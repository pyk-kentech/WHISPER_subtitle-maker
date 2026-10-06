from __future__ import annotations

import html
import random
import re
import time
import unicodedata
import warnings
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

from google.generativeai.types import HarmBlockThreshold, HarmCategory

from .config import (
    DEFAULT_TRANSLATION_MODELS,
    DEFAULT_TRANSLATION_REQUEST_DELAY_JITTER_SECONDS,
    TRANSLATION_CONTEXT_LINES,
    TRANSLATION_ECHO_MIN_SIMILARITY,
)


LogCallback = Callable[[str], None]

USER_CONTENT_TEMPLATE = """<main id=\"source\">
{{slot}}
</main>
<main id=\"translation\">"""

CONTEXT_CONTENT_TEMPLATE = """<main id=\"context\">
{{slot}}
</main>
"""

_LINE_PATTERN = re.compile(r'<p id="([^"]+)">(.*?)</p>', re.DOTALL)
_ECHO_PATTERN = re.compile(r"\s*<o>(.*?)</o>(.*)\Z", re.DOTALL)

TARGET_LANGUAGE_LABELS = {
    "ko": "한국어",
    "en": "English",
    "ja": "日本語",
}


class TranslationError(RuntimeError):
    pass


class RetryableTranslationError(TranslationError):
    pass


class QuotaExceededError(RetryableTranslationError):
    pass


class TransientServiceError(RetryableTranslationError):
    pass


class SafetyBlockedError(TranslationError):
    pass


class ResponseFormatError(TranslationError):
    pass


class TranslationUnavailableError(TranslationError):
    """키·모델·할당량 문제라 청크를 쪼개 다시 보내도 해결되지 않는 실패."""


@dataclass(slots=True)
class TranslationConfig:
    keys: list[str]
    preferred_model: str
    target_language: str
    system_prompt: str
    translation_note: str
    temperature: float
    top_p: float
    reasoning_level: str
    chunk_size: int
    request_delay_seconds: float
    wait_seconds_when_exhausted: int = 60
    max_exhausted_waits: int = 3
    min_adaptive_delay_seconds: float = 1.0
    max_adaptive_delay_seconds: float = 12.0
    max_retry_per_chunk: int = 8


class GeminiTranslator:
    def __init__(self, config: TranslationConfig, log_callback: LogCallback) -> None:
        self.config = config
        self.log_callback = log_callback
        self.error_count = 0
        self._key_index = 0
        self._last_request_monotonic = 0.0
        self._records: list = []
        self._record_positions: dict[str, int] = {}
        self._unavailable_models: set[str] = set()
        self._quota_exhausted = False
        self.token_usage = {"prompt": 0, "output": 0, "thinking": 0}
        self._adaptive_delay_seconds = max(
            self.config.min_adaptive_delay_seconds,
            float(self.config.request_delay_seconds),
        )

    @property
    def current_key_display(self) -> str:
        return f"{self._key_index + 1}/{len(self.config.keys)}"

    def translate_lines(
        self,
        records,
        on_chunk_progress: Callable[[int, int, str, str, int], None],
        file_name: str,
    ) -> dict[str, str]:
        """번역에 실패한 줄은 결과에서 빠진다. 키·모델·할당량 문제로 중단되면
        TranslationUnavailableError를 던지며, 그때까지의 결과는 예외의 partial에 담긴다."""
        if self._quota_exhausted:
            raise TranslationUnavailableError("Gemini 할당량이 소진되어 이번 작업에서는 Gemini 번역을 건너뜁니다.")

        chunks = [
            records[index : index + self.config.chunk_size]
            for index in range(0, len(records), self.config.chunk_size)
        ]
        self._records = list(records)
        self._record_positions = {record.line_id: position for position, record in enumerate(records)}

        translated: dict[str, str] = {}
        for chunk_index, chunk in enumerate(chunks, start=1):
            try:
                translated.update(self._translate_chunk_with_split(chunk, file_name, chunk_index, len(chunks)))
            except TranslationUnavailableError as exc:
                exc.partial = translated
                raise
            candidates = self._resolve_model_candidates()
            on_chunk_progress(
                chunk_index,
                len(chunks),
                self.current_key_display,
                candidates[0] if candidates else "",
                self.error_count,
            )
        return translated

    def _translate_chunk_with_split(
        self,
        chunk,
        file_name: str,
        chunk_index: int,
        chunk_total: int,
        split_depth: int = 0,
    ) -> dict[str, str]:
        try:
            translated, rejected_ids = self._translate_chunk(chunk, file_name, chunk_index, chunk_total)
        except TranslationUnavailableError:
            raise
        except TranslationError as exc:
            return self._split_failed_chunk(chunk, file_name, chunk_index, chunk_total, split_depth, str(exc))

        if not rejected_ids:
            return translated

        if len(rejected_ids) == len(chunk):
            return self._split_failed_chunk(
                chunk,
                file_name,
                chunk_index,
                chunk_total,
                split_depth,
                "every line failed the source echo check",
            )

        retry_chunk = [record for record in chunk if record.line_id in rejected_ids]
        self.log_callback(
            f"[{file_name}] chunk {chunk_index}/{chunk_total} | {len(retry_chunk)}/{len(chunk)} lines misaligned "
            f"(first: {retry_chunk[0].line_id}) -> retrying those lines"
        )
        translated.update(
            self._translate_chunk_with_split(retry_chunk, file_name, chunk_index, chunk_total, split_depth + 1)
        )
        return translated

    def _split_failed_chunk(
        self,
        chunk,
        file_name: str,
        chunk_index: int,
        chunk_total: int,
        split_depth: int,
        reason: str,
    ) -> dict[str, str]:
        if len(chunk) <= 1:
            line_id = getattr(chunk[0], "line_id", "?") if chunk else "?"
            self.log_callback(
                f"[{file_name}] chunk {chunk_index}/{chunk_total} | line {line_id} failed after split retries -> keeping source text ({reason})"
            )
            return {}

        if not self._should_split_for_error(reason):
            self.log_callback(
                f"[{file_name}] chunk {chunk_index}/{chunk_total} | chunk failed without split fallback -> keeping source text for {len(chunk)} lines ({reason})"
            )
            return {}

        midpoint = max(1, len(chunk) // 2)
        left_chunk = chunk[:midpoint]
        right_chunk = chunk[midpoint:]
        self.log_callback(
            f"[{file_name}] chunk {chunk_index}/{chunk_total} | splitting failed chunk depth={split_depth + 1} size={len(chunk)} -> {len(left_chunk)} + {len(right_chunk)} ({reason})"
        )
        translated: dict[str, str] = {}
        translated.update(
            self._translate_chunk_with_split(left_chunk, file_name, chunk_index, chunk_total, split_depth + 1)
        )
        translated.update(
            self._translate_chunk_with_split(right_chunk, file_name, chunk_index, chunk_total, split_depth + 1)
        )
        return translated

    def _build_user_prompt(self, chunk) -> str:
        payload = "\n".join(f'<p id="{record.line_id}">{record.text}</p>' for record in chunk)
        user_prompt = USER_CONTENT_TEMPLATE.replace("{{slot}}", payload)
        first_position = self._record_positions.get(chunk[0].line_id) if chunk else None
        if not first_position:
            return user_prompt
        context_records = self._records[max(0, first_position - TRANSLATION_CONTEXT_LINES) : first_position]
        context = "\n".join(record.text for record in context_records)
        return CONTEXT_CONTENT_TEMPLATE.replace("{{slot}}", context) + user_prompt

    def _translate_chunk(
        self, chunk, file_name: str, chunk_index: int, chunk_total: int
    ) -> tuple[dict[str, str], list[str]]:
        user_prompt = self._build_user_prompt(chunk)
        note = self.config.translation_note.strip()
        language_note = _build_target_language_note(self.config.target_language)
        reasoning_note = _build_reasoning_note(self.config.reasoning_level)
        extra_notes = "\n".join(part for part in (language_note, note, reasoning_note) if part).strip()
        system_prompt = self.config.system_prompt.replace("{{note}}", extra_notes)

        exhausted_waits = 0
        while True:
            content_failures = 0
            retryable_failures = 0
            quota_failures = 0
            attempts = 0
            last_error = ""
            for model_name in self._resolve_model_candidates():
                for offset in range(len(self.config.keys)):
                    if attempts >= self.config.max_retry_per_chunk:
                        break
                    key_index = (self._key_index + offset) % len(self.config.keys)
                    api_key = self.config.keys[key_index]
                    self._key_index = key_index
                    self.log_callback(
                        f"[{file_name}] chunk {chunk_index}/{chunk_total} | model={model_name} | key {key_index + 1}/{len(self.config.keys)}"
                    )
                    try:
                        self._sleep_for_request_spacing()
                        response_text = self._request_translation(api_key, model_name, system_prompt, user_prompt)
                        self._last_request_monotonic = time.monotonic()
                        self._on_request_success()
                        return self._parse_response(response_text, chunk, file_name)
                    except (SafetyBlockedError, ResponseFormatError) as exc:
                        # 내용 문제는 키를 바꿔도 같으므로, 여러 줄이면 바로 쪼개서 문제 줄을 찾고
                        # 한 줄까지 좁혀졌을 때만 다른 모델로 한 번씩 시도한다.
                        self._last_request_monotonic = time.monotonic()
                        self.error_count += 1
                        if len(chunk) > 1:
                            self.log_callback(f"Blocked or unreadable response on {model_name} -> splitting chunk: {exc}")
                            raise
                        attempts += 1
                        content_failures += 1
                        last_error = str(exc)
                        self.log_callback(f"Line blocked or unreadable on {model_name}, trying next model: {exc}")
                        break
                    except RetryableTranslationError as exc:
                        self._last_request_monotonic = time.monotonic()
                        self.error_count += 1
                        retryable_failures += 1
                        if isinstance(exc, QuotaExceededError):
                            quota_failures += 1
                            self._on_quota_error()
                        last_error = str(exc)
                        self.log_callback(f"Rate limited or temporarily unavailable, switching key: {exc}")
                        continue
                    except TranslationError as exc:
                        self._last_request_monotonic = time.monotonic()
                        self.error_count += 1
                        attempts += 1
                        message = str(exc)
                        last_error = message
                        if _is_model_unavailable_error(message):
                            self._unavailable_models.add(model_name)
                            self.log_callback(f"Model {model_name} is unavailable, skipping it from now on: {exc}")
                            break
                        if _is_auth_error(message):
                            self.log_callback(f"Key {key_index + 1} was rejected, trying next key: {exc}")
                            continue
                        self.log_callback(f"Model/key combination failed, trying next candidate: {exc}")
                        break

            if content_failures:
                raise TranslationError(f"Gemini could not translate this chunk: {last_error}")
            if retryable_failures:
                if exhausted_waits >= self.config.max_exhausted_waits:
                    if quota_failures:
                        self._quota_exhausted = True
                    raise TranslationUnavailableError(
                        f"Gemini kept failing after {exhausted_waits} waits (quota or service outage): {last_error}"
                    )
                exhausted_waits += 1
                self.log_callback(
                    f"All keys/models are rate-limited or unavailable. Waiting {self.config.wait_seconds_when_exhausted} seconds "
                    f"before retry ({exhausted_waits}/{self.config.max_exhausted_waits})."
                )
                time.sleep(self.config.wait_seconds_when_exhausted)
                continue
            raise TranslationUnavailableError(
                f"No available Gemini key/model combination could complete the translation: {last_error or 'no usable model'}"
            )

    def _should_split_for_error(self, message: str) -> bool:
        lowered = message.lower()
        if _is_quota_error(lowered):
            return False
        return True

    def _request_translation(self, api_key: str, model_name: str, system_prompt: str, user_prompt: str) -> str:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_prompt,
            generation_config={
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "candidate_count": 1,
                "response_mime_type": "text/plain",
            },
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
        try:
            response = model.generate_content(user_prompt, request_options={"timeout": 1200})
        except Exception as exc:
            message = str(exc) or exc.__class__.__name__
            if _is_quota_error(message):
                raise QuotaExceededError(message) from exc
            if _is_safety_error(message):
                raise SafetyBlockedError(message) from exc
            if _is_transient_error(message):
                raise TransientServiceError(message) from exc
            raise TranslationError(message) from exc
        self._log_token_usage(model_name, response)

        finish_reason = ""
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            finish_reason = str(getattr(candidates[0], "finish_reason", ""))
            if "SAFETY" in finish_reason.upper():
                raise SafetyBlockedError(f"finish_reason={finish_reason}")

        try:
            text = response.text
        except Exception as exc:
            message = str(exc)
            if _is_safety_error(message) or "SAFETY" in finish_reason.upper():
                raise SafetyBlockedError(message or finish_reason) from exc
            raise ResponseFormatError(message or f"Failed to read response text (finish_reason={finish_reason}).") from exc

        if not text.strip():
            raise ResponseFormatError("Received an empty response.")
        return text

    def _log_token_usage(self, model_name: str, response) -> None:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return
        counts = {
            "prompt": int(getattr(usage, "prompt_token_count", 0) or 0),
            "output": int(getattr(usage, "candidates_token_count", 0) or 0),
            "thinking": int(getattr(usage, "thoughts_token_count", 0) or 0),
        }
        for name, value in counts.items():
            self.token_usage[name] += value
        self.log_callback(
            f"tokens | model={model_name} | prompt={counts['prompt']} output={counts['output']} thinking={counts['thinking']}"
        )

    @property
    def token_usage_summary(self) -> str:
        usage = self.token_usage
        return f"Gemini 누적 토큰 | 입력 {usage['prompt']:,} | 출력 {usage['output']:,} | thinking {usage['thinking']:,}"

    def _parse_response(self, response_text: str, chunk, file_name: str) -> tuple[dict[str, str], list[str]]:
        pairs = dict(_LINE_PATTERN.findall(response_text))
        if not pairs:
            raise ResponseFormatError("The response does not contain any <p id=\"...\"> pairs.")

        has_echo = any(_ECHO_PATTERN.match(body) for body in pairs.values())
        if not has_echo:
            self.log_callback(f"[{file_name}] response has no <o> source echo -> line alignment not verified")

        translated: dict[str, str] = {}
        rejected: list[str] = []
        for position, record in enumerate(chunk):
            body = pairs.get(record.line_id)
            if body is None:
                rejected.append(record.line_id)
                continue
            if has_echo:
                match = _ECHO_PATTERN.match(body)
                neighbors = [item.text for item in chunk[max(0, position - 2) : position + 3] if item is not record]
                if match is None or not _echo_matches(record.text, match.group(1), neighbors):
                    rejected.append(record.line_id)
                    continue
                body = match.group(2)
            text = html.unescape(body.strip())
            if not text and _normalize_for_echo(record.text):
                rejected.append(record.line_id)
                continue
            translated[record.line_id] = text
        return translated, rejected

    def _resolve_model_candidates(self) -> list[str]:
        preferred = self.config.preferred_model.strip()
        candidates = [preferred] if preferred else []
        candidates += [item for item in DEFAULT_TRANSLATION_MODELS if item != preferred]
        return [item for item in candidates if item not in self._unavailable_models]

    def _sleep_for_request_spacing(self) -> None:
        base_delay = max(self.config.min_adaptive_delay_seconds, self._adaptive_delay_seconds)
        if base_delay <= 0:
            return
        jitter = random.uniform(0.0, DEFAULT_TRANSLATION_REQUEST_DELAY_JITTER_SECONDS)
        target_delay = base_delay + jitter
        if self._last_request_monotonic <= 0:
            if target_delay > 0:
                self.log_callback(f"Request delay {target_delay:.1f}s before Gemini call")
                time.sleep(target_delay)
            return

        elapsed = time.monotonic() - self._last_request_monotonic
        remaining = target_delay - elapsed
        if remaining > 0:
            self.log_callback(f"Request delay {remaining:.1f}s before Gemini call")
            time.sleep(remaining)

    def _on_request_success(self) -> None:
        self._adaptive_delay_seconds = max(
            self.config.min_adaptive_delay_seconds,
            self._adaptive_delay_seconds - 0.2,
        )

    def _on_quota_error(self) -> None:
        self._adaptive_delay_seconds = min(
            self.config.max_adaptive_delay_seconds,
            self._adaptive_delay_seconds + 2.0,
        )


def _is_quota_error(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ("429", "quota", "resource_exhausted", "rate limit", "too many requests"))


def _is_transient_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in ("500", "502", "503", "504", "internal", "unavailable", "overloaded", "deadline", "timed out", "timeout", "connection")
    )


def _is_auth_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in ("api key not valid", "api_key_invalid", "api key expired", "permission_denied", "permission denied", "unauthenticated", "401", "403")
    )


def _is_model_unavailable_error(message: str) -> bool:
    lowered = message.lower()
    return "404" in lowered or "not found" in lowered or "not supported for generatecontent" in lowered


def _is_safety_error(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ("safety", "blocked", "block_reason", "prohibited", "recitation"))


def _normalize_for_echo(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", html.unescape(text)).casefold()
    return "".join(char for char in normalized if unicodedata.category(char)[0] in {"L", "N"})


def _echo_matches(source_text: str, echoed_text: str, neighbor_texts: list[str] | None = None) -> bool:
    source = _normalize_for_echo(source_text)
    if not source:
        return True
    echoed = _normalize_for_echo(echoed_text)
    similarity = SequenceMatcher(None, source, echoed).ratio()
    if similarity < TRANSLATION_ECHO_MIN_SIMILARITY:
        return False
    return all(
        SequenceMatcher(None, _normalize_for_echo(neighbor), echoed).ratio() <= similarity
        for neighbor in neighbor_texts or []
    )


def _build_target_language_note(target_language: str) -> str:
    label = TARGET_LANGUAGE_LABELS.get(target_language, target_language)
    return (
        f"Translate all text content into {label}. Keep every HTML tag, line id, ordering, and subtitle structure unchanged. "
        "Do not alter timestamps or merge/split lines.\n"
        'Output every line as <p id="ID"><o>source</o>translation</p>, where <o> holds an exact copy of the source text '
        "of that same ID, followed by the translation of that line only. Never move text from one ID to another, even "
        "when a sentence continues across lines.\n"
        'Lines inside <main id="context"> are earlier dialogue for reference only. Do not translate or output them.'
    )


def _build_reasoning_note(reasoning_level: str) -> str:
    normalized = reasoning_level.strip().lower()
    if normalized == "minimal":
        return "Reasoning level: minimal. Prioritize structure preservation and direct output."
    if normalized == "low":
        return "Reasoning level: low. Preserve structure and output only the translation result."
    if normalized == "medium":
        return "Reasoning level: medium. Improve translation quality while keeping structure exact."
    if normalized == "high":
        return "Reasoning level: high. Maximize translation quality but preserve structure exactly."
    return ""
