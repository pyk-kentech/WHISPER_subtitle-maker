from __future__ import annotations

import html
import random
import re
import time
import warnings
from dataclasses import dataclass
from typing import Callable

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

from google.generativeai.types import HarmBlockThreshold, HarmCategory

from .config import (
    DEFAULT_TRANSLATION_BACKOFF_BASE_SECONDS,
    DEFAULT_TRANSLATION_BACKOFF_MAX_SECONDS,
    DEFAULT_TRANSLATION_MODELS,
    DEFAULT_TRANSLATION_REQUEST_DELAY_JITTER_SECONDS,
)


LogCallback = Callable[[str], None]

USER_CONTENT_TEMPLATE = """<main id=\"source\">
{{slot}}
</main>
<main id=\"translation\">"""

TARGET_LANGUAGE_LABELS = {
    "ko": "한국어",
    "en": "English",
    "ja": "日本語",
}


class TranslationError(RuntimeError):
    pass


class QuotaExceededError(TranslationError):
    pass


class SafetyBlockedError(TranslationError):
    pass


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
        chunks = [
            records[index : index + self.config.chunk_size]
            for index in range(0, len(records), self.config.chunk_size)
        ]

        translated: dict[str, str] = {}
        for chunk_index, chunk in enumerate(chunks, start=1):
            translated.update(self._translate_chunk_with_split(chunk, file_name, chunk_index, len(chunks)))
            on_chunk_progress(
                chunk_index,
                len(chunks),
                self.current_key_display,
                self._resolve_model_candidates()[0],
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
            return self._translate_chunk(chunk, file_name, chunk_index, chunk_total)
        except TranslationError as exc:
            if len(chunk) <= 1:
                line_id = getattr(chunk[0], "line_id", "?") if chunk else "?"
                self.log_callback(
                    f"[{file_name}] chunk {chunk_index}/{chunk_total} | line {line_id} failed after split retries -> keeping source text ({exc})"
                )
                return {}

            if not self._should_split_for_error(str(exc)):
                self.log_callback(
                    f"[{file_name}] chunk {chunk_index}/{chunk_total} | chunk failed without split fallback -> keeping source text for {len(chunk)} lines ({exc})"
                )
                return {}

            midpoint = max(1, len(chunk) // 2)
            left_chunk = chunk[:midpoint]
            right_chunk = chunk[midpoint:]
            self.log_callback(
                f"[{file_name}] chunk {chunk_index}/{chunk_total} | splitting blocked chunk depth={split_depth + 1} size={len(chunk)} -> {len(left_chunk)} + {len(right_chunk)}"
            )
            translated: dict[str, str] = {}
            translated.update(
                self._translate_chunk_with_split(left_chunk, file_name, chunk_index, chunk_total, split_depth + 1)
            )
            translated.update(
                self._translate_chunk_with_split(right_chunk, file_name, chunk_index, chunk_total, split_depth + 1)
            )
            return translated

    def _translate_chunk(self, chunk, file_name: str, chunk_index: int, chunk_total: int) -> dict[str, str]:
        payload = "\n".join(f'<p id="{record.line_id}">{record.text}</p>' for record in chunk)
        user_prompt = USER_CONTENT_TEMPLATE.replace("{{slot}}", payload)
        note = self.config.translation_note.strip()
        language_note = _build_target_language_note(self.config.target_language)
        reasoning_note = _build_reasoning_note(self.config.reasoning_level)
        extra_notes = "\n".join(part for part in (language_note, note, reasoning_note) if part).strip()
        system_prompt = self.config.system_prompt.replace("{{note}}", extra_notes)

        while True:
            any_quota_error = False
            quota_error_count = 0
            retry_count = 0
            for model_name in self._resolve_model_candidates():
                for offset in range(len(self.config.keys)):
                    if retry_count >= self.config.max_retry_per_chunk:
                        raise TranslationError("Retry limit reached for this translation chunk.")
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
                        return self._parse_response(response_text, [record.line_id for record in chunk])
                    except SafetyBlockedError as exc:
                        self.error_count += 1
                        retry_count += 1
                        self.log_callback(f"Safety blocked, switching key: {exc}")
                        continue
                    except QuotaExceededError as exc:
                        self.error_count += 1
                        any_quota_error = True
                        quota_error_count += 1
                        retry_count += 1
                        self._on_quota_error()
                        backoff_seconds = self._compute_backoff_seconds(quota_error_count)
                        self.log_callback(f"Quota exceeded, backing off for {backoff_seconds:.1f}s before switching key: {exc}")
                        time.sleep(backoff_seconds)
                        continue
                    except TranslationError as exc:
                        self.error_count += 1
                        retry_count += 1
                        self.log_callback(f"Model/key combination failed, trying next candidate: {exc}")
                        break

            if any_quota_error:
                self.log_callback(
                    f"All current keys are rate-limited. Waiting {self.config.wait_seconds_when_exhausted} seconds before retry."
                )
                time.sleep(self.config.wait_seconds_when_exhausted)
                continue
            raise TranslationError("No available Gemini key/model combination could complete the translation.")

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
            message = str(exc)
            if _is_quota_error(message):
                raise QuotaExceededError(message) from exc
            if _is_safety_error(message):
                raise SafetyBlockedError(message) from exc
            raise TranslationError(message) from exc

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
            raise TranslationError(message or "Failed to read response text.") from exc

        if not text.strip():
            raise TranslationError("Received an empty response.")
        return text

    def _parse_response(self, response_text: str, expected_ids: list[str]) -> dict[str, str]:
        pairs = dict(re.findall(r'<p id="([^"]+)">(.*?)</p>', response_text, flags=re.DOTALL))
        if not pairs:
            raise TranslationError("The response does not contain any <p id=\"...\"> pairs.")

        translated: dict[str, str] = {}
        missing: list[str] = []
        for line_id in expected_ids:
            if line_id not in pairs:
                missing.append(line_id)
                continue
            translated[line_id] = html.unescape(pairs[line_id].strip())

        if missing:
            raise TranslationError(f"Some expected ids are missing from the response: {', '.join(missing[:5])}")
        return translated

    def _resolve_model_candidates(self) -> list[str]:
        preferred = self.config.preferred_model.strip()
        if preferred:
            return [preferred, *[item for item in DEFAULT_TRANSLATION_MODELS if item != preferred]]
        return list(DEFAULT_TRANSLATION_MODELS)

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

    def _compute_backoff_seconds(self, quota_error_count: int) -> float:
        exponent = max(0, quota_error_count - 1)
        return min(
            DEFAULT_TRANSLATION_BACKOFF_MAX_SECONDS,
            DEFAULT_TRANSLATION_BACKOFF_BASE_SECONDS * (2 ** exponent),
        )


def _is_quota_error(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ("429", "quota", "resource_exhausted", "rate limit", "too many requests"))


def _is_safety_error(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ("safety", "blocked", "block_reason", "prohibited", "recitation"))


def _build_target_language_note(target_language: str) -> str:
    label = TARGET_LANGUAGE_LABELS.get(target_language, target_language)
    return (
        f"Translate all text content into {label}. Keep every HTML tag, line id, ordering, and subtitle structure unchanged. "
        "Do not alter timestamps or merge/split lines."
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
