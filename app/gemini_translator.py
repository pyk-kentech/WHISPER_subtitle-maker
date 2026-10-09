from __future__ import annotations

import html
import random
import re
import socket
import time
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, TypeVar
from urllib import error as urllib_error

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from .config import (
    DEFAULT_TRANSLATION_MODELS,
    DEFAULT_TRANSLATION_REQUEST_DELAY_JITTER_SECONDS,
    NETWORK_OUTAGE_MAX_WAIT_SECONDS,
    NETWORK_OUTAGE_RETRY_SECONDS,
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

REQUEST_TIMEOUT_MS = 1_200_000
THINKING_LEVELS = ["minimal", "low", "medium", "high"]
_SAFETY_SETTINGS = [
    genai_types.SafetySetting(category=category, threshold=genai_types.HarmBlockThreshold.BLOCK_NONE)
    for category in (
        genai_types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        genai_types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        genai_types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        genai_types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
]
_BLOCKED_FINISH_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}

# 모델이 </p>를 빠뜨려도 다음 줄을 삼키지 않도록 다음 <p id=나 </main>에서도 끊는다.
_LINE_PATTERN = re.compile(r'<p id="([^"]+)">(.*?)(?=</p>|<p id="|</main>|\Z)', re.DOTALL)
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


class TranslationCancelled(Exception):
    """프로그램 종료 등으로 번역을 멈출 때 쓴다. 번역 실패(TranslationError)와 구분된다."""


class TranslationUnavailableError(TranslationError):
    """키·모델·할당량 문제라 청크를 쪼개 다시 보내도 해결되지 않는 실패."""


class NetworkUnavailableError(TranslationUnavailableError):
    """인터넷 연결이 끊긴 채 기다려도 돌아오지 않아 번역을 멈출 때 쓴다."""


T = TypeVar("T")

# DNS 조회 실패·경로 없음 등 "서버에 닿지도 못한" 상태를 나타내는 문구(Windows/Linux/macOS).
_NETWORK_DOWN_TOKENS = (
    "getaddrinfo failed",
    "name or service not known",
    "temporary failure in name resolution",
    "nodename nor servname",
    "no address associated with hostname",
    "network is unreachable",
    "no route to host",
)


def is_network_down_error(exc: BaseException) -> bool:
    """키를 바꿔도 소용없는 연결 실패(인터넷 끊김)인지 예외 체인을 따라가며 확인한다."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (httpx.ConnectError, httpx.ConnectTimeout, socket.gaierror)):
            return True
        if isinstance(current, urllib_error.URLError) and isinstance(
            current.reason, (socket.gaierror, ConnectionRefusedError, TimeoutError)
        ):
            return True
        lowered = str(current).lower()
        if any(token in lowered for token in _NETWORK_DOWN_TOKENS):
            return True
        current = current.__cause__ or current.__context__
    return False


class NetworkWaiter:
    """인터넷이 끊기면 키·모델을 돌려 가며 헛도는 대신 연결이 돌아올 때까지 같은 요청을 다시 보낸다.
    Gemini와 DeepL이 한 객체를 같이 써서, 한 번 기다리다 포기한 뒤에는 연결이 돌아오기 전까지 곧바로 포기한다
    (음성 인식 등 나머지 작업이 막히지 않도록)."""

    def __init__(
        self,
        log_callback: LogCallback,
        max_wait_seconds: float = NETWORK_OUTAGE_MAX_WAIT_SECONDS,
        retry_seconds: tuple[float, ...] = NETWORK_OUTAGE_RETRY_SECONDS,
    ) -> None:
        self.log_callback = log_callback
        self.max_wait_seconds = max_wait_seconds
        self.retry_seconds = retry_seconds
        self._outage_started: float | None = None
        self._gave_up = False

    def run(self, request: Callable[[], T], sleep: Callable[[float], None]) -> T:
        attempt = 0
        while True:
            try:
                result = request()
            except Exception as exc:
                if not is_network_down_error(exc):
                    # 서버까지는 닿았다는 뜻이므로 연결은 살아 있다.
                    self._mark_online()
                    raise
                if self._gave_up:
                    raise NetworkUnavailableError(f"인터넷 연결 없음: {exc}") from exc
                now = time.monotonic()
                if self._outage_started is None:
                    self._outage_started = now
                    self.log_callback(
                        f"인터넷 연결이 끊겼습니다 ({exc}). 키·모델은 그대로 두고 최대 "
                        f"{self.max_wait_seconds / 60:.0f}분 동안 연결을 기다립니다."
                    )
                elapsed = now - self._outage_started
                if elapsed >= self.max_wait_seconds:
                    self._gave_up = True
                    raise NetworkUnavailableError(
                        f"인터넷 연결이 {elapsed / 60:.0f}분 넘게 끊겨 있어 번역을 멈춥니다: {exc}"
                    ) from exc
                wait = self.retry_seconds[min(attempt, len(self.retry_seconds) - 1)]
                wait = min(wait, self.max_wait_seconds - elapsed)
                attempt += 1
                self.log_callback(f"연결 대기 중... {wait:.0f}초 후 다시 시도 (끊긴 지 {elapsed:.0f}초)")
                sleep(wait)
                continue
            self._mark_online()
            return result

    def _mark_online(self) -> None:
        if self._outage_started is not None or self._gave_up:
            self.log_callback("인터넷 연결이 돌아왔습니다. 번역을 이어갑니다.")
        self._outage_started = None
        self._gave_up = False


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
        self._daily_exhausted_models: set[str] = set()
        self._quota_exhausted = False
        self.token_usage = {"prompt": 0, "output": 0, "thinking": 0}
        self._clients: dict[str, genai.Client] = {}
        self.cancel_check: Callable[[], bool] | None = None
        self.network = NetworkWaiter(log_callback)
        self._thinking_levels: dict[str, str | None] = {}
        # 사용자가 정한 요청 간격은 최소값으로 지킨다(무료 티어 분당 한도 보호).
        self._base_delay_seconds = max(self.config.min_adaptive_delay_seconds, float(self.config.request_delay_seconds))
        self._adaptive_delay_seconds = self._base_delay_seconds

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
                _carry_partial(exc, translated)
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
        try:
            translated.update(
                self._translate_chunk_with_split(retry_chunk, file_name, chunk_index, chunk_total, split_depth + 1)
            )
        except TranslationUnavailableError as exc:
            _carry_partial(exc, translated)
            raise
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
        try:
            translated.update(
                self._translate_chunk_with_split(left_chunk, file_name, chunk_index, chunk_total, split_depth + 1)
            )
            translated.update(
                self._translate_chunk_with_split(right_chunk, file_name, chunk_index, chunk_total, split_depth + 1)
            )
        except TranslationUnavailableError as exc:
            _carry_partial(exc, translated)
            raise
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
        extra_notes = "\n".join(part for part in (language_note, note) if part).strip()
        system_prompt = self.config.system_prompt.replace("{{note}}", extra_notes)

        exhausted_waits = 0
        while True:
            content_failures = 0
            retryable_failures = 0
            quota_failures = 0
            attempts = 0
            last_error = ""
            start_key_index = self._key_index
            candidates = self._resolve_model_candidates()
            if not candidates and self._daily_exhausted_models:
                self._quota_exhausted = True
                raise TranslationUnavailableError("모든 Gemini 모델의 오늘 무료 한도가 소진되었습니다.")
            for model_name in candidates:
                model_quota_failures = 0
                for offset in range(len(self.config.keys)):
                    if attempts >= self.config.max_retry_per_chunk:
                        break
                    key_index = (start_key_index + offset) % len(self.config.keys)
                    api_key = self.config.keys[key_index]
                    self._key_index = key_index
                    self.log_callback(
                        f"[{file_name}] chunk {chunk_index}/{chunk_total} | model={model_name} | key {key_index + 1}/{len(self.config.keys)}"
                    )
                    try:
                        self._sleep_for_request_spacing()
                        response_text = self.network.run(
                            lambda: self._request_translation(api_key, model_name, system_prompt, user_prompt),
                            self._sleep,
                        )
                        self._last_request_monotonic = time.monotonic()
                        self._on_request_success()
                        return self._parse_response(response_text, chunk, file_name)
                    except NetworkUnavailableError:
                        # 연결이 끊긴 동안은 다른 키·모델도 똑같이 실패하므로 돌려 보지 않는다.
                        self._last_request_monotonic = time.monotonic()
                        raise
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
                            model_quota_failures += 1
                            if _is_daily_quota_error(str(exc)) and model_quota_failures >= len(self.config.keys):
                                # 하루 한도는 기다려도 풀리지 않으므로 이번 작업에서는 이 모델을 건너뛴다.
                                self._daily_exhausted_models.add(model_name)
                                self._unavailable_models.add(model_name)
                                self.log_callback(f"Model {model_name} reached its daily quota on every key; skipping it from now on.")
                            else:
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
            if retryable_failures and not self._resolve_model_candidates():
                continue
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
                self._sleep(self.config.wait_seconds_when_exhausted)
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
        client = self._clients.get(api_key)
        if client is None:
            client = genai.Client(api_key=api_key, http_options=genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_MS))
            self._clients[api_key] = client

        while True:
            thinking_level = self._thinking_level_for(model_name)
            config = genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                candidate_count=1,
                response_mime_type="text/plain",
                safety_settings=_SAFETY_SETTINGS,
                thinking_config=(
                    genai_types.ThinkingConfig(thinking_level=thinking_level.upper()) if thinking_level else None
                ),
            )
            try:
                response = client.models.generate_content(model=model_name, contents=user_prompt, config=config)
                break
            except genai_errors.APIError as exc:
                message = str(exc) or exc.__class__.__name__
                if exc.code == 400 and thinking_level and "thinking" in message.lower():
                    self._downgrade_thinking_level(model_name, thinking_level, message)
                    continue
                raise _classify_api_error(message, exc.code) from exc
            except httpx.TransportError as exc:
                raise TransientServiceError(str(exc) or exc.__class__.__name__) from exc
            except Exception as exc:
                raise _classify_api_error(str(exc) or exc.__class__.__name__, None) from exc

        self._log_token_usage(model_name, response)
        feedback = getattr(response, "prompt_feedback", None)
        if feedback is not None and feedback.block_reason:
            raise SafetyBlockedError(f"prompt blocked: {_enum_name(feedback.block_reason)}")
        candidates = response.candidates or []
        finish_reason = _enum_name(candidates[0].finish_reason) if candidates else ""
        if finish_reason in _BLOCKED_FINISH_REASONS:
            raise SafetyBlockedError(f"finish_reason={finish_reason}")
        text = response.text or ""
        if not text.strip():
            raise ResponseFormatError(f"Received an empty response (finish_reason={finish_reason or 'none'}).")
        return text

    def _sleep(self, seconds: float) -> None:
        # 긴 대기(최대 60초) 중에도 종료 요청을 빨리 반영하도록 잘게 나눠 잔다.
        remaining = max(0.0, seconds)
        while True:
            if self.cancel_check is not None and self.cancel_check():
                raise TranslationCancelled()
            if remaining <= 0:
                return
            step = min(0.5, remaining)
            time.sleep(step)
            remaining -= step

    def _thinking_level_for(self, model_name: str) -> str | None:
        if model_name in self._thinking_levels:
            return self._thinking_levels[model_name]
        level = self.config.reasoning_level.strip().lower()
        return level if level in THINKING_LEVELS else None

    def _downgrade_thinking_level(self, model_name: str, rejected_level: str, message: str) -> None:
        # 모델마다 허용하는 thinking 수준이 달라서(예: 3.7/3.8 Flash는 minimal 불가) 거절되면 한 단계 올리고,
        # 끝까지 안 되면 thinking 설정 없이 보낸다. 결과는 모델별로 기억한다.
        position = THINKING_LEVELS.index(rejected_level)
        next_level = THINKING_LEVELS[position + 1] if position + 1 < len(THINKING_LEVELS) else None
        self._thinking_levels[model_name] = next_level
        self.log_callback(
            f"Model {model_name} rejected thinking level '{rejected_level}' -> using '{next_level or 'model default'}' ({message[:120]})"
        )

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
            text = _strip_echo_leftovers(html.unescape(body.strip()))
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
        base_delay = max(self._base_delay_seconds, self._adaptive_delay_seconds)
        if base_delay <= 0:
            return
        jitter = random.uniform(0.0, DEFAULT_TRANSLATION_REQUEST_DELAY_JITTER_SECONDS)
        target_delay = base_delay + jitter
        if self._last_request_monotonic <= 0:
            if target_delay > 0:
                self.log_callback(f"Request delay {target_delay:.1f}s before Gemini call")
                self._sleep(target_delay)
            return

        elapsed = time.monotonic() - self._last_request_monotonic
        remaining = target_delay - elapsed
        if remaining > 0:
            self.log_callback(f"Request delay {remaining:.1f}s before Gemini call")
            self._sleep(remaining)

    def _on_request_success(self) -> None:
        self._adaptive_delay_seconds = max(self._base_delay_seconds, self._adaptive_delay_seconds - 0.2)

    def _on_quota_error(self) -> None:
        ceiling = max(self.config.max_adaptive_delay_seconds, self._base_delay_seconds)
        self._adaptive_delay_seconds = min(ceiling, self._adaptive_delay_seconds + 2.0)


def _is_quota_error(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ("429", "quota", "resource_exhausted", "rate limit", "too many requests"))


def _is_daily_quota_error(message: str) -> bool:
    lowered = message.lower()
    return "perday" in lowered or "per day" in lowered or "daily" in lowered


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


_LEADING_ECHO = re.compile(r"\A\s*<o>.*?</o>", re.DOTALL)


def _strip_echo_leftovers(text: str) -> str:
    """모델이 <o>를 두 번 쓰는 등(<o><o>원문</o>번역) 되풀이가 번역 앞에 남으면 지운다. 남은 <o>, </o> 태그도 지운다."""
    while _LEADING_ECHO.match(text):
        text = _LEADING_ECHO.sub("", text, count=1)
    return text.replace("<o>", "").replace("</o>", "").strip()


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


def _carry_partial(exc: TranslationUnavailableError, translated: dict[str, str]) -> None:
    """중단 예외에 지금까지 번역된 줄을 모아 둔다(안쪽 재시도·분할에서 얻은 결과 포함)."""
    partial = getattr(exc, "partial", None) or {}
    exc.partial = {**translated, **partial}


def _enum_name(value) -> str:
    return str(getattr(value, "name", None) or value or "").split(".")[-1].upper()


def _classify_api_error(message: str, code: int | None) -> TranslationError:
    if code == 429 or _is_quota_error(message):
        return QuotaExceededError(message)
    if code in {401, 403} or _is_auth_error(message):
        return TranslationError(message)
    if (code is not None and code >= 500) or _is_transient_error(message):
        return TransientServiceError(message)
    return TranslationError(message)
