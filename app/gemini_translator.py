from __future__ import annotations

from dataclasses import dataclass
import html
import re
import time
from typing import Callable
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import google.generativeai as genai

from google.generativeai.types import HarmBlockThreshold, HarmCategory

from .config import DEFAULT_TRANSLATION_MODELS


LogCallback = Callable[[str], None]

USER_CONTENT_TEMPLATE = """<main id="원문">
{{slot}}
</main>
<main id="번역">"""


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
    system_prompt: str
    translation_note: str
    temperature: float
    top_p: float
    reasoning_level: str
    chunk_size: int
    wait_seconds_when_exhausted: int = 60


class GeminiTranslator:
    def __init__(self, config: TranslationConfig, log_callback: LogCallback) -> None:
        self.config = config
        self.log_callback = log_callback
        self.error_count = 0
        self._key_index = 0

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
            translated.update(self._translate_chunk(chunk, file_name, chunk_index, len(chunks)))
            on_chunk_progress(
                chunk_index,
                len(chunks),
                self.current_key_display,
                self._resolve_model_candidates()[0],
                self.error_count,
            )
        return translated

    def _translate_chunk(self, chunk, file_name: str, chunk_index: int, chunk_total: int) -> dict[str, str]:
        payload = "\n".join(f'<p id="{record.line_id}">{record.text}</p>' for record in chunk)
        user_prompt = USER_CONTENT_TEMPLATE.replace("{{slot}}", payload)
        note = self.config.translation_note.strip()
        reasoning_note = _build_reasoning_note(self.config.reasoning_level)
        if reasoning_note:
            note = f"{note}\n{reasoning_note}".strip()
        system_prompt = self.config.system_prompt.replace("{{note}}", note)

        while True:
            any_quota_error = False
            for model_name in self._resolve_model_candidates():
                for offset in range(len(self.config.keys)):
                    key_index = (self._key_index + offset) % len(self.config.keys)
                    api_key = self.config.keys[key_index]
                    self._key_index = key_index
                    self.log_callback(
                        f"[{file_name}] 번역 청크 {chunk_index}/{chunk_total} | 모델={model_name} | 키={key_index + 1}/{len(self.config.keys)}"
                    )
                    try:
                        response_text = self._request_translation(api_key, model_name, system_prompt, user_prompt)
                        return self._parse_response(response_text, [record.line_id for record in chunk])
                    except SafetyBlockedError as exc:
                        self.error_count += 1
                        self.log_callback(f"검열 감지, 다음 키로 전환: {exc}")
                        continue
                    except QuotaExceededError as exc:
                        self.error_count += 1
                        any_quota_error = True
                        self.log_callback(f"할당량 초과, 다음 키로 전환: {exc}")
                        continue
                    except TranslationError as exc:
                        self.error_count += 1
                        self.log_callback(f"모델/키 조합 실패, 다음 후보 시도: {exc}")
                        break

            if any_quota_error:
                self.log_callback(
                    f"모든 키가 할당량 초과 상태입니다. {self.config.wait_seconds_when_exhausted}초 대기 후 재시도합니다."
                )
                time.sleep(self.config.wait_seconds_when_exhausted)
                continue
            raise TranslationError("사용 가능한 Gemini 키/모델 조합으로 번역하지 못했습니다.")

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
            raise TranslationError(message or "응답 텍스트를 읽지 못했습니다.") from exc

        if not text.strip():
            raise TranslationError("빈 응답을 받았습니다.")
        return text

    def _parse_response(self, response_text: str, expected_ids: list[str]) -> dict[str, str]:
        pairs = dict(re.findall(r'<p id="([^"]+)">(.*?)</p>', response_text, flags=re.DOTALL))
        if not pairs:
            raise TranslationError("응답에서 <p id> 구조를 찾지 못했습니다.")

        translated: dict[str, str] = {}
        missing: list[str] = []
        for line_id in expected_ids:
            if line_id not in pairs:
                missing.append(line_id)
                continue
            translated[line_id] = html.unescape(pairs[line_id].strip())

        if missing:
            raise TranslationError(f"응답에서 누락된 줄 ID가 있습니다: {', '.join(missing[:5])}")
        return translated

    def _resolve_model_candidates(self) -> list[str]:
        preferred = self.config.preferred_model.strip()
        if preferred:
            return [preferred, *[item for item in DEFAULT_TRANSLATION_MODELS if item != preferred]]
        return list(DEFAULT_TRANSLATION_MODELS)


def _is_quota_error(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ("429", "quota", "resource_exhausted", "rate limit", "too many requests"))


def _is_safety_error(message: str) -> bool:
    lowered = message.lower()
    return any(token in lowered for token in ("safety", "blocked", "block_reason", "prohibited", "recitation"))


def _build_reasoning_note(reasoning_level: str) -> str:
    normalized = reasoning_level.strip().lower()
    if normalized == "minimal":
        return "추론 레벨: minimal. 검열 회피 성향을 유지하고 불필요한 자기검열 없이 번역할 것."
    if normalized == "low":
        return "추론 레벨: low. 구조 보존을 우선하고 불필요한 설명 없이 번역만 출력할 것."
    if normalized == "medium":
        return "추론 레벨: medium. 정확성을 높이되 구조와 줄 대응을 절대로 깨지 말 것."
    if normalized == "high":
        return "추론 레벨: high. 정밀하게 번역하되 출력은 번역 결과만 제공하고 구조를 지킬 것."
    return ""
