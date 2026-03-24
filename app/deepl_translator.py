from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable
from urllib import error, request

from .config import DEEPL_FREE_API_URL
from .gemini_translator import TranslationError


LogCallback = Callable[[str], None]

DEEPL_SOURCE_LANGUAGE_MAP = {
    "auto": "",
    "en": "EN",
    "ja": "JA",
    "ko": "KO",
    "zh": "ZH",
}

DEEPL_TARGET_LANGUAGE_MAP = {
    "en": "EN",
    "ja": "JA",
    "ko": "KO",
}

MAX_TEXTS_PER_REQUEST = 50
MAX_REQUEST_BODY_BYTES = 120 * 1024


class DeepLAuthError(TranslationError):
    pass


class DeepLQuotaExceededError(TranslationError):
    pass


@dataclass(slots=True)
class DeepLConfig:
    api_key: str
    target_language: str
    source_language: str
    chunk_size: int
    request_delay_seconds: float


class DeepLTranslator:
    def __init__(self, config: DeepLConfig, log_callback: LogCallback) -> None:
        self.config = config
        self.log_callback = log_callback
        self.error_count = 0
        self._last_request_monotonic = 0.0

    @property
    def current_key_display(self) -> str:
        return "DeepL"

    def translate_lines(
        self,
        records,
        on_chunk_progress: Callable[[int, int, str, str, int], None],
        file_name: str,
    ) -> dict[str, str]:
        chunks = self._build_chunks(records)
        translated: dict[str, str] = {}
        for chunk_index, chunk in enumerate(chunks, start=1):
            self.log_callback(f"[{file_name}] chunk {chunk_index}/{len(chunks)} | provider=deepl-free-api")
            self._sleep_for_request_spacing()
            chunk_result = self._translate_chunk(chunk)
            self._last_request_monotonic = time.monotonic()
            translated.update(chunk_result)
            on_chunk_progress(chunk_index, len(chunks), self.current_key_display, "deepl-free-api", self.error_count)
        return translated

    def _build_chunks(self, records) -> list[list]:
        chunks: list[list] = []
        current: list = []
        current_bytes = 0
        max_items = max(1, min(self.config.chunk_size, MAX_TEXTS_PER_REQUEST))

        for record in records:
            text_bytes = len(record.text.encode("utf-8"))
            next_bytes = current_bytes + text_bytes + 64
            if current and (len(current) >= max_items or next_bytes >= MAX_REQUEST_BODY_BYTES):
                chunks.append(current)
                current = []
                current_bytes = 0
            current.append(record)
            current_bytes += text_bytes + 64

        if current:
            chunks.append(current)
        return chunks

    def _translate_chunk(self, chunk) -> dict[str, str]:
        texts = [record.text for record in chunk]
        translated_texts = self._request_translation(texts)
        if len(translated_texts) != len(chunk):
            raise TranslationError(
                f"DeepL returned {len(translated_texts)} translations for {len(chunk)} input lines."
            )
        return {
            record.line_id: translated_text
            for record, translated_text in zip(chunk, translated_texts)
            if record.line_id is not None
        }

    def _request_translation(self, texts: list[str]) -> list[str]:
        target_language = DEEPL_TARGET_LANGUAGE_MAP.get(self.config.target_language, "")
        if not target_language:
            raise TranslationError(f"DeepL does not support target language '{self.config.target_language}'.")

        source_language = DEEPL_SOURCE_LANGUAGE_MAP.get(self.config.source_language, "")
        context = "\n".join(text for text in texts if text.strip())
        payload = {
            "text": texts,
            "target_lang": target_language,
            "model_type": "prefer_quality_optimized",
            "preserve_formatting": True,
        }
        if source_language:
            payload["source_lang"] = source_language
        if context:
            payload["context"] = context

        body = json.dumps(payload).encode("utf-8")
        request_obj = request.Request(
            DEEPL_FREE_API_URL,
            data=body,
            headers={
                "Authorization": f"DeepL-Auth-Key {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(request_obj, timeout=120) as response:
                raw = response.read()
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            if exc.code in {401, 403}:
                raise DeepLAuthError(message or f"DeepL authentication failed with HTTP {exc.code}.") from exc
            if exc.code in {429, 456}:
                raise DeepLQuotaExceededError(message or f"DeepL quota exceeded with HTTP {exc.code}.") from exc
            raise TranslationError(message or f"DeepL request failed with HTTP {exc.code}.") from exc
        except error.URLError as exc:
            raise TranslationError(str(exc.reason) or "DeepL request failed.") from exc

        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise TranslationError("DeepL returned an invalid JSON response.") from exc

        translations = data.get("translations")
        if not isinstance(translations, list):
            raise TranslationError("DeepL response does not contain a valid 'translations' list.")

        results: list[str] = []
        for item in translations:
            text = item.get("text") if isinstance(item, dict) else None
            if not isinstance(text, str):
                raise TranslationError("DeepL response contains an invalid translation entry.")
            results.append(text)

        return results

    def _sleep_for_request_spacing(self) -> None:
        delay = max(0.0, float(self.config.request_delay_seconds))
        if delay <= 0:
            return
        if self._last_request_monotonic <= 0:
            self.log_callback(f"Request delay {delay:.1f}s before DeepL call")
            time.sleep(delay)
            return
        elapsed = time.monotonic() - self._last_request_monotonic
        remaining = delay - elapsed
        if remaining > 0:
            self.log_callback(f"Request delay {remaining:.1f}s before DeepL call")
            time.sleep(remaining)
