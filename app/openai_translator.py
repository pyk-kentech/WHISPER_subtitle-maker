"""OpenAI 호환(Chat Completions) 번역기: Runpod에 띄운 LLM, llama.cpp `llama-server`, Ollama, vLLM 등.

Gemini 번역기와 같은 시스템 프롬프트, 같은 입력 형식(<p id>·<o> 원문 되풀이, 번역 노트, 문맥 줄),
같은 청크 분할·줄 밀림 검사·재시도·네트워크 대기를 쓴다. 요청 부분만 OpenAI 형식으로 바꾼다.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import httpx

from .gemini_translator import (
    GeminiTranslator,
    TranslationUnavailableError,
    QuotaExceededError,
    ResponseFormatError,
    SafetyBlockedError,
    TransientServiceError,
    TranslationConfig,
    TranslationError,
    _classify_api_error,
)

# 추론 모델이 내놓는 생각 블록은 번역 결과가 아니므로 지운다.
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
REQUEST_TIMEOUT_SECONDS = 600.0
DEFAULT_MAX_TOKENS = 8192


@dataclass(slots=True)
class OpenAICompatConfig:
    base_url: str
    model: str
    api_key: str = ""
    target_language: str = "ko"
    system_prompt: str = ""
    translation_note: str = ""
    temperature: float = 0.7
    top_p: float = 0.9
    chunk_size: int = 60
    # 자기 서버라 무료 한도 보호용 지연이 필요 없다.
    request_delay_seconds: float = 0.0
    max_tokens: int = DEFAULT_MAX_TOKENS
    # 한 번에 보내는 청크 요청 수. llama-server --parallel N이면 N개를 동시에 처리해 처리량이 크게 는다.
    parallel_requests: int = 2


def normalize_base_url(url: str) -> str:
    """'https://host/v1', 'https://host/v1/', 'https://host' 모두 받는다(마지막에 /v1이 없으면 붙인다)."""
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith("/chat/completions"):
        cleaned = cleaned[: -len("/chat/completions")]
    if not re.search(r"/v\d+$", cleaned):
        cleaned += "/v1"
    return cleaned


class OpenAICompatTranslator(GeminiTranslator):
    provider_name = "OpenAI-compatible LLM"

    def __init__(self, config: OpenAICompatConfig, log_callback) -> None:
        super().__init__(
            TranslationConfig(
                keys=[config.api_key or "-"],
                preferred_model=config.model,
                target_language=config.target_language,
                system_prompt=config.system_prompt,
                translation_note=config.translation_note,
                temperature=config.temperature,
                top_p=config.top_p,
                reasoning_level="",
                chunk_size=config.chunk_size,
                request_delay_seconds=config.request_delay_seconds,
                min_adaptive_delay_seconds=0.0,
                wait_seconds_when_exhausted=20,
            ),
            log_callback,
        )
        self.compat = config
        self.base_url = normalize_base_url(config.base_url)
        self._http = httpx.Client(timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=20.0))

    def translate_lines(self, records, on_chunk_progress, file_name: str) -> dict[str, str]:
        """청크를 여러 개 동시에 보낸다(순서·검사·재시도는 청크마다 기존과 같다)."""
        workers = max(1, int(self.compat.parallel_requests))
        if self._quota_exhausted:
            raise TranslationUnavailableError("LLM 서버가 계속 요청을 거절해(429) 이번 작업에서는 건너뜁니다.")
        if workers == 1:
            return super().translate_lines(records, on_chunk_progress, file_name)
        size = self.config.chunk_size
        chunks = [records[index : index + size] for index in range(0, len(records), size)]
        self._records = list(records)
        self._record_positions = {record.line_id: position for position, record in enumerate(records)}
        translated: dict[str, str] = {}
        done = 0
        failure: TranslationUnavailableError | None = None
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="llm-chunk") as pool:
            futures = [
                pool.submit(self._translate_chunk_with_split, chunk, file_name, index, len(chunks))
                for index, chunk in enumerate(chunks, start=1)
            ]
            for future in futures:
                try:
                    translated.update(future.result())
                except TranslationUnavailableError as exc:
                    translated.update(getattr(exc, "partial", None) or {})
                    failure = failure or exc
                done += 1
                on_chunk_progress(done, len(chunks), self.current_key_display, self.compat.model, self.error_count)
        if failure is not None:
            failure.partial = translated
            raise failure
        return translated

    def _resolve_model_candidates(self) -> list[str]:
        model = self.compat.model.strip()
        return [model] if model and model not in self._unavailable_models else []

    def _request_translation(self, api_key: str, model_name: str, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if api_key and api_key != "-":
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.compat.max_tokens,
            "stream": False,
        }
        try:
            response = self._http.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise TransientServiceError(f"timeout: {exc}") from exc
        except httpx.TransportError as exc:
            # 연결 자체가 안 되면(DNS 실패 등) NetworkWaiter가 원인 예외를 보고 연결을 기다린다.
            raise TransientServiceError(str(exc) or exc.__class__.__name__) from exc
        if response.status_code >= 400:
            message = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code == 429:
                raise QuotaExceededError(message)
            if response.status_code in {502, 503, 504} or response.status_code >= 520:
                # Runpod 프록시는 서버가 아직 뜨는 중이거나 워커가 깨는 중일 때 502/503/524를 돌려준다.
                raise TransientServiceError(message)
            raise _classify_api_error(message, response.status_code)
        try:
            data = response.json()
        except ValueError as exc:
            raise ResponseFormatError(f"not JSON: {response.text[:200]}") from exc
        self._log_openai_usage(model_name, data)
        choices = data.get("choices") or []
        if not choices:
            raise ResponseFormatError("Received no choices.")
        choice = choices[0]
        finish_reason = str(choice.get("finish_reason") or "")
        if finish_reason == "content_filter":
            raise SafetyBlockedError("finish_reason=content_filter")
        message = choice.get("message") or {}
        text = _THINK_PATTERN.sub("", str(message.get("content") or "")).strip()
        if not text:
            raise ResponseFormatError(f"Received an empty response (finish_reason={finish_reason or 'none'}).")
        return text

    def _log_openai_usage(self, model_name: str, data: dict) -> None:
        usage = data.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        output = int(usage.get("completion_tokens") or 0)
        self.token_usage["prompt"] += prompt
        self.token_usage["output"] += output
        if prompt or output:
            self.log_callback(f"tokens | model={model_name} | prompt={prompt} output={output}")

    @property
    def token_usage_summary(self) -> str:
        usage = self.token_usage
        return f"LLM({self.compat.model}) 누적 토큰 | 입력 {usage['prompt']:,} | 출력 {usage['output']:,}"

    def close(self) -> None:
        self._http.close()


def check_openai_endpoint(base_url: str, api_key: str = "", timeout: float = 15.0) -> list[str]:
    """/v1/models로 연결과 키를 확인하고 모델 이름 목록을 돌려준다(설정 화면의 '연결 확인'용)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    response = httpx.get(f"{normalize_base_url(base_url)}/models", headers=headers, timeout=timeout)
    if response.status_code >= 400:
        raise TranslationError(f"HTTP {response.status_code}: {response.text[:200]}")
    data = response.json()
    return [str(item.get("id")) for item in data.get("data", []) if item.get("id")]
