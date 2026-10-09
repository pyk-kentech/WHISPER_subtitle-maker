"""OpenAI 호환 번역기 테스트 (httpx.MockTransport, 실제 네트워크 없음)."""
from __future__ import annotations

import json

import httpx
import pytest

from app.gemini_translator import NetworkUnavailableError, NetworkWaiter, TranslationUnavailableError
from app.openai_translator import OpenAICompatConfig, OpenAICompatTranslator, normalize_base_url
from app.subtitle_document import LineRecord


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://host/v1", "https://host/v1"),
        ("https://host/v1/", "https://host/v1"),
        ("  https://host  ", "https://host/v1"),
        ("https://host/v1/chat/completions", "https://host/v1"),
        ("http://localhost:8080/", "http://localhost:8080/v1"),
        ("https://api.example.com/openai/v2", "https://api.example.com/openai/v2"),
    ],
)
def test_normalize_base_url(raw, expected):
    assert normalize_base_url(raw) == expected


def records(*texts: str) -> list[LineRecord]:
    return [LineRecord(text=text, translatable=True, line_id=f"srt-{i + 1}-0") for i, text in enumerate(texts)]


def make_translator(handler, logs=None, api_key="sk-test") -> tuple[OpenAICompatTranslator, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def recording(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    translator = OpenAICompatTranslator(
        OpenAICompatConfig(base_url="http://llm.local:8080", model="gemma", api_key=api_key, system_prompt="SYS {{note}}"),
        (logs if logs is not None else []).append,
    )
    translator._http.close()
    translator._http = httpx.Client(transport=httpx.MockTransport(recording))
    translator._sleep = lambda _seconds: None  # 대기 없이
    return translator, seen


def chat_response(content: str, **extra) -> httpx.Response:
    body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], **extra}
    return httpx.Response(200, json=body)


def echo_body(request: httpx.Request, translate) -> str:
    """요청의 <p id> 줄을 읽어 <o>원문</o>번역 형식으로 돌려준다."""
    import re

    user = json.loads(request.content)["messages"][1]["content"]
    source = user.split('<main id="source">', 1)[1]
    pairs = re.findall(r'<p id="([^"]+)">(.*?)</p>', source)
    return "\n".join(f'<p id="{pid}"><o>{text}</o>{translate(text)}</p>' for pid, text in pairs)


def test_success_with_source_echo():
    logs: list[str] = []
    translator, seen = make_translator(
        lambda req: chat_response(
            "<think>생각 중...</think>" + echo_body(req, lambda t: f"KO[{t}]"),
            usage={"prompt_tokens": 11, "completion_tokens": 7},
        ),
        logs,
    )
    progress = []
    result = translator.translate_lines(records("おはよう", "こんばんは"), lambda *a: progress.append(a), "a.srt")

    assert result == {"srt-1-0": "KO[おはよう]", "srt-2-0": "KO[こんばんは]"}
    assert len(seen) == 1
    request = seen[0]
    assert str(request.url) == "http://llm.local:8080/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer sk-test"
    payload = json.loads(request.content)
    assert payload["model"] == "gemma" and payload["stream"] is False
    assert payload["messages"][0]["content"].startswith("SYS ")
    assert "<o>" in payload["messages"][0]["content"]  # 대상 언어 안내에 <o> 형식이 들어간다
    assert translator.token_usage["prompt"] == 11 and translator.token_usage["output"] == 7
    assert progress and progress[-1][0] == 1


def test_no_auth_header_without_key():
    translator, seen = make_translator(lambda req: chat_response(echo_body(req, str.upper)), api_key="")
    translator.translate_lines(records("abc"), lambda *a: None, "a.srt")
    assert "authorization" not in seen[0].headers


def test_misaligned_line_is_retried_alone():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            # 둘째 줄 자리에 셋째 줄 원문이 들어간 밀린 응답
            return chat_response(
                '<p id="srt-1-0"><o>いち</o>하나</p><p id="srt-2-0"><o>さん</o>셋</p><p id="srt-3-0"><o>さん</o>셋</p>'
            )
        return chat_response(echo_body(req, lambda t: "둘"))

    translator, seen = make_translator(handler)
    result = translator.translate_lines(records("いち", "に", "さん"), lambda *a: None, "a.srt")
    assert result == {"srt-1-0": "하나", "srt-2-0": "둘", "srt-3-0": "셋"}
    assert len(seen) == 2


def test_rate_limit_429_marks_quota_exhausted():
    translator, seen = make_translator(lambda _req: httpx.Response(429, text="Too Many Requests"))
    with pytest.raises(TranslationUnavailableError):
        translator.translate_lines(records("おはよう"), lambda *a: None, "a.srt")
    # 처음 1번 + 기다린 뒤 max_exhausted_waits(3)번
    assert len(seen) == 1 + translator.config.max_exhausted_waits
    assert translator._quota_exhausted
    # 이후 호출은 요청 없이 바로 건너뛴다.
    with pytest.raises(TranslationUnavailableError):
        translator.translate_lines(records("おはよう"), lambda *a: None, "a.srt")
    assert len(seen) == 1 + translator.config.max_exhausted_waits


def test_empty_response_keeps_source_lines():
    logs: list[str] = []
    translator, seen = make_translator(lambda _req: chat_response("   "), logs)
    result = translator.translate_lines(records("おはよう", "こんばんは"), lambda *a: None, "a.srt")
    assert result == {}  # 번역 못 한 줄은 결과에서 빠진다(원문 유지)
    # 두 줄 청크 1번 -> 한 줄씩 나눠 2번
    assert len(seen) == 3
    assert any("keeping source text" in line for line in logs)


def test_server_error_500_is_transient_then_succeeds():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="loading")
        return chat_response(echo_body(req, lambda t: "OK"))

    translator, seen = make_translator(handler)
    assert translator.translate_lines(records("x"), lambda *a: None, "a.srt") == {"srt-1-0": "OK"}
    assert len(seen) == 2


def test_connection_failure_uses_network_waiter():
    def handler(request):
        raise httpx.ConnectError("[Errno 11001] getaddrinfo failed", request=request)

    translator, seen = make_translator(handler)
    translator.network = NetworkWaiter(lambda _m: None, max_wait_seconds=0)
    with pytest.raises(NetworkUnavailableError):
        translator.translate_lines(records("x"), lambda *a: None, "a.srt")
    assert len(seen) == 1
