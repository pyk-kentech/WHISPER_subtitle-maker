"""Gemini 응답 파싱(<o> 원문 되풀이 검사)과 번역기 묶음(create_translators/translate_records) 테스트."""
from __future__ import annotations

import pytest

from app.gemini_translator import (
    GeminiTranslator,
    ResponseFormatError,
    TranslationConfig,
    TranslationUnavailableError,
    _echo_matches,
)
from app.openai_translator import OpenAICompatTranslator
from app.subtitle_document import LineRecord
from app.translator_store import TranslatorSettings
from app.workers import TranslatorChain, create_translators, translate_records


def make_gemini(logs=None) -> GeminiTranslator:
    config = TranslationConfig(
        keys=["k1"],
        preferred_model="",
        target_language="ko",
        system_prompt="{{note}}",
        translation_note="",
        temperature=1.0,
        top_p=0.8,
        reasoning_level="minimal",
        chunk_size=10,
        request_delay_seconds=0.0,
    )
    return GeminiTranslator(config, (logs if logs is not None else []).append)


def records(*texts: str) -> list[LineRecord]:
    return [LineRecord(text=text, translatable=True, line_id=f"srt-{i + 1}-0") for i, text in enumerate(texts)]


# ----- _echo_matches -----

def test_echo_exact_match():
    assert _echo_matches("こんにちは、世界", "こんにちは、世界")


def test_echo_ignores_punctuation_and_width():
    assert _echo_matches("ＡＢＣ、テスト！", "ABC テスト")


def test_echo_mismatch_rejected():
    assert not _echo_matches("おはようございます", "ありがとう")


def test_echo_closer_to_neighbor_rejected():
    # 되풀이가 이웃 줄과 더 닮았으면(줄 밀림) 거절한다.
    assert not _echo_matches("今日はいい天気ですね", "今日はいい天気ですねえ", ["今日はいい天気ですねえ"])


def test_echo_empty_source_always_ok():
    assert _echo_matches("…!?", "whatever")


# ----- _parse_response -----

def test_parse_response_with_echo():
    translator = make_gemini()
    chunk = records("おはよう", "こんばんは")
    text = (
        '<p id="srt-1-0"><o>おはよう</o>좋은 아침</p>\n'
        '<p id="srt-2-0"><o>こんばんは</o>좋은 저녁 &amp; 밤</p>'
    )
    translated, rejected = translator._parse_response(text, chunk, "a.srt")
    assert translated == {"srt-1-0": "좋은 아침", "srt-2-0": "좋은 저녁 & 밤"}
    assert rejected == []


def test_parse_response_rejects_shifted_and_missing_lines():
    translator = make_gemini()
    chunk = records("おはよう", "こんばんは", "さようなら")
    text = (
        '<p id="srt-1-0"><o>おはよう</o>좋은 아침</p>'
        '<p id="srt-2-0"><o>さようなら</o>안녕히</p>'  # 다음 줄 원문이 들어옴 -> 밀림
    )
    translated, rejected = translator._parse_response(text, chunk, "a.srt")
    assert translated == {"srt-1-0": "좋은 아침"}
    assert rejected == ["srt-2-0", "srt-3-0"]


def test_parse_response_without_echo_is_accepted_with_log():
    logs: list[str] = []
    translator = make_gemini(logs)
    chunk = records("おはよう")
    translated, rejected = translator._parse_response('<p id="srt-1-0">좋은 아침</p>', chunk, "a.srt")
    assert translated == {"srt-1-0": "좋은 아침"}
    assert rejected == []
    assert any("no <o> source echo" in line for line in logs)


def test_parse_response_empty_translation_rejected():
    translator = make_gemini()
    chunk = records("おはよう")
    translated, rejected = translator._parse_response('<p id="srt-1-0"><o>おはよう</o>  </p>', chunk, "a.srt")
    assert translated == {}
    assert rejected == ["srt-1-0"]


def test_parse_response_without_pairs_raises():
    with pytest.raises(ResponseFormatError):
        make_gemini()._parse_response("sorry, I can't", records("x"), "a.srt")


# ----- create_translators -----

def test_create_translators_share_one_network_waiter():
    settings = TranslatorSettings(
        use_deepl_fallback=True,
        use_openai_fallback=True,
        openai_base_url="http://localhost:8080",
        openai_model="local-model",
    )
    cancel = lambda: False  # noqa: E731
    chain = create_translators(settings, ["g1"], "deepl-key", "ja", lambda _m: None, cancel, "sk")
    assert isinstance(chain.primary, GeminiTranslator) and not isinstance(chain.primary, OpenAICompatTranslator)
    assert isinstance(chain.secondary, OpenAICompatTranslator)
    assert chain.deepl is not None
    assert chain.primary.network is chain.secondary.network is chain.deepl.network
    assert chain.primary.cancel_check is cancel and chain.deepl.cancel_check is cancel
    chain.secondary.close()


def test_create_translators_openai_provider_requires_url():
    with pytest.raises(RuntimeError):
        create_translators(TranslatorSettings(provider="openai"), ["g1"], "", "ja", lambda _m: None)


def test_create_translators_openai_provider_is_primary():
    settings = TranslatorSettings(provider="openai", openai_base_url="http://h/v1", openai_model="m")
    chain = create_translators(settings, ["g1"], "", "ja", lambda _m: None)
    assert isinstance(chain.primary, OpenAICompatTranslator)
    assert chain.secondary is None
    chain.primary.close()


def test_create_translators_without_any_translator_raises():
    with pytest.raises(RuntimeError):
        create_translators(TranslatorSettings(), [], "", "ja", lambda _m: None)


# ----- translate_records -----

class FakeTranslator:
    def __init__(self, result=None, error=None, partial=None) -> None:
        self.result = result or {}
        self.error = error
        self.partial = partial
        self.received: list[str] = []
        self.compat = type("Compat", (), {"model": "fake"})()

    def translate_lines(self, recs, _progress, _file_name):
        self.received = [record.line_id for record in recs]
        if self.error is not None:
            exc = self.error
            if self.partial is not None:
                exc.partial = dict(self.partial)
            raise exc
        return {line_id: text for line_id, text in self.result.items() if line_id in self.received}


def test_translate_records_all_lines_by_primary():
    recs = records("a", "b")
    primary = FakeTranslator({"srt-1-0": "A", "srt-2-0": "B"})
    outcome = translate_records(recs, TranslatorChain(primary, None, None), None, "f", lambda _m: None)
    assert outcome.translations == {"srt-1-0": "A", "srt-2-0": "B"}
    assert outcome.missing_count == 0 and not outcome.failed


def test_translate_records_secondary_fills_after_primary_abort():
    recs = records("a", "b", "c")
    primary = FakeTranslator(error=TranslationUnavailableError("quota"), partial={"srt-1-0": "A"})
    secondary = FakeTranslator({"srt-2-0": "B", "srt-3-0": "C"})
    outcome = translate_records(recs, TranslatorChain(primary, secondary, None), None, "f", lambda _m: None)
    assert secondary.received == ["srt-2-0", "srt-3-0"]
    assert outcome.translations == {"srt-1-0": "A", "srt-2-0": "B", "srt-3-0": "C"}
    assert not outcome.failed


def test_translate_records_primary_abort_with_missing_is_failed():
    recs = records("a", "b")
    primary = FakeTranslator(error=TranslationUnavailableError("quota"), partial={"srt-1-0": "A"})
    outcome = translate_records(recs, TranslatorChain(primary, None, None), None, "f", lambda _m: None)
    assert outcome.failed
    assert outcome.missing_count == 1


def test_translate_records_partial_without_abort_is_not_failed():
    """번역기가 끝까지 돌았는데 몇 줄만 빠졌으면 실패가 아니라 '원문 유지'로 저장한다."""
    recs = records("a", "b")
    primary = FakeTranslator({"srt-1-0": "A"})
    outcome = translate_records(recs, TranslatorChain(primary, None, None), None, "f", lambda _m: None)
    assert not outcome.failed
    assert outcome.missing_count == 1


def test_translate_records_deepl_last_resort():
    recs = records("a", "b")
    primary = FakeTranslator({"srt-1-0": "A"})
    deepl = FakeTranslator({"srt-2-0": "B"})
    outcome = translate_records(recs, TranslatorChain(primary, None, deepl), None, "f", lambda _m: None)
    assert deepl.received == ["srt-2-0"]
    assert outcome.missing_count == 0 and not outcome.failed
