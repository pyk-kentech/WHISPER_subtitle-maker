from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from sudachipy import tokenizer as sudachi_tokenizer

from .dictionary_pack import create_dictionary_tokenizer
from .srt_writer import SubtitleSegment


JP_COMMA = "\u3001"
JP_PERIOD = "\u3002"
JP_EXCL = "\uff01"
JP_QUESTION = "\uff1f"
JP_OPEN_QUOTES = {"\u300c", "\u300e", "\uff08"}
JP_CLOSE_QUOTES = {"\u300d", "\u300f", "\uff09"}
JP_PUNCT = {JP_COMMA, JP_PERIOD, JP_EXCL, JP_QUESTION, *JP_CLOSE_QUOTES}

_JAPANESE_CHAR = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
_JP_BOUNDARY = re.compile(rf"(?<=[{_JAPANESE_CHAR}])\s+(?=[{_JAPANESE_CHAR}])")
_SPACE_BEFORE_PUNCT = re.compile(rf"\s+([{JP_COMMA}{JP_PERIOD}{JP_EXCL}{JP_QUESTION}\u30fb\u300d\u300f\uff09])")
_SPACE_AFTER_OPEN = re.compile(r"([\u300c\u300e\uff08])\s+")
_MULTI_SPACE = re.compile(r"\s+")
_DUPLICATED_PUNCT = re.compile(rf"([{JP_COMMA}{JP_PERIOD}{JP_EXCL}{JP_QUESTION}])\1+")
_JP_DOT_BETWEEN = re.compile(rf"(?<=[{_JAPANESE_CHAR}])\.(?=[{_JAPANESE_CHAR}])")
_JP_DOT_BEFORE_END = re.compile(rf"(?<=[{_JAPANESE_CHAR}])\s*\.\s*(?=$)")
_JP_DOT_AT_END = re.compile(rf"(?<=[{_JAPANESE_CHAR}])\.$")
_ASCII_SPACE_BEFORE_JP_PUNCT = re.compile(rf"(?<=[0-9A-Za-z])\s+(?=[{JP_COMMA}{JP_PERIOD}{JP_EXCL}{JP_QUESTION}])")
_SPACE_AFTER_JP_PUNCT = re.compile(rf"(?<=[{JP_COMMA}{JP_PERIOD}{JP_EXCL}{JP_QUESTION}])\s+")
_ENDS_WITH_JP_CHAR = re.compile(rf"[{_JAPANESE_CHAR}]$")
_STARTS_WITH_JP_CHAR = re.compile(rf"^[{_JAPANESE_CHAR}]")
_NFKC_TRIGGER = re.compile(r"[\uff00-\uffef\u3000]")
_TOKENIZER_CACHE: dict[str, sudachi_tokenizer.Tokenizer] = {}

_CONTINUATION_ENDINGS = (
    "\u306f",
    "\u304c",
    "\u3092",
    "\u306b",
    "\u3067",
    "\u3068",
    "\u3082",
    "\u306e",
    "\u3078",
    "\u304b\u3089",
    "\u307e\u3067",
    "\u3088\u308a",
    "\u3063\u3066",
    "\u3057",
    "\u3066",
    "\u3051\u3069",
    "\u3051\u308c\u3069",
    "\u306e\u3067",
    "\u306e\u306b",
)
_CONTINUATION_STARTS = (
    "\u3066",
    "\u3067",
    "\u3068",
    "\u306e",
    "\u304c",
    "\u3092",
    "\u306b",
    "\u306f",
    "\u3082",
    "\u304b\u3089",
    "\u3051\u3069",
    "\u3051\u308c\u3069",
    "\u306e\u3067",
    "\u306e\u306b",
)
_TERMINAL_ENDINGS = (
    "\u3067\u3059",
    "\u307e\u3059",
    "\u3067\u3057\u305f",
    "\u307e\u3057\u305f",
    "\u3060",
    "\u3060\u3063\u305f",
    "\u3067\u3057\u3087\u3046",
    "\u307e\u3059\u306d",
    "\u3067\u3059\u306d",
    "\u304b\u306a",
    "\u304b\u3082",
    "\u3088",
    "\u306d",
    "\u3088\u306d",
    "\u305e",
    "\u305c",
    "\u304b",
)
_SINGLE_WORD_POS = {
    "\u540d\u8a5e",
    "\u52d5\u8a5e",
    "\u5f62\u5bb9\u8a5e",
    "\u526f\u8a5e",
    "\u611f\u52d5\u8a5e",
}


@dataclass(slots=True)
class PostprocessOptions:
    enabled: bool = True
    enhanced: bool = False
    sentence: bool = True
    standard_asia: bool = True
    max_comma: int = 2
    max_gap: float = 0.35
    one_word: bool = True


def _normalize_punctuation(text: str) -> str:
    normalized = text
    normalized = normalized.replace("\uff64", JP_COMMA).replace("\uff61", JP_PERIOD)
    normalized = normalized.replace("\uff0c", JP_COMMA).replace(",", JP_COMMA)
    normalized = normalized.replace("\uff0e", JP_PERIOD)
    normalized = _JP_DOT_BETWEEN.sub(JP_PERIOD, normalized)
    normalized = _JP_DOT_BEFORE_END.sub(JP_PERIOD, normalized)
    normalized = _JP_DOT_AT_END.sub(JP_PERIOD, normalized)
    normalized = normalized.replace("\uff62", "\u300c").replace("\uff63", "\u300d")
    normalized = normalized.replace("(", "\uff08").replace(")", "\uff09")
    normalized = normalized.replace("\uff08 ", "\uff08").replace(" \uff09", "\uff09")
    normalized = normalized.replace(" \u30fb ", "\u30fb").replace("\u30fb ", "\u30fb").replace(" \u30fb", "\u30fb")
    normalized = _SPACE_BEFORE_PUNCT.sub(r"\1", normalized)
    normalized = _SPACE_AFTER_OPEN.sub(r"\1", normalized)
    normalized = _DUPLICATED_PUNCT.sub(r"\1", normalized)
    return normalized


def _limit_commas(text: str, max_comma: int) -> str:
    if max_comma < 0:
        return text
    count = 0
    output: list[str] = []
    for char in text:
        if char == JP_COMMA:
            count += 1
            if count > max_comma:
                output.append(" ")
                continue
        output.append(char)
    return "".join(output)


def _get_tokenizer(split_mode: sudachi_tokenizer.Tokenizer.SplitMode) -> sudachi_tokenizer.Tokenizer:
    cache_key = str(split_mode)
    tokenizer_obj = _TOKENIZER_CACHE.get(cache_key)
    if tokenizer_obj is None:
        tokenizer_obj = create_dictionary_tokenizer(split_mode)
        _TOKENIZER_CACHE[cache_key] = tokenizer_obj
    return tokenizer_obj


def _normalize_input_text(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    if _NFKC_TRIGGER.search(stripped):
        return unicodedata.normalize("NFKC", stripped)
    return stripped


def postprocess_japanese_text(text: str, options: PostprocessOptions | None = None) -> str:
    config = options or PostprocessOptions()
    if not config.enabled:
        return text.strip()

    normalized = _normalize_input_text(text)
    if not normalized:
        return ""

    normalized = _MULTI_SPACE.sub(" ", normalized)
    if config.standard_asia:
        normalized = _normalize_punctuation(normalized)
        normalized = _JP_BOUNDARY.sub("", normalized)
        normalized = _ASCII_SPACE_BEFORE_JP_PUNCT.sub("", normalized)
        normalized = _SPACE_AFTER_JP_PUNCT.sub("", normalized)

    normalized = _limit_commas(normalized, config.max_comma)
    normalized = _MULTI_SPACE.sub(" ", normalized)
    return normalized.strip()


def _is_terminal_text(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return False
    if _has_terminal_punctuation(stripped):
        return True
    return any(stripped.endswith(ending) for ending in _TERMINAL_ENDINGS)


def _has_terminal_punctuation(text: str) -> bool:
    stripped = text.rstrip()
    return stripped.endswith((JP_PERIOD, JP_EXCL, JP_QUESTION, "!", "?", "\u300d", "\u300f"))


def _should_add_period(text: str) -> bool:
    stripped = text.rstrip()
    if len(stripped) < 6:
        return False
    if _is_terminal_text(stripped):
        return False
    if any(stripped.endswith(ending) for ending in _CONTINUATION_ENDINGS):
        return False
    return any(stripped.endswith(ending) for ending in _TERMINAL_ENDINGS)


def _should_merge(previous: SubtitleSegment, current: SubtitleSegment, options: PostprocessOptions) -> bool:
    prev_text = previous.text.rstrip()
    curr_text = current.text.lstrip()
    if not prev_text or not curr_text:
        return False

    gap = max(0.0, current.start - previous.end)
    if gap > options.max_gap:
        return False
    if _is_terminal_text(prev_text):
        return False
    if len(prev_text) <= 4 and options.one_word:
        return True
    if any(prev_text.endswith(ending) for ending in _CONTINUATION_ENDINGS):
        return True
    return any(curr_text.startswith(token) for token in _CONTINUATION_STARTS)


def _build_token_text(text: str, split_mode: sudachi_tokenizer.Tokenizer.SplitMode) -> str:
    tokenizer_obj = _get_tokenizer(split_mode)
    morphemes = tokenizer_obj.tokenize(text)
    parts: list[str] = []
    for morpheme in morphemes:
        surface = morpheme.surface()
        if not surface:
            continue
        if not parts:
            parts.append(surface)
            continue

        previous = parts[-1]
        if surface in JP_PUNCT:
            parts[-1] = previous.rstrip()
            parts.append(surface)
            continue
        if previous in JP_OPEN_QUOTES:
            parts.append(surface)
            continue
        if _ENDS_WITH_JP_CHAR.search(previous) and _STARTS_WITH_JP_CHAR.match(surface):
            parts[-1] = previous.rstrip()
            parts.append(surface)
            continue
        parts.append(f" {surface}")

    return "".join(parts)


def _is_terminal_with_dictionary(text: str) -> bool:
    if _is_terminal_text(text):
        return True

    tokenizer_obj = _get_tokenizer(sudachi_tokenizer.Tokenizer.SplitMode.C)
    morphemes = tokenizer_obj.tokenize(text)
    if not morphemes:
        return False

    last = morphemes[-1]
    pos = last.part_of_speech()
    surface = last.surface()
    if surface in {"\u3067\u3059", "\u307e\u3059", "\u3060", "\u3067\u3057\u305f", "\u307e\u3057\u305f"}:
        return True
    return pos[0] in {"\u52a9\u52d5\u8a5e", "\u52d5\u8a5e"} and surface.endswith(
        ("\u305f", "\u3060", "\u307e\u3059", "\u3067\u3059")
    )


def _should_merge_with_dictionary(
    previous: SubtitleSegment,
    current: SubtitleSegment,
    options: PostprocessOptions,
) -> bool:
    if _should_merge(previous, current, options):
        return True

    gap = max(0.0, current.start - previous.end)
    if gap > options.max_gap:
        return False

    tokenizer_obj = _get_tokenizer(sudachi_tokenizer.Tokenizer.SplitMode.C)
    prev_tokens = tokenizer_obj.tokenize(previous.text)
    curr_tokens = tokenizer_obj.tokenize(current.text)
    if not prev_tokens or not curr_tokens:
        return False

    prev_last = prev_tokens[-1]
    curr_first = curr_tokens[0]
    prev_pos = prev_last.part_of_speech()
    curr_pos = curr_first.part_of_speech()
    if prev_pos[0] == "\u52a9\u8a5e":
        return True
    if options.one_word and len(prev_tokens) == 1 and prev_pos[0] in _SINGLE_WORD_POS:
        return True
    return curr_pos[0] in {"\u52a9\u8a5e", "\u52a9\u52d5\u8a5e"}


def _enhance_text_with_dictionary(text: str, options: PostprocessOptions) -> str:
    tokenized = _build_token_text(text, sudachi_tokenizer.Tokenizer.SplitMode.C)
    return postprocess_japanese_text(tokenized, options)


def postprocess_japanese_segments(
    segments: list[SubtitleSegment],
    options: PostprocessOptions | None = None,
) -> list[SubtitleSegment]:
    config = options or PostprocessOptions()
    process_text = _enhance_text_with_dictionary if config.enhanced else postprocess_japanese_text
    if not config.enabled:
        return [
            SubtitleSegment(start=segment.start, end=segment.end, text=segment.text.strip())
            for segment in segments
            if segment.text.strip()
        ]

    normalized_segments: list[SubtitleSegment] = []
    for segment in segments:
        text = process_text(segment.text, config)
        if not text:
            continue
        normalized_segments.append(SubtitleSegment(start=segment.start, end=segment.end, text=text))

    if not normalized_segments:
        return []

    merged_segments: list[SubtitleSegment] = [normalized_segments[0]]
    for segment in normalized_segments[1:]:
        previous = merged_segments[-1]
        should_merge = (
            _should_merge_with_dictionary(previous, segment, config)
            if config.enhanced
            else _should_merge(previous, segment, config)
        )
        if should_merge:
            previous.text = process_text(f"{previous.text}{segment.text}", config)
            previous.end = max(previous.end, segment.end)
            continue
        merged_segments.append(segment)

    for segment in merged_segments:
        segment.text = process_text(segment.text, config)
        if config.sentence:
            if config.enhanced:
                if _is_terminal_with_dictionary(segment.text) and not _has_terminal_punctuation(segment.text):
                    segment.text = f"{segment.text}{JP_PERIOD}"
            elif _should_add_period(segment.text) and not _has_terminal_punctuation(segment.text):
                segment.text = f"{segment.text}{JP_PERIOD}"

    return merged_segments
