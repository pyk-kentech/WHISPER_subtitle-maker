from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(slots=True)
class SubtitleSegment:
    start: float
    end: float
    text: str


def format_srt_timestamp(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0

    total_milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def build_srt_text(segments: Iterable[SubtitleSegment]) -> str:
    blocks: list[str] = []
    index = 1

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        start = format_srt_timestamp(segment.start)
        end = format_srt_timestamp(max(segment.end, segment.start))
        blocks.append(f"{index}\n{start} --> {end}\n{text}")
        index += 1

    if not blocks:
        return ""

    return "\n\n".join(blocks) + "\n"


def write_srt(path: Path, segments: Iterable[SubtitleSegment]) -> None:
    srt_text = build_srt_text(segments)
    if not srt_text:
        raise ValueError("인식 결과가 비어 있어 SRT를 생성할 수 없습니다.")

    path.write_text(srt_text, encoding="utf-8-sig")
