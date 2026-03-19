from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class LineRecord:
    text: str
    translatable: bool
    line_id: str | None = None


@dataclass(slots=True)
class SubtitleDocument:
    records: list[LineRecord]

    def get_translatable_records(self) -> list[LineRecord]:
        return [record for record in self.records if record.translatable and record.line_id is not None]

    def apply_translations(self, translations: dict[str, str]) -> None:
        for record in self.records:
            if record.translatable and record.line_id in translations:
                record.text = translations[record.line_id]

    def render(self) -> str:
        return "\n".join(record.text for record in self.records)


def load_subtitle_document(path: Path) -> SubtitleDocument:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    return load_subtitle_document_from_text(path.suffix.lower(), text)


def load_subtitle_document_from_text(suffix: str, text: str) -> SubtitleDocument:
    lines = text.splitlines()
    if suffix == ".srt":
        return SubtitleDocument(records=_parse_srt(lines))
    if suffix == ".vtt":
        return SubtitleDocument(records=_parse_vtt(lines))
    return SubtitleDocument(records=_parse_txt(lines))


def _parse_srt(lines: list[str]) -> list[LineRecord]:
    records: list[LineRecord] = []
    active_text = False
    cue_index = 0
    text_index = 0

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            records.append(LineRecord(text=raw_line, translatable=False))
            active_text = False
            text_index = 0
            continue
        if stripped.isdigit():
            cue_index += 1
            records.append(LineRecord(text=raw_line, translatable=False))
            active_text = False
            text_index = 0
            continue
        if "-->" in raw_line:
            records.append(LineRecord(text=raw_line, translatable=False))
            active_text = True
            text_index = 0
            continue
        if active_text:
            records.append(LineRecord(text=raw_line, translatable=True, line_id=f"srt-{cue_index}-{text_index}"))
            text_index += 1
            continue
        records.append(LineRecord(text=raw_line, translatable=False))
    return records


def _parse_vtt(lines: list[str]) -> list[LineRecord]:
    records: list[LineRecord] = []
    active_text = False
    cue_index = 0
    text_index = 0

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            records.append(LineRecord(text=raw_line, translatable=False))
            active_text = False
            text_index = 0
            continue
        if stripped == "WEBVTT" or stripped.startswith("NOTE") or stripped.startswith("STYLE") or stripped.startswith("REGION"):
            records.append(LineRecord(text=raw_line, translatable=False))
            active_text = False
            continue
        if "-->" in raw_line:
            cue_index += 1
            records.append(LineRecord(text=raw_line, translatable=False))
            active_text = True
            text_index = 0
            continue
        if active_text:
            records.append(LineRecord(text=raw_line, translatable=True, line_id=f"vtt-{cue_index}-{text_index}"))
            text_index += 1
            continue
        records.append(LineRecord(text=raw_line, translatable=False))
    return records


def _parse_txt(lines: list[str]) -> list[LineRecord]:
    records: list[LineRecord] = []
    line_index = 0
    for raw_line in lines:
        if raw_line.strip():
            records.append(LineRecord(text=raw_line, translatable=True, line_id=f"txt-{line_index}"))
            line_index += 1
            continue
        records.append(LineRecord(text=raw_line, translatable=False))
    return records
