from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import SUPPORTED_EXTENSIONS


STATUS_PENDING = "대기중"
STATUS_TRANSCRIBING = "자막 생성중"
STATUS_TRANSLATING = "번역중"
STATUS_DONE = "완료"
STATUS_FAILED = "실패"
STATUS_SKIPPED = "스킵"


@dataclass(slots=True)
class QueueItem:
    source_path: Path
    status: str = STATUS_PENDING
    message: str = ""
    output_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.output_path = self.source_path.with_suffix(".srt")


def normalize_input_files(paths: Iterable[str | Path]) -> tuple[list[QueueItem], list[str]]:
    items: list[QueueItem] = []
    errors: list[str] = []
    seen: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser()
        try:
            resolved = path.resolve(strict=False)
        except OSError as exc:
            errors.append(f"경로 확인 실패: {path} ({exc})")
            continue

        if resolved in seen:
            continue
        if resolved.suffix.lower() not in SUPPORTED_EXTENSIONS:
            errors.append(f"지원하지 않는 파일 형식: {resolved}")
            continue
        if not resolved.exists() or not resolved.is_file():
            errors.append(f"파일을 찾을 수 없음: {resolved}")
            continue

        seen.add(resolved)
        items.append(QueueItem(source_path=resolved))

    return items, errors
