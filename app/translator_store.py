from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .config import (
    DEFAULT_TRANSLATION_CHUNK_SIZE,
    DEFAULT_TRANSLATION_REASONING_LEVEL,
    DEFAULT_TRANSLATION_SYSTEM_PROMPT,
    DEFAULT_TRANSLATION_TEMPERATURE,
    DEFAULT_TRANSLATION_TOP_P,
    get_translator_keys_path,
    get_translator_settings_path,
)


@dataclass(slots=True)
class TranslatorSettings:
    preferred_model: str = ""
    chunk_size: int = DEFAULT_TRANSLATION_CHUNK_SIZE
    temperature: float = DEFAULT_TRANSLATION_TEMPERATURE
    top_p: float = DEFAULT_TRANSLATION_TOP_P
    reasoning_level: str = DEFAULT_TRANSLATION_REASONING_LEVEL
    system_prompt: str = DEFAULT_TRANSLATION_SYSTEM_PROMPT
    translation_note: str = ""


def load_translator_settings() -> TranslatorSettings:
    path = get_translator_settings_path()
    if not path.is_file():
        return TranslatorSettings()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return TranslatorSettings()

    return TranslatorSettings(
        preferred_model=str(data.get("preferred_model", "")),
        chunk_size=int(data.get("chunk_size", DEFAULT_TRANSLATION_CHUNK_SIZE)),
        temperature=float(data.get("temperature", DEFAULT_TRANSLATION_TEMPERATURE)),
        top_p=float(data.get("top_p", DEFAULT_TRANSLATION_TOP_P)),
        reasoning_level=str(data.get("reasoning_level", DEFAULT_TRANSLATION_REASONING_LEVEL)),
        system_prompt=str(data.get("system_prompt", DEFAULT_TRANSLATION_SYSTEM_PROMPT)),
        translation_note=str(data.get("translation_note", "")),
    )


def save_translator_settings(settings: TranslatorSettings) -> None:
    path = get_translator_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")


def load_api_keys() -> list[str]:
    path = get_translator_keys_path()
    if not path.is_file():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_api_keys(keys_text: str) -> None:
    path = get_translator_keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = "\n".join(line.strip() for line in keys_text.splitlines() if line.strip())
    path.write_text((normalized + "\n") if normalized else "", encoding="utf-8")
