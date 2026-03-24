from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from .config import (
    API_KEYS_CREDENTIAL_NAME,
    DEEPL_API_KEY_CREDENTIAL_NAME,
    DEFAULT_OUTPUT_LANGUAGE,
    DEFAULT_TRANSLATION_CHUNK_SIZE,
    DEFAULT_TRANSLATION_REQUEST_DELAY_SECONDS,
    DEFAULT_TRANSLATION_REASONING_LEVEL,
    DEFAULT_TRANSLATION_SYSTEM_PROMPT,
    DEFAULT_TRANSLATION_TEMPERATURE,
    DEFAULT_TRANSLATION_TOP_P,
    get_translator_keys_path,
    get_translator_settings_path,
)
from .credential_store import CredentialStoreError, delete_secret, load_secret, save_secret


@dataclass(slots=True)
class TranslatorSettings:
    preferred_model: str = ""
    target_language: str = DEFAULT_OUTPUT_LANGUAGE
    chunk_size: int = DEFAULT_TRANSLATION_CHUNK_SIZE
    request_delay_seconds: float = DEFAULT_TRANSLATION_REQUEST_DELAY_SECONDS
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
        target_language=str(data.get("target_language", DEFAULT_OUTPUT_LANGUAGE)),
        chunk_size=int(data.get("chunk_size", DEFAULT_TRANSLATION_CHUNK_SIZE)),
        request_delay_seconds=float(data.get("request_delay_seconds", DEFAULT_TRANSLATION_REQUEST_DELAY_SECONDS)),
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
    try:
        stored = load_secret(API_KEYS_CREDENTIAL_NAME)
    except CredentialStoreError:
        stored = ""

    if stored:
        return [line.strip() for line in stored.splitlines() if line.strip()]

    path = get_translator_keys_path()
    if not path.is_file():
        return []

    plaintext = path.read_text(encoding="utf-8")
    normalized = "\n".join(line.strip() for line in plaintext.splitlines() if line.strip())
    if normalized:
        try:
            save_secret(API_KEYS_CREDENTIAL_NAME, normalized)
            path.unlink(missing_ok=True)
        except CredentialStoreError:
            pass
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def save_api_keys(keys_text: str) -> None:
    normalized = "\n".join(line.strip() for line in keys_text.splitlines() if line.strip())
    path = get_translator_keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if normalized:
        save_secret(API_KEYS_CREDENTIAL_NAME, normalized)
    else:
        delete_secret(API_KEYS_CREDENTIAL_NAME)
    if path.exists():
        path.unlink(missing_ok=True)


def load_deepl_api_key() -> str:
    try:
        return load_secret(DEEPL_API_KEY_CREDENTIAL_NAME).strip()
    except CredentialStoreError:
        return ""


def save_deepl_api_key(api_key: str) -> None:
    normalized = api_key.strip()
    if normalized:
        save_secret(DEEPL_API_KEY_CREDENTIAL_NAME, normalized)
    else:
        delete_secret(DEEPL_API_KEY_CREDENTIAL_NAME)
