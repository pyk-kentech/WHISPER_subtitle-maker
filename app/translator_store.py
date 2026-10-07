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
    LEGACY_TRANSLATION_CHUNK_SIZE,
    LEGACY_TRANSLATION_SYSTEM_PROMPT,
    get_translator_keys_path,
    get_translator_settings_path,
)
from .credential_store import CredentialStoreError, delete_secret, load_secret, save_secret


SETTINGS_VERSION = 2


@dataclass(slots=True)
class TranslatorSettings:
    preferred_model: str = ""
    target_language: str = DEFAULT_OUTPUT_LANGUAGE
    use_deepl_fallback: bool = False
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

    chunk_size = int(data.get("chunk_size", DEFAULT_TRANSLATION_CHUNK_SIZE))
    system_prompt = str(data.get("system_prompt", DEFAULT_TRANSLATION_SYSTEM_PROMPT))
    if int(data.get("settings_version", 1)) < SETTINGS_VERSION:
        # 이전 버전 기본값을 그대로 쓰던 경우에만 새 기본값으로 올린다. 사용자가 바꾼 값은 유지.
        if chunk_size == LEGACY_TRANSLATION_CHUNK_SIZE:
            chunk_size = DEFAULT_TRANSLATION_CHUNK_SIZE
        if system_prompt == LEGACY_TRANSLATION_SYSTEM_PROMPT:
            system_prompt = DEFAULT_TRANSLATION_SYSTEM_PROMPT

    return TranslatorSettings(
        preferred_model=str(data.get("preferred_model", "")),
        target_language=str(data.get("target_language", DEFAULT_OUTPUT_LANGUAGE)),
        use_deepl_fallback=bool(data.get("use_deepl_fallback", False)),
        chunk_size=chunk_size,
        request_delay_seconds=float(data.get("request_delay_seconds", DEFAULT_TRANSLATION_REQUEST_DELAY_SECONDS)),
        temperature=float(data.get("temperature", DEFAULT_TRANSLATION_TEMPERATURE)),
        top_p=float(data.get("top_p", DEFAULT_TRANSLATION_TOP_P)),
        reasoning_level=str(data.get("reasoning_level", DEFAULT_TRANSLATION_REASONING_LEVEL)),
        system_prompt=system_prompt,
        translation_note=str(data.get("translation_note", "")),
    )


def save_translator_settings(settings: TranslatorSettings) -> None:
    path = get_translator_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {**asdict(settings), "settings_version": SETTINGS_VERSION}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# Windows 일반 자격 증명 하나에는 2560바이트(UTF-16으로 약 1280자)까지만 담기므로 키가 많으면 나눠 저장한다.
_KEYS_PART_LIMIT_BYTES = 2400
_MAX_KEY_PARTS = 64


def _key_part_name(index: int) -> str:
    return API_KEYS_CREDENTIAL_NAME if index == 0 else f"{API_KEYS_CREDENTIAL_NAME}#{index + 1}"


def _split_key_parts(normalized: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    for line in normalized.splitlines():
        candidate = "\n".join([*current, line])
        if current and len(candidate.encode("utf-16-le")) > _KEYS_PART_LIMIT_BYTES:
            parts.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(current))
    return parts


def _load_key_parts() -> str:
    lines: list[str] = []
    for index in range(_MAX_KEY_PARTS):
        part = load_secret(_key_part_name(index))
        if not part:
            break
        lines.append(part)
    return "\n".join(lines)


def _save_key_parts(normalized: str) -> None:
    parts = _split_key_parts(normalized) if normalized else []
    for index, part in enumerate(parts):
        save_secret(_key_part_name(index), part)
    for index in range(len(parts), _MAX_KEY_PARTS):
        if not load_secret(_key_part_name(index)):
            break
        delete_secret(_key_part_name(index))


def load_api_keys() -> list[str]:
    try:
        stored = _load_key_parts()
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
            _save_key_parts(normalized)
            path.unlink(missing_ok=True)
        except CredentialStoreError:
            pass
    return [line.strip() for line in normalized.splitlines() if line.strip()]


def save_api_keys(keys_text: str) -> None:
    normalized = "\n".join(line.strip() for line in keys_text.splitlines() if line.strip())
    path = get_translator_keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _save_key_parts(normalized)
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
