from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
import urllib.request
import zipfile

from sudachipy import dictionary, tokenizer

from .config import get_dictionary_cache_dir, get_dictionary_pack_dir


StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int, str, int, int], None]

DICT_PACKAGE_NAME = "sudachidict_core"
DICT_MANIFEST_NAME = "manifest.json"
_SUDACHI_DICTIONARY: dictionary.Dictionary | None = None
_TOKENIZER_CACHE: dict[str, tokenizer.Tokenizer] = {}


def get_dictionary_path() -> Path:
    return get_dictionary_pack_dir() / "system.dic"


def is_dictionary_pack_ready() -> bool:
    return get_dictionary_path().is_file()


def _read_manifest() -> dict[str, str]:
    manifest_path = get_dictionary_pack_dir() / DICT_MANIFEST_NAME
    if not manifest_path.is_file():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        pass
    return {}


def _write_manifest(data: dict[str, str]) -> None:
    pack_dir = get_dictionary_pack_dir()
    pack_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = pack_dir / DICT_MANIFEST_NAME
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_package_metadata() -> tuple[str, str, str]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{DICT_PACKAGE_NAME}/json") as response:
        metadata = json.load(response)

    version = str(metadata["info"]["version"])
    for file_info in metadata["releases"][version]:
        filename = str(file_info["filename"])
        if "py3-none-any.whl" in filename and filename.endswith(".whl"):
            return version, str(file_info["url"]), filename
    raise RuntimeError(f"Wheel not found for {DICT_PACKAGE_NAME}")


def _download_file(
    url: str,
    destination: Path,
    filename: str,
    progress_callback: ProgressCallback,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as stream:
        total_bytes = int(response.headers.get("Content-Length", "0"))
        downloaded_bytes = 0
        chunk_size = 1024 * 1024
        progress_callback(0, total_bytes, filename, 0, total_bytes)

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            stream.write(chunk)
            downloaded_bytes += len(chunk)
            progress_callback(downloaded_bytes, total_bytes, filename, downloaded_bytes, total_bytes)


def _extract_dictionary(wheel_path: Path, status_callback: StatusCallback) -> Path:
    pack_dir = get_dictionary_pack_dir()
    pack_dir.mkdir(parents=True, exist_ok=True)
    target_path = get_dictionary_path()

    with zipfile.ZipFile(wheel_path) as archive:
        member_name = next(
            (name for name in archive.namelist() if name.endswith("/resources/system.dic")),
            None,
        )
        if member_name is None:
            raise RuntimeError("Dictionary file not found in downloaded pack.")

        with archive.open(member_name) as source_stream, target_path.open("wb") as target_stream:
            target_stream.write(source_stream.read())

    status_callback(f"일본어 사전팩 설치 완료: {target_path}")
    return target_path


def ensure_dictionary_pack(
    progress_callback: ProgressCallback,
    status_callback: StatusCallback,
) -> Path:
    if is_dictionary_pack_ready():
        progress_callback(1, 1, "", 1, 1)
        status_callback(f"일본어 사전팩 캐시 사용: {get_dictionary_path()}")
        return get_dictionary_path()

    cache_dir = get_dictionary_cache_dir()
    pack_dir = get_dictionary_pack_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    pack_dir.mkdir(parents=True, exist_ok=True)

    version, url, filename = _get_package_metadata()
    wheel_path = cache_dir / filename
    manifest = _read_manifest()

    if not wheel_path.is_file():
        status_callback(f"일본어 사전팩 다운로드 중: {filename}")
        _download_file(url, wheel_path, filename, progress_callback)
    else:
        progress_callback(1, 1, filename, 1, 1)
        status_callback(f"일본어 사전팩 캐시 사용: {filename}")

    if manifest.get("version") != version or not is_dictionary_pack_ready():
        status_callback(f"일본어 사전팩 설치 중: {version}")
        _extract_dictionary(wheel_path, status_callback)
        _write_manifest({"version": version, "wheel": filename})

    if not is_dictionary_pack_ready():
        raise RuntimeError("일본어 사전팩 다운로드 후에도 system.dic를 찾을 수 없습니다.")

    status_callback(f"일본어 사전팩 준비 완료: {get_dictionary_path()}")
    return get_dictionary_path()


def create_dictionary_tokenizer(
    split_mode: tokenizer.Tokenizer.SplitMode = tokenizer.Tokenizer.SplitMode.C,
) -> tokenizer.Tokenizer:
    global _SUDACHI_DICTIONARY, _TOKENIZER_CACHE
    dictionary_path = get_dictionary_path()
    if not dictionary_path.is_file():
        raise RuntimeError("일본어 사전팩이 준비되지 않았습니다.")
    if _SUDACHI_DICTIONARY is None:
        _SUDACHI_DICTIONARY = dictionary.Dictionary(dict=str(dictionary_path))
    cache_key = str(split_mode)
    if cache_key not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[cache_key] = _SUDACHI_DICTIONARY.create(mode=split_mode)
    return _TOKENIZER_CACHE[cache_key]
