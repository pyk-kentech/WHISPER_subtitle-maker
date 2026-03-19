from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable
import urllib.request
import zipfile

import ctranslate2

from .config import get_cuda_cache_dir, get_cuda_runtime_dir


StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int, str, int, int], None]

CUDA_PACKAGES = (
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cuda-nvrtc-cu12",
)
REQUIRED_DLL_PREFIXES = (
    "cublas64_12",
    "cudart64_12",
    "cudnn64_9",
)


def get_cuda_device_count() -> int:
    try:
        return max(0, int(ctranslate2.get_cuda_device_count()))
    except Exception:
        return 0


def has_cuda_device() -> bool:
    return get_cuda_device_count() > 0


def get_runtime_bin_dir() -> Path:
    return get_cuda_runtime_dir() / "bin"


def _read_manifest() -> dict[str, object]:
    manifest_path = get_cuda_runtime_dir() / "manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_manifest(data: dict[str, object]) -> None:
    runtime_dir = get_cuda_runtime_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runtime_dir / "manifest.json"
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _find_dll_by_prefix(prefix: str) -> Path | None:
    bin_dir = get_runtime_bin_dir()
    if not bin_dir.is_dir():
        return None
    for path in bin_dir.glob(f"{prefix}*.dll"):
        return path
    return None


def is_cuda_runtime_ready() -> bool:
    return all(_find_dll_by_prefix(prefix) is not None for prefix in REQUIRED_DLL_PREFIXES)


def add_cuda_runtime_to_path() -> Path | None:
    bin_dir = get_runtime_bin_dir()
    if not bin_dir.is_dir():
        return None

    path_value = os.environ.get("PATH", "")
    path_parts = path_value.split(os.pathsep) if path_value else []
    bin_dir_str = str(bin_dir)
    if bin_dir_str not in path_parts:
        os.environ["PATH"] = f"{bin_dir_str}{os.pathsep}{path_value}" if path_value else bin_dir_str

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is not None:
        try:
            add_dll_directory(bin_dir_str)
        except OSError:
            pass
    return bin_dir


def _get_package_metadata(package_name: str) -> tuple[str, str, str]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package_name}/json") as response:
        metadata = json.load(response)

    version = str(metadata["info"]["version"])
    for file_info in metadata["releases"][version]:
        filename = str(file_info["filename"])
        if "win_amd64" in filename and filename.endswith(".whl"):
            return version, str(file_info["url"]), filename
    raise RuntimeError(f"Windows wheel not found for {package_name}")


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


def _extract_dlls(wheel_path: Path, status_callback: StatusCallback) -> int:
    bin_dir = get_runtime_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    extracted_count = 0

    with zipfile.ZipFile(wheel_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            if not member.filename.lower().endswith(".dll"):
                continue

            target_path = bin_dir / Path(member.filename).name
            with archive.open(member) as source_stream, target_path.open("wb") as target_stream:
                target_stream.write(source_stream.read())
            extracted_count += 1

    status_callback(f"CUDA DLL 추출 완료: {wheel_path.name} ({extracted_count}개)")
    return extracted_count


def ensure_cuda_runtime(
    progress_callback: ProgressCallback,
    status_callback: StatusCallback,
) -> Path:
    if not has_cuda_device():
        raise RuntimeError("CUDA GPU를 찾지 못했습니다.")

    add_cuda_runtime_to_path()
    if is_cuda_runtime_ready():
        progress_callback(1, 1, "", 1, 1)
        status_callback("CUDA 런타임 캐시가 이미 준비되어 있습니다.")
        return get_runtime_bin_dir()

    cache_dir = get_cuda_cache_dir()
    runtime_dir = get_cuda_runtime_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_manifest()
    installed_versions = dict(manifest.get("packages", {})) if isinstance(manifest.get("packages", {}), dict) else {}

    package_versions: dict[str, str] = {}
    for package_name in CUDA_PACKAGES:
        version, url, filename = _get_package_metadata(package_name)
        package_versions[package_name] = version
        wheel_path = cache_dir / filename
        if not wheel_path.is_file():
            status_callback(f"CUDA 런타임 다운로드 중: {filename}")
            _download_file(url, wheel_path, filename, progress_callback)
        else:
            progress_callback(1, 1, filename, 1, 1)
            status_callback(f"CUDA 런타임 캐시 사용: {filename}")

        if installed_versions.get(package_name) != version:
            status_callback(f"CUDA 런타임 설치 중: {package_name} {version}")
            _extract_dlls(wheel_path, status_callback)

    _write_manifest({"packages": package_versions})
    add_cuda_runtime_to_path()

    if not is_cuda_runtime_ready():
        raise RuntimeError("CUDA 런타임 다운로드 후에도 필수 DLL이 준비되지 않았습니다.")

    status_callback(f"CUDA 런타임 준비 완료: {get_runtime_bin_dir()}")
    return get_runtime_bin_dir()
