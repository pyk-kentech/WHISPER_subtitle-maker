from __future__ import annotations

import ctypes
import fnmatch
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Callable
import urllib.request
import zipfile

import ctranslate2
from packaging.version import InvalidVersion, Version

from .config import get_cuda_cache_dir, get_cuda_runtime_dir


StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int, str, int, int], None]

CUDA_PACKAGES = (
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cuda-nvrtc-cu12",
)
_IS_WINDOWS = sys.platform == "win32"
# 휠 안에서 꺼낼 라이브러리 파일과, 준비 완료로 볼 필수 라이브러리(Windows는 DLL, 리눅스는 .so).
if _IS_WINDOWS:
    REQUIRED_LIBRARY_PATTERNS = ("cublas64_12*.dll", "cudart64_12*.dll", "cudnn64_9*.dll")
    SYSTEM_LIBRARY_NAMES: tuple[str, ...] = ()
else:
    REQUIRED_LIBRARY_PATTERNS = ("libcublas.so.12*", "libcudart.so.12*", "libcudnn.so.9*")
    # 시스템에 CUDA 12·cuDNN 9가 깔려 있으면 따로 받지 않고 그것을 쓴다.
    SYSTEM_LIBRARY_NAMES = ("libcublas.so.12", "libcudnn.so.9")
PACKAGE_MAJOR_VERSIONS = {
    "nvidia-cudnn-cu12": 9,
}


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


_PRELOADED_LIBRARIES: set[str] = set()


def _is_library_file(name: str) -> bool:
    lowered = name.lower()
    if _IS_WINDOWS:
        return lowered.endswith(".dll")
    return ".so" in lowered and fnmatch.fnmatch(lowered, "*.so*")


def _find_library(pattern: str) -> Path | None:
    bin_dir = get_runtime_bin_dir()
    if not bin_dir.is_dir():
        return None
    for path in bin_dir.glob(pattern):
        return path
    return None


def is_system_cuda_available() -> bool:
    if not SYSTEM_LIBRARY_NAMES:
        return False
    try:
        for name in SYSTEM_LIBRARY_NAMES:
            ctypes.CDLL(name)
    except OSError:
        return False
    return True


def is_cuda_runtime_ready() -> bool:
    if all(_find_library(pattern) is not None for pattern in REQUIRED_LIBRARY_PATTERNS):
        return True
    return is_system_cuda_available()


def find_missing_cuda_libraries() -> list[str]:
    """CTranslate2가 실제로 불러올 CUDA 라이브러리 중 로드되지 않는 것의 이름을 돌려준다."""
    names = ("cublas64_12.dll",) if _IS_WINDOWS else ("libcublas.so.12", "libcudnn.so.9")
    loader = getattr(ctypes, "WinDLL", ctypes.CDLL) if _IS_WINDOWS else ctypes.CDLL
    missing: list[str] = []
    for name in names:
        try:
            loader(name)
        except OSError:
            missing.append(name)
    return missing


def _preload_libraries(bin_dir: Path) -> None:
    # 리눅스는 실행 중에 LD_LIBRARY_PATH를 바꿔도 dlopen이 보지 않는다. 받은 .so를 RTLD_GLOBAL로 미리 올려 두면
    # CTranslate2가 같은 이름(soname)으로 찾을 때 이미 로드된 것을 쓴다. 의존 순서를 몰라도 되도록 될 때까지 반복한다.
    pending = sorted(
        path for path in bin_dir.iterdir() if _is_library_file(path.name) and str(path) not in _PRELOADED_LIBRARIES
    )
    while pending:
        failed: list[Path] = []
        for path in pending:
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(path)
                continue
            _PRELOADED_LIBRARIES.add(str(path))
        if len(failed) == len(pending):
            return
        pending = failed


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
    if not _IS_WINDOWS:
        _preload_libraries(bin_dir)
    return bin_dir


def _select_version(package_name: str, metadata: dict) -> str:
    major = PACKAGE_MAJOR_VERSIONS.get(package_name)
    latest = str(metadata["info"]["version"])
    if major is None:
        return latest
    candidates: list[Version] = []
    for version_text, files in metadata["releases"].items():
        try:
            version = Version(version_text)
        except InvalidVersion:
            continue
        if version.major == major and not version.is_prerelease and files:
            candidates.append(version)
    if not candidates:
        raise RuntimeError(f"No {major}.x release found for {package_name}")
    return str(max(candidates))


def _get_package_metadata(package_name: str) -> tuple[str, str, str, str]:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package_name}/json") as response:
        metadata = json.load(response)

    version = _select_version(package_name, metadata)
    for file_info in metadata["releases"][version]:
        filename = str(file_info["filename"])
        if filename.endswith(".whl") and _wheel_matches_platform(filename):
            return version, str(file_info["url"]), filename, str(file_info["digests"]["sha256"])
    raise RuntimeError(f"{sys.platform}/{platform.machine()}용 wheel을 찾지 못했습니다: {package_name}")


def _wheel_matches_platform(filename: str) -> bool:
    if _IS_WINDOWS:
        return "win_amd64" in filename
    machine = platform.machine().lower()
    arch = "aarch64" if machine in {"aarch64", "arm64"} else "x86_64"
    return "manylinux" in filename and arch in filename


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(
    url: str,
    destination: Path,
    filename: str,
    expected_sha256: str,
    progress_callback: ProgressCallback,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial_path = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, partial_path.open("wb") as stream:
        total_bytes = int(response.headers.get("Content-Length", "0"))
        downloaded_bytes = 0
        chunk_size = 1024 * 1024
        progress_callback(0, total_bytes, filename, 0, total_bytes)

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            stream.write(chunk)
            digest.update(chunk)
            downloaded_bytes += len(chunk)
            progress_callback(downloaded_bytes, total_bytes, filename, downloaded_bytes, total_bytes)

    if digest.hexdigest() != expected_sha256:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file failed SHA-256 verification: {filename}")
    partial_path.replace(destination)


def _extract_libraries(wheel_path: Path, status_callback: StatusCallback) -> int:
    bin_dir = get_runtime_bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    extracted_count = 0

    with zipfile.ZipFile(wheel_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            if not _is_library_file(Path(member.filename).name):
                continue

            target_path = bin_dir / Path(member.filename).name
            partial_path = target_path.with_name(target_path.name + ".part")
            with archive.open(member) as source_stream, partial_path.open("wb") as target_stream:
                target_stream.write(source_stream.read())
            partial_path.replace(target_path)
            extracted_count += 1

    status_callback(f"CUDA 라이브러리 추출 완료: {wheel_path.name} ({extracted_count}개)")
    return extracted_count


def ensure_cuda_runtime(
    progress_callback: ProgressCallback,
    status_callback: StatusCallback,
) -> Path:
    if not has_cuda_device():
        raise RuntimeError("CUDA GPU를 찾지 못했습니다.")

    add_cuda_runtime_to_path()
    if is_system_cuda_available():
        progress_callback(1, 1, "", 1, 1)
        status_callback("시스템에 설치된 CUDA 12 / cuDNN 9 라이브러리를 사용합니다.")
        return get_runtime_bin_dir()
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
        version, url, filename, sha256 = _get_package_metadata(package_name)
        package_versions[package_name] = version
        wheel_path = cache_dir / filename
        if wheel_path.is_file() and _sha256_of(wheel_path) != sha256:
            status_callback(f"손상된 CUDA 런타임 캐시를 다시 받습니다: {filename}")
            wheel_path.unlink()
            installed_versions.pop(package_name, None)
        if not wheel_path.is_file():
            status_callback(f"CUDA 런타임 다운로드 중: {filename}")
            _download_file(url, wheel_path, filename, sha256, progress_callback)
        else:
            progress_callback(1, 1, filename, 1, 1)
            status_callback(f"CUDA 런타임 캐시 사용: {filename}")

        if installed_versions.get(package_name) != version or not is_cuda_runtime_ready():
            status_callback(f"CUDA 런타임 설치 중: {package_name} {version}")
            _extract_libraries(wheel_path, status_callback)

    _write_manifest({"packages": package_versions})
    add_cuda_runtime_to_path()

    if not is_cuda_runtime_ready():
        raise RuntimeError("CUDA 런타임 다운로드 후에도 필수 라이브러리가 준비되지 않았습니다.")

    status_callback(f"CUDA 런타임 준비 완료: {get_runtime_bin_dir()}")
    return get_runtime_bin_dir()
