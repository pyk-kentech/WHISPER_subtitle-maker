from __future__ import annotations

import ctypes
import gc
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import av
import ctranslate2
from faster_whisper import WhisperModel

from .config import DEFAULT_MODEL_KEY
from .cuda_runtime import add_cuda_runtime_to_path, has_cuda_device, is_cuda_runtime_ready
from .model_manager import get_model_dir, is_model_ready
from .srt_writer import SubtitleSegment

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


CPU_COMPUTE_TYPE_OPTIONS = ["auto", "int8", "int8_float32", "float32"]
CUDA_COMPUTE_TYPE_OPTIONS = ["auto", "float16", "int8_float16", "int8_float32", "float32"]
MEMORY_PROFILE_OPTIONS = [
    ("unlimited", "제한 없음 (기본)"),
    ("save", "절약"),
    ("strict", "강한 절약"),
]


@dataclass(slots=True)
class RuntimeConfig:
    device: str
    compute_type: str
    cpu_threads: int
    num_workers: int
    memory_profile: str
    label: str


@dataclass(slots=True)
class VADSettings:
    enabled: bool = True
    min_silence_duration_ms: int = 500
    speech_pad_ms: int = 200


@dataclass(slots=True)
class RuntimeTuningOptions:
    compute_type: str = "auto"
    cpu_threads: int | None = None
    num_workers: int = 1
    memory_profile: str = "unlimited"
    auto_unload_after_job: bool = True


@dataclass(slots=True)
class LoadedModelInfo:
    model_key: str
    device: str
    compute_type: str
    cpu_threads: int
    num_workers: int
    loaded_at: float
    last_used_at: float


_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str, str, int, int], WhisperModel] = {}
_MODEL_META: dict[tuple[str, str, str, int, int], LoadedModelInfo] = {}


def is_cuda_runtime_available() -> tuple[bool, str]:
    if not has_cuda_device():
        return False, "CUDA GPU was not detected."

    add_cuda_runtime_to_path()
    if not is_cuda_runtime_ready():
        return False, "CUDA runtime is not ready yet."

    missing_libraries: list[str] = []
    for library_name in ("cublas64_12.dll",):
        try:
            ctypes.WinDLL(library_name)
        except OSError:
            missing_libraries.append(library_name)

    if missing_libraries:
        return False, f"Required CUDA library is missing: {', '.join(missing_libraries)}"
    return True, ""


def is_cuda_runtime_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "cublas",
            "cudnn",
            "cudart",
            "cuda",
            "curand",
            "cufft",
            "cannot be loaded",
            "load library failed",
        )
    )


def get_available_runtime_choices() -> list[tuple[str, str]]:
    choices = [("cpu", "CPU")]
    if has_cuda_device():
        choices.insert(0, ("cuda", "GPU (CUDA)"))
    return choices


def get_default_runtime_choice() -> str:
    for device, _label in get_available_runtime_choices():
        if device == "cuda":
            return "cuda"
    return "cpu"


def get_compute_type_choices(device: str) -> list[tuple[str, str]]:
    normalized = device.lower().strip()
    options = CUDA_COMPUTE_TYPE_OPTIONS if normalized == "cuda" else CPU_COMPUTE_TYPE_OPTIONS
    return [(item, item) for item in options]


def get_memory_profile_choices() -> list[tuple[str, str]]:
    return list(MEMORY_PROFILE_OPTIONS)


def get_default_cpu_threads() -> int:
    cpu_count = os.cpu_count() or 4
    return max(1, min(cpu_count - 1, 8))


def get_max_worker_count() -> int:
    return max(1, min(os.cpu_count() or 4, 32))


def build_runtime_config(device: str, options: RuntimeTuningOptions | None = None) -> RuntimeConfig:
    normalized = device.lower().strip()
    tuning = options or RuntimeTuningOptions()
    selected_compute_type = tuning.compute_type.strip().lower() or "auto"
    memory_profile = tuning.memory_profile.strip().lower() or "unlimited"
    selected_workers = max(1, int(tuning.num_workers))
    if memory_profile in {"save", "strict"}:
        selected_workers = 1

    if normalized == "cuda":
        available, reason = is_cuda_runtime_available()
        if not available:
            raise RuntimeError(f"GPU is unavailable: {reason}")

        supported = ctranslate2.get_supported_compute_types("cuda")
        preferred_order = ["float16", "int8_float16", "int8_float32", "float32"]
        if selected_compute_type != "auto":
            preferred_order = [selected_compute_type]
        elif memory_profile == "save":
            preferred_order = ["int8_float16", "float16", "int8_float32", "float32"]
        elif memory_profile == "strict":
            preferred_order = ["int8_float16", "int8_float32", "float16", "float32"]

        for compute_type in preferred_order:
            if compute_type in supported:
                return RuntimeConfig(
                    device="cuda",
                    compute_type=compute_type,
                    cpu_threads=0,
                    num_workers=selected_workers,
                    memory_profile=memory_profile,
                    label=f"GPU (CUDA, {compute_type}, workers={selected_workers}, mem={memory_profile})",
                )
        raise RuntimeError("No compatible GPU compute type was found.")

    supported = ctranslate2.get_supported_compute_types("cpu")
    preferred_order = ["int8", "int8_float32", "float32"]
    if selected_compute_type != "auto":
        preferred_order = [selected_compute_type]

    cpu_threads = tuning.cpu_threads if tuning.cpu_threads is not None else get_default_cpu_threads()
    cpu_threads = max(1, int(cpu_threads))
    if memory_profile == "save":
        cpu_threads = min(cpu_threads, 4)
    elif memory_profile == "strict":
        cpu_threads = min(cpu_threads, 2)
    for compute_type in preferred_order:
        if compute_type in supported:
            return RuntimeConfig(
                device="cpu",
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=selected_workers,
                memory_profile=memory_profile,
                label=f"CPU ({compute_type}, threads={cpu_threads}, workers={selected_workers}, mem={memory_profile})",
            )
    raise RuntimeError("No compatible CPU compute type was found.")


def unload_loaded_models(model_key: str | None = None) -> int:
    with _MODEL_CACHE_LOCK:
        keys = [key for key in _MODEL_CACHE if model_key is None or key[0] == model_key]
        for key in keys:
            del _MODEL_CACHE[key]
            _MODEL_META.pop(key, None)
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return len(keys)


def get_loaded_model_info(model_key: str | None = None) -> list[LoadedModelInfo]:
    with _MODEL_CACHE_LOCK:
        items = list(_MODEL_META.items())
    result: list[LoadedModelInfo] = []
    for cache_key, info in items:
        if model_key is None or cache_key[0] == model_key:
            result.append(info)
    return sorted(result, key=lambda item: item.loaded_at, reverse=True)


def describe_loaded_model_state(model_key: str, device: str) -> str:
    infos = get_loaded_model_info(model_key)
    if not infos:
        return "모델 없음"
    for info in infos:
        if info.device == device:
            scope = "GPU 사용 중" if info.device == "cuda" else "CPU 사용 중"
            return f"모델 로딩됨 | {scope} | {info.compute_type}"
    return "모델 로딩됨"


class TranscriptionEngine:
    def __init__(self, runtime_config: RuntimeConfig, model_key: str = DEFAULT_MODEL_KEY) -> None:
        self.runtime_config = runtime_config
        self.model_key = model_key

    def _cache_key(self) -> tuple[str, str, str, int, int]:
        return (
            self.model_key,
            self.runtime_config.device,
            self.runtime_config.compute_type,
            self.runtime_config.cpu_threads,
            self.runtime_config.num_workers,
        )

    def _load_model(self) -> WhisperModel:
        cache_key = self._cache_key()
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                meta = _MODEL_META.get(cache_key)
                if meta is not None:
                    meta.last_used_at = time.time()
                return cached

        model_dir = get_model_dir(self.model_key)
        if not is_model_ready(model_dir, self.model_key):
            raise RuntimeError("The selected model is not ready yet. Download it first.")

        model = WhisperModel(
            str(model_dir),
            device=self.runtime_config.device,
            compute_type=self.runtime_config.compute_type,
            cpu_threads=self.runtime_config.cpu_threads,
            num_workers=self.runtime_config.num_workers,
        )

        now = time.time()
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[cache_key] = model
            _MODEL_META[cache_key] = LoadedModelInfo(
                model_key=self.model_key,
                device=self.runtime_config.device,
                compute_type=self.runtime_config.compute_type,
                cpu_threads=self.runtime_config.cpu_threads,
                num_workers=self.runtime_config.num_workers,
                loaded_at=now,
                last_used_at=now,
            )
        return model

    def get_media_duration(self, source_path: Path) -> float:
        try:
            with av.open(str(source_path)) as container:
                if container.duration is not None:
                    return max(0.0, float(container.duration / av.time_base))
                stream = next((s for s in container.streams if s.type in {"audio", "video"}), None)
                if stream is not None and stream.duration is not None and stream.time_base is not None:
                    return max(0.0, float(stream.duration * stream.time_base))
        except Exception:
            pass
        return 0.0

    def transcribe_file(
        self,
        source_path: Path,
        progress_callback: Callable[[int, float, float], None] | None = None,
        language_code: str | None = None,
        vad_settings: VADSettings | None = None,
    ) -> list[SubtitleSegment]:
        model = self._load_model()
        media_duration = self.get_media_duration(source_path)
        vad = vad_settings or VADSettings()
        memory_profile = self.runtime_config.memory_profile
        vad_parameters = {
            "min_silence_duration_ms": max(0, int(vad.min_silence_duration_ms)),
            "speech_pad_ms": max(0, int(vad.speech_pad_ms)),
        }

        beam_size = 5 if self.runtime_config.device == "cuda" else 3
        chunk_length = 30
        condition_on_previous_text = True
        if memory_profile == "save":
            beam_size = 3 if self.runtime_config.device == "cuda" else 2
            chunk_length = 20
        elif memory_profile == "strict":
            beam_size = 2 if self.runtime_config.device == "cuda" else 1
            chunk_length = 15
            condition_on_previous_text = False

        segments, _ = model.transcribe(
            str(source_path),
            language=language_code,
            task="transcribe",
            beam_size=beam_size,
            vad_filter=bool(vad.enabled),
            vad_parameters=vad_parameters,
            condition_on_previous_text=condition_on_previous_text,
            chunk_length=chunk_length,
        )

        result: list[SubtitleSegment] = []
        last_percent = -1
        for segment in segments:
            text = segment.text.strip()
            if not text:
                if progress_callback is not None and media_duration > 0:
                    percent = min(100, int(float(segment.end) * 100 / media_duration))
                    if percent != last_percent:
                        last_percent = percent
                        progress_callback(percent, float(segment.end), media_duration)
                continue

            result.append(SubtitleSegment(start=float(segment.start), end=float(segment.end), text=text))
            if progress_callback is not None and media_duration > 0:
                percent = min(100, int(float(segment.end) * 100 / media_duration))
                if percent != last_percent:
                    last_percent = percent
                    progress_callback(percent, float(segment.end), media_duration)

        with _MODEL_CACHE_LOCK:
            meta = _MODEL_META.get(self._cache_key())
            if meta is not None:
                meta.last_used_at = time.time()

        if progress_callback is not None:
            progress_callback(100, media_duration, media_duration)
        return result
