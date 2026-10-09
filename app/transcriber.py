from __future__ import annotations

import gc
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import av
import ctranslate2
import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio
from faster_whisper.transcribe import restore_speech_timestamps
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps

from .config import (
    DEFAULT_CLIPS_TRANSCRIBE_OPTIONS,
    DEFAULT_MAX_CLIP_SECONDS,
    GPU_DEFAULT_MODEL_KEY,
    DEFAULT_MODEL_KEY,
    VAD_MODEL_KEY,
    WHISPERSEG_DEFAULT_THRESHOLD,
)
from .cuda_runtime import (
    add_cuda_runtime_to_path,
    find_missing_cuda_libraries,
    has_cuda_device,
    is_cuda_runtime_ready,
)
from .model_manager import get_model_dir, get_model_languages, get_model_preset, is_model_ready
from .srt_writer import SubtitleSegment

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


SAMPLE_RATE = 16000
# vad_clips 방식에서 한 번에 넘기는 구간 수. 배치 파이프라인이 넘긴 구간의 멜 특징을 한꺼번에 메모리에 올리므로
# (구간당 약 1.5MB) 긴 파일도 메모리가 일정하도록 나눠 보낸다.
VAD_CLIP_GROUP_SIZE = 32

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
    # "silero"(faster-whisper 내장) 또는 "whisperseg"(ASMR용, 속삭임에 강함)
    backend: str = "silero"
    # "normal": Whisper가 스스로 자막 시간을 냄 / "clips": VAD 구간마다 잘라 인식하고 구간 경계를 자막 시간으로 씀
    segmentation: str = "normal"
    # None이면 VAD별 기본값(Silero 0.5, WhisperSeg 0.5)
    threshold: float | None = None
    # 구간별 인식에서 한 구간의 최대 길이(초). 길면 가장 조용한 지점에서 나눈다.
    max_clip_seconds: float = DEFAULT_MAX_CLIP_SECONDS


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


def uses_vad_clips(model_key: str, vad: VADSettings) -> bool:
    preset = get_model_preset(model_key)
    return preset.get("segmentation") == "vad_clips" or (bool(vad.enabled) and vad.segmentation == "clips")


def needs_whisperseg_model(model_key: str, vad: VADSettings) -> bool:
    return (bool(vad.enabled) or uses_vad_clips(model_key, vad)) and vad.backend == "whisperseg"


_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str, str, int, int], WhisperModel] = {}
_MODEL_META: dict[tuple[str, str, str, int, int], LoadedModelInfo] = {}


def is_cuda_runtime_available() -> tuple[bool, str]:
    if not has_cuda_device():
        return False, "CUDA GPU was not detected."

    add_cuda_runtime_to_path()
    if not is_cuda_runtime_ready():
        return False, "CUDA runtime is not ready yet."

    missing_libraries = find_missing_cuda_libraries()
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


def get_default_model_key() -> str:
    """GPU가 있으면 더 무겁지만 정확한 모델, CPU만 있으면 CPU에서도 쓸 만한 속도의 모델."""
    return GPU_DEFAULT_MODEL_KEY if has_cuda_device() else DEFAULT_MODEL_KEY


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
        self.last_detected_language: str | None = None
        # 평가·튜닝용: beam_size 등 디코딩 옵션을 덮어쓴다(앱은 비워 둔다).
        self.decode_overrides: dict = {}

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
        preset = get_model_preset(self.model_key)
        supported_languages = get_model_languages(self.model_key)
        if supported_languages and language_code not in supported_languages:
            # 일본어 전용 모델 등: 자동 감지나 다른 언어를 고르면 모델이 아는 언어로 고정한다.
            language_code = supported_languages[0]
        memory_profile = self.runtime_config.memory_profile
        vad_enabled = bool(vad.enabled)
        vad_parameters = None
        if vad_enabled:
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

        overrides = dict(self.decode_overrides)
        beam_size = int(overrides.pop("beam_size", beam_size))

        if uses_vad_clips(self.model_key, vad):
            return self._transcribe_vad_clips(
                model, source_path, progress_callback, language_code, vad, beam_size, media_duration, preset, overrides
            )

        transcribe_input = str(source_path)
        speech_chunks = None
        progress_offset = 0
        if vad_enabled and vad.backend == "whisperseg":
            # faster-whisper의 vad_filter와 같은 방식(발화 구간만 이어 붙여 인식하고 시간을 되돌림)에 VAD만 바꾼다.
            audio = decode_audio(str(source_path), sampling_rate=SAMPLE_RATE)
            media_duration = audio.shape[0] / SAMPLE_RATE or media_duration
            speech_chunks = self.detect_speech(audio, vad, float("inf"), progress_callback, media_duration)
            progress_offset = 10
            if speech_chunks:
                audio_chunks, _metadata = collect_chunks(audio, speech_chunks)
                transcribe_input = np.concatenate(audio_chunks, axis=0)
            vad_enabled = False
            vad_parameters = None
        elif vad_enabled and vad.threshold is not None:
            vad_parameters["threshold"] = float(vad.threshold)

        if speech_chunks is not None and not speech_chunks:
            segments, info = iter(()), None
        else:
            segments, info = model.transcribe(
                transcribe_input,
                language=language_code,
                task="transcribe",
                beam_size=beam_size,
                vad_filter=vad_enabled,
                vad_parameters=vad_parameters,
                condition_on_previous_text=condition_on_previous_text,
                chunk_length=chunk_length,
                **overrides,
            )
            if speech_chunks is not None:
                segments = restore_speech_timestamps(segments, speech_chunks, SAMPLE_RATE)
        self.last_detected_language = getattr(info, "language", None) or language_code

        result: list[SubtitleSegment] = []
        last_percent = -1
        span = 100 - progress_offset
        for segment in segments:
            text = segment.text.strip()
            if not text:
                if progress_callback is not None and media_duration > 0:
                    percent = min(100, progress_offset + int(float(segment.end) * span / media_duration))
                    if percent != last_percent:
                        last_percent = percent
                        progress_callback(percent, float(segment.end), media_duration)
                continue

            result.append(SubtitleSegment(start=float(segment.start), end=float(segment.end), text=text))
            if progress_callback is not None and media_duration > 0:
                percent = min(100, progress_offset + int(float(segment.end) * span / media_duration))
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

    def detect_speech(
        self,
        audio,
        vad: VADSettings,
        max_seconds: float,
        progress_callback: Callable[[int, float, float], None] | None = None,
        duration: float = 0.0,
    ) -> list[dict]:
        """발화 구간을 샘플 단위 [{"start", "end"}]로 돌려준다. WhisperSeg는 진행률의 앞 10%를 쓴다."""
        if vad.backend == "whisperseg":
            from .whisperseg_vad import WhisperSegOptions, get_whisperseg_vad, probs_to_segments

            model_dir = get_model_dir(VAD_MODEL_KEY)
            if not is_model_ready(model_dir, VAD_MODEL_KEY):
                raise RuntimeError("ASMR VAD(WhisperSeg) 모델이 아직 준비되지 않았습니다.")
            threads = self.runtime_config.cpu_threads or None

            def on_vad_progress(ratio: float) -> None:
                if progress_callback is not None:
                    progress_callback(int(ratio * 10), ratio * duration, duration)

            probs = get_whisperseg_vad(model_dir, threads).speech_probs(audio, on_vad_progress)
            options = WhisperSegOptions(
                threshold=float(vad.threshold) if vad.threshold is not None else WHISPERSEG_DEFAULT_THRESHOLD,
                min_silence_duration_ms=max(0, int(vad.min_silence_duration_ms)),
                speech_pad_ms=max(0, int(vad.speech_pad_ms)),
                max_speech_duration_s=max_seconds,
            )
            return [
                {"start": int(seg["start"] * SAMPLE_RATE), "end": min(len(audio), int(seg["end"] * SAMPLE_RATE))}
                for seg in probs_to_segments(probs, options)
            ]
        vad_options = VadOptions(
            min_silence_duration_ms=max(0, int(vad.min_silence_duration_ms)),
            speech_pad_ms=max(0, int(vad.speech_pad_ms)),
            max_speech_duration_s=max_seconds,
        )
        if vad.threshold is not None:
            vad_options.threshold = float(vad.threshold)
        return get_speech_timestamps(audio, vad_options)

    def _transcribe_vad_clips(
        self,
        model: WhisperModel,
        source_path: Path,
        progress_callback: Callable[[int, float, float], None] | None,
        language_code: str | None,
        vad: VADSettings,
        beam_size: int,
        media_duration: float,
        preset: dict,
        overrides: dict | None = None,
    ) -> list[SubtitleSegment]:
        """VAD로 발화 구간을 나눠 구간마다 따로 인식한다. 자막 시간은 VAD 구간 경계를 쓴다
        (타임스탬프를 스스로 내지 못하는 짧은 대사용 모델, 긴 무음이 많은 ASMR용)."""
        audio = decode_audio(str(source_path), sampling_rate=SAMPLE_RATE)
        duration = audio.shape[0] / SAMPLE_RATE or media_duration
        max_seconds = float(vad.max_clip_seconds or preset.get("max_clip_seconds", DEFAULT_MAX_CLIP_SECONDS))
        clips = [
            {"start": clip["start"] / SAMPLE_RATE, "end": clip["end"] / SAMPLE_RATE}
            for clip in self.detect_speech(audio, vad, max_seconds, progress_callback, duration)
        ]
        progress_offset = 10 if vad.backend == "whisperseg" else 0
        self.last_detected_language = language_code
        if self.runtime_config.device == "cuda":
            batch_size = {"save": 4, "strict": 2}.get(self.runtime_config.memory_profile, 8)
        else:
            batch_size = 1 if self.runtime_config.memory_profile == "strict" else 4
        pipeline = BatchedInferencePipeline(model)
        extra_options = dict(preset.get("transcribe_options", DEFAULT_CLIPS_TRANSCRIBE_OPTIONS))
        extra_options.update(overrides or {})
        if not extra_options.get("no_repeat_ngram_size"):
            extra_options.pop("no_repeat_ngram_size", None)

        result: list[SubtitleSegment] = []
        last_percent = -1
        span = 99 - progress_offset
        for group_start in range(0, len(clips), VAD_CLIP_GROUP_SIZE):
            group = clips[group_start : group_start + VAD_CLIP_GROUP_SIZE]
            segments, info = pipeline.transcribe(
                audio,
                language=language_code,
                task="transcribe",
                beam_size=beam_size,
                clip_timestamps=group,
                without_timestamps=True,
                batch_size=batch_size,
                **extra_options,
            )
            if language_code is None and getattr(info, "language", None):
                # 자동 감지면 첫 묶음에서 정한 언어를 나머지에도 쓴다(묶음마다 언어가 바뀌지 않게).
                language_code = info.language
                self.last_detected_language = language_code
            for segment in segments:
                text = segment.text.strip()
                if text:
                    result.append(SubtitleSegment(start=float(segment.start), end=float(segment.end), text=text))
                if progress_callback is not None and duration > 0:
                    percent = min(99, progress_offset + int(float(segment.end) * span / duration))
                    if percent != last_percent:
                        last_percent = percent
                        progress_callback(percent, float(segment.end), duration)

        with _MODEL_CACHE_LOCK:
            meta = _MODEL_META.get(self._cache_key())
            if meta is not None:
                meta.last_used_at = time.time()

        if progress_callback is not None:
            progress_callback(100, duration, duration)
        return result
