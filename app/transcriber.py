from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable

import av
import ctranslate2
from faster_whisper import WhisperModel

from .config import MODEL_LANGUAGE
from .model_manager import get_model_dir, is_model_ready
from .srt_writer import SubtitleSegment


@dataclass(slots=True)
class RuntimeConfig:
    device: str
    compute_type: str
    cpu_threads: int
    label: str


def get_available_runtime_choices() -> list[tuple[str, str]]:
    choices = [("cpu", "CPU")]
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            choices.insert(0, ("cuda", "GPU (CUDA)"))
    except Exception:
        pass
    return choices


def get_default_runtime_choice() -> str:
    for device, _label in get_available_runtime_choices():
        if device == "cuda":
            return "cuda"
    return "cpu"


def build_runtime_config(device: str) -> RuntimeConfig:
    normalized = device.lower().strip()
    if normalized == "cuda":
        try:
            if ctranslate2.get_cuda_device_count() <= 0:
                raise RuntimeError("CUDA GPU를 찾지 못했습니다.")
        except Exception as exc:
            raise RuntimeError(f"GPU 사용 불가: {exc}") from exc

        supported = ctranslate2.get_supported_compute_types("cuda")
        for compute_type in ("float16", "int8_float16", "int8_float32", "float32"):
            if compute_type in supported:
                return RuntimeConfig(
                    device="cuda",
                    compute_type=compute_type,
                    cpu_threads=0,
                    label=f"GPU (CUDA, {compute_type})",
                )
        raise RuntimeError("GPU에서 사용할 수 있는 compute type을 찾지 못했습니다.")

    supported = ctranslate2.get_supported_compute_types("cpu")
    for compute_type in ("int8", "int8_float32", "float32"):
        if compute_type in supported:
            cpu_count = os.cpu_count() or 4
            cpu_threads = max(1, min(cpu_count - 1, 8))
            return RuntimeConfig(
                device="cpu",
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                label=f"CPU ({compute_type}, {cpu_threads} threads)",
            )
    raise RuntimeError("CPU에서 사용할 수 있는 compute type을 찾지 못했습니다.")


class TranscriptionEngine:
    def __init__(self, runtime_config: RuntimeConfig) -> None:
        self.runtime_config = runtime_config
        self._model: WhisperModel | None = None

    def _load_model(self) -> WhisperModel:
        if self._model is not None:
            return self._model

        model_dir = get_model_dir()
        if not is_model_ready(model_dir):
            raise RuntimeError("모델이 준비되지 않았습니다. 먼저 다운로드를 완료하세요.")

        self._model = WhisperModel(
            str(model_dir),
            device=self.runtime_config.device,
            compute_type=self.runtime_config.compute_type,
            cpu_threads=self.runtime_config.cpu_threads,
            num_workers=1,
        )

        return self._model

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
    ) -> list[SubtitleSegment]:
        model = self._load_model()
        media_duration = self.get_media_duration(source_path)
        segments, _ = model.transcribe(
            str(source_path),
            language=MODEL_LANGUAGE,
            task="transcribe",
            beam_size=5 if self.runtime_config.device == "cuda" else 3,
            vad_filter=True,
            condition_on_previous_text=True,
            chunk_length=30,
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
            result.append(
                SubtitleSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                )
            )
            if progress_callback is not None and media_duration > 0:
                percent = min(100, int(float(segment.end) * 100 / media_duration))
                if percent != last_percent:
                    last_percent = percent
                    progress_callback(percent, float(segment.end), media_duration)

        if progress_callback is not None:
            progress_callback(100, media_duration, media_duration)

        return result
