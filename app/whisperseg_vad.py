"""ASMR용 VAD: WhisperSeg(TransWithAI/Whisper-Vad-EncDec-ASMR-onnx, MIT).

whisper-base 인코더 + 2층 디코더를 ONNX로 내보낸 모델로, 30초 창마다 20ms 프레임별 발화 확률을 낸다.
Silero가 놓치는 속삭임·귓속말을 잘 잡는다(평가셋에서 대사 구간 검출 66% -> 96%).

- 특징값: whisper-base 80-mel. faster-whisper의 FeatureExtractor(feature_size=80)가 transformers의
  WhisperFeatureExtractor와 같은 값을 낸다(최대 오차 4e-6)는 것을 확인했으므로 torch·transformers 없이 만든다.
- 후처리: 제작자 inference.py의 get_speech_timestamps와 같다(임계값 히스테리시스, 최소 무음, 패딩).
  다만 최대 길이를 넘는 구간은 그 자리에서 자르지 않고 구간 안에서 가장 확률이 낮은 프레임에서 나눈다.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE
FRAME_MS = 20
FRAMES_PER_CHUNK = CHUNK_SECONDS * 1000 // FRAME_MS
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# 모델 파일 이름(Hugging Face 저장소 그대로)
MODEL_FILENAME = "model.onnx"
METADATA_FILENAME = "model_metadata.json"


@dataclass(slots=True)
class WhisperSegOptions:
    threshold: float = 0.5
    # None이면 threshold - 0.15 (제작자 기본)
    neg_threshold: float | None = None
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 200
    max_speech_duration_s: float = float("inf")


class WhisperSegVad:
    """ONNX 세션을 감싼다. 같은 모델 폴더면 프로세스 안에서 한 번만 연다(get_whisperseg_vad)."""

    def __init__(self, model_dir: Path, cpu_threads: int | None = None) -> None:
        import onnxruntime as ort
        from faster_whisper.feature_extractor import FeatureExtractor

        model_path = Path(model_dir) / MODEL_FILENAME
        if not model_path.is_file():
            raise FileNotFoundError(f"WhisperSeg 모델 파일이 없습니다: {model_path}")
        metadata_path = Path(model_dir) / METADATA_FILENAME
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if int(metadata.get("frame_duration_ms", FRAME_MS)) != FRAME_MS:
                raise RuntimeError("지원하지 않는 WhisperSeg 모델입니다(프레임 길이가 20ms가 아님).")

        options = ort.SessionOptions()
        threads = cpu_threads or max(1, min(os.cpu_count() or 4, 8))
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 30초 창 하나에 수십 ms라 CPU로 충분하다. GPU 실행 공급자는 cuDNN 버전 문제를 일으키기 쉬워 쓰지 않는다.
        self._session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._feature_extractor = FeatureExtractor(feature_size=80)
        self._lock = threading.Lock()

    def speech_probs(
        self,
        audio: np.ndarray,
        progress_callback: Callable[[float], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        """16kHz 모노 float32 음성 -> 20ms 프레임별 발화 확률(길이 = ceil(초 * 50))."""
        audio = np.asarray(audio, dtype=np.float32)
        total_frames = int(np.ceil(len(audio) / FRAME_SAMPLES))
        probs: list[np.ndarray] = []
        chunk_count = max(1, int(np.ceil(len(audio) / CHUNK_SAMPLES)))
        for index in range(chunk_count):
            if cancel_check is not None and cancel_check():
                break
            chunk = audio[index * CHUNK_SAMPLES : (index + 1) * CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
            features = self._feature_extractor(chunk, padding=0)[:, :3000]
            features = features[np.newaxis].astype(np.float32)
            with self._lock:
                logits = self._session.run(None, {self._input_name: features})[0][0]
            probs.append(1.0 / (1.0 + np.exp(-logits.astype(np.float64))))
            if progress_callback is not None:
                progress_callback((index + 1) / chunk_count)
        if not probs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(probs)[:total_frames].astype(np.float32)


def probs_to_segments(probs: np.ndarray, options: WhisperSegOptions | None = None) -> list[dict]:
    """프레임 확률 -> 발화 구간 [{"start": 초, "end": 초}, ...] (제작자 get_speech_timestamps와 같은 규칙)."""
    opts = options or WhisperSegOptions()
    threshold = float(opts.threshold)
    neg_threshold = opts.neg_threshold if opts.neg_threshold is not None else max(threshold - 0.15, 0.01)
    min_speech_frames = int(opts.min_speech_duration_ms / FRAME_MS)
    min_silence_frames = int(opts.min_silence_duration_ms / FRAME_MS)
    pad_frames = int(opts.speech_pad_ms / FRAME_MS)
    total = len(probs)

    speeches: list[list[int]] = []
    triggered = False
    start = 0
    temp_end = 0
    for i, prob in enumerate(probs):
        if prob >= threshold and not triggered:
            triggered = True
            start = i
            temp_end = 0
            continue
        if not triggered:
            continue
        if prob < neg_threshold:
            if not temp_end:
                temp_end = i
            if i - temp_end >= min_silence_frames:
                if temp_end - start >= min_speech_frames:
                    speeches.append([start, temp_end])
                triggered = False
                temp_end = 0
        elif prob >= threshold and temp_end:
            temp_end = 0
    if triggered and total - start >= min_speech_frames:
        speeches.append([start, total])

    for i, speech in enumerate(speeches):
        lower = speeches[i - 1][1] if i > 0 else 0
        upper = speeches[i + 1][0] if i + 1 < len(speeches) else total
        speech[0] = max(lower, speech[0] - pad_frames)
        speech[1] = min(upper, speech[1] + pad_frames)

    if np.isfinite(opts.max_speech_duration_s) and opts.max_speech_duration_s > 0:
        max_frames = max(2, int(opts.max_speech_duration_s * 1000 / FRAME_MS))
        speeches = [part for speech in speeches for part in split_long_segment(speech, probs, max_frames)]

    seconds = FRAME_MS / 1000
    return [{"start": round(s * seconds, 3), "end": round(e * seconds, 3)} for s, e in speeches if e > s]


def split_long_segment(speech: list[int], probs: np.ndarray, max_frames: int) -> list[list[int]]:
    """max_frames보다 긴 구간을 가장 조용한(확률이 가장 낮은) 프레임에서 반복해서 나눈다.
    너무 짧은 조각이 생기지 않게 앞 조각 길이를 max의 절반~max 사이에서 고른다."""
    start, end = speech
    parts: list[list[int]] = []
    while end - start > max_frames:
        lo = start + max_frames // 2
        hi = start + max_frames
        cut = lo + int(np.argmin(probs[lo:hi]))
        parts.append([start, cut])
        start = cut
    parts.append([start, end])
    return parts


_VAD_CACHE: dict[str, WhisperSegVad] = {}
_VAD_CACHE_LOCK = threading.Lock()


def get_whisperseg_vad(model_dir: Path, cpu_threads: int | None = None) -> WhisperSegVad:
    key = str(Path(model_dir).resolve())
    with _VAD_CACHE_LOCK:
        vad = _VAD_CACHE.get(key)
        if vad is None:
            vad = WhisperSegVad(Path(model_dir), cpu_threads)
            _VAD_CACHE[key] = vad
        return vad


def unload_whisperseg_vad() -> None:
    with _VAD_CACHE_LOCK:
        _VAD_CACHE.clear()
