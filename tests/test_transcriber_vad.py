"""VADSettings / uses_vad_clips / needs_whisperseg_model / detect_speech 테스트 (모델 다운로드 없음)."""
from __future__ import annotations

import numpy as np
import pytest

import app.transcriber as transcriber
import app.whisperseg_vad as whisperseg_vad
from app.transcriber import RuntimeConfig, TranscriptionEngine, VADSettings, needs_whisperseg_model, uses_vad_clips


def cpu_engine(model_key: str = "medium") -> TranscriptionEngine:
    config = RuntimeConfig(device="cpu", compute_type="int8", cpu_threads=2, num_workers=1, memory_profile="unlimited", label="CPU")
    return TranscriptionEngine(config, model_key)


def test_vad_settings_defaults():
    vad = VADSettings()
    assert vad.enabled and vad.backend == "silero" and vad.segmentation == "normal"
    assert vad.threshold is None and vad.max_clip_seconds == 12


@pytest.mark.parametrize(
    "model_key, vad, expected",
    [
        ("medium", VADSettings(enabled=True, segmentation="clips"), True),
        ("medium", VADSettings(enabled=True, segmentation="normal"), False),
        ("medium", VADSettings(enabled=False, segmentation="clips"), False),
        # anime-whisper는 VAD 설정과 관계없이 항상 구간별 인식
        ("anime-whisper", VADSettings(enabled=False, segmentation="normal"), True),
    ],
)
def test_uses_vad_clips(model_key, vad, expected):
    assert uses_vad_clips(model_key, vad) is expected


@pytest.mark.parametrize(
    "model_key, vad, expected",
    [
        ("medium", VADSettings(enabled=True, backend="whisperseg"), True),
        ("medium", VADSettings(enabled=True, backend="silero"), False),
        ("medium", VADSettings(enabled=False, backend="whisperseg"), False),
        ("anime-whisper", VADSettings(enabled=False, backend="whisperseg"), True),
        ("anime-whisper", VADSettings(enabled=False, backend="silero"), False),
    ],
)
def test_needs_whisperseg_model(model_key, vad, expected):
    assert needs_whisperseg_model(model_key, vad) is expected


def test_silero_detect_speech_on_silence_returns_nothing():
    """faster-whisper에 들어 있는 Silero ONNX로 실제 실행(다운로드 없음)."""
    audio = np.zeros(16000 * 3, dtype=np.float32)
    assert cpu_engine().detect_speech(audio, VADSettings(backend="silero"), 12.0) == []


def test_silero_detect_speech_passes_options(monkeypatch):
    captured = {}

    def fake_get_speech_timestamps(audio, options):
        captured["options"] = options
        return [{"start": 0, "end": 16000}]

    monkeypatch.setattr(transcriber, "get_speech_timestamps", fake_get_speech_timestamps)
    vad = VADSettings(backend="silero", min_silence_duration_ms=-5, speech_pad_ms=150, threshold=0.42)
    result = cpu_engine().detect_speech(np.zeros(16000, dtype=np.float32), vad, 12.0)
    options = captured["options"]
    assert result == [{"start": 0, "end": 16000}]
    assert options.min_silence_duration_ms == 0  # 음수는 0으로
    assert options.speech_pad_ms == 150
    assert options.max_speech_duration_s == 12.0
    assert options.threshold == pytest.approx(0.42)


def test_whisperseg_detect_speech_converts_frames_to_samples(monkeypatch):
    probs = np.concatenate([np.zeros(50), np.full(100, 0.9), np.zeros(50)]).astype(np.float32)
    progress: list[int] = []

    class FakeVad:
        def speech_probs(self, audio, progress_callback=None, cancel_check=None):
            progress_callback(1.0)
            return probs

    monkeypatch.setattr(transcriber, "is_model_ready", lambda *_a, **_k: True)
    monkeypatch.setattr(whisperseg_vad, "get_whisperseg_vad", lambda *_a, **_k: FakeVad())
    audio = np.zeros(16000 * 4, dtype=np.float32)
    vad = VADSettings(backend="whisperseg", min_silence_duration_ms=100, speech_pad_ms=0)
    result = cpu_engine().detect_speech(audio, vad, 12.0, lambda p, _c, _t: progress.append(p), 4.0)
    assert result == [{"start": 16000, "end": 48000}]
    assert progress == [10]  # WhisperSeg는 진행률 앞 10%를 쓴다


def test_whisperseg_detect_speech_requires_model(monkeypatch):
    monkeypatch.setattr(transcriber, "is_model_ready", lambda *_a, **_k: False)
    with pytest.raises(RuntimeError):
        cpu_engine().detect_speech(np.zeros(16000, dtype=np.float32), VADSettings(backend="whisperseg"), 12.0)
