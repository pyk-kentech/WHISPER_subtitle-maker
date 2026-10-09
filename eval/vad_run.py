"""Detect speech regions with one VAD over the whole eval set.
Usage:
  .venv python  vad_run.py silero       (needs faster_whisper -> run with the project venv)
  eval venv     vad_run.py whisperseg   (needs onnxruntime + transformers + av)
Writes vad/<name>/<id>.json = [{"start": s, "end": s}, ...]"""
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
AUDIO = HERE / "data" / "audio"
LIMITS = {"w8sleep": 600}
name = sys.argv[1]
out_dir = HERE / "vad" / name
out_dir.mkdir(parents=True, exist_ok=True)


def load(path: Path) -> np.ndarray:
    if name == "silero":
        from faster_whisper import decode_audio

        audio = decode_audio(str(path), sampling_rate=16000)
    else:
        import av

        chunks = []
        with av.open(str(path)) as container:
            resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray()[0])
        audio = np.concatenate(chunks).astype(np.float32)
    limit = LIMITS.get(path.stem)
    return audio[: limit * 16000] if limit else audio


if name == "silero":
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    # 앱과 같은 설정(VAD 최소 무음 500ms, 패딩 200ms)
    def detect(audio):
        return get_speech_timestamps(audio, VadOptions(min_silence_duration_ms=500, speech_pad_ms=200))

    frame_rate = 16000
else:
    # 제작자 inference.py는 모듈 첫 줄에서 torch/librosa를 부르지만, 여기서 쓰는 경로는 numpy만 쓴다.
    # transformers가 torch 유무를 먼저 확인하도록 가짜 모듈보다 먼저 불러 둔다.
    from transformers import WhisperFeatureExtractor  # noqa: F401
    sys.modules.setdefault("torch", types.SimpleNamespace(is_tensor=lambda _x: False))
    sys.modules.setdefault("librosa", types.ModuleType("librosa"))
    sys.path.insert(0, str(HERE / "whisperseg"))
    import inference as ws

    model = ws.WhisperVADOnnxWrapper(str(HERE / "whisperseg" / "model.onnx"),
                                     str(HERE / "whisperseg" / "model_metadata.json"), force_cpu=True, num_threads=8)

    def detect(audio):
        # 제작자 기본값(임계 0.5, 최소 무음 100ms, 패딩 30ms)에 앱과 같은 패딩만 맞춘다
        return ws.get_speech_timestamps(audio, model, threshold=0.5, min_silence_duration_ms=100,
                                        speech_pad_ms=200, return_seconds=False)

    frame_rate = 16000

total_audio = 0.0
started = time.monotonic()
for path in sorted(AUDIO.glob("*.mp3")):
    audio = load(path)
    total_audio += len(audio) / 16000
    segments = [{"start": s["start"] / frame_rate, "end": s["end"] / frame_rate} for s in detect(audio)]
    (out_dir / f"{path.stem}.json").write_text(json.dumps(segments), encoding="utf-8")
    print(f"{name} {path.stem}: {len(segments)} regions, {sum(s['end'] - s['start'] for s in segments):.0f}s speech", flush=True)
print(f"{name}: {total_audio / 60:.0f} min audio in {time.monotonic() - started:.0f}s")
