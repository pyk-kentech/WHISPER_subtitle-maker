"""Transcribe one track using a given VAD's regions as clips (app's vad_clips approach).
Usage: .venv python asr_on_vad.py <model_key> <vad_name> <track>  -> out/<model>+<vad>/<track>.json"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, r"D:\WHISPER_subtitle-maker")
from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio  # noqa: E402

from app.model_manager import get_model_dir  # noqa: E402

HERE = Path(__file__).parent
model_key, vad_name, track = sys.argv[1:4]
MAX_CLIP = 12.0  # 앱 anime-whisper 프리셋과 같은 최대 길이

regions = json.loads((HERE / "vad" / vad_name / f"{track}.json").read_text())
clips = []
for r in regions:  # 긴 구간은 12초 이하로 고르게 나눈다
    length = r["end"] - r["start"]
    parts = max(1, int(-(-length // MAX_CLIP)))
    step = length / parts
    clips += [{"start": r["start"] + i * step, "end": r["start"] + (i + 1) * step} for i in range(parts)]

audio = decode_audio(str(HERE / "data" / "audio" / f"{track}.mp3"), sampling_rate=16000)
model = WhisperModel(str(get_model_dir(model_key)), device="cpu", compute_type="int8", cpu_threads=12)
pipe = BatchedInferencePipeline(model)
started = time.monotonic()
segments = []
for i in range(0, len(clips), 32):
    result, _ = pipe.transcribe(audio, language="ja", beam_size=3, clip_timestamps=clips[i:i + 32],
                                without_timestamps=True, no_repeat_ngram_size=5, batch_size=4)
    segments += [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in result if s.text.strip()]
    print(f"{i + 32}/{len(clips)} clips", flush=True)
out = HERE / "out" / f"{model_key}+{vad_name}"
out.mkdir(parents=True, exist_ok=True)
(out / f"{track}.json").write_text(json.dumps({"elapsed": time.monotonic() - started, "segments": segments}, ensure_ascii=False), encoding="utf-8")
print(f"done {model_key}+{vad_name} {track}: {len(segments)} segments in {time.monotonic() - started:.0f}s")
