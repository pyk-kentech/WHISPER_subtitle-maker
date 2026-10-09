"""Run the app's own pipeline (TranscriptionEngine + Japanese postprocess, same options as PipelineWorker) on the Runpod pod.
Usage: python run_app.py <model_key> <system_name> [key=value ...]
Extra key=value pairs tune the run (vad=silero|whisperseg, mode=normal|clips, beam=N, nrng=N, thr=.., min_silence=ms, pad=ms, max_clip=s).
Writes /root/asr-eval/out/<system>/<id>.json (post-processed) and <system>-raw/<id>.json (before sentence merge)."""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/root/asr-eval/code")
from app.config import MODEL_PRESETS  # noqa: E402

# 평가용 임시 프리셋(앱에 없을 수 있음)
_CT2_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json")
MODEL_PRESETS.setdefault("ja-1.5b", {"label": "whisper-ja-1.5B", "repo_id": "TransWithAI/whisper-ja-1.5B-ct2", "files": _CT2_FILES})
MODEL_PRESETS.setdefault("kotoba-v2", {"label": "kotoba-whisper-v2.0", "repo_id": "kotoba-tech/kotoba-whisper-v2.0-faster", "files": _CT2_FILES})

from app.japanese_postprocess import PostprocessOptions, postprocess_japanese_segments  # noqa: E402
from app.model_manager import download_model, is_model_ready  # noqa: E402
from app import transcriber as T  # noqa: E402

DATA = Path("/root/asr-eval/data")
model_key, system = sys.argv[1], sys.argv[2]
opts = dict(a.split("=", 1) for a in sys.argv[3:])
only = set(os.environ["ONLY"].split(",")) if os.environ.get("ONLY") else None
out_dir = Path("/root/asr-eval/out") / system
out_dir.mkdir(parents=True, exist_ok=True)
raw_dir = out_dir.parent / f"{system}-raw"
raw_dir.mkdir(exist_ok=True)

if not is_model_ready(model_key=model_key):
    download_model(model_key, lambda *a: None, lambda m: print(m, flush=True))
from app.config import VAD_MODEL_KEY  # noqa: E402
if opts.get("vad") == "whisperseg" and not is_model_ready(model_key=VAD_MODEL_KEY):
    download_model(VAD_MODEL_KEY, lambda *a: None, lambda m: print(m, flush=True))

device = os.environ.get("DEVICE", "cuda")
tuning = T.RuntimeTuningOptions(cpu_threads=int(os.environ.get("THREADS", "8"))) if device == "cpu" else T.RuntimeTuningOptions()
engine = T.TranscriptionEngine(T.build_runtime_config(device, tuning), model_key)
vad_kwargs = {}
if "min_silence" in opts:
    vad_kwargs["min_silence_duration_ms"] = int(opts["min_silence"])
if "pad" in opts:
    vad_kwargs["speech_pad_ms"] = int(opts["pad"])
vad = T.VADSettings(**vad_kwargs)
# 새 옵션(앱에 들어간 뒤에만 존재): 없으면 무시
for field, key in (("backend", "vad"), ("segmentation", "mode"), ("threshold", "thr"), ("max_clip_seconds", "max_clip")):
    if key in opts and hasattr(vad, field):
        cast = float if field in ("threshold", "max_clip_seconds") else str
        setattr(vad, field, cast(opts[key]))
if "beam" in opts:
    engine.decode_overrides["beam_size"] = int(opts["beam"])
if "nrng" in opts:
    engine.decode_overrides["no_repeat_ngram_size"] = int(opts["nrng"])
if opts.get("vad") == "off":
    vad.enabled = False
options = PostprocessOptions(
    enabled=True, enhanced=False, sentence=True, standard_asia=True, max_comma=2, max_gap=0.35, one_word=True
)
manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
total = 0.0
for item in manifest:
    if only and item["id"] not in only:
        continue
    target = out_dir / f"{item['id']}.json"
    if target.exists():
        continue
    started = time.monotonic()
    raw = engine.transcribe_file(DATA / "audio" / f"{item['id']}.mp3", None, language_code="ja", vad_settings=vad)
    elapsed = time.monotonic() - started
    total += elapsed
    raw_rows = [{"start": s.start, "end": s.end, "text": s.text} for s in raw]
    segments = postprocess_japanese_segments(raw, options)
    rows = [{"start": s.start, "end": s.end, "text": s.text} for s in segments]
    target.write_text(json.dumps({"elapsed": elapsed, "segments": rows}, ensure_ascii=False), encoding="utf-8")
    (raw_dir / f"{item['id']}.json").write_text(json.dumps({"elapsed": elapsed, "segments": raw_rows}, ensure_ascii=False), encoding="utf-8")
    print(f"{system} {item['id']} segments={len(rows)} {elapsed:.1f}s", flush=True)
print(f"{system} DONE {total:.1f}s", flush=True)
