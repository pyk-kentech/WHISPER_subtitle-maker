"""Whisper-family model + Qwen3-ASR on the SAME WhisperSeg clips (<=12 s), per clip:
  W      : Whisper model text (faster-whisper batched, beam 5, no_repeat_ngram 5) + avg_logprob / compression ratio
  Q      : Qwen3-ASR-1.7B plain (language=Japanese)
  Qctx   : Qwen3-ASR with the Whisper text as context (system prompt)   -> item (b)
  sel*   : per-clip choice between W and Q/Qctx by confidence rules       -> item (c)
  W-fa   : W text, subtitle start/end re-timed with Qwen3-ForcedAligner   -> item (3)
Usage (wj-venv): python combo_clips.py <whisper_model_key> [qwen_repo]
Writes /root/asr-eval/out/cmb-<key>-<variant>/<id>.json and per-clip details in /root/asr-eval/combo/<key>/<id>.json"""
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/asr-eval/code")
from faster_whisper import BatchedInferencePipeline, WhisperModel, decode_audio  # noqa: E402

from app.config import VAD_MODEL_KEY  # noqa: E402
from app.config import get_model_cache_dir_for as get_model_dir  # noqa: E402  (wj-venv의 huggingface_hub가 낮아 model_manager를 못 씀)
from app.whisperseg_vad import WhisperSegOptions, get_whisperseg_vad, probs_to_segments  # noqa: E402

E = Path("/root/asr-eval")
DATA = E / "data"
wkey = sys.argv[1]
qwen_repo = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-ASR-1.7B"
qtag = "" if qwen_repo == "Qwen/Qwen3-ASR-1.7B" else "-" + qwen_repo.split("/")[-1][:12]
only = set(os.environ["ONLY"].split(",")) if os.environ.get("ONLY") else None
detail_dir = E / "combo" / f"{wkey}{qtag}"
detail_dir.mkdir(parents=True, exist_ok=True)
REP = re.compile(r"(.{2,6})\1{4,}")
SR = 16000


def norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def is_bad(text: str) -> bool:
    return not norm(text) or bool(REP.search(norm(text)))


manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
vad = get_whisperseg_vad(get_model_dir(VAD_MODEL_KEY))
wmodel = WhisperModel(str(get_model_dir(wkey)), device="cuda", compute_type="float16")
wpipe = BatchedInferencePipeline(wmodel)
from qwen_asr import Qwen3ASRModel  # noqa: E402

qmodel = Qwen3ASRModel.from_pretrained(
    qwen_repo,
    forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=16,
    max_new_tokens=256,
)
extra_opts = {}
if wkey == "anime-whisper":
    extra_opts["repetition_penalty"] = 1.0

timing = {}
for item in manifest:
    tid = item["id"]
    if only and tid not in only:
        continue
    detail_path = detail_dir / f"{tid}.json"
    if detail_path.exists():
        rows = json.loads(detail_path.read_text(encoding="utf-8"))["clips"]
    else:
        audio = decode_audio(str(DATA / "audio" / f"{tid}.mp3"), sampling_rate=SR)
        t0 = time.monotonic()
        clips = probs_to_segments(vad.speech_probs(audio), WhisperSegOptions(min_silence_duration_ms=100, speech_pad_ms=200, max_speech_duration_s=12))
        t_vad = time.monotonic() - t0
        rows = [{"start": c["start"], "end": c["end"], "W": "", "W_lp": None, "W_cr": None} for c in clips]
        t0 = time.monotonic()
        for g in range(0, len(clips), 32):
            group = clips[g : g + 32]
            segs, _ = wpipe.transcribe(audio, language="ja", task="transcribe", beam_size=5, clip_timestamps=group,
                                       without_timestamps=True, batch_size=8, no_repeat_ngram_size=5, **extra_opts)
            for s in segs:
                idx = g + int(np.argmin([abs(s.start - c["start"]) for c in group]))
                r = rows[idx]
                r["W"] += s.text.strip()
                r["W_lp"] = s.avg_logprob if r["W_lp"] is None else min(r["W_lp"], s.avg_logprob)
                r["W_cr"] = s.compression_ratio if r["W_cr"] is None else max(r["W_cr"], s.compression_ratio)
        t_w = time.monotonic() - t0
        pieces = [(audio[int(r["start"] * SR) : int(r["end"] * SR)], SR) for r in rows]
        t0 = time.monotonic()
        q = qmodel.transcribe(audio=pieces, language="Japanese") if pieces else []
        t_q = time.monotonic() - t0
        t0 = time.monotonic()
        qc = qmodel.transcribe(audio=pieces, context=[r["W"] for r in rows], language="Japanese") if pieces else []
        t_qc = time.monotonic() - t0
        # ForcedAligner: W 글자를 구간 안에서 정렬해 첫·마지막 글자 시각으로 자막 시간을 다시 잡는다
        t0 = time.monotonic()
        fa_idx = [i for i, r in enumerate(rows) if norm(r["W"])]
        aligned = qmodel.forced_aligner.align(audio=[pieces[i] for i in fa_idx], text=[rows[i]["W"] for i in fa_idx],
                                              language="Japanese") if fa_idx else []
        t_fa = time.monotonic() - t0
        for i, r in enumerate(rows):
            r["Q"] = q[i].text if i < len(q) else ""
            r["Qctx"] = qc[i].text if i < len(qc) else ""
        for i, res in zip(fa_idx, aligned):
            items = list(res)
            if items:
                rows[i]["fa"] = [float(items[0].start_time), float(items[-1].end_time)]
        timing[tid] = {"vad": t_vad, "W": t_w, "Q": t_q, "Qctx": t_qc, "FA": t_fa}
        detail_path.write_text(json.dumps({"timing": timing[tid], "clips": rows}, ensure_ascii=False), encoding="utf-8")
        print(f"{wkey} {tid}: {len(rows)} clips vad {t_vad:.0f}s W {t_w:.0f}s Q {t_q:.0f}s Qctx {t_qc:.0f}s FA {t_fa:.0f}s", flush=True)

    def choose_sel1(r):  # Qctx 우선, 비거나 반복이면 W
        return r["Qctx"] if not is_bad(r["Qctx"]) else r["W"]

    def choose_sel2(r):  # W가 자신 있고(로그확률 >= -0.6, 압축비 <= 2.4) 정상이면 W, 아니면 Q
        good_w = not is_bad(r["W"]) and (r["W_lp"] or -9) >= -0.6 and (r["W_cr"] or 9) <= 2.4
        return r["W"] if good_w or is_bad(r["Q"]) else r["Q"]

    def choose_sel3(r):  # 더 긴 쪽(반복 제외): 놓친 말이 적은 쪽
        cands = [t for t in (r["W"], r["Q"]) if not is_bad(t)]
        return max(cands, key=lambda t: len(norm(t))) if cands else ""

    variants = {
        "W": lambda r: (r["W"], r["start"], r["end"]),
        "Q": lambda r: (r["Q"], r["start"], r["end"]),
        "Qctx": lambda r: (r["Qctx"], r["start"], r["end"]),
        "sel1": lambda r: (choose_sel1(r), r["start"], r["end"]),
        "sel2": lambda r: (choose_sel2(r), r["start"], r["end"]),
        "sel3": lambda r: (choose_sel3(r), r["start"], r["end"]),
        "W-fa": lambda r: (r["W"],) + ((max(r["start"], r["start"] + r["fa"][0] - 0.1), min(r["end"], r["start"] + r["fa"][1] + 0.2)) if r.get("fa") else (r["start"], r["end"])),
    }
    for name, fn in variants.items():
        out = E / "out" / f"cmb-{wkey}{qtag}-{name}"
        out.mkdir(parents=True, exist_ok=True)
        segs = []
        for r in rows:
            text, s, e = fn(r)
            if norm(text):
                segs.append({"start": s, "end": e, "text": text})
        (out / f"{tid}.json").write_text(json.dumps({"elapsed": None, "segments": segs}, ensure_ascii=False), encoding="utf-8")
print(f"combo {wkey} DONE", flush=True)
