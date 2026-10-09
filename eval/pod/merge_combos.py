"""Item 4(a) without re-running ensembles: merge existing pass outputs with WhisperJAV's own MergeEngine.
pass1 = Whisper-family result, pass2 = Qwen3-ASR (wj-qwen), strategies = pass1_primary / smart_merge / longest.
Usage (wj-venv): python merge_combos.py   -> /root/asr-eval/out/mrg-<pass1>-<strategy>/<id>.json"""
import json
import sys
import tempfile
from pathlib import Path

from whisperjav.ensemble.merge import MergeEngine

OUT = Path("/root/asr-eval/out")
PASS1 = {"anime": "wj-qwen-anime", "ja15b": "wj-balanced-ja15b", "turbo": "cmb-large-v3-turbo-W"}
PASS2 = "wj-qwen"
STRATEGIES = ["pass1_primary", "smart_merge", "longest"]


def fmt(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def to_srt(segments, path: Path) -> None:
    lines = []
    for i, s in enumerate(segments, 1):
        lines += [str(i), f"{fmt(s['start'])} --> {fmt(s['end'])}", s["text"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def from_srt(path: Path):
    blocks = path.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n\n")
    segs = []
    for block in blocks:
        rows = [r for r in block.strip().split("\n") if r.strip()]
        if len(rows) < 3 or "-->" not in rows[1]:
            continue
        a, b = [x.strip() for x in rows[1].split("-->")]
        conv = lambda x: int(x[:2]) * 3600 + int(x[3:5]) * 60 + int(x[6:8]) + int(x[9:12]) / 1000
        segs.append({"start": conv(a), "end": conv(b), "text": "".join(rows[2:])})
    return segs


engine = MergeEngine()
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for p1name, p1 in PASS1.items():
        if not (OUT / p1).is_dir() or not (OUT / PASS2).is_dir():
            print(f"skip {p1name}: missing {p1} or {PASS2}", file=sys.stderr)
            continue
        for strategy in STRATEGIES:
            target = OUT / f"mrg-{p1name}-{strategy}"
            target.mkdir(parents=True, exist_ok=True)
            done = 0
            for f1 in sorted((OUT / p1).glob("*.json")):
                if f1.name.startswith("_"):
                    continue
                f2 = OUT / PASS2 / f1.name
                if not f2.exists():
                    continue
                a, b, m = tmp / "a.srt", tmp / "b.srt", tmp / "m.srt"
                to_srt(json.loads(f1.read_text(encoding="utf-8"))["segments"], a)
                to_srt(json.loads(f2.read_text(encoding="utf-8"))["segments"], b)
                engine.merge(a, b, m, strategy)
                (target / f1.name).write_text(json.dumps({"elapsed": None, "segments": from_srt(m)}, ensure_ascii=False), encoding="utf-8")
                done += 1
            print(f"mrg-{p1name}-{strategy}: {done} tracks")
