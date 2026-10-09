"""Score every system in ./out against ./data (script text = reference words, Chinese subs = reference timeline).
Usage: python metrics.py   -> prints a table and writes results.json"""
import json
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

from rapidfuzz.distance import Levenshtein
from sudachipy import dictionary, tokenizer

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "out"
TOKENIZER = dictionary.Dictionary(dict="core").tokenizer()
SPLIT_MODE = tokenizer.Tokenizer.SplitMode.C
HALLUCINATION_PHRASES = ("ご視聴ありがとうございました", "チャンネル登録", "おやすみなさいおやすみなさいおやすみなさい")


def strip_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


def to_kana(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    readings = []
    for chunk in re.split(r"[^\w]+", text):
        if not chunk:
            continue
        for morpheme in TOKENIZER.tokenize(chunk, SPLIT_MODE):
            reading = morpheme.reading_form() or morpheme.surface()
            readings.append(reading)
    kana = "".join(readings)
    # 가타카나 -> 히라가나, 장음/작은 글자 차이는 그대로 둔다
    kana = "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in kana)
    return strip_text(kana)


def cer(reference: str, hypothesis: str) -> dict:
    if not reference:
        return {"cer": None}
    ops = Counter(op.tag for op in Levenshtein.editops(reference, hypothesis))
    n = len(reference)
    return {
        "cer": (ops["replace"] + ops["delete"] + ops["insert"]) / n,
        "sub": ops["replace"] / n,
        "del": ops["delete"] / n,
        "ins": ops["insert"] / n,
    }


def overlaps(a_start, a_end, b_start, b_end, pad=0.0):
    return a_start < b_end + pad and b_start < a_end + pad


MAX_SEGMENT_SECONDS = 15.0  # 이보다 긴 자막 한 줄은 타임스탬프가 깨진 것으로 본다


def covered_fraction(cue: dict, segments: list[dict], pad: float = 0.3) -> float:
    length = max(1e-6, cue["end"] - cue["start"])
    total = 0.0
    for s in segments:
        lo = max(cue["start"], s["start"] - pad)
        hi = min(cue["end"], s["end"] + pad)
        total += max(0.0, hi - lo)
    return min(1.0, total / length)


def timeline_scores(cues: list[dict], segments: list[dict], duration: float) -> dict:
    broken = [s for s in segments if s["end"] - s["start"] > MAX_SEGMENT_SECONDS]
    valid = [s for s in segments if s["end"] - s["start"] <= MAX_SEGMENT_SECONDS]
    broken_ratio = sum(s["end"] - s["start"] for s in broken) / max(duration, 1e-6)
    if not cues:
        return {"broken_ratio": broken_ratio, "segments": len(segments)}
    covered = sum(1 for c in cues if covered_fraction(c, valid) >= 0.5)
    phantom = [s for s in segments if not any(overlaps(c["start"], c["end"], s["start"], s["end"], 0.5) for c in cues)]
    total_dur = sum(max(0.0, s["end"] - s["start"]) for s in segments) or 1e-9
    cue_starts = [c["start"] for c in cues]
    offsets = []
    for s in segments:
        if s in phantom:
            continue
        offsets.append(min(abs(s["start"] - cs) for cs in cue_starts))
    return {
        "recall": covered / len(cues),
        "phantom_count": len(phantom),
        "phantom_dur_ratio": sum(max(0.0, s["end"] - s["start"]) for s in phantom) / total_dur,
        "start_offset_median": statistics.median(offsets) if offsets else None,
        "segments": len(segments),
        "broken_ratio": broken_ratio,
    }


def repetition_flags(segments: list[dict]) -> int:
    flags = 0
    for s in segments:
        text = strip_text(s["text"])
        if any(p in text for p in HALLUCINATION_PHRASES):
            flags += 1
            continue
        # 같은 2~6글자가 5번 넘게 연속 반복되면 반복 환각으로 본다
        if re.search(r"(.{2,6})\1{4,}", text):
            flags += 1
    return flags


def main() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    durations = json.loads((DATA / "durations.json").read_text(encoding="utf-8"))
    refs = {m["id"]: (DATA / "gt" / f"{m['id']}.txt").read_text(encoding="utf-8") for m in manifest}
    ref_strip = {k: strip_text(v) for k, v in refs.items()}
    ref_kana = {k: to_kana(v) for k, v in refs.items()}
    results = {}
    for system_dir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        system = system_dir.name
        per_item = {}
        for m in manifest:
            path = system_dir / f"{m['id']}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            segments = data["segments"]
            hyp = "".join(s["text"] for s in segments)
            cues = json.loads((DATA / "timeline" / f"{m['id']}.json").read_text(encoding="utf-8"))
            row = {
                "elapsed": data.get("elapsed"),
                "duration": durations[m["id"]],
                "hyp_chars": len(strip_text(hyp)),
                "repetition_flags": repetition_flags(segments),
            }
            if ref_strip[m["id"]]:
                row["char"] = cer(ref_strip[m["id"]], strip_text(hyp))
                row["kana"] = cer(ref_kana[m["id"]], to_kana(hyp))
            row["timeline"] = timeline_scores(cues, segments, durations[m["id"]])
            per_item[m["id"]] = row
        results[system] = per_item
    (HERE / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    def pooled(items, key):
        num = sum(r[key]["cer"] * len(ref_kana[i] if key == "kana" else ref_strip[i]) for i, r in items if key in r)
        den = sum(len(ref_kana[i] if key == "kana" else ref_strip[i]) for i, r in items if key in r)
        return num / den if den else float("nan")

    print(f"{'system':20} {'n':>2} {'CER':>6} {'kanaCER':>7} {'recall':>6} {'broken%':>7} {'phantom%':>8} {'offset':>6} {'rep':>3} {'sleep':>5} {'RTF':>6}")
    for system, per_item in results.items():
        items = [(i, r) for i, r in per_item.items() if i != "w8sleep"]
        tl = [r["timeline"] for _, r in items if r["timeline"]]
        sleep = per_item.get("w8sleep", {})
        meta = OUT / system / "_meta.json"
        elapsed = json.loads(meta.read_text())["elapsed_total"] if meta.exists() else sum(r["elapsed"] or 0 for r in per_item.values())
        duration = sum(r["duration"] for r in per_item.values())
        broken = sum(r["timeline"]["broken_ratio"] * r["duration"] for _, r in items) / sum(r["duration"] for _, r in items)
        print(
            f"{system:20} {len(per_item):2} {pooled(items, 'char'):6.3f} {pooled(items, 'kana'):7.3f} "
            f"{statistics.mean(t['recall'] for t in tl):6.3f} {100 * broken:6.1f}% {100 * statistics.mean(t['phantom_dur_ratio'] for t in tl):7.1f}% "
            f"{statistics.median(t['start_offset_median'] for t in tl if t['start_offset_median'] is not None):6.2f} "
            f"{sum(r['repetition_flags'] for _, r in items):3} {sleep.get('hyp_chars', '-'):>5} {elapsed / duration:6.3f}"
        )


if __name__ == "__main__":
    main()
