"""ASR-input condition: for each GT passage in set_gt.json, take the ASR subtitles (system in ../out/<system>/) that cover
the same speech, found by fuzzy-aligning the passage text against the track's ASR text.
Usage (eval venv): python build_asr_set.py <system>   -> set_asr.json (one row per ASR subtitle line)"""
import json
import re
import sys
import unicodedata
from pathlib import Path

from rapidfuzz import fuzz

HERE = Path(__file__).parent
OUT = HERE.parent / "out"


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in "LN")


def main() -> None:
    system = sys.argv[1]
    gt_rows = json.loads((HERE / "set_gt.json").read_text(encoding="utf-8"))
    passages: dict[str, list[dict]] = {}
    for row in gt_rows:
        if row["work"] != "syn":
            passages.setdefault(row["track"], []).append(row)
    rows = []
    for track, prows in passages.items():
        segs = json.loads((OUT / system / f"{track}.json").read_text(encoding="utf-8"))["segments"]
        texts = [norm(s["text"]) for s in segs]
        offsets, total = [], 0
        for t in texts:
            offsets.append(total)
            total += len(t)
        full = "".join(texts)
        target = norm("".join(r["text"] for r in prows))
        align = fuzz.partial_ratio_alignment(target, full) if len(target) <= len(full) else None
        if align is None:
            continue
        lo, hi = align.dest_start, align.dest_end
        picked = [i for i, off in enumerate(offsets) if off + len(texts[i]) > lo and off < hi and texts[i]]
        for n, i in enumerate(picked):
            rows.append({"id": f"{track}-asr{n + 1:03d}", "work": prows[0]["work"], "track": track, "tag": prows[0]["tag"],
                         "text": segs[i]["text"].strip(), "start": segs[i]["start"], "end": segs[i]["end"]})
        print(f"{track}: passage {len(target)} chars ~ ASR span {hi - lo} chars, score {align.score:.0f}, {len(picked)} lines")
    (HERE / "set_asr.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} ASR lines from {system}")


if __name__ == "__main__":
    main()
