"""Score systems in ./out on a split of the eval set and print a markdown table (same metrics as metrics.py).
Usage: python score_split.py <all|tune|val> [system glob ...] [--json out.json]
Split is by WORK so tuning never sees validation works:
  tune = w1~w4 (RJ01060913, RJ01077014, RJ01099308, RJ01111194)
  val  = w5~w7 + w8sleep (RJ01183891, RJ01494389, RJ427560; w8sleep is RJ01183891's breathing-only track)"""
import fnmatch
import json
import statistics
import sys
from pathlib import Path

import metrics as M

SPLITS = {
    "tune": ["w1t01", "w1t05", "w2t01", "w2t09", "w3t03", "w3t06", "w4t02", "w4t06"],
    "val": ["w5t02", "w5t06", "w6t1", "w6t7", "w7t2", "w8sleep"],
}
SPLITS["all"] = SPLITS["tune"] + SPLITS["val"]


def main() -> None:
    args = [a for a in sys.argv[1:]]
    json_out = None
    if "--json" in args:
        i = args.index("--json")
        json_out = args[i + 1]
        del args[i : i + 2]
    split = args[0]
    patterns = args[1:] or ["*"]
    tracks = SPLITS[split]
    durations = json.loads((M.DATA / "durations.json").read_text(encoding="utf-8"))
    refs = {t: (M.DATA / "gt" / f"{t}.txt").read_text(encoding="utf-8") for t in tracks}
    ref_strip = {t: M.strip_text(v) for t, v in refs.items()}
    ref_kana = {t: M.to_kana(v) for t, v in refs.items()}
    systems = sorted(p.name for p in M.OUT.iterdir() if p.is_dir() and any(fnmatch.fnmatch(p.name, pat) for pat in patterns))
    rows = {}
    for system in systems:
        per = {}
        for t in tracks:
            path = M.OUT / system / f"{t}.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            segs = data["segments"]
            hyp = "".join(s["text"] for s in segs)
            cues = json.loads((M.DATA / "timeline" / f"{t}.json").read_text(encoding="utf-8"))
            row = {"elapsed": data.get("elapsed"), "hyp_chars": len(M.strip_text(hyp)), "rep": M.repetition_flags(segs),
                   "tl": M.timeline_scores(cues, segs, durations[t])}
            if ref_strip[t]:
                row["char"] = M.cer(ref_strip[t], M.strip_text(hyp))
                row["kana"] = M.cer(ref_kana[t], M.to_kana(hyp))
            per[t] = row
        speech = [t for t in tracks if ref_strip[t]]
        if not all(t in per for t in speech):
            missing = [t for t in speech if t not in per]
            print(f"<!-- {system}: incomplete, missing {missing} -->", file=sys.stderr)
            continue
        items = [(t, per[t]) for t in speech]
        pooled = lambda key, refd: sum(r[key]["cer"] * len(refd[t]) for t, r in items) / sum(len(refd[t]) for t, _ in items)
        tl = [r["tl"] for _, r in items]
        dur = sum(durations[t] for t, _ in items)
        elapsed = [r["elapsed"] for r in per.values() if r["elapsed"] is not None]
        meta = M.OUT / system / "_meta.json"
        total_elapsed = json.loads(meta.read_text())["elapsed_total"] if meta.exists() else (sum(elapsed) if len(elapsed) == len(per) else None)
        rows[system] = {
            "cer": pooled("char", ref_strip),
            "kana": pooled("kana", ref_kana),
            "recall": statistics.mean(x["recall"] for x in tl),
            "broken": sum(r["tl"]["broken_ratio"] * durations[t] for t, r in items) / dur,
            "phantom": statistics.mean(x["phantom_dur_ratio"] for x in tl),
            "rep": sum(r["rep"] for _, r in items),
            "sleep": per.get("w8sleep", {}).get("hyp_chars"),
            "rtf_parallel": (total_elapsed / sum(durations[t] for t in per)) if total_elapsed else None,
            "del": sum(r["kana"]["del"] * len(ref_kana[t]) for t, r in items) / sum(len(ref_kana[t]) for t, _ in items),
            "ins": sum(r["kana"]["ins"] * len(ref_kana[t]) for t, r in items) / sum(len(ref_kana[t]) for t, _ in items),
        }
    print(f"| system ({split}) | CER | 읽는 소리 CER | (누락/삽입) | 잡은 대사 | 깨진 자막 | 가짜 자막 | 반복 | 숨소리 트랙 글자 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for system, r in sorted(rows.items(), key=lambda kv: kv[1]["kana"]):
        sleep = "-" if r["sleep"] is None else str(r["sleep"])
        print(f"| {system} | {r['cer']:.3f} | **{r['kana']:.3f}** | {r['del']:.2f}/{r['ins']:.2f} | {100 * r['recall']:.0f}% | "
              f"{100 * r['broken']:.0f}% | {100 * r['phantom']:.1f}% | {r['rep']} | {sleep} |")
    if json_out:
        Path(json_out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
