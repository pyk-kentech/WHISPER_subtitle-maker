"""Compare VAD outputs in ./vad/<name>/ against the Chinese-subtitle timeline."""
import json
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
durations = json.loads((DATA / "durations.json").read_text())
names = sorted(p.name for p in (HERE / "vad").iterdir() if p.is_dir())


def covered(cue, regions):
    length = max(1e-6, cue["end"] - cue["start"])
    hit = sum(max(0.0, min(cue["end"], r["end"]) - max(cue["start"], r["start"])) for r in regions)
    return min(1.0, hit / length)


def outside(regions, cues, pad=1.0):
    """VAD가 말이라고 했지만 중국어 자막 어느 줄과도 겹치지 않는 시간(초)."""
    total = 0.0
    for r in regions:
        overlap = sum(max(0.0, min(r["end"], c["end"] + pad) - max(r["start"], c["start"] - pad)) for c in cues)
        total += max(0.0, (r["end"] - r["start"]) - overlap)
    return total


rows = {}
for name in names:
    per = {}
    for path in sorted((HERE / "vad" / name).glob("*.json")):
        item = path.stem
        regions = json.loads(path.read_text())
        cues = json.loads((DATA / "timeline" / f"{item}.json").read_text())
        speech = sum(r["end"] - r["start"] for r in regions)
        row = {"speech_s": speech}
        if cues:
            row["recall"] = sum(1 for c in cues if covered(c, regions) >= 0.5) / len(cues)
            row["false_s_per_min"] = outside(regions, cues) / (durations[item] / 60)
        per[item] = row
    rows[name] = per

items = sorted(next(iter(rows.values())))
print(f"{'track':8} " + " ".join(f"{n + ' recall':>18} {'false s/min':>11}" for n in names))
for item in items:
    cells = []
    for n in names:
        r = rows[n].get(item, {})
        if "recall" in r:
            cells.append(f"{r['recall']:18.2f} {r['false_s_per_min']:11.1f}")
        else:
            cells.append(f"{'(no speech) ' + format(r.get('speech_s', 0), '.0f') + 's found':>30}")
    print(f"{item:8} " + " ".join(cells))
for n in names:
    rs = [r for i, r in rows[n].items() if "recall" in r]
    print(f"{n:10} mean recall {sum(r['recall'] for r in rs) / len(rs):.3f} | mean false {sum(r['false_s_per_min'] for r in rs) / len(rs):.1f} s/min"
          f" | sleep-track speech {rows[n].get('w8sleep', {}).get('speech_s', 0):.0f}s")

# 속삭임 구간(w7t2 60~600초) 집중 확인
for n in names:
    regions = json.loads((HERE / "vad" / n / "w7t2.json").read_text())
    inside = sum(max(0.0, min(r["end"], 600) - max(r["start"], 60)) for r in regions)
    print(f"{n}: speech found in w7t2 60-600s whisper section = {inside:.0f}s of 540s")
(HERE / "vad_results.json").write_text(json.dumps(rows, indent=1))
