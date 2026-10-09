import json, sys
import metrics as M
track = sys.argv[1]
ref = (M.DATA / "gt" / f"{track}.txt").read_text(encoding="utf-8")
cues = json.loads((M.DATA / "timeline" / f"{track}.json").read_text())
dur = json.loads((M.DATA / "durations.json").read_text())[track]
for system in sys.argv[2:]:
    p = M.OUT / system / f"{track}.json"
    if not p.exists():
        print(f"{system}: (not yet)"); continue
    segs = json.loads(p.read_text(encoding="utf-8"))["segments"]
    hyp = "".join(s["text"] for s in segs)
    k = M.cer(M.to_kana(ref), M.to_kana(hyp))
    t = M.timeline_scores(cues, segs, dur)
    print(f"{system:26} kanaCER {k['cer']:.3f} (del {k['del']:.3f} ins {k['ins']:.3f} sub {k['sub']:.3f}) | recall {t['recall']:.2f} | broken {100*t['broken_ratio']:.0f}% | phantom {100*t['phantom_dur_ratio']:.1f}% | rep {M.repetition_flags(segs)}")
