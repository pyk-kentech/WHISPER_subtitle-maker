"""Convert WhisperJAV SRT output to the eval JSON format. Usage: python srt_to_json.py <srt_dir> <out_dir> <elapsed_seconds>"""
import json
import re
import sys
from pathlib import Path

srt_dir, out_dir, elapsed = Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
out_dir.mkdir(parents=True, exist_ok=True)
stamp = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
count = 0
for srt in sorted(srt_dir.rglob("*.srt")):
    item_id = srt.name.split(".")[0]
    segments = []
    for block in re.split(r"\n\s*\n", srt.read_text(encoding="utf-8-sig").replace("\r\n", "\n")):
        lines = block.strip().split("\n")
        for index, line in enumerate(lines):
            m = stamp.search(line)
            if m:
                g = [int(x) for x in m.groups()]
                start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
                end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
                text = "".join(lines[index + 1 :]).strip()
                if text:
                    segments.append({"start": start, "end": end, "text": text})
                break
    (out_dir / f"{item_id}.json").write_text(json.dumps({"elapsed": None, "segments": segments}, ensure_ascii=False), encoding="utf-8")
    count += 1
(out_dir / "_meta.json").write_text(json.dumps({"elapsed_total": elapsed}), encoding="utf-8")
print(f"converted {count} srt files from {srt_dir}")
