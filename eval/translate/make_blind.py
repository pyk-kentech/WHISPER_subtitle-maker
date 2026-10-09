"""Blind review sheet + automatic counts for the translation comparison.
Usage: python make_blind.py <set_name> <system> <system> [...]
  reads out/<set>__<system>.json, writes out/blind_<set>.md (A/B/C shuffled per group) and out/blind_<set>_key.json,
  and prints automatic counts per system (missing lines, misalignment retries, refusal phrases, leftover Japanese)."""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
GROUP = 8
KANA = re.compile(r"[぀-ヿ]")
REFUSAL = re.compile(r"죄송|할 수 없|도와드릴 수|부적절|I can't|I cannot|cannot assist|申し訳|不適切|規約|정책|policy", re.I)


def main() -> None:
    set_name, systems = sys.argv[1], sys.argv[2:]
    rows = json.loads((HERE / f"{set_name}.json").read_text(encoding="utf-8"))
    results = {s: json.loads((HERE / "out" / f"{set_name}__{s}.json").read_text(encoding="utf-8")) for s in systems}
    stats = {}
    for system, data in results.items():
        lines = data["lines"]
        events = data.get("events", [])
        missing = [i for i, v in lines.items() if v is None]
        empty = [i for i, v in lines.items() if v is not None and not v.strip()]
        leftover = [i for i, v in lines.items() if v and len(KANA.findall(v)) >= 3]
        refusal = [i for i, v in lines.items() if v and REFUSAL.search(v)]
        misaligned = sum(int(m.group(1)) for e in events for m in [re.search(r"\| (\d+)/\d+ lines misaligned", e)] if m)
        splits = sum(1 for e in events if "splitting failed chunk" in e)
        blocked = sum(1 for e in events if "Blocked" in e or "blocked" in e)
        syn = [r["id"] for r in rows if r["work"] == "syn"]
        stats[system] = {
            "lines": len(lines),
            "missing": len(missing),
            "missing_syn": sum(1 for i in missing if i in syn),
            "empty": len(empty),
            "leftover_japanese": len(leftover),
            "refusal_phrase": len(refusal),
            "misaligned_first_pass": misaligned,
            "chunk_splits": splits,
            "blocked_events": blocked,
            "elapsed_s": round(data.get("elapsed", 0), 1),
            "requests": data.get("requests"),
            "tokens": data.get("tokens"),
        }
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    (HERE / "out" / f"stats_{set_name}.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")

    rng = random.Random(20261009)
    letters = "ABCDEFG"[: len(systems)]
    key = {}
    out = [f"# 블라인드 비교: {set_name}\n", "각 묶음마다 A/B/C가 어느 시스템인지 섞었다(키는 별도 파일).\n"]
    for g in range(0, len(rows), GROUP):
        group = rows[g : g + GROUP]
        order = systems[:]
        rng.shuffle(order)
        gid = f"G{g // GROUP + 1:02d}"
        key[gid] = dict(zip(letters, order))
        out.append(f"\n## {gid} ({group[0]['track']}, {group[0]['tag']})\n")
        for r in group:
            out.append(f"- `{r['id']}` 원문: {r['text']}")
            for letter, system in zip(letters, order):
                value = results[system]["lines"].get(r["id"])
                out.append(f"  - {letter}: {value if value is not None else '⟨번역 없음⟩'}")
    (HERE / "out" / f"blind_{set_name}.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    (HERE / "out" / f"blind_{set_name}_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote out/blind_{set_name}.md ({len(key)} groups)")


if __name__ == "__main__":
    main()
