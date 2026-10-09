"""Fill the table placeholders in RESULTS.md from score_split.py output (and the solo benchmark log).
Usage (eval venv): python fill_results.py"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def table(split: str, *patterns: str, raw: bool = False) -> str:
    out = subprocess.run([sys.executable, str(HERE / "score_split.py"), split, *patterns], capture_output=True, text=True,
                         encoding="utf-8", cwd=HERE).stdout
    lines = [ln for ln in out.splitlines() if ln.startswith("|") and (raw or "-raw |" not in ln)]
    return "\n".join(lines)


def bench_table() -> str:
    log = HERE / "pod_logs" / "lane_bench.txt"
    durations = 634.538662
    rows = ["| 구성 | 처리 시간(초) | RTF | 최대 VRAM(MiB) |", "|---|---|---|---|"]
    if not log.exists():
        return "(벤치마크 로그 없음)"
    for line in log.read_text(encoding="utf-8").splitlines():
        m = re.search(r"end (b-\S+) exit=0 peak_vram_mib=(\d+)", line)
        if m:
            meta = HERE / "out" / m.group(1) / "w7t2.json"
            elapsed = json.loads(meta.read_text(encoding="utf-8"))["elapsed"] if meta.exists() else None
            rows.append(f"| {m.group(1)[2:]} (앱) | {elapsed:.0f} | {elapsed / durations:.3f} | {m.group(2)} |" if elapsed else f"| {m.group(1)} | - | - | {m.group(2)} |")
        m = re.search(r"bench (bw-\S+) elapsed=([\d.]+)s peak_vram_mib=(\d+)", line)
        if m:
            e = float(m.group(2))
            rows.append(f"| WhisperJAV {m.group(1)[3:]} | {e:.0f} | {e / durations:.3f} | {m.group(3)} |")
        m = re.search(r"bench combo elapsed=(\d+)s peak_vram_mib=(\d+)", line)
        if m:
            e = float(m.group(1))
            rows.append(f"| combo(anime-whisper W + Qwen3-ASR Q·Qctx + ForcedAligner, 모델 로딩 포함) | {e:.0f} | {e / durations:.3f} | {m.group(2)} |")
    return "\n".join(rows)


def main() -> None:
    p = HERE / "RESULTS.md"
    s = p.read_text(encoding="utf-8")
    s = s.replace("ALL_TABLE", table("all", "app-*", "wj-*", "cmb-anime-whisper-*", "cmb-large-v3-turbo-*", "mrg-*", "final-*"))
    s = s.replace("TUNE_GRID", table("tune", "p2-*", raw=True))
    s = s.replace("TUNE_SWEEP", table("tune", "t-*", "p2-anime-whisper-whisperseg-clips"))
    s = s.replace("VAL_TABLE", table("val", "final-*", "app-medium", "app-anime", "app-turbo", "wj-qwen", "wj-qwen-anime", "wj-ens19",
                                      "cmb-anime-whisper-W", "cmb-anime-whisper-Qctx", "cmb-large-v3-turbo-sel2"))
    s = s.replace("BENCH_TABLE", bench_table())
    p.write_text(s, encoding="utf-8", newline="\n")
    print("filled")


if __name__ == "__main__":
    main()
