"""Build the translation test set (GT input): passages from each eval work's Japanese script + a synthetic adult subset.
Writes translate/set_gt.json = [{"id", "work", "track", "tag", "text"}]. Content stays local (gitignored)."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
GT = HERE.parent / "data" / "gt"
# (track, first line, line count, tag)  tag = 장르 특성 메모(채점 때 참고)
PASSAGES = [
    ("w1t01", 1, 34, "일상 대화·츤데레 말투"),
    ("w2t01", 1, 32, "일상·보쿠(ボク)·늘어지는 말투"),
    ("w3t03", 1, 24, "츤데레 모드"),
    ("w3t06", 1, 14, "아가씨(ですわ) 말투"),
    ("w4t02", 1, 34, "메이드 경어·의성어(がらがら·あわあわ)"),
    ("w5t06", 1, 30, "목욕·속삭임"),
    ("w6t1", 1, 34, "게임 캐릭터(닥터 호칭)"),
    ("w7t2", 1, 48, "갸루어·속삭임 의성어"),
]


def main() -> None:
    rows = []
    for track, first, count, tag in PASSAGES:
        lines = [ln.strip() for ln in (GT / f"{track}.txt").read_text(encoding="utf-8").splitlines()]
        lines = [ln for ln in lines if re.sub(r"[「」『』\s…]", "", ln)]
        for index, text in enumerate(lines[first - 1 : first - 1 + count]):
            rows.append({"id": f"{track}-{first + index:03d}", "work": track[:2], "track": track, "tag": tag, "text": text})
    scene = "misc"
    counter = 0
    for raw in (HERE / "synthetic_r18.txt").read_text(encoding="utf-8").splitlines():
        if raw.startswith("# scene:") or raw.startswith("# misc"):
            scene = raw.split(":", 1)[1].split("(")[0].strip()
            continue
        if not raw.strip() or raw.startswith("#"):
            continue
        counter += 1
        tag = "말투(비성인)" if scene == "speech styles" else "성인 대사(합성)"
        rows.append({"id": f"syn-{counter:03d}", "work": "syn", "track": f"syn-{scene}", "tag": tag, "text": raw.strip()})
    (HERE / "set_gt.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(rows)} lines ({sum(1 for r in rows if r['work'] == 'syn')} synthetic)")


if __name__ == "__main__":
    main()
