"""Build the evaluation set: audio list, Japanese reference text (from scripts), and speech timeline (from Chinese subs).
Writes into ./data (next to this file). Media stays local until uploaded to /dev/shm on the server."""
import json
import re
import shutil
import unicodedata
from pathlib import Path

from pypdf import PdfReader

ROOT = Path("D:/samples")
OUT = Path(__file__).parent / "data"

SELECTION = [
    # id, work, mp3 (relative to work), script source key, track key
    ("w1t01", "RJ01060913", "mp3/n08_01_『在海边散步』.mp3", "txt:附赠/剧本_日文.txt", 1),
    ("w1t05", "RJ01060913", "mp3/n08_05_『清洁左耳』.mp3", "txt:附赠/剧本_日文.txt", 5),
    ("w2t01", "RJ01077014", "mp3/h19_01_会站着睡着的仅限一日的恋人？.mp3", "txt:附赠/剧本_日文.txt", 1),
    ("w2t09", "RJ01077014", "mp3/h19_09_陪睡摸摸＆钛制颂钵.mp3", "txt:附赠/剧本_日文.txt", 9),
    ("w3t03", "RJ01099308", "MP3/h23_03_用傲娇模式医疗用掏耳（右耳）.mp3", "txt:omake/剧本_日文.txt", 3),
    ("w3t06", "RJ01099308", "MP3/h23_06_用钻头双马尾大小姐模式进行同人活动.mp3", "txt:omake/剧本_日文.txt", 6),
    ("w4t02", "RJ01111194", "MP3/mr03_02_『主人，请张嘴』.mp3", "txt:附赠/剧本_日文.txt", 2),
    ("w4t06", "RJ01111194", "MP3/mr03_06_『我帮您掏耳哦主人（左）』.mp3", "txt:附赠/剧本_日文.txt", 6),
    ("w5t02", "RJ01183891", "正篇/mp3/02_EP01：雪音的请求.mp3", "w5pdf", "EP01"),
    ("w5t06", "RJ01183891", "正篇/mp3/06_EP04：如泡沫一般嬉戏.mp3", "w5pdf", "EP04"),
    ("w6t1", "RJ01494389", "MP3/1-卡兹戴尔的夜晚（导入）.mp3", "w6pdf", 1),
    ("w6t7", "RJ01494389", "MP3/7-伴您身侧（陪睡）.mp3", "w6pdf", 7),
    ("w7t2", "RJ427560", "Track2.使用绵柔辣妹用语的创新ASMR.mp3", "plain:日语剧本/2.txt", None),
    # 숨소리만 있는 트랙: 정답은 "대사 없음" (앞 10분만 사용)
    ("w8sleep", "RJ01183891", "正篇/mp3/12_特典音轨：睡息(正面).mp3", "none", None),
]

PAREN = re.compile(r"（[^（）]*）|\([^()]*\)")
FW_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp932", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(path)


def clean_dialogue(text: str) -> str:
    for _ in range(2):
        text = PAREN.sub("", text)
    return text


def split_txt_script(text: str) -> dict[int, str]:
    """W1~W4: トラックN 표시로 나누고 대사(「」『』 및 그 연속 줄)만 남긴다."""
    tracks: dict[int, list[str]] = {}
    current = None
    last_number = None
    marker = re.compile(r"^[/;●―\s]*トラック\s*([0-9０-９]+)\s*[:：]")
    for line in text.splitlines():
        match = marker.match(line)
        if match:
            number = int(match.group(1).translate(FW_DIGITS))
            if last_number is not None and number <= last_number:
                number = last_number + 1  # 대본 오타(같은 번호 두 번) 보정
            last_number = number
            current = number
            tracks[current] = []
            continue
        if current is None:
            continue
        stripped = line.strip().strip("\u3000")
        if not stripped or stripped[0] in "/;●―ー◆" or stripped.startswith("【"):
            continue
        tracks[current].append(stripped)
    return {number: clean_dialogue("\n".join(lines)) for number, lines in tracks.items()}


def w5_script() -> dict[str, str]:
    reader = PdfReader(str(ROOT / "RJ01183891/台本/花样年华-被少女饲养的宠物的我-』脚本_日语.pdf"))
    text = "".join((page.extract_text() or "") for page in reader.pages)
    text = re.sub(r"\s+", "", text)
    parts = re.split(r"【(Prologue|EP\d+-S|EP\d+|Epilogue)[：:][^】]*】", text)
    sections = {}
    for index in range(1, len(parts), 2):
        body = parts[index + 1]
        quotes = re.findall(r"「(.*?)」", body)
        sections[parts[index]] = clean_dialogue("\n".join(quotes))
    return sections


def w6_script() -> dict[int, str]:
    reader = PdfReader(str(ROOT / "RJ01494389/赠品/日文台本.pdf"))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    tracks: dict[int, list[str]] = {}
    current = None
    for line in text.splitlines():
        line = re.sub(r"\d+\s*$", "", line).strip()  # 줄 끝 줄번호 제거
        match = re.match(r"P(\d+)\s", line)
        if match:
            current = int(match.group(1))
            tracks[current] = []
            continue
        if current is None or not line or line[0] in "■▲★(（" or "【" in line or "】" in line:
            continue
        tracks[current].append(line)
    return {number: clean_dialogue("\n".join(lines)) for number, lines in tracks.items()}


def parse_vtt(path: Path) -> list[dict]:
    cues = []
    stamp = re.compile(r"(\d+):(\d+):(\d+)[.,](\d+)\s*-->\s*(\d+):(\d+):(\d+)[.,](\d+)")
    for line in read_text(path).splitlines():
        m = stamp.search(line)
        if m:
            g = [int(x) for x in m.groups()]
            start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
            end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
            cues.append({"start": round(start, 3), "end": round(end, 3)})
    return cues


def parse_lrc(path: Path) -> list[dict]:
    starts = []
    for line in read_text(path).splitlines():
        for mm, ss in re.findall(r"\[(\d+):(\d+(?:\.\d+)?)\]", line):
            text = re.sub(r"\[[^\]]*\]", "", line).strip()
            if text:
                starts.append(int(mm) * 60 + float(ss))
    starts.sort()
    cues = []
    for i, start in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else start + 6
        cues.append({"start": round(start, 3), "end": round(min(nxt, start + 8), 3)})
    return cues


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "audio").mkdir(parents=True)
    (OUT / "gt").mkdir()
    (OUT / "timeline").mkdir()
    cache: dict[str, dict] = {}
    manifest = []
    for item_id, work, rel, source, key in SELECTION:
        mp3 = ROOT / work / rel
        shutil.copyfile(mp3, OUT / "audio" / f"{item_id}.mp3")
        if source.startswith("txt:"):
            if (work, source) not in cache:
                cache[(work, source)] = split_txt_script(read_text(ROOT / work / source[4:]))
            reference = cache.get((work, source), cache.get(source))[key]
        elif source == "w5pdf":
            cache.setdefault(source, w5_script())
            reference = cache.get((work, source), cache.get(source))[key]
        elif source == "w6pdf":
            cache.setdefault(source, w6_script())
            reference = cache.get((work, source), cache.get(source))[key]
        elif source.startswith("plain:"):
            reference = clean_dialogue(read_text(ROOT / work / source[6:]))
        else:
            reference = ""
        (OUT / "gt" / f"{item_id}.txt").write_text(reference, encoding="utf-8")

        vtt = Path(str(mp3) + ".vtt")
        lrc = ROOT / work / "LRC字幕文件" / (mp3.stem + ".lrc")
        if vtt.exists():
            timeline, kind = parse_vtt(vtt), "vtt"
        elif lrc.exists():
            timeline, kind = parse_lrc(lrc), "lrc"
        else:
            timeline, kind = [], "none"
        (OUT / "timeline" / f"{item_id}.json").write_text(json.dumps(timeline), encoding="utf-8")
        chars = len(re.sub(r"[^\w]", "", unicodedata.normalize("NFKC", reference)))
        manifest.append({"id": item_id, "work": work, "source": rel, "ref_chars": chars,
                         "timeline": kind, "cues": len(timeline), "clip_seconds": 600 if item_id == "w8sleep" else None})
        print(f"{item_id:8} ref_chars={chars:6} timeline={kind}:{len(timeline):4}  {rel}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
