"""Translate a test set with the app's own translator code (same system prompt, <p id>/<o> echo format, context lines,
chunking, echo check, split retries). Usage (app .venv, from the project root):
  python eval/translate/run_translate.py <set.json> gemini
  python eval/translate/run_translate.py <set.json> llm <name> <base_url> <model>
Writes eval/translate/out/<set>__<system>.json with per-line output, timing, token usage and translator events.
API keys come from the Windows credential store and are never printed."""
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.credential_store import load_secret  # noqa: E402
from app.gemini_translator import GeminiTranslator, TranslationConfig  # noqa: E402
from app.openai_translator import OpenAICompatConfig, OpenAICompatTranslator  # noqa: E402
from app.translator_store import load_api_keys, load_translator_settings  # noqa: E402

HERE = Path(__file__).parent


@dataclass
class Record:
    line_id: str
    text: str


def main() -> None:
    set_path = Path(sys.argv[1])
    kind = sys.argv[2]
    rows = json.loads(set_path.read_text(encoding="utf-8"))
    settings = load_translator_settings()
    events: list[str] = []

    def log(message: str) -> None:
        events.append(message)
        if any(k in message for k in ("misaligned", "split", "failed", "Blocked", "blocked", "unreadable", "중단", "quota", "Rate")):
            print("  ", message[:160], flush=True)

    if kind == "gemini":
        keys = list(dict.fromkeys(load_api_keys() + [k for k in [load_secret("DongeumSubMaker/EvalGeminiKey").strip()] if k]))
        print(f"gemini keys available: {len(keys)} | preferred model: {settings.preferred_model or '(auto)'}")
        translator = GeminiTranslator(
            TranslationConfig(
                keys=keys,
                preferred_model=settings.preferred_model,
                target_language="ko",
                system_prompt=settings.system_prompt,
                translation_note=settings.translation_note,
                temperature=settings.temperature,
                top_p=settings.top_p,
                reasoning_level=settings.reasoning_level,
                chunk_size=settings.chunk_size,
                request_delay_seconds=settings.request_delay_seconds,
            ),
            log,
        )
        system = "gemini"
    else:
        name, base_url, model = sys.argv[3], sys.argv[4], sys.argv[5]
        translator = OpenAICompatTranslator(
            OpenAICompatConfig(base_url=base_url, model=model, target_language="ko", system_prompt=settings.system_prompt,
                               translation_note=settings.translation_note, chunk_size=settings.chunk_size),
            log,
        )
        system = name
    records = [Record(r["id"], r["text"]) for r in rows]
    started = time.monotonic()
    # 작품(트랙)마다 따로 번역한다(앱이 파일 하나씩 번역하는 것과 같게, 문맥이 다른 작품으로 넘어가지 않게).
    out: dict[str, str] = {}
    by_track: dict[str, list[Record]] = {}
    for r, rec in zip(rows, records):
        by_track.setdefault(r["track"], []).append(rec)
    request_count = 0
    for track, recs in by_track.items():
        before = len(events)
        try:
            out.update(translator.translate_lines(recs, lambda *a: None, track))
        except Exception as exc:  # noqa: BLE001
            out.update(getattr(exc, "partial", None) or {})
            log(f"[{track}] aborted: {exc}")
        request_count += sum(1 for e in events[before:] if "| model=" in e)
        print(f"{track}: {sum(1 for r in recs if r.line_id in out)}/{len(recs)} lines", flush=True)
    elapsed = time.monotonic() - started
    result = {
        "system": system,
        "model": getattr(translator, "compat", None).model if kind != "gemini" else settings.preferred_model,
        "elapsed": elapsed,
        "requests": request_count,
        "tokens": translator.token_usage,
        "lines": {r["id"]: out.get(r["id"]) for r in rows},
        "events": events,
    }
    (HERE / "out").mkdir(exist_ok=True)
    target = HERE / "out" / f"{set_path.stem}__{system}.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    missing = sum(1 for v in result["lines"].values() if v is None)
    print(f"{system}: {len(rows) - missing}/{len(rows)} translated, {elapsed:.0f}s, {request_count} requests -> {target}")


if __name__ == "__main__":
    main()
