"""번역이 모두 끝나면 중간 원문 자막(a.ja.srt)을 지우고, 아니면 남기는지 확인한다."""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication

import app.workers as workers
from app.file_queue import STATUS_DONE, STATUS_FAILED
from app.gemini_translator import TranslationUnavailableError
from app.srt_writer import SubtitleSegment
from app.transcriber import RuntimeTuningOptions, VADSettings
from app.translator_store import TranslatorSettings
from app.workers import (
    PipelineJob,
    PipelineWorker,
    SaveTask,
    TranslatorChain,
    build_translated_subtitle_output_path,
)

ORIGINAL_SRT = "1\n00:00:01,000 --> 00:00:02,000\nおはよう\n\n2\n00:00:03,000 --> 00:00:04,000\nこんばんは\n"


def make_job(paths: list[str], enable_translation: bool = True, source_language: str = "ja") -> PipelineJob:
    return PipelineJob(
        source_paths=paths,
        runtime_device="cpu",
        model_key="medium",
        source_language=source_language,
        runtime_tuning=RuntimeTuningOptions(auto_unload_after_job=False),
        vad_settings=VADSettings(),
        enable_postprocess=False,
        enable_enhanced_postprocess=False,
        translator_settings=TranslatorSettings(),
        api_keys=["dummy"],
        deepl_api_key="",
        enable_translation=enable_translation,
    )


class Recorder:
    def __init__(self, worker: PipelineWorker) -> None:
        self.statuses: list[tuple[str, str, str]] = []
        self.logs: list[str] = []
        self.failures: list[str] = []
        worker.failed.connect(self.failures.append)
        worker.item_status_changed.connect(lambda *args: self.statuses.append(args))
        worker.log_message.connect(self.logs.append)


# ----- build_translated_subtitle_output_path -----

def test_build_translated_subtitle_output_path(tmp_path):
    out = tmp_path / "a.srt"
    assert build_translated_subtitle_output_path(out, "ja") == tmp_path / "a.ja.srt"
    assert build_translated_subtitle_output_path(out, " KO ") == tmp_path / "a.ko.srt"
    assert build_translated_subtitle_output_path(out, "") == tmp_path / "a.translated.srt"


# ----- 저장 스레드(SaveTask 큐) 직접 사용 -----

def run_save_tasks(worker: PipelineWorker, tasks: list[SaveTask]) -> None:
    worker._start_save_worker()
    for task in tasks:
        worker._enqueue_save(task)
    worker._wait_for_save_completion()
    QCoreApplication.processEvents()  # 저장 스레드에서 보낸 신호 전달


def test_save_removes_intermediate_when_requested(qapp, tmp_path):
    original = tmp_path / "a.ja.srt"
    original.write_text(ORIGINAL_SRT, encoding="utf-8")
    worker = PipelineWorker(make_job([]))
    rec = Recorder(worker)

    run_save_tasks(
        worker,
        [SaveTask(source_path="a.mp3", output_path=tmp_path / "a.srt", content="1\nx\n", remove_after_save=original)],
    )

    assert (tmp_path / "a.srt").is_file()
    assert not original.exists()
    assert worker._success_count == 1
    assert rec.statuses[-1][1] == STATUS_DONE
    assert any("원문 자막 삭제" in line for line in rec.logs)


def test_save_keeps_intermediate_without_remove_flag(qapp, tmp_path):
    original = tmp_path / "a.ja.srt"
    original.write_text(ORIGINAL_SRT, encoding="utf-8")
    worker = PipelineWorker(make_job([]))
    run_save_tasks(
        worker,
        [SaveTask(source_path="a.mp3", output_path=tmp_path / "a.srt", content="1\nx\n", note="1줄 번역 실패 (원문 유지)")],
    )
    assert (tmp_path / "a.srt").is_file()
    assert original.is_file()


def test_failed_primary_save_does_not_remove_intermediate(qapp, tmp_path):
    original = tmp_path / "a.ja.srt"
    original.write_text(ORIGINAL_SRT, encoding="utf-8")
    worker = PipelineWorker(make_job([]))
    rec = Recorder(worker)
    # 빈 내용은 write_srt_text가 거절한다 -> 저장 실패
    run_save_tasks(
        worker,
        [SaveTask(source_path="a.mp3", output_path=tmp_path / "a.srt", content="  ", remove_after_save=original)],
    )
    assert original.is_file()
    assert worker._failure_count == 1
    assert rec.statuses[-1][1] == STATUS_FAILED


def test_non_primary_then_primary_order(qapp, tmp_path):
    """원문 저장(비주 작업)이 먼저 처리되고, 번역본 저장 뒤 그 원문이 지워진다."""
    original = tmp_path / "a.ja.srt"
    worker = PipelineWorker(make_job([]))
    run_save_tasks(
        worker,
        [
            SaveTask(source_path="a.mp3", output_path=original, content=ORIGINAL_SRT, is_primary=False),
            SaveTask(source_path="a.mp3", output_path=tmp_path / "a.srt", content="1\nx\n", remove_after_save=original),
        ],
    )
    assert (tmp_path / "a.srt").is_file()
    assert not original.exists()
    assert worker._success_count == 1  # 비주 작업은 성공 수에 들어가지 않는다


# ----- PipelineWorker._run 전체 흐름(스레드 없이) -----

class FakeTranslator:
    current_key_display = "1/1"
    error_count = 0
    token_usage_summary = "fake tokens"

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def translate_lines(self, recs, progress, _file_name):
        if self.mode == "all":
            return {record.line_id: f"KO:{record.text}" for record in recs}
        if self.mode == "partial":
            return {recs[0].line_id: f"KO:{recs[0].text}"}
        exc = TranslationUnavailableError("quota exhausted")
        exc.partial = {recs[0].line_id: "KO"}
        raise exc


class FakeEngine:
    detected_language = "ja"

    def __init__(self, *_args, **_kwargs) -> None:
        self.last_detected_language = self.detected_language
        self.calls = 0

    def transcribe_file(self, *_args, **_kwargs):
        self.calls += 1
        return [SubtitleSegment(1.0, 2.0, "おはよう"), SubtitleSegment(3.0, 4.0, "こんばんは")]


@pytest.fixture
def pipeline_env(monkeypatch):
    state = {"mode": "all", "engines": []}

    def fake_create_translators(*_args, **_kwargs):
        return TranslatorChain(primary=FakeTranslator(state["mode"]), secondary=None, deepl=None)

    def fake_engine(*args, **kwargs):
        engine = FakeEngine(*args, **kwargs)
        state["engines"].append(engine)
        return engine

    monkeypatch.setattr(workers, "create_translators", fake_create_translators)
    monkeypatch.setattr(workers, "build_runtime_config", lambda *_a, **_k: type("RC", (), {"label": "CPU", "device": "cpu"})())
    monkeypatch.setattr(workers, "TranscriptionEngine", fake_engine)
    monkeypatch.setattr(workers, "needs_whisperseg_model", lambda *_a: False)
    monkeypatch.setattr(workers, "ensure_cuda_runtime", lambda *_a: pytest.fail("CUDA를 건드리면 안 된다"))
    monkeypatch.setattr(workers, "download_model", lambda *_a: pytest.fail("다운로드하면 안 된다"))
    return state


def run_pipeline(tmp_path: Path, *, with_existing_original: bool) -> tuple[PipelineWorker, Recorder, Path, Path]:
    source = tmp_path / "a.mp3"
    source.write_bytes(b"fake audio")
    original = tmp_path / "a.ja.srt"
    if with_existing_original:
        original.write_text(ORIGINAL_SRT, encoding="utf-8")
    worker = PipelineWorker(make_job([str(source)]))
    rec = Recorder(worker)
    worker._remote_session = None  # run()이 정하는 값(로컬 실행)
    worker._run()
    QCoreApplication.processEvents()
    assert rec.failures == [], (rec.failures, rec.logs)
    return worker, rec, tmp_path / "a.srt", original


@pytest.mark.parametrize("with_existing_original", [True, False])
def test_pipeline_full_translation_removes_original(qapp, tmp_path, pipeline_env, with_existing_original):
    pipeline_env["mode"] = "all"
    worker, rec, output, original = run_pipeline(tmp_path, with_existing_original=with_existing_original)
    assert output.is_file()
    text = output.read_text(encoding="utf-8-sig")
    assert "KO:おはよう" in text and "KO:こんばんは" in text
    assert not original.exists()
    assert worker._success_count == 1 and worker._failure_count == 0
    engine_calls = sum(engine.calls for engine in pipeline_env["engines"])
    # 기존 원문 자막이 있으면 음성 인식을 건너뛴다.
    assert engine_calls == (0 if with_existing_original else 1)


@pytest.mark.parametrize("with_existing_original", [True, False])
def test_pipeline_partial_translation_keeps_original(qapp, tmp_path, pipeline_env, with_existing_original):
    pipeline_env["mode"] = "partial"
    worker, rec, output, original = run_pipeline(tmp_path, with_existing_original=with_existing_original)
    assert output.is_file()
    assert original.is_file()
    assert "おはよう" in original.read_text(encoding="utf-8-sig")
    assert any("원문 유지" in status[2] for status in rec.statuses if status[1] == STATUS_DONE)


def test_pipeline_failed_translation_keeps_only_original(qapp, tmp_path, pipeline_env):
    pipeline_env["mode"] = "abort"
    worker, rec, output, original = run_pipeline(tmp_path, with_existing_original=False)
    assert not output.exists()
    assert original.is_file()
    assert worker._failure_count == 1
    assert rec.statuses[-1][1] == STATUS_FAILED


def test_pipeline_does_not_delete_unrelated_existing_subtitle(qapp, tmp_path, pipeline_env, monkeypatch):
    monkeypatch.setattr(FakeEngine, "detected_language", "ko")
    pipeline_env["mode"] = "all"
    source = tmp_path / "a.mp3"
    source.write_bytes(b"fake audio")
    user_file = tmp_path / "a.ko.srt"
    user_file.write_text("1\n00:00:01,000 --> 00:00:02,000\n사용자가 만든 자막\n", encoding="utf-8")
    worker = PipelineWorker(make_job([str(source)], source_language="auto"))
    rec = Recorder(worker)
    worker._remote_session = None
    worker._run()
    QCoreApplication.processEvents()
    assert rec.failures == []
    assert (tmp_path / "a.srt").is_file()
    assert user_file.is_file()  # 이번 실행과 무관한 파일은 남아야 한다
