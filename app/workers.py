from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .config import INPUT_LANGUAGE_OPTIONS
from .audio_splitter import (
    AudioSplitCancelled,
    build_cut_points,
    find_quiet_cut_points,
    format_timestamp,
    measure_audio_duration,
    probe_audio,
    split_audio,
)
from .cuda_runtime import ensure_cuda_runtime
from .deepl_translator import DeepLConfig, DeepLTranslator
from .dictionary_pack import ensure_dictionary_pack
from .file_queue import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PAUSED,
    STATUS_PENDING,
    STATUS_REMOVED,
    STATUS_SAVING,
    STATUS_SKIPPED,
    STATUS_TRANSCRIBING,
    STATUS_TRANSLATING,
)
from .gemini_translator import GeminiTranslator, TranslationConfig, TranslationError
from .japanese_postprocess import PostprocessOptions, postprocess_japanese_segments
from .model_manager import download_model, get_download_plan, get_model_dir, is_model_ready
from .srt_writer import build_srt_text, write_srt_text
from .subtitle_document import load_subtitle_document, load_subtitle_document_from_text
from .transcriber import (
    RuntimeTuningOptions,
    TranscriptionEngine,
    VADSettings,
    build_runtime_config,
    is_cuda_runtime_error,
    unload_loaded_models,
)
from .translator_store import TranslatorSettings


def format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


@dataclass(slots=True)
class PipelineJob:
    source_paths: list[str]
    runtime_device: str
    model_key: str
    source_language: str
    runtime_tuning: RuntimeTuningOptions
    vad_settings: VADSettings
    enable_postprocess: bool
    enable_enhanced_postprocess: bool
    translator_settings: TranslatorSettings
    api_keys: list[str]
    deepl_api_key: str
    enable_translation: bool = True


@dataclass(slots=True)
class SubtitleTranslationJob:
    source_paths: list[str]
    source_language: str
    translator_settings: TranslatorSettings
    api_keys: list[str]
    deepl_api_key: str


@dataclass(slots=True)
class AudioSplitJob:
    source_path: Path
    titles: list[str]
    end_times: list[float]
    silence_search_seconds: float
    output_dir: Path


@dataclass(slots=True)
class SaveTask:
    source_path: str
    output_path: Path
    content: str
    is_primary: bool = True
    note: str = ""


@dataclass(slots=True)
class TranslationOutcome:
    translations: dict[str, str]
    missing_count: int
    failed: bool


def create_translators(
    settings: TranslatorSettings,
    api_keys: list[str],
    deepl_api_key: str,
    source_language: str,
    log_callback,
) -> tuple[GeminiTranslator | None, DeepLTranslator | None]:
    gemini_translator = None
    if api_keys:
        gemini_translator = GeminiTranslator(
            TranslationConfig(
                keys=api_keys,
                preferred_model=settings.preferred_model,
                target_language=settings.target_language,
                system_prompt=settings.system_prompt,
                translation_note=settings.translation_note,
                temperature=settings.temperature,
                top_p=settings.top_p,
                reasoning_level=settings.reasoning_level,
                chunk_size=settings.chunk_size,
                request_delay_seconds=settings.request_delay_seconds,
            ),
            log_callback,
        )

    deepl_translator = None
    if settings.use_deepl_fallback and deepl_api_key:
        deepl_translator = DeepLTranslator(
            DeepLConfig(
                api_key=deepl_api_key,
                target_language=settings.target_language,
                source_language=source_language,
                chunk_size=settings.chunk_size,
                request_delay_seconds=settings.request_delay_seconds,
            ),
            log_callback,
        )

    if gemini_translator is None and deepl_translator is None:
        raise RuntimeError("번역용 Gemini API 키가 비어 있습니다. 번역 설정 탭에서 입력하세요.")
    return gemini_translator, deepl_translator


def translate_records(records, gemini_translator, deepl_translator, progress_callback, file_name: str, log_callback) -> TranslationOutcome:
    translations: dict[str, str] = {}
    gemini_aborted = False
    if gemini_translator is not None:
        try:
            translations = gemini_translator.translate_lines(records, progress_callback, file_name)
        except Exception as exc:
            translations = dict(getattr(exc, "partial", None) or {})
            gemini_aborted = True
            log_callback(f"{file_name} | Gemini 번역 중단 -> {str(exc) or exc.__class__.__name__}")

    missing = [record for record in records if record.line_id not in translations]
    if missing and deepl_translator is not None:
        log_callback(f"{file_name} | 번역되지 않은 {len(missing)}줄을 DeepL Free API로 번역합니다.")
        try:
            translations.update(deepl_translator.translate_lines(missing, progress_callback, file_name))
        except Exception as exc:
            translations.update(getattr(exc, "partial", None) or {})
            log_callback(f"{file_name} | DeepL 번역 실패 -> {str(exc) or exc.__class__.__name__}")
        missing = [record for record in records if record.line_id not in translations]

    failed = bool(missing) and (gemini_aborted or gemini_translator is None or not translations)
    return TranslationOutcome(translations=translations, missing_count=len(missing), failed=failed)


class ModelDownloadWorker(QThread):
    progress_changed = Signal(int)
    status_changed = Signal(str)
    log_message = Signal(str)
    finished_success = Signal(str)
    failed = Signal(str)

    def __init__(self, model_key: str, parent=None) -> None:
        super().__init__(parent)
        self.model_key = model_key

    def run(self) -> None:
        try:
            if is_model_ready(model_key=self.model_key):
                self.progress_changed.emit(100)
                self.status_changed.emit("모델 캐시 준비 완료")
                self.finished_success.emit(str(get_model_dir(self.model_key)))
                return

            plan = get_download_plan(self.model_key)
            if plan.total_bytes > 0:
                self.log_message.emit(f"모델 다운로드 필요: {format_bytes(plan.total_bytes)}")
            else:
                self.log_message.emit("모델 캐시 상태를 확인하는 중입니다.")

            def on_progress(downloaded_bytes: int, total_bytes: int, filename: str, file_downloaded: int, file_total: int) -> None:
                percent = 100 if total_bytes <= 0 else min(100, int(downloaded_bytes * 100 / total_bytes))
                self.progress_changed.emit(percent)
                if filename:
                    self.status_changed.emit(
                        f"모델 다운로드 중: {filename} ({format_bytes(file_downloaded)}/{format_bytes(file_total)})"
                    )

            def on_status(message: str) -> None:
                self.status_changed.emit(message)
                self.log_message.emit(message)

            model_dir = download_model(self.model_key, on_progress, on_status)
            self.progress_changed.emit(100)
            self.finished_success.emit(str(model_dir))
        except Exception as exc:
            self.failed.emit(str(exc) or exc.__class__.__name__)


class PipelineWorker(QThread):
    item_status_changed = Signal(str, str, str)
    log_message = Signal(str)
    queue_progress_changed = Signal(int, int)
    file_started = Signal(int, int, str)
    stage_changed = Signal(str, str)
    stage_progress_changed = Signal(int, float, float)
    translation_progress_changed = Signal(int, int, str, str, int)
    summary_ready = Signal(int, int, int)
    failed = Signal(str)
    processing_state_changed = Signal(str, str)

    def __init__(self, job: PipelineJob, parent=None) -> None:
        super().__init__(parent)
        self.job = job
        self._condition = threading.Condition()
        self._pending_paths = deque(job.source_paths)
        self._queued_paths = set(job.source_paths)
        self._current_path: str | None = None
        self._pause_requested = False
        self._stop_requested = False
        self._processed = 0
        self._success_count = 0
        self._failure_count = 0
        self._skipped_count = 0
        self._counter_lock = threading.Lock()
        self._save_queue: queue.Queue[SaveTask | None] = queue.Queue()
        self._save_thread: threading.Thread | None = None

    @property
    def current_path(self) -> str | None:
        with self._condition:
            return self._current_path

    def add_paths(self, paths: list[str]) -> int:
        added = 0
        with self._condition:
            for path in paths:
                if path in self._queued_paths or path == self._current_path:
                    continue
                self._pending_paths.append(path)
                self._queued_paths.add(path)
                added += 1
            total = self._processed + len(self._pending_paths) + (1 if self._current_path else 0)
            self._condition.notify_all()
        if added:
            self.queue_progress_changed.emit(self._processed, total)
        return added

    def remove_paths(self, paths: list[str]) -> tuple[list[str], list[str]]:
        removed: list[str] = []
        blocked: list[str] = []
        with self._condition:
            pending_list = list(self._pending_paths)
            keep: list[str] = []
            for path in pending_list:
                if path in paths:
                    removed.append(path)
                    self._queued_paths.discard(path)
                else:
                    keep.append(path)
            for path in paths:
                if path == self._current_path:
                    blocked.append(path)
            self._pending_paths = deque(keep)
            total = self._processed + len(self._pending_paths) + (1 if self._current_path else 0)
            self._condition.notify_all()
        for path in removed:
            self.item_status_changed.emit(path, STATUS_REMOVED, "대기열에서 제거됨")
        if removed:
            self.queue_progress_changed.emit(self._processed, total)
        return removed, blocked

    def pause_processing(self) -> bool:
        with self._condition:
            if self._pause_requested:
                return False
            self._pause_requested = True
            self._condition.notify_all()
        self.processing_state_changed.emit(STATUS_PAUSED, "진행 중인 단계에서 일시 중지합니다.")
        return True

    def resume_processing(self) -> bool:
        with self._condition:
            if not self._pause_requested:
                return False
            self._pause_requested = False
            self._condition.notify_all()
        self.processing_state_changed.emit(STATUS_PENDING, "남은 대기열 처리를 재개합니다.")
        return True

    def run(self) -> None:
        try:
            self._start_save_worker()
            if self.job.runtime_device == "cuda":
                self.stage_changed.emit("환경 준비", "CUDA runtime 확인")
                ensure_cuda_runtime(self._report_runtime_progress, self.log_message.emit)

            if self.job.enable_enhanced_postprocess:
                self.stage_changed.emit("환경 준비", "강화 후처리 사전 확인")
                ensure_dictionary_pack(self._report_dictionary_progress, self.log_message.emit)

            gemini_translator = None
            deepl_translator = None
            if self.job.enable_translation:
                gemini_translator, deepl_translator = create_translators(
                    self.job.translator_settings,
                    self.job.api_keys,
                    self.job.deepl_api_key,
                    self.job.source_language,
                    self.log_message.emit,
                )

            runtime_config = build_runtime_config(self.job.runtime_device, self.job.runtime_tuning)
            engine = TranscriptionEngine(runtime_config, self.job.model_key)
            self.log_message.emit(f"실행 장치: {runtime_config.label}")

            postprocess_options = PostprocessOptions(
                enabled=self.job.enable_postprocess,
                enhanced=self.job.enable_enhanced_postprocess,
                sentence=True,
                standard_asia=True,
                max_comma=2,
                max_gap=0.35,
                one_word=True,
            )

            while True:
                source_str, index, total = self._next_source()
                if source_str is None:
                    break

                source_path = Path(source_str)
                output_path = source_path.with_suffix(".srt")
                self.file_started.emit(index, total, str(source_path))
                self.stage_progress_changed.emit(0, 0.0, 0.0)
                if gemini_translator is not None:
                    self.translation_progress_changed.emit(
                        0, 0, gemini_translator.current_key_display, "", gemini_translator.error_count
                    )

                try:
                    if not source_path.exists():
                        raise FileNotFoundError("입력 파일을 찾을 수 없습니다.")

                    if output_path.exists():
                        with self._counter_lock:
                            self._skipped_count += 1
                        message = "기존 자막 파일 존재 -> 스킵"
                        self.item_status_changed.emit(source_str, STATUS_SKIPPED, message)
                        self.log_message.emit(f"{source_path} | {message}")
                        self.stage_changed.emit("스킵", "기존 자막 파일 존재")
                        self.stage_progress_changed.emit(100, 0.0, 0.0)
                        continue

                    language_code = None if self.job.source_language == "auto" else self.job.source_language
                    existing_original = self._find_existing_original(output_path)
                    if existing_original is not None:
                        original_output_path, source_language = existing_original
                        subtitle_text = original_output_path.read_text(encoding="utf-8-sig")
                        self.log_message.emit(f"{source_path} | 기존 원문 자막 재사용 (음성 인식 생략) -> {original_output_path}")
                    else:
                        self.item_status_changed.emit(source_str, STATUS_TRANSCRIBING, "")
                        self.stage_changed.emit("자막 생성", "음성 인식")
                        try:
                            segments = engine.transcribe_file(
                                source_path,
                                self._make_stage_progress_callback(),
                                language_code=language_code,
                                vad_settings=self.job.vad_settings,
                            )
                        except Exception as exc:
                            if runtime_config.device == "cuda" and is_cuda_runtime_error(str(exc)):
                                if "out of memory" in str(exc).lower():
                                    self.log_message.emit("GPU 메모리가 부족하여 CPU로 자동 전환합니다.")
                                else:
                                    self.log_message.emit("GPU 초기화에 실패하여 CPU로 자동 전환합니다.")
                                runtime_config = build_runtime_config(
                                    "cpu",
                                    RuntimeTuningOptions(
                                        compute_type="auto",
                                        cpu_threads=self.job.runtime_tuning.cpu_threads,
                                        num_workers=self.job.runtime_tuning.num_workers,
                                        memory_profile=self.job.runtime_tuning.memory_profile,
                                        auto_unload_after_job=self.job.runtime_tuning.auto_unload_after_job,
                                    ),
                                )
                                engine = TranscriptionEngine(runtime_config, self.job.model_key)
                                self.log_message.emit(f"자동 전환된 실행 장치: {runtime_config.label}")
                                segments = engine.transcribe_file(
                                    source_path,
                                    self._make_stage_progress_callback(),
                                    language_code=language_code,
                                    vad_settings=self.job.vad_settings,
                                )
                            else:
                                raise

                        self._pause_checkpoint("현재 파일 일시 중지됨")

                        if self.job.enable_postprocess:
                            self.stage_changed.emit("자막 생성", "후처리")
                            segments = postprocess_japanese_segments(segments, postprocess_options)

                        subtitle_text = build_srt_text(segments)
                        source_language = engine.last_detected_language or language_code or "source"
                        original_output_path = build_translated_subtitle_output_path(output_path, source_language)

                    if not subtitle_text.strip():
                        raise RuntimeError("자막 생성 결과가 비어 있습니다.")

                    if not self.job.enable_translation:
                        self.item_status_changed.emit(source_str, STATUS_SAVING, str(output_path))
                        self._enqueue_save(SaveTask(source_path=source_str, output_path=output_path, content=subtitle_text))
                        self.log_message.emit(f"{source_path} | 번역 없이 저장 큐 등록 -> {output_path}")
                        self.stage_changed.emit("저장", "자막 저장 큐 등록")
                        self.stage_progress_changed.emit(100, 0.0, 0.0)
                        continue

                    if existing_original is None and not original_output_path.exists():
                        # 번역 도중 예기치 못한 오류가 나도 음성 인식 결과는 남도록 먼저 저장한다.
                        self._enqueue_save(
                            SaveTask(
                                source_path=source_str,
                                output_path=original_output_path,
                                content=subtitle_text,
                                is_primary=False,
                            )
                        )

                    self.item_status_changed.emit(source_str, STATUS_TRANSLATING, "")
                    self.stage_changed.emit("번역", "자막 번역")
                    document = load_subtitle_document_from_text(".srt", subtitle_text)
                    records = document.get_translatable_records()
                    outcome = translate_records(
                        records,
                        gemini_translator,
                        deepl_translator,
                        self._make_translation_progress_callback(),
                        source_path.name,
                        self.log_message.emit,
                    )
                    self._pause_checkpoint("현재 파일 일시 중지됨")

                    if outcome.failed:
                        raise TranslationError(
                            f"번역 실패 -> 원문 자막만 저장: {original_output_path} "
                            "(다시 시작하면 음성 인식 없이 번역만 다시 시도합니다)"
                        )

                    document.apply_translations(outcome.translations)
                    note = ""
                    if outcome.missing_count:
                        note = f"{outcome.missing_count}줄 번역 실패 (원문 유지)"
                        self.log_message.emit(f"{source_path} | {note}")
                    self.item_status_changed.emit(source_str, STATUS_SAVING, str(output_path))
                    self._enqueue_save(
                        SaveTask(
                            source_path=source_str,
                            output_path=output_path,
                            content=document.render() + "\n",
                            note=note,
                        )
                    )
                    self.log_message.emit(f"{source_path} | 저장 큐 등록 -> {output_path}")
                    self.stage_changed.emit("저장", "자막 저장 큐 등록")
                    self.stage_progress_changed.emit(100, 0.0, 0.0)
                except Exception as exc:
                    with self._counter_lock:
                        self._failure_count += 1
                    message = str(exc) or exc.__class__.__name__
                    self.item_status_changed.emit(source_str, STATUS_FAILED, message)
                    self.log_message.emit(f"{source_path} | 실패 -> {message}")
                    self.stage_changed.emit("실패", message)
                finally:
                    if self.job.runtime_tuning.memory_profile == "strict":
                        unloaded = unload_loaded_models(self.job.model_key)
                        if unloaded > 0:
                            self.log_message.emit("강한 절약 모드로 현재 파일 처리 후 모델을 메모리에서 해제했습니다.")
                    self._complete_current()

            self._wait_for_save_completion()
            if self.job.runtime_tuning.auto_unload_after_job:
                unloaded = unload_loaded_models(self.job.model_key)
                if unloaded > 0:
                    self.log_message.emit(f"작업 완료 후 모델 {unloaded}개를 메모리에서 자동 해제했습니다.")
            if gemini_translator is not None:
                self.log_message.emit(gemini_translator.token_usage_summary)
            self.summary_ready.emit(self._success_count, self._failure_count, self._skipped_count)
        except Exception as exc:
            self._wait_for_save_completion()
            if self.job.runtime_tuning.auto_unload_after_job:
                unload_loaded_models(self.job.model_key)
            self.failed.emit(str(exc) or exc.__class__.__name__)

    def _find_existing_original(self, output_path: Path) -> tuple[Path, str] | None:
        if self.job.source_language != "auto":
            languages = [self.job.source_language]
        else:
            # 자동 감지면 번역 대상 언어를 뺀 입력 언어 후보를 모두 확인한다(번역 결과 파일과 혼동 방지).
            target = self.job.translator_settings.target_language
            languages = [code for code, _label in INPUT_LANGUAGE_OPTIONS if code not in {"auto", target}]
        candidates = [(build_translated_subtitle_output_path(output_path, code), code) for code in languages]
        candidates.append((build_translated_subtitle_output_path(output_path, "jp"), "ja"))
        for path, language in candidates:
            if path.is_file():
                return path, language
        return None

    def _start_save_worker(self) -> None:
        if self._save_thread is not None and self._save_thread.is_alive():
            return
        self._save_thread = threading.Thread(target=self._save_worker_main, name="SubtitleSaveWorker", daemon=True)
        self._save_thread.start()

    def _enqueue_save(self, task: SaveTask) -> None:
        self._save_queue.put(task)

    def _wait_for_save_completion(self) -> None:
        if self._save_thread is None:
            return
        self._save_queue.join()
        self._save_queue.put(None)
        self._save_queue.join()
        self._save_thread.join()
        self._save_thread = None

    def _save_worker_main(self) -> None:
        while True:
            task = self._save_queue.get()
            try:
                if task is None:
                    return
                write_srt_text(task.output_path, task.content)
                if not task.is_primary:
                    self.log_message.emit(f"{task.source_path} | 원문 자막 저장 -> {task.output_path}")
                    continue
                with self._counter_lock:
                    self._success_count += 1
                done_message = f"{task.output_path} | {task.note}" if task.note else str(task.output_path)
                self.item_status_changed.emit(task.source_path, STATUS_DONE, done_message)
                self.log_message.emit(f"{task.source_path} | 완료 -> {done_message}")
            except Exception as exc:
                message = str(exc) or exc.__class__.__name__
                if not task.is_primary:
                    self.log_message.emit(f"{task.source_path} | 원문 자막 저장 실패 -> {message}")
                    continue
                with self._counter_lock:
                    self._failure_count += 1
                self.item_status_changed.emit(task.source_path, STATUS_FAILED, message)
                self.log_message.emit(f"{task.source_path} | 저장 실패 -> {message}")
            finally:
                self._save_queue.task_done()

    def _next_source(self) -> tuple[str | None, int, int]:
        with self._condition:
            while True:
                while self._pause_requested:
                    self.processing_state_changed.emit(STATUS_PAUSED, "일시 중지됨")
                    self._condition.wait()

                if self._stop_requested:
                    return None, self._processed, self._processed

                if self._pending_paths:
                    source_str = self._pending_paths.popleft()
                    self._current_path = source_str
                    total = self._processed + len(self._pending_paths) + 1
                    index = self._processed + 1
                    self.processing_state_changed.emit(STATUS_TRANSCRIBING, "처리 중")
                    return source_str, index, total

                return None, self._processed, self._processed

    def _complete_current(self) -> None:
        with self._condition:
            if self._current_path is not None:
                self._queued_paths.discard(self._current_path)
                self._current_path = None
                self._processed += 1
            total = self._processed + len(self._pending_paths)
        self.queue_progress_changed.emit(self._processed, total)

    def _requeue_current(self, source_str: str) -> None:
        with self._condition:
            self._pending_paths.appendleft(source_str)

    def _pause_checkpoint(self, detail: str) -> None:
        with self._condition:
            while self._pause_requested:
                self.processing_state_changed.emit(STATUS_PAUSED, detail)
                self._condition.wait()

    def _make_stage_progress_callback(self):
        def callback(percent: int, current: float, total: float) -> None:
            self._pause_checkpoint("현재 파일 일시 중지됨")
            self.stage_progress_changed.emit(percent, current, total)

        return callback

    def _make_translation_progress_callback(self):
        def callback(chunk_index: int, chunk_total: int, key_display: str, model_name: str, error_count: int) -> None:
            self._pause_checkpoint("현재 파일 일시 중지됨")
            self.translation_progress_changed.emit(chunk_index, chunk_total, key_display, model_name, error_count)

        return callback

    def _report_runtime_progress(
        self,
        downloaded_bytes: int,
        total_bytes: int,
        filename: str,
        file_downloaded: int,
        file_total: int,
    ) -> None:
        self._emit_prep_progress("CUDA runtime", downloaded_bytes, total_bytes, filename, file_downloaded, file_total)

    def _report_dictionary_progress(
        self,
        downloaded_bytes: int,
        total_bytes: int,
        filename: str,
        file_downloaded: int,
        file_total: int,
    ) -> None:
        self._emit_prep_progress("Japanese dictionary pack", downloaded_bytes, total_bytes, filename, file_downloaded, file_total)

    def _emit_prep_progress(
        self,
        label: str,
        downloaded_bytes: int,
        total_bytes: int,
        filename: str,
        file_downloaded: int,
        file_total: int,
    ) -> None:
        if filename:
            if total_bytes > 0:
                percent = min(100, int(downloaded_bytes * 100 / total_bytes))
            elif file_total > 0:
                percent = min(100, int(file_downloaded * 100 / file_total))
            else:
                percent = 100
            self.stage_progress_changed.emit(percent, 0.0, 0.0)
            self.log_message.emit(
                f"{label} download {percent}% | {filename} ({format_bytes(file_downloaded)}/{format_bytes(file_total)})"
            )


def build_translated_subtitle_output_path(source_path: Path, language_code: str) -> Path:
    normalized = language_code.strip().lower() or "translated"
    return source_path.with_name(f"{source_path.stem}.{normalized}{source_path.suffix}")


class SubtitleTranslationWorker(QThread):
    item_status_changed = Signal(str, str, str)
    log_message = Signal(str)
    queue_progress_changed = Signal(int, int)
    file_started = Signal(int, int, str)
    stage_changed = Signal(str, str)
    translation_progress_changed = Signal(int, int, str, str, int)
    summary_ready = Signal(int, int, int)
    failed = Signal(str)

    def __init__(self, job: SubtitleTranslationJob, parent=None) -> None:
        super().__init__(parent)
        self.job = job
        self._pending_paths = deque(job.source_paths)
        self._queued_paths = set(job.source_paths)
        self._current_path: str | None = None
        self._processed = 0
        self._success_count = 0
        self._failure_count = 0
        self._skipped_count = 0

    def add_paths(self, paths: list[str]) -> int:
        added = 0
        for path in paths:
            if path in self._queued_paths or path == self._current_path:
                continue
            self._pending_paths.append(path)
            self._queued_paths.add(path)
            added += 1
        if added:
            self.queue_progress_changed.emit(self._processed, self._processed + len(self._pending_paths) + (1 if self._current_path else 0))
        return added

    def remove_paths(self, paths: list[str]) -> tuple[list[str], list[str]]:
        removed: list[str] = []
        blocked: list[str] = []
        keep: list[str] = []
        for path in list(self._pending_paths):
            if path in paths:
                removed.append(path)
                self._queued_paths.discard(path)
            else:
                keep.append(path)
        for path in paths:
            if path == self._current_path:
                blocked.append(path)
        self._pending_paths = deque(keep)
        if removed:
            self.queue_progress_changed.emit(self._processed, self._processed + len(self._pending_paths) + (1 if self._current_path else 0))
        return removed, blocked

    def run(self) -> None:
        try:
            gemini_translator, deepl_translator = create_translators(
                self.job.translator_settings,
                self.job.api_keys,
                self.job.deepl_api_key,
                self.job.source_language,
                self.log_message.emit,
            )

            while self._pending_paths:
                source_str = self._pending_paths.popleft()
                self._current_path = source_str
                source_path = Path(source_str)
                output_path = build_translated_subtitle_output_path(source_path, self.job.translator_settings.target_language)
                total = self._processed + len(self._pending_paths) + 1
                index = self._processed + 1
                self.file_started.emit(index, total, source_str)
                if gemini_translator is not None:
                    self.translation_progress_changed.emit(
                        0, 0, gemini_translator.current_key_display, "", gemini_translator.error_count
                    )

                try:
                    if not source_path.exists():
                        raise FileNotFoundError("입력 자막 파일을 찾을 수 없습니다.")
                    target = self.job.translator_settings.target_language.strip().lower()
                    if target and source_path.stem.lower().endswith(f".{target}"):
                        self._skipped_count += 1
                        message = "이미 번역된 자막 파일로 보여 스킵 (파일 이름이 출력 언어 코드로 끝남)"
                        self.item_status_changed.emit(source_str, STATUS_SKIPPED, message)
                        self.log_message.emit(f"{source_path} | {message}")
                        self.stage_changed.emit("스킵", "번역 결과 파일")
                        continue
                    if output_path.exists():
                        self._skipped_count += 1
                        message = "기존 번역 자막 파일 존재 -> 스킵"
                        self.item_status_changed.emit(source_str, STATUS_SKIPPED, message)
                        self.log_message.emit(f"{source_path} | {message}")
                        self.stage_changed.emit("스킵", "기존 번역 자막 파일 존재")
                        continue

                    self.item_status_changed.emit(source_str, STATUS_TRANSLATING, "")
                    self.stage_changed.emit("번역", "자막 번역")
                    document = load_subtitle_document(source_path)
                    records = document.get_translatable_records()
                    outcome = translate_records(
                        records,
                        gemini_translator,
                        deepl_translator,
                        self.translation_progress_changed.emit,
                        source_path.name,
                        self.log_message.emit,
                    )
                    if outcome.failed:
                        raise TranslationError("번역 실패 (원본 자막은 그대로 둡니다)")

                    document.apply_translations(outcome.translations)
                    self.item_status_changed.emit(source_str, STATUS_SAVING, str(output_path))
                    write_srt_text(output_path, document.render() + "\n")
                    self._success_count += 1
                    done_message = str(output_path)
                    if outcome.missing_count:
                        done_message += f" | {outcome.missing_count}줄 번역 실패 (원문 유지)"
                    self.item_status_changed.emit(source_str, STATUS_DONE, done_message)
                    self.log_message.emit(f"{source_path} | 완료 -> {done_message}")
                    self.stage_changed.emit("저장", "자막 저장 완료")
                except Exception as exc:
                    self._failure_count += 1
                    message = str(exc) or exc.__class__.__name__
                    self.item_status_changed.emit(source_str, STATUS_FAILED, message)
                    self.log_message.emit(f"{source_path} | 실패 -> {message}")
                    self.stage_changed.emit("실패", message)
                finally:
                    if self._current_path is not None:
                        self._queued_paths.discard(self._current_path)
                        self._current_path = None
                        self._processed += 1
                    self.queue_progress_changed.emit(self._processed, self._processed + len(self._pending_paths))

            if gemini_translator is not None:
                self.log_message.emit(gemini_translator.token_usage_summary)
            self.summary_ready.emit(self._success_count, self._failure_count, self._skipped_count)
        except Exception as exc:
            self.failed.emit(str(exc) or exc.__class__.__name__)


class AudioSplitWorker(QThread):
    progress_changed = Signal(int)
    log_message = Signal(str)
    finished_success = Signal(list)
    failed = Signal(str)

    def __init__(self, job: AudioSplitJob, parent=None) -> None:
        super().__init__(parent)
        self.job = job

    def run(self) -> None:
        try:
            info = probe_audio(self.job.source_path)
            actual = measure_audio_duration(self.job.source_path, self.isInterruptionRequested)
            if actual > 0 and abs(actual - info.duration) > 0.5:
                self.log_message.emit(
                    f"파일에 기록된 길이({format_timestamp(info.duration, True)})와 실제 오디오 길이"
                    f"({format_timestamp(actual, True)})가 달라 실제 길이를 기준으로 나눕니다."
                )
            if actual > 0:
                info.duration = actual
            cut_points = build_cut_points(self.job.end_times, info.duration)
            search = self.job.silence_search_seconds
            if search > 0:
                self.log_message.emit(f"자르는 지점 앞뒤 {search:.1f}초 안에서 가장 조용한 곳을 찾는 중...")
                snapped = find_quiet_cut_points(
                    self.job.source_path,
                    cut_points,
                    info.duration,
                    search,
                    lambda ratio: self.progress_changed.emit(int(ratio * 30)),
                    self.isInterruptionRequested,
                )
                for index, (original, (point, before, after)) in enumerate(zip(cut_points, snapped), start=1):
                    shift = point - original
                    if abs(shift) < 0.005:
                        self.log_message.emit(f"{index}번 구간 끝 {format_timestamp(original, True)} | 보정 없음 (이미 가장 조용한 지점)")
                    else:
                        self.log_message.emit(
                            f"{index}번 구간 끝 {format_timestamp(original, True)} -> {format_timestamp(point, True)} "
                            f"({shift:+.2f}초, 음량 {before:.0f} dB -> {after:.0f} dB)"
                        )
                cut_points = build_cut_points([point for point, _, _ in snapped], info.duration)
            self.progress_changed.emit(30)
            results = split_audio(
                self.job.source_path,
                info,
                self.job.titles,
                cut_points,
                self.job.output_dir,
                lambda ratio: self.progress_changed.emit(30 + int(ratio * 70)),
                self.isInterruptionRequested,
            )
            for result in results:
                self.log_message.emit(
                    f"{result.path.name} | {format_timestamp(result.start, True)} ~ {format_timestamp(result.end, True)}"
                )
            self.progress_changed.emit(100)
            self.finished_success.emit([str(result.path) for result in results])
        except AudioSplitCancelled:
            self.log_message.emit("분할을 취소했습니다. 만들던 파일은 지웠습니다.")
        except Exception as exc:
            self.failed.emit(str(exc) or exc.__class__.__name__)
