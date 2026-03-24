from __future__ import annotations

import queue
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal

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


class PauseRequested(Exception):
    pass


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


@dataclass(slots=True)
class SubtitleTranslationJob:
    source_paths: list[str]
    source_language: str
    translator_settings: TranslatorSettings
    api_keys: list[str]
    deepl_api_key: str


@dataclass(slots=True)
class SaveTask:
    source_path: str
    output_path: Path
    content: str


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
        self.processing_state_changed.emit(STATUS_PAUSED, "현재 파일 마무리 후 일시 중지합니다.")
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

            if not self.job.api_keys:
                raise RuntimeError("번역용 Gemini API 키가 비어 있습니다. 번역 설정 탭에서 입력하세요.")

            gemini_translator = GeminiTranslator(
                TranslationConfig(
                    keys=self.job.api_keys,
                    preferred_model=self.job.translator_settings.preferred_model,
                    target_language=self.job.translator_settings.target_language,
                    system_prompt=self.job.translator_settings.system_prompt,
                    translation_note=self.job.translator_settings.translation_note,
                    temperature=self.job.translator_settings.temperature,
                    top_p=self.job.translator_settings.top_p,
                    reasoning_level=self.job.translator_settings.reasoning_level,
                    chunk_size=self.job.translator_settings.chunk_size,
                    request_delay_seconds=self.job.translator_settings.request_delay_seconds,
                ),
                self.log_message.emit,
            )

            deepl_translator = None
            if self.job.translator_settings.use_deepl_fallback and self.job.deepl_api_key:
                deepl_translator = DeepLTranslator(
                    DeepLConfig(
                        api_key=self.job.deepl_api_key,
                        target_language=self.job.translator_settings.target_language,
                        source_language=self.job.source_language,
                        chunk_size=self.job.translator_settings.chunk_size,
                        request_delay_seconds=self.job.translator_settings.request_delay_seconds,
                    ),
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
                jp_output_path = source_path.with_suffix(".jp.srt")
                self.file_started.emit(index, total, str(source_path))
                self.stage_progress_changed.emit(0, 0.0, 0.0)
                progress_key_display = gemini_translator.current_key_display
                progress_error_count = gemini_translator.error_count
                self.translation_progress_changed.emit(0, 0, progress_key_display, "", progress_error_count)

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
                    if jp_output_path.exists():
                        with self._counter_lock:
                            self._skipped_count += 1
                        message = "기존 일본어 자막 파일 존재 -> 스킵"
                        self.item_status_changed.emit(source_str, STATUS_SKIPPED, message)
                        self.log_message.emit(f"{source_path} | {message}")
                        self.stage_changed.emit("스킵", "기존 일본어 자막 파일 존재")
                        self.stage_progress_changed.emit(100, 0.0, 0.0)
                        continue

                    self.item_status_changed.emit(source_str, STATUS_TRANSCRIBING, "")
                    self.stage_changed.emit("자막 생성", "음성 인식")

                    language_code = None if self.job.source_language == "auto" else self.job.source_language
                    try:
                        segments = engine.transcribe_file(
                            source_path,
                            self._make_stage_progress_callback(),
                            language_code=language_code,
                            vad_settings=self.job.vad_settings,
                        )
                    except Exception as exc:
                        if runtime_config.device == "cuda" and is_cuda_runtime_error(str(exc)):
                            self.log_message.emit("GPU 초기화에 실패하여 CPU로 자동 전환합니다.")
                            runtime_config = build_runtime_config(
                                "cpu",
                                RuntimeTuningOptions(
                                    compute_type="auto",
                                    cpu_threads=self.job.runtime_tuning.cpu_threads,
                                    num_workers=self.job.runtime_tuning.num_workers,
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
                    if not subtitle_text:
                        raise RuntimeError("자막 생성 결과가 비어 있습니다.")

                    self.item_status_changed.emit(source_str, STATUS_TRANSLATING, "")
                    self.stage_changed.emit("번역", "자막 번역")
                    document = load_subtitle_document_from_text(".srt", subtitle_text)
                    records = document.get_translatable_records()
                    translated = None
                    gemini_error: str | None = None
                    output_path = source_path.with_suffix(".srt")

                    try:
                        translated = gemini_translator.translate_lines(
                            records,
                            self._make_translation_progress_callback(),
                            source_path.name,
                        )
                    except TranslationError as exc:
                        gemini_error = str(exc) or exc.__class__.__name__
                        self.log_message.emit(
                            f"{source_path.name} | Gemini 번역 후보를 모두 시도했지만 실패했습니다 -> {gemini_error}"
                        )

                    if translated is None:
                        if self.job.translator_settings.use_deepl_fallback:
                            if deepl_translator is None:
                                raise RuntimeError("DeepL 폴백이 활성화되었지만 API 키가 비어 있습니다.")
                            self.stage_changed.emit("번역", "DeepL Free API 폴백")
                            self.log_message.emit(f"{source_path.name} | DeepL Free API로 폴백합니다.")
                            translated = deepl_translator.translate_lines(
                                records,
                                self._make_translation_progress_callback(),
                                source_path.name,
                            )
                            self._pause_checkpoint("현재 파일 일시 중지됨")
                            document.apply_translations(translated)
                        else:
                            output_path = jp_output_path
                            self.stage_changed.emit("저장", "일본어 자막 저장")
                            self.log_message.emit(
                                f"{source_path.name} | Gemini 실패로 번역 없이 일본어 자막을 저장합니다 -> {output_path}"
                            )
                    else:
                        self._pause_checkpoint("현재 파일 일시 중지됨")
                        document.apply_translations(translated)

                    self.item_status_changed.emit(source_str, STATUS_SAVING, str(output_path))
                    self._enqueue_save(
                        SaveTask(
                            source_path=source_str,
                            output_path=output_path,
                            content=(document.render() if translated is not None else subtitle_text) + "\n",
                        )
                    )
                    self.log_message.emit(f"{source_path} | 저장 큐 등록 -> {output_path}")
                    self.stage_changed.emit("저장", "자막 저장 큐 등록")
                    self.stage_progress_changed.emit(100, 0.0, 0.0)
                except PauseRequested:
                    self.item_status_changed.emit(source_str, STATUS_PENDING, "일시 중지됨")
                    self._requeue_current(source_str)
                    self.processing_state_changed.emit(STATUS_PAUSED, "현재 파일 일시 중지됨")
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
            self.summary_ready.emit(self._success_count, self._failure_count, self._skipped_count)
        except Exception as exc:
            self._wait_for_save_completion()
            if self.job.runtime_tuning.auto_unload_after_job:
                unload_loaded_models(self.job.model_key)
            self.failed.emit(str(exc) or exc.__class__.__name__)


def build_translated_subtitle_output_path(source_path: Path, language_code: str) -> Path:
    normalized = language_code.strip().lower() or "translated"
    return source_path.with_name(f"{source_path.stem}.{normalized}{source_path.suffix}")


def build_japanese_subtitle_output_path(source_path: Path) -> Path:
    return source_path.with_name(f"{source_path.stem}.jp{source_path.suffix}")


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
            if not self.job.api_keys:
                raise RuntimeError("번역용 Gemini API 키가 비어 있습니다. 번역 설정 탭에서 입력하세요.")

            gemini_translator = GeminiTranslator(
                TranslationConfig(
                    keys=self.job.api_keys,
                    preferred_model=self.job.translator_settings.preferred_model,
                    target_language=self.job.translator_settings.target_language,
                    system_prompt=self.job.translator_settings.system_prompt,
                    translation_note=self.job.translator_settings.translation_note,
                    temperature=self.job.translator_settings.temperature,
                    top_p=self.job.translator_settings.top_p,
                    reasoning_level=self.job.translator_settings.reasoning_level,
                    chunk_size=self.job.translator_settings.chunk_size,
                    request_delay_seconds=self.job.translator_settings.request_delay_seconds,
                ),
                self.log_message.emit,
            )

            deepl_translator = None
            if self.job.translator_settings.use_deepl_fallback and self.job.deepl_api_key:
                deepl_translator = DeepLTranslator(
                    DeepLConfig(
                        api_key=self.job.deepl_api_key,
                        target_language=self.job.translator_settings.target_language,
                        source_language=self.job.source_language,
                        chunk_size=self.job.translator_settings.chunk_size,
                        request_delay_seconds=self.job.translator_settings.request_delay_seconds,
                    ),
                    self.log_message.emit,
                )

            while self._pending_paths:
                source_str = self._pending_paths.popleft()
                self._current_path = source_str
                source_path = Path(source_str)
                output_path = build_translated_subtitle_output_path(source_path, self.job.translator_settings.target_language)
                jp_output_path = build_japanese_subtitle_output_path(source_path)
                total = self._processed + len(self._pending_paths) + 1
                index = self._processed + 1
                self.file_started.emit(index, total, source_str)
                self.translation_progress_changed.emit(0, 0, gemini_translator.current_key_display, "", gemini_translator.error_count)

                try:
                    if not source_path.exists():
                        raise FileNotFoundError("입력 자막 파일을 찾을 수 없습니다.")
                    if output_path.exists():
                        self._skipped_count += 1
                        message = "기존 번역 자막 파일 존재 -> 스킵"
                        self.item_status_changed.emit(source_str, STATUS_SKIPPED, message)
                        self.log_message.emit(f"{source_path} | {message}")
                        self.stage_changed.emit("스킵", "기존 번역 자막 파일 존재")
                        continue
                    if jp_output_path.exists():
                        self._skipped_count += 1
                        message = "기존 일본어 자막 파일 존재 -> 스킵"
                        self.item_status_changed.emit(source_str, STATUS_SKIPPED, message)
                        self.log_message.emit(f"{source_path} | {message}")
                        self.stage_changed.emit("스킵", "기존 일본어 자막 파일 존재")
                        continue

                    self.item_status_changed.emit(source_str, STATUS_TRANSLATING, "")
                    self.stage_changed.emit("번역", "자막 번역")
                    document = load_subtitle_document(source_path)
                    records = document.get_translatable_records()
                    translated = None

                    try:
                        translated = gemini_translator.translate_lines(
                            records,
                            self.translation_progress_changed.emit,
                            source_path.name,
                        )
                    except TranslationError as exc:
                        self.log_message.emit(
                            f"{source_path.name} | Gemini 번역 후보를 모두 시도했지만 실패했습니다 -> {str(exc) or exc.__class__.__name__}"
                        )

                    if translated is None:
                        if self.job.translator_settings.use_deepl_fallback:
                            if deepl_translator is None:
                                raise RuntimeError("DeepL 폴백이 활성화되었지만 API 키가 비어 있습니다.")
                            self.stage_changed.emit("번역", "DeepL Free API 폴백")
                            self.log_message.emit(f"{source_path.name} | DeepL Free API로 폴백합니다.")
                            translated = deepl_translator.translate_lines(
                                records,
                                self.translation_progress_changed.emit,
                                source_path.name,
                            )
                            document.apply_translations(translated)
                        else:
                            output_path = jp_output_path
                            self.stage_changed.emit("저장", "일본어 자막 저장")
                            self.log_message.emit(
                                f"{source_path.name} | Gemini 실패로 번역 없이 일본어 자막을 저장합니다 -> {output_path}"
                            )
                    else:
                        document.apply_translations(translated)

                    self.item_status_changed.emit(source_str, STATUS_SAVING, str(output_path))
                    write_srt_text(output_path, document.render() + "\n")
                    self._success_count += 1
                    self.item_status_changed.emit(source_str, STATUS_DONE, str(output_path))
                    self.log_message.emit(f"{source_path} | 완료 -> {output_path}")
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

            self.summary_ready.emit(self._success_count, self._failure_count, self._skipped_count)
        except Exception as exc:
            self.failed.emit(str(exc) or exc.__class__.__name__)

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
                with self._counter_lock:
                    self._success_count += 1
                self.item_status_changed.emit(task.source_path, STATUS_DONE, str(task.output_path))
                self.log_message.emit(f"{task.source_path} | 완료 -> {task.output_path}")
            except Exception as exc:
                with self._counter_lock:
                    self._failure_count += 1
                message = str(exc) or exc.__class__.__name__
                self.item_status_changed.emit(task.source_path, STATUS_FAILED, message)
                self.log_message.emit(f"{task.source_path} | 저장 실패 -> {message}")
            finally:
                self._save_queue.task_done()
