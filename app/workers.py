from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .cuda_runtime import ensure_cuda_runtime
from .dictionary_pack import ensure_dictionary_pack
from .file_queue import STATUS_DONE, STATUS_FAILED, STATUS_PENDING, STATUS_SKIPPED, STATUS_TRANSCRIBING, STATUS_TRANSLATING
from .gemini_translator import GeminiTranslator, TranslationConfig
from .japanese_postprocess import PostprocessOptions, postprocess_japanese_segments
from .model_manager import download_model, get_download_plan, get_model_dir, is_model_ready
from .srt_writer import build_srt_text, write_srt
from .subtitle_document import load_subtitle_document_from_text
from .transcriber import TranscriptionEngine, build_runtime_config, is_cuda_runtime_error
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
    enable_postprocess: bool
    enable_enhanced_postprocess: bool
    translator_settings: TranslatorSettings
    api_keys: list[str]


class ModelDownloadWorker(QThread):
    progress_changed = Signal(int)
    status_changed = Signal(str)
    log_message = Signal(str)
    finished_success = Signal(str)
    failed = Signal(str)

    def run(self) -> None:
        try:
            if is_model_ready():
                self.progress_changed.emit(100)
                self.status_changed.emit("모델 캐시 준비 완료")
                self.finished_success.emit(str(get_model_dir()))
                return

            plan = get_download_plan()
            if plan.total_bytes > 0:
                self.log_message.emit(f"모델 다운로드 필요: {format_bytes(plan.total_bytes)}")
            else:
                self.log_message.emit("모델 캐시를 확인하는 중입니다.")

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

            model_dir = download_model(on_progress, on_status)
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

    def __init__(self, job: PipelineJob, parent=None) -> None:
        super().__init__(parent)
        self.job = job

    def run(self) -> None:
        try:
            total = len(self.job.source_paths)
            success_count = 0
            failure_count = 0
            skipped_count = 0
            processed = 0

            if self.job.runtime_device == "cuda":
                self.stage_changed.emit("환경 준비", "CUDA 런타임 확인")
                ensure_cuda_runtime(self._report_runtime_progress, self.log_message.emit)

            if self.job.enable_enhanced_postprocess:
                self.stage_changed.emit("환경 준비", "강화 후처리 사전팩 확인")
                ensure_dictionary_pack(self._report_dictionary_progress, self.log_message.emit)

            if not self.job.api_keys:
                raise RuntimeError("번역용 Gemini API 키가 비어 있습니다. 번역 설정 탭에서 키를 입력하세요.")

            translator = GeminiTranslator(
                TranslationConfig(
                    keys=self.job.api_keys,
                    preferred_model=self.job.translator_settings.preferred_model,
                    system_prompt=self.job.translator_settings.system_prompt,
                    translation_note=self.job.translator_settings.translation_note,
                    temperature=self.job.translator_settings.temperature,
                    top_p=self.job.translator_settings.top_p,
                    reasoning_level=self.job.translator_settings.reasoning_level,
                    chunk_size=self.job.translator_settings.chunk_size,
                ),
                self.log_message.emit,
            )

            runtime_config = build_runtime_config(self.job.runtime_device)
            engine = TranscriptionEngine(runtime_config)
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

            for index, source_str in enumerate(self.job.source_paths, start=1):
                source_path = Path(source_str)
                output_path = source_path.with_suffix(".srt")
                self.file_started.emit(index, total, str(source_path))
                self.stage_progress_changed.emit(0, 0.0, 0.0)
                self.translation_progress_changed.emit(0, 0, translator.current_key_display, "", translator.error_count)

                try:
                    if not source_path.exists():
                        raise FileNotFoundError("입력 파일을 찾을 수 없습니다.")

                    if output_path.exists():
                        skipped_count += 1
                        message = "기존 자막 파일 존재 -> 스킵"
                        self.item_status_changed.emit(source_str, STATUS_SKIPPED, message)
                        self.log_message.emit(f"{source_path} | {message}")
                        self.stage_changed.emit("스킵", "기존 한국어 자막 존재")
                        self.stage_progress_changed.emit(100, 0.0, 0.0)
                        continue

                    self.item_status_changed.emit(source_str, STATUS_TRANSCRIBING, "")
                    self.stage_changed.emit("자막 생성", "일본어 음성 인식")

                    try:
                        segments = engine.transcribe_file(source_path, self.stage_progress_changed.emit)
                    except Exception as exc:
                        if runtime_config.device == "cuda" and is_cuda_runtime_error(str(exc)):
                            self.log_message.emit(
                                "GPU 초기화에 실패하여 CPU로 자동 전환합니다. CUDA 런타임 설치 상태를 확인하세요."
                            )
                            runtime_config = build_runtime_config("cpu")
                            engine = TranscriptionEngine(runtime_config)
                            self.log_message.emit(f"자동 전환된 실행 장치: {runtime_config.label}")
                            segments = engine.transcribe_file(source_path, self.stage_progress_changed.emit)
                        else:
                            raise

                    if self.job.enable_postprocess:
                        self.stage_changed.emit("자막 생성", "일본어 후처리")
                        segments = postprocess_japanese_segments(segments, postprocess_options)

                    japanese_srt_text = build_srt_text(segments)
                    if not japanese_srt_text:
                        raise RuntimeError("일본어 자막 생성 결과가 비어 있습니다.")

                    self.item_status_changed.emit(source_str, STATUS_TRANSLATING, "")
                    self.stage_changed.emit("번역", "한국어 번역")
                    document = load_subtitle_document_from_text(".srt", japanese_srt_text)
                    translated = translator.translate_lines(
                        document.get_translatable_records(),
                        self.translation_progress_changed.emit,
                        source_path.name,
                    )
                    document.apply_translations(translated)
                    output_path.write_text(document.render() + "\n", encoding="utf-8-sig")

                    success_count += 1
                    self.item_status_changed.emit(source_str, STATUS_DONE, str(output_path))
                    self.log_message.emit(f"{source_path} | 완료 -> {output_path}")
                    self.stage_changed.emit("완료", "한국어 자막 저장 완료")
                    self.stage_progress_changed.emit(100, 0.0, 0.0)
                except Exception as exc:
                    failure_count += 1
                    message = str(exc) or exc.__class__.__name__
                    self.item_status_changed.emit(source_str, STATUS_FAILED, message)
                    self.log_message.emit(f"{source_path} | 실패 -> {message}")
                    self.stage_changed.emit("실패", message)
                finally:
                    processed += 1
                    self.queue_progress_changed.emit(processed, total)

            self.summary_ready.emit(success_count, failure_count, skipped_count)
        except Exception as exc:
            self.failed.emit(str(exc) or exc.__class__.__name__)

    def _report_runtime_progress(
        self,
        downloaded_bytes: int,
        total_bytes: int,
        filename: str,
        file_downloaded: int,
        file_total: int,
    ) -> None:
        self._emit_prep_progress("CUDA 런타임", downloaded_bytes, total_bytes, filename, file_downloaded, file_total)

    def _report_dictionary_progress(
        self,
        downloaded_bytes: int,
        total_bytes: int,
        filename: str,
        file_downloaded: int,
        file_total: int,
    ) -> None:
        self._emit_prep_progress("일본어 사전팩", downloaded_bytes, total_bytes, filename, file_downloaded, file_total)

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
                f"{label} 다운로드 {percent}% | {filename} ({format_bytes(file_downloaded)}/{format_bytes(file_total)})"
            )
