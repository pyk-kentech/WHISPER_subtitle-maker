from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .file_queue import STATUS_DONE, STATUS_FAILED, STATUS_PROCESSING, STATUS_SKIPPED
from .model_manager import download_model, get_download_plan, get_model_dir, is_model_ready
from .srt_writer import write_srt
from .transcriber import TranscriptionEngine, build_runtime_config


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
                self.log_message.emit(
                    f"모델 다운로드 필요: {format_bytes(plan.total_bytes)}"
                )
            else:
                self.log_message.emit("모델 캐시를 확인하는 중입니다.")

            def on_progress(
                downloaded_bytes: int,
                total_bytes: int,
                filename: str,
                file_downloaded: int,
                file_total: int,
            ) -> None:
                if total_bytes <= 0:
                    percent = 100
                else:
                    percent = min(100, int(downloaded_bytes * 100 / total_bytes))
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


class TranscriptionWorker(QThread):
    item_status_changed = Signal(str, str, str)
    log_message = Signal(str)
    progress_changed = Signal(int, int)
    current_file_changed = Signal(int, int, str)
    current_file_progress = Signal(int, float, float)
    summary_ready = Signal(int, int, int)
    failed = Signal(str)

    def __init__(self, source_paths: list[str], runtime_device: str, parent=None) -> None:
        super().__init__(parent)
        self.source_paths = source_paths
        self.runtime_device = runtime_device

    def run(self) -> None:
        try:
            success_count = 0
            failure_count = 0
            skipped_count = 0
            processed = 0
            total = len(self.source_paths)
            runtime_config = build_runtime_config(self.runtime_device)
            engine = TranscriptionEngine(runtime_config)
            self.log_message.emit(f"실행 장치: {runtime_config.label}")

            for index, source_str in enumerate(self.source_paths, start=1):
                source_path = Path(source_str)
                output_path = source_path.with_suffix(".srt")
                self.current_file_changed.emit(index, total, str(source_path))
                self.current_file_progress.emit(0, 0.0, 0.0)

                try:
                    self.item_status_changed.emit(source_str, STATUS_PROCESSING, "")

                    if not source_path.exists():
                        raise FileNotFoundError("입력 파일을 찾을 수 없습니다.")

                    if output_path.exists():
                        skipped_count += 1
                        message = "기존 자막 파일 존재 -> 스킵"
                        self.item_status_changed.emit(source_str, STATUS_SKIPPED, message)
                        self.log_message.emit(f"{source_path} | {message}")
                        self.current_file_progress.emit(100, 0.0, 0.0)
                        continue

                    segments = engine.transcribe_file(source_path, self.current_file_progress.emit)
                    write_srt(output_path, segments)
                    success_count += 1
                    self.item_status_changed.emit(source_str, STATUS_DONE, str(output_path))
                    self.log_message.emit(f"{source_path} | 완료 -> {output_path}")
                except Exception as exc:
                    failure_count += 1
                    message = str(exc) or exc.__class__.__name__
                    self.item_status_changed.emit(source_str, STATUS_FAILED, message)
                    self.log_message.emit(f"{source_path} | 실패 -> {message}")
                finally:
                    processed += 1
                    self.progress_changed.emit(processed, total)

            self.summary_ready.emit(success_count, failure_count, skipped_count)
        except Exception as exc:
            self.failed.emit(str(exc) or exc.__class__.__name__)
