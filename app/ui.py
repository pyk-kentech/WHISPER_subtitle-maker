from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import APP_NAME, MODEL_LABEL
from .file_queue import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_PROCESSING,
    STATUS_SKIPPED,
    QueueItem,
    normalize_input_files,
)
from .model_manager import get_model_dir, is_model_ready
from .transcriber import get_available_runtime_choices, get_default_runtime_choice
from .workers import ModelDownloadWorker, TranscriptionWorker


class DropArea(QFrame):
    files_dropped = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("dropArea")
        self.setStyleSheet(
            "#dropArea {"
            "border: 2px dashed #6a7b8c;"
            "border-radius: 10px;"
            "background: #f5f8fb;"
            "}"
        )
        layout = QVBoxLayout(self)
        label = QLabel("여기에 .mp3 / .mp4 파일을 드래그 앤 드롭")
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(90)
        layout.addWidget(label)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                paths.append(url.toLocalFile())
        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()


class MainWindow(QMainWindow):
    def __init__(self, auto_download_on_startup: bool = True) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1100, 760)

        self._auto_download_on_startup = auto_download_on_startup
        self._items: dict[str, QueueItem] = {}
        self._rows_by_path: dict[str, int] = {}
        self._download_worker: ModelDownloadWorker | None = None
        self._transcription_worker: TranscriptionWorker | None = None
        self._model_ready = False
        self._processing = False
        self._settings = QSettings("Codex", APP_NAME)
        self._session_success_count = 0
        self._session_failure_count = 0
        self._session_skipped_count = 0
        self._current_file_prefix = "현재 파일: 대기 중"

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        title_label = QLabel("일본어 음성 -> SRT 자막 생성")
        title_label.setStyleSheet("font-size: 20px; font-weight: 700;")
        root_layout.addWidget(title_label)

        fixed_config_label = QLabel(f"고정 설정: {MODEL_LABEL}")
        fixed_config_label.setStyleSheet("color: #43556a;")
        root_layout.addWidget(fixed_config_label)

        runtime_row = QHBoxLayout()
        runtime_label = QLabel("실행 장치")
        runtime_row.addWidget(runtime_label)

        self.runtime_combo = QComboBox()
        self._populate_runtime_choices()
        self.runtime_combo.currentIndexChanged.connect(self.on_runtime_changed)
        runtime_row.addWidget(self.runtime_combo)
        runtime_row.addStretch(1)
        root_layout.addLayout(runtime_row)

        self.model_status_label = QLabel("모델 상태 확인 중...")
        root_layout.addWidget(self.model_status_label)

        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 100)
        self.model_progress.setValue(0)
        root_layout.addWidget(self.model_progress)

        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self.add_files)
        root_layout.addWidget(self.drop_area)

        button_row = QHBoxLayout()
        self.add_button = QPushButton("파일 추가")
        self.add_button.clicked.connect(self.open_file_dialog)
        button_row.addWidget(self.add_button)

        self.start_button = QPushButton("시작")
        self.start_button.clicked.connect(self.start_transcription)
        button_row.addWidget(self.start_button)

        self.retry_download_button = QPushButton("모델 다운로드 재시도")
        self.retry_download_button.clicked.connect(self.start_model_download)
        button_row.addWidget(self.retry_download_button)

        button_row.addStretch(1)
        root_layout.addLayout(button_row)

        splitter = QSplitter(Qt.Vertical)
        root_layout.addWidget(splitter, 1)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["파일명", "전체 경로", "상태"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        splitter.addWidget(self.table)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        splitter.addWidget(self.log_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.current_file_label = QLabel(self._current_file_prefix)
        root_layout.addWidget(self.current_file_label)

        self.current_file_progress = QProgressBar()
        self.current_file_progress.setRange(0, 100)
        self.current_file_progress.setValue(0)
        root_layout.addWidget(self.current_file_progress)

        self.batch_progress_label = QLabel("작업 진행률: 0 / 0")
        root_layout.addWidget(self.batch_progress_label)

        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setValue(0)
        root_layout.addWidget(self.batch_progress)

        self.setCentralWidget(central)

        self.log("프로그램 시작")
        self.refresh_model_status()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.is_busy():
            QMessageBox.warning(self, APP_NAME, "다운로드 또는 변환 작업이 진행 중입니다.")
            event.ignore()
            return
        super().closeEvent(event)

    def is_busy(self) -> bool:
        return (
            (self._download_worker is not None and self._download_worker.isRunning())
            or (self._transcription_worker is not None and self._transcription_worker.isRunning())
        )

    def _populate_runtime_choices(self) -> None:
        saved = str(self._settings.value("runtime_device", get_default_runtime_choice()))
        choices = get_available_runtime_choices()
        self.runtime_combo.clear()
        selected_index = 0

        for index, (device, label) in enumerate(choices):
            self.runtime_combo.addItem(label, device)
            if device == saved:
                selected_index = index

        self.runtime_combo.setCurrentIndex(selected_index)

    def current_runtime_device(self) -> str:
        data = self.runtime_combo.currentData()
        return str(data or "cpu")

    def on_runtime_changed(self) -> None:
        device = self.current_runtime_device()
        self._settings.setValue("runtime_device", device)
        self.log(f"실행 장치 선택: {self.runtime_combo.currentText()}")

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.append(f"[{timestamp}] {message}")

    def refresh_model_status(self) -> None:
        self._model_ready = is_model_ready()
        if self._model_ready:
            self.model_status_label.setText(f"모델 준비 완료: {get_model_dir()}")
            self.model_progress.setValue(100)
            self.log(f"모델 캐시 사용: {get_model_dir()}")
        else:
            self.model_status_label.setText("모델 다운로드 필요")
            self.model_progress.setValue(0)
            if self._auto_download_on_startup:
                self.log("최초 실행 또는 캐시 없음: 모델 다운로드 시작")
                self.start_model_download()
            else:
                self.log("모델 다운로드 대기 중")
        self.update_controls()

    def update_controls(self) -> None:
        has_pending = any(item.status in {STATUS_PENDING, STATUS_FAILED} for item in self._items.values())
        download_running = self._download_worker is not None and self._download_worker.isRunning()
        processing_running = self._transcription_worker is not None and self._transcription_worker.isRunning()
        self.add_button.setEnabled(True)
        self.drop_area.setEnabled(True)
        self.runtime_combo.setEnabled(not processing_running and not download_running)
        self.start_button.setEnabled(self._model_ready and has_pending and not processing_running and not download_running)
        self.retry_download_button.setEnabled(not download_running and not self._model_ready)

    def open_file_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "음성/영상 파일 선택",
            "",
            "Audio / Video Files (*.mp3 *.mp4)",
        )
        if files:
            self.add_files(files)

    def add_files(self, paths: list[str]) -> None:
        items, errors = normalize_input_files(paths)

        for error in errors:
            self.log(error)

        added_count = 0
        for item in items:
            key = str(item.source_path)
            if key in self._items:
                self.log(f"이미 목록에 존재함: {item.source_path}")
                continue

            row = self.table.rowCount()
            self.table.insertRow(row)
            self.table.setItem(row, 0, QTableWidgetItem(item.source_path.name))
            self.table.setItem(row, 1, QTableWidgetItem(str(item.source_path)))
            self.table.setItem(row, 2, QTableWidgetItem(item.status))
            self._items[key] = item
            self._rows_by_path[key] = row
            added_count += 1

        if added_count:
            self.log(f"파일 {added_count}개 추가")
            if self._processing:
                self.log("진행 중 추가된 파일은 현재 작업이 끝난 뒤 자동으로 이어서 처리됩니다.")
            self.refresh_queue_progress()
        self.update_controls()

    def update_item_status(self, source_path: str, status: str, message: str = "") -> None:
        item = self._items.get(source_path)
        row = self._rows_by_path.get(source_path)
        if item is None or row is None:
            return

        item.status = status
        item.message = message
        status_text = status if not message else f"{status} | {message}"
        status_item = self.table.item(row, 2)
        if status_item is None:
            status_item = QTableWidgetItem()
            self.table.setItem(row, 2, status_item)
        status_item.setText(status_text)

    def start_model_download(self) -> None:
        if self._download_worker is not None and self._download_worker.isRunning():
            return

        self._download_worker = ModelDownloadWorker(self)
        self._download_worker.progress_changed.connect(self.model_progress.setValue)
        self._download_worker.status_changed.connect(self.model_status_label.setText)
        self._download_worker.log_message.connect(self.log)
        self._download_worker.finished_success.connect(self.on_model_download_success)
        self._download_worker.failed.connect(self.on_model_download_failed)
        self._download_worker.start()
        self.update_controls()

    def on_model_download_success(self, model_path: str) -> None:
        self._model_ready = True
        self.model_progress.setValue(100)
        self.model_status_label.setText(f"모델 준비 완료: {model_path}")
        self.log(f"모델 다운로드 완료: {model_path}")
        self.update_controls()

    def on_model_download_failed(self, message: str) -> None:
        self._model_ready = False
        self.model_status_label.setText(f"모델 다운로드 실패: {message}")
        self.log(f"모델 다운로드 실패: {message}")
        QMessageBox.warning(
            self,
            APP_NAME,
            f"모델 다운로드에 실패했습니다.\n\n{message}\n\n재시도 버튼으로 다시 시도하세요.",
        )
        self.update_controls()

    def start_transcription(self) -> None:
        if not self._model_ready:
            self.log("모델 준비 전에는 변환을 시작할 수 없습니다.")
            return

        source_paths = [
            path
            for path, item in self._items.items()
            if item.status in {STATUS_PENDING, STATUS_FAILED}
        ]
        if not source_paths:
            self.log("처리할 파일이 없습니다.")
            return

        self._processing = True
        self._session_success_count = 0
        self._session_failure_count = 0
        self._session_skipped_count = 0
        self._start_next_pending_batch(include_failed=True)

    def _start_next_pending_batch(self, include_failed: bool = False) -> None:
        allowed = {STATUS_PENDING, STATUS_FAILED} if include_failed else {STATUS_PENDING}
        source_paths = [path for path, item in self._items.items() if item.status in allowed]
        if not source_paths:
            return

        for source_path in source_paths:
            self.update_item_status(source_path, STATUS_PENDING)

        self.refresh_queue_progress()
        self._current_file_prefix = "현재 파일: 준비 중"
        self.current_file_label.setText(self._current_file_prefix)
        self.current_file_progress.setValue(0)

        runtime_device = self.current_runtime_device()
        self._transcription_worker = TranscriptionWorker(source_paths, runtime_device, self)
        self._transcription_worker.item_status_changed.connect(self.update_item_status)
        self._transcription_worker.log_message.connect(self.log)
        self._transcription_worker.progress_changed.connect(self.on_batch_progress_changed)
        self._transcription_worker.current_file_changed.connect(self.on_current_file_changed)
        self._transcription_worker.current_file_progress.connect(self.on_current_file_progress_changed)
        self._transcription_worker.summary_ready.connect(self.on_batch_summary_ready)
        self._transcription_worker.failed.connect(self.on_batch_failed)
        self._transcription_worker.start()
        self.update_controls()
        self.log(f"변환 시작: {len(source_paths)}개 파일, 장치={self.runtime_combo.currentText()}")

    def on_batch_progress_changed(self, processed: int, total: int) -> None:
        self.refresh_queue_progress()

    def refresh_queue_progress(self) -> None:
        total = len(self._items)
        finished = sum(
            1 for item in self._items.values() if item.status in {STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED}
        )
        percent = 100 if total == 0 else int(finished * 100 / total)
        self.batch_progress.setValue(percent)
        self.batch_progress_label.setText(f"전체 대기열 진행률: {finished} / {total}")

    def on_current_file_changed(self, index: int, total: int, source_path: str) -> None:
        self._current_file_prefix = f"현재 파일: {index} / {total} | {source_path}"
        self.current_file_label.setText(self._current_file_prefix)
        self.current_file_progress.setValue(0)

    def on_current_file_progress_changed(self, percent: int, position: float, duration: float) -> None:
        self.current_file_progress.setValue(max(0, min(100, percent)))
        if duration > 0:
            self.current_file_label.setText(
                f"{self._current_file_prefix} | {position:.1f}s / {duration:.1f}s"
            )

    def on_batch_summary_ready(self, success_count: int, failure_count: int, skipped_count: int) -> None:
        self._session_success_count += success_count
        self._session_failure_count += failure_count
        self._session_skipped_count += skipped_count
        self.refresh_queue_progress()

        pending_exists = any(item.status == STATUS_PENDING for item in self._items.values())
        if pending_exists:
            self.log("대기열에 남은 파일이 있어 다음 순차 작업을 이어서 시작합니다.")
            self._start_next_pending_batch()
            return

        self._processing = False
        self._current_file_prefix = "현재 파일: 작업 완료"
        self.current_file_label.setText(self._current_file_prefix)
        self.current_file_progress.setValue(100)
        summary = (
            f"작업 완료\n성공: {self._session_success_count}\n실패: {self._session_failure_count}\n스킵: {self._session_skipped_count}"
        )
        self.log(summary.replace("\n", " | "))
        QMessageBox.information(self, APP_NAME, summary)
        self.update_controls()

    def on_batch_failed(self, message: str) -> None:
        self._processing = False
        self._current_file_prefix = "현재 파일: 시작 실패"
        self.current_file_label.setText(self._current_file_prefix)
        self.log(f"작업 시작 실패: {message}")
        QMessageBox.warning(self, APP_NAME, f"작업을 시작할 수 없습니다.\n\n{message}")
        self.update_controls()
