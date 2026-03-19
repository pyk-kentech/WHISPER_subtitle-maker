from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import APP_NAME, MODEL_LABEL
from .file_queue import (
    QueueItem,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_SKIPPED,
    STATUS_TRANSCRIBING,
    STATUS_TRANSLATING,
    normalize_input_files,
)
from .model_manager import get_model_dir, is_model_ready
from .transcriber import get_available_runtime_choices, get_default_runtime_choice
from .translator_store import TranslatorSettings, load_api_keys, load_translator_settings, save_api_keys, save_translator_settings
from .workers import ModelDownloadWorker, PipelineJob, PipelineWorker


GREEN_BAR_STYLE = """
QProgressBar {
    border: 1px solid #9bb49f;
    border-radius: 6px;
    background: #eef3ef;
    text-align: center;
}
QProgressBar::chunk {
    background-color: #2f9e44;
    border-radius: 5px;
}
"""


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
        paths: list[str] = []
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
        self.resize(1200, 860)

        self._auto_download_on_startup = auto_download_on_startup
        self._items: dict[str, QueueItem] = {}
        self._rows_by_path: dict[str, int] = {}
        self._download_worker: ModelDownloadWorker | None = None
        self._pipeline_worker: PipelineWorker | None = None
        self._model_ready = False
        self._processing = False
        self._session_success_count = 0
        self._session_failure_count = 0
        self._session_skipped_count = 0
        self._queue_total = 0
        self._queue_processed = 0
        self._current_file_prefix = "현재 파일: 대기 중"
        self._stage_title = "대기 중"
        self._stage_detail = ""

        self._translator_settings = load_translator_settings()

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        title_label = QLabel("일본 음성 -> 한국어 자막 생성기")
        title_label.setStyleSheet("font-size: 20px; font-weight: 700;")
        root_layout.addWidget(title_label)

        subtitle_label = QLabel(f"고정 STT 설정: {MODEL_LABEL}")
        subtitle_label.setStyleSheet("color: #43556a;")
        root_layout.addWidget(subtitle_label)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_main_tab(), "작업")
        self.tabs.addTab(self._build_translation_tab(), "번역 설정")
        root_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(central)
        self.log("프로그램 시작")
        self.refresh_model_status()

    def _build_main_tab(self) -> QWidget:
        page = QWidget()
        root_layout = QVBoxLayout(page)
        root_layout.setSpacing(10)

        runtime_row = QHBoxLayout()
        runtime_row.addWidget(QLabel("실행 장치"))

        self.runtime_combo = QComboBox()
        self._populate_runtime_choices()
        self.runtime_combo.currentIndexChanged.connect(self.on_runtime_changed)
        runtime_row.addWidget(self.runtime_combo)

        self.postprocess_checkbox = QCheckBox("기본 일본어 후처리")
        self.postprocess_checkbox.setChecked(True)
        self.postprocess_checkbox.toggled.connect(self.on_postprocess_changed)
        runtime_row.addWidget(self.postprocess_checkbox)

        self.enhanced_postprocess_checkbox = QCheckBox("강화 후처리(사전팩)")
        self.enhanced_postprocess_checkbox.setChecked(False)
        self.enhanced_postprocess_checkbox.toggled.connect(self.on_enhanced_postprocess_changed)
        runtime_row.addWidget(self.enhanced_postprocess_checkbox)
        runtime_row.addStretch(1)
        root_layout.addLayout(runtime_row)

        self.model_status_label = QLabel("모델 상태 확인 중...")
        root_layout.addWidget(self.model_status_label)

        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 100)
        self.model_progress.setValue(0)
        self.model_progress.setStyleSheet(GREEN_BAR_STYLE)
        root_layout.addWidget(self.model_progress)

        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self.add_files)
        root_layout.addWidget(self.drop_area)

        button_row = QHBoxLayout()
        self.add_button = QPushButton("파일 추가")
        self.add_button.clicked.connect(self.open_file_dialog)
        button_row.addWidget(self.add_button)

        self.remove_button = QPushButton("선택 제거")
        self.remove_button.clicked.connect(self.remove_selected_files)
        button_row.addWidget(self.remove_button)

        self.start_button = QPushButton("시작")
        self.start_button.clicked.connect(self.start_pipeline)
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
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        splitter.addWidget(self.table)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        splitter.addWidget(self.log_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.current_file_label = QLabel(self._current_file_prefix)
        root_layout.addWidget(self.current_file_label)

        self.current_stage_label = QLabel("현재 단계: 대기 중")
        root_layout.addWidget(self.current_stage_label)

        self.stage_progress = QProgressBar()
        self.stage_progress.setRange(0, 100)
        self.stage_progress.setValue(0)
        self.stage_progress.setStyleSheet(GREEN_BAR_STYLE)
        root_layout.addWidget(self.stage_progress)

        self.translation_progress_label = QLabel("번역 진행: 대기 중")
        root_layout.addWidget(self.translation_progress_label)

        self.translation_progress = QProgressBar()
        self.translation_progress.setRange(0, 100)
        self.translation_progress.setValue(0)
        self.translation_progress.setStyleSheet(GREEN_BAR_STYLE)
        root_layout.addWidget(self.translation_progress)

        self.queue_progress_label = QLabel("전체 대기열 진행률 0 / 0")
        root_layout.addWidget(self.queue_progress_label)

        self.queue_progress = QProgressBar()
        self.queue_progress.setRange(0, 100)
        self.queue_progress.setValue(0)
        self.queue_progress.setStyleSheet(GREEN_BAR_STYLE)
        root_layout.addWidget(self.queue_progress)

        return page

    def _build_translation_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(10)

        api_box = QGroupBox("API 키")
        api_layout = QVBoxLayout(api_box)
        api_layout.addWidget(QLabel("여러 키를 한 줄에 하나씩 입력하면 자동 저장됩니다."))
        self.keys_edit = QPlainTextEdit("\n".join(load_api_keys()))
        self.keys_edit.setPlaceholderText("AIza...")
        self.keys_edit.setMinimumHeight(140)
        self.keys_edit.textChanged.connect(self.save_translation_inputs)
        api_layout.addWidget(self.keys_edit)
        layout.addWidget(api_box)

        option_box = QGroupBox("Gemini 번역 옵션")
        form = QFormLayout(option_box)

        self.model_edit = QLineEdit(self._translator_settings.preferred_model)
        self.model_edit.setPlaceholderText("비우면 자동 모델 순서 사용")
        self.model_edit.textChanged.connect(self.save_translation_inputs)
        form.addRow("모델 지정", self.model_edit)

        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(10, 200)
        self.chunk_size_spin.setValue(self._translator_settings.chunk_size)
        self.chunk_size_spin.valueChanged.connect(self.save_translation_inputs)
        form.addRow("청크 크기", self.chunk_size_spin)

        self.reasoning_combo = QComboBox()
        self.reasoning_combo.addItems(["minimal", "low", "medium", "high"])
        self.reasoning_combo.setCurrentText(self._translator_settings.reasoning_level)
        self.reasoning_combo.currentTextChanged.connect(self.save_translation_inputs)
        form.addRow("추론 레벨", self.reasoning_combo)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(self._translator_settings.temperature)
        self.temperature_spin.valueChanged.connect(self.save_translation_inputs)
        form.addRow("온도", self.temperature_spin)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(self._translator_settings.top_p)
        self.top_p_spin.valueChanged.connect(self.save_translation_inputs)
        form.addRow("Top-P", self.top_p_spin)
        layout.addWidget(option_box)

        prompt_box = QGroupBox("프롬프트")
        prompt_layout = QVBoxLayout(prompt_box)
        prompt_layout.addWidget(QLabel("시스템 프롬프트"))
        self.system_prompt_edit = QPlainTextEdit(self._translator_settings.system_prompt)
        self.system_prompt_edit.setMinimumHeight(240)
        self.system_prompt_edit.textChanged.connect(self.save_translation_inputs)
        prompt_layout.addWidget(self.system_prompt_edit)

        prompt_layout.addWidget(QLabel("번역 노트"))
        self.note_edit = QPlainTextEdit(self._translator_settings.translation_note)
        self.note_edit.setMinimumHeight(140)
        self.note_edit.textChanged.connect(self.save_translation_inputs)
        prompt_layout.addWidget(self.note_edit)
        layout.addWidget(prompt_box, 1)

        return page

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.is_busy():
            QMessageBox.warning(self, APP_NAME, "다운로드 또는 작업이 진행 중입니다.")
            event.ignore()
            return
        super().closeEvent(event)

    def is_busy(self) -> bool:
        return (
            (self._download_worker is not None and self._download_worker.isRunning())
            or (self._pipeline_worker is not None and self._pipeline_worker.isRunning())
        )

    def _populate_runtime_choices(self) -> None:
        saved = get_default_runtime_choice()
        choices = get_available_runtime_choices()
        self.runtime_combo.clear()
        selected_index = 0
        for index, (device, label) in enumerate(choices):
            self.runtime_combo.addItem(label, device)
            if device == saved:
                selected_index = index
        self.runtime_combo.setCurrentIndex(selected_index)

    def on_runtime_changed(self) -> None:
        self.log(f"실행 장치 선택: {self.runtime_combo.currentText()}")

    def on_enhanced_postprocess_changed(self, checked: bool) -> None:
        if checked and not self.postprocess_checkbox.isChecked():
            self.postprocess_checkbox.setChecked(True)

    def on_postprocess_changed(self, checked: bool) -> None:
        if not checked and self.enhanced_postprocess_checkbox.isChecked():
            self.enhanced_postprocess_checkbox.setChecked(False)

    def current_runtime_device(self) -> str:
        return str(self.runtime_combo.currentData() or "cpu")

    def translator_settings(self) -> TranslatorSettings:
        return TranslatorSettings(
            preferred_model=self.model_edit.text().strip(),
            chunk_size=self.chunk_size_spin.value(),
            temperature=self.temperature_spin.value(),
            top_p=self.top_p_spin.value(),
            reasoning_level=self.reasoning_combo.currentText(),
            system_prompt=self.system_prompt_edit.toPlainText(),
            translation_note=self.note_edit.toPlainText(),
        )

    def save_translation_inputs(self) -> None:
        save_api_keys(self.keys_edit.toPlainText())
        save_translator_settings(self.translator_settings())

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
        has_rows = self.table.rowCount() > 0
        download_running = self._download_worker is not None and self._download_worker.isRunning()
        processing_running = self._pipeline_worker is not None and self._pipeline_worker.isRunning()
        enabled = not download_running and not processing_running

        self.add_button.setEnabled(True)
        self.drop_area.setEnabled(True)
        self.remove_button.setEnabled(has_rows)
        self.runtime_combo.setEnabled(enabled)
        self.postprocess_checkbox.setEnabled(enabled)
        self.enhanced_postprocess_checkbox.setEnabled(enabled)
        self.start_button.setEnabled(self._model_ready and has_pending and enabled)
        self.retry_download_button.setEnabled(not download_running and not self._model_ready)
        self.tabs.setTabEnabled(1, True)

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
                self.log("진행 중에 추가된 파일은 현재 작업이 끝난 뒤 자동으로 이어서 처리합니다.")
            self.refresh_queue_progress()
        self.update_controls()

    def _rebuild_row_index(self) -> None:
        self._rows_by_path.clear()
        for row in range(self.table.rowCount()):
            path_item = self.table.item(row, 1)
            if path_item is not None:
                self._rows_by_path[path_item.text()] = row

    def _remove_paths(self, paths: list[str]) -> int:
        rows_to_remove: set[int] = set()
        for source_path in paths:
            row = self._rows_by_path.get(source_path)
            if row is not None:
                rows_to_remove.add(row)

        if not rows_to_remove:
            return 0

        for row in sorted(rows_to_remove, reverse=True):
            path_item = self.table.item(row, 1)
            if path_item is not None:
                self._items.pop(path_item.text(), None)
            self.table.removeRow(row)

        self._rebuild_row_index()
        self.refresh_queue_progress()
        self.update_controls()
        return len(rows_to_remove)

    def remove_selected_files(self) -> None:
        selection_model = self.table.selectionModel()
        if selection_model is None:
            return
        selected_rows = selection_model.selectedRows()
        if not selected_rows:
            self.log("제거할 파일이 선택되지 않았습니다.")
            return

        removable_paths: list[str] = []
        for index in selected_rows:
            path_item = self.table.item(index.row(), 1)
            if path_item is not None:
                removable_paths.append(path_item.text())

        removed_count = self._remove_paths(removable_paths)
        if removed_count:
            self.log(f"선택한 파일 {removed_count}개를 목록에서 제거했습니다.")

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

        if status == STATUS_DONE:
            removed_count = self._remove_paths([source_path])
            if removed_count:
                self.log(f"완료 파일 자동 제거: {source_path}")

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
        QMessageBox.warning(self, APP_NAME, f"모델 다운로드에 실패했습니다.\n\n{message}")
        self.update_controls()

    def start_pipeline(self) -> None:
        self.save_translation_inputs()
        api_keys = load_api_keys()
        if not api_keys:
            self.tabs.setCurrentIndex(1)
            QMessageBox.warning(self, APP_NAME, "번역 설정 탭에 Gemini API 키를 입력하세요.")
            return
        if not self._model_ready:
            self.log("모델 준비 전에는 작업을 시작할 수 없습니다.")
            return

        source_paths = [path for path, item in self._items.items() if item.status in {STATUS_PENDING, STATUS_FAILED}]
        if not source_paths:
            self.log("처리할 파일이 없습니다.")
            return

        for source_path in source_paths:
            self.update_item_status(source_path, STATUS_PENDING)

        self._processing = True
        self._session_success_count = 0
        self._session_failure_count = 0
        self._session_skipped_count = 0
        self._queue_total = len(source_paths)
        self._queue_processed = 0
        self.current_file_label.setText("현재 파일: 준비 중")
        self._stage_title = "작업 준비"
        self._stage_detail = ""
        self.current_stage_label.setText("현재 단계: 작업 준비")
        self.stage_progress.setValue(0)
        self.translation_progress.setValue(0)
        self.translation_progress_label.setText("번역 진행: 대기 중")

        job = PipelineJob(
            source_paths=source_paths,
            runtime_device=self.current_runtime_device(),
            enable_postprocess=self.postprocess_checkbox.isChecked(),
            enable_enhanced_postprocess=self.enhanced_postprocess_checkbox.isChecked(),
            translator_settings=self.translator_settings(),
            api_keys=api_keys,
        )
        self._pipeline_worker = PipelineWorker(job, self)
        self._pipeline_worker.item_status_changed.connect(self.update_item_status)
        self._pipeline_worker.log_message.connect(self.log)
        self._pipeline_worker.queue_progress_changed.connect(self.on_queue_progress_changed)
        self._pipeline_worker.file_started.connect(self.on_file_started)
        self._pipeline_worker.stage_changed.connect(self.on_stage_changed)
        self._pipeline_worker.stage_progress_changed.connect(self.on_stage_progress_changed)
        self._pipeline_worker.translation_progress_changed.connect(self.on_translation_progress_changed)
        self._pipeline_worker.summary_ready.connect(self.on_pipeline_summary_ready)
        self._pipeline_worker.failed.connect(self.on_pipeline_failed)
        self._pipeline_worker.start()
        self.update_controls()
        self.log(
            f"작업 시작: {len(source_paths)}개 파일, 장치={self.runtime_combo.currentText()}, "
            f"후처리={'강화' if self.enhanced_postprocess_checkbox.isChecked() else ('기본' if self.postprocess_checkbox.isChecked() else '꺼짐')}"
        )

    def refresh_queue_progress(self) -> None:
        if self._queue_total > 0:
            total = self._queue_total
            finished = self._queue_processed
        else:
            total = len(self._items)
            finished = sum(1 for item in self._items.values() if item.status in {STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED})
        percent = 100 if total == 0 else int(finished * 100 / total)
        self.queue_progress.setValue(percent)
        self.queue_progress_label.setText(f"전체 대기열 진행률 {finished} / {total}")

    def on_queue_progress_changed(self, processed: int, total: int) -> None:
        self._queue_processed = processed
        self._queue_total = total
        self.refresh_queue_progress()

    def on_file_started(self, index: int, total: int, source_path: str) -> None:
        self.current_file_label.setText(f"현재 파일: {index} / {total} | {source_path}")
        self.stage_progress.setValue(0)
        self.translation_progress.setValue(0)

    def on_stage_changed(self, stage_title: str, stage_detail: str) -> None:
        self.current_stage_label.setText(f"현재 단계: {stage_title} | {stage_detail}")
        self._stage_title = stage_title
        self._stage_detail = stage_detail

    def on_stage_progress_changed(self, percent: int, position: float, duration: float) -> None:
        self.stage_progress.setValue(max(0, min(100, percent)))
        if duration > 0:
            detail = f"{self._stage_detail} | {position:.1f}s / {duration:.1f}s" if self._stage_detail else f"{position:.1f}s / {duration:.1f}s"
            self.current_stage_label.setText(f"현재 단계: {self._stage_title} | {detail}")

    def on_translation_progress_changed(self, chunk_index: int, chunk_total: int, key_display: str, model_name: str, error_count: int) -> None:
        percent = 0 if chunk_total <= 0 else int(chunk_index * 100 / chunk_total)
        self.translation_progress.setValue(max(0, min(100, percent)))
        if chunk_total <= 0:
            self.translation_progress_label.setText("번역 진행: 대기 중")
            return
        self.translation_progress_label.setText(
            f"번역 진행: 청크 {chunk_index}/{chunk_total} | 키 {key_display} | 모델 {model_name or '-'} | 에러 {error_count}"
        )

    def on_pipeline_summary_ready(self, success_count: int, failure_count: int, skipped_count: int) -> None:
        self._processing = False
        self._session_success_count += success_count
        self._session_failure_count += failure_count
        self._session_skipped_count += skipped_count
        self.current_file_label.setText("현재 파일: 작업 완료")
        self._stage_title = "완료"
        self._stage_detail = ""
        self.current_stage_label.setText("현재 단계: 완료")
        self.stage_progress.setValue(100)
        self.translation_progress.setValue(100)
        self.refresh_queue_progress()
        self._queue_total = 0
        self._queue_processed = 0
        self.update_controls()

        summary = (
            f"작업 완료\n성공: {self._session_success_count}\n실패: {self._session_failure_count}\n스킵: {self._session_skipped_count}"
        )
        self.log(summary.replace("\n", " | "))
        QMessageBox.information(self, APP_NAME, summary)

    def on_pipeline_failed(self, message: str) -> None:
        self._processing = False
        self._stage_title = "실패"
        self._stage_detail = message
        self.current_stage_label.setText(f"현재 단계: 실패 | {message}")
        self._queue_total = 0
        self._queue_processed = 0
        self.log(f"작업 실패: {message}")
        self.update_controls()
        QMessageBox.warning(self, APP_NAME, message)
