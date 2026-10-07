from __future__ import annotations

from datetime import datetime
import dataclasses
import logging
import subprocess

from pathlib import Path

from PySide6.QtCore import QPoint, QRect, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QCloseEvent, QColor, QDesktopServices, QDragEnterEvent, QDropEvent, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
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
    QLayout,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedLayout,
    QStyle,
    QSystemTrayIcon,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from .config import (
    APP_NAME,
    DEFAULT_INPUT_LANGUAGE,
    DEFAULT_MODEL_KEY,
    DEFAULT_VAD_ENABLED,
    DEFAULT_VAD_MIN_SILENCE_MS,
    DEFAULT_VAD_SPEECH_PAD_MS,
    INPUT_LANGUAGE_OPTIONS,
    MODEL_PRESETS,
    OUTPUT_LANGUAGE_OPTIONS,
    get_preferred_icon_path,
)
from .app_logging import get_logger, set_ui_log_callback
from .audio_splitter import (
    DEFAULT_SILENCE_SEARCH_SECONDS,
    SPLIT_SUPPORTED_EXTENSIONS,
    AudioSplitError,
    build_cut_points,
    default_output_dir,
    format_timestamp,
    parse_timestamp,
    probe_audio,
)
from .file_queue import (
    QueueItem,
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PAUSED,
    STATUS_PENDING,
    STATUS_REMOVED,
    STATUS_SAVING,
    STATUS_SKIPPED,
    STATUS_TRANSCRIBING,
    STATUS_TRANSLATING,
    normalize_input_files,
    normalize_translation_files,
)
from .credential_store import CredentialStoreError
from .model_manager import get_model_dir, get_model_label, is_model_ready
from .transcriber import (
    RuntimeTuningOptions,
    VADSettings,
    describe_loaded_model_state,
    get_compute_type_choices,
    get_available_runtime_choices,
    get_default_runtime_choice,
    get_default_cpu_threads,
    get_memory_profile_choices,
    get_max_worker_count,
    unload_loaded_models,
)
from .translator_store import (
    TranslatorSettings,
    load_api_keys,
    load_deepl_api_key,
    load_translator_settings,
    save_api_keys,
    save_deepl_api_key,
    save_translator_settings,
)
from .workers import (
    AudioSplitJob,
    AudioSplitWorker,
    ModelDownloadWorker,
    PipelineJob,
    PipelineWorker,
    SubtitleTranslationJob,
    SubtitleTranslationWorker,
)


# --windowed 빌드에서 shutdown 명령을 실행할 때 콘솔 창이 깜빡이지 않게 한다(Windows 전용 플래그).
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

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

DROP_AREA_BASE_STYLE = (
    "#dropArea {"
    "border: 2px dashed #6a7b8c;"
    "border-radius: 10px;"
    "background: #f5f8fb;"
    "}"
)

DROP_AREA_ACTIVE_STYLE = (
    "#dropArea {"
    "border: 2px solid #2f9e44;"
    "border-radius: 10px;"
    "background: #eaf7ee;"
    "}"
)

DROP_TABLE_BASE_STYLE = (
    "QTableWidget {"
    "border: 1px solid #c9d4df;"
    "border-radius: 8px;"
    "background: #ffffff;"
    "gridline-color: #dde5ee;"
    "}"
)

DROP_TABLE_ACTIVE_STYLE = (
    "QTableWidget {"
    "border: 2px solid #2f9e44;"
    "border-radius: 8px;"
    "background: #eef9f1;"
    "gridline-color: #dde5ee;"
    "}"
)


class DropArea(QFrame):
    files_dropped = Signal(list)

    def __init__(
        self,
        label_text: str = "여기에 .mp3 / .mp4 파일을 드래그 앤 드롭",
        min_height: int = 150,
        word_wrap: bool = True,
    ) -> None:
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
        label = QLabel(label_text)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(min_height)
        label.setWordWrap(word_wrap)
        if not word_wrap:
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        label.setStyleSheet("font-size: 18px; font-weight: 600; color: #31455a;")
        layout.addWidget(label)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            self.setStyleSheet(DROP_AREA_ACTIVE_STYLE)
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet(DROP_AREA_BASE_STYLE)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        paths: list[str] = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                paths.append(url.toLocalFile())
        self.setStyleSheet(DROP_AREA_BASE_STYLE)
        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()


class FlowLayout(QLayout):
    """창 너비가 좁아지면 다음 줄로 넘어가는 가로 배치."""

    def __init__(self, parent=None, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items = []
        self.setSpacing(spacing)
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)
        height = self._do_layout(rect, test_only=False)
        # 상위 레이아웃(스플리터 안)이 heightForWidth를 반영하지 않아 줄바꿈된 버튼이 잘리므로 직접 최소 높이를 맞춘다.
        host = self.parentWidget()
        if host is not None and host.minimumHeight() != height:
            host.setMinimumHeight(height)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x, y, line_height = area.x(), area.y(), 0
        spacing = self.spacing()
        for item in self._items:
            hint = item.sizeHint()
            if x > area.x() and x + hint.width() > area.right() + 1:
                x = area.x()
                y += line_height + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + spacing
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


class DropTableWidget(QTableWidget):
    files_dropped = Signal(list)

    def __init__(self, rows: int, columns: int) -> None:
        super().__init__(rows, columns)
        self.setAcceptDrops(True)
        self.setStyleSheet(DROP_TABLE_BASE_STYLE)
        self.setWordWrap(False)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            self.setStyleSheet(DROP_TABLE_ACTIVE_STYLE)
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet(DROP_TABLE_BASE_STYLE)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:
        paths: list[str] = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                paths.append(url.toLocalFile())
        self.setStyleSheet(DROP_TABLE_BASE_STYLE)
        if paths:
            self.files_dropped.emit(paths)
        event.acceptProposedAction()


class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget, expanded: bool = True, parent=None) -> None:
        super().__init__(parent)
        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggle_button.clicked.connect(self._toggle)
        self.toggle_button.setStyleSheet("font-weight: 600; padding: 6px 0;")

        self.content = content
        self.content.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content)

    def _toggle(self, checked: bool) -> None:
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content.setVisible(checked)


class MainWindow(QMainWindow):
    ui_log_signal = Signal(str)

    def __init__(self, auto_download_on_startup: bool = True) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(980, 520)
        self.setMinimumSize(720, 420)
        self._app_icon = self._load_app_icon()
        if self._app_icon is not None:
            self.setWindowIcon(self._app_icon)

        self._auto_download_on_startup = auto_download_on_startup
        self._items: dict[str, QueueItem] = {}
        self._rows_by_path: dict[str, int] = {}
        self._download_worker: ModelDownloadWorker | None = None
        self._pipeline_worker: PipelineWorker | None = None
        self._subtitle_translation_worker: SubtitleTranslationWorker | None = None
        self._model_ready = False
        self._processing = False
        self._subtitle_translation_processing = False
        self._session_success_count = 0
        self._session_failure_count = 0
        self._session_skipped_count = 0
        self._subtitle_success_count = 0
        self._subtitle_failure_count = 0
        self._subtitle_skipped_count = 0
        self._allow_close = False
        self._tray_message_shown = False
        self.tray_icon: QSystemTrayIcon | None = None
        self._queue_total = 0
        self._queue_processed = 0
        self._subtitle_queue_total = 0
        self._subtitle_queue_processed = 0
        self._current_file_prefix = "현재 파일: 없음"
        self._stage_title = "대기 중"
        self._stage_detail = ""
        self._shutdown_scheduled = False
        self._subtitle_items: dict[str, QueueItem] = {}
        self._subtitle_rows_by_path: dict[str, int] = {}
        self._split_worker: AudioSplitWorker | None = None
        self._split_source: Path | None = None
        self._split_duration: float | None = None
        self._split_output_dir: Path | None = None
        self._pending_shutdown_label: str | None = None

        self._translator_settings = load_translator_settings()
        self._saved_api_keys_text = "\n".join(load_api_keys())
        self._saved_deepl_api_key = load_deepl_api_key()
        self._settings_dirty = False
        self._logger = get_logger()
        self.ui_log_signal.connect(self._append_ui_log)
        set_ui_log_callback(self.ui_log_signal.emit)

        central = QWidget()
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(10)

        title_label = QLabel("일본어 음성 -> 한국어 자막 생성기")
        title_label.setStyleSheet("font-size: 20px; font-weight: 700;")
        root_layout.addWidget(title_label)

        self.subtitle_label = QLabel("")
        self.subtitle_label.setStyleSheet("color: #43556a;")
        root_layout.addWidget(self.subtitle_label)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_main_tab(), "작업")
        self.tabs.addTab(self._build_subtitle_translate_tab(), "자막 번역")
        self.tabs.addTab(self._build_translation_tab(), "번역 설정")
        self.tabs.addTab(self._build_audio_split_tab(), "음성 분할")
        root_layout.addWidget(self.tabs, 1)

        self.setCentralWidget(central)
        self._create_tray_icon()
        self.log("Program started")
        self.refresh_model_status()

    def _load_app_icon(self) -> QIcon | None:
        icon_path = get_preferred_icon_path()
        if icon_path is None:
            return None
        icon = QIcon(str(icon_path))
        if icon.isNull():
            return None
        return icon

    def _build_main_tab(self) -> QWidget:
        content = QWidget()
        root_layout = QVBoxLayout(content)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(8)

        self.runtime_combo = QComboBox()
        self._populate_runtime_choices()
        self.runtime_combo.currentIndexChanged.connect(self.on_runtime_changed)

        self.model_combo = QComboBox()
        for model_key, preset in MODEL_PRESETS.items():
            self.model_combo.addItem(str(preset["label"]), model_key)
        self.model_combo.setCurrentIndex(max(0, self.model_combo.findData(DEFAULT_MODEL_KEY)))
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

        self.input_language_combo = QComboBox()
        for code, label in INPUT_LANGUAGE_OPTIONS:
            self.input_language_combo.addItem(label, code)
        self.input_language_combo.setCurrentIndex(max(0, self.input_language_combo.findData(DEFAULT_INPUT_LANGUAGE)))

        self.output_language_combo = QComboBox()
        for code, label in OUTPUT_LANGUAGE_OPTIONS:
            self.output_language_combo.addItem(label, code)
        self.output_language_combo.setCurrentIndex(max(0, self.output_language_combo.findData(self._translator_settings.target_language)))
        self.output_language_combo.currentIndexChanged.connect(self.mark_translation_inputs_dirty)

        self.translate_checkbox = QCheckBox("번역 사용")
        self.translate_checkbox.setChecked(True)

        self.postprocess_checkbox = QCheckBox("기본 후처리")
        self.postprocess_checkbox.setChecked(True)
        self.postprocess_checkbox.toggled.connect(self.on_postprocess_changed)

        self.enhanced_postprocess_checkbox = QCheckBox("강화 후처리")
        self.enhanced_postprocess_checkbox.setChecked(False)
        self.enhanced_postprocess_checkbox.toggled.connect(self.on_enhanced_postprocess_changed)

        self.compute_type_combo = QComboBox()
        self._populate_compute_type_choices()

        self.cpu_threads_spin = QSpinBox()
        self.cpu_threads_spin.setRange(1, get_max_worker_count())
        self.cpu_threads_spin.setValue(get_default_cpu_threads())

        self.num_workers_spin = QSpinBox()
        self.num_workers_spin.setRange(1, get_max_worker_count())
        self.num_workers_spin.setValue(1)

        self.memory_profile_combo = QComboBox()
        for value, label in get_memory_profile_choices():
            self.memory_profile_combo.addItem(label, value)
        self.memory_profile_combo.setCurrentIndex(max(0, self.memory_profile_combo.findData("unlimited")))

        self.auto_unload_checkbox = QCheckBox("작업 완료 후 모델 자동 해제")
        self.auto_unload_checkbox.setChecked(True)

        self.shutdown_after_complete_checkbox = QCheckBox("작업 완료 후 2분 뒤 시스템 종료")
        self.shutdown_after_complete_checkbox.setChecked(False)
        self.shutdown_after_complete_checkbox.toggled.connect(self.on_shutdown_after_complete_changed)

        self.vad_mode_combo = QComboBox()
        self.vad_mode_combo.addItem("ON", True)
        self.vad_mode_combo.addItem("OFF (Whisper only)", False)
        self.vad_mode_combo.setCurrentIndex(max(0, self.vad_mode_combo.findData(DEFAULT_VAD_ENABLED)))
        self.vad_mode_combo.currentIndexChanged.connect(self.on_vad_changed)

        self.vad_min_silence_spin = QSpinBox()
        self.vad_min_silence_spin.setRange(0, 5000)
        self.vad_min_silence_spin.setValue(DEFAULT_VAD_MIN_SILENCE_MS)

        self.vad_speech_pad_spin = QSpinBox()
        self.vad_speech_pad_spin.setRange(0, 5000)
        self.vad_speech_pad_spin.setValue(DEFAULT_VAD_SPEECH_PAD_MS)

        self.model_status_label = QLabel("모델 상태 확인 중...")
        self.model_runtime_state_label = QLabel("모델 없음")
        self.model_progress = QProgressBar()
        self.model_progress.setRange(0, 100)
        self.model_progress.setValue(0)
        self.model_progress.setStyleSheet(GREEN_BAR_STYLE)

        settings_bar = QHBoxLayout()
        settings_bar.setSpacing(6)
        settings_bar.addWidget(QLabel("작업 설정"))
        settings_bar.addWidget(self._build_settings_menu_button("STT 설정", self._create_stt_settings_menu()))
        settings_bar.addWidget(self._build_settings_menu_button("성능 설정", self._create_performance_settings_menu()))
        settings_bar.addWidget(self._build_settings_menu_button("처리 설정", self._create_process_settings_menu()))
        settings_bar.addStretch(1)
        root_layout.addLayout(settings_bar)

        self.main_settings_summary_label = QLabel("")
        self.main_settings_summary_label.setWordWrap(True)
        self.main_settings_summary_label.setStyleSheet(
            "padding: 8px 10px; border: 1px solid #d7e0ea; border-radius: 8px; background: #f6f9fc; color: #32465a;"
        )
        root_layout.addWidget(self.main_settings_summary_label)

        self._wire_main_settings_summary_signals()
        self.refresh_main_settings_summary()

        self.current_file_label = QLabel(self._current_file_prefix)
        self.current_stage_label = QLabel("현재 단계: 대기 중")
        self.stage_progress = QProgressBar()
        self.stage_progress.setRange(0, 100)
        self.stage_progress.setValue(0)
        self.stage_progress.setStyleSheet(GREEN_BAR_STYLE)
        self.translation_progress_label = QLabel("번역 진행: 대기 중")
        self.translation_progress = QProgressBar()
        self.translation_progress.setRange(0, 100)
        self.translation_progress.setValue(0)
        self.translation_progress.setStyleSheet(GREEN_BAR_STYLE)
        self.queue_progress_label = QLabel("전체 대기열 진행률 0 / 0")
        self.queue_progress = QProgressBar()
        self.queue_progress.setRange(0, 100)
        self.queue_progress.setValue(0)
        self.queue_progress.setStyleSheet(GREEN_BAR_STYLE)

        button_host = QWidget()
        button_row = FlowLayout(button_host)
        self.add_button = QPushButton("파일 추가")
        self.add_button.clicked.connect(self.open_file_dialog)
        button_row.addWidget(self.add_button)

        self.add_folder_button = QPushButton("폴더 추가")
        self.add_folder_button.clicked.connect(self.open_folder_dialog)
        button_row.addWidget(self.add_folder_button)

        self.include_subdirs_checkbox = QCheckBox("하위 폴더 포함")
        self.include_subdirs_checkbox.setChecked(True)
        button_row.addWidget(self.include_subdirs_checkbox)

        self.remove_button = QPushButton("선택 제거")
        self.remove_button.clicked.connect(self.remove_selected_files)
        button_row.addWidget(self.remove_button)

        self.start_button = QPushButton("시작")
        self.start_button.clicked.connect(self.start_pipeline)
        button_row.addWidget(self.start_button)
        self.pause_button = QPushButton("일시 중지")
        self.pause_button.clicked.connect(self.pause_pipeline)
        button_row.addWidget(self.pause_button)
        self.resume_button = QPushButton("재개")
        self.resume_button.clicked.connect(self.resume_pipeline)
        button_row.addWidget(self.resume_button)

        self.retry_download_button = QPushButton("모델 다운로드")
        self.retry_download_button.clicked.connect(self.start_model_download)
        button_row.addWidget(self.retry_download_button)

        self.unload_model_button = QPushButton("모델 해제")
        self.unload_model_button.clicked.connect(self.unload_model)
        button_row.addWidget(self.unload_model_button)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(self.model_status_label)
        left_layout.addWidget(self.model_runtime_state_label)
        left_layout.addWidget(self.model_progress)
        left_layout.addWidget(self.main_settings_summary_label)
        left_layout.addWidget(button_host)
        left_layout.addWidget(self.current_file_label)
        left_layout.addWidget(self.current_stage_label)
        left_layout.addWidget(self.stage_progress)
        left_layout.addWidget(self.translation_progress_label)
        left_layout.addWidget(self.translation_progress)
        left_layout.addWidget(self.queue_progress_label)
        left_layout.addWidget(self.queue_progress)
        left_layout.addStretch(1)
        for label in (
            self.model_status_label,
            self.model_runtime_state_label,
            self.current_file_label,
            self.current_stage_label,
            self.translation_progress_label,
            self.queue_progress_label,
            self.main_settings_summary_label,
        ):
            # 긴 경로·상태 문구가 왼쪽 패널의 최소 너비를 키워 분할선이 움직이지 않던 문제 방지
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            label.setWordWrap(True)

        right_splitter = QSplitter(Qt.Vertical)

        self.drop_area = DropArea()
        self.drop_area.files_dropped.connect(self.add_files)

        self.table = DropTableWidget(0, 3)
        self.table.files_dropped.connect(self.add_files)
        self.table.setHorizontalHeaderLabels(["파일명", "전체 경로", "상태"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setAlternatingRowColors(True)
        self.table.setTextElideMode(Qt.ElideMiddle)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setShowGrid(False)

        self.list_stack_host = QWidget()
        self.list_stack = QStackedLayout(self.list_stack_host)
        self.list_stack.setContentsMargins(0, 0, 0, 0)
        self.list_stack.addWidget(self.drop_area)
        self.list_stack.addWidget(self.table)
        self.list_stack.setCurrentWidget(self.drop_area)
        right_splitter.addWidget(self.list_stack_host)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        right_splitter.addWidget(self.log_view)
        right_splitter.setStretchFactor(0, 3)
        right_splitter.setStretchFactor(1, 2)
        right_splitter.setSizes([360, 240])

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)
        left_scroll.viewport().setAutoFillBackground(False)
        left_panel.setAutoFillBackground(False)

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(left_scroll)
        main_splitter.addWidget(right_splitter)
        main_splitter.setStretchFactor(0, 2)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setSizes([430, 770])
        root_layout.addWidget(main_splitter, 1)
        return content

    def _build_audio_split_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        extensions = " ".join(sorted(SPLIT_SUPPORTED_EXTENSIONS))
        self.split_drop_area = DropArea(
            f"여기에 나눌 음성 파일을 드래그 앤 드롭 ({extensions})", min_height=28, word_wrap=False
        )
        self.split_drop_area.files_dropped.connect(self.on_split_files_dropped)
        layout.addWidget(self.split_drop_area)

        file_row = QHBoxLayout()
        self.split_choose_button = QPushButton("파일 선택")
        self.split_choose_button.clicked.connect(self.open_split_file_dialog)
        file_row.addWidget(self.split_choose_button)
        self.split_file_label = QLabel("선택된 파일 없음")
        self.split_file_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        file_row.addWidget(self.split_file_label, 1)
        layout.addLayout(file_row)

        self.split_info_label = QLabel("")
        self.split_info_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.split_info_label.setStyleSheet("color: #43556a;")
        layout.addWidget(self.split_info_label)

        options = QHBoxLayout()
        options.addWidget(QLabel("구간 수"))
        self.split_count_spin = QSpinBox()
        self.split_count_spin.setRange(2, 999)
        self.split_count_spin.setValue(2)
        self.split_count_spin.valueChanged.connect(self._resize_split_table)
        options.addWidget(self.split_count_spin)
        self.split_snap_checkbox = QCheckBox("무음 지점으로 자동 보정  ±")
        self.split_snap_checkbox.setChecked(True)
        self.split_snap_checkbox.setToolTip(
            "입력한 시간 앞뒤로 가장 조용한 지점을 찾아 자릅니다.\n대사·소리가 이어지는 도중에 잘리는 것을 막아 줍니다."
        )
        options.addWidget(self.split_snap_checkbox)
        self.split_snap_spin = QDoubleSpinBox()
        self.split_snap_spin.setRange(0.1, 3.0)
        self.split_snap_spin.setSingleStep(0.1)
        self.split_snap_spin.setDecimals(1)
        self.split_snap_spin.setValue(DEFAULT_SILENCE_SEARCH_SECONDS)
        self.split_snap_spin.setSuffix(" 초 이내")
        self.split_snap_checkbox.toggled.connect(self.split_snap_spin.setEnabled)
        options.addWidget(self.split_snap_spin)
        options.addStretch(1)
        layout.addLayout(options)

        hint = QLabel("각 구간의 끝 시간만 입력 (예: 03:25, 12:34.50, 1:02:03) · 첫 구간은 00:00부터, 마지막 구간은 파일 끝까지")
        hint.setToolTip(
            "제목은 파일 이름과 태그 제목으로 쓰이며, 비워 두면 원래 파일 이름에 번호를 붙입니다.\n"
            "다시 인코딩하지 않아 음질과 표지 이미지가 그대로 유지됩니다."
        )
        hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        hint.setStyleSheet("color: #5b6b7b;")
        layout.addWidget(hint)

        self.split_table = QTableWidget(0, 2)
        self.split_table.setHorizontalHeaderLabels(["제목", "끝 시간"])
        self.split_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.split_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.split_table.setColumnWidth(1, 170)
        self.split_table.setAlternatingRowColors(True)
        self.split_table.setMinimumHeight(90)
        self.split_table.setToolTip("제목은 파일 이름과 태그 제목으로 쓰입니다. 비워 두면 원래 파일 이름에 번호를 붙입니다.")

        split_action_host = QWidget()
        action_row = QHBoxLayout(split_action_host)
        action_row.setContentsMargins(0, 0, 0, 0)
        self.split_start_button = QPushButton("분할 시작")
        self.split_start_button.clicked.connect(self.start_audio_split)
        action_row.addWidget(self.split_start_button)
        self.split_open_button = QPushButton("저장 폴더 열기")
        self.split_open_button.clicked.connect(self.open_split_output_dir)
        action_row.addWidget(self.split_open_button)
        self.split_output_label = QLabel("")
        self.split_output_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        action_row.addWidget(self.split_output_label, 1)

        self.split_progress = QProgressBar()
        self.split_progress.setRange(0, 100)
        self.split_progress.setValue(0)
        self.split_progress.setStyleSheet(GREEN_BAR_STYLE)

        self.split_log_view = QTextEdit()
        self.split_log_view.setReadOnly(True)
        self.split_log_view.setMinimumHeight(40)

        split_splitter = QSplitter(Qt.Vertical)
        split_splitter.addWidget(self.split_table)
        split_splitter.addWidget(self.split_log_view)
        split_splitter.setChildrenCollapsible(False)
        split_splitter.setStretchFactor(0, 3)
        split_splitter.setStretchFactor(1, 1)
        split_splitter.setSizes([300, 100])

        layout.addWidget(split_splitter, 1)
        layout.addWidget(split_action_host)
        layout.addWidget(self.split_progress)

        self._resize_split_table(self.split_count_spin.value())
        self._refresh_split_controls()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        scroll.viewport().setAutoFillBackground(False)
        content.setAutoFillBackground(False)
        return scroll

    def open_split_file_dialog(self) -> None:
        patterns = " ".join(f"*{extension}" for extension in sorted(SPLIT_SUPPORTED_EXTENSIONS))
        path, _ = QFileDialog.getOpenFileName(self, "나눌 음성 파일 선택", "", f"음성 파일 ({patterns})")
        if path:
            self.set_split_source(Path(path))

    def on_split_files_dropped(self, paths: list[str]) -> None:
        if len(paths) > 1:
            self.split_log_view.append("한 번에 한 파일만 나눌 수 있어 첫 번째 파일만 사용합니다.")
        self.set_split_source(Path(paths[0]))

    def set_split_source(self, path: Path) -> None:
        if self._split_worker is not None and self._split_worker.isRunning():
            return
        if path.suffix.lower() not in SPLIT_SUPPORTED_EXTENSIONS:
            QMessageBox.warning(self, APP_NAME, f"지원하지 않는 파일 형식입니다: {path.name}")
            return
        try:
            info = probe_audio(path)
        except AudioSplitError as exc:
            QMessageBox.warning(self, APP_NAME, f"{path.name}\n{exc}")
            return

        self._split_source = path
        self._split_duration = info.duration
        self._split_output_dir = default_output_dir(path)
        self.split_file_label.setText(str(path))
        self.split_file_label.setToolTip(str(path))
        details = [f"길이 {format_timestamp(info.duration, True)}", info.codec]
        if info.bit_rate:
            details.append(f"{info.bit_rate // 1000} kbps")
        if info.sample_rate:
            details.append(f"{info.sample_rate} Hz")
        details.append("표지 있음" if info.has_cover else "표지 없음")
        details.append(f"저장 형식 {info.output_suffix}")
        if info.has_video:
            details.append("영상은 빼고 음성만 저장")
        self.split_info_label.setText(" | ".join(details))
        self.split_output_label.setText(f"저장 위치: {self._split_output_dir}")
        self.split_output_label.setToolTip(str(self._split_output_dir))
        self.split_progress.setValue(0)
        self.split_log_view.append(f"파일 선택: {path}")
        self._resize_split_table(self.split_count_spin.value())
        self._refresh_split_controls()

    def _resize_split_table(self, count: int) -> None:
        table = self.split_table
        table.setRowCount(count)
        for row in range(count):
            for column in (0, 1):
                if table.item(row, column) is None:
                    table.setItem(row, column, QTableWidgetItem(""))
            end_item = table.item(row, 1)
            if row == count - 1:
                end_text = format_timestamp(self._split_duration, True) if self._split_duration is not None else ""
                end_item.setText(f"{end_text} (파일 끝)".strip())
                end_item.setFlags(end_item.flags() & ~Qt.ItemIsEditable)
                end_item.setForeground(QColor("#7a8794"))
                end_item.setData(Qt.UserRole, "auto")
            elif end_item.data(Qt.UserRole) == "auto":
                end_item.setText("")
                end_item.setFlags(end_item.flags() | Qt.ItemIsEditable)
                end_item.setData(Qt.ForegroundRole, None)
                end_item.setData(Qt.UserRole, None)
        table.setVerticalHeaderLabels([str(row) for row in range(1, count + 1)])

    def start_audio_split(self) -> None:
        if self._split_source is None or self._split_duration is None:
            QMessageBox.information(self, APP_NAME, "먼저 나눌 음성 파일을 선택하세요.")
            return
        count = self.split_count_spin.value()
        titles = [self.split_table.item(row, 0).text().strip() for row in range(count)]
        end_times: list[float] = []
        for row in range(count - 1):
            try:
                end_times.append(parse_timestamp(self.split_table.item(row, 1).text()))
            except ValueError as exc:
                self.split_table.setCurrentCell(row, 1)
                QMessageBox.warning(self, APP_NAME, f"{row + 1}번 구간 끝 시간: {exc}")
                return
        try:
            build_cut_points(end_times, self._split_duration)
        except AudioSplitError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return

        search = self.split_snap_spin.value() if self.split_snap_checkbox.isChecked() else 0.0
        job = AudioSplitJob(
            source_path=self._split_source,
            titles=titles,
            end_times=end_times,
            silence_search_seconds=search,
            output_dir=self._split_output_dir or default_output_dir(self._split_source),
        )
        self._split_worker = AudioSplitWorker(job, self)
        self._split_worker.progress_changed.connect(self.split_progress.setValue)
        self._split_worker.log_message.connect(self.split_log_view.append)
        self._split_worker.finished_success.connect(self.on_audio_split_finished)
        self._split_worker.failed.connect(self.on_audio_split_failed)
        self._split_worker.finished.connect(self._refresh_split_controls)
        self._split_worker.finished.connect(self._run_pending_shutdown)
        self.split_progress.setValue(0)
        self.split_log_view.append(f"분할 시작: {self._split_source.name} -> {count}개 구간")
        self._split_worker.start()
        self._refresh_split_controls()

    def on_audio_split_finished(self, paths: list[str]) -> None:
        self.split_log_view.append(f"완료: {len(paths)}개 파일 저장 -> {self._split_output_dir}")
        self.log(f"음성 분할 완료: {self._split_source} -> {len(paths)}개 파일")

    def on_audio_split_failed(self, message: str) -> None:
        self.split_log_view.append(f"실패: {message}")
        self.log(f"음성 분할 실패: {message}", logging.ERROR)
        QMessageBox.warning(self, APP_NAME, f"음성 분할에 실패했습니다.\n{message}")

    def open_split_output_dir(self) -> None:
        if self._split_output_dir is not None and self._split_output_dir.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._split_output_dir)))

    def _refresh_split_controls(self) -> None:
        running = self._split_worker is not None and self._split_worker.isRunning()
        for widget in (
            self.split_drop_area,
            self.split_choose_button,
            self.split_count_spin,
            self.split_table,
            self.split_snap_checkbox,
        ):
            widget.setEnabled(not running)
        self.split_snap_spin.setEnabled(not running and self.split_snap_checkbox.isChecked())
        self.split_start_button.setEnabled(not running and self._split_source is not None)
        self.split_open_button.setEnabled(self._split_output_dir is not None and self._split_output_dir.is_dir())

    def _build_settings_menu_button(self, title: str, menu: QMenu) -> QToolButton:
        button = QToolButton()
        button.setText(title)
        button.setPopupMode(QToolButton.InstantPopup)
        button.setToolButtonStyle(Qt.ToolButtonTextOnly)
        button.setMenu(menu)
        button.setStyleSheet(
            "QToolButton { padding: 6px 10px; border: 1px solid #c9d4df; border-radius: 8px; background: #ffffff; }"
            "QToolButton::menu-indicator { subcontrol-position: right center; }"
        )
        return button

    def _create_menu_field(self, label_text: str, widget: QWidget) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(8)
        label = QLabel(label_text)
        label.setMinimumWidth(112)
        layout.addWidget(label)
        widget.setMinimumWidth(170)
        layout.addWidget(widget, 1)
        return container

    def _add_widget_action(self, menu: QMenu, label_text: str, widget: QWidget) -> None:
        action = QWidgetAction(menu)
        action.setDefaultWidget(self._create_menu_field(label_text, widget))
        menu.addAction(action)

    def _create_stt_settings_menu(self) -> QMenu:
        menu = QMenu(self)
        runtime_menu = menu.addMenu("실행 / 모델")
        self._add_widget_action(runtime_menu, "실행 장치", self.runtime_combo)
        self._add_widget_action(runtime_menu, "Whisper 모델", self.model_combo)

        language_menu = menu.addMenu("언어")
        self._add_widget_action(language_menu, "입력 언어", self.input_language_combo)
        self._add_widget_action(language_menu, "출력 언어", self.output_language_combo)
        self._add_widget_action(language_menu, "번역", self.translate_checkbox)
        return menu

    def _create_performance_settings_menu(self) -> QMenu:
        menu = QMenu(self)
        self._add_widget_action(menu, "compute_type", self.compute_type_combo)
        self._add_widget_action(menu, "CPU threads", self.cpu_threads_spin)
        self._add_widget_action(menu, "num_workers", self.num_workers_spin)
        self._add_widget_action(menu, "RAM 사용량", self.memory_profile_combo)
        return menu

    def _create_process_settings_menu(self) -> QMenu:
        menu = QMenu(self)
        postprocess_menu = menu.addMenu("후처리")
        self._add_widget_action(postprocess_menu, "기본 후처리", self.postprocess_checkbox)
        self._add_widget_action(postprocess_menu, "강화 후처리", self.enhanced_postprocess_checkbox)

        vad_menu = menu.addMenu("VAD")
        self._add_widget_action(vad_menu, "VAD 필터", self.vad_mode_combo)
        self._add_widget_action(vad_menu, "최소 침묵(ms)", self.vad_min_silence_spin)
        self._add_widget_action(vad_menu, "패딩(ms)", self.vad_speech_pad_spin)

        finish_menu = menu.addMenu("완료 후 동작")
        self._add_widget_action(finish_menu, "모델 자동 해제", self.auto_unload_checkbox)
        self._add_widget_action(finish_menu, "2분 후 종료", self.shutdown_after_complete_checkbox)
        return menu

    def _wire_main_settings_summary_signals(self) -> None:
        self.runtime_combo.currentIndexChanged.connect(self.refresh_main_settings_summary)
        self.model_combo.currentIndexChanged.connect(self.refresh_main_settings_summary)
        self.input_language_combo.currentIndexChanged.connect(self.refresh_main_settings_summary)
        self.output_language_combo.currentIndexChanged.connect(self.refresh_main_settings_summary)
        self.translate_checkbox.toggled.connect(self.refresh_main_settings_summary)
        self.compute_type_combo.currentIndexChanged.connect(self.refresh_main_settings_summary)
        self.cpu_threads_spin.valueChanged.connect(self.refresh_main_settings_summary)
        self.num_workers_spin.valueChanged.connect(self.refresh_main_settings_summary)
        self.memory_profile_combo.currentIndexChanged.connect(self.refresh_main_settings_summary)
        self.auto_unload_checkbox.toggled.connect(self.refresh_main_settings_summary)
        self.shutdown_after_complete_checkbox.toggled.connect(self.refresh_main_settings_summary)
        self.vad_mode_combo.currentIndexChanged.connect(self.refresh_main_settings_summary)
        self.vad_min_silence_spin.valueChanged.connect(self.refresh_main_settings_summary)
        self.vad_speech_pad_spin.valueChanged.connect(self.refresh_main_settings_summary)
        self.postprocess_checkbox.toggled.connect(self.refresh_main_settings_summary)
        self.enhanced_postprocess_checkbox.toggled.connect(self.refresh_main_settings_summary)

    def refresh_main_settings_summary(self, *_args) -> None:
        if not hasattr(self, "main_settings_summary_label"):
            return
        summary = (
            f"실행 {self.runtime_combo.currentText()} | 모델 {self.model_combo.currentText()} | "
            f"{self.input_language_combo.currentText()} -> "
            f"{self.output_language_combo.currentText() if self.translate_checkbox.isChecked() else '번역 안 함'} | "
            f"compute {self.compute_type_combo.currentText()} | CPU {self.cpu_threads_spin.value()} | "
            f"workers {self.num_workers_spin.value()} | RAM {self.memory_profile_combo.currentText()} | "
            f"VAD {'ON' if self.is_vad_enabled() else 'OFF'} | "
            f"기본 후처리 {'ON' if self.postprocess_checkbox.isChecked() else 'OFF'} | "
            f"강화 후처리 {'ON' if self.enhanced_postprocess_checkbox.isChecked() else 'OFF'} | "
            f"완료 후 모델 해제 {'ON' if self.auto_unload_checkbox.isChecked() else 'OFF'} | "
            f"2분 후 종료 {'ON' if self.shutdown_after_complete_checkbox.isChecked() else 'OFF'}"
        )
        self.main_settings_summary_label.setText(summary)

    def _create_tray_icon(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        tray_menu = QMenu(self)

        show_action = QAction("Show Window", self)
        show_action.triggered.connect(self.restore_from_tray)
        tray_menu.addAction(show_action)

        hide_action = QAction("Hide to Tray", self)
        hide_action.triggered.connect(self.hide_to_tray)
        tray_menu.addAction(hide_action)

        tray_menu.addSeparator()

        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.quit_from_tray)
        tray_menu.addAction(quit_action)

        icon = self._app_icon or self.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.tray_icon = QSystemTrayIcon(icon, self)
        self.tray_icon.setToolTip(APP_NAME)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_activated)
        self.tray_icon.show()

    def hide_to_tray(self) -> None:
        if self.isMinimized():
            self.showNormal()
        self.hide()
        if self.tray_icon is not None and not self._tray_message_shown:
            self.tray_icon.showMessage(
                APP_NAME,
                "앱은 시스템 트레이에서 계속 실행됩니다.",
                QSystemTrayIcon.Information,
                3000,
            )
            self._tray_message_shown = True

    def restore_from_tray(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.Trigger:
            if self.isVisible():
                self.hide_to_tray()
            else:
                self.restore_from_tray()

    def quit_from_tray(self) -> None:
        self._allow_close = True
        self._stop_workers_for_exit()
        if self._shutdown_scheduled:
            self._cancel_scheduled_shutdown()
        if self.tray_icon is not None:
            self.tray_icon.hide()
            self.tray_icon.deleteLater()
            self.tray_icon = None
        self.close()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)

    def _build_subtitle_translate_tab(self) -> QWidget:
        content = QWidget()
        root_layout = QVBoxLayout(content)
        root_layout.setSpacing(10)

        info_row = QHBoxLayout()
        self.subtitle_translate_info_label = QLabel("입력 언어와 출력 언어는 작업 탭의 현재 설정을 사용합니다.")
        self.subtitle_translate_info_label.setStyleSheet("color: #43556a;")
        info_row.addWidget(self.subtitle_translate_info_label)
        info_row.addStretch(1)
        root_layout.addLayout(info_row)

        button_row = QHBoxLayout()
        self.subtitle_add_button = QPushButton("자막 파일 추가")
        self.subtitle_add_button.clicked.connect(self.open_subtitle_file_dialog)
        button_row.addWidget(self.subtitle_add_button)

        self.subtitle_add_folder_button = QPushButton("자막 폴더 추가")
        self.subtitle_add_folder_button.clicked.connect(self.open_subtitle_folder_dialog)
        button_row.addWidget(self.subtitle_add_folder_button)

        self.subtitle_include_subdirs_checkbox = QCheckBox("하위 폴더 포함")
        self.subtitle_include_subdirs_checkbox.setChecked(True)
        button_row.addWidget(self.subtitle_include_subdirs_checkbox)

        self.subtitle_remove_button = QPushButton("선택 제거")
        self.subtitle_remove_button.clicked.connect(self.remove_selected_subtitle_files)
        button_row.addWidget(self.subtitle_remove_button)

        self.subtitle_start_button = QPushButton("번역 시작")
        self.subtitle_start_button.clicked.connect(self.start_subtitle_translation)
        button_row.addWidget(self.subtitle_start_button)
        button_row.addStretch(1)
        root_layout.addLayout(button_row)

        splitter = QSplitter(Qt.Vertical)
        root_layout.addWidget(splitter, 1)

        self.subtitle_drop_area = DropArea("여기에 .srt / .vtt / .txt 파일을 드래그 앤 드롭")
        self.subtitle_drop_area.files_dropped.connect(self.add_subtitle_files)

        self.subtitle_table = DropTableWidget(0, 3)
        self.subtitle_table.files_dropped.connect(self.add_subtitle_files)
        self.subtitle_table.setHorizontalHeaderLabels(["파일명", "전체 경로", "상태"])
        self.subtitle_table.horizontalHeader().setStretchLastSection(False)
        self.subtitle_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.subtitle_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.subtitle_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.subtitle_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.subtitle_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.subtitle_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.subtitle_table.setAlternatingRowColors(True)
        self.subtitle_table.setTextElideMode(Qt.ElideMiddle)
        self.subtitle_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.subtitle_table.setShowGrid(False)

        self.subtitle_list_stack_host = QWidget()
        self.subtitle_list_stack = QStackedLayout(self.subtitle_list_stack_host)
        self.subtitle_list_stack.setContentsMargins(0, 0, 0, 0)
        self.subtitle_list_stack.addWidget(self.subtitle_drop_area)
        self.subtitle_list_stack.addWidget(self.subtitle_table)
        self.subtitle_list_stack.setCurrentWidget(self.subtitle_drop_area)
        splitter.addWidget(self.subtitle_list_stack_host)

        self.subtitle_log_view = QTextEdit()
        self.subtitle_log_view.setReadOnly(True)
        splitter.addWidget(self.subtitle_log_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.subtitle_current_file_label = QLabel("현재 파일: 없음")
        root_layout.addWidget(self.subtitle_current_file_label)

        self.subtitle_stage_label = QLabel("현재 단계: 대기 중")
        root_layout.addWidget(self.subtitle_stage_label)

        self.subtitle_translation_progress_label = QLabel("번역 진행: 대기 중")
        root_layout.addWidget(self.subtitle_translation_progress_label)

        self.subtitle_translation_progress = QProgressBar()
        self.subtitle_translation_progress.setRange(0, 100)
        self.subtitle_translation_progress.setValue(0)
        self.subtitle_translation_progress.setStyleSheet(GREEN_BAR_STYLE)
        root_layout.addWidget(self.subtitle_translation_progress)

        self.subtitle_queue_progress_label = QLabel("전체 대기열 진행률 0 / 0")
        root_layout.addWidget(self.subtitle_queue_progress_label)

        self.subtitle_queue_progress = QProgressBar()
        self.subtitle_queue_progress.setRange(0, 100)
        self.subtitle_queue_progress.setValue(0)
        self.subtitle_queue_progress.setStyleSheet(GREEN_BAR_STYLE)
        root_layout.addWidget(self.subtitle_queue_progress)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(content)

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)
        return page

    def _build_translation_tab(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(10)

        save_row = QHBoxLayout()
        self.translation_settings_status = QLabel("")
        self.translation_settings_status.setStyleSheet("color: #2f6f3e;")
        save_row.addWidget(self.translation_settings_status)
        save_row.addStretch(1)
        self.save_translation_button = QPushButton("저장")
        self.save_translation_button.setEnabled(False)
        self.save_translation_button.clicked.connect(self.save_translation_inputs)
        save_row.addWidget(self.save_translation_button)
        layout.addLayout(save_row)

        api_box = QGroupBox("API 키")
        api_layout = QVBoxLayout(api_box)
        api_layout.addWidget(QLabel("Gemini 키는 여러 개를 한 줄에 하나씩 입력하면 자동으로 순환 사용합니다."))
        api_layout.addWidget(QLabel("Gemini가 끝까지 실패하면 저장된 DeepL Free API 키로 자동 폴백합니다."))
        api_layout.addWidget(QLabel("Gemini API 키"))
        self.keys_edit = QPlainTextEdit(self._saved_api_keys_text)
        self.keys_edit.setPlaceholderText("AIza...")
        self.keys_edit.setMinimumHeight(140)
        self.keys_edit.textChanged.connect(self.mark_translation_inputs_dirty)
        api_layout.addWidget(self.keys_edit)

        api_layout.addWidget(QLabel("DeepL Free API 키"))
        self.deepl_key_edit = QLineEdit(self._saved_deepl_api_key)
        self.deepl_key_edit.setPlaceholderText("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx:fx")
        self.deepl_key_edit.setEchoMode(QLineEdit.PasswordEchoOnEdit)
        self.deepl_key_edit.textChanged.connect(self.mark_translation_inputs_dirty)
        api_layout.addWidget(self.deepl_key_edit)
        layout.addWidget(CollapsibleSection("API Keys", api_box, expanded=True))

        option_box = QGroupBox("Gemini 번역 옵션")
        form = QFormLayout(option_box)

        self.model_edit = QLineEdit(self._translator_settings.preferred_model)
        self.model_edit.setPlaceholderText("비우면 자동 모델 순서 사용")
        self.model_edit.textChanged.connect(self.mark_translation_inputs_dirty)
        form.addRow("선호 모델", self.model_edit)

        self.use_deepl_fallback_checkbox = QCheckBox("Gemini 실패 시 DeepL Free API 사용")
        self.use_deepl_fallback_checkbox.setChecked(self._translator_settings.use_deepl_fallback)
        self.use_deepl_fallback_checkbox.toggled.connect(self.mark_translation_inputs_dirty)
        form.addRow("실패 폴백", self.use_deepl_fallback_checkbox)

        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(10, 200)
        self.chunk_size_spin.setValue(self._translator_settings.chunk_size)
        self.chunk_size_spin.valueChanged.connect(self.mark_translation_inputs_dirty)
        form.addRow("청크 크기", self.chunk_size_spin)

        self.request_delay_spin = QDoubleSpinBox()
        self.request_delay_spin.setRange(0.0, 30.0)
        self.request_delay_spin.setSingleStep(0.5)
        self.request_delay_spin.setValue(self._translator_settings.request_delay_seconds)
        self.request_delay_spin.valueChanged.connect(self.mark_translation_inputs_dirty)
        form.addRow("API 요청 간 지연(초)", self.request_delay_spin)

        self.reasoning_combo = QComboBox()
        self.reasoning_combo.addItems(["minimal", "low", "medium", "high"])
        self.reasoning_combo.setCurrentText(self._translator_settings.reasoning_level)
        self.reasoning_combo.currentTextChanged.connect(self.mark_translation_inputs_dirty)
        form.addRow("추론 레벨", self.reasoning_combo)

        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(self._translator_settings.temperature)
        self.temperature_spin.valueChanged.connect(self.mark_translation_inputs_dirty)
        form.addRow("온도", self.temperature_spin)

        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.05)
        self.top_p_spin.setValue(self._translator_settings.top_p)
        self.top_p_spin.valueChanged.connect(self.mark_translation_inputs_dirty)
        form.addRow("Top-P", self.top_p_spin)
        layout.addWidget(CollapsibleSection("Gemini Translation Options", option_box, expanded=True))

        prompt_box = QGroupBox("프롬프트")
        prompt_layout = QVBoxLayout(prompt_box)
        prompt_layout.addWidget(QLabel("시스템 프롬프트"))
        self.system_prompt_edit = QPlainTextEdit(self._translator_settings.system_prompt)
        self.system_prompt_edit.setMinimumHeight(240)
        self.system_prompt_edit.textChanged.connect(self.mark_translation_inputs_dirty)
        prompt_layout.addWidget(self.system_prompt_edit)

        prompt_layout.addWidget(QLabel("번역 노트"))
        self.note_edit = QPlainTextEdit(self._translator_settings.translation_note)
        self.note_edit.setMinimumHeight(140)
        self.note_edit.textChanged.connect(self.mark_translation_inputs_dirty)
        prompt_layout.addWidget(self.note_edit)
        layout.addWidget(CollapsibleSection("Prompts", prompt_box, expanded=True), 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidget(content)

        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)
        return page

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._allow_close:
            if self.tray_icon is not None:
                self.tray_icon.hide()
            super().closeEvent(event)
            return

        if self.tray_icon is not None:
            self.hide_to_tray()
            event.ignore()
            return

        if self.is_busy():
            QMessageBox.warning(self, APP_NAME, "다운로드 또는 작업이 진행 중입니다.")
            event.ignore()
            return
        super().closeEvent(event)
        if event.isAccepted():
            # 트레이가 없으면 창을 닫는 것이 곧 종료다(QuitOnLastWindowClosed가 꺼져 있어 직접 끝낸다).
            app = QApplication.instance()
            if app is not None:
                app.quit()

    def is_busy(self) -> bool:
        return (
            (self._download_worker is not None and self._download_worker.isRunning())
            or (self._pipeline_worker is not None and self._pipeline_worker.isRunning())
            or (self._subtitle_translation_worker is not None and self._subtitle_translation_worker.isRunning())
            or (self._split_worker is not None and self._split_worker.isRunning())
        )

    def _stop_workers_for_exit(self) -> None:
        if self._split_worker is not None and self._split_worker.isRunning():
            # 분할 작업은 취소 요청을 확인하고 만들던 파일을 지운 뒤 스스로 끝난다.
            self._split_worker.requestInterruption()
            if not self._split_worker.wait(10000):
                self._split_worker.terminate()
                self._split_worker.wait(2000)
        if self._pipeline_worker is not None and self._pipeline_worker.isRunning():
            self._pipeline_worker.request_stop()
        for worker in (self._download_worker, self._pipeline_worker, self._subtitle_translation_worker):
            if worker is None or not worker.isRunning():
                continue
            worker.requestInterruption()
            # 작업 스레드는 진행률 콜백·파일 사이에서 중단 요청을 확인하고 스스로 끝난다.
            # 모델 다운로드처럼 중간에 멈출 수 없는 경우에만 마지막 수단으로 강제 종료한다.
            if not worker.wait(8000):
                worker.terminate()
                worker.wait(2000)

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

    def _populate_compute_type_choices(self) -> None:
        current_value = ""
        if hasattr(self, "compute_type_combo"):
            current_value = str(self.compute_type_combo.currentData() or "auto")
        self.compute_type_combo.clear()
        selected_index = 0
        for index, (value, label) in enumerate(get_compute_type_choices(self.current_runtime_device())):
            self.compute_type_combo.addItem(label, value)
            if value == current_value:
                selected_index = index
        self.compute_type_combo.setCurrentIndex(selected_index)

    def on_runtime_changed(self) -> None:
        self._populate_compute_type_choices()
        self.refresh_model_status()
        self.refresh_main_settings_summary()
        self.log(f"실행 장치 선택: {self.runtime_combo.currentText()}")

    def on_enhanced_postprocess_changed(self, checked: bool) -> None:
        if checked and not self.postprocess_checkbox.isChecked():
            self.postprocess_checkbox.setChecked(True)

    def on_postprocess_changed(self, checked: bool) -> None:
        if not checked and self.enhanced_postprocess_checkbox.isChecked():
            self.enhanced_postprocess_checkbox.setChecked(False)

    def on_vad_changed(self, _index: int) -> None:
        vad_enabled = self.is_vad_enabled()
        self.vad_min_silence_spin.setEnabled(vad_enabled)
        self.vad_speech_pad_spin.setEnabled(vad_enabled)

    def on_shutdown_after_complete_changed(self, checked: bool) -> None:
        if not checked and self._shutdown_scheduled:
            self._cancel_scheduled_shutdown()

    def current_runtime_device(self) -> str:
        return str(self.runtime_combo.currentData() or "cpu")

    def current_model_key(self) -> str:
        return str(self.model_combo.currentData() or DEFAULT_MODEL_KEY)

    def current_input_language(self) -> str:
        return str(self.input_language_combo.currentData() or DEFAULT_INPUT_LANGUAGE)

    def current_vad_settings(self) -> VADSettings:
        return VADSettings(
            enabled=self.is_vad_enabled(),
            min_silence_duration_ms=self.vad_min_silence_spin.value(),
            speech_pad_ms=self.vad_speech_pad_spin.value(),
        )

    def is_vad_enabled(self) -> bool:
        return bool(self.vad_mode_combo.currentData())

    def current_runtime_tuning(self) -> RuntimeTuningOptions:
        cpu_threads = self.cpu_threads_spin.value() if self.current_runtime_device() == "cpu" else None
        return RuntimeTuningOptions(
            compute_type=str(self.compute_type_combo.currentData() or "auto"),
            cpu_threads=cpu_threads,
            num_workers=self.num_workers_spin.value(),
            memory_profile=str(self.memory_profile_combo.currentData() or "unlimited"),
            auto_unload_after_job=self.auto_unload_checkbox.isChecked(),
        )

    def on_model_changed(self) -> None:
        self.refresh_model_status()
        self.refresh_main_settings_summary()

    def collect_translation_inputs(self) -> TranslatorSettings:
        return TranslatorSettings(
            preferred_model=self.model_edit.text().strip(),
            target_language=str(self.output_language_combo.currentData() or "ko"),
            use_deepl_fallback=self.use_deepl_fallback_checkbox.isChecked(),
            chunk_size=self.chunk_size_spin.value(),
            request_delay_seconds=self.request_delay_spin.value(),
            temperature=self.temperature_spin.value(),
            top_p=self.top_p_spin.value(),
            reasoning_level=self.reasoning_combo.currentText(),
            system_prompt=self.system_prompt_edit.toPlainText(),
            translation_note=self.note_edit.toPlainText(),
        )

    def translator_settings(self) -> TranslatorSettings:
        return self._translator_settings

    def _job_translator_settings(self) -> TranslatorSettings:
        # 출력 언어는 작업 설정 메뉴에서 바로 바꿀 수 있으므로, 저장 여부와 관계없이 현재 선택값을 쓴다.
        target = str(self.output_language_combo.currentData() or self._translator_settings.target_language)
        return dataclasses.replace(self._translator_settings, target_language=target)

    def mark_translation_inputs_dirty(self) -> None:
        self._settings_dirty = True
        self.sync_translation_save_state()
        self.translation_settings_status.setText("저장되지 않은 변경사항")

    def save_translation_inputs(self) -> None:
        keys_text = self.keys_edit.toPlainText()
        deepl_key = self.deepl_key_edit.text().strip()
        settings = self.collect_translation_inputs()
        # 디스크·자격 증명 저장이 모두 성공한 뒤에만 메모리 값을 바꾼다(일부만 반영되는 상태 방지).
        try:
            save_translator_settings(settings)
            save_api_keys(keys_text)
            save_deepl_api_key(deepl_key)
        except CredentialStoreError as exc:
            self.log(f"API 키 보안 저장 실패: {exc}", logging.ERROR)
            QMessageBox.warning(self, APP_NAME, f"API 키를 안전하게 저장하지 못했습니다.\n\n{exc}")
            return
        except OSError as exc:
            self.log(f"번역 설정 저장 실패: {exc}", logging.ERROR)
            QMessageBox.warning(self, APP_NAME, f"번역 설정 파일을 저장하지 못했습니다.\n\n{exc}")
            return
        self._saved_api_keys_text = keys_text
        self._saved_deepl_api_key = deepl_key
        self._translator_settings = settings
        self._settings_dirty = False
        self.sync_translation_save_state()
        self.translation_settings_status.setText("설정이 저장되었습니다")
        self.log("번역 설정이 저장되었습니다.")

    def update_file_area_mode(self) -> None:
        if getattr(self, "list_stack", None) is None:
            return
        target = self.drop_area if self.table.rowCount() == 0 else self.table
        self.list_stack.setCurrentWidget(target)

    def update_row_appearance(self, row: int, status: str) -> None:
        color = None
        if status in {STATUS_TRANSCRIBING, STATUS_TRANSLATING, STATUS_SAVING}:
            color = QColor("#e8f2ff")
        elif status == STATUS_DONE:
            color = QColor("#e7f7ea")
        elif status == STATUS_FAILED:
            color = QColor("#fdeaea")
        elif status == STATUS_SKIPPED:
            color = QColor("#f7f3e8")

        for column in range(self.table.columnCount()):
            item = self.table.item(row, column)
            if item is None:
                continue
            item.setBackground(color if color is not None else QColor("#ffffff"))

    def sync_translation_save_state(self) -> None:
        if not hasattr(self, "save_translation_button"):
            return
        self.save_translation_button.setEnabled(self._settings_dirty)

    def _append_ui_log(self, formatted_message: str) -> None:
        self.log_view.append(formatted_message)

    def log(self, message: str, level: int = logging.INFO) -> None:
        self._logger.log(level, message)

    def refresh_model_status(self) -> None:
        model_key = self.current_model_key()
        self.subtitle_label.setText(f"선택 STT 설정: {get_model_label(model_key)}")
        self._model_ready = is_model_ready(model_key=model_key)
        self.model_runtime_state_label.setText(
            f"메모리 상태: {describe_loaded_model_state(model_key, self.current_runtime_device())}"
        )
        if self._model_ready:
            model_dir = get_model_dir(model_key)
            self.model_status_label.setText(f"모델 준비 완료: {model_dir}")
            self.model_progress.setValue(100)
            self.log(f"모델 캐시 사용: {model_dir}")
        else:
            self.model_status_label.setText(f"모델 다운로드 필요: {get_model_label(model_key)}")
            self.model_progress.setValue(0)
            if self._auto_download_on_startup:
                self.log(f"선택 모델 다운로드 시작: {get_model_label(model_key)}")
                self.start_model_download()
        self.update_controls()

    def update_controls(self) -> None:
        has_pending = any(item.status in {STATUS_PENDING, STATUS_FAILED} for item in self._items.values())
        has_rows = self.table.rowCount() > 0
        has_subtitle_pending = any(item.status in {STATUS_PENDING, STATUS_FAILED} for item in self._subtitle_items.values())
        has_subtitle_rows = self.subtitle_table.rowCount() > 0 if hasattr(self, "subtitle_table") else False
        download_running = self._download_worker is not None and self._download_worker.isRunning()
        processing_running = self._pipeline_worker is not None and self._pipeline_worker.isRunning()
        subtitle_processing_running = self._subtitle_translation_worker is not None and self._subtitle_translation_worker.isRunning()
        enabled = not download_running and not processing_running and not subtitle_processing_running

        self.add_button.setEnabled(True)
        self.add_folder_button.setEnabled(True)
        self.include_subdirs_checkbox.setEnabled(True)
        self.drop_area.setEnabled(True)
        self.remove_button.setEnabled(has_rows)
        self.runtime_combo.setEnabled(enabled)
        self.model_combo.setEnabled(enabled)
        self.input_language_combo.setEnabled(enabled)
        self.output_language_combo.setEnabled(True)
        self.compute_type_combo.setEnabled(enabled)
        self.cpu_threads_spin.setEnabled(enabled and self.current_runtime_device() == "cpu")
        self.num_workers_spin.setEnabled(enabled)
        self.memory_profile_combo.setEnabled(enabled)
        self.auto_unload_checkbox.setEnabled(True)
        self.shutdown_after_complete_checkbox.setEnabled(True)
        self.vad_mode_combo.setEnabled(enabled)
        self.vad_min_silence_spin.setEnabled(enabled and self.is_vad_enabled())
        self.vad_speech_pad_spin.setEnabled(enabled and self.is_vad_enabled())
        self.postprocess_checkbox.setEnabled(enabled)
        self.enhanced_postprocess_checkbox.setEnabled(enabled)
        self.translate_checkbox.setEnabled(enabled)
        self.start_button.setEnabled(self._model_ready and has_pending and enabled)
        self.pause_button.setEnabled(processing_running and not getattr(self._pipeline_worker, "_pause_requested", False))
        self.resume_button.setEnabled(processing_running and getattr(self._pipeline_worker, "_pause_requested", False))
        self.subtitle_add_button.setEnabled(True)
        self.subtitle_add_folder_button.setEnabled(True)
        self.subtitle_include_subdirs_checkbox.setEnabled(True)
        self.subtitle_drop_area.setEnabled(True)
        self.subtitle_remove_button.setEnabled(has_subtitle_rows)
        self.subtitle_start_button.setEnabled(has_subtitle_pending and enabled)
        self.retry_download_button.setEnabled(not download_running and not self._model_ready)
        self.unload_model_button.setEnabled(not processing_running)
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(2, True)
        self.sync_translation_save_state()

    def open_file_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "음성/영상 파일 선택",
            "",
            "Audio / Video Files (*.mp3 *.mp4)",
        )
        if files:
            self.add_files(files)

    def open_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if folder:
            self.add_files([folder])

    def add_files(self, paths: list[str]) -> None:
        items, errors = normalize_input_files(paths, include_subdirs=self.include_subdirs_checkbox.isChecked())
        for error in errors:
            self.log(error)

        added_count = 0
        added_paths: list[str] = []
        for item in items:
            key = str(item.source_path)
            if key in self._items:
                self.log(f"이미 목록에 존재함: {item.source_path}")
                continue

            row = self.table.rowCount()
            self.table.insertRow(row)
            name_item = QTableWidgetItem(item.source_path.name)
            name_item.setToolTip(str(item.source_path))
            path_item = QTableWidgetItem(str(item.source_path))
            path_item.setToolTip(str(item.source_path))
            status_item = QTableWidgetItem(item.status)
            status_item.setToolTip(item.status)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, path_item)
            self.table.setItem(row, 2, status_item)
            self.update_row_appearance(row, item.status)
            self._items[key] = item
            self._rows_by_path[key] = row
            added_count += 1
            added_paths.append(key)

        if added_count:
            self.log(f"파일 {added_count}개 추가")
            if self._processing and self._pipeline_worker is not None and self._pipeline_worker.isRunning():
                if self._pipeline_worker.add_paths(added_paths):
                    self.log("진행 중 추가된 파일은 현재 작업에 이어서 처리됩니다.")
                else:
                    self.log("현재 작업이 마무리되는 중이라 추가한 파일은 다음에 '시작'을 누르면 처리됩니다.")
            self.refresh_queue_progress()
            self.update_file_area_mode()
        self.update_controls()

    def open_subtitle_file_dialog(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "자막 파일 선택",
            "",
            "Subtitle Files (*.srt *.vtt *.txt)",
        )
        if files:
            self.add_subtitle_files(files)

    def open_subtitle_folder_dialog(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "자막 폴더 선택")
        if folder:
            self.add_subtitle_files([folder])

    def add_subtitle_files(self, paths: list[str]) -> None:
        items, errors = normalize_translation_files(paths, include_subdirs=self.subtitle_include_subdirs_checkbox.isChecked())
        for error in errors:
            self.log(error)
            self.subtitle_log_view.append(error)

        added_count = 0
        added_paths: list[str] = []
        for item in items:
            key = str(item.source_path)
            if key in self._subtitle_items:
                self.subtitle_log_view.append(f"이미 목록에 존재함: {item.source_path}")
                continue

            row = self.subtitle_table.rowCount()
            self.subtitle_table.insertRow(row)
            name_item = QTableWidgetItem(item.source_path.name)
            name_item.setToolTip(str(item.source_path))
            path_item = QTableWidgetItem(str(item.source_path))
            path_item.setToolTip(str(item.source_path))
            status_item = QTableWidgetItem(item.status)
            status_item.setToolTip(item.status)
            self.subtitle_table.setItem(row, 0, name_item)
            self.subtitle_table.setItem(row, 1, path_item)
            self.subtitle_table.setItem(row, 2, status_item)
            self.update_subtitle_row_appearance(row, item.status)
            self._subtitle_items[key] = item
            self._subtitle_rows_by_path[key] = row
            added_count += 1
            added_paths.append(key)

        if added_count:
            self.subtitle_log_view.append(f"자막 파일 {added_count}개 추가")
            if self._subtitle_translation_processing and self._subtitle_translation_worker is not None and self._subtitle_translation_worker.isRunning():
                if self._subtitle_translation_worker.add_paths(added_paths):
                    self.subtitle_log_view.append("진행 중 추가된 파일은 현재 작업 뒤에 이어서 처리됩니다.")
                else:
                    self.subtitle_log_view.append("현재 작업이 마무리되는 중이라 추가한 파일은 다음에 시작하면 처리됩니다.")
            self.refresh_subtitle_queue_progress()
            self.update_subtitle_file_area_mode()
        self.update_controls()

    def update_subtitle_file_area_mode(self) -> None:
        if getattr(self, "subtitle_list_stack", None) is None:
            return
        target = self.subtitle_drop_area if self.subtitle_table.rowCount() == 0 else self.subtitle_table
        self.subtitle_list_stack.setCurrentWidget(target)

    def update_subtitle_row_appearance(self, row: int, status: str) -> None:
        color = None
        if status in {STATUS_TRANSLATING, STATUS_SAVING}:
            color = QColor("#e8f2ff")
        elif status == STATUS_DONE:
            color = QColor("#e7f7ea")
        elif status == STATUS_FAILED:
            color = QColor("#fdeaea")
        elif status == STATUS_SKIPPED:
            color = QColor("#f7f3e8")

        for column in range(self.subtitle_table.columnCount()):
            item = self.subtitle_table.item(row, column)
            if item is None:
                continue
            item.setBackground(color if color is not None else QColor("#ffffff"))

    def refresh_subtitle_queue_progress(self) -> None:
        if self._subtitle_queue_total > 0:
            total = self._subtitle_queue_total
            finished = self._subtitle_queue_processed
        else:
            total = len(self._subtitle_items)
            finished = sum(
                1 for item in self._subtitle_items.values() if item.status in {STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED}
            )
        percent = 100 if total == 0 else int(finished * 100 / total)
        self.subtitle_queue_progress.setValue(percent)
        self.subtitle_queue_progress_label.setText(f"전체 대기열 진행률 {finished} / {total}")

    def _rebuild_subtitle_row_index(self) -> None:
        self._subtitle_rows_by_path.clear()
        for row in range(self.subtitle_table.rowCount()):
            path_item = self.subtitle_table.item(row, 1)
            if path_item is not None:
                self._subtitle_rows_by_path[path_item.text()] = row

    def _remove_subtitle_paths(self, paths: list[str]) -> int:
        rows_to_remove: set[int] = set()
        for source_path in paths:
            row = self._subtitle_rows_by_path.get(source_path)
            if row is not None:
                rows_to_remove.add(row)

        if not rows_to_remove:
            return 0

        for row in sorted(rows_to_remove, reverse=True):
            path_item = self.subtitle_table.item(row, 1)
            if path_item is not None:
                self._subtitle_items.pop(path_item.text(), None)
            self.subtitle_table.removeRow(row)

        self._rebuild_subtitle_row_index()
        self.refresh_subtitle_queue_progress()
        self.update_subtitle_file_area_mode()
        self.update_controls()
        return len(rows_to_remove)

    def remove_selected_subtitle_files(self) -> None:
        selection_model = self.subtitle_table.selectionModel()
        if selection_model is None:
            return
        selected_rows = selection_model.selectedRows()
        if not selected_rows:
            self.subtitle_log_view.append("제거할 자막 파일을 선택하지 않았습니다.")
            return

        removable_paths: list[str] = []
        for index in selected_rows:
            path_item = self.subtitle_table.item(index.row(), 1)
            if path_item is not None:
                removable_paths.append(path_item.text())
        removed_count = 0
        blocked_count = 0
        if self._subtitle_translation_worker is not None and self._subtitle_translation_worker.isRunning():
            removed_paths, blocked_paths = self._subtitle_translation_worker.remove_paths(removable_paths)
            blocked_count = len(blocked_paths)
            finished_paths = [path for path in removable_paths if path not in removed_paths and path not in blocked_paths]
            removed_count = len(set(removed_paths)) + self._remove_subtitle_paths(finished_paths + removed_paths)
        else:
            removed_count = self._remove_subtitle_paths(removable_paths)
        if removed_count:
            self.subtitle_log_view.append(f"선택한 자막 파일 {removed_count}개를 목록에서 제거했습니다.")
        if blocked_count:
            self.subtitle_log_view.append("현재 처리 중인 자막 파일은 즉시 제거할 수 없습니다.")

    def update_subtitle_item_status(self, source_path: str, status: str, message: str = "") -> None:
        item = self._subtitle_items.get(source_path)
        row = self._subtitle_rows_by_path.get(source_path)
        if item is None or row is None:
            return

        item.status = status
        item.message = message
        status_text = status if not message else f"{status} | {message}"
        status_item = self.subtitle_table.item(row, 2)
        if status_item is None:
            status_item = QTableWidgetItem()
            self.subtitle_table.setItem(row, 2, status_item)
        status_item.setText(status_text)
        status_item.setToolTip(status_text)
        self.update_subtitle_row_appearance(row, status)

        if status == STATUS_DONE:
            removed_count = self._remove_subtitle_paths([source_path])
            if removed_count:
                self.subtitle_log_view.append(f"완료 파일 자동 제거: {source_path}")
        elif status == STATUS_REMOVED:
            self._remove_subtitle_paths([source_path])

    def _check_translation_keys(self, api_keys: list[str], deepl_api_key: str) -> bool:
        use_deepl = self.translator_settings().use_deepl_fallback
        if use_deepl and not deepl_api_key:
            self.tabs.setCurrentIndex(2)
            QMessageBox.warning(self, APP_NAME, "DeepL 폴백을 사용하려면 DeepL Free API 키를 입력해 주세요.")
            return False
        if not api_keys and not use_deepl:
            self.tabs.setCurrentIndex(2)
            QMessageBox.warning(
                self,
                APP_NAME,
                "번역 설정 탭에 Gemini API 키를 입력해 주세요.\n번역 없이 자막만 만들려면 작업 설정의 '번역 사용'을 끄세요.",
            )
            return False
        return True

    def start_subtitle_translation(self) -> None:
        api_keys = load_api_keys()
        deepl_api_key = load_deepl_api_key()
        if not self._check_translation_keys(api_keys, deepl_api_key):
            return

        source_paths = [path for path, item in self._subtitle_items.items() if item.status in {STATUS_PENDING, STATUS_FAILED}]
        if not source_paths:
            self.subtitle_log_view.append("번역할 자막 파일이 없습니다.")
            return

        for source_path in source_paths:
            self.update_subtitle_item_status(source_path, STATUS_PENDING)

        self._subtitle_translation_processing = True
        self._subtitle_success_count = 0
        self._subtitle_failure_count = 0
        self._subtitle_skipped_count = 0
        self._subtitle_queue_total = len(source_paths)
        self._subtitle_queue_processed = 0
        self.subtitle_current_file_label.setText("현재 파일: 없음")
        self.subtitle_stage_label.setText("현재 단계: 작업 준비")
        self.subtitle_translation_progress.setValue(0)
        self.subtitle_translation_progress_label.setText("번역 진행: 대기 중")

        job = SubtitleTranslationJob(
            source_paths=source_paths,
            source_language=self.current_input_language(),
            translator_settings=self._job_translator_settings(),
            api_keys=api_keys,
            deepl_api_key=deepl_api_key,
        )
        self._subtitle_translation_worker = SubtitleTranslationWorker(job, self)
        self._subtitle_translation_worker.item_status_changed.connect(self.update_subtitle_item_status)
        self._subtitle_translation_worker.log_message.connect(self.subtitle_log_view.append)
        self._subtitle_translation_worker.queue_progress_changed.connect(self.on_subtitle_queue_progress_changed)
        self._subtitle_translation_worker.file_started.connect(self.on_subtitle_file_started)
        self._subtitle_translation_worker.stage_changed.connect(self.on_subtitle_stage_changed)
        self._subtitle_translation_worker.translation_progress_changed.connect(self.on_subtitle_translation_progress_changed)
        self._subtitle_translation_worker.summary_ready.connect(self.on_subtitle_summary_ready)
        self._subtitle_translation_worker.failed.connect(self.on_subtitle_failed)
        self._subtitle_translation_worker.finished.connect(self._run_pending_shutdown)
        self._subtitle_translation_worker.start()
        self.update_controls()
        self.subtitle_log_view.append(
            f"자막 번역 시작: {len(source_paths)}개 파일 | 입력 언어={self.input_language_combo.currentText()} | "
            f"출력 언어={self.output_language_combo.currentText()}"
        )

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
        self.update_file_area_mode()
        self.update_controls()
        return len(rows_to_remove)

    def remove_selected_files(self) -> None:
        selection_model = self.table.selectionModel()
        if selection_model is None:
            return
        selected_rows = selection_model.selectedRows()
        if not selected_rows:
            self.log("제거할 파일을 선택하지 않았습니다.")
            return

        removable_paths: list[str] = []
        for index in selected_rows:
            path_item = self.table.item(index.row(), 1)
            if path_item is not None:
                removable_paths.append(path_item.text())
        removed_count = 0
        blocked_count = 0
        if self._pipeline_worker is not None and self._pipeline_worker.isRunning():
            removed_paths, blocked_paths = self._pipeline_worker.remove_paths(removable_paths)
            blocked_count = len(blocked_paths)
            # 대기열에 없는(이미 끝난·실패·스킵) 행은 작업과 무관하므로 바로 지운다.
            finished_paths = [path for path in removable_paths if path not in removed_paths and path not in blocked_paths]
            removed_count = len(set(removed_paths)) + self._remove_paths(finished_paths + removed_paths)
        else:
            removed_count = self._remove_paths(removable_paths)
        if removed_count:
            self.log(f"선택한 파일 {removed_count}개를 목록에서 제거했습니다.")
        if blocked_count:
            self.log("현재 처리중인 파일은 즉시 제거할 수 없습니다.")

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
        status_item.setToolTip(status_text)
        self.update_row_appearance(row, status)

        if status == STATUS_DONE:
            removed_count = self._remove_paths([source_path])
            if removed_count:
                self.log(f"완료 파일 자동 제거: {source_path}")
        elif status == STATUS_REMOVED:
            self._remove_paths([source_path])

    def start_model_download(self) -> None:
        if self._download_worker is not None and self._download_worker.isRunning():
            return

        self._download_worker = ModelDownloadWorker(self.current_model_key(), self)
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

    def unload_model(self) -> None:
        unloaded = unload_loaded_models(self.current_model_key())
        if unloaded > 0:
            self.log(f"메모리에서 모델 {unloaded}개를 해제했습니다.")
        else:
            self.log("해제할 로딩된 모델이 없습니다.")
        self.refresh_model_status()

    def pause_pipeline(self) -> None:
        if self._pipeline_worker is None or not self._pipeline_worker.isRunning():
            return
        if self._pipeline_worker.pause_processing():
            self.current_stage_label.setText("현재 단계: 현재 파일 마무리 후 일시 중지 예정")
            self.log("일시 중지를 요청했습니다.")
        self.update_controls()

    def resume_pipeline(self) -> None:
        if self._pipeline_worker is None or not self._pipeline_worker.isRunning():
            return
        if self._pipeline_worker.resume_processing():
            self.log("일시 중지된 작업을 재개합니다.")
        self.update_controls()

    def start_pipeline(self) -> None:
        enable_translation = self.translate_checkbox.isChecked()
        api_keys = load_api_keys() if enable_translation else []
        deepl_api_key = load_deepl_api_key() if enable_translation else ""
        if enable_translation and not self._check_translation_keys(api_keys, deepl_api_key):
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
        self.current_file_label.setText("현재 파일: 없음")
        self._stage_title = "작업 준비"
        self._stage_detail = ""
        self.current_stage_label.setText("현재 단계: 작업 준비")
        self.stage_progress.setValue(0)
        self.translation_progress.setValue(0)
        self.translation_progress_label.setText("번역 진행: 대기 중")

        job = PipelineJob(
            source_paths=source_paths,
            runtime_device=self.current_runtime_device(),
            model_key=self.current_model_key(),
            source_language=self.current_input_language(),
            runtime_tuning=self.current_runtime_tuning(),
            vad_settings=self.current_vad_settings(),
            enable_postprocess=self.postprocess_checkbox.isChecked(),
            enable_enhanced_postprocess=self.enhanced_postprocess_checkbox.isChecked(),
            translator_settings=self._job_translator_settings(),
            api_keys=api_keys,
            deepl_api_key=deepl_api_key,
            enable_translation=enable_translation,
        )
        self._pipeline_worker = PipelineWorker(job, self)
        self._pipeline_worker.item_status_changed.connect(self.update_item_status)
        self._pipeline_worker.log_message.connect(self.log)
        self._pipeline_worker.queue_progress_changed.connect(self.on_queue_progress_changed)
        self._pipeline_worker.file_started.connect(self.on_file_started)
        self._pipeline_worker.stage_changed.connect(self.on_stage_changed)
        self._pipeline_worker.stage_progress_changed.connect(self.on_stage_progress_changed)
        self._pipeline_worker.translation_progress_changed.connect(self.on_translation_progress_changed)
        self._pipeline_worker.processing_state_changed.connect(self.on_processing_state_changed)
        self._pipeline_worker.summary_ready.connect(self.on_pipeline_summary_ready)
        self._pipeline_worker.failed.connect(self.on_pipeline_failed)
        self._pipeline_worker.finished.connect(self._run_pending_shutdown)
        self._pipeline_worker.start()
        self.update_controls()
        self.log(
            f"작업 시작: {len(source_paths)}개 파일, 장치={self.runtime_combo.currentText()}, "
            f"모델={self.model_combo.currentText()}, 입력 언어={self.input_language_combo.currentText()}, "
            f"출력 언어={self.output_language_combo.currentText() if enable_translation else '번역 안 함'}, "
            f"VAD={'ON' if self.is_vad_enabled() else 'OFF (Whisper only)'}"
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

    def on_processing_state_changed(self, state: str, detail: str) -> None:
        if state == STATUS_PAUSED:
            self.current_stage_label.setText(f"현재 단계: {detail}")
            self.current_file_label.setText("현재 파일: 일시 중지됨")
        self.update_controls()

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

    def on_subtitle_queue_progress_changed(self, processed: int, total: int) -> None:
        self._subtitle_queue_processed = processed
        self._subtitle_queue_total = total
        self.refresh_subtitle_queue_progress()

    def on_subtitle_file_started(self, index: int, total: int, source_path: str) -> None:
        self.subtitle_current_file_label.setText(f"현재 파일: {index} / {total} | {source_path}")
        self.subtitle_translation_progress.setValue(0)

    def on_subtitle_stage_changed(self, stage_title: str, stage_detail: str) -> None:
        self.subtitle_stage_label.setText(f"현재 단계: {stage_title} | {stage_detail}")

    def on_subtitle_translation_progress_changed(
        self, chunk_index: int, chunk_total: int, key_display: str, model_name: str, error_count: int
    ) -> None:
        percent = 0 if chunk_total <= 0 else int(chunk_index * 100 / chunk_total)
        self.subtitle_translation_progress.setValue(max(0, min(100, percent)))
        if chunk_total <= 0:
            self.subtitle_translation_progress_label.setText("번역 진행: 대기 중")
            return
        self.subtitle_translation_progress_label.setText(
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
        self.refresh_model_status()
        self.update_controls()

        summary = (
            f"작업 완료\n성공: {self._session_success_count}\n실패: {self._session_failure_count}\n스킵: {self._session_skipped_count}"
        )
        self.log(summary.replace("\n", " | "))
        self._request_shutdown_after_completion("메인 작업")
        QMessageBox.information(self, APP_NAME, summary)

    def on_pipeline_failed(self, message: str) -> None:
        self._processing = False
        self._stage_title = "실패"
        self._stage_detail = message
        self.current_stage_label.setText(f"현재 단계: 실패 | {message}")
        self._queue_total = 0
        self._queue_processed = 0
        self.log(f"작업 실패: {message}")
        self.refresh_model_status()
        self.update_controls()
        QMessageBox.warning(self, APP_NAME, message)

    def on_subtitle_summary_ready(self, success_count: int, failure_count: int, skipped_count: int) -> None:
        self._subtitle_translation_processing = False
        self._subtitle_success_count += success_count
        self._subtitle_failure_count += failure_count
        self._subtitle_skipped_count += skipped_count
        self.subtitle_current_file_label.setText("현재 파일: 작업 완료")
        self.subtitle_stage_label.setText("현재 단계: 완료")
        self.subtitle_translation_progress.setValue(100)
        self.refresh_subtitle_queue_progress()
        self._subtitle_queue_total = 0
        self._subtitle_queue_processed = 0
        self.update_controls()

        summary = (
            f"자막 번역 완료\n성공: {self._subtitle_success_count}\n실패: {self._subtitle_failure_count}\n스킵: {self._subtitle_skipped_count}"
        )
        self.subtitle_log_view.append(summary.replace("\n", " | "))
        self._request_shutdown_after_completion("자막 번역 작업")
        QMessageBox.information(self, APP_NAME, summary)

    def on_subtitle_failed(self, message: str) -> None:
        self._subtitle_translation_processing = False
        self.subtitle_stage_label.setText(f"현재 단계: 실패 | {message}")
        self._subtitle_queue_total = 0
        self._subtitle_queue_processed = 0
        self.subtitle_log_view.append(f"작업 실패: {message}")
        self.update_controls()
        QMessageBox.warning(self, APP_NAME, message)

    def _request_shutdown_after_completion(self, job_label: str) -> None:
        if not self.shutdown_after_complete_checkbox.isChecked() or self._shutdown_scheduled:
            return
        self._pending_shutdown_label = job_label
        self._run_pending_shutdown()

    def _run_pending_shutdown(self) -> None:
        # 다른 작업(자막 번역·음성 분할·모델 다운로드)이 남아 있으면 모두 끝난 뒤에 예약한다.
        label = self._pending_shutdown_label
        if label is None or self.is_busy():
            if label is not None:
                self.log(f"{label} 완료: 진행 중인 다른 작업이 끝나면 자동 종료를 예약합니다.")
            return
        self._pending_shutdown_label = None
        self._schedule_shutdown_after_completion(label)

    def _schedule_shutdown_after_completion(self, job_label: str) -> None:
        if not self.shutdown_after_complete_checkbox.isChecked() or self._shutdown_scheduled:
            return
        try:
            subprocess.run(["shutdown", "/s", "/t", "120"], check=True, capture_output=True, text=True, creationflags=_NO_WINDOW)
        except Exception as exc:
            message = f"{job_label} 완료 후 자동 종료 예약에 실패했습니다: {str(exc) or exc.__class__.__name__}"
            self.log(message)
            if self.tray_icon is not None:
                self.tray_icon.showMessage(APP_NAME, message, QSystemTrayIcon.Warning, 4000)
            return

        self._shutdown_scheduled = True
        message = f"{job_label} 완료: 2분 후 시스템 종료가 예약되었습니다."
        self.log(message)
        self.subtitle_log_view.append(message)
        if self.tray_icon is not None:
            self.tray_icon.showMessage(APP_NAME, message, QSystemTrayIcon.Information, 4000)

    def _cancel_scheduled_shutdown(self) -> None:
        try:
            subprocess.run(["shutdown", "/a"], check=True, capture_output=True, text=True, creationflags=_NO_WINDOW)
        except Exception as exc:
            self.log(f"예약된 시스템 종료 취소에 실패했습니다: {str(exc) or exc.__class__.__name__}")
            return

        self._shutdown_scheduled = False
        message = "예약된 시스템 종료를 취소했습니다."
        self.log(message)
        self.subtitle_log_view.append(message)
        if self.tray_icon is not None:
            self.tray_icon.showMessage(APP_NAME, message, QSystemTrayIcon.Information, 3000)
