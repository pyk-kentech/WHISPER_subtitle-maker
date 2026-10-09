from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


# 밝은/어두운 테마 색. 위젯 코드는 색을 직접 쓰지 않고 여기 값(또는 앱 스타일시트)을 쓴다.
LIGHT = {
    "window": "#f3f5f8",
    "tab": "#e3e8ee",
    "base": "#ffffff",
    "alt_base": "#f6f8fa",
    "text": "#1f2a37",
    "muted": "#5c6b7c",
    "disabled": "#9aa5b1",
    "border": "#cfd8e2",
    "border_strong": "#a9b6c4",
    "button": "#ffffff",
    "button_hover": "#eef2f6",
    "card": "#f8fafc",
    "highlight": "#2f6fd0",
    "highlight_text": "#ffffff",
    "accent": "#2f9e44",
    "accent_hover": "#2b8a3e",
    "accent_soft": "#e9f6ec",
    "accent_text": "#ffffff",
    "track": "#e8edf2",
    "success": "#2f7d3e",
    "drop_bg": "#f6f9fc",
    "drop_border": "#8a99a8",
    "row_running": "#e5f0ff",
    "row_done": "#e5f5e8",
    "row_failed": "#fce8e8",
    "row_skipped": "#f7f2e4",
}

DARK = {
    "window": "#25282d",
    "tab": "#2f3338",
    "base": "#1c1f23",
    "alt_base": "#22262a",
    "text": "#e3e7ec",
    "muted": "#9ca8b5",
    "disabled": "#6b747f",
    "border": "#454c55",
    "border_strong": "#5a646f",
    "button": "#353a41",
    "button_hover": "#3e444c",
    "card": "#2c3036",
    "highlight": "#3d7fe0",
    "highlight_text": "#ffffff",
    "accent": "#37a24e",
    "accent_hover": "#40b358",
    "accent_soft": "#22392a",
    "accent_text": "#ffffff",
    "track": "#1c1f23",
    "success": "#6fcf83",
    "drop_bg": "#2a2e33",
    "drop_border": "#6b7785",
    "row_running": "#1f3552",
    "row_done": "#1f3b29",
    "row_failed": "#4a2428",
    "row_skipped": "#3d3725",
}

_ROW_KEYS = {"running": "row_running", "done": "row_done", "failed": "row_failed", "skipped": "row_skipped"}
_current: dict[str, str] | None = None
_forced_dark: bool | None = None


def system_prefers_dark(app: QApplication) -> bool:
    try:
        scheme = app.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            return True
        if scheme == Qt.ColorScheme.Light:
            return False
    except AttributeError:
        pass
    return app.palette().color(QPalette.ColorRole.Window).lightness() < 128


def colors() -> dict[str, str]:
    return _current or LIGHT


def is_dark() -> bool:
    return colors() is DARK


def color(key: str) -> QColor:
    return QColor(colors()[key])


def row_color(kind: str | None) -> QColor:
    """대기열 표의 상태별 행 배경색. kind가 None이면 기본 배경."""
    return color(_ROW_KEYS.get(kind or "", "base"))


def _build_palette(c: dict[str, str]) -> QPalette:
    palette = QPalette()
    roles = {
        QPalette.ColorRole.Window: c["window"],
        QPalette.ColorRole.WindowText: c["text"],
        QPalette.ColorRole.Base: c["base"],
        QPalette.ColorRole.AlternateBase: c["alt_base"],
        QPalette.ColorRole.ToolTipBase: c["base"],
        QPalette.ColorRole.ToolTipText: c["text"],
        QPalette.ColorRole.PlaceholderText: c["disabled"],
        QPalette.ColorRole.Text: c["text"],
        QPalette.ColorRole.Button: c["button"],
        QPalette.ColorRole.ButtonText: c["text"],
        QPalette.ColorRole.BrightText: "#ffffff",
        QPalette.ColorRole.Highlight: c["highlight"],
        QPalette.ColorRole.HighlightedText: c["highlight_text"],
        QPalette.ColorRole.Link: c["highlight"],
        QPalette.ColorRole.Light: c["button_hover"],
        QPalette.ColorRole.Midlight: c["border"],
        QPalette.ColorRole.Mid: c["border_strong"],
        QPalette.ColorRole.Dark: c["border_strong"],
        QPalette.ColorRole.Shadow: "#000000",
    }
    for role, value in roles.items():
        palette.setColor(role, QColor(value))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor(c["disabled"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, QColor(c["window"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, QColor(c["window"]))
    return palette


def build_stylesheet(c: dict[str, str]) -> str:
    return f"""
QLabel[role="title"] {{ font-size: 19px; font-weight: 700; }}
QLabel[role="hint"] {{ color: {c["muted"]}; }}
QLabel[role="success"] {{ color: {c["success"]}; }}
QLabel[role="card"] {{
    padding: 8px 10px;
    border: 1px solid {c["border"]};
    border-radius: 6px;
    background: {c["card"]};
    color: {c["muted"]};
}}

QTabWidget::pane {{
    border: 1px solid {c["border"]};
    background: {c["window"]};
    top: -1px;
}}
QTabBar::tab {{
    background: {c["tab"]};
    color: {c["muted"]};
    border: 1px solid {c["border"]};
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    padding: 5px 14px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {c["window"]};
    color: {c["text"]};
    border-bottom-color: {c["window"]};
}}
QTabBar::tab:!selected {{ margin-top: 2px; }}
QTabBar::tab:!selected:hover {{ color: {c["text"]}; }}

QTextEdit, QPlainTextEdit {{
    border: 1px solid {c["border"]};
    border-radius: 4px;
    background: {c["base"]};
}}
QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {c["highlight"]}; }}

QGroupBox {{
    border: 1px solid {c["border"]};
    border-radius: 6px;
    margin-top: 10px;
    padding: 10px 8px 6px 8px;
}}
QGroupBox[untitled="true"] {{ margin-top: 0px; padding-top: 6px; }}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: {c["muted"]};
}}

QProgressBar {{
    border: 1px solid {c["border"]};
    border-radius: 5px;
    background: {c["track"]};
    color: {c["text"]};
    text-align: center;
    min-height: 16px;
    max-height: 18px;
}}
QProgressBar::chunk {{
    background-color: {c["accent"]};
    border-radius: 4px;
}}

QPushButton[role="primary"] {{
    background: {c["accent"]};
    color: {c["accent_text"]};
    border: 1px solid {c["accent_hover"]};
    border-radius: 4px;
    padding: 4px 16px;
    font-weight: 600;
}}
QPushButton[role="primary"]:hover {{ background: {c["accent_hover"]}; }}
QPushButton[role="primary"]:disabled {{
    background: {c["button"]};
    color: {c["disabled"]};
    border-color: {c["border"]};
}}

QToolButton[role="settingsMenu"] {{
    padding: 4px 10px;
    padding-right: 18px;
    border: 1px solid {c["border_strong"]};
    border-radius: 4px;
    background: {c["button"]};
}}
QToolButton[role="settingsMenu"]:hover {{ background: {c["button_hover"]}; }}
QToolButton[role="settingsMenu"]::menu-indicator {{
    subcontrol-origin: padding;
    subcontrol-position: right center;
    right: 5px;
}}
QToolButton[role="sectionToggle"] {{
    font-weight: 600;
    padding: 4px 2px;
    border: none;
    background: transparent;
}}

#dropArea {{
    border: 2px dashed {c["drop_border"]};
    border-radius: 8px;
    background: {c["drop_bg"]};
}}
#dropArea[dragActive="true"] {{
    border: 2px solid {c["accent"]};
    background: {c["accent_soft"]};
}}
#dropAreaLabel {{
    font-size: 16px;
    font-weight: 600;
    color: {c["muted"]};
    background: transparent;
}}

QTableWidget[dropTarget="true"] {{
    border: 1px solid {c["border"]};
    border-radius: 6px;
    background: {c["base"]};
    gridline-color: {c["border"]};
}}
QTableWidget[dropTarget="true"][dragActive="true"] {{
    border: 2px solid {c["accent"]};
    background: {c["accent_soft"]};
}}
"""


def apply_theme(app: QApplication | None = None, dark: bool | None = None) -> None:
    """Fusion 스타일 + 밝은/어두운 팔레트 + 앱 공통 스타일시트를 적용한다. dark=None이면 시스템 설정을 따른다."""
    global _current, _forced_dark
    app = app or QApplication.instance()
    if app is None:
        return
    first = _current is None
    if dark is not None:
        _forced_dark = dark
    if first:
        app.setStyle("Fusion")
    use_dark = _forced_dark if _forced_dark is not None else system_prefers_dark(app)
    _current = DARK if use_dark else LIGHT
    app.setPalette(_build_palette(_current))
    app.setStyleSheet(build_stylesheet(_current))
    if first:
        try:
            app.styleHints().colorSchemeChanged.connect(lambda *_: apply_theme(app))
        except AttributeError:
            pass


def ensure_theme(app: QApplication | None = None) -> None:
    if _current is None:
        apply_theme(app)


def set_dynamic_property(widget, name: str, value) -> None:
    """스타일시트 선택자에 쓰는 동적 속성을 바꾸고 다시 그린다."""
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
