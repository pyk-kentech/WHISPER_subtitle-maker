from __future__ import annotations

import ctypes
import logging
import sys
import traceback

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QMessageBox

from .app_logging import get_logger, setup_logging
from .config import APP_NAME, get_font_paths, get_log_dir
from .cuda_runtime import add_cuda_runtime_to_path
from .ui import MainWindow


_SINGLE_INSTANCE_MUTEX = None


def _show_fatal_error(exc_type, exc_value, exc_traceback) -> None:
    logger = get_logger()
    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "fatal-error.log"
    error_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    log_path.write_text(error_text, encoding="utf-8")
    logger.critical("Fatal startup/runtime error", exc_info=(exc_type, exc_value, exc_traceback))

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    message = f"프로그램 실행 중 치명적 오류가 발생했습니다.\n\n{exc_value}\n\n로그 파일:\n{log_path}"
    QMessageBox.critical(None, APP_NAME, message)


sys.excepthook = _show_fatal_error


def _set_windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Codex.DongeumSubMaker")
    except Exception:
        pass


def _acquire_single_instance() -> bool:
    global _SINGLE_INSTANCE_MUTEX
    if sys.platform != "win32":
        return True
    try:
        kernel32 = ctypes.windll.kernel32
        mutex = kernel32.CreateMutexW(None, False, "Global\\DongeumSubMakerSingleInstance")
        if not mutex:
            return True
        _SINGLE_INSTANCE_MUTEX = mutex
        return kernel32.GetLastError() != 183
    except Exception:
        return True


def _apply_bundled_font(app: QApplication) -> None:
    for font_path in get_font_paths():
        try:
            font_id = QFontDatabase.addApplicationFont(str(font_path))
        except Exception:
            continue
        if font_id < 0:
            continue
        families = QFontDatabase.applicationFontFamilies(font_id)
        if not families:
            continue
        app.setFont(QFont(families[0]))
        return


def main() -> int:
    setup_logging()
    logger = get_logger()
    add_cuda_runtime_to_path()
    _set_windows_app_id()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Codex")
    app.setQuitOnLastWindowClosed(False)
    _apply_bundled_font(app)
    if not _acquire_single_instance():
        QMessageBox.information(None, APP_NAME, "이미 실행 중입니다.\n기존 창이나 트레이 아이콘을 확인해 주세요.")
        return 0
    logger.info("Application startup")
    window = MainWindow(auto_download_on_startup=True)
    if not window.windowIcon().isNull():
        app.setWindowIcon(window.windowIcon())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
