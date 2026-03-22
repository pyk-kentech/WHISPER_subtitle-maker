from __future__ import annotations

import logging
import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from .app_logging import get_logger, setup_logging
from .config import APP_NAME, get_log_dir
from .cuda_runtime import add_cuda_runtime_to_path
from .ui import MainWindow


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


def main() -> int:
    setup_logging()
    logger = get_logger()
    add_cuda_runtime_to_path()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Codex")
    app.setQuitOnLastWindowClosed(False)
    logger.info("Application startup")
    window = MainWindow(auto_download_on_startup=True)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
