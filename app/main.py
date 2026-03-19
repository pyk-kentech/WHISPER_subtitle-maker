from __future__ import annotations

import traceback
import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from .config import APP_NAME, get_log_dir
from .ui import MainWindow


def _show_fatal_error(exc_type, exc_value, exc_traceback) -> None:
    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "fatal-error.log"
    error_text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    log_path.write_text(error_text, encoding="utf-8")

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    message = (
        "프로그램 실행 중 치명적인 오류가 발생했습니다.\n\n"
        f"{exc_value}\n\n"
        f"로그 파일:\n{log_path}"
    )
    QMessageBox.critical(None, APP_NAME, message)


sys.excepthook = _show_fatal_error


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Codex")
    auto_download_on_startup = os.getenv("DONGEUM_SKIP_STARTUP_DOWNLOAD") != "1"
    window = MainWindow(auto_download_on_startup=auto_download_on_startup)
    window.show()
    if os.getenv("DONGEUM_TEST_AUTOQUIT") == "1":
        QTimer.singleShot(3000, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
