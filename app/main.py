from __future__ import annotations

import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from .ui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Dongeum Sub Maker")
    app.setOrganizationName("Codex")
    window = MainWindow()
    window.show()
    if os.getenv("DONGEUM_TEST_AUTOQUIT") == "1":
        QTimer.singleShot(3000, app.quit)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
