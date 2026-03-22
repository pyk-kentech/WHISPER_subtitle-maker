from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable

from .config import APP_NAME, get_log_dir


LOGGER_NAME = "dongeum_sub_maker"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
DATE_FORMAT = "%H:%M:%S"
MAX_LOG_BYTES = 1_048_576
BACKUP_COUNT = 5

_ui_callback: Callable[[str], None] | None = None


class UILogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if _ui_callback is None:
            return
        if record.levelno < logging.INFO:
            return
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        _ui_callback(message)


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def set_ui_log_callback(callback: Callable[[str], None] | None) -> None:
    global _ui_callback
    _ui_callback = callback


def setup_logging() -> Path:
    log_dir = get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app.log"

    logger = get_logger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=MAX_LOG_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(file_handler)

    if not any(isinstance(handler, UILogHandler) for handler in logger.handlers):
        ui_handler = UILogHandler()
        ui_handler.setLevel(logging.INFO)
        ui_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(ui_handler)

    return log_path
