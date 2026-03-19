from __future__ import annotations

from pathlib import Path
import os


APP_NAME = "Dongeum Sub Maker"
APP_AUTHOR = "Codex"

MODEL_REPO_ID = "Systran/faster-whisper-medium"
MODEL_LANGUAGE = "ja"
MODEL_LABEL = "Faster-Whisper-XXL / medium / japanese"
SUPPORTED_EXTENSIONS = {".mp3", ".mp4"}
MODEL_REQUIRED_FILES = (
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.txt",
)


def get_app_data_dir() -> Path:
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "DongeumSubMaker"
    return Path.home() / ".dongeum-sub-maker"


def get_model_cache_dir() -> Path:
    return get_app_data_dir() / "models" / "faster-whisper-medium"


def get_hf_cache_dir() -> Path:
    return get_app_data_dir() / "hf-cache"


def get_log_dir() -> Path:
    return get_app_data_dir() / "logs"
