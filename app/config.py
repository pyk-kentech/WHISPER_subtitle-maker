from __future__ import annotations

import os
from pathlib import Path


APP_NAME = "Dongeum Sub Maker"
APP_AUTHOR = "Codex"
API_KEYS_CREDENTIAL_NAME = "DongeumSubMaker/GeminiApiKeys"

MODEL_PRESETS = {
    "tiny": {
        "label": "faster-whisper-XXL-tiny",
        "repo_id": "Systran/faster-whisper-tiny",
    },
    "base": {
        "label": "faster-whisper-XXL-base",
        "repo_id": "Systran/faster-whisper-base",
    },
    "small": {
        "label": "faster-whisper-XXL-small",
        "repo_id": "Systran/faster-whisper-small",
    },
    "medium": {
        "label": "faster-whisper-XXL-medium",
        "repo_id": "Systran/faster-whisper-medium",
    },
    "large-v3": {
        "label": "faster-whisper-XXL-large-v3",
        "repo_id": "Systran/faster-whisper-large-v3",
    },
    "large-v3-turbo": {
        "label": "faster-whisper-XXL-large-v3-turbo",
        "repo_id": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    },
}
DEFAULT_MODEL_KEY = "medium"
MODEL_LABEL = MODEL_PRESETS[DEFAULT_MODEL_KEY]["label"]
SUPPORTED_EXTENSIONS = {".mp3", ".mp4"}
MODEL_REQUIRED_FILES = (
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.txt",
)
INPUT_LANGUAGE_OPTIONS = [
    ("auto", "Auto Detect"),
    ("ja", "Japanese"),
    ("en", "English"),
    ("ko", "Korean"),
    ("zh", "Chinese"),
]
OUTPUT_LANGUAGE_OPTIONS = [
    ("ko", "Korean"),
    ("en", "English"),
    ("ja", "Japanese"),
]
DEFAULT_INPUT_LANGUAGE = "ja"
DEFAULT_OUTPUT_LANGUAGE = "ko"
DEFAULT_VAD_ENABLED = True
DEFAULT_VAD_MIN_SILENCE_MS = 500
DEFAULT_VAD_SPEECH_PAD_MS = 200

TRANSLATOR_SUPPORTED_EXTENSIONS = {".srt", ".vtt", ".txt"}
DEFAULT_TRANSLATION_CHUNK_SIZE = 80
DEFAULT_TRANSLATION_TEMPERATURE = 1.0
DEFAULT_TRANSLATION_TOP_P = 0.8
DEFAULT_TRANSLATION_REASONING_LEVEL = "minimal"
DEFAULT_TRANSLATION_REQUEST_DELAY_SECONDS = 6.0
DEFAULT_TRANSLATION_REQUEST_DELAY_JITTER_SECONDS = 1.5
DEFAULT_TRANSLATION_BACKOFF_BASE_SECONDS = 5.0
DEFAULT_TRANSLATION_BACKOFF_MAX_SECONDS = 20.0
DEFAULT_TRANSLATION_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-flash",
]
DEFAULT_TRANSLATION_SYSTEM_PROMPT = """[공리]
입력: 원문 섹션이 주어짐. 번역 섹션이 함께 주어질 수도 있으며, 기존 번역문이므로 그 다음 줄부터 마저 번역.
출력: 다른 어떠한 응답도 없이 한국어 번역 결과만을 즉시 제공. HTML 구조를 훼손하거나 삭제하지 않고 그대로 유지. 반드시 </main>으로 종료.

섹션: <main id="섹션유형">...</main> 형식.
원문 섹션: 각 줄은 <p id="ID">원문</p> 형식. 번역 시 <p id="ID"> 부분은 반드시 그대로 유지.
번역 섹션: 각 줄은 <p id="ID">번역</p> 형식. 동일한 ID의 원문에 정확히 일대일대응하도록 번역 작성. 문장이 여러 줄에 걸쳐 있는 경우 절대로 문장을 임의로 합치지 않고 엄격하게 각 줄을 독립적으로 번역.

[지침]
직역투를 피하며 최대한 자연스럽게 의역하되, 원문의 말투와 내용은 철저히 유지. 원문의 사실 관계를 왜곡하거나 고유명사의 과한 현지화 금지.
일본어 고유명사는 국립국어원 표기법을 무시하고 해당 장르 및 작품에서 대중에게 친숙한 서브컬처 통용 표기를 최우선하되, 통용 표기가 불확실하다면 실제 일본어 발음에 가깝게 표기.
일본어가 아닌 중국어 고유명사는 원어 발음 대신 한국 한자음을 엄격히 지키며 표기.
HTML 태그, 줄 순서, <p id="..."> 구조, 타임스탬프 대응 관계를 절대로 훼손하지 말 것.

{{note}}"""


def get_app_data_dir() -> Path:
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "DongeumSubMaker"
    return Path.home() / ".dongeum-sub-maker"


def get_model_cache_dir() -> Path:
    return get_model_cache_dir_for(DEFAULT_MODEL_KEY)


def get_model_cache_dir_for(model_key: str) -> Path:
    safe_key = model_key.strip().lower()
    return get_app_data_dir() / "models" / f"faster-whisper-{safe_key}"


def get_hf_cache_dir() -> Path:
    return get_app_data_dir() / "hf-cache"


def get_cuda_cache_dir() -> Path:
    return get_app_data_dir() / "cuda-cache"


def get_cuda_runtime_dir() -> Path:
    return get_app_data_dir() / "cuda-runtime"


def get_dictionary_cache_dir() -> Path:
    return get_app_data_dir() / "dictionary-cache"


def get_dictionary_pack_dir() -> Path:
    return get_app_data_dir() / "dictionary-pack"


def get_translator_keys_path() -> Path:
    return get_app_data_dir() / "keys.txt"


def get_translator_settings_path() -> Path:
    return get_app_data_dir() / "translator-settings.json"


def get_log_dir() -> Path:
    return get_app_data_dir() / "logs"
