from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "Dongeum Sub Maker"
APP_AUTHOR = "Codex"
API_KEYS_CREDENTIAL_NAME = "DongeumSubMaker/GeminiApiKeys"
DEEPL_API_KEY_CREDENTIAL_NAME = "DongeumSubMaker/DeepLApiKey"
OPENAI_COMPAT_API_KEY_CREDENTIAL_NAME = "DongeumSubMaker/OpenAICompatApiKey"
RUNPOD_API_KEY_CREDENTIAL_NAME = "DongeumSubMaker/RunpodApiKey"

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
        "files": ("config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"),
    },
    "large-v3-turbo": {
        "label": "faster-whisper-XXL-large-v3-turbo",
        "repo_id": "dropbox-dash/faster-whisper-large-v3-turbo",
        "files": ("config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"),
    },
    # 애니·게임 대사로 파인튜닝한 일본어 전용 모델(litagin/anime-whisper의 CTranslate2 변환본, 약 3GB).
    # 짧은 대사 단위로 학습되어 타임스탬프 토큰을 거의 내지 못하고 30초 창을 통째로 주면 반복 환각이 생긴다.
    # 그래서 VAD로 발화마다 잘라 따로 인식하고(vad_clips), 자막 시간은 VAD 구간 경계를 쓴다.
    # ASMR 평가셋에서 가장 정확했다(eval/RESULTS.md): WhisperSeg VAD와 함께 쓰면 읽는 소리 CER 0.146(medium 0.313).
    "anime-whisper": {
        "label": "anime-whisper (일본어 전용, ASMR 추천)",
        "repo_id": "quantumcookie/anime-whisper-ct2",
        "files": ("config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"),
        "languages": ("ja",),
        "segmentation": "vad_clips",
        "max_clip_seconds": 12,
        # 모델 카드 권장: 초기 프롬프트 금지, 반복 환각은 no_repeat_ngram_size로 억제(카드 평가값 5).
        "transcribe_options": {"no_repeat_ngram_size": 5, "repetition_penalty": 1.0},
        "note": "일본어 전용이라 입력 언어가 일본어로 고정되고, 말소리 구간마다 잘라 인식하므로 VAD가 켜짐으로 고정됩니다. "
        "CPU로도 쓸 수 있지만 medium보다 1.4배쯤 느립니다(노트북 CPU 기준 음성 길이의 약 0.8배).",
    },
}
# 음성 인식 모델이 아닌 보조 모델(VAD). 모델 목록에는 나오지 않고, 고른 기능을 처음 쓸 때 받는다.
VAD_MODEL_KEY = "whisperseg-vad"
AUX_MODEL_PRESETS = {
    VAD_MODEL_KEY: {
        "label": "WhisperSeg ASMR VAD",
        "repo_id": "TransWithAI/Whisper-Vad-EncDec-ASMR-onnx",
        "files": ("model.onnx", "model_metadata.json"),
    },
}
# 기본 모델(eval/RESULTS.md): ASMR 평가셋에서 가장 정확하고(검증 작품 읽는 소리 CER 0.18, 예전 기본 medium 0.53),
# CPU만 있는 PC에서도 쓸 만한 속도(노트북 CPU 기준 음성 길이의 약 0.8배)라 CPU·GPU 모두 같은 모델을 기본으로 쓴다.
# GPU가 있으면 실행 장치가 GPU(float16, beam 5, 배치 8)로 자동 선택된다.
DEFAULT_MODEL_KEY = "anime-whisper"
GPU_DEFAULT_MODEL_KEY = "anime-whisper"
# 구간별 인식에서 모델 프리셋에 따로 정하지 않았을 때 쓰는 디코딩 옵션(짧은 구간에서 같은 말이 반복되는 환각 억제)
DEFAULT_CLIPS_TRANSCRIBE_OPTIONS = {"no_repeat_ngram_size": 5}
MODEL_LABEL = MODEL_PRESETS[DEFAULT_MODEL_KEY]["label"]
SUPPORTED_EXTENSIONS = {".mp3", ".mp4"}
# large-v3 계열은 vocabulary.json + preprocessor_config.json(128 mel)을 쓰므로 프리셋의 "files"가 우선한다.
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
# VAD 종류: Silero(faster-whisper 내장)와 ASMR용 WhisperSeg(속삭임에 강함, 처음 쓸 때 모델 약 120MB를 받음)
VAD_BACKEND_OPTIONS = [
    ("whisperseg", "ASMR (WhisperSeg)"),
    ("silero", "Silero"),
]
DEFAULT_VAD_BACKEND = "whisperseg"
# WhisperSeg 권장값(제작자 기본: 임계 0.5, 최소 무음 100ms)
WHISPERSEG_DEFAULT_THRESHOLD = 0.5
WHISPERSEG_DEFAULT_MIN_SILENCE_MS = 100
# 인식 방식: 일반(Whisper가 30초 창마다 스스로 자막 시간을 냄) / 구간별(VAD 구간마다 잘라 인식하고 구간 경계를 자막 시간으로 씀)
SEGMENTATION_OPTIONS = [
    ("clips", "구간별 (VAD 구간마다 인식)"),
    ("normal", "일반 (Whisper 시간 사용)"),
]
DEFAULT_SEGMENTATION = "clips"
DEFAULT_MAX_CLIP_SECONDS = 12

TRANSLATOR_SUPPORTED_EXTENSIONS = {".srt", ".vtt", ".txt"}
DEFAULT_TRANSLATION_CHUNK_SIZE = 60
LEGACY_TRANSLATION_CHUNK_SIZE = 80
TRANSLATION_CONTEXT_LINES = 3
TRANSLATION_ECHO_MIN_SIMILARITY = 0.8
DEFAULT_TRANSLATION_TEMPERATURE = 1.0
DEFAULT_TRANSLATION_TOP_P = 0.8
DEFAULT_TRANSLATION_REASONING_LEVEL = "minimal"
DEFAULT_TRANSLATION_REQUEST_DELAY_SECONDS = 6.0
DEFAULT_TRANSLATION_REQUEST_DELAY_JITTER_SECONDS = 1.5
# 인터넷이 끊기면 이 시간까지 연결을 기다리고, 넘으면 그 파일은 원문 자막만 남기고 다음으로 넘어간다.
NETWORK_OUTAGE_MAX_WAIT_SECONDS = 600
NETWORK_OUTAGE_RETRY_SECONDS = (5, 10, 20, 30, 60)
# 무료 티어 하루 요청 한도가 모델별로 따로 잡히므로, 한도가 큰 Flash-Lite(500회/일)를 앞에 둔다.
DEFAULT_TRANSLATION_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
]
DEEPL_FREE_API_URL = "https://api-free.deepl.com/v2/translate"
# 번역 공급자: Gemini(기본) 또는 OpenAI 호환 LLM 서버(Runpod에 띄운 LLM, llama.cpp, Ollama, vLLM 등)
TRANSLATION_PROVIDER_OPTIONS = [
    ("gemini", "Gemini"),
    ("openai", "LLM 서버 (OpenAI 호환 · Runpod 등)"),
]
DEFAULT_TRANSLATION_PROVIDER = "gemini"
DEFAULT_OPENAI_COMPAT_TEMPERATURE = 0.7
DEFAULT_OPENAI_COMPAT_TOP_P = 0.9
LEGACY_TRANSLATION_SYSTEM_PROMPT = """[공리]
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
DEFAULT_TRANSLATION_SYSTEM_PROMPT = """[공리]
입력: 원문 섹션이 주어짐. 참고 섹션이 함께 주어질 수 있으며, 이는 직전 대사로 흐름 파악용이므로 번역하거나 출력하지 않음.
출력: 다른 어떠한 응답도 없이 한국어 번역 결과만을 즉시 제공. HTML 구조를 훼손하거나 삭제하지 않고 그대로 유지. 반드시 </main>으로 종료.

섹션: <main id="섹션유형">...</main> 형식.
원문 섹션: 각 줄은 <p id="ID">원문</p> 형식. 번역 시 <p id="ID"> 부분은 반드시 그대로 유지.
번역 섹션: 각 줄은 <p id="ID"><o>원문</o>번역</p> 형식. <o> 안에는 동일한 ID의 원문을 한 글자도 바꾸지 않고 그대로 복사하고, 그 뒤에 그 줄의 번역만 작성. 문장이 여러 줄에 걸쳐 있는 경우 절대로 문장을 임의로 합치거나 다른 줄로 옮기지 않고 엄격하게 각 줄을 독립적으로 번역.

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
    if sys.platform.startswith("linux"):
        # 리눅스 관례(XDG): ~/.local/share/DongeumSubMaker
        data_home = os.getenv("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        return Path(data_home) / "DongeumSubMaker"
    return Path.home() / ".dongeum-sub-maker"


def get_model_cache_dir() -> Path:
    return get_model_cache_dir_for(DEFAULT_MODEL_KEY)


def get_model_cache_dir_for(model_key: str) -> Path:
    safe_key = model_key.strip().lower()
    if safe_key in AUX_MODEL_PRESETS:
        return get_app_data_dir() / "models" / safe_key
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


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return get_project_root()


def get_preferred_icon_path() -> Path | None:
    icon_files = sorted(get_runtime_base_dir().glob("*.ico"))
    if not icon_files and getattr(sys, "frozen", False):
        icon_files = sorted(Path(sys.executable).resolve().parent.glob("*.ico"))
    if not icon_files:
        return None
    return icon_files[0]


def get_font_paths() -> list[Path]:
    base_dir = get_runtime_base_dir()
    font_files = sorted(base_dir.glob("*.ttf")) + sorted(base_dir.glob("*.otf"))
    if not font_files and getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        font_files = sorted(exe_dir.glob("*.ttf")) + sorted(exe_dir.glob("*.otf"))
    return font_files
