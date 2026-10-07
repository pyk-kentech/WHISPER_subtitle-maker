# Dongeum Sub Maker

Windows 전용 GUI 프로그램입니다. `.mp3`, `.mp4` 파일에서 자막을 만들고, 필요하면 번역까지 이어서 처리할 수 있습니다.

## 설치와 실행 (Windows)

### 1. 준비 (처음 한 번)

1. **Python 3.12 또는 3.13**을 설치합니다: <https://www.python.org/downloads/>
   - 설치 첫 화면에서 **"Add python.exe to PATH"를 반드시 체크**합니다.
   - Python 3.14 이상은 아직 지원하지 않습니다. 빌드 스크립트가 3.10~3.13만 허용합니다.
   - 설치 확인: PowerShell에서 `python --version`을 실행해 `Python 3.12.x` 또는 `3.13.x`가 나오면 됩니다.
2. 이 저장소를 받습니다. 둘 중 하나를 고르세요.
   - **ZIP:** GitHub 저장소 화면에서 **Code → Download ZIP**을 누르고 원하는 폴더에 압축을 풉니다. 예: `D:\SubMaker`
   - **Git:** `git clone https://github.com/pyk-kentech/WHISPER_subtitle-maker.git`
3. 압축을 푼 폴더에서 빈 곳을 **Shift + 우클릭 → "PowerShell 창 열기"** 또는 **"터미널에서 열기"**를 누릅니다.

### 2-A. exe로 빌드해서 쓰기 (권장)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\build.ps1
```

- 첫 줄은 이 PowerShell 창에서만 스크립트 실행을 허용합니다. 창을 닫으면 원래대로 돌아갑니다.
- `build.ps1`이 하는 일
  - `.venv` 가상환경을 만듭니다(이미 있으면 그대로 씁니다).
  - `requirements.txt`의 라이브러리를 설치합니다.
  - PyInstaller로 exe를 만듭니다.
  - 첫 빌드는 10~20분 정도 걸립니다. 중간에 실패하면 그 단계 이름과 함께 바로 멈춥니다.
- 결과
  - 실행 파일은 **`dist\DongeumSubMaker\DongeumSubMaker.exe`**입니다.
  - `dist\DongeumSubMaker` 폴더 안의 파일을 함께 써야 하니 **폴더째로 두고**, exe의 바로가기를 바탕화면 등에 만들어 쓰세요.
- 파일 하나짜리 exe가 필요하면 `.\build-onefile.ps1`을 실행합니다.
  - 결과는 `dist\DongeumSubMaker-OneFile.exe`입니다.
  - 실행할 때마다 임시 폴더에 풀어서 시작이 조금 느립니다.

### 2-B. 빌드하지 않고 소스에서 바로 실행

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m app.main
```

다음부터는 마지막 줄만 실행하면 됩니다.

### 3. 처음 실행할 때

- **Whisper 모델 자동 다운로드:** 처음 켜면 선택된 모델(기본 `medium`, 약 1.5GB)을 받습니다. 작업 탭 위쪽 진행률에 표시되고, 다 받아야 **시작** 버튼이 켜집니다.

  | 모델 | 크기 | 비고 |
  |---|---|---|
  | `tiny` | 약 75MB | |
  | `base` | 약 145MB | |
  | `small` | 약 480MB | |
  | `medium` | 약 1.5GB | |
  | `large-v3` | 약 3GB | |
  | `large-v3-turbo` | 약 1.6GB | NVIDIA GPU가 있으면 추천 |

  다른 모델을 고르면 그 모델도 바로 받습니다.
- **GPU(CUDA):** NVIDIA 그래픽카드가 있으면 실행 장치가 GPU로 잡힙니다. 첫 GPU 작업 때 CUDA 파일(약 1.4GB)을 자동으로 받습니다. GPU 초기화에 실패하면 CPU로 자동 전환합니다.
- **번역 키**
  1. <https://aistudio.google.com/apikey> 에서 Gemini API 키를 만듭니다.
  2. **번역 설정** 탭에 붙여넣고 **저장**을 누릅니다.
  3. 번역 없이 원문 자막만 만들려면 **작업 탭 → STT 설정 → 언어 → 번역**의 체크를 끄세요. 이때는 키가 필요 없습니다.
- **무료 한도**
  - 본인 한도는 <https://aistudio.google.com/rate-limit> 에서 확인할 수 있습니다.
  - 한도는 **API 키가 아니라 Google 프로젝트 단위**입니다. 같은 프로젝트에서 만든 키를 여러 개 넣어도 한도는 늘지 않습니다.
  - 기본 모델인 `gemini-3.5-flash-lite`는 무료로 하루 약 500회입니다. 청크 60줄 기준으로 하루 약 30시간 분량입니다.

### 4. 사용법

- **작업 탭:** mp3·mp4 파일이나 폴더를 끌어다 놓고 **시작**을 누릅니다. 결과는 원본 옆에 저장됩니다(아래 "출력 규칙" 참고).
- **자막 번역 탭:** 이미 있는 `.srt`, `.vtt`, `.txt` 자막을 번역만 합니다. 결과는 원본 옆에 `이름.<출력 언어>.<원래 확장자>`로 저장됩니다(예: `a.srt` → `a.ko.srt`). 이름이 이미 출력 언어 코드로 끝나는 파일(`a.ko.srt`)은 번역 결과로 보고 건너뜁니다.
- **번역 설정 탭:** API 키, 선호 모델, 청크 크기, 요청 간 지연, 추론 레벨, 프롬프트, 번역 노트를 정합니다. 바꾼 뒤 **저장**을 누르세요.
  - 출력 언어는 작업 탭 설정에서 고른 값이 저장하지 않아도 바로 적용됩니다.
  - 시스템 프롬프트의 `{{note}}`는 지우면 안 됩니다. 번역 노트와 줄 밀림 검사 지시가 이 자리에 들어갑니다.
- **음성 분할 탭:** 아래 "음성 분할"을 참고하세요.

### 5. 새 버전으로 업데이트

1. 최신 코드를 받습니다. ZIP을 다시 받아 같은 폴더에 덮어쓰거나, Git이면 `git pull`을 실행합니다.
2. 다시 빌드합니다. `.\build.ps1`이 `requirements.txt`대로 라이브러리 버전을 맞춥니다.
   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\build.ps1
   ```
   소스로 실행 중이라면 대신 `.\.venv\Scripts\python -m pip install -r requirements.txt`를 실행합니다.
3. 그래도 이상하면 `.venv`, `build`, `dist` 폴더를 지우고 2번을 다시 합니다.

설정, 키, 받은 모델은 `%LOCALAPPDATA%\DongeumSubMaker`에 있어서 업데이트해도 그대로 유지됩니다.

### 6. 문제 해결

| 증상 | 해결 |
|---|---|
| `open() got an unexpected keyword argument 'metadata_errors'` | 설치된 PyAV가 19 이상이라 faster-whisper 1.2.1과 맞지 않습니다. `.\.venv\Scripts\python -m pip install -r requirements.txt`로 `av<19`를 설치하거나 `.venv`를 지우고 다시 빌드하세요. |
| `이 시스템에서 스크립트를 실행할 수 없으므로…` | 빌드 전에 `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`를 먼저 실행하세요. |
| `Python was not found` 또는 Microsoft Store가 열림 | Python 설치 때 PATH 체크를 빠뜨렸습니다. 다시 설치하면서 체크하거나, **설정 → 앱 → 고급 앱 설정 → 앱 실행 별칭**에서 `python.exe` 별칭을 끄세요. |
| `Python 3.10-3.13 is required` | `.venv`가 다른 버전의 Python으로 만들어졌습니다. `.venv` 폴더를 지우고 3.12 또는 3.13으로 다시 빌드하세요. |
| "이미 실행 중입니다" | 작업 표시줄 오른쪽 트레이 아이콘을 클릭해 창을 여세요. 창의 X는 트레이로 숨기기이고, 완전히 끄려면 트레이 아이콘 우클릭 메뉴의 **Quit**을 누르세요. |
| 번역이 429·할당량 오류로 멈춤 | 무료 하루 한도가 끝난 것입니다. 원문 자막(`a.ja.srt`)은 저장되어 있으니, 한도가 초기화된 뒤(한국 시간 오후 4~5시) 다시 **시작**하면 음성 인식 없이 번역만 다시 합니다. DeepL Free 키를 넣고 폴백을 켜면 나머지를 DeepL이 번역합니다. |
| 일부 줄이 원문(일본어) 그대로 남음 | 검열(안전 필터) 등으로 그 줄만 번역되지 않은 것입니다. 완료 메시지에 줄 수가 표시됩니다. DeepL 폴백을 켜면 그 줄을 DeepL이 채웁니다. |
| 기타 오류 | `%LOCALAPPDATA%\DongeumSubMaker\logs\app.log`에서 자세한 내용을 확인할 수 있습니다. |

## 주요 기능

- 입력 파일: `.mp3`, `.mp4`
- 입력 방식: 파일 추가, 폴더 추가, 드래그 앤 드롭
- 폴더 추가 시 하위 폴더 포함 옵션 지원
- Whisper 모델 선택 지원
  - `faster-whisper-XXL-tiny`
  - `faster-whisper-XXL-base`
  - `faster-whisper-XXL-small`
  - `faster-whisper-XXL-medium`
  - `faster-whisper-XXL-large-v3`
  - `faster-whisper-XXL-large-v3-turbo`
- 번역 사용 여부 선택 (끄면 Gemini 키 없이 원문 자막만 생성)
- 입력 언어 선택 지원
  - `auto`, `ja`, `en`, `ko`, `zh`
- 출력 번역 언어 선택 지원
  - `ko`, `en`, `ja`
- 실행 장치 선택
  - CPU
  - GPU (CUDA)
- VAD 세부 설정 지원
  - 사용 여부
  - 최소 침묵 시간
  - speech pad
- 추론 튜닝 지원
  - `compute_type`
  - `CPU threads`
  - `num_workers`
  - 작업 완료 후 모델 자동 해제
- 파일 1개씩 순차 처리
- 작업 중 파일 추가 가능
- 기존 `.srt` 존재 시 덮어쓰기 없이 스킵
- 번역용 Gemini API 키는 Windows Credential Manager에 저장
- 로그 파일 저장 및 UI 로그 표시 지원
- 음성 분할: 긴 음성 파일을 끝 시간만 입력해 여러 파일로 나누기 (음질·표지·태그 유지, 무음 지점 자동 보정)
- 번역만 하기: 이미 있는 자막 파일(`.srt`, `.vtt`, `.txt`) 번역

## 음성 분할

`음성 분할` 탭에서 긴 음성 파일 하나를 여러 구간 파일로 나눕니다.

- 입력: `.mp3`, `.m4a`, `.mp4`, `.flac`, `.wav` (영상이 있는 `.mp4`는 음성만 `.m4a`로 저장)
- 구간 수(2 이상)를 정하고, 각 구간의 **끝 시간만** 입력합니다 (`mm:ss`, `mm:ss.xx`, `h:mm:ss`). 첫 구간은 00:00부터, 마지막 구간은 파일 끝까지입니다.
- 구간별 제목은 파일 이름(`01 제목.mp3`)과 태그 제목으로 쓰입니다. 비워 두면 `원본이름 01.mp3`가 됩니다.
- mp3·m4a·wav는 다시 인코딩하지 않고 원본 오디오 데이터를 그대로 잘라 담습니다. flac은 무손실로 다시 인코딩해 샘플 단위로 정확히 자릅니다(음질 변화 없음, 파일 크기는 조금 다를 수 있음). 어느 쪽이든 표지 이미지와 아티스트·앨범 등 태그가 유지되고, 트랙 번호는 `1/n` 형식으로 기록됩니다. 원본 전체에만 맞는 태그(전체 길이, 큐시트, 리플레이게인)는 복사하지 않습니다.
- `무음 지점으로 자동 보정`을 켜면 입력한 시간 앞뒤(기본 ±0.5초)에서 가장 조용한 지점을 찾아 자릅니다. 대사나 소리 중간에서 잘리는 것을 줄여 줍니다. 보정 결과는 로그에 표시됩니다.
- 결과는 원본 옆 `<원본이름> 분할` 폴더에 저장되며, 같은 이름의 파일이 있으면 덮어쓰지 않고 ` (2)`를 붙입니다.
- mp3·m4a·wav는 원본 데이터를 자르는 방식이라 자르는 위치가 오디오 프레임 단위(약 0.02초)로 맞춰집니다.
- 파일에 기록된 길이가 실제와 다르면(일부 VBR mp3) 실제 오디오 길이를 기준으로 나눕니다. 실패하거나 취소되면 만들던 파일은 지웁니다.

## 출력 규칙

- 출력 파일은 원본과 같은 폴더에 저장됩니다.
- 파일명은 원본 basename 기준 `.srt`입니다.
- 예:
  - `D:\media\a.mp4 -> D:\media\a.srt`
  - `E:\audio\b.mp3 -> E:\audio\b.srt`
- 기존 `.srt`가 있으면 스킵합니다.
- 번역을 사용하면 번역 전 원문 자막도 `<basename>.<원문 언어 코드>.srt`로 함께 저장합니다. (예: `a.ja.srt`)
- 번역에 실패하면 원문 자막만 저장하고 실패로 표시합니다. 다시 시작하면 원문 자막(`a.ja.srt`, 예전 버전의 `a.jp.srt`)을 재사용해 음성 인식 없이 번역만 다시 시도합니다.
- 일부 줄만 번역에 실패하면 그 줄은 원문으로 남기고 완료 메시지에 줄 수를 표시합니다.

## SRT 형식

- 인덱스는 1부터 시작
- 시간 형식은 `HH:MM:SS,mmm`
- 구분자는 `-->`
- 블록 사이 빈 줄 1개
- UTF-8 BOM으로 저장

## 폴더 구조

```text
Dongeum sub maker/
  app/
    app_logging.py
    audio_splitter.py
    config.py
    credential_store.py
    cuda_runtime.py
    deepl_translator.py
    dictionary_pack.py
    file_queue.py
    gemini_translator.py
    japanese_postprocess.py
    main.py
    model_manager.py
    srt_writer.py
    subtitle_document.py
    transcriber.py
    translator_store.py
    ui.py
    workers.py
  build.ps1
  build-onefile.ps1
  requirements.txt
  LICENSE.md
  README.md
```

## 캐시 및 저장 위치

앱 폴더가 아니라 Windows 사용자 계정 쪽에 저장되므로, 앱을 지우고 다시 받아도 설정·키·모델이 그대로 남습니다.

- 앱 데이터 폴더: `%LOCALAPPDATA%\DongeumSubMaker`
- Whisper 모델: `models\faster-whisper-<모델>` (다운로드 캐시 `hf-cache`)
- GPU용 CUDA 파일: `cuda-runtime` (받은 원본 `cuda-cache`)
- 일본어 사전팩: `dictionary-pack` (받은 원본 `dictionary-cache`)
- 로그: `logs\app.log` (자동 순환), 치명적 오류 기록 `logs\fatal-error.log`
- 번역 설정(모델, 청크 크기, 프롬프트, 번역 노트 등): `translator-settings.json`
- Gemini / DeepL API 키: Windows 자격 증명 관리자 → Windows 자격 증명 → 일반 자격 증명의 `DongeumSubMaker/...` 항목 (키가 많으면 `#2`, `#3`… 으로 나눠 저장)

완전히 초기화하려면 위 폴더를 지우고, 자격 증명 관리자에서 `DongeumSubMaker/...` 항목을 제거하세요.

## 로깅

- 파일 로그와 UI 로그를 모두 지원합니다.
- 파일 로그는 `INFO`, `WARNING`, `ERROR`, `DEBUG`를 기록합니다.
- UI 로그는 중요한 로그 위주로 표시합니다.
- 로그 로테이션이 적용됩니다.

## 번역 처리

- Gemini 요청 간 지연 시간 설정 지원
- adaptive throttling 적용
- 429 / 일시 장애 시 다른 키·모델로 즉시 전환하고, 모두 막히면 60초씩 최대 3번 대기 후 그 파일의 Gemini 번역을 중단 (할당량 소진 시 남은 파일은 Gemini를 건너뜀)
- 키 오류·없는 모델은 청크를 쪼개 재시도하지 않고 바로 다음 후보로 전환, 404 모델은 이후 요청에서 제외
- DeepL 폴백을 켜면 Gemini가 번역하지 못한 줄만 DeepL로 채움
- 검열(안전 필터)·응답 형식 오류는 키를 바꿔 재시도하지 않고 청크를 반씩 나눠 문제 줄만 찾아낸 뒤, 그 줄만 다른 모델로 한 번씩 시도 (실패 시 원문 유지 또는 DeepL)
- 기본 모델 순서는 무료 하루 한도가 큰 `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` → `gemini-3.6-flash` → `gemini-3-flash-preview` (무료 한도는 모델별로 따로 계산)
- 요청마다 입력/출력/thinking 토큰 수를 로그로 남기고 작업 끝에 누적 합계 표시
- Gemini 호출은 `google-genai` SDK 사용. 번역 설정의 추론 레벨이 실제 thinking 수준(`thinking_level`)으로 적용되며, 모델이 해당 수준을 지원하지 않으면 한 단계씩 올리고 끝내 안 되면 모델 기본값을 사용
- 여러 API 키를 순환 사용
- 응답의 각 줄에 원문을 그대로 되돌려 받아(`<o>원문</o>번역`) 같은 ID의 원문과 대조하고, 줄이 밀린 부분은 그 줄만 다시 요청
- 청크마다 직전 대사 몇 줄을 번역하지 않는 참고 문맥으로 함께 전달

## 라이선스

이 프로젝트의 라이선스는 MIT License입니다. 자세한 내용은 [LICENSE.md](LICENSE.md)를 참고하세요.
