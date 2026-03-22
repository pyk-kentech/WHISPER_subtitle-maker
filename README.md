# Dongeum Sub Maker

Windows 전용 GUI 프로그램입니다. `.mp3`, `.mp4` 파일에서 자막을 만들고, 필요하면 번역까지 이어서 처리할 수 있습니다.

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

## 출력 규칙

- 출력 파일은 원본과 같은 폴더에 저장됩니다.
- 파일명은 원본 basename 기준 `.srt`입니다.
- 예:
  - `D:\media\a.mp4 -> D:\media\a.srt`
  - `E:\audio\b.mp3 -> E:\audio\b.srt`
- 기존 `.srt`가 있으면 스킵합니다.

## SRT 형식

- 인덱스는 1부터 시작
- 시간 형식은 `HH:MM:SS,mmm`
- 구분자는 `-->`
- 블록 사이 빈 줄 1개
- UTF-8 BOM으로 저장

## 실행 방법

```powershell
cd "d:\PotPlayer\Dongeum sub maker"
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m app.main
```

최초 실행 시 선택한 Whisper 모델이 자동 다운로드됩니다.

## 빌드 방법

```powershell
cd "d:\PotPlayer\Dongeum sub maker"
.\build.ps1
```

`one-file` 빌드:

```powershell
cd "d:\PotPlayer\Dongeum sub maker"
.\build-onefile.ps1
```

## 폴더 구조

```text
Dongeum sub maker/
  app/
    app_logging.py
    config.py
    credential_store.py
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

- 앱 데이터 폴더: `%LOCALAPPDATA%\DongeumSubMaker`
- Whisper 모델 캐시: `%LOCALAPPDATA%\DongeumSubMaker\models\...`
- CUDA 런타임 캐시: `%LOCALAPPDATA%\DongeumSubMaker\cuda-runtime`
- 사전팩 캐시: `%LOCALAPPDATA%\DongeumSubMaker\dictionary-pack`
- 로그 파일: `%LOCALAPPDATA%\DongeumSubMaker\logs\app.log`
- 번역 설정: `%LOCALAPPDATA%\DongeumSubMaker\translator-settings.json`
- Gemini API 키: Windows Credential Manager

## 로깅

- 파일 로그와 UI 로그를 모두 지원합니다.
- 파일 로그는 `INFO`, `WARNING`, `ERROR`, `DEBUG`를 기록합니다.
- UI 로그는 중요한 로그 위주로 표시합니다.
- 로그 로테이션이 적용됩니다.

## 번역 처리

- Gemini 요청 간 지연 시간 설정 지원
- adaptive throttling 적용
- 429 / quota 발생 시 exponential backoff 적용
- 여러 API 키를 순환 사용

## 라이선스

이 프로젝트의 라이선스는 MIT License입니다. 자세한 내용은 [LICENSE.md](LICENSE.md)를 참고하세요.
