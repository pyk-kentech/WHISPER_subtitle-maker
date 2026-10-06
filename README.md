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

## 실행 방법

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m app.main
```

최초 실행 시 선택한 Whisper 모델이 자동 다운로드됩니다.

## 빌드 방법

```powershell
.\build.ps1
```

`one-file` 빌드:

```powershell
.\build-onefile.ps1
```

## 폴더 구조

```text
Dongeum sub maker/
  app/
    app_logging.py
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
- 429 / 일시 장애 시 다른 키·모델로 즉시 전환하고, 모두 막히면 60초씩 최대 3번 대기 후 그 파일의 Gemini 번역을 중단 (할당량 소진 시 남은 파일은 Gemini를 건너뜀)
- 키 오류·없는 모델은 청크를 쪼개 재시도하지 않고 바로 다음 후보로 전환, 404 모델은 이후 요청에서 제외
- DeepL 폴백을 켜면 Gemini가 번역하지 못한 줄만 DeepL로 채움
- 검열(안전 필터)·응답 형식 오류는 키를 바꿔 재시도하지 않고 청크를 반씩 나눠 문제 줄만 찾아낸 뒤, 그 줄만 다른 모델로 한 번씩 시도 (실패 시 원문 유지 또는 DeepL)
- 기본 모델 순서는 무료 하루 한도가 큰 `gemini-3.5-flash-lite` → `gemini-3.1-flash-lite` → `gemini-3.6-flash` → `gemini-3-flash-preview` (무료 한도는 모델별로 따로 계산)
- 요청마다 입력/출력/thinking 토큰 수를 로그로 남기고 작업 끝에 누적 합계 표시
- 여러 API 키를 순환 사용
- 응답의 각 줄에 원문을 그대로 되돌려 받아(`<o>원문</o>번역`) 같은 ID의 원문과 대조하고, 줄이 밀린 부분은 그 줄만 다시 요청
- 청크마다 직전 대사 몇 줄을 번역하지 않는 참고 문맥으로 함께 전달

## 라이선스

이 프로젝트의 라이선스는 MIT License입니다. 자세한 내용은 [LICENSE.md](LICENSE.md)를 참고하세요.
