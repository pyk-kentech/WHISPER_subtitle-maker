# Dongeum Sub Maker

Windows 전용 일본어 음성 자막 생성 GUI 프로그램입니다.

`.mp3`, `.mp4` 파일을 넣으면 일본어 음성을 인식해서 같은 폴더에 `.srt` 자막 파일을 생성합니다. 기존 PotPlayer 프로젝트와는 분리된 별도 프로그램입니다.

## 핵심 기능

- 입력 파일: `.mp3`, `.mp4`
- 출력 파일: 원본과 같은 폴더, 같은 basename의 `.srt`
- 언어 고정: Japanese (`ja`)
- 모델 고정: `Systran/faster-whisper-medium`
- 실행 장치 선택: CPU / GPU(CUDA)
- 처리 방식: 파일 1개씩 순차 처리
- 진행 중 파일 추가 가능: 현재 파일 완료 후 대기열에 자동 이어서 처리
- 기존 `.srt` 존재 시 덮어쓰기 금지, 스킵 처리
- 모델 파일은 저장소에 포함하지 않음
- 최초 실행 시 모델 자동 다운로드, 이후 로컬 캐시 재사용

## GUI 구성

- 드래그 앤 드롭 영역
- 파일 추가 버튼
- 파일 목록
- 시작 버튼
- 모델 다운로드 진행률
- 전체 대기열 진행률
- 현재 파일 진행률
- 로그 영역
- 파일별 상태 표시: 대기중 / 처리중 / 완료 / 실패 / 스킵

## 처리 방식

1. 사용자가 `.mp3` 또는 `.mp4` 파일을 추가합니다.
2. 각 파일에 대해 같은 폴더의 `.srt` 경로를 계산합니다.
3. 기존 `.srt`가 있으면 `기존 자막 파일 존재 -> 스킵`으로 처리합니다.
4. 없는 파일만 Faster-Whisper로 순차 처리합니다.
5. 결과를 SRT 규격으로 저장합니다.

중요:

- 여러 파일을 한꺼번에 병렬 처리하지 않습니다.
- 항상 한 파일씩 처리한 뒤 다음 파일로 넘어갑니다.
- 작업 중에도 새 파일을 대기열에 추가할 수 있습니다.

## 모델 다운로드 / 캐시

- 모델은 `%LOCALAPPDATA%\DongeumSubMaker\models\faster-whisper-medium`에 저장됩니다.
- Hugging Face Hub에서 최초 1회 자동 다운로드합니다.
- 다운로드 완료 전에는 변환을 시작할 수 없습니다.
- 다운로드 실패 시 GUI에서 재시도할 수 있습니다.

## 실행 최적화

- GPU 선택 시 CUDA 가능 여부를 확인하고 `float16` 우선으로 사용합니다.
- CPU 선택 시 `int8` 우선으로 사용합니다.
- CPU 스레드는 시스템 응답성을 해치지 않도록 제한합니다.
- 변환은 안정성 우선으로 파일 단위 순차 처리합니다.

## 프로젝트 구조

```text
app/
  config.py
  file_queue.py
  main.py
  model_manager.py
  srt_writer.py
  transcriber.py
  ui.py
  workers.py
build.ps1
requirements.txt
requirement.txt
README.md
```

## 의존성

- Python 3.11 권장
- PySide6
- faster-whisper
- huggingface_hub
- tqdm
- PyInstaller

## 실행 방법

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m app.main
```

## 빌드 방법

```powershell
.\build.ps1
```

빌드 결과물은 `dist\DongeumSubMaker` 아래에 생성됩니다.

## SRT 형식

- 인덱스는 1부터 시작
- 시간 형식은 `HH:MM:SS,mmm`
- 구분자는 `-->`
- 블록 사이 빈 줄 1개
- UTF-8 BOM 저장
