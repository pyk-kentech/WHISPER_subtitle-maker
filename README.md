# Dongeum Sub Maker

일본어 음성 `.mp3` / `.mp4` 파일에서 `.srt` 자막을 생성하는 Windows 전용 GUI 프로그램입니다.

이 프로그램은 기존 PotPlayer 프로젝트와 분리된 별도 프로그램이며, `Faster-Whisper-XXL / medium / japanese` 고정 설정으로 동작합니다.

## 핵심 동작

- 지원 입력: `.mp3`, `.mp4`
- 출력 위치: 각 원본 파일과 같은 폴더
- 출력 이름: 원본 basename + `.srt`
- 언어 고정: Japanese (`ja`)
- 모델 고정: `Systran/faster-whisper-medium`
- 실행 장치 선택: CPU 또는 GPU(CUDA)
- 처리 방식: 파일 1개씩 순차 처리
- 진행 중 추가: 작업 중에도 새 파일을 대기열에 추가 가능, 현재 파일 완료 후 자동 이어서 처리
- 기존 `.srt` 존재 시: 덮어쓰기 금지, 해당 파일 스킵

## 폴더 구조

```text
Dongeum sub maker/
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
  README.md
```

## requirement

- Python 3.11 권장
- PySide6
- faster-whisper
- huggingface_hub
- tqdm
- PyInstaller

## 실행 방법

```powershell
cd "d:\PotPlayer\Dongeum sub maker"
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m app.main
```

최초 실행 시 모델이 자동 다운로드됩니다. 다운로드가 완료되기 전까지는 변환 시작 버튼이 비활성화됩니다.

## 빌드 방법

```powershell
cd "d:\PotPlayer\Dongeum sub maker"
.\build.ps1
```

빌드 결과물은 `dist\DongeumSubMaker` 아래에 생성됩니다. `--windowed` 옵션으로 콘솔 없이 실행되는 Windows GUI EXE를 기준으로 구성했습니다.

## 모델 다운로드 및 캐싱 방식

- 모델 파일은 저장소에 포함하지 않습니다.
- 로컬 캐시는 `%LOCALAPPDATA%\DongeumSubMaker\models\faster-whisper-medium` 경로를 사용합니다.
- Hugging Face Hub에서 최초 1회 자동 다운로드합니다.
- 이후 실행에서는 캐시된 모델을 재사용합니다.
- 다운로드 진행률은 GUI 진행바와 상태 텍스트에 표시됩니다.
- 다운로드 실패 시 GUI의 재시도 버튼으로 다시 받을 수 있습니다.

## 자막 생성 로직

1. 사용자가 `.mp3` / `.mp4` 파일을 드래그 앤 드롭하거나 파일 선택으로 추가합니다.
2. 각 파일마다 같은 폴더의 `.srt` 경로를 계산합니다.
3. 이미 `.srt`가 있으면 `기존 자막 파일 존재 -> 스킵`으로 처리합니다.
4. 없는 파일만 Faster-Whisper로 순차 변환합니다.
5. 인식 결과를 SRT 규격 `HH:MM:SS,mmm` 형식으로 저장합니다.

## SRT 형식

출력은 다음 규칙을 지킵니다.

- 인덱스는 1부터 시작
- 시간은 `HH:MM:SS,mmm`
- 구분자는 `-->`
- 블록 사이 빈 줄 1개
- UTF-8 BOM으로 저장

## 구조 설명

- `app/file_queue.py`: 입력 파일 정규화, 중복 제거, 지원 확장자 필터링
- `app/model_manager.py`: 모델 다운로드 계획, 진행률 수집, 로컬 캐시 관리
- `app/transcriber.py`: Faster-Whisper 고정 설정 로딩 및 음성 인식
- `app/srt_writer.py`: 안전한 SRT 포맷 생성과 파일 저장
- `app/ui.py`: 드래그 앤 드롭, 파일 목록, 로그, 버튼, 상태 표시
- `app/workers.py`: 다운로드/변환 백그라운드 스레드와 요약 집계

## 실행 최적화

- GPU 선택 시 CUDA 사용 가능 여부를 확인하고 `float16` 우선으로 로딩합니다.
- CPU 선택 시 `int8` 우선으로 로딩하고 CPU 스레드는 전체 코어보다 약간 낮게 제한해 시스템 응답성을 유지합니다.
- 변환은 항상 파일 단위 순차 처리로 고정합니다.
- GUI에서 현재 몇 번째 파일을 처리 중인지와 현재 파일 진행률을 함께 표시합니다.
