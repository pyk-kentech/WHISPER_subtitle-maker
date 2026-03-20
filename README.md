# Dongeum Sub Maker

일본어 음성 파일(`.mp3`, `.mp4`)에서 `.srt` 자막을 생성하는 **Windows 전용 GUI 프로그램**입니다.

본 프로그램은 기존 PotPlayer 프로젝트와는 별도로 개발되었으며,
고정된 설정(`Faster-Whisper medium / Japanese`)으로 동작합니다.

---

## ✨ 주요 기능

* 🎧 입력 지원: `.mp3`, `.mp4`
* 📄 출력 형식: `.srt` (원본 파일과 동일한 폴더)
* 🏷 파일 이름: `원본파일명.srt`
* 🌏 언어: 일본어 (`ja`) 고정
* 🤖 모델: `Systran/faster-whisper-medium` 고정
* ⚙ 실행 장치 선택:

  * CPU
  * GPU (CUDA)
* 🔄 처리 방식:

  * 파일 단위 **순차 처리**
  * 작업 중에도 파일 추가 가능 (큐 자동 이어서 처리)
* 🚫 기존 자막 파일 존재 시:

  * 덮어쓰기 금지
  * 자동 스킵

---

## 📁 프로젝트 구조

```text
WHISPER_subtitle-maker/
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

---

## 🧩 요구 사항

* Python 3.11 (권장)
* PySide6
* faster-whisper
* huggingface_hub
* tqdm
* PyInstaller

---

## 🚀 실행 방법

### 1. 코드 다운로드

```bash
git clone https://github.com/pyk-kentech/WHISPER_subtitle-maker.git
cd WHISPER_subtitle-maker
```

### 2. 가상환경 생성 및 실행

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m app.main
```

---

## 📦 빌드 방법

### 일반 빌드

```powershell
.\build.ps1
```

### 단일 EXE 빌드 (One-file)

```powershell
.\build-onefile.ps1
```

📌 결과물 위치:

```
dist\DongeumSubMaker
```

* `--windowed` 옵션 적용 (콘솔 없이 실행되는 GUI)

---

## 🤖 모델 다운로드 및 캐싱

* 모델은 저장소에 포함되지 않음
* 최초 실행 시 자동 다운로드
* 캐시 경로:

  ```
  %LOCALAPPDATA%\DongeumSubMaker\models\faster-whisper-medium
  ```

### 특징

* 1회 다운로드 후 재사용
* GUI에서 다운로드 진행률 표시
* 실패 시 재시도 버튼 제공

---

## 📝 자막 생성 프로세스

1. 파일 드래그 앤 드롭 또는 선택
2. 출력 `.srt` 경로 자동 생성
3. 기존 `.srt` 존재 시 → 스킵
4. 없는 파일만 순차 처리
5. Faster-Whisper로 음성 인식 수행
6. SRT 형식으로 저장

---

## 📄 SRT 출력 규칙

* 인덱스: 1부터 시작
* 시간 형식: `HH:MM:SS,mmm`
* 구분자: `-->`
* 블록 간 빈 줄 1개
* 인코딩: UTF-8 BOM

---

## 🧠 내부 구조

| 파일                 | 설명                        |
| ------------------ | ------------------------- |
| `file_queue.py`    | 파일 정규화, 중복 제거, 확장자 필터링    |
| `model_manager.py` | 모델 다운로드 및 캐시 관리           |
| `transcriber.py`   | Faster-Whisper 로딩 및 음성 인식 |
| `srt_writer.py`    | SRT 포맷 생성 및 저장            |
| `ui.py`            | GUI (드래그 앤 드롭, 리스트, 로그 등) |
| `workers.py`       | 백그라운드 스레드 및 작업 관리         |

---

## ⚡ 성능 최적화

* GPU 사용 시:

  * CUDA 가능 여부 확인
  * `float16` 우선 사용
* CPU 사용 시:

  * `int8` 우선 사용
  * CPU 스레드 수를 약간 제한하여 시스템 응답성 유지
* 처리 방식:

  * 항상 **파일 단위 순차 처리**
* GUI 표시:

  * 전체 진행 상태
  * 현재 파일 진행률




---
