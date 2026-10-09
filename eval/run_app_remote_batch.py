"""Drive the real app (offscreen) to process folders with the Runpod remote mode, like a user would:
add folders -> Start -> confirm cost -> wait. Prints ONLY counts (no file names, no subtitle text).
Usage (app .venv, project root):
  python eval/run_app_remote_batch.py <worker_url> <token_file> <llm_choice:role|none>  (role = primary | fallback) <stop_epoch> <folder> [<folder> ...]
stop_epoch: unix time after which no new file is started (the current one is finished)."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

import app.ui as U  # noqa: E402
from app.file_queue import STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED  # noqa: E402


def main() -> None:
    worker_url, token_file, llm_spec, stop_epoch = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    llm_choice, _, llm_role = llm_spec.partition(":")
    folders = sys.argv[5:]
    token = Path(token_file).read_text(encoding="utf-8").strip()
    QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.Ok)
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.Ok)
    app = QApplication([])
    window = U.MainWindow(auto_download_on_startup=False)
    # 앱 일반 사용과 같게 설정: 실행 위치 Runpod(수동 연결), 모델 anime-whisper, ASMR VAD 구간별
    window.execution_combo.setCurrentIndex(window.execution_combo.findData("runpod"))
    window.model_combo.setCurrentIndex(window.model_combo.findData("anime-whisper"))
    window.vad_backend_combo.setCurrentIndex(window.vad_backend_combo.findData("whisperseg"))
    window.remote_mode_combo.setCurrentIndex(window.remote_mode_combo.findData("manual"))
    window.remote_manual_url_edit.setText(worker_url)
    window.remote_manual_token_edit.setText(token)
    window.remote_llm_combo.setCurrentIndex(window.remote_llm_combo.findData(llm_choice))
    window.remote_llm_role_combo.setCurrentIndex(window.remote_llm_role_combo.findData(llm_role or "primary"))
    window.include_subdirs_checkbox.setChecked(True)
    window.add_files(folders)
    total = len(window._items)
    print(f"queued files: {total}", flush=True)
    started = time.time()
    state = {"stopping": False}

    def tick() -> None:
        worker = window._pipeline_worker
        if not state["stopping"] and time.time() > stop_epoch and worker is not None and worker.isRunning():
            # 예산 한도: 지금 파일까지만 끝내고 남은 대기열을 비운다.
            state["stopping"] = True
            pending = [p for p, it in window._items.items() if it.status not in {STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED}]
            removed, _blocked = worker.remove_paths(pending)
            print(f"budget stop: removed {len(removed)} pending files from the queue", flush=True)
        if worker is not None and not worker.isRunning() and not window._processing:
            app.quit()

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(5000)
    window.start_pipeline()
    app.exec()
    counts = {"done": window._session_success_count, "failed": window._session_failure_count, "skipped": window._session_skipped_count}
    print(f"result: {counts} of {total} queued, {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
