"""공용 픽스처: 오프스크린 Qt, 임시 앱 데이터 폴더, 메모리 자격 증명 저장소."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# PySide6를 불러오기 전에 정해야 한다.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_app_data(tmp_path, monkeypatch):
    """실제 %LOCALAPPDATA%\\DongeumSubMaker 대신 임시 폴더를 쓴다."""
    data_root = tmp_path / "appdata"
    data_root.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(data_root))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_root))
    return data_root / "DongeumSubMaker"


@pytest.fixture(autouse=True)
def memory_secrets(monkeypatch):
    """Windows 자격 증명 관리자 대신 메모리 dict를 쓴다."""
    store: dict[str, str] = {}

    def save_secret(name, secret, user_name="DongeumSubMaker"):
        store[name] = secret

    def load_secret(name):
        return store.get(name, "")

    def delete_secret(name):
        store.pop(name, None)

    import app.credential_store as credential_store
    import app.translator_store as translator_store

    for module in (credential_store, translator_store):
        monkeypatch.setattr(module, "save_secret", save_secret)
        monkeypatch.setattr(module, "load_secret", load_secret)
        monkeypatch.setattr(module, "delete_secret", delete_secret)
    if "app.ui" in sys.modules:
        ui = sys.modules["app.ui"]
        monkeypatch.setattr(ui, "save_secret", save_secret)
        monkeypatch.setattr(ui, "load_secret", load_secret)
        monkeypatch.setattr(ui, "delete_secret", delete_secret)
    return store


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class MessageBoxRecorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.answer = None  # question()이 돌려줄 값

    def question(self, _parent, _title, text, *args, **kwargs):
        self.calls.append(("question", text))
        return self.answer

    def warning(self, _parent, _title, text, *args, **kwargs):
        from PySide6.QtWidgets import QMessageBox

        self.calls.append(("warning", text))
        return QMessageBox.Ok

    def information(self, _parent, _title, text, *args, **kwargs):
        from PySide6.QtWidgets import QMessageBox

        self.calls.append(("information", text))
        return QMessageBox.Ok

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.calls]


@pytest.fixture
def message_boxes(monkeypatch):
    """QMessageBox 정적 함수는 오프스크린에서 멈추므로 바로 돌려주게 바꾼다."""
    from PySide6.QtWidgets import QMessageBox

    recorder = MessageBoxRecorder()
    recorder.answer = QMessageBox.No
    monkeypatch.setattr(QMessageBox, "question", staticmethod(recorder.question))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(recorder.warning))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(recorder.information))
    return recorder


@pytest.fixture
def window(qapp, monkeypatch, message_boxes, memory_secrets):
    """실제 자격 증명·GPU·다운로드 없이 MainWindow를 만든다."""
    import app.ui as ui
    from app.app_logging import set_ui_log_callback
    from app.remote_runpod import RemoteSettings
    from app.translator_store import TranslatorSettings

    monkeypatch.setattr(ui, "load_api_keys", lambda: [])
    monkeypatch.setattr(ui, "load_deepl_api_key", lambda: "")
    monkeypatch.setattr(ui, "load_openai_api_key", lambda: "")
    monkeypatch.setattr(ui, "load_runpod_api_key", lambda: "")
    monkeypatch.setattr(ui, "load_secret", lambda name: "")
    monkeypatch.setattr(ui, "save_secret", lambda name, secret, user_name="": None)
    monkeypatch.setattr(ui, "delete_secret", lambda name: None)
    monkeypatch.setattr(ui, "load_translator_settings", lambda: TranslatorSettings())
    monkeypatch.setattr(ui, "load_remote_settings", lambda path: RemoteSettings())
    monkeypatch.setattr(ui, "get_available_runtime_choices", lambda: [("cpu", "CPU")])
    monkeypatch.setattr(ui, "get_default_runtime_choice", lambda: "cpu")

    win = ui.MainWindow(auto_download_on_startup=False)
    yield win
    win._download_worker = None
    win._pipeline_worker = None
    win._subtitle_translation_worker = None
    win._split_worker = None
    win._allow_close = True
    win.close()
    win.deleteLater()
    set_ui_log_callback(None)
    qapp.processEvents()


class FakeRunningWorker:
    """isRunning()만 흉내 내는 작업 스레드 대역."""

    def __init__(self, running: bool = True) -> None:
        self._running = running
        self._pause_requested = False

    def isRunning(self) -> bool:
        return self._running


@pytest.fixture
def fake_worker_cls():
    return FakeRunningWorker
