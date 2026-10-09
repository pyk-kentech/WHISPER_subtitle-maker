"""MainWindow 테스트(오프스크린): 버튼 활성화, 모델별 언어·VAD 고정과 복원, Runpod 비용 확인."""
from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtWidgets import QMessageBox

import app.ui as ui
from app.config import DEFAULT_VAD_MIN_SILENCE_MS, WHISPERSEG_DEFAULT_MIN_SILENCE_MS
from app.file_queue import STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED
from app.remote_runpod import GPU_CHOICES, RemoteSettings, RunpodApi, fallback_price

SRT = "1\n00:00:01,000 --> 00:00:02,000\nおはよう\n"


def select(combo, value) -> None:
    index = combo.findData(value)
    assert index >= 0, value
    combo.setCurrentIndex(index)


def add_subtitle(window, tmp_path: Path, name: str = "a.srt") -> str:
    path = tmp_path / name
    path.write_text(SRT, encoding="utf-8")
    window.add_subtitle_files([str(path)])
    return str(path.resolve())


# ----- 3. 자막 번역 탭 '번역 시작' 버튼 -----

def test_subtitle_start_disabled_without_items(window):
    window.update_controls()
    assert not window.subtitle_start_button.isEnabled()
    assert not window.subtitle_remove_button.isEnabled()


def test_subtitle_start_enabled_with_pending_item(window, tmp_path):
    add_subtitle(window, tmp_path)
    assert window.subtitle_start_button.isEnabled()
    assert window.subtitle_remove_button.isEnabled()


def test_subtitle_start_disabled_while_subtitle_worker_runs(window, tmp_path, fake_worker_cls):
    add_subtitle(window, tmp_path)
    window._subtitle_translation_worker = fake_worker_cls(running=True)
    window.update_controls()
    assert not window.subtitle_start_button.isEnabled()
    window._subtitle_translation_worker = fake_worker_cls(running=False)
    window.update_controls()
    assert window.subtitle_start_button.isEnabled()


def test_subtitle_start_independent_of_pipeline_and_download(window, tmp_path, fake_worker_cls):
    add_subtitle(window, tmp_path)
    window._download_worker = fake_worker_cls(running=True)
    window._pipeline_worker = fake_worker_cls(running=True)
    window.update_controls()
    assert window.subtitle_start_button.isEnabled()
    # 메인 작업 쪽은 막힌다.
    assert not window.start_button.isEnabled()
    assert not window.model_combo.isEnabled()


def test_subtitle_start_follows_item_status(window, tmp_path):
    key = add_subtitle(window, tmp_path)
    for status, expected in ((STATUS_SKIPPED, False), (STATUS_FAILED, True)):
        window._subtitle_items[key].status = status
        window.update_controls()
        assert window.subtitle_start_button.isEnabled() is expected, status
    window.update_subtitle_item_status(key, STATUS_DONE)  # 완료되면 목록에서 빠진다
    assert not window._subtitle_items
    assert not window.subtitle_start_button.isEnabled()


# ----- 메인 '시작' 버튼과 실행 위치 -----

def test_main_start_requires_model_or_remote(window, tmp_path):
    media = tmp_path / "a.mp3"
    media.write_bytes(b"fake")
    window.add_files([str(media)])
    assert not window._model_ready
    assert not window.start_button.isEnabled()

    select(window.execution_combo, "runpod")
    assert window.current_execution() == "runpod"
    assert window.start_button.isEnabled()  # 원격은 로컬 모델이 없어도 된다
    assert not window.runtime_combo.isEnabled()
    assert "Runpod" in window.main_settings_summary_label.text()

    select(window.execution_combo, "local")
    assert not window.start_button.isEnabled()
    assert window.runtime_combo.isEnabled()


# ----- 4. 모델별 언어·VAD 고정과 복원 -----

def test_default_model_is_anime_whisper_with_constraints_applied(window):
    """기본 모델(anime-whisper)은 시작할 때부터 언어·VAD·구간별 인식이 고정돼 있어야 한다."""
    assert window.current_model_key() == "anime-whisper"
    assert window.current_input_language() == "ja"
    assert window.is_vad_enabled() and window.current_segmentation() == "clips"
    assert not window.vad_mode_combo.isEnabled() and not window.segmentation_combo.isEnabled()
    assert window.current_vad_backend() == "whisperseg"


def test_anime_whisper_locks_language_vad_and_segmentation_then_restores(window):
    select(window.model_combo, "medium")
    select(window.input_language_combo, "auto")
    select(window.vad_mode_combo, False)
    select(window.segmentation_combo, "normal")
    window.update_controls()
    assert window.vad_mode_combo.isEnabled() and window.input_language_combo.isEnabled()

    select(window.model_combo, "anime-whisper")
    assert window.current_input_language() == "ja"
    assert window.is_vad_enabled() is True
    assert window.current_segmentation() == "clips"
    assert not window.vad_mode_combo.isEnabled()
    assert not window.segmentation_combo.isEnabled()
    assert not window.input_language_combo.isEnabled()
    assert window.vad_backend_combo.isEnabled()  # VAD 종류는 고를 수 있다
    assert window.current_vad_settings().segmentation == "clips"

    select(window.model_combo, "medium")
    assert window.current_input_language() == "auto"
    assert window.is_vad_enabled() is False
    assert window.current_segmentation() == "normal"
    assert window.vad_mode_combo.isEnabled()
    assert window.input_language_combo.isEnabled()
    assert not window.segmentation_combo.isEnabled()  # VAD OFF로 돌아왔으므로
    assert window._language_before_model_lock is None
    assert window._vad_before_model_lock is None
    assert window._segmentation_before_model_lock is None


def test_restore_keeps_default_choices(window):
    before = (window.current_input_language(), window.is_vad_enabled(), window.current_segmentation())
    select(window.model_combo, "anime-whisper")
    select(window.model_combo, "large-v3")
    assert (window.current_input_language(), window.is_vad_enabled(), window.current_segmentation()) == before


def test_vad_off_disables_vad_detail_controls(window):
    select(window.model_combo, "medium")
    select(window.vad_mode_combo, False)
    assert not window.vad_backend_combo.isEnabled()
    assert not window.segmentation_combo.isEnabled()
    assert not window.vad_min_silence_spin.isEnabled()
    select(window.vad_mode_combo, True)
    assert window.vad_backend_combo.isEnabled() and window.segmentation_combo.isEnabled()


def test_vad_backend_switch_adjusts_default_min_silence(window):
    assert window.current_vad_backend() == "whisperseg"
    assert window.vad_min_silence_spin.value() == WHISPERSEG_DEFAULT_MIN_SILENCE_MS
    select(window.vad_backend_combo, "silero")
    assert window.vad_min_silence_spin.value() == DEFAULT_VAD_MIN_SILENCE_MS
    select(window.vad_backend_combo, "whisperseg")
    assert window.vad_min_silence_spin.value() == WHISPERSEG_DEFAULT_MIN_SILENCE_MS


def test_vad_backend_switch_keeps_custom_min_silence(window):
    window.vad_min_silence_spin.setValue(300)
    select(window.vad_backend_combo, "silero")
    assert window.vad_min_silence_spin.value() == 300
    select(window.vad_backend_combo, "whisperseg")
    assert window.vad_min_silence_spin.value() == 300


def test_current_vad_settings_reflects_ui(window):
    select(window.vad_backend_combo, "silero")
    window.vad_speech_pad_spin.setValue(321)
    vad = window.current_vad_settings()
    assert vad.backend == "silero" and vad.speech_pad_ms == 321 and vad.min_silence_duration_ms == 500


# ----- 8. Runpod 비용 확인 -----

@pytest.fixture
def remote_mocks(monkeypatch):
    state = {"price": 0.31, "price_calls": [], "probe_calls": []}

    def fake_probe(paths):
        state["probe_calls"].append(paths)
        return 3600.0

    def fake_gpu_price(self, gpu_id, cloud):
        state["price_calls"].append((gpu_id, cloud))
        return state["price"]

    monkeypatch.setattr(ui, "probe_audio_seconds", fake_probe)
    monkeypatch.setattr(RunpodApi, "gpu_price", fake_gpu_price)
    return state


def test_confirm_remote_cost_auto_without_key(window, message_boxes, remote_mocks):
    result = window.confirm_remote_cost(RemoteSettings(mode="auto", api_key=""), ["a.mp3"])
    assert result is False
    assert message_boxes.kinds() == ["warning"]
    assert window.tabs.currentIndex() == window.tabs.count() - 1  # Runpod 탭으로 이동
    assert remote_mocks["price_calls"] == [] and remote_mocks["probe_calls"] == []


def test_confirm_remote_cost_user_says_no(window, message_boxes, remote_mocks):
    message_boxes.answer = QMessageBox.No
    assert window.confirm_remote_cost(RemoteSettings(api_key="k"), ["a.mp3"]) is False
    assert message_boxes.kinds() == ["question"]


def test_confirm_remote_cost_yes_returns_live_price(window, message_boxes, remote_mocks):
    message_boxes.answer = QMessageBox.Yes
    result = window.confirm_remote_cost(RemoteSettings(api_key="k", cloud_type="SECURE"), ["a.mp3"])
    assert result == pytest.approx(0.31) and isinstance(result, float)
    assert remote_mocks["price_calls"] == [(GPU_CHOICES[0][0], "SECURE")]
    assert "$0.31" in message_boxes.calls[0][1]
    assert "1.0시간" in message_boxes.calls[0][1]


def test_confirm_remote_cost_yes_with_fallback_price(window, message_boxes, remote_mocks):
    remote_mocks["price"] = None
    message_boxes.answer = QMessageBox.Yes
    result = window.confirm_remote_cost(RemoteSettings(api_key="k"), ["a.mp3"])
    assert result == pytest.approx(fallback_price(GPU_CHOICES[0][0], "COMMUNITY"))
    assert "참고 가격" in message_boxes.calls[0][1]


def test_confirm_remote_cost_manual_mode(window, message_boxes, remote_mocks):
    message_boxes.answer = QMessageBox.Yes
    settings = RemoteSettings(mode="manual", manual_url="https://w", manual_token="t")
    assert window.confirm_remote_cost(settings, ["a.mp3"]) is None
    assert remote_mocks["price_calls"] == []  # 수동 연결은 가격 조회 없음


def test_collect_remote_settings_from_tab(window):
    window.runpod_key_edit.setText("  rp-key  ")
    select(window.remote_mode_combo, "manual")
    window.remote_manual_url_edit.setText("https://w")
    settings = window.collect_remote_settings()
    assert settings.api_key == "rp-key" and settings.mode == "manual" and settings.manual_url == "https://w"
    assert not window.runpod_key_edit.isEnabled() and window.remote_manual_url_edit.isEnabled()
