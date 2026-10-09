"""Runpod 원격 실행 테스트: 비용 추정, Pod 본문, REST 클라이언트, 워커 클라이언트, 세션 정리(모두 MockTransport)."""
from __future__ import annotations

import base64
import json
import time as real_time
import zlib

import httpx
import pytest

import app.remote_runpod as rr
from app.remote_runpod import (
    GPU_CHOICES,
    RemoteError,
    RemoteSession,
    RemoteSettings,
    RunpodApi,
    WorkerClient,
    build_pod_body,
    estimate_cost,
    fallback_price,
    prepare_cost_confirmation,
)


class FakeTime:
    """remote_runpod.time 대역: sleep만 기록하고 나머지는 실제 time을 쓴다."""

    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def __getattr__(self, name):
        return getattr(real_time, name)


@pytest.fixture
def fake_time(monkeypatch):
    fake = FakeTime()
    monkeypatch.setattr(rr, "time", fake)
    return fake


def mock_client(handler, **kwargs) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), **kwargs)


# ----- 가격·비용 -----

def test_fallback_price_table():
    gpu_id, _label, prices = GPU_CHOICES[0]
    assert fallback_price(gpu_id, "COMMUNITY") == prices["COMMUNITY"]
    assert fallback_price(gpu_id, "SECURE") == prices["SECURE"]
    assert fallback_price(gpu_id, "UNKNOWN") == max(prices.values())
    assert fallback_price("no-such-gpu", "COMMUNITY") == 1.0


def test_estimate_cost_without_and_with_llm():
    est = estimate_cost(0.6, "src", audio_hours=1.0, use_llm=False, max_hours=6.0)
    assert est.minutes == pytest.approx(rr.BOOT_MINUTES + 60 * rr.ASR_RTF)
    assert est.cost == pytest.approx(0.6 * est.minutes / 60)
    assert est.max_cost == pytest.approx(3.6)
    with_llm = estimate_cost(0.6, "src", audio_hours=1.0, use_llm=True, max_hours=6.0)
    assert with_llm.minutes == pytest.approx(est.minutes + rr.LLM_MINUTES_PER_AUDIO_HOUR + 5)


def test_prepare_cost_confirmation_uses_live_price():
    calls = []

    def lookup(gpu, cloud):
        calls.append((gpu, cloud))
        return 0.31

    settings = RemoteSettings(api_key="k", cloud_type="SECURE")
    estimate, message = prepare_cost_confirmation(settings, 7200, lookup)
    assert calls == [(settings.gpu_type_id, "SECURE")]
    assert estimate.price_per_hour == 0.31
    assert estimate.price_source == "Runpod 실시간 가격"
    assert "$0.31" in message and "2.0시간" in message


def test_prepare_cost_confirmation_falls_back_when_lookup_fails():
    settings = RemoteSettings(api_key="k")
    estimate, message = prepare_cost_confirmation(settings, 3600, lambda _g, _c: None)
    assert estimate.price_per_hour == fallback_price(settings.gpu_type_id, settings.cloud_type)
    assert "참고 가격" in estimate.price_source
    assert "참고 가격" in message


def test_prepare_cost_confirmation_without_lookup_uses_fallback():
    estimate, _message = prepare_cost_confirmation(RemoteSettings(api_key="k"), 0, None)
    assert estimate.price_per_hour == fallback_price(GPU_CHOICES[0][0], "COMMUNITY")


def test_prepare_cost_confirmation_manual_mode_has_no_estimate():
    def lookup(_g, _c):
        raise AssertionError("수동 연결은 가격을 조회하지 않는다")

    settings = RemoteSettings(mode="manual", manual_url="https://abc-8000.proxy.runpod.net")
    estimate, message = prepare_cost_confirmation(settings, 1800, lookup)
    assert estimate is None
    assert "https://abc-8000.proxy.runpod.net" in message


# ----- Pod 본문 -----

def test_build_pod_body_without_llm():
    body = build_pod_body(RemoteSettings(idle_minutes=20, max_hours=3.5, llm_choice="none"), "tok")
    env = body["env"]
    assert body["name"].startswith(rr.POD_NAME_PREFIX)
    assert body["gpuTypeIds"] == [GPU_CHOICES[0][0]] and body["gpuCount"] == 1
    assert body["ports"] == [f"{rr.WORKER_PORT}/http"]
    assert body["containerDiskInGb"] == 25
    assert env["DSM_TOKEN"] == "tok" and env["DSM_IDLE_MINUTES"] == "20" and env["DSM_MAX_HOURS"] == "3.5"
    assert "DSM_LLM_REPO" not in env
    boot = zlib.decompress(base64.b64decode(env["DSM_BOOT"])).decode("utf-8")
    assert boot == rr.worker_source()


def test_build_pod_body_with_llm():
    body = build_pod_body(RemoteSettings(llm_choice="gemma4-26b"), "tok")
    assert body["containerDiskInGb"] == 40
    assert body["env"]["DSM_LLM_REPO"] == rr.LLM_CHOICES["gemma4-26b"]["repo"]
    assert body["env"]["DSM_LLM_ALIAS"] == "gemma4-26b"


# ----- RunpodApi -----

def test_runpod_api_requires_key():
    with pytest.raises(RemoteError):
        RunpodApi("  ")


def test_runpod_api_requests_and_errors():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.headers.get("authorization")))
        if request.method == "POST" and request.url.path == "/v1/pods":
            return httpx.Response(200, json={"id": "p1", "body": json.loads(request.content)["name"]})
        if request.url.path == "/v1/pods/gone":
            return httpx.Response(404)
        if request.url.path == "/v1/pods/bad":
            return httpx.Response(500, text="server exploded")
        if request.url.path == "/v1/pods/p1":
            return httpx.Response(200, json={"id": "p1"})
        if request.url.path == "/v1/pods":
            return httpx.Response(200, json={"pods": [{"id": "p1"}]})
        return httpx.Response(401)

    api = RunpodApi(" key ", client=mock_client(handler))
    assert api.create_pod({"name": "dsm-remote-x"}) == {"id": "p1", "body": "dsm-remote-x"}
    assert api.get_pod("p1") == {"id": "p1"}
    assert api.get_pod("gone") is None
    assert api.list_pods() == [{"id": "p1"}]
    api.delete_pod("gone")  # 404는 이미 지워진 것으로 본다
    with pytest.raises(RemoteError, match="500"):
        api.delete_pod("bad")
    with pytest.raises(RemoteError, match="401"):
        api.get_pod("other")
    assert all(auth == "Bearer key" for _m, _p, auth in seen)


def test_runpod_api_gpu_price():
    def handler(request: httpx.Request) -> httpx.Response:
        query = json.loads(request.content)["query"]
        if "missing" in query:
            return httpx.Response(200, json={"data": {"gpuTypes": []}})
        if "broken" in query:
            return httpx.Response(502, text="<html>bad gateway</html>")
        return httpx.Response(200, json={"data": {"gpuTypes": [{"id": "g", "communityPrice": 0.22, "securePrice": 0.5}]}})

    api = RunpodApi("key", client=mock_client(handler))
    assert api.gpu_price("g", "COMMUNITY") == 0.22
    assert api.gpu_price("g", "SECURE") == 0.5
    assert api.gpu_price("missing", "COMMUNITY") is None
    assert api.gpu_price("broken", "COMMUNITY") is None


# ----- WorkerClient.transcribe -----

class FakeWorker:
    """remote_worker.py 흉내: 업로드(오프셋 이어 받기) -> 시작 -> 진행 조회 -> 삭제."""

    def __init__(self, final_status: str = "done", resume_at: int | None = None) -> None:
        self.final_status = final_status
        self.resume_at = resume_at
        self.audio = b""
        self.calls: list[tuple[str, str]] = []
        self.polls = 0
        self.tokens: set[str] = set()
        self.started_options = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        self.tokens.add(request.headers.get("x-dsm-token", ""))
        path = request.url.path
        if request.method == "PUT" and path.endswith("/audio"):
            offset = int(request.url.params["offset"])
            if self.resume_at is not None and offset == 0 and not self.audio:
                # 이전 연결에서 일부를 이미 받아 둔 상태를 흉내 낸다.
                self.audio = self.pending[: self.resume_at]
                return httpx.Response(409, json={"size": len(self.audio)})
            if offset != len(self.audio):
                return httpx.Response(409, json={"size": len(self.audio)})
            self.audio += request.content
            return httpx.Response(200, json={"size": len(self.audio)})
        if request.method == "POST" and path.endswith("/start"):
            self.started_options = json.loads(request.content)
            return httpx.Response(200, json={"ok": True})
        if request.method == "GET" and "/jobs/" in path:
            self.polls += 1
            if self.polls == 1:
                return httpx.Response(200, json={"status": "running", "detail": "loading model", "progress": 10, "position": 1.0, "duration": 10.0})
            if self.final_status == "error":
                return httpx.Response(200, json={"status": "error", "error": "CUDA OOM"})
            return httpx.Response(
                200,
                json={
                    "status": "done",
                    "language": "ja",
                    "audio_deleted": True,
                    "segments": [{"start": 0.5, "end": 1.5, "text": "おはよう"}],
                    "duration": 10.0,
                    "progress": 100,
                    "position": 10.0,
                },
            )
        if request.method == "DELETE":
            return httpx.Response(200, json={"audio_exists": False})
        return httpx.Response(404)


def make_worker_client(fake: FakeWorker, monkeypatch, sleeps: list[float]) -> WorkerClient:
    monkeypatch.setattr(rr, "UPLOAD_PART_BYTES", 4)
    client = WorkerClient("https://pod-8000.proxy.runpod.net/", "secret-token", sleep=sleeps.append)
    client._http.close()
    client._http = mock_client(fake, headers={"X-DSM-Token": "secret-token"})
    return client


def test_worker_transcribe_upload_poll_delete(tmp_path, monkeypatch):
    audio = tmp_path / "audio.ogg"
    audio.write_bytes(b"0123456789")
    fake = FakeWorker()
    sleeps: list[float] = []
    client = make_worker_client(fake, monkeypatch, sleeps)
    progress: list[tuple] = []
    logs: list[str] = []

    segments, language = client.transcribe(audio, {"model_key": "medium"}, lambda *a: progress.append(a), logs.append)

    assert fake.audio == b"0123456789"
    methods = [method for method, _ in fake.calls]
    assert methods == ["PUT", "PUT", "PUT", "POST", "GET", "GET", "DELETE"]
    assert fake.started_options == {"model_key": "medium"}
    assert [(s.start, s.end, s.text) for s in segments] == [(0.5, 1.5, "おはよう")]
    assert language == "ja"
    assert progress[0] == (10, 1.0, 10.0) and progress[-1] == (100, 10.0, 10.0)
    assert sleeps == [2.0]
    assert "원격: loading model" in logs
    assert fake.tokens == {"secret-token"}
    assert client.llm_base_url == "https://pod-8000.proxy.runpod.net/v1"


def test_worker_transcribe_resumes_upload_after_409(tmp_path, monkeypatch):
    audio = tmp_path / "audio.ogg"
    audio.write_bytes(b"abcdefghij")
    fake = FakeWorker(resume_at=6)
    fake.pending = b"abcdefghij"
    client = make_worker_client(fake, monkeypatch, [])
    client.transcribe(audio, {}, None, lambda _m: None)
    assert fake.audio == b"abcdefghij"
    puts = [path for method, path in fake.calls if method == "PUT"]
    assert len(puts) == 2  # 409 응답 1번 + 남은 4바이트 1번


def test_worker_transcribe_error_still_deletes_job(tmp_path, monkeypatch):
    audio = tmp_path / "audio.ogg"
    audio.write_bytes(b"xyz")
    fake = FakeWorker(final_status="error")
    client = make_worker_client(fake, monkeypatch, [])
    with pytest.raises(RemoteError, match="CUDA OOM"):
        client.transcribe(audio, {}, None, lambda _m: None)
    assert fake.calls[-1][0] == "DELETE"


def test_worker_transcribe_cancel_still_deletes_job(tmp_path, monkeypatch):
    audio = tmp_path / "audio.ogg"
    audio.write_bytes(b"xyz")
    fake = FakeWorker()
    client = make_worker_client(fake, monkeypatch, [])
    with pytest.raises(RemoteError, match="취소"):
        client.transcribe(audio, {}, None, lambda _m: None, cancel_check=lambda: True)
    assert fake.calls[-1][0] == "DELETE"
    assert not any(method == "GET" for method, _ in fake.calls)


def test_worker_rejects_bad_token(monkeypatch):
    client = WorkerClient("https://w", "t")
    client._http = mock_client(lambda _r: httpx.Response(401))
    with pytest.raises(RemoteError, match="401"):
        client.health()


# ----- RemoteSession -----

class FakeWorkerHandle:
    def __init__(self) -> None:
        self.shutdown_called = False
        self.closed = False

    def shutdown(self) -> None:
        self.shutdown_called = True

    def close(self) -> None:
        self.closed = True


def test_session_close_deletes_pod_and_verifies(fake_time):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(404)

    logs: list[str] = []
    worker = FakeWorkerHandle()
    session = RemoteSession(RemoteSettings(api_key="k"), logs.append, api=RunpodApi("k", client=mock_client(handler)))
    session.pod_id = "pod1"
    session.worker = worker
    session.cost_per_hour = 0.3
    session.close()

    assert calls == [("DELETE", "/v1/pods/pod1"), ("GET", "/v1/pods/pod1")]
    assert session.pod_id is None
    assert any("삭제 완료" in line for line in logs)
    assert not any("경고" in line for line in logs)
    assert worker.closed and not worker.shutdown_called
    assert fake_time.sleeps == []


def test_session_close_warns_when_deletion_unconfirmed(fake_time):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "DELETE":
            return httpx.Response(204)
        return httpx.Response(200, json={"id": "pod1", "desiredStatus": "RUNNING"})

    logs: list[str] = []
    worker = FakeWorkerHandle()
    session = RemoteSession(RemoteSettings(api_key="k"), logs.append, api=RunpodApi("k", client=mock_client(handler)))
    session.pod_id = "pod1"
    session.worker = worker
    session.close()

    assert calls.count("DELETE") == 5
    assert session.pod_id == "pod1"
    assert worker.shutdown_called and worker.closed  # 워커에게 스스로 지우라고 요청
    assert any("경고: Pod pod1 삭제를 확인하지 못했습니다" in line for line in logs)
    assert fake_time.sleeps == [3, 6, 9, 12, 15]


def test_session_close_retries_after_api_error(fake_time):
    state = {"deletes": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "DELETE":
            state["deletes"] += 1
            return httpx.Response(500, text="try later") if state["deletes"] == 1 else httpx.Response(200)
        return httpx.Response(404)

    logs: list[str] = []
    session = RemoteSession(RemoteSettings(api_key="k"), logs.append, api=RunpodApi("k", client=mock_client(handler)))
    session.pod_id = "pod1"
    session.close()
    assert session.pod_id is None
    assert any("재시도 1/5" in line for line in logs)
    assert fake_time.sleeps == [3]


def test_session_start_rejects_pod_more_expensive_than_confirmed():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        return httpx.Response(200, json={"id": "p9", "costPerHr": 0.9})

    session = RemoteSession(RemoteSettings(api_key="k"), lambda _m: None, api=RunpodApi("k", client=mock_client(handler)))
    with pytest.raises(RemoteError, match="취소"):
        session.start(max_price=0.22)
    assert session.pod_id == "p9"  # close()가 이 Pod를 지울 수 있게 남는다
    assert session.worker is None


def test_session_manual_mode_requires_url_and_token():
    session = RemoteSession(RemoteSettings(mode="manual", manual_url="https://w"), lambda _m: None)
    with pytest.raises(RemoteError):
        session.start()


def test_session_manual_mode_uses_worker_and_never_touches_pods(monkeypatch):
    calls = []
    state = {"bundle": False}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/health":
            stage = "ready" if state["bundle"] else "waiting_bundle"
            return httpx.Response(200, json={"stage": stage, "gpu": "RTX 3090"})
        if request.url.path == "/bundle":
            state["bundle"] = True
            return httpx.Response(200)
        return httpx.Response(404)

    class MockWorkerClient(WorkerClient):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._http.close()
            self._http = mock_client(handler, headers={"X-DSM-Token": self.token})

    monkeypatch.setattr(rr, "WorkerClient", MockWorkerClient)
    monkeypatch.setattr(rr, "RunpodApi", lambda *_a, **_k: pytest.fail("수동 연결은 Runpod API를 쓰지 않는다"))
    logs: list[str] = []
    settings = RemoteSettings(mode="manual", manual_url="https://w-8000.proxy.runpod.net", manual_token="tok")
    session = RemoteSession(settings, logs.append)
    worker = session.start()
    assert worker.base_url == "https://w-8000.proxy.runpod.net"
    assert calls == [("GET", "/health"), ("PUT", "/bundle"), ("GET", "/health")]
    assert session.pod_id is None and session.api is None
    session.close()
    assert calls[-1] == ("GET", "/health")  # 닫을 때 Pod 삭제·종료 요청 없음


def test_remote_settings_save_load_skips_secrets(tmp_path):
    path = tmp_path / "remote.json"
    rr.save_remote_settings(path, RemoteSettings(api_key="secret", manual_token="tok", idle_minutes=30, max_hours=2.5))
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "api_key" not in data and "manual_token" not in data
    loaded = rr.load_remote_settings(path)
    assert loaded.idle_minutes == 30 and loaded.max_hours == 2.5 and loaded.api_key == ""
    assert rr.load_remote_settings(tmp_path / "missing.json") == RemoteSettings()


def test_probe_audio_seconds_ignores_unreadable(tmp_path):
    bogus = tmp_path / "x.mp3"
    bogus.write_bytes(b"not audio")
    assert rr.probe_audio_seconds([bogus, tmp_path / "missing.mp3"]) == 0.0

