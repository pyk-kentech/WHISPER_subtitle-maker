"""NetworkWaiter / is_network_down_error 테스트 (가짜 시계·sleep 사용)."""
from __future__ import annotations

import socket
from types import SimpleNamespace
from urllib import error as urllib_error

import httpx
import pytest

import app.gemini_translator as gt
from app.gemini_translator import (
    NetworkUnavailableError,
    NetworkWaiter,
    TransientServiceError,
    is_network_down_error,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(gt, "time", SimpleNamespace(monotonic=fake.monotonic))
    return fake


def connect_error() -> httpx.ConnectError:
    return httpx.ConnectError("[Errno 11001] getaddrinfo failed")


class FlakyRequest:
    """앞의 failures번은 예외를 던지고 그다음부터 성공한다."""

    def __init__(self, failures: int, exc_factory=connect_error, result="ok") -> None:
        self.failures = failures
        self.exc_factory = exc_factory
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.failures:
            raise self.exc_factory()
        return self.result


# ----- is_network_down_error -----

@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("boom"),
        httpx.ConnectTimeout("slow"),
        socket.gaierror(11001, "getaddrinfo failed"),
        urllib_error.URLError(socket.gaierror(-2, "Name or service not known")),
        urllib_error.URLError(ConnectionRefusedError()),
        RuntimeError("Temporary failure in name resolution"),
        OSError("Network is unreachable"),
    ],
)
def test_network_down_detected(exc):
    assert is_network_down_error(exc)


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("bad value"),
        httpx.ReadTimeout("read timeout"),
        RuntimeError("HTTP 500: internal"),
        urllib_error.URLError("some other reason"),
    ],
)
def test_other_errors_not_network_down(exc):
    assert not is_network_down_error(exc)


def test_network_down_found_through_cause_chain():
    try:
        try:
            raise httpx.ConnectError("dns")
        except httpx.ConnectError as inner:
            raise TransientServiceError("wrapped") from inner
    except TransientServiceError as outer:
        assert is_network_down_error(outer)


def test_cyclic_exception_chain_terminates():
    first = ValueError("a")
    second = ValueError("b")
    first.__context__ = second
    second.__context__ = first
    assert not is_network_down_error(first)


# ----- NetworkWaiter -----

def test_waits_and_retries_until_connection_returns(clock):
    logs: list[str] = []
    waiter = NetworkWaiter(logs.append, max_wait_seconds=600, retry_seconds=(5, 10, 20))
    request = FlakyRequest(failures=3)

    assert waiter.run(request, clock.sleep) == "ok"
    assert request.calls == 4
    assert clock.sleeps == [5, 10, 20]
    assert any("돌아왔습니다" in line for line in logs)
    # 성공 뒤에는 상태가 초기화된다.
    assert waiter._outage_started is None and waiter._gave_up is False


def test_gives_up_after_max_wait(clock):
    waiter = NetworkWaiter(lambda _m: None, max_wait_seconds=60, retry_seconds=(5, 10, 20, 30, 60))
    request = FlakyRequest(failures=10_000)

    with pytest.raises(NetworkUnavailableError):
        waiter.run(request, clock.sleep)
    # 마지막 대기는 남은 시간(60-35=25초)으로 잘린다.
    assert clock.sleeps == [5, 10, 20, 25]
    assert sum(clock.sleeps) == 60
    assert request.calls == 5
    assert waiter._gave_up is True


def test_fails_fast_after_giving_up_until_success(clock):
    waiter = NetworkWaiter(lambda _m: None, max_wait_seconds=30, retry_seconds=(10,))
    with pytest.raises(NetworkUnavailableError):
        waiter.run(FlakyRequest(failures=10_000), clock.sleep)
    sleeps_before = len(clock.sleeps)

    # 포기한 뒤: 기다리지 않고 바로 실패한다.
    again = FlakyRequest(failures=10_000)
    with pytest.raises(NetworkUnavailableError):
        waiter.run(again, clock.sleep)
    assert again.calls == 1
    assert len(clock.sleeps) == sleeps_before

    # 한 번 성공하면 연결이 돌아온 것으로 보고 다시 기다린다.
    assert waiter.run(FlakyRequest(failures=0), clock.sleep) == "ok"
    assert waiter._gave_up is False
    recovered = FlakyRequest(failures=1)
    assert waiter.run(recovered, clock.sleep) == "ok"
    assert recovered.calls == 2
    assert len(clock.sleeps) == sleeps_before + 1


def test_non_network_error_propagates_immediately(clock):
    waiter = NetworkWaiter(lambda _m: None, max_wait_seconds=600, retry_seconds=(5,))
    request = FlakyRequest(failures=1, exc_factory=lambda: ValueError("HTTP 400"))
    with pytest.raises(ValueError):
        waiter.run(request, clock.sleep)
    assert request.calls == 1
    assert clock.sleeps == []


def test_non_network_error_marks_online_after_give_up(clock):
    waiter = NetworkWaiter(lambda _m: None, max_wait_seconds=0, retry_seconds=(5,))
    with pytest.raises(NetworkUnavailableError):
        waiter.run(FlakyRequest(failures=1), clock.sleep)
    assert waiter._gave_up is True
    # 서버까지 닿은 오류(예: 400)는 연결이 살아 있다는 뜻이다.
    with pytest.raises(ValueError):
        waiter.run(FlakyRequest(failures=1, exc_factory=lambda: ValueError("x")), clock.sleep)
    assert waiter._gave_up is False


def test_wrapped_transport_error_is_waited_on(clock):
    """번역기가 ConnectError를 TransientServiceError로 감싸도 원인 체인을 보고 기다린다."""

    def wrapped():
        try:
            raise httpx.ConnectError("getaddrinfo failed")
        except httpx.ConnectError as exc:
            raise TransientServiceError(str(exc)) from exc

    waiter = NetworkWaiter(lambda _m: None, max_wait_seconds=600, retry_seconds=(7,))
    request = FlakyRequest(failures=2, exc_factory=lambda: _capture(wrapped))
    assert waiter.run(request, clock.sleep) == "ok"
    assert clock.sleeps == [7, 7]


def _capture(func):
    try:
        func()
    except Exception as exc:  # noqa: BLE001
        return exc
    raise AssertionError("예외가 나야 한다")
