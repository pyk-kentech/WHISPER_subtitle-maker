"""WhisperSeg 후처리(probs_to_segments, split_long_segment) 테스트. 프레임 = 20ms."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.whisperseg_vad import FRAME_MS, WhisperSegOptions, probs_to_segments, split_long_segment

FPS = 1000 // FRAME_MS  # 초당 50프레임


def frames(*parts: tuple[float, int]) -> np.ndarray:
    """(확률, 프레임 수) 묶음을 이어 붙인다."""
    return np.concatenate([np.full(count, value, dtype=np.float32) for value, count in parts])


def opts(**kwargs) -> WhisperSegOptions:
    base = dict(threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=100, speech_pad_ms=0)
    base.update(kwargs)
    return WhisperSegOptions(**base)


def spans(segments: list[dict]) -> list[tuple[float, float]]:
    return [(seg["start"], seg["end"]) for seg in segments]


def test_default_options():
    o = WhisperSegOptions()
    assert o.threshold == 0.5 and o.neg_threshold is None
    assert o.min_silence_duration_ms == 100 and o.speech_pad_ms == 200
    assert math.isinf(o.max_speech_duration_s)


# ----- 히스테리시스 -----

def test_basic_segment_without_padding():
    probs = frames((0.0, 50), (0.9, 100), (0.0, 100))
    assert spans(probs_to_segments(probs, opts())) == [(1.0, 3.0)]


def test_speech_starts_at_threshold_inclusive():
    assert probs_to_segments(frames((0.0, 10), (0.49, 100), (0.0, 20)), opts()) == []
    assert spans(probs_to_segments(frames((0.0, 10), (0.5, 100), (0.0, 20)), opts())) == [(0.2, 2.2)]


def test_dip_between_neg_and_threshold_does_not_end_speech():
    # 0.4는 threshold(0.5)보다 낮지만 neg_threshold(0.35)보다 높다.
    probs = frames((0.9, 100), (0.4, 50), (0.9, 100), (0.0, 50))
    assert spans(probs_to_segments(probs, opts())) == [(0.0, 5.0)]


def test_dip_below_neg_threshold_ends_speech():
    probs = frames((0.9, 100), (0.3, 50), (0.9, 100), (0.0, 50))
    assert spans(probs_to_segments(probs, opts())) == [(0.0, 2.0), (3.0, 5.0)]


def test_short_silence_shorter_than_min_silence_is_bridged():
    # 4프레임(80ms) 무음 < 최소 무음 100ms -> 하나로 이어진다.
    probs = frames((0.9, 100), (0.0, 4), (0.9, 100), (0.0, 50))
    assert spans(probs_to_segments(probs, opts())) == [(0.0, 4.08)]


def test_long_silence_splits_and_end_is_first_silent_frame():
    probs = frames((0.9, 100), (0.0, 10), (0.9, 100), (0.0, 50))
    assert spans(probs_to_segments(probs, opts())) == [(0.0, 2.0), (2.2, 4.2)]


def test_min_silence_option_changes_bridging():
    probs = frames((0.9, 100), (0.0, 10), (0.9, 100), (0.0, 50))
    # 최소 무음 500ms(25프레임)이면 10프레임 무음은 이어 붙인다.
    assert spans(probs_to_segments(probs, opts(min_silence_duration_ms=500))) == [(0.0, 4.2)]


def test_neg_threshold_explicit_and_clamped():
    probs = frames((0.9, 100), (0.4, 50), (0.9, 100), (0.0, 50))
    # neg_threshold를 0.45로 올리면 0.4 구간에서 끊긴다.
    assert spans(probs_to_segments(probs, opts(neg_threshold=0.45))) == [(0.0, 2.0), (3.0, 5.0)]
    # threshold가 낮으면 neg_threshold는 0.01 아래로 내려가지 않는다.
    low = frames((0.2, 100), (0.005, 50))
    assert spans(probs_to_segments(low, opts(threshold=0.1))) == [(0.0, 2.0)]


def test_min_speech_filters_short_bursts():
    probs = frames((0.0, 20), (0.9, 10), (0.0, 50), (0.9, 20), (0.0, 50))
    # 10프레임(200ms) < 250ms 는 버리고 20프레임(400ms)은 남긴다.
    assert spans(probs_to_segments(probs, opts())) == [(1.6, 2.0)]


def test_trailing_speech_until_end_of_audio():
    probs = frames((0.0, 50), (0.9, 100))
    assert spans(probs_to_segments(probs, opts())) == [(1.0, 3.0)]
    # 끝에 걸린 짧은 발화는 버린다.
    assert probs_to_segments(frames((0.0, 50), (0.9, 5)), opts()) == []


def test_empty_and_silent_input():
    assert probs_to_segments(np.zeros(0, dtype=np.float32), opts()) == []
    assert probs_to_segments(np.zeros(500, dtype=np.float32), opts()) == []


# ----- 패딩 -----

def test_padding_extends_and_clamps_to_audio_bounds():
    probs = frames((0.0, 2), (0.9, 100), (0.0, 50))
    # 앞은 0초에서 멈추고, 뒤는 200ms 늘어난다.
    assert spans(probs_to_segments(probs, opts(speech_pad_ms=200))) == [(0.0, 2.24)]
    tail = frames((0.0, 50), (0.9, 100))
    assert spans(probs_to_segments(tail, opts(speech_pad_ms=200))) == [(0.8, 3.0)]


def test_padding_clamps_against_neighbours():
    probs = frames((0.0, 20), (0.9, 50), (0.0, 10), (0.9, 50), (0.0, 20))
    segments = spans(probs_to_segments(probs, opts(speech_pad_ms=200)))
    assert segments == [(0.2, 1.6), (1.6, 2.8)]


def test_padding_never_overlaps_with_small_gap():
    probs = frames((0.0, 30), (0.9, 50), (0.0, 7), (0.9, 50), (0.0, 30))
    segments = spans(probs_to_segments(probs, opts(speech_pad_ms=200)))
    assert len(segments) == 2
    assert segments[0][1] <= segments[1][0]


# ----- 최대 길이(12초) 분할 -----

def test_long_segment_split_at_lowest_probability_frame():
    probs = np.full(40 * FPS, 0.9, dtype=np.float32)
    probs[450] = 0.55  # 첫 창 [300, 600) 안의 가장 낮은 프레임
    probs[700] = 0.51  # 두 번째 창 [750, 1050) 밖이므로 쓰지 않는다
    probs[900] = 0.6
    segments = spans(probs_to_segments(probs, opts(max_speech_duration_s=12)))
    assert segments[0] == (0.0, 9.0)
    assert segments[1] == (9.0, 18.0)
    assert segments[-1][1] == 40.0
    for (_s1, e1), (s2, _e2) in zip(segments, segments[1:]):
        assert e1 == s2  # 이어 붙으면 원래 구간이 된다
    assert all(end - start <= 12.0 + 1e-9 for start, end in segments)


def test_split_never_exceeds_max_on_random_input():
    rng = np.random.default_rng(1234)
    probs = rng.uniform(0.5, 1.0, size=180 * FPS).astype(np.float32)
    probs[: 2 * FPS] = 0.0
    segments = probs_to_segments(probs, opts(max_speech_duration_s=12, speech_pad_ms=200))
    assert segments
    assert all(seg["end"] - seg["start"] <= 12.0 + 1e-9 for seg in segments)
    assert all(seg["end"] - seg["start"] >= 6.0 - 1e-9 for seg in segments[:-1])


def test_short_segments_left_alone_by_max_duration():
    probs = frames((0.0, 50), (0.9, 5 * FPS), (0.0, 50))
    with_max = spans(probs_to_segments(probs, opts(max_speech_duration_s=12)))
    without = spans(probs_to_segments(probs, opts()))
    assert with_max == without == [(1.0, 6.0)]


def test_no_split_when_max_is_infinite():
    probs = np.full(40 * FPS, 0.9, dtype=np.float32)
    assert spans(probs_to_segments(probs, opts())) == [(0.0, 40.0)]


# ----- split_long_segment -----

def test_split_long_segment_exact_max_unchanged():
    probs = np.ones(1000, dtype=np.float32)
    assert split_long_segment([100, 700], probs, 600) == [[100, 700]]


def test_split_long_segment_one_over_max():
    probs = np.ones(1000, dtype=np.float32)
    parts = split_long_segment([0, 601], probs, 600)
    assert len(parts) == 2
    # 확률이 같으면 창의 첫 프레임(max의 절반)에서 자른다.
    assert parts == [[0, 300], [300, 601]]


def test_split_long_segment_uses_absolute_offsets():
    probs = np.ones(2000, dtype=np.float32)
    probs[1500] = 0.1
    parts = split_long_segment([1000, 1900], probs, 600)
    assert parts == [[1000, 1500], [1500, 1900]]


def test_split_long_segment_many_parts_contiguous():
    rng = np.random.default_rng(7)
    probs = rng.uniform(size=5000).astype(np.float32)
    parts = split_long_segment([123, 4987], probs, 600)
    assert parts[0][0] == 123 and parts[-1][1] == 4987
    for (_a, b), (c, _d) in zip(parts, parts[1:]):
        assert b == c
    assert all(end - start <= 600 for start, end in parts)
    assert all(end - start >= 300 for start, end in parts[:-1])


@pytest.mark.parametrize("length", [3, 5, 9])
def test_split_long_segment_tiny_max(length):
    probs = np.ones(20, dtype=np.float32)
    parts = split_long_segment([0, length], probs, 2)
    assert all(0 < end - start <= 2 for start, end in parts)
    assert parts[-1][1] == length
