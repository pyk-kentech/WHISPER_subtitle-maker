from __future__ import annotations

import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av
import numpy as np


SPLIT_SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".mp4", ".flac", ".wav"}
DEFAULT_SILENCE_SEARCH_SECONDS = 0.5
MIN_SEGMENT_SECONDS = 0.5

# 원본 코덱을 다시 인코딩하지 않고 담을 수 있는 컨테이너
_OUTPUT_FORMATS = {
    "mp3": ("mp3", ".mp3"),
    "aac": ("mp4", ".m4a"),
    "alac": ("mp4", ".m4a"),
    "flac": ("flac", ".flac"),
}
_COVER_FORMATS = {"mp3", "mp4", "flac"}
_INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_RMS_WINDOW_SECONDS = 0.02
_RMS_HOP_SECONDS = 0.005
_DECODE_WARMUP_SECONDS = 0.3

ProgressCallback = Callable[[float], None]
CancelCallback = Callable[[], bool]
LogCallback = Callable[[str], None]

_WHOLE_FILE_TAGS = {
    "encoder", "major_brand", "minor_version", "compatible_brands", "tlen", "length", "duration",
    "cuesheet", "itunsmpb", "itunnorm", "track", "tracktotal", "totaltracks", "itunes_cddb_1",
}
_WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


class AudioSplitError(RuntimeError):
    pass


class AudioSplitCancelled(AudioSplitError):
    pass


@dataclass(slots=True)
class AudioInfo:
    path: Path
    duration: float
    codec: str
    sample_rate: int
    channels: int
    bit_rate: int
    has_cover: bool
    has_video: bool
    output_format: str
    output_suffix: str


@dataclass(slots=True)
class SplitResult:
    path: Path
    title: str
    start: float
    end: float


def parse_timestamp(text: str) -> float:
    """'mm:ss', 'm:ss.xx', 'h:mm:ss(.xx)' 또는 초('95.5')를 초 단위로 바꾼다."""
    value = text.strip().replace("：", ":")
    if not value:
        raise ValueError("시간이 비어 있습니다.")
    parts = value.split(":")
    if len(parts) > 3:
        raise ValueError(f"시간 형식이 올바르지 않습니다: {text}")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError(f"시간 형식이 올바르지 않습니다: {text}") from exc
    if any(number < 0 for number in numbers) or any(not math.isfinite(number) for number in numbers):
        raise ValueError(f"시간 형식이 올바르지 않습니다: {text}")
    if len(parts) > 1 and numbers[-1] >= 60:
        raise ValueError(f"초는 60보다 작아야 합니다: {text}")
    if len(parts) == 3 and numbers[1] >= 60:
        raise ValueError(f"분은 60보다 작아야 합니다: {text}")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def format_timestamp(seconds: float, precise: bool = False) -> str:
    seconds = max(0.0, seconds)
    if precise:
        centiseconds = int(round(seconds * 100))
        whole, fraction = divmod(centiseconds, 100)
    else:
        whole, fraction = int(round(seconds)), 0
    hours, remainder = divmod(whole, 3600)
    minutes, secs = divmod(remainder, 60)
    text = f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
    return f"{text}.{fraction:02d}" if precise else text


def probe_audio(path: Path) -> AudioInfo:
    try:
        container = av.open(str(path))
    except av.error.FFmpegError as exc:
        raise AudioSplitError(f"파일을 열 수 없습니다: {exc}") from exc
    with container:
        if not container.streams.audio:
            raise AudioSplitError("오디오 스트림이 없는 파일입니다.")
        audio = container.streams.audio[0]
        codec = audio.codec_context.codec.canonical_name
        cover = _find_cover_stream(container)
        has_video = any(stream is not cover for stream in container.streams.video)
        if container.duration is not None:
            duration = container.duration / av.time_base
        elif audio.duration is not None and audio.time_base is not None:
            duration = float(audio.duration * audio.time_base)
        else:
            raise AudioSplitError("파일 길이를 알 수 없습니다.")
        output_format, output_suffix = _resolve_output_format(codec)
        return AudioInfo(
            path=path,
            duration=float(duration),
            codec=codec,
            sample_rate=int(audio.codec_context.sample_rate or 0),
            channels=int(audio.codec_context.channels or 0),
            bit_rate=int(audio.codec_context.bit_rate or container.bit_rate or 0),
            has_cover=cover is not None,
            has_video=has_video,
            output_format=output_format,
            output_suffix=output_suffix,
        )


def build_cut_points(end_times: list[float], duration: float) -> list[float]:
    """사용자가 입력한 구간별 끝 시간(마지막 구간 제외)을 검증해 자를 지점 목록으로 돌려준다."""
    previous = 0.0
    for index, end in enumerate(end_times, start=1):
        if end - previous < MIN_SEGMENT_SECONDS:
            raise AudioSplitError(
                f"{index}번 구간의 끝 시간({format_timestamp(end, True)})이 "
                f"시작({format_timestamp(previous, True)})보다 {MIN_SEGMENT_SECONDS}초 이상 뒤여야 합니다."
            )
        previous = end
    if duration - previous < MIN_SEGMENT_SECONDS:
        raise AudioSplitError(
            f"{len(end_times)}번 구간의 끝 시간({format_timestamp(previous, True)})이 "
            f"파일 길이({format_timestamp(duration, True)})보다 {MIN_SEGMENT_SECONDS}초 이상 앞이어야 합니다."
        )
    return list(end_times)


def measure_audio_duration(path: Path, should_cancel: CancelCallback | None = None) -> float:
    """컨테이너가 알려주는 길이(VBR mp3 등에서 틀릴 수 있음) 대신 실제 오디오 패킷 끝을 잰다."""
    end = 0.0
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        for count, packet in enumerate(container.demux(stream)):
            if packet.pts is None or packet.size == 0:
                continue
            duration = float(packet.duration * packet.time_base) if packet.duration else 0.0
            end = max(end, float(packet.pts * packet.time_base) + duration)
            if should_cancel is not None and count % 2000 == 0 and should_cancel():
                raise AudioSplitCancelled("취소되었습니다.")
    return end


def find_quiet_cut_points(
    path: Path,
    cut_points: list[float],
    duration: float,
    search_seconds: float,
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> list[tuple[float, float, float]]:
    """각 자를 지점 앞뒤 search_seconds 안에서 가장 조용한 곳을 찾는다.

    반환값은 (보정된 지점, 원래 지점 음량 dBFS, 보정 지점 음량 dBFS) 목록이다.
    """
    if not cut_points or search_seconds <= 0:
        return [(point, float("nan"), float("nan")) for point in cut_points]

    bounds = [0.0, *cut_points, duration]
    windows: list[tuple[float, float]] = []
    for index, point in enumerate(cut_points, start=1):
        # 이웃 지점이 반대쪽으로 끝까지 움직여도 구간이 MIN_SEGMENT_SECONDS 이상 남도록 범위를 제한한다.
        before = (point - bounds[index - 1] - MIN_SEGMENT_SECONDS) / (1 if index == 1 else 2)
        after = (bounds[index + 1] - point - MIN_SEGMENT_SECONDS) / (1 if index == len(cut_points) else 2)
        limit = max(0.0, min(search_seconds, before, after))
        windows.append((point - limit, point + limit))

    samples: list[list[tuple[float, np.ndarray]]] = [[] for _ in cut_points]
    sample_rate = 0
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        decoder = stream.codec_context
        resampler = None
        window_index = 0
        skipped = False
        for count, packet in enumerate(container.demux(stream)):
            if packet.pts is None or packet.size == 0:
                continue
            if should_cancel is not None and count % 500 == 0 and should_cancel():
                raise AudioSplitCancelled("취소되었습니다.")
            start = float(packet.pts * packet.time_base)
            while window_index < len(windows) and start > windows[window_index][1] + 0.1:
                window_index += 1
            if window_index >= len(windows):
                break
            if progress_callback is not None and duration > 0:
                progress_callback(min(1.0, start / duration))
            low, _high = windows[window_index]
            if start < low - _DECODE_WARMUP_SECONDS:
                skipped = True
                continue
            if skipped:
                # 패킷을 건너뛴 뒤에만 디코더를 비운다. 이어지는 구간에서 비우면 워밍업 없이
                # 첫 프레임이 무음처럼 디코딩되어 가짜 무음 지점이 잡힌다.
                decoder.flush_buffers()
                skipped = False
            for frame in decoder.decode(packet):
                if resampler is None:
                    sample_rate = frame.sample_rate
                    resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
                frame_time = float(frame.time) if frame.time is not None else start
                for converted in resampler.resample(frame):
                    data = converted.to_ndarray().reshape(-1).astype(np.float32)
                    frame_end = frame_time + len(data) / sample_rate
                    for target, (target_low, target_high) in enumerate(windows):
                        if frame_end >= target_low - _RMS_WINDOW_SECONDS and frame_time <= target_high + _RMS_WINDOW_SECONDS:
                            samples[target].append((frame_time, data))
                    frame_time = frame_end

    results: list[tuple[float, float, float]] = []
    for point, (low, high), chunks in zip(cut_points, windows, samples):
        if not chunks or sample_rate <= 0 or high <= low:
            level = _rms_db_at(chunks, sample_rate, point)
            results.append((point, level, level))
            continue
        best_time, best_level = point, math.inf
        candidate = low
        while candidate <= high + 1e-9:
            level = _rms_db_at(chunks, sample_rate, candidate)
            score = level + 3.0 * abs(candidate - point) / max(high - low, 1e-6)
            if score < best_level:
                best_time, best_level = candidate, score
            candidate += _RMS_HOP_SECONDS
        results.append((best_time, _rms_db_at(chunks, sample_rate, point), _rms_db_at(chunks, sample_rate, best_time)))
    return results


def split_audio(
    path: Path,
    info: AudioInfo,
    titles: list[str],
    cut_points: list[float],
    output_dir: Path,
    progress_callback: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> list[SplitResult]:
    """구간별 파일로 나눈다(표지·태그 유지). 실패하거나 취소되면 이번에 만든 파일을 지운다.

    mp3/aac/wav는 원본 패킷을 다시 인코딩하지 않고 그대로 옮긴다. flac은 프레임마다 원본 기준
    샘플 번호가 기록돼 있어 그대로 옮기면 2번째 조각부터 시간이 어긋나므로, 무손실로 다시 인코딩해
    샘플 단위로 정확히 자른다(음질 변화 없음).
    """
    if len(titles) != len(cut_points) + 1:
        raise AudioSplitError("구간 제목 수와 자를 지점 수가 맞지 않습니다.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = build_output_paths(path, titles, output_dir, info.output_suffix)
    created: list[Path] = []
    try:
        if info.output_format == "flac":
            results = _split_lossless_flac(path, info, titles, cut_points, output_paths, created, progress_callback, should_cancel)
        else:
            results = _split_by_packets(path, info, titles, cut_points, output_paths, created, progress_callback, should_cancel)
        if len(results) != len(titles):
            raise AudioSplitError(f"{len(titles)}개 구간 중 {len(results)}개만 만들어졌습니다. 자르는 시간을 확인해 주세요.")
    except BaseException:
        for created_path in created:
            created_path.unlink(missing_ok=True)
        raise
    return results


def _split_by_packets(path, info, titles, cut_points, output_paths, created, progress_callback, should_cancel) -> list[SplitResult]:
    with av.open(str(path)) as source:
        audio = source.streams.audio[0]
        cover_stream, cover_bytes, base_metadata = _segment_extras(path, source, info)
        results: list[SplitResult] = []
        segment_index = -1
        output = None
        out_audio = None
        offset = 0
        segment_start = 0.0
        last_end = 0.0
        try:
            for count, packet in enumerate(source.demux(audio)):
                if packet.pts is None or packet.size == 0:
                    continue
                if should_cancel is not None and count % 500 == 0 and should_cancel():
                    raise AudioSplitCancelled("취소되었습니다.")
                packet_start = float(packet.pts * packet.time_base)
                packet_duration = float(packet.duration * packet.time_base) if packet.duration else 0.0
                target = _segment_for(packet_start + packet_duration / 2, cut_points)
                if target != segment_index:
                    if output is not None:
                        output.close()
                        output = None
                        results.append(SplitResult(output_paths[segment_index], titles[segment_index], segment_start, last_end))
                    segment_index = target
                    segment_start = packet_start if segment_index > 0 else 0.0
                    offset = packet.pts if segment_index > 0 else 0
                    created.append(output_paths[segment_index])
                    output, out_audio = _open_segment_output(
                        output_paths[segment_index], info, audio, None, cover_stream, cover_bytes, base_metadata,
                        titles[segment_index], segment_index, len(titles),
                    )
                packet.pts -= offset
                if packet.dts is not None:
                    packet.dts -= offset
                packet.stream = out_audio
                output.mux(packet)
                last_end = packet_start + packet_duration
                if progress_callback is not None and info.duration > 0:
                    progress_callback(min(1.0, packet_start / info.duration))
            if output is not None:
                output.close()
                output = None
                results.append(SplitResult(output_paths[segment_index], titles[segment_index], segment_start, last_end))
        finally:
            if output is not None:
                output.close()
    return results


def _split_lossless_flac(path, info, titles, cut_points, output_paths, created, progress_callback, should_cancel) -> list[SplitResult]:
    with av.open(str(path)) as source:
        audio = source.streams.audio[0]
        cover_stream, cover_bytes, base_metadata = _segment_extras(path, source, info)
        rate = info.sample_rate
        # PyAV 18 이하는 이 속성이 없다. 그때는 FFmpeg flac 인코더 기본값(32비트 샘플 → 24비트 기록)을 따른다.
        bits = int(getattr(audio.codec_context, "bits_per_raw_sample", 0) or 0)
        boundaries = [int(round(point * rate)) for point in cut_points]
        results: list[SplitResult] = []
        segment_index = -1
        segment_first_sample = 0
        output = None
        out_audio = None
        written = 0
        position = 0

        def open_segment(index: int):
            nonlocal output, out_audio, written, segment_first_sample
            created.append(output_paths[index])
            output, out_audio = _open_segment_output(
                output_paths[index], info, audio, bits, cover_stream, cover_bytes, base_metadata,
                titles[index], index, len(titles),
            )
            written = 0
            segment_first_sample = position

        def close_segment(index: int, end_sample: int):
            nonlocal output
            for packet in out_audio.encode(None):
                output.mux(packet)
            output.close()
            output = None
            results.append(SplitResult(output_paths[index], titles[index], segment_first_sample / rate, end_sample / rate))

        try:
            for count, frame in enumerate(source.decode(audio)):
                if should_cancel is not None and count % 200 == 0 and should_cancel():
                    raise AudioSplitCancelled("취소되었습니다.")
                data = frame.to_ndarray()
                planar = frame.format.is_planar
                channels = len(frame.layout.channels)
                total = frame.samples
                consumed = 0
                while consumed < total:
                    target = _segment_for((position + 0.5) / rate, cut_points)
                    if target != segment_index:
                        if output is not None:
                            close_segment(segment_index, position)
                        segment_index = target
                        open_segment(segment_index)
                    limit = boundaries[segment_index] if segment_index < len(boundaries) else None
                    take = total - consumed if limit is None else min(total - consumed, limit - position)
                    take = max(1, take)
                    if planar:
                        piece = data[:, consumed:consumed + take]
                    else:
                        piece = data[:, consumed * channels:(consumed + take) * channels]
                    out_frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(piece), format=frame.format.name, layout=frame.layout.name)
                    out_frame.sample_rate = rate
                    out_frame.pts = written
                    out_frame.time_base = Fraction(1, rate)
                    for packet in out_audio.encode(out_frame):
                        output.mux(packet)
                    written += take
                    consumed += take
                    position += take
                if progress_callback is not None and info.duration > 0:
                    progress_callback(min(1.0, position / rate / info.duration))
            if output is not None:
                close_segment(segment_index, position)
        finally:
            if output is not None:
                output.close()
    return results


def _segment_extras(path: Path, source, info: AudioInfo):
    cover_stream = _find_cover_stream(source)
    cover_bytes = _read_cover_packet(path) if cover_stream is not None and info.output_format in _COVER_FORMATS else None
    base_metadata = {key: value for key, value in source.metadata.items() if not _is_whole_file_tag(key)}
    return cover_stream, cover_bytes, base_metadata


def _is_whole_file_tag(key: str) -> bool:
    # 원본 전체에만 맞는 태그(전체 길이, 큐시트, 리플레이게인, 갭리스 정보 등)는 조각에 복사하지 않는다.
    lowered = key.lower()
    return lowered in _WHOLE_FILE_TAGS or lowered.startswith("replaygain_")


def build_output_paths(source: Path, titles: list[str], output_dir: Path, suffix: str) -> list[Path]:
    width = max(2, len(str(len(titles))))
    paths: list[Path] = []
    used: set[str] = set()
    for index, title in enumerate(titles, start=1):
        clean = sanitize_filename(title)
        stem = f"{index:0{width}d} {clean}" if clean else f"{source.stem} {index:0{width}d}"
        candidate = output_dir / f"{stem}{suffix}"
        counter = 2
        while candidate.exists() or candidate.name.lower() in used:
            candidate = output_dir / f"{stem} ({counter}){suffix}"
            counter += 1
        used.add(candidate.name.lower())
        paths.append(candidate)
    return paths


def default_output_dir(source: Path) -> Path:
    return source.parent / f"{source.stem} 분할"


def sanitize_filename(text: str) -> str:
    clean = _INVALID_FILENAME_CHARS.sub("_", text.strip()).rstrip(" .")[:120].rstrip(" .")
    if clean.split(".")[0].upper() in _WINDOWS_RESERVED_NAMES:
        clean = f"_{clean}"
    return clean


def _segment_for(time_seconds: float, cut_points: list[float]) -> int:
    for index, point in enumerate(cut_points):
        if time_seconds < point:
            return index
    return len(cut_points)


def _resolve_output_format(codec: str) -> tuple[str, str]:
    if codec in _OUTPUT_FORMATS:
        return _OUTPUT_FORMATS[codec]
    if codec.startswith("pcm_"):
        return "wav", ".wav"
    raise AudioSplitError(f"이 오디오 코덱({codec})은 음질 손실 없이 나눌 수 없습니다.")


def _find_cover_stream(container):
    for stream in container.streams.video:
        if stream.disposition & av.stream.Disposition.attached_pic:
            return stream
    return None


def _read_cover_packet(path: Path):
    with av.open(str(path)) as container:
        cover = _find_cover_stream(container)
        if cover is None:
            return None
        for packet in container.demux(cover):
            if packet.size:
                return bytes(packet)
    return None


def _open_segment_output(
    output_path: Path,
    info: AudioInfo,
    audio_stream,
    encode_bits: int | None,
    cover_stream,
    cover_bytes: bytes | None,
    base_metadata: dict[str, str],
    title: str,
    index: int,
    total: int,
):
    """encode_bits가 None이면 원본 스트림을 그대로 복사하고, 아니면 같은 설정의 FLAC 인코더를 만든다."""
    output = av.open(str(output_path), mode="w", format=info.output_format)
    try:
        metadata = dict(base_metadata)
        if title.strip():
            metadata["title"] = title.strip()
        metadata["track"] = f"{index + 1}/{total}"
        output.metadata.update(metadata)
        if encode_bits is None:
            out_audio = output.add_stream_from_template(audio_stream)
        else:
            source_context = audio_stream.codec_context
            out_audio = output.add_stream("flac", rate=info.sample_rate, layout=source_context.layout.name)
            out_audio.format = source_context.format.name
            if encode_bits and hasattr(out_audio.codec_context, "bits_per_raw_sample"):
                out_audio.codec_context.bits_per_raw_sample = encode_bits
        out_audio.metadata.update(audio_stream.metadata)
        out_cover = None
        if cover_bytes is not None:
            out_cover = output.add_stream_from_template(cover_stream)
            out_cover.disposition = av.stream.Disposition.attached_pic
            out_cover.metadata.update(cover_stream.metadata)
        output.start_encoding()
        if out_cover is not None:
            cover_packet = av.Packet(cover_bytes)
            cover_packet.stream = out_cover
            cover_packet.pts = 0
            cover_packet.dts = 0
            cover_packet.time_base = out_cover.time_base or Fraction(1, 90000)
            cover_packet.is_keyframe = True
            output.mux(cover_packet)
    except Exception:
        output.close()
        output_path.unlink(missing_ok=True)
        raise
    return output, out_audio


def _rms_db_at(chunks: list[tuple[float, np.ndarray]], sample_rate: int, center: float) -> float:
    if not chunks or sample_rate <= 0:
        return float("nan")
    half = _RMS_WINDOW_SECONDS / 2
    low, high = center - half, center + half
    pieces = []
    for start, data in chunks:
        end = start + len(data) / sample_rate
        if end <= low or start >= high:
            continue
        first = max(0, int((low - start) * sample_rate))
        last = min(len(data), int(math.ceil((high - start) * sample_rate)))
        if last > first:
            pieces.append(data[first:last])
    if not pieces:
        return float("nan")
    joined = np.concatenate(pieces)
    rms = float(np.sqrt(np.mean(np.square(joined, dtype=np.float64))))
    return 20 * math.log10(max(rms, 1e-7))
