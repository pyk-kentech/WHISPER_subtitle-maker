"""data/audio/*.mp3 -> data/audio16k/*.opus (16 kHz mono, 48 kbps): what every compared model actually consumes."""
from pathlib import Path

import av

HERE = Path(__file__).parent
SRC = HERE / "data" / "audio"
DST = HERE / "data" / "audio16k"
DST.mkdir(exist_ok=True)
LIMITS = {"w8sleep": 600.0}

for mp3 in sorted(SRC.glob("*.mp3")):
    target = DST / f"{mp3.stem}.opus"
    limit = LIMITS.get(mp3.stem)
    with av.open(str(mp3)) as source, av.open(str(target), "w", format="ogg") as sink:
        stream = sink.add_stream("libopus", rate=16000, layout="mono")
        stream.bit_rate = 48000
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in source.decode(audio=0):
            if limit is not None and frame.time is not None and frame.time >= limit:
                break
            for out in resampler.resample(frame):
                for packet in stream.encode(out):
                    sink.mux(packet)
        for out in resampler.resample(None):
            for packet in stream.encode(out):
                sink.mux(packet)
        for packet in stream.encode(None):
            sink.mux(packet)
    print(f"{target.name}: {target.stat().st_size / 2**20:.1f} MB")
