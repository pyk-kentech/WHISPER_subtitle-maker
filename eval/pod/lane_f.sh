#!/usr/bin/env bash
# Lane F: one-factor-at-a-time tuning on the TUNING split (base = anime-whisper + WhisperSeg + clips, thr .5, ms 100, pad 200, max 12, beam 5, nrng 5)
source /root/asr-eval/code/eval/pod/lib.sh
export ONLY=w1t01,w1t05,w2t01,w2t09,w3t03,w3t06,w4t02,w4t06
B="vad=whisperseg mode=clips"
run() { local name=$1; shift; wait_mem 8000; app anime-whisper t-anime-$name $B "$@"; }
run thr35 min_silence=100 thr=0.35
run thr65 min_silence=100 thr=0.65
run ms300 min_silence=300
run ms500 min_silence=500
run pad100 min_silence=100 pad=100
run pad400 min_silence=100 pad=400
run max8 min_silence=100 max_clip=8
run max15 min_silence=100 max_clip=15
run beam1 min_silence=100 beam=1
run nrng0 min_silence=100 nrng=0
wait_mem 8000; app medium t-medium-ws-clips-nrng5 vad=whisperseg mode=clips min_silence=100 nrng=5
wait_mem 8000; app large-v3-turbo t-turbo-ws-clips-nrng5 vad=whisperseg mode=clips min_silence=100 nrng=5
echo "[$(date -u +%T)] LANE F DONE"
