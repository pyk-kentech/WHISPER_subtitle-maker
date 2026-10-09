#!/usr/bin/env bash
# Phase 2 grid on the TUNING split (works w1-w4) with the app code as-is
source /root/asr-eval/code/eval/pod/lib.sh
export ONLY=w1t01,w1t05,w2t01,w2t09,w3t03,w3t06,w4t02,w4t06
for m in medium large-v3-turbo ja-1.5b; do
  for v in silero whisperseg; do
    ms=$([[ $v == whisperseg ]] && echo 100 || echo 500)
    for mode in normal clips; do
      app $m p2-$m-$v-$mode vad=$v mode=$mode min_silence=$ms
    done
  done
done
app anime-whisper p2-anime-whisper-silero-clips vad=silero mode=clips min_silence=500
app anime-whisper p2-anime-whisper-whisperseg-clips vad=whisperseg mode=clips min_silence=100
echo "[$(date -u +%T)] LANE P2GRID DONE"
