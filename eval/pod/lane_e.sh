#!/usr/bin/env bash
# Lane E: one job at a time, each waits for free GPU memory (phase-2 grid, remaining combos, ForcedAligner run)
source /root/asr-eval/code/eval/pod/lib.sh
TUNE=w1t01,w1t05,w2t01,w2t09,w3t03,w3t06,w4t02,w4t06
for m in medium large-v3-turbo ja-1.5b; do
  for v in silero whisperseg; do
    ms=$([[ $v == whisperseg ]] && echo 100 || echo 500)
    for mode in normal clips; do
      wait_mem 8000; ONLY=$TUNE app $m p2-$m-$v-$mode vad=$v mode=$mode min_silence=$ms
    done
  done
done
wait_mem 8000; ONLY=$TUNE app anime-whisper p2-anime-whisper-silero-clips vad=silero mode=clips min_silence=500
wait_mem 8000; ONLY=$TUNE app anime-whisper p2-anime-whisper-whisperseg-clips vad=whisperseg mode=clips min_silence=100
wait_mem 10000; combo large-v3-turbo
wait_mem 10000; combo ja-1.5b
rmdir $E/lock/wj-qwen-fa 2>/dev/null
wait_mem 10000; wj wj-qwen-fa --mode qwen --qwen-timestamp-mode aligner_interpolation
echo "[$(date -u +%T)] LANE E DONE"
