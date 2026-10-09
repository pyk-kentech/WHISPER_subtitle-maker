#!/usr/bin/env bash
# Lane H: final candidates through the app pipeline with the NEW defaults (WhisperSeg + clips, nrng 5), all 14 tracks
source /root/asr-eval/code/eval/pod/lib.sh
while pgrep -f "llama-b11518/llama-serve[r]" > /dev/null; do sleep 20; done
for m in anime-whisper medium large-v3-turbo; do
  wait_mem 8000; app $m final-$m-ws-clips vad=whisperseg mode=clips min_silence=100
done
echo "[$(date -u +%T)] LANE H DONE"
