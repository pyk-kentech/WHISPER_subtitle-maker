#!/usr/bin/env bash
# Lane C: Whisper model + Qwen3-ASR on the same WhisperSeg clips (items 3, 4b, 4c), then WJ qwen + ForcedAligner
source /root/asr-eval/code/eval/pod/lib.sh
combo() {
  local key=$1 tag=${2:-}
  echo "[$(date -u +%T)] start combo $key $tag"
  /root/wj-venv/bin/python $E/code/eval/pod/combo_clips.py $key $tag > $E/logs/combo-$key.log 2>&1
  echo "[$(date -u +%T)] end combo $key exit=$?"
}
combo anime-whisper
combo large-v3-turbo
combo ja-1.5b
wj wj-qwen-fa --mode qwen --qwen-timestamp-mode aligner_interpolation
echo "[$(date -u +%T)] LANE C DONE"
