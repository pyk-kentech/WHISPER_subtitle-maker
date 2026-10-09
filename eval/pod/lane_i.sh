#!/bin/bash
source /root/asr-eval/code/eval/pod/lib.sh
while pgrep -f "code/eval/pod/lane_[g].sh" > /dev/null; do sleep 20; done
wait_mem 10000
echo "[$(date -u +%T)] start combo jaykwok"
/root/wj-venv/bin/python $E/code/eval/pod/combo_clips.py anime-whisper jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame > $E/logs/combo-jaykwok.log 2>&1
echo "[$(date -u +%T)] end combo jaykwok exit=$?"
