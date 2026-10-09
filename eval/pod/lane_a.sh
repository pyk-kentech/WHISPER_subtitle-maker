#!/usr/bin/env bash
# Lane A: current app pipeline (faster-whisper + Silero + app postprocess), 6 models
source /root/asr-eval/code/eval/pod/lib.sh
app medium app-medium
app large-v3-turbo app-turbo
app anime-whisper app-anime
app ja-1.5b app-ja15b
app kotoba-v2 app-kotoba
app large-v3 app-large-v3
echo "[$(date -u +%T)] LANE A DONE"
