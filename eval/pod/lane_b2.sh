#!/usr/bin/env bash
# Lane B2: adopt the running wj-qwen (started by lane B), then continue lane B's list with locks
source /root/asr-eval/code/eval/pod/lib.sh
PID=$1
while kill -0 $PID 2>/dev/null; do sleep 20; done
start=$(date -d "2026-10-09T09:04:58Z" +%s); now=$(date +%s)
$STT $E/code/eval/server/srt_to_json.py $E/wj/wj-qwen $E/out/wj-qwen $((now - start)) >> $E/logs/wj-qwen.log 2>&1
echo "[$(date -u +%T)] end wj-qwen (adopted) elapsed=$((now - start))s"
mkdir -p $E/lock/wj-qwen
wj wj-qwen-anime --mode qwen --qwen-generator anime-whisper
wj wj-balanced --mode balanced
wj wj-qwen-jaykwok --mode qwen --qwen-model-id jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame
wj wj-balanced-ja15b --mode balanced --model TransWithAI/whisper-ja-1.5B-ct2
wj wj-kotoba --mode transformers --hf-model-id kotoba-tech/kotoba-whisper-v2.0
wj wj-ens19 --ensemble --pass1-pipeline qwen --pass1-qwen-params '{"generator_backend": "anime-whisper"}' --pass1-speech-segmenter whisperseg --pass2-pipeline qwen --pass2-speech-segmenter ten --merge-strategy pass1_primary
echo "[$(date -u +%T)] LANE B2 DONE"
