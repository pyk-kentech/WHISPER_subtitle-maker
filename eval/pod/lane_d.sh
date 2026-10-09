#!/usr/bin/env bash
# Lane D: same WJ list from the other end (locks prevent duplicates)
source /root/asr-eval/code/eval/pod/lib.sh
wj wj-ens19 --ensemble --pass1-pipeline qwen --pass1-qwen-params '{"generator_backend": "anime-whisper"}' --pass1-speech-segmenter whisperseg --pass2-pipeline qwen --pass2-speech-segmenter ten --merge-strategy pass1_primary
wj wj-kotoba --mode transformers --hf-model-id kotoba-tech/kotoba-whisper-v2.0
wj wj-balanced-ja15b --mode balanced --model TransWithAI/whisper-ja-1.5B-ct2
wj wj-qwen-jaykwok --mode qwen --qwen-model-id jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame
wj wj-balanced --mode balanced
echo "[$(date -u +%T)] LANE D DONE"
