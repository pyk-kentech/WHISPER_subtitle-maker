#!/usr/bin/env bash
# Lane G: remaining WhisperJAV runs after the Gemma translation test frees the GPU
source /root/asr-eval/code/eval/pod/lib.sh
while pgrep -f "llama-b11518/llama-serve[r]" > /dev/null; do sleep 20; done
for n in wj-qwen-jaykwok wj-balanced-ja15b; do rmdir $E/lock/$n 2>/dev/null; rm -rf $E/out/$n; done
wait_mem 10000; wj wj-qwen-jaykwok --mode qwen --qwen-model-id jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame
wait_mem 10000; wj wj-balanced-ja15b --mode balanced --model TransWithAI/whisper-ja-1.5B-ct2
# ensemble: only w1t01 failed (OOM at 09:30) -> rerun that track alone and add it
if [[ ! -f $E/out/wj-ens19/w1t01.json ]]; then
  mkdir -p $E/retry_audio && ln -sf $E/data/audio/w1t01.mp3 $E/retry_audio/w1t01.mp3
  rm -rf $E/wj/ens19-w1 $E/tmp/ens19-w1; mkdir -p $E/wj/ens19-w1 $E/tmp/ens19-w1
  wait_mem 12000; start=$(date +%s)
  $WJ $E/retry_audio --output-dir $E/wj/ens19-w1 --temp-dir $E/tmp/ens19-w1 --no-progress --ensemble --pass1-pipeline qwen --pass1-qwen-params '{"generator_backend": "anime-whisper"}' --pass1-speech-segmenter whisperseg --pass2-pipeline qwen --pass2-speech-segmenter ten --merge-strategy pass1_primary > $E/logs/wj-ens19-w1.log 2>&1
  $STT $E/code/eval/server/srt_to_json.py $E/wj/ens19-w1 $E/tmp/ens19-w1-json 0 >> $E/logs/wj-ens19-w1.log 2>&1
  cp $E/tmp/ens19-w1-json/w1t01.json $E/out/wj-ens19/ && echo "[$(date -u +%T)] ens19 w1t01 retry done $(( $(date +%s) - start ))s"
fi
/root/wj-venv/bin/python $E/code/eval/pod/merge_combos.py > $E/logs/merge.log 2>&1; echo "[$(date -u +%T)] merges: $(grep -c tracks $E/logs/merge.log)"
echo "[$(date -u +%T)] LANE G DONE"
