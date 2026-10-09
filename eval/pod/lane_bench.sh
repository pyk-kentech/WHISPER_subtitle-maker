#!/usr/bin/env bash
# Solo benchmark (run ONLY when nothing else uses the GPU): RTF + peak VRAM on one track (w7t2, 634.5 s)
source /root/asr-eval/code/eval/pod/lib.sh
mkdir -p $E/bench_audio && ln -sf $E/data/audio/w7t2.mp3 $E/bench_audio/w7t2.mp3
export ONLY=w7t2
for spec in "medium silero normal 500" "medium whisperseg clips 100" "large-v3-turbo silero normal 500" "large-v3-turbo whisperseg clips 100" \
            "anime-whisper silero clips 500" "anime-whisper whisperseg clips 100" "ja-1.5b whisperseg clips 100"; do
  set -- $spec
  app $1 b-$1-$2-$3 vad=$2 mode=$3 min_silence=$4
done
wjb() {  # wjb <system> <args...> : WhisperJAV on the bench folder
  local name=$1; shift
  [[ -f $E/out/$name/_meta.json ]] && return 0
  rm -rf $E/wj/$name $E/tmp/$name; mkdir -p $E/wj/$name $E/tmp/$name
  vram_start $name; local start=$(date +%s.%N)
  $WJ $E/bench_audio --output-dir $E/wj/$name --temp-dir $E/tmp/$name --no-progress "$@" > $E/logs/$name.log 2>&1
  local elapsed=$(python3 -c "print($(date +%s.%N) - $start)"); local peak=$(vram_stop $name)
  $STT $E/code/eval/server/srt_to_json.py $E/wj/$name $E/out/$name $elapsed >> $E/logs/$name.log 2>&1
  echo "[$(date -u +%T)] bench $name elapsed=${elapsed}s $peak"
}
wjb bw-qwen --mode qwen
wjb bw-qwen-anime --mode qwen --qwen-generator anime-whisper
wjb bw-qwen-jaykwok --mode qwen --qwen-model-id jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame
wjb bw-balanced --mode balanced
wjb bw-qwen-fa --mode qwen --qwen-timestamp-mode aligner_interpolation
wjb bw-ens19 --ensemble --pass1-pipeline qwen --pass1-qwen-params '{"generator_backend": "anime-whisper"}' --pass1-speech-segmenter whisperseg --pass2-pipeline qwen --pass2-speech-segmenter ten --merge-strategy pass1_primary
rm -f $E/combo/anime-whisper/w7t2.json; echo "[$(date -u +%T)] start combo bench"; vram_start b-combo; t=$(date +%s)
ONLY=w7t2 /root/wj-venv/bin/python $E/code/eval/pod/combo_clips.py anime-whisper > $E/logs/b-combo.log 2>&1
echo "[$(date -u +%T)] bench combo elapsed=$(( $(date +%s) - t ))s $(vram_stop b-combo)"
echo "[$(date -u +%T)] LANE BENCH DONE"
