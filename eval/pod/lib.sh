# shared helpers for pod runs (source me)
set -u
E=/root/asr-eval
mkdir -p $E/out $E/wj $E/tmp $E/logs $E/vram
export HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUNBUFFERED=1
STT=/root/fw-venv/bin/python
WJ=/root/wj-venv/bin/whisperjav

vram_start() {  # sample GPU memory every second into $E/vram/<name>.txt
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > $E/vram/$1.txt 2>/dev/null &
  VRAM_PID=$!
}
vram_stop() { kill $VRAM_PID 2>/dev/null; wait $VRAM_PID 2>/dev/null; echo "peak_vram_mib=$(sort -n $E/vram/$1.txt | tail -1)"; }

app() {  # app <model_key> <system> [k=v...]
  local key=$1 name=$2; shift 2
  [[ -f $E/out/$name/w8sleep.json || -f $E/out/$name/_done ]] && return 0
  echo "[$(date -u +%T)] start $name"
  vram_start $name
  $STT $E/code/eval/pod/run_app.py "$key" "$name" "$@" > $E/logs/$name.log 2>&1
  local code=$?
  [[ $code == 0 ]] && touch $E/out/$name/_done
  echo "[$(date -u +%T)] end $name exit=$code $(vram_stop $name)"
}

wj() {  # wj <system> <whisperjav args...>
  local name=$1; shift
  [[ -f $E/out/$name/_meta.json ]] && return 0
  mkdir -p $E/lock; mkdir $E/lock/$name 2>/dev/null || { echo "[$(date -u +%T)] skip $name (other lane)"; return 0; }
  echo "[$(date -u +%T)] start $name $*"
  rm -rf $E/wj/$name $E/tmp/$name $E/out/$name; mkdir -p $E/wj/$name $E/tmp/$name
  vram_start $name
  local start=$(date +%s.%N)
  $WJ $E/data/audio --output-dir $E/wj/$name --temp-dir $E/tmp/$name --no-progress "$@" > $E/logs/$name.log 2>&1
  local code=$?
  local elapsed=$(python3 -c "print($(date +%s.%N) - $start)")
  local peak=$(vram_stop $name)
  $STT $E/code/eval/server/srt_to_json.py $E/wj/$name $E/out/$name $elapsed >> $E/logs/$name.log 2>&1
  rm -rf $E/tmp/$name
  if grep -q "failed [1-9]" $E/logs/$name.log; then
    rm -f $E/out/$name/_meta.json
    echo "[$(date -u +%T)] FAILED $name exit=$code elapsed=${elapsed}s $peak"
    return 1
  fi
  echo "[$(date -u +%T)] end $name exit=$code elapsed=${elapsed}s $peak"
}

wait_mem() {  # wait_mem <MiB>: block until that much GPU memory is free (other lanes share the GPU)
  local need=$1
  while true; do
    local free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [[ $free -ge $need ]] && return 0
    sleep 15
  done
}

combo() {  # combo <whisper_model_key>
  echo "[$(date -u +%T)] start combo $1"
  /root/wj-venv/bin/python $E/code/eval/pod/combo_clips.py $1 > $E/logs/combo-$1.log 2>&1
  echo "[$(date -u +%T)] end combo $1 exit=$?"
}
