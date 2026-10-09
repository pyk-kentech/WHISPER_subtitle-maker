#!/usr/bin/env bash
# Lane C2: after lane C's combos (its wj-qwen-fa is pre-locked so it ends), run the phase-2 grid, then wj-qwen-fa
source /root/asr-eval/code/eval/pod/lib.sh
while pgrep -f "bash code/eval/pod/lane_c.sh" > /dev/null; do sleep 30; done
bash $E/code/eval/pod/lane_p2grid.sh
rmdir $E/lock/wj-qwen-fa 2>/dev/null
wj wj-qwen-fa --mode qwen --qwen-timestamp-mode aligner_interpolation
echo "[$(date -u +%T)] LANE C2 DONE"
