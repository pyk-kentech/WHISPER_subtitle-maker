#!/usr/bin/env bash
# Pull pod results (json only, no audio) into eval/out and eval/pod_logs
cd "$(dirname "$0")"
mkdir -p out pod_logs
ssh rp 'cd /root/asr-eval && tar cf - out logs vram 2>/dev/null' | tar xf - -C . --transform 's,^logs,pod_logs,;s,^vram,pod_logs/vram,' 2>/dev/null
ls out | wc -l
