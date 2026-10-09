#!/usr/bin/env bash
# Linux build: produces dist/DongeumSubMaker/ (onedir) and dist/DongeumSubMaker-linux-<arch>.tar.gz
# Usage: ./build-linux.sh            (onedir)
#        ./build-linux.sh --onefile  (single executable)
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

onefile=0
if [[ "${1:-}" == "--onefile" ]]; then
  onefile=1
fi

python_bin="${PYTHON:-python3}"
if [[ ! -x ".venv-linux/bin/python" ]]; then
  if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "Python was not found. Install Python 3.10-3.13 (e.g. sudo apt install python3 python3-venv)." >&2
    exit 1
  fi
  # Windows용 .venv와 섞이지 않도록 별도 폴더를 쓴다.
  "$python_bin" -m venv .venv-linux
fi

py=".venv-linux/bin/python"
"$py" -c "import sys; v=sys.version_info[:2]; sys.exit(0 if (3, 10) <= v <= (3, 13) else print('Python 3.10-3.13 is required, found %d.%d. Delete .venv-linux and recreate it.' % v) or 1)"
"$py" -m pip install --upgrade pip
"$py" -m pip install -r requirements.txt

name="DongeumSubMaker"
args=(
  --noconfirm
  --clean
  --windowed
  --paths "$root"
  --collect-all faster_whisper
  --collect-all ctranslate2
  --collect-all av
  --collect-all tokenizers
  --collect-all huggingface_hub
  --collect-all google.genai
  --collect-all dotenv
  --collect-all onnxruntime
  # Runpod 원격 실행은 앱 코드(.py)를 원격 Pod로 올리므로 소스를 함께 넣는다.
  --add-data "$root/app/*.py:remote_bundle/app"
)
if [[ $onefile -eq 1 ]]; then
  name="DongeumSubMaker-OneFile"
  args+=(--onefile)
fi
args+=(--name "$name")

shopt -s nullglob
for icon in "$root"/*.ico; do
  args+=(--add-data "$icon:.")
  break
done
for font in "$root"/*.ttf "$root"/*.otf; do
  args+=(--add-data "$font:.")
done
shopt -u nullglob

"$py" -m PyInstaller "${args[@]}" launcher.py

if [[ $onefile -eq 0 ]]; then
  arch="$(uname -m)"
  tar -C dist -czf "dist/${name}-linux-${arch}.tar.gz" "$name"
  echo "Built: dist/$name/$name  (archive: dist/${name}-linux-${arch}.tar.gz)"
else
  echo "Built: dist/$name"
fi
