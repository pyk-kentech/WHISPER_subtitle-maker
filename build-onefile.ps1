$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  python -m venv .venv
}

.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --onefile `
  --name "DongeumSubMaker-OneFile" `
  --paths "$root" `
  --collect-all faster_whisper `
  --collect-all ctranslate2 `
  --collect-all av `
  --collect-all tokenizers `
  --collect-all huggingface_hub `
  --collect-all google.generativeai `
  --collect-all google.ai.generativelanguage `
  --collect-all dotenv `
  launcher.py
