$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Windows PowerShell 5.1 does not stop on a failing native command even with "Stop",
# so check every exit code explicitly.
function Invoke-Checked {
  param([string]$Step, [scriptblock]$Command)
  & $Command
  if ($LASTEXITCODE -ne 0) {
    throw "$Step failed (exit code $LASTEXITCODE)"
  }
}

$icon = Get-ChildItem -Path $root -File -Filter *.ico | Select-Object -First 1
$fontFiles = @(
  Get-ChildItem -Path $root -File -Filter *.ttf
  Get-ChildItem -Path $root -File -Filter *.otf
) | Sort-Object Name -Unique
$iconArgs = @()
if ($icon) {
  $iconArgs = @("--icon", $icon.FullName)
}

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
  if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.12 or 3.13 and check 'Add python.exe to PATH'."
  }
  Invoke-Checked "Creating .venv" { python -m venv .venv }
}

Invoke-Checked "Python version check" {
  .\.venv\Scripts\python -c "import sys; v=sys.version_info[:2]; sys.exit(0 if (3, 10) <= v <= (3, 13) else print('Python 3.10-3.13 is required, found %d.%d. Delete .venv and recreate it with Python 3.12 or 3.13.' % v) or 1)"
}
Invoke-Checked "Upgrading pip" { .\.venv\Scripts\python -m pip install --upgrade pip }
Invoke-Checked "Installing requirements" { .\.venv\Scripts\python -m pip install -r requirements.txt }
$pyinstallerArgs = @(
  "--noconfirm",
  "--clean",
  "--windowed",
  "--onefile",
  "--name", "DongeumSubMaker-OneFile",
  "--paths", $root,
  "--collect-all", "faster_whisper",
  "--collect-all", "ctranslate2",
  "--collect-all", "av",
  "--collect-all", "tokenizers",
  "--collect-all", "huggingface_hub",
  "--collect-all", "google.genai",
  "--collect-all", "dotenv",
  "--collect-all", "onnxruntime",
  # Runpod 원격 실행은 앱 코드(.py)를 원격 Pod로 올리므로 소스를 함께 넣는다.
  "--add-data", "$root\app\*.py;remote_bundle\app"
)

if ($iconArgs.Count -gt 0) {
  $pyinstallerArgs += $iconArgs
  $pyinstallerArgs += @("--add-data", "$($icon.FullName);.")
}

foreach ($fontFile in $fontFiles) {
  $pyinstallerArgs += @("--add-data", "$($fontFile.FullName);.")
}

$pyinstallerArgs += "launcher.py"

Invoke-Checked "PyInstaller build" { .\.venv\Scripts\python -m PyInstaller @pyinstallerArgs }
