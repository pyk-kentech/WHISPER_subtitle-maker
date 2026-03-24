$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

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
  python -m venv .venv
}

.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
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
  "--collect-all", "google.generativeai",
  "--collect-all", "google.ai.generativelanguage",
  "--collect-all", "dotenv"
)

if ($iconArgs.Count -gt 0) {
  $pyinstallerArgs += $iconArgs
  $pyinstallerArgs += @("--add-data", "$($icon.FullName);.")
}

foreach ($fontFile in $fontFiles) {
  $pyinstallerArgs += @("--add-data", "$($fontFile.FullName);.")
}

$pyinstallerArgs += "launcher.py"

.\.venv\Scripts\python -m PyInstaller @pyinstallerArgs
