<#
.SYNOPSIS
  Build the portable Windows bundle:  dist\DocQA-Runtime\docqa-runtime.exe  (+ runtime.toml, bin\, models\)
  Needs Python 3.11+ on the BUILD machine only; the target laptop needs nothing installed.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\build_exe.ps1
  powershell -ExecutionPolicy Bypass -File scripts\windows\build_exe.ps1 -IncludeBin -Zip
#>
param(
  [switch]$IncludeBin,     # copy .\bin (llama.cpp) into the bundle
  [switch]$IncludeModels,  # copy .\models\*.gguf into the bundle (large!)
  [switch]$Zip
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root

if (-not (Test-Path ".venv-build")) { py -3 -m venv .venv-build }
& .\.venv-build\Scripts\python.exe -m pip install --upgrade pip | Out-Null
& .\.venv-build\Scripts\python.exe -m pip install . "pyinstaller>=6" | Out-Null
& .\.venv-build\Scripts\pyinstaller.exe --noconfirm --clean packaging\docqa-runtime.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$out = Join-Path $root "dist\DocQA-Runtime"
Copy-Item runtime.toml $out -Force
New-Item -ItemType Directory -Force -Path (Join-Path $out "bin"), (Join-Path $out "models"), (Join-Path $out "runs") | Out-Null
Copy-Item models\shortlist.json (Join-Path $out "models") -Force
Copy-Item packaging\start-runtime.cmd, packaging\README-RUN.txt $out -Force
New-Item -ItemType Directory -Force -Path (Join-Path $out "scripts") | Out-Null
Copy-Item scripts\windows\block-network.ps1, scripts\windows\demo.ps1 (Join-Path $out "scripts") -Force
if ($IncludeBin -and (Test-Path bin\llama-server.exe)) { Copy-Item bin\* (Join-Path $out "bin") -Recurse -Force }
if ($IncludeModels) { Copy-Item models\*.gguf (Join-Path $out "models") -Force -ErrorAction SilentlyContinue }

& (Join-Path $out "docqa-runtime.exe") --version
if ($Zip) {
  $zip = Join-Path $root "dist\DocQA-Runtime.zip"
  if (Test-Path $zip) { Remove-Item $zip }
  Compress-Archive -Path $out -DestinationPath $zip
  Write-Host "Zipped: $zip"
}
Write-Host "Bundle ready: $out"
