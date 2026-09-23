<#
.SYNOPSIS
  Download the official llama.cpp Windows build into .\bin (run this on a machine WITH internet, then copy the folder).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\fetch_runtime.ps1
  powershell -ExecutionPolicy Bypass -File scripts\windows\fetch_runtime.ps1 -Tag b6500 -Variant cpu-x64
#>
param(
  [string]$Tag = "latest",            # a release tag like b6500, or "latest"
  [string]$Variant = "cpu-x64",       # cpu-x64 (reference laptop), vulkan-x64, cuda-12.4-x64 ...
  [string]$Dest = (Join-Path $PSScriptRoot "..\..\bin")
)
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$api = if ($Tag -eq "latest") { "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" }
       else { "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$Tag" }
Write-Host "Looking up llama.cpp release ($Tag)..."
$rel = Invoke-RestMethod -Uri $api -Headers @{ "User-Agent" = "docqa-runtime" }
$asset = $rel.assets | Where-Object { $_.name -like "*bin-win-$Variant.zip" } | Select-Object -First 1
if (-not $asset) {
  $names = ($rel.assets | ForEach-Object { $_.name }) -join "`n  "
  throw "No asset matching '*bin-win-$Variant.zip' in $($rel.tag_name). Available:`n  $names"
}
$tmp = Join-Path $env:TEMP $asset.name
Write-Host "Downloading $($asset.name) ($([math]::Round($asset.size / 1MB)) MB)..."
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $tmp -UseBasicParsing

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Expand-Archive -Path $tmp -DestinationPath $Dest -Force
# Some releases nest the binaries in a sub-folder; flatten so bin\llama-server.exe exists.
$server = Get-ChildItem -Path $Dest -Recurse -Filter "llama-server.exe" | Select-Object -First 1
if (-not $server) { throw "llama-server.exe not found after extracting $($asset.name)" }
if ($server.DirectoryName -ne (Resolve-Path $Dest).Path) {
  Get-ChildItem $server.DirectoryName | Move-Item -Destination $Dest -Force
}
Get-ChildItem -Path $Dest -Recurse | Unblock-File
Set-Content -Path (Join-Path $Dest "LLAMA_CPP_VERSION.txt") -Value "$($rel.tag_name) $($asset.name)"
Remove-Item $tmp -Force
Write-Host "Installed llama.cpp $($rel.tag_name) into $((Resolve-Path $Dest).Path)"
& (Join-Path $Dest "llama-server.exe") --version
