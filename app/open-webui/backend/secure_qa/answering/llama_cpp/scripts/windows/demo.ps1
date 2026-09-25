<#
.SYNOPSIS
  Friday demo (Week 2): one command starts the runtime, endpoints are shown, a chat request succeeds,
  then two compression levels are compared.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
  powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 -Stub          # dry run without real models
#>
param(
  [switch]$Stub,
  [int]$Port = 8080,
  [string]$Exe = ""
)
$ErrorActionPreference = "Stop"
$root = if (Test-Path (Join-Path $PSScriptRoot "..\runtime.toml")) { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path } else { (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path }
if (-not $Exe) {
  $Exe = if (Test-Path (Join-Path $root "docqa-runtime.exe")) { Join-Path $root "docqa-runtime.exe" } else { "docqa-runtime" }
}
$extra = @()
if ($Stub) {
  $stubDir = Join-Path $root "runs\stub-models"
  & $Exe stub-models --out $stubDir | Out-Null
  $extra += @("--server-bin", "stub", "--models-dir", $stubDir)
}
$base = "http://127.0.0.1:$Port"

function Step($t) { Write-Host "`n=== $t ===" -ForegroundColor Cyan }

Step "1. Pre-flight checks"
& $Exe doctor @extra --port $Port

Step "2. One command: docqa-runtime up (opens in its own window)"
$proc = Start-Process -FilePath $Exe -ArgumentList (@("up", "--port", "$Port") + $extra) -PassThru
$deadline = (Get-Date).AddMinutes(3)
do {
  Start-Sleep -Milliseconds 500
  try { $h = Invoke-RestMethod "$base/health" -TimeoutSec 2 } catch { $h = $null }
} until (($h -and $h.status -eq "ok") -or (Get-Date) -gt $deadline -or $proc.HasExited)
if (-not $h -or $h.status -ne "ok") { throw "Runtime did not become ready - see its window for the error." }

Step "3. Endpoints"
Write-Host "GET  $base/health"; $h | ConvertTo-Json -Depth 5
Write-Host "`nGET  $base/v1/models"; Invoke-RestMethod "$base/v1/models" | ConvertTo-Json -Depth 5

Step "4. Chat request"
$body = @{ messages = @(@{ role = "user"; content = "In one sentence: why does running the model locally help keep documents private?" }); max_tokens = 80 } | ConvertTo-Json -Depth 5
$r = Invoke-RestMethod "$base/v1/chat/completions" -Method Post -ContentType "application/json" -Body $body
Write-Host $r.choices[0].message.content -ForegroundColor Green
Write-Host ("{0} tokens at {1:N1} tok/s" -f $r.usage.completion_tokens, $r.timings.predicted_per_second)

Step "5. Actionable error example (unknown model)"
try { Invoke-RestMethod "$base/v1/chat/completions" -Method Post -ContentType "application/json" -Body '{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}' }
catch { $_.ErrorDetails.Message }

Stop-Process -Id $proc.Id -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

Step "6. Compare compression levels (quantization matrix)"
& $Exe bench @extra --runs 2 --max-tokens 96
Write-Host "`nOpen the report.md printed above to show RAM / startup / tokens-per-second side by side."
