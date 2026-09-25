<#
.SYNOPSIS
  Belt-and-braces network isolation: Windows Firewall rules that block ALL outbound and inbound traffic
  for llama-server.exe and docqa-runtime.exe. Loopback (127.0.0.1) is not filtered by Windows Firewall,
  so the local API keeps working. Requires an elevated (Administrator) PowerShell.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\windows\block-network.ps1            # add rules
  powershell -ExecutionPolicy Bypass -File scripts\windows\block-network.ps1 -Remove    # remove rules
  powershell -ExecutionPolicy Bypass -File scripts\windows\block-network.ps1 -Status
#>
param(
  [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
  [switch]$Remove,
  [switch]$Status
)
$ErrorActionPreference = "Stop"
$group = "DocQA Runtime Isolation"

if ($Status) { Get-NetFirewallRule -Group $group -ErrorAction SilentlyContinue | Format-Table DisplayName, Direction, Action, Enabled; exit 0 }

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) { throw "Run this from an Administrator PowerShell (right-click > Run as administrator)." }

Get-NetFirewallRule -Group $group -ErrorAction SilentlyContinue | Remove-NetFirewallRule
if ($Remove) { Write-Host "Removed '$group' rules."; exit 0 }

$programs = @(
  (Join-Path $Root "bin\llama-server.exe"),
  (Join-Path $Root "docqa-runtime.exe")
) | Where-Object { Test-Path $_ }
if (-not $programs) { throw "Neither bin\llama-server.exe nor docqa-runtime.exe found under $Root" }

foreach ($p in $programs) {
  foreach ($dir in "Outbound", "Inbound") {
    New-NetFirewallRule -DisplayName "DocQA block $dir - $(Split-Path $p -Leaf)" -Group $group `
      -Direction $dir -Action Block -Program $p -Profile Any | Out-Null
  }
  Write-Host "Blocked network for $p"
}
Write-Host "`nVerify with: docqa-runtime doctor   (and the baseline record's 'socket audit')"
