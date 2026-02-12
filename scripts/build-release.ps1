# Build a Windows NSIS installer for DeepSeek OCR Client using bundled uv.
#
# Usage:
#   .\scripts\build-release.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

Write-Host "=============================================="
Write-Host "  DeepSeek OCR Client Release Build (Windows)"
Write-Host "=============================================="
Write-Host ""
Write-Host "Project root: $ProjectRoot"
Write-Host ""

Set-Location $ProjectRoot

Write-Host "=== Step 1/3: Downloading uv runtime binary ==="
& "$ScriptDir\download-uv.ps1"
Write-Host ""

Write-Host "=== Step 2/3: Installing Node dependencies ==="
npm install
Write-Host ""

Write-Host "=== Step 3/3: Building NSIS installer ==="
npm run dist:win
Write-Host ""

$Installer = Get-ChildItem -Path (Join-Path $ProjectRoot "dist") -Filter "*.exe" -File -ErrorAction SilentlyContinue | Select-Object -First 1

Write-Host "=============================================="
Write-Host "  Build Complete"
Write-Host "=============================================="

if ($Installer) {
    $SizeMB = [math]::Round($Installer.Length / 1MB, 1)
    Write-Host "Installer: $($Installer.FullName)"
    Write-Host "Size: $SizeMB MB"
} else {
    Write-Host "Installer not found in dist/. Check build logs."
}
