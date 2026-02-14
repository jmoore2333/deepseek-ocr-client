# Build a Windows NSIS installer for DeepSeek OCR Client using bundled uv.
#
# Usage:
#   .\scripts\build-release.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

function Test-SymlinkPrivilege {
    $TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("deepseek-ocr-symlink-test-" + [guid]::NewGuid().ToString("N"))
    $TargetFile = Join-Path $TempRoot "target.txt"
    $LinkFile = Join-Path $TempRoot "link.txt"

    New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
    Set-Content -Path $TargetFile -Value "symlink-test"

    try {
        New-Item -ItemType SymbolicLink -Path $LinkFile -Target $TargetFile -ErrorAction Stop | Out-Null
        return $true
    } catch {
        return $false
    } finally {
        Remove-Item -Path $TempRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "=============================================="
Write-Host "  DeepSeek OCR Client Release Build (Windows)"
Write-Host "=============================================="
Write-Host ""
Write-Host "Project root: $ProjectRoot"
Write-Host ""

Set-Location $ProjectRoot

Write-Host "=== Preflight: Checking symbolic link privilege ==="
$HasSymlinkPrivilege = Test-SymlinkPrivilege
if ($HasSymlinkPrivilege) {
    Write-Host "Symlink creation is available."
} else {
    Write-Host "Symlink creation is NOT available."
    Write-Host "Will use compatibility build mode to avoid winCodeSign extraction."
    Write-Host "Compatibility tradeoff: app EXE icon/version metadata editing is disabled."
}
Write-Host ""

Write-Host "=== Step 1/3: Downloading uv runtime binary ==="
& "$ScriptDir\download-uv.ps1"
Write-Host ""

Write-Host "=== Step 2/3: Installing Node dependencies ==="
npm install
Write-Host ""

Write-Host "=== Step 3/3: Building NSIS installer ==="
$env:CSC_IDENTITY_AUTO_DISCOVERY = "false"
if ($HasSymlinkPrivilege) {
    npm run dist:win
} else {
    npm exec electron-builder -- --win nsis -c.win.signAndEditExecutable=false -c.win.verifyUpdateCodeSignature=false
}
if ($LASTEXITCODE -ne 0) {
    Write-Error "Windows installer build failed."
    exit $LASTEXITCODE
}
Write-Host ""

$Installers = Get-ChildItem -Path (Join-Path $ProjectRoot "dist") -Filter "*.exe" -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
$Installer = $Installers | Select-Object -First 1

Write-Host "=============================================="
Write-Host "  Build Complete"
Write-Host "=============================================="

if ($Installer) {
    $SizeMB = [math]::Round($Installer.Length / 1MB, 1)
    Write-Host "Installer: $($Installer.FullName)"
    Write-Host "Size: $SizeMB MB"
    if ($Installers.Count -gt 1) {
        Write-Host "Note: Multiple installers found in dist/. Using newest by timestamp."
    }
} else {
    Write-Host "Installer not found in dist/. Check build logs."
}
