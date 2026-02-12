# Download the uv binary for Windows and stage it in runtime/uv.exe.
#
# Usage:
#   .\scripts\download-uv.ps1

param(
    [string]$UvVersion = "0.6.6"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$RuntimeDir = Join-Path $ProjectRoot "runtime"

if (-not (Test-Path $RuntimeDir)) {
    New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
}

$Arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
switch ($Arch) {
    "X64"   { $Platform = "x86_64-pc-windows-msvc" }
    "Arm64" { $Platform = "aarch64-pc-windows-msvc" }
    default {
        Write-Error "Unsupported architecture: $Arch"
        exit 1
    }
}

Write-Host "Detected platform: $Platform"
Write-Host "Downloading uv $UvVersion..."

$ArchiveName = "uv-$Platform.zip"
$DownloadUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$ArchiveName"
$TempDir = Join-Path $env:TEMP "uv-download-$(Get-Random)"
New-Item -ItemType Directory -Path $TempDir -Force | Out-Null

try {
    $ArchivePath = Join-Path $TempDir $ArchiveName
    $ExtractDir = Join-Path $TempDir "extract"

    Invoke-WebRequest -Uri $DownloadUrl -OutFile $ArchivePath -UseBasicParsing
    Expand-Archive -Path $ArchivePath -DestinationPath $ExtractDir -Force

    $UvExe = Get-ChildItem -Path $ExtractDir -Filter "uv.exe" -Recurse | Select-Object -First 1
    if (-not $UvExe) {
        Write-Error "Could not find uv.exe in downloaded archive"
        exit 1
    }

    $DestPath = Join-Path $RuntimeDir "uv.exe"
    Copy-Item -Path $UvExe.FullName -Destination $DestPath -Force

    Write-Host "Staged uv binary: $DestPath"
    & $DestPath --version
}
finally {
    Remove-Item -Path $TempDir -Recurse -Force -ErrorAction SilentlyContinue
}
