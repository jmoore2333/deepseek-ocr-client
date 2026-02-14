#!/bin/bash
# Build a release bundle for macOS/Linux using bundled uv runtime.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "  DeepSeek OCR Client Release Build"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"

echo "=== Step 1/3: Downloading uv runtime binary ==="
bash "$SCRIPT_DIR/download-uv.sh"
echo ""

echo "=== Step 2/3: Installing Node dependencies ==="
if [[ -d "$PROJECT_ROOT/node_modules" && "${FORCE_NPM_INSTALL:-0}" != "1" ]]; then
  echo "Found existing node_modules; skipping npm install."
  echo "Set FORCE_NPM_INSTALL=1 to force reinstall."
else
  npm install
fi
echo ""

OS="$(uname -s)"
if [[ "$OS" == "Darwin" ]]; then
  BUILD_CMD="dist:mac"
elif [[ "$OS" == "Linux" ]]; then
  BUILD_CMD="dist:linux"
else
  echo "Unsupported OS for this script: $OS"
  exit 1
fi

echo "=== Step 3/3: Building installer artifacts ($BUILD_CMD) ==="
export CSC_IDENTITY_AUTO_DISCOVERY=false
export ELECTRON_BUILDER_CACHE="${ELECTRON_BUILDER_CACHE:-$PROJECT_ROOT/.cache/electron-builder}"
mkdir -p "$ELECTRON_BUILDER_CACHE"
if [[ "$OS" == "Linux" ]]; then
  LINUX_TARGETS="${LINUX_TARGETS:-AppImage deb}"
  read -r -a LINUX_TARGET_ARGS <<< "$LINUX_TARGETS"
  echo "Linux targets: $LINUX_TARGETS"
  npm exec electron-builder -- --linux "${LINUX_TARGET_ARGS[@]}"
else
  npm run "$BUILD_CMD"
fi
echo ""

echo "Build complete."
echo "Artifacts are in: $PROJECT_ROOT/dist"
