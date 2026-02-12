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
npm install
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
npm run "$BUILD_CMD"
echo ""

echo "Build complete."
echo "Artifacts are in: $PROJECT_ROOT/dist"
