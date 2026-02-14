#!/bin/bash
# Download the uv binary for macOS/Linux and stage it in runtime/uv.
#
# Usage:
#   ./scripts/download-uv.sh

set -euo pipefail

UV_VERSION="${1:-0.6.6}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RUNTIME_DIR="$PROJECT_ROOT/runtime"
TARGET_UV="$RUNTIME_DIR/uv"
FORCE_DOWNLOAD="${FORCE_UV_DOWNLOAD:-0}"

mkdir -p "$RUNTIME_DIR"

if [[ "$FORCE_DOWNLOAD" != "1" && -x "$TARGET_UV" ]]; then
  echo "Using existing bundled uv: $TARGET_UV"
  "$TARGET_UV" --version
  exit 0
fi

if [[ "$FORCE_DOWNLOAD" != "1" ]] && command -v uv >/dev/null 2>&1; then
  SYSTEM_UV="$(command -v uv)"
  echo "Using system uv from PATH: $SYSTEM_UV"
  cp "$SYSTEM_UV" "$TARGET_UV"
  chmod +x "$TARGET_UV"
  "$TARGET_UV" --version
  exit 0
fi

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin) OS_PART="apple-darwin" ;;
  Linux) OS_PART="unknown-linux-gnu" ;;
  *)
    echo "Unsupported OS: $OS"
    exit 1
    ;;
esac

case "$ARCH" in
  x86_64|amd64) ARCH_PART="x86_64" ;;
  arm64|aarch64) ARCH_PART="aarch64" ;;
  *)
    echo "Unsupported architecture: $ARCH"
    exit 1
    ;;
esac

PLATFORM="${ARCH_PART}-${OS_PART}"
ARCHIVE_NAME="uv-${PLATFORM}.tar.gz"
DOWNLOAD_URL="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${ARCHIVE_NAME}"

TMP_DIR="$(mktemp -d /tmp/uv-download.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Detected platform: $PLATFORM"
echo "Downloading uv ${UV_VERSION}..."

if ! curl -fL "$DOWNLOAD_URL" -o "$TMP_DIR/$ARCHIVE_NAME"; then
  echo "Failed to download uv from GitHub."
  echo "Set FORCE_UV_DOWNLOAD=0 and ensure either runtime/uv or system 'uv' exists for offline builds."
  exit 1
fi
tar -xzf "$TMP_DIR/$ARCHIVE_NAME" -C "$TMP_DIR"

UV_BIN="$(find "$TMP_DIR" -type f -name 'uv' | head -n 1)"
if [[ -z "$UV_BIN" ]]; then
  echo "Could not locate uv binary in archive"
  exit 1
fi

cp "$UV_BIN" "$TARGET_UV"
chmod +x "$TARGET_UV"

echo "Staged uv binary: $TARGET_UV"
"$TARGET_UV" --version
