# Branch Analysis: feature/cuda-queue-optimization

## Overview

The `feature/cuda-queue-optimization` branch represents the first major divergence from the original upstream `ihatecsv/deepseek-ocr-client`. It introduced GPU acceleration, batch queue processing, a managed Python runtime, and cross-platform installation infrastructure.

**Branch tip:** `ef7a24b` (2026-02-12)
**Base:** Forked from upstream main after commit `109629b` (Fix for newer nvidia cards)
**Status:** Merged into main at `3308c95`

---

## Commit History

| Hash | Date | Description |
|------|------|-------------|
| `8633168` | 2025-10-26 | feat: Add CUDA optimization, queue processing, and critical bug fixes |
| `580d95c` | 2025-11-27 | Fix blank terminal output on Windows |
| `c3d819a` | 2025-11-27 | Add Python version check to shell script and fix bare except |
| `9ab9ce9` | 2025-11-29 | Fix blank terminal: remove TextIOWrapper, use flush=True |
| `ef7a24b` | 2026-02-12 | feat: add uv-based packaged runtime and cross-platform install flow |

---

## Features Introduced

### 1. CUDA GPU Optimization
- `torch.compile` integration for optimized inference
- Optimal dtype selection per GPU capability
- Device-aware model loading (CUDA / MPS / CPU fallback)

### 2. Batch Queue Processing
- Multi-file and folder selection
- Real-time preview during queue processing
- Progress tracking and status indicators
- Organized output folders with auto-save
- `open_latest_queue.bat` helper for quick access to results

### 3. Managed Python Runtime (uv)
- Bundled `uv` binary for fast, portable Python environment setup
- Cross-platform install scripts (`download-uv.sh`, `download-uv.ps1`)
- Build/release scripts for all platforms (`build-release.sh`, `build-release.ps1`)
- Python 3.10-3.12 auto-detection (3.13+ excluded for PyTorch compatibility)

### 4. Critical Bug Fixes
- Fixed syntax error in DeepSeek OCR model (line 914)
- Fixed dtype mismatch in `masked_scatter_` operation
- Fixed blank terminal output on Windows (replaced codecs.StreamWriter with io.TextIOWrapper)
- Fixed bare `except` clauses to use `except Exception`

### 5. Windows Terminal Reliability
- UTF-8 encoding with `line_buffering=True` for immediate output
- Graceful encoding fallback with `errors='replace'`

---

## Architecture at This Point

```
Electron (main.js) ──IPC──> Flask Backend (ocr_server.py)
       │                          │
       │                     PyTorch Model
       │                     (CUDA/MPS/CPU)
       │
  uv Runtime Manager
  (Python venv bootstrap)
```

- **Electron main process** manages Flask subprocess lifecycle and IPC
- **Flask backend** handles OCR inference with device-aware model loading
- **uv runtime** provides portable Python environment without system Python dependency
- **No preload.js** - direct nodeIntegration in renderer (security hardened later)

---

## File Inventory (23 files at branch tip)

| File | Purpose |
|------|---------|
| `main.js` | Electron main process + runtime management |
| `renderer.js` | Frontend UI logic |
| `index.html` | Application shell |
| `styles.css` | Styling |
| `backend/ocr_server.py` | Flask OCR backend with queue processing |
| `backend/__init__.py` | Python package marker |
| `package.json` | Node dependencies + electron-builder config |
| `package-lock.json` | Locked Node dependencies |
| `requirements.txt` | Python dependencies (PyTorch, transformers, Flask, etc.) |
| `start-client.sh` | macOS/Linux launcher with Python version detection |
| `start-client.bat` | Windows launcher |
| `start.py` | Python bootstrap helper |
| `runtime/README.md` | uv binary bundling documentation |
| `scripts/download-uv.sh` | Download uv binary (Unix) |
| `scripts/download-uv.ps1` | Download uv binary (Windows) |
| `scripts/build-release.sh` | Build release (Unix) |
| `scripts/build-release.ps1` | Build release (Windows) |
| `open_latest_queue.bat` | Open latest queue output folder (Windows) |
| `.gitignore` | Ignore rules |
| `LICENSE.md` | License |
| `README.md` | Documentation |
| `docs/images/document.gif` | Demo screenshot |
| `docs/images/document2.png` | Demo screenshot |

---

## Diff Statistics (from upstream base)

```
19 files changed, 6,008 insertions(+), 776 deletions(-)
```

Major files by change volume:
- `package-lock.json`: +4,376 (dependency tree expansion)
- `main.js`: +539 (runtime management, IPC handlers)
- `backend/ocr_server.py`: +532 (CUDA optimization, queue processing)
- `renderer.js`: +392 (queue UI, progress tracking)
- `start.py`: +242 (Python bootstrap expansion)
- `styles.css`: +170 (queue visualization styling)

---

## Known State

- **Windows + CUDA:** Verified working (primary development/test platform)
- **macOS + MPS:** Functional but performance not optimized
- **Linux:** Untested at this branch point
- **Model:** Uses `deepseek-ai/deepseek-ocr-base` for CUDA, `Dogacel/DeepSeek-OCR-Metal-MPS` for MPS/CPU
