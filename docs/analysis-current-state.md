# Current State Analysis: DeepSeek OCR Client

**Date:** 2026-02-13
**Branch:** main (consolidated from feature/cuda-queue-optimization + feature/upgrades)
**Commit:** `158176b`

---

## Application Summary

DeepSeek OCR Client is a cross-platform Electron desktop application providing a GUI for DeepSeek-OCR model inference. It is a fork of [ihatecsv/deepseek-ocr-client](https://github.com/ihatecsv/deepseek-ocr-client) that has been substantially extended with production-grade features.

---

## Complete Feature Inventory

### Core OCR
- Single-file OCR (image + PDF input)
- Multiple prompt types: document, ocr, free, figure, describe
- Configurable parameters: base_size, image_size, crop_mode
- Real-time character generation progress

### Queue Processing
- Multi-file and folder batch processing
- Pause / resume / cancel operations
- Per-item retry on failure
- Organized output folders with timestamps
- `open_latest_queue.bat` for quick result access (Windows)

### PDF Support
- PDF file input alongside images
- Page range selection (e.g., "1-3,5,7-9")
- Per-page progress tracking
- Live page preview during processing
- `pypdfium2` for page extraction

### Hardware Acceleration
- CUDA GPU optimization with `torch.compile`
- MPS (Apple Silicon) support via Dogacel fork model
- CPU fallback mode
- Automatic device detection and selection
- Device-specific dtype optimization

### Managed Runtime
- Bundled `uv` binary for portable Python environment
- First-run setup flow with progress UI
- Python 3.10-3.12 auto-detection
- Cross-platform install scripts (shell + PowerShell)

### UI/UX
- Basic and Advanced mode toggle
- Startup setup progress banner
- Preflight checks (disk space, download estimates, time estimates)
- Diagnostics export (ZIP bundle)
- Retention policy management (output/cache cleanup)
- Branded application icons (all platforms)

### Security
- Context isolation via preload.js
- Restricted IPC API surface (appAPI)
- CSP headers via Electron session (not HTML meta tag)
- No nodeIntegration in renderer

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  Electron App                    │
│                                                  │
│  ┌──────────┐   IPC    ┌──────────────────────┐ │
│  │ renderer  │◄───────►│     main.js           │ │
│  │  .js      │ preload  │  - Runtime bootstrap  │ │
│  │  .html    │  .js     │  - Flask lifecycle    │ │
│  │  .css     │          │  - IPC handlers       │ │
│  └──────────┘          │  - Setup flow         │ │
│                         │  - Diagnostics        │ │
│                         └──────────┬───────────┘ │
│                                    │ HTTP         │
│                         ┌──────────▼───────────┐ │
│                         │  Flask Backend        │ │
│                         │  (ocr_server.py)      │ │
│                         │  - OCR endpoints      │ │
│                         │  - Queue processing   │ │
│                         │  - Model management   │ │
│                         │  - Progress streaming │ │
│                         └──────────┬───────────┘ │
│                                    │              │
│                         ┌──────────▼───────────┐ │
│                         │  PyTorch + HF         │ │
│                         │  Transformers         │ │
│                         │  (CUDA/MPS/CPU)       │ │
│                         └──────────────────────┘ │
│                                                  │
│  ┌──────────────────────────────────────────────┐│
│  │  uv Runtime Manager                          ││
│  │  (Python venv bootstrap + dep installation)  ││
│  └──────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
```

---

## Code Inventory

| File | Lines | Role |
|------|-------|------|
| `backend/ocr_server.py` | 1,909 | Flask OCR backend, queue processing, model management |
| `renderer.js` | 1,894 | Frontend UI logic, state management, event handlers |
| `main.js` | 1,496 | Electron main process, IPC, runtime management |
| `styles.css` | 1,368 | Application styling |
| `start.py` | 436 | Python bootstrap helper |
| `index.html` | 275 | Application shell |
| `tests/e2e/mock_backend.js` | 523 | Mock Flask backend for testing |
| `tests/e2e/app.e2e.spec.js` | 195 | Playwright E2E test suite |
| `preload.js` | 55 | Security bridge (context isolation) |
| `playwright.config.js` | 15 | Test configuration |
| **Total core** | **~7,433** | |

---

## Dependency Audit

### Node.js (package.json)
**Runtime (4):** axios, electron, jszip, marked, form-data
**Dev (4):** @playwright/test, electron-builder, playwright

### Python (requirements.txt, 17 packages)
**ML Core:** torch>=2.4.0, torchvision>=0.19.0, torchaudio>=2.4.0, transformers==4.46.3, tokenizers>=0.20.0, accelerate>=0.25.0
**Web:** flask>=3.0.0, flask-cors>=4.0.0
**Image/PDF:** Pillow>=10.0.0, pypdfium2>=4.30.0
**Utilities:** addict, matplotlib, einops, easydict, safetensors, hf-xet, packaging

---

## Platform Support Matrix

| Platform | Device | Model | Status |
|----------|--------|-------|--------|
| Windows | CUDA | deepseek-ai/deepseek-ocr-base | Verified working |
| Windows | CPU | Dogacel/DeepSeek-OCR-Metal-MPS | Expected working |
| macOS | MPS | Dogacel/DeepSeek-OCR-Metal-MPS | Working but slow |
| macOS | CPU | Dogacel/DeepSeek-OCR-Metal-MPS | Expected working |
| Linux | CUDA | deepseek-ai/deepseek-ocr-base | Untested |
| Linux | CPU | Dogacel/DeepSeek-OCR-Metal-MPS | Untested |

---

## Build & Packaging

| Target | Format | Command |
|--------|--------|---------|
| Windows | NSIS installer | `npm run dist:win` |
| macOS | DMG | `npm run dist:mac` |
| Linux | AppImage + deb | `npm run dist:linux` |

Build pipeline includes:
- `scripts/build-release.sh` / `build-release.ps1` for full release builds
- `scripts/download-uv.sh` / `download-uv.ps1` for runtime binary download
- `scripts/test-build-smoke.js` for post-build validation
- `scripts/test-dist-host.js` for distribution testing

---

## CI/CD Coverage

### ci-e2e.yml (on PR/push/manual)
- Runs on: Ubuntu, Windows, macOS
- E2E tests with Playwright
- Build smoke tests
- Python syntax validation

### dist-matrix.yml (manual dispatch)
- Full distribution packaging for all platforms
- Artifact upload for each target

---

## Known Issues & Gaps

1. **MPS Performance:** Apple Silicon inference is significantly slower than CUDA. Current Dogacel fork model may not be optimal.
2. **Linux Untested:** No verification of CUDA or CPU inference on Linux.
3. **Hardcoded Models:** No user-configurable model selection; device type determines model.
4. **No Unit Tests:** Backend endpoints and frontend logic lack unit test coverage.
5. **Mock-Only E2E:** E2E tests use mock backend; no real-model integration tests.
6. **No MCP Integration:** No programmatic LLM control interface.
7. **No Performance Benchmarks:** No automated performance measurement infrastructure.
