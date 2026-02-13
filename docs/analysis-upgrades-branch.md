# Branch Analysis: feature/upgrades

## Overview

The `feature/upgrades` branch builds on top of `main` (which already contains the cuda-queue-optimization work). It represents a comprehensive production-readiness push: PDF support, security hardening, E2E testing, CI/CD pipelines, UI refinements, MPS performance tuning, and branded assets.

**Branch tip:** `1c0f7a3` (2026-02-13)
**Base:** Branched from `main` at `3308c95` (Merge feature/cuda-queue-optimization into main)
**Status:** 9 commits ahead of main (10 including merged codex branch)

---

## Commit History

| Hash | Date | Description |
|------|------|-------------|
| `7457e65` | 2026-02-12 | feat: add startup setup progress banner |
| `38eff2c` | 2026-02-12 | feat: add PDF OCR input support and update repo links |
| `48ff59b` | 2026-02-12 | feat: show uniform page progress in queue status |
| `fc50bdc` | 2026-02-12 | feat: add branded app icons and smooth progress behavior |
| `6ffbd1e` | 2026-02-12 | test: add cross-platform e2e and build/dist matrix |
| `1147624` | 2026-02-12 | Refine UI and improve OCR runtime reliability |
| `8ab97c8` | 2026-02-12 | Merge branch 'codex/frontend-design-ui-evaluation' |
| `3528eef` | 2026-02-12 | Speed up MPS PDF OCR with tighter defaults and cap logging |
| `1c0f7a3` | 2026-02-13 | fix: resolve Defender false positive, add PDF preview stage, add form-data dep |

---

## Features Added (on top of cuda branch)

### 1. PDF OCR Input Support (`38eff2c`)
- Accept PDF files alongside images
- Page range selection (e.g., "1-3,5,7-9")
- `pypdfium2` for PDF page extraction
- Per-page progress tracking during queue processing

### 2. Startup Setup Progress Banner (`7457e65`)
- Visual progress indicator during first-run Python environment setup
- Stage-by-stage feedback (downloading uv, creating venv, installing deps, downloading model)

### 3. Branded Application Icons (`fc50bdc`)
- Custom `.icns` (macOS), `.ico` (Windows), `.png` (Linux) icons
- Smooth progress bar behavior improvements

### 4. Queue Page Progress (`48ff59b`)
- Uniform per-page progress display in queue status
- Individual page tracking for multi-page PDFs

### 5. E2E Test Suite & CI/CD (`6ffbd1e`)
- Playwright-based E2E tests (`tests/e2e/app.e2e.spec.js`, 195 lines)
- Mock Flask backend for testing without real model (`tests/e2e/mock_backend.js`, 523 lines)
- GitHub Actions CI workflow (`ci-e2e.yml`) - runs on Ubuntu/Windows/macOS
- Distribution packaging matrix (`dist-matrix.yml`) - manual dispatch
- Build smoke test (`scripts/test-build-smoke.js`)
- Distribution host test (`scripts/test-dist-host.js`)
- Playwright configuration (`playwright.config.js`)

### 6. UI Refinements (`1147624`, merged via `8ab97c8`)
- Codex-generated frontend design evaluation
- Improved layout, styling, and interaction patterns
- OCR runtime reliability improvements

### 7. Security Hardening (`1c0f7a3`)
- **preload.js** - Context isolation security bridge (55 lines)
- Content-Security-Policy moved from HTML meta tag to Electron session headers
- Resolves Windows Defender `Trojan:JS/ChatGPTStealer.GVA!MTB` false positive

### 8. MPS PDF Performance Tuning (`3528eef`)
- Tighter default parameters for MPS device
- Generation token cap enforcement
- Reduced logging overhead during inference

### 9. PDF Preview Stage (`1c0f7a3`)
- Live PDF page preview during OCR processing
- `/current_page_image` backend endpoint
- Frontend polling for real-time page rendering display

### 10. Dependency Addition (`1c0f7a3`)
- `form-data` npm package added for multipart form data in IPC handlers

---

## Architecture Evolution

```
Electron (main.js)
  │
  ├── preload.js (security bridge, context isolation)
  │      │
  │      └── appAPI (restricted IPC surface)
  │              │
  └── renderer.js (UI logic, no direct Node access)
         │
    Flask Backend (ocr_server.py)
         │
    PyTorch Model (CUDA/MPS/CPU)
         │
    pypdfium2 (PDF extraction)

Testing:
  Playwright ──> Electron App ──> Mock Backend

CI/CD:
  GitHub Actions ──> E2E Tests (3 platforms)
                ──> Build Smoke Tests
                ──> Distribution Packaging Matrix
```

Key architectural changes from cuda branch:
- **Context isolation** via preload.js (security hardening)
- **E2E test infrastructure** with mock backend
- **CI/CD pipelines** for automated testing and distribution
- **PDF processing pipeline** integrated into existing OCR flow

---

## File Inventory (35 files at branch tip)

### New files (not in cuda branch)
| File | Lines | Purpose |
|------|-------|---------|
| `preload.js` | 55 | Electron security bridge |
| `playwright.config.js` | 15 | E2E test configuration |
| `tests/e2e/app.e2e.spec.js` | 195 | Playwright E2E test suite |
| `tests/e2e/mock_backend.js` | 523 | Mock Flask backend for testing |
| `scripts/test-build-smoke.js` | 58 | Post-build validation |
| `scripts/test-dist-host.js` | 70 | Distribution packaging test |
| `.github/workflows/ci-e2e.yml` | 64 | E2E + build CI workflow |
| `.github/workflows/dist-matrix.yml` | 59 | Distribution packaging workflow |
| `assets/icon.png` | - | Application icon (base) |
| `assets/icons/icon.icns` | - | macOS application icon |
| `assets/icons/icon.ico` | - | Windows application icon |
| `assets/icons/icon.png` | - | Linux application icon |

### Modified files (from cuda branch)
| File | Change | Purpose |
|------|--------|---------|
| `backend/ocr_server.py` | +1,558/-varies | PDF support, MPS tuning, page preview endpoint |
| `main.js` | +839/-varies | Setup progress, preload integration, diagnostics |
| `renderer.js` | +947/-varies | Queue UI, mode toggle, PDF workflow |
| `index.html` | +268/-varies | UI restructure for new features |
| `styles.css` | +1,569/-varies | Complete styling overhaul |
| `README.md` | +217/-varies | Comprehensive documentation rewrite |
| `package.json` | +16/-varies | New deps, build targets, test scripts |
| `requirements.txt` | +1 | Additional Python dependency |
| `.gitignore` | +2 | Additional ignore patterns |

---

## Diff Statistics (from main)

```
22 files changed, 5,218 insertions(+), 1,306 deletions(-)
```

Core code growth:
- `backend/ocr_server.py`: ~1,909 lines total (from ~530 at cuda tip)
- `main.js`: ~1,496 lines total (from ~657 at cuda tip)
- `renderer.js`: ~1,894 lines total (from ~947 at cuda tip)
- `styles.css`: ~1,569 lines total (from ~170 at cuda tip)

---

## Known Issues

1. **MPS (Apple Silicon) Performance:** Logging suggests OCR is working but processing is notably slow. The Dogacel/DeepSeek-OCR-Metal-MPS model may not be optimal for Apple Silicon. MPS PDF defaults were tightened in `3528eef` but fundamental performance gap remains.

2. **Linux + CUDA:** Untested. Expected to work since Windows CUDA is verified, but needs validation.

3. **Model Selection:** Hardcoded model paths based on device type. No user-configurable model selection.

4. **E2E Tests:** Use mock backend only. No real-model integration tests yet.

5. **No Unit Tests:** Backend Flask endpoints and frontend renderer logic lack unit test coverage.
