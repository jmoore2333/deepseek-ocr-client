# Testing Strategy: DeepSeek OCR Client

**Date:** 2026-02-13

---

## Current Test Coverage

| Category | Status | Location |
|----------|--------|----------|
| E2E (mock backend) | Implemented | `tests/e2e/app.e2e.spec.js` |
| Mock backend server | Implemented | `tests/e2e/mock_backend.js` |
| Build smoke test | Implemented | `scripts/test-build-smoke.js` |
| Distribution test | Implemented | `scripts/test-dist-host.js` |
| CI pipeline | Implemented | `.github/workflows/ci-e2e.yml` |
| Unit tests (backend) | **Missing** | - |
| Unit tests (frontend) | **Missing** | - |
| Integration tests | **Missing** | - |
| Performance benchmarks | **Missing** | - |
| Real-model E2E | **Missing** | - |

---

## Proposed Test Architecture

```
tests/
├── unit/
│   ├── backend/
│   │   ├── test_ocr_endpoints.py      # Flask endpoint tests
│   │   ├── test_queue_processing.py    # Queue logic tests
│   │   ├── test_device_selection.py    # Device detection tests
│   │   ├── test_pdf_extraction.py      # PDF page extraction
│   │   └── conftest.py                 # Shared fixtures
│   └── frontend/
│       ├── renderer.test.js            # Renderer logic tests
│       └── setup.js                    # Test setup
├── integration/
│   ├── test_flask_inference.py         # Real model inference
│   ├── test_electron_flask.js          # IPC → HTTP → response
│   └── test_mcp_server.py             # MCP → Flask integration
├── e2e/                                # (existing)
│   ├── app.e2e.spec.js                 # Mock backend E2E
│   ├── app.real.e2e.spec.js            # Real backend E2E (new)
│   └── mock_backend.js                 # Mock server
├── performance/
│   ├── benchmark_ocr.py                # OCR latency benchmarks
│   ├── benchmark_queue.py              # Queue throughput benchmarks
│   └── benchmark_report.py             # Report generator
└── fixtures/
    ├── images/
    │   ├── simple_text.png             # Simple OCR test image
    │   ├── complex_layout.png          # Multi-column layout
    │   ├── handwritten.png             # Handwriting sample
    │   └── corrupted.png               # Invalid image for error testing
    └── pdfs/
        ├── single_page.pdf             # 1-page test PDF
        ├── multi_page.pdf              # 5-page test PDF
        └── corrupted.pdf               # Invalid PDF for error testing
```

---

## 1. Unit Tests

### Backend (Python, pytest)

**Flask Endpoint Tests** (`test_ocr_endpoints.py`):
- `POST /ocr` with valid image returns OCR text
- `POST /ocr` with invalid file returns appropriate error
- `POST /ocr` with PDF + page range extracts correct pages
- `GET /health` returns model status and device info
- `GET /progress` returns current operation state
- `GET /model_info` returns model configuration
- `GET /diagnostics` returns runtime diagnostics
- `POST /load_model` triggers model loading

Mock the PyTorch model to avoid actual inference in unit tests.

**Queue Processing Tests** (`test_queue_processing.py`):
- Add files to queue → queue status reflects additions
- Start processing → items transition through states
- Pause → processing stops, items remain pending
- Resume → processing continues from where it stopped
- Cancel → all pending items marked cancelled
- Retry failed → failed items reset to pending
- Clear → all items removed
- Concurrent access → thread safety verified

**Device Selection Tests** (`test_device_selection.py`):
- CUDA available → selects CUDA with correct model
- MPS available (no CUDA) → selects MPS with Dogacel model
- Neither available → falls back to CPU
- Device-specific dtype selection verified

**PDF Extraction Tests** (`test_pdf_extraction.py`):
- Single page PDF → extracts 1 page image
- Multi-page PDF → extracts all pages
- Page range "1-3" → extracts pages 1, 2, 3
- Page range "1,3,5" → extracts pages 1, 3, 5
- Invalid page range → returns error
- Corrupted PDF → returns error gracefully

### Frontend (JavaScript, Playwright component testing or Vitest)

**Renderer Logic Tests** (`renderer.test.js`):
- Mode toggle switches between Basic/Advanced
- File selection populates queue display
- Queue status updates render correctly
- Progress bar calculations are accurate
- Error states display appropriate messages
- Settings panel reads/writes correctly

---

## 2. Integration Tests

### Flask + PyTorch (`test_flask_inference.py`)
Requires actual model download. Run selectively (CI tag: `@slow`).
- Load model on available device
- Submit known test image → verify OCR output contains expected text
- Submit known test PDF → verify per-page output
- Queue 3 files → process → verify all results saved

### Electron + Flask (`test_electron_flask.js`)
- Start Electron app → verify Flask backend starts
- IPC `run-ocr` → HTTP `POST /ocr` → result returned to renderer
- IPC `load-model` → model loads → health check passes
- IPC `export-diagnostics` → ZIP file created with correct contents

### MCP + Flask (`test_mcp_server.py`)
- MCP `ocr_process_file` tool → Flask `POST /ocr` → result returned
- MCP `deepseek-ocr://health` resource → correct health data
- MCP queue workflow: add → start → poll → complete
- MCP tools handle Flask down gracefully

---

## 3. E2E Tests

### Mock Backend E2E (existing, `app.e2e.spec.js`)
Already covers:
- Application launch
- File selection dialogs
- Queue operations with mock responses
- OCR invocation flow

### Real Backend E2E (new, `app.real.e2e.spec.js`)
Requires model and GPU. Run selectively (CI tag: `@gpu`).
- Full workflow: launch → setup → load model → select image → OCR → verify output
- PDF workflow: select PDF → set page range → OCR → verify per-page output
- Queue workflow: select folder → process all → verify output folder
- Error recovery: corrupt file in queue → skip → continue remaining
- Retention policy: set policy → trigger cleanup → verify files removed

---

## 4. Platform Matrix Tests

| Test Suite | Windows CUDA | macOS MPS | macOS CPU | Linux CUDA | Linux CPU |
|------------|:----------:|:---------:|:---------:|:----------:|:---------:|
| Unit (backend) | Run | Run | Run | Run | Run |
| Unit (frontend) | Run | Run | Run | Run | Run |
| Integration (real model) | Run | Run | Skip | Run | Skip |
| E2E (mock) | Run | Run | Run | Run | Run |
| E2E (real) | Run | Run | Skip | Run | Skip |
| Performance benchmarks | Run | Run | N/A | Run | N/A |
| Build smoke | Run | Run | N/A | Run | N/A |

### Platform-Specific Test Cases

**Windows:**
- Terminal output encoding (UTF-8 via TextIOWrapper)
- Windows Defender does not flag built binary
- NSIS installer creates correct shortcuts
- `open_latest_queue.bat` opens correct folder

**macOS:**
- MPS device detection works on Apple Silicon
- DMG mounts and app launches
- Gatekeeper does not block unsigned dev builds

**Linux:**
- AppImage runs without additional deps
- xvfb-run works for headless E2E
- CUDA runtime detection with nvidia-smi

---

## 5. Performance Benchmarks

### Metrics to Track

| Metric | Unit | Target (CUDA) | Target (MPS) |
|--------|------|---------------|--------------|
| Model load time | seconds | <30s | <60s |
| Single image OCR (simple) | seconds | <5s | <15s |
| Single image OCR (complex) | seconds | <15s | <45s |
| PDF page extraction | ms/page | <200ms | <200ms |
| PDF OCR (per page) | seconds | <10s | <30s |
| Queue throughput | images/min | >10 | >3 |
| Peak memory (inference) | GB | <6 | <8 |
| Idle memory (model loaded) | GB | <4 | <6 |

### Benchmark Implementation (`tests/performance/`)

```python
# benchmark_ocr.py
def benchmark_single_image(image_path, device, iterations=5):
    """Measure OCR latency for a single image."""
    # Warm-up run (excluded from timing)
    # Timed runs with statistics (mean, std, min, max)

def benchmark_pdf(pdf_path, device, iterations=3):
    """Measure per-page OCR latency for a PDF."""

def benchmark_queue(folder_path, device):
    """Measure queue throughput (images/minute)."""

def benchmark_model_load(device):
    """Measure model loading time."""
```

### Benchmark CI Integration
- Run on schedule (weekly) or manual dispatch
- Store results as JSON artifacts
- Compare against previous baseline
- Alert on >20% regression

---

## 6. MCP Server Tests

### Tool Tests
- Each tool returns expected schema
- Error handling for missing Flask backend
- File path validation in `ocr_process_file`
- Queue operations match Flask endpoint behavior

### Resource Tests
- Each resource URI returns valid JSON
- Health resource reflects actual server state
- Progress resource updates during operations

### End-to-End MCP
- Claude Code can discover and connect to MCP server
- Full OCR workflow via MCP tools produces correct output
- Queue workflow via MCP tools matches GUI behavior

---

## 7. Test Fixtures

### Required Test Assets
Create minimal test fixtures that are small enough to commit:
- `fixtures/images/simple_text.png` - Black text on white, ~10 words (~5KB)
- `fixtures/images/complex_layout.png` - Multi-column with headers (~20KB)
- `fixtures/pdfs/single_page.pdf` - One page with text (~10KB)
- `fixtures/pdfs/multi_page.pdf` - 5 pages with varied content (~50KB)
- `fixtures/images/corrupted.png` - Invalid binary data (~1KB)

---

## 8. Running Tests

```bash
# Unit tests (fast, no model needed)
pytest tests/unit/ -v

# Integration tests (needs model, slow)
pytest tests/integration/ -v -m "not slow"
pytest tests/integration/ -v -m "slow"  # includes real inference

# E2E tests (mock backend)
npm run test:e2e

# E2E tests (real backend, needs model + GPU)
npx playwright test tests/e2e/app.real.e2e.spec.js

# Performance benchmarks
python tests/performance/benchmark_ocr.py --device cuda
python tests/performance/benchmark_ocr.py --device mps

# All fast tests (CI default)
pytest tests/unit/ -v && npm run test:e2e

# Full test suite (CI comprehensive)
pytest tests/ -v && npm run test:e2e && npx playwright test tests/e2e/app.real.e2e.spec.js
```

---

## 9. Implementation Priority

1. **Test fixtures** - Create minimal test images/PDFs (required by everything else)
2. **Backend unit tests** - Highest coverage impact, fastest to write
3. **Real-model E2E** - Validates the full user workflow
4. **Performance benchmarks** - Establishes baselines before MPS optimization
5. **Frontend unit tests** - Lower priority, E2E covers most UI paths
6. **MCP server tests** - After MCP server is implemented
7. **Platform matrix expansion** - After Linux testing infrastructure is set up
