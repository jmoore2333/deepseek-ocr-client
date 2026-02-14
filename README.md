# DeepSeek OCR Client (Fork)

Desktop Electron client for [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR), with cross-platform packaging, queue workflows, GPU-aware setup, and first-run managed Python runtime.

This project is unaffiliated with DeepSeek.

## What This Fork Adds

- Managed first-run runtime via bundled `uv` (packaged builds)
- Hardware-aware setup path (Apple Silicon MLX, NVIDIA CUDA, CPU)
- Dual inference backends:
  - Apple Silicon: `mlx` + `mlx-vlm` with `mlx-community/DeepSeek-OCR-2-8bit`
  - Windows/Linux/Intel macOS: PyTorch backend (`torch` + `transformers`)
- Queue processing for mixed image + PDF inputs
- Queue controls: pause, resume, cancel, retry failed
- PDF page-range support (`1-3,5`) for single OCR and queue OCR
- Startup/setup progress UI
- Preflight estimator (disk + download/time estimates)
- Diagnostics bundle export (for debugging/support)
- Retention policy UI (outputs/cache cleanup)
- Security hardening: preload-based API with `contextIsolation: true`
- Cross-platform E2E and packaging test coverage

## Platform Support

- macOS: source mode + DMG packaging
- Windows: source mode + NSIS packaging
- Linux: source mode + AppImage/deb packaging

## Requirements

### End users (installer builds)

- No preinstalled Python required
- Internet access on first run (runtime + dependencies + model)

### Source/development mode

- Node.js 18+
- Python 3.10-3.12
- Optional: NVIDIA driver/CUDA-capable GPU for CUDA acceleration

## Quick Start

### 1. Clone and install Node deps

```bash
npm ci
```

### 2. Run in source mode

macOS/Linux:

```bash
./start-client.sh
```

Windows:

```bat
start-client.bat
```

### 3. In the app

1. Click `Load Model` (first time downloads model)
2. Drop/select an image or PDF
3. Click `Run OCR`

## Queue Workflow

- Add files or add a folder
- Optional PDF page range: `1-3,5`
- Click `Process Queue`
- Use queue controls while running:
  - `Pause`
  - `Resume`
  - `Cancel`
  - `Retry Failed`

Queue outputs are written under timestamped folders in the app cache output directory.

## Basic vs Advanced Mode

- `Basic`: simplified UI for common OCR flow
- `Advanced`: exposes tuning + diagnostics + retention controls

Mode is persisted locally.

## Preflight, Diagnostics, and Retention

- `Run Preflight`: estimates required disk/download and expected setup time
- `Export Diagnostics`: writes a ZIP with app/backend diagnostics and log tails
- Retention policy panel:
  - Output retention days
  - Max queue run folders
  - Download-cache retention days
  - Optional cleanup on startup

## Build Installers / Distributions

### Windows (NSIS)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build-release.ps1
```

### macOS/Linux

```bash
bash ./scripts/build-release.sh
```

### Manual dist commands

```bash
npm run dist:win
npm run dist:mac
npm run dist:linux
```

## Testing

### Electron E2E (mock backend)

```bash
npm run test:e2e
```

### Host build smoke

```bash
npm run test:build:smoke
```

### Host dist packaging

```bash
npm run test:dist:host
```

## CI

- `.github/workflows/ci-e2e.yml`
  - E2E + build smoke on Linux/macOS/Windows
- `.github/workflows/dist-matrix.yml`
  - Full host distribution packaging matrix (manual dispatch)

## Security Model

Renderer access is exposed through a restricted preload API:

- `contextIsolation: true`
- `nodeIntegration: false`
- IPC channels wrapped by explicit preload methods

## MCP Server / Claude Code Plugin

This project includes an MCP (Model Context Protocol) server that enables full programmatic control of the OCR application through Claude Code or any MCP-compatible LLM client.

**What you can do:** Submit files for OCR, manage the processing queue, monitor progress, load models, export diagnostics, and manage retention — all through natural language in Claude Code.

**Quick start:** The project-level config at `.claude/mcp_servers.json` activates automatically when you open Claude Code in this directory. Or install the plugin:

```bash
claude plugin install --path ./mcp-plugin
```

See [`docs/mcp-server.md`](docs/mcp-server.md) for full documentation, available tools/resources, and usage examples.

## Project Layout

- `main.js`: Electron main process, runtime bootstrap, IPC, setup flow
- `preload.js`: secure renderer bridge
- `renderer.js`: UI logic
- `backend/ocr_server.py`: Flask OCR backend + queue processing
- `backend/mcp_server.py`: MCP server for LLM integration
- `mcp-plugin/`: Claude Code plugin package
- `runtime/`: bundled `uv` binaries
- `scripts/`: release/test helpers
- `tests/e2e/`: Playwright E2E suite + mock backend
- `docs/`: analysis, testing strategy, and MCP documentation

## Troubleshooting

- Model/setup issues:
  - Run `Run Preflight` first, confirm disk/network are sufficient
- Runtime/setup anomalies:
  - Export diagnostics and inspect `diagnostics.json`
- Queue stalls:
  - Check queue state in UI (`paused`, `cancel requested`, failed items)

## Upstream and Fork Context

Original project:

- [ihatecsv/deepseek-ocr-client](https://github.com/ihatecsv/deepseek-ocr-client)

This repository is an actively maintained fork focused on installer/runtime reliability and multi-platform operation.

## License

MIT
