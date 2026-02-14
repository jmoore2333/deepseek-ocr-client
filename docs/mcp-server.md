# DeepSeek OCR MCP Server

## What It Does

The MCP (Model Context Protocol) server in `backend/mcp_server.py` exposes the same OCR backend used by the Electron UI. Any MCP-compatible client can:

- Run OCR for images and PDFs
- Manage queue workflows (add/start/pause/resume/cancel/retry/clear)
- Check health/progress/model info
- Trigger model load
- Export diagnostics
- Manage retention policy

## Architecture

```text
MCP Client (Codex / Claude / other)
            │ stdio
            ▼
MCP Server (backend/mcp_server.py)
            │ HTTP (DEEPSEEK_OCR_URL)
            ▼
Flask Backend (backend/ocr_server.py)
            │
            ├─ Apple Silicon: MLX backend + mlx-community/DeepSeek-OCR-2-8bit
            └─ CUDA/CPU: PyTorch backend + deepseek-ai/DeepSeek-OCR-2
```

## Prerequisites

- Flask backend running at `DEEPSEEK_OCR_URL` (default `http://127.0.0.1:5000`)
- Python with MCP dependencies (`mcp[cli]`, `httpx`)

Backend start options:

1. Open the desktop app (recommended): backend is started automatically.
2. Run manually: `python backend/ocr_server.py`.

## Dependency Setup (Dev)

```bash
cd /absolute/path/to/deepseek-ocr-client
uv venv venv --python 3.11
uv pip install --python venv/bin/python3 "mcp[cli]>=1.0.0" "httpx>=0.27.0"
```

## Codex App / Codex CLI Setup

No MCP server code changes are required for Codex. The only adjustments needed are:

- Register the server in Codex MCP config
- Use absolute paths for `command` and script args
- Set `DEEPSEEK_OCR_URL` if your backend is not on `127.0.0.1:5000`

### Option A: Register via CLI (recommended)

```bash
cd /absolute/path/to/deepseek-ocr-client
codex mcp add deepseek-ocr --env DEEPSEEK_OCR_URL=http://127.0.0.1:5000 -- \
  "$(pwd)/venv/bin/python3" "$(pwd)/backend/mcp_server.py"
codex mcp list
```

### Option B: Manual `~/.codex/config.toml`

```toml
[mcp_servers.deepseek-ocr]
command = "/absolute/path/to/deepseek-ocr-client/venv/bin/python3"
args = ["/absolute/path/to/deepseek-ocr-client/backend/mcp_server.py"]

[mcp_servers.deepseek-ocr.env]
DEEPSEEK_OCR_URL = "http://127.0.0.1:5000"
```

After saving config, restart Codex App/CLI session and verify with `codex mcp list`.

## Claude Code Setup

- Project-local config exists at `.claude/mcp_servers.json`
- Optional plugin exists at `mcp-plugin/`

```bash
claude plugin install --path ./mcp-plugin
```

## Project Config Files

- `.mcp.json`: generic MCP server config (project-local)
- `.claude/mcp_servers.json`: Claude project config
- `mcp-plugin/.mcp.json`: plugin-scoped MCP config

## Available Tools

| Tool | Description |
|---|---|
| `ocr_process_file` | OCR a single image/PDF (`file_path`, `prompt_type`, `base_size`, `image_size`, `crop_mode`, `pdf_page_range`) |
| `queue_add_files` | Add files to queue |
| `queue_start_processing` | Start queue processing |
| `queue_wait_for_completion` | Wait for queue completion |
| `queue_pause` | Pause queue |
| `queue_resume` | Resume queue |
| `queue_cancel` | Cancel queue |
| `queue_retry_failed` | Retry failed queue items |
| `queue_clear` | Clear queue |
| `queue_remove_item` | Remove queue item by ID |
| `model_load` | Load/download model |
| `retention_policy_update` | Update retention settings |
| `retention_cleanup_run` | Trigger cleanup now |
| `diagnostics_export` | Export diagnostics (inline or to file) |

## Available Resources

| URI | Description |
|---|---|
| `deepseek-ocr://health` | Runtime health and backend/device state |
| `deepseek-ocr://progress` | Current progress (loading/processing/idle) |
| `deepseek-ocr://model-info` | Model name, cache, backend, device |
| `deepseek-ocr://queue/status` | Queue summary and item-level status |
| `deepseek-ocr://diagnostics` | Runtime diagnostics snapshot |
| `deepseek-ocr://retention-policy` | Current retention policy |
| `deepseek-ocr://preflight` | Setup/disk/download preflight report |

## Troubleshooting

**Server not reachable / backend down**

- Confirm app is open, or run `python backend/ocr_server.py`
- Check `DEEPSEEK_OCR_URL` matches the backend URL/port

**MCP server fails to start**

- Verify dependencies: `pip install "mcp[cli]" httpx`
- Validate script: `python -m py_compile backend/mcp_server.py`
- Re-add server with absolute paths

**Model not loaded**

- Call `model_load`
- Poll `deepseek-ocr://progress` until `status=loaded`

