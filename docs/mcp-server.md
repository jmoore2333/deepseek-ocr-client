# DeepSeek OCR — MCP Server & Claude Code Plugin

## What It Does

The MCP (Model Context Protocol) server gives Claude Code (or any MCP-compatible LLM client) full programmatic control over the DeepSeek OCR desktop application. Everything you can do in the GUI, you can do through the MCP server:

- Submit images and PDFs for OCR
- Manage the processing queue (add, pause, resume, cancel, retry, clear)
- Monitor progress in real time
- Load/check model status
- Export diagnostics
- Manage retention policy
- Run preflight checks

The server wraps the same Flask HTTP API that the Electron UI uses, so behavior is identical to using the desktop app directly.

## Architecture

```
Claude Code ──stdio──> MCP Server (mcp_server.py)
                            │
                       HTTP (localhost:5000)
                            │
                       Flask Backend (ocr_server.py)
                            │
                       PyTorch Model (CUDA/MPS/CPU)
```

The MCP server is a Python process that communicates with the Flask backend over HTTP. It uses stdio transport so Claude Code can spawn it as a subprocess. The Flask backend must be running — either through the Electron desktop app or started manually.

## Installation

### Option A: Claude Code Plugin (Recommended)

The plugin is bundled in the `mcp-plugin/` directory of this repository.

```bash
# From the project root, install as a local plugin
claude plugin install --path ./mcp-plugin
```

This registers the MCP server and adds three skills:
- `/deepseek-ocr:ocr-single-file` — Process a single image or PDF
- `/deepseek-ocr:ocr-batch-folder` — Batch process a folder of files
- `/deepseek-ocr:system-check` — Full system diagnostic

### Option B: Project-Level MCP Config

If you prefer not to install the plugin globally, the project already includes a `.claude/mcp_servers.json` that configures the MCP server for this project:

```json
{
  "deepseek-ocr": {
    "command": "python",
    "args": ["backend/mcp_server.py"],
    "env": {
      "DEEPSEEK_OCR_URL": "http://127.0.0.1:5000"
    }
  }
}
```

This activates automatically when you open Claude Code in the project directory.

### Option C: Manual MCP Configuration

Add to your Claude Code settings (`~/.claude/settings.json`) under `mcpServers`:

```json
{
  "mcpServers": {
    "deepseek-ocr": {
      "command": "python",
      "args": ["/absolute/path/to/deepseek-ocr-client/backend/mcp_server.py"],
      "env": {
        "DEEPSEEK_OCR_URL": "http://127.0.0.1:5000"
      }
    }
  }
}
```

### Dependencies

The MCP server requires two additional Python packages (already in `requirements.txt`):

```
mcp[cli]>=1.0.0
httpx>=0.27.0
```

If using the app's managed Python environment, these install automatically. For development:

```bash
pip install "mcp[cli]>=1.0.0" "httpx>=0.27.0"
```

## Prerequisites

The Flask backend must be running before using the MCP server. Two ways:

1. **Open the desktop app** — the Electron app starts the Flask backend automatically
2. **Start manually** — `python backend/ocr_server.py` (uses port 5000)

## Available Tools

Tools are actions that change state or perform work.

| Tool | Description |
|------|-------------|
| `ocr_process_file` | Submit a single image or PDF for OCR. Args: `file_path`, `prompt_type`, `base_size`, `image_size`, `crop_mode`, `pdf_page_range` |
| `queue_add_files` | Add files to the processing queue. Args: `file_paths` (list), same OCR params |
| `queue_start_processing` | Start queue processing (non-blocking, returns immediately) |
| `queue_wait_for_completion` | Block until queue finishes. Args: `timeout_seconds`, `poll_interval` |
| `queue_pause` | Pause active queue processing |
| `queue_resume` | Resume paused queue processing |
| `queue_cancel` | Cancel active queue processing |
| `queue_retry_failed` | Reset failed items to pending for retry |
| `queue_clear` | Clear all items from queue |
| `queue_remove_item` | Remove a specific item. Args: `item_id` |
| `model_load` | Trigger model download/loading |
| `retention_policy_update` | Update cleanup settings. Args: `output_retention_days`, `max_queue_runs`, etc. |
| `retention_cleanup_run` | Run retention cleanup immediately |
| `diagnostics_export` | Export diagnostics to file or return inline. Args: `output_path` (optional) |

## Available Resources

Resources are read-only data you can inspect.

| URI | Description |
|-----|-------------|
| `deepseek-ocr://health` | Backend health: model status, device, GPU availability |
| `deepseek-ocr://progress` | Current operation progress (loading, processing, idle) |
| `deepseek-ocr://model-info` | Model name, cache dir, device, GPU name |
| `deepseek-ocr://queue/status` | Full queue: total/pending/processing/completed/failed items |
| `deepseek-ocr://diagnostics` | Runtime diagnostics snapshot |
| `deepseek-ocr://retention-policy` | Current retention/cleanup policy |
| `deepseek-ocr://preflight` | Setup status, disk space, backend reachability |

## Usage Examples

### Single file OCR

```
> Use the deepseek-ocr MCP server to OCR this file: /Users/me/Documents/receipt.png
```

Claude Code will:
1. Check health → load model if needed → call `ocr_process_file` → return text

### Batch process a folder

```
> Process all the PDFs in /Users/me/Documents/scans/ with the OCR queue
```

Claude Code will:
1. List files → `queue_add_files` → `queue_start_processing` → `queue_wait_for_completion` → report results

### System check

```
> Run a system check on the DeepSeek OCR app
```

Or with the plugin skill:

```
> /deepseek-ocr:system-check
```

### Monitor queue progress

```
> What's the current status of the OCR queue?
```

Claude Code reads `deepseek-ocr://queue/status` and reports.

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_OCR_URL` | `http://127.0.0.1:5000` | Flask backend URL |

### Custom Flask Port

If the Flask backend runs on a different port:

```json
{
  "env": {
    "DEEPSEEK_OCR_URL": "http://127.0.0.1:8080"
  }
}
```

## Troubleshooting

**"Flask backend not reachable"**
- Start the desktop app, or run `python backend/ocr_server.py` manually
- Check that port 5000 is not blocked by another process

**"Model not loaded"**
- Call `model_load` — first run downloads ~3-5 GB
- Check `deepseek-ocr://progress` for download/loading status

**"Queue already processing"**
- Only one queue run at a time. Use `queue_pause` or `queue_cancel` first.

**MCP server won't start**
- Ensure `mcp` and `httpx` are installed: `pip install "mcp[cli]" httpx`
- Verify Python syntax: `python -m py_compile backend/mcp_server.py`

## File Locations

| File | Purpose |
|------|---------|
| `backend/mcp_server.py` | MCP server implementation (605 lines) |
| `mcp-plugin/` | Claude Code plugin package |
| `mcp-plugin/.claude-plugin/plugin.json` | Plugin manifest |
| `mcp-plugin/.mcp.json` | Plugin MCP server config |
| `mcp-plugin/skills/` | Plugin skills (OCR, batch, diagnostics) |
| `.claude/mcp_servers.json` | Project-level MCP config (auto-activates) |
