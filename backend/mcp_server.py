#!/usr/bin/env python3
"""MCP Server for DeepSeek OCR Client.

Provides full programmatic control of the DeepSeek OCR desktop application
via the Model Context Protocol (MCP). Wraps the Flask backend HTTP API and
adds Electron-only features (retention policy, preflight, diagnostics export).

Usage:
    python backend/mcp_server.py          # stdio transport (default)
    DEEPSEEK_OCR_URL=http://... python backend/mcp_server.py

Requires the Flask backend (ocr_server.py) to be running.
"""

import json
import mimetypes
import os
import platform
import shutil
import time
from pathlib import Path
from threading import Thread

import httpx
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FLASK_URL = os.environ.get("DEEPSEEK_OCR_URL", "http://127.0.0.1:5000")
HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)

# Electron userData paths (platform-specific)
if platform.system() == "Darwin":
    _app_data = Path.home() / "Library" / "Application Support" / "DeepSeek OCR Client"
elif platform.system() == "Windows":
    _app_data = Path(os.environ.get("APPDATA", "")) / "DeepSeek OCR Client"
else:
    _app_data = Path.home() / ".config" / "DeepSeek OCR Client"

RETENTION_POLICY_PATH = _app_data / "retention-policy.json"

SUPPORTED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".pdf"
}

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "deepseek-ocr",
    instructions="Full control of the DeepSeek OCR desktop application. "
    "Provides tools for OCR processing, queue management, model loading, "
    "diagnostics, and retention policy management.",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client() -> httpx.Client:
    """Return a configured httpx client."""
    return httpx.Client(base_url=FLASK_URL, timeout=HTTP_TIMEOUT)


def _guess_mime(file_path: str) -> str:
    mime, _ = mimetypes.guess_type(file_path)
    return mime or "application/octet-stream"


def _flask_available() -> bool:
    """Check if the Flask backend is reachable."""
    try:
        with _client() as c:
            r = c.get("/health")
            return r.status_code == 200
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return False


def _read_retention_policy() -> dict:
    """Read retention policy from the Electron userData JSON file."""
    defaults = {
        "outputRetentionDays": 0,
        "maxQueueRuns": 0,
        "downloadCacheRetentionDays": 0,
        "cleanupOnStartup": False,
    }
    if RETENTION_POLICY_PATH.is_file():
        try:
            data = json.loads(RETENTION_POLICY_PATH.read_text())
            defaults.update(data)
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def _write_retention_policy(policy: dict) -> dict:
    """Write retention policy to the Electron userData JSON file."""
    RETENTION_POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RETENTION_POLICY_PATH.write_text(json.dumps(policy, indent=2))
    return policy


# ---------------------------------------------------------------------------
# TOOLS — state-changing actions
# ---------------------------------------------------------------------------


@mcp.tool()
def ocr_process_file(
    file_path: str,
    prompt_type: str = "document",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    pdf_page_range: str = "",
) -> dict:
    """Submit a single image or PDF file for OCR processing.

    Args:
        file_path: Absolute path to the image or PDF file.
        prompt_type: One of: document, ocr, free, figure, describe.
        base_size: Base resolution for OCR processing.
        image_size: Image size parameter for OCR.
        crop_mode: Whether to enable crop mode.
        pdf_page_range: Optional page range for PDFs (e.g. "1-3,5").
    """
    p = Path(file_path)
    if not p.is_file():
        return {"status": "error", "message": f"File not found: {file_path}"}
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return {"status": "error", "message": f"Unsupported file type: {p.suffix}"}

    with _client() as c:
        with open(file_path, "rb") as f:
            files = {"image": (p.name, f, _guess_mime(file_path))}
            data = {
                "prompt_type": prompt_type,
                "base_size": str(base_size),
                "image_size": str(image_size),
                "crop_mode": str(crop_mode).lower(),
            }
            if pdf_page_range:
                data["pdf_page_range"] = pdf_page_range
            r = c.post("/ocr", files=files, data=data)
    return r.json()


@mcp.tool()
def queue_add_files(
    file_paths: list[str],
    prompt_type: str = "document",
    base_size: int = 1024,
    image_size: int = 640,
    crop_mode: bool = True,
    pdf_page_range: str = "",
) -> dict:
    """Add one or more files to the OCR processing queue.

    Args:
        file_paths: List of absolute paths to image or PDF files.
        prompt_type: One of: document, ocr, free, figure, describe.
        base_size: Base resolution for OCR processing.
        image_size: Image size parameter for OCR.
        crop_mode: Whether to enable crop mode.
        pdf_page_range: Optional page range for PDFs (e.g. "1-3,5").
    """
    valid_files = []
    for fp in file_paths:
        p = Path(fp)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
            valid_files.append(p)

    if not valid_files:
        return {"status": "error", "message": "No valid image/PDF files provided"}

    with _client() as c:
        files_list = []
        open_handles = []
        try:
            for p in valid_files:
                fh = open(p, "rb")
                open_handles.append(fh)
                files_list.append(("files", (p.name, fh, _guess_mime(str(p)))))

            data = {
                "prompt_type": prompt_type,
                "base_size": str(base_size),
                "image_size": str(image_size),
                "crop_mode": str(crop_mode).lower(),
            }
            if pdf_page_range:
                data["pdf_page_range"] = pdf_page_range

            r = c.post("/queue/add", files=files_list, data=data)
        finally:
            for fh in open_handles:
                fh.close()

    return r.json()


# Background thread reference for queue processing
_queue_thread = None


@mcp.tool()
def queue_start_processing() -> dict:
    """Start processing the OCR queue (non-blocking).

    Returns immediately. Poll the deepseek-ocr://progress and
    deepseek-ocr://queue/status resources for updates, or use
    queue_wait_for_completion to block until done.
    """
    global _queue_thread

    # Check if already processing
    with _client() as c:
        status = c.get("/queue/status").json()
    if status.get("is_processing"):
        return {"status": "error", "message": "Queue is already processing"}

    def _run():
        try:
            with _client() as c:
                c.post("/queue/process")
        except Exception:
            pass

    _queue_thread = Thread(target=_run, daemon=True)
    _queue_thread.start()

    return {
        "status": "started",
        "message": "Queue processing started. Read deepseek-ocr://progress for updates.",
    }


@mcp.tool()
def queue_wait_for_completion(
    timeout_seconds: int = 600, poll_interval: int = 5
) -> dict:
    """Wait for queue processing to complete, polling progress.

    Args:
        timeout_seconds: Maximum time to wait (default 600 = 10 min).
        poll_interval: Seconds between polls (default 5).
    """
    start = time.time()
    while time.time() - start < timeout_seconds:
        with _client() as c:
            progress = c.get("/progress").json()
            queue = c.get("/queue/status").json()

        if not queue.get("is_processing") and progress.get("status") in (
            "idle",
            "error",
            "loaded",
        ):
            return {
                "status": "completed",
                "queue_summary": queue,
                "final_progress": progress,
            }
        time.sleep(poll_interval)

    return {
        "status": "timeout",
        "message": f"Queue did not complete within {timeout_seconds}s",
    }


@mcp.tool()
def queue_pause() -> dict:
    """Pause active queue processing."""
    with _client() as c:
        r = c.post("/queue/pause")
    return r.json()


@mcp.tool()
def queue_resume() -> dict:
    """Resume paused queue processing."""
    with _client() as c:
        r = c.post("/queue/resume")
    return r.json()


@mcp.tool()
def queue_cancel() -> dict:
    """Cancel active queue processing."""
    with _client() as c:
        r = c.post("/queue/cancel")
    return r.json()


@mcp.tool()
def queue_retry_failed() -> dict:
    """Retry all failed queue items by resetting them to pending."""
    with _client() as c:
        r = c.post("/queue/retry_failed")
    return r.json()


@mcp.tool()
def queue_clear() -> dict:
    """Clear all items from the processing queue."""
    with _client() as c:
        r = c.post("/queue/clear")
    return r.json()


@mcp.tool()
def queue_remove_item(item_id: int) -> dict:
    """Remove a specific item from the queue.

    Args:
        item_id: The numeric ID of the queue item to remove.
    """
    with _client() as c:
        r = c.request("DELETE", f"/queue/remove/{item_id}")
    return r.json()


@mcp.tool()
def model_load() -> dict:
    """Trigger model loading on the backend.

    The model downloads on first use (~3-5 GB). Loading may take 30-120 seconds
    depending on hardware. Poll deepseek-ocr://progress for loading status.
    """
    with _client() as c:
        r = c.post("/load_model")
    return r.json()


@mcp.tool()
def retention_policy_update(
    output_retention_days: int | None = None,
    max_queue_runs: int | None = None,
    download_cache_retention_days: int | None = None,
    cleanup_on_startup: bool | None = None,
) -> dict:
    """Update the retention/cleanup policy settings.

    Args:
        output_retention_days: Days to keep OCR output files (0 = forever).
        max_queue_runs: Maximum number of queue run folders to keep (0 = unlimited).
        download_cache_retention_days: Days to keep model download cache (0 = forever).
        cleanup_on_startup: Whether to run cleanup on app startup.
    """
    policy = _read_retention_policy()
    if output_retention_days is not None:
        policy["outputRetentionDays"] = max(0, output_retention_days)
    if max_queue_runs is not None:
        policy["maxQueueRuns"] = max(0, max_queue_runs)
    if download_cache_retention_days is not None:
        policy["downloadCacheRetentionDays"] = max(0, download_cache_retention_days)
    if cleanup_on_startup is not None:
        policy["cleanupOnStartup"] = bool(cleanup_on_startup)

    saved = _write_retention_policy(policy)
    return {"status": "success", "policy": saved}


@mcp.tool()
def retention_cleanup_run() -> dict:
    """Manually trigger retention cleanup based on current policy.

    Removes old OCR output folders and model cache according to the
    configured retention policy.
    """
    policy = _read_retention_policy()
    removed_runs = 0
    bytes_freed = 0

    # Clean output folders based on maxQueueRuns
    outputs_dir = _app_data / "outputs"
    if policy.get("maxQueueRuns", 0) > 0 and outputs_dir.is_dir():
        runs = sorted(outputs_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        excess = len(runs) - policy["maxQueueRuns"]
        if excess > 0:
            for run_dir in runs[:excess]:
                if run_dir.is_dir():
                    size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
                    shutil.rmtree(run_dir, ignore_errors=True)
                    removed_runs += 1
                    bytes_freed += size

    return {
        "status": "success",
        "removed_runs": removed_runs,
        "bytes_freed": bytes_freed,
        "bytes_freed_human": f"{bytes_freed / (1024 * 1024):.1f} MB",
    }


@mcp.tool()
def diagnostics_export(output_path: str = "") -> dict:
    """Export full diagnostics to a JSON file or return inline.

    Args:
        output_path: Optional absolute path to save diagnostics JSON.
                     If omitted, returns diagnostics data inline.
    """
    with _client() as c:
        backend_diag = c.get("/diagnostics").json()

    diag = {
        "backend": backend_diag,
        "mcp_server": {
            "flask_url": FLASK_URL,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "app_data_dir": str(_app_data),
        },
        "retention_policy": _read_retention_policy(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(diag, indent=2))
        return {"status": "success", "saved_to": output_path}

    return diag


# ---------------------------------------------------------------------------
# RESOURCES — read-only data
# ---------------------------------------------------------------------------


@mcp.resource("deepseek-ocr://health")
def get_health() -> str:
    """Server health status including model state and device info."""
    with _client() as c:
        r = c.get("/health")
    return json.dumps(r.json(), indent=2)


@mcp.resource("deepseek-ocr://progress")
def get_progress() -> str:
    """Current operation progress (model loading, OCR processing, etc.)."""
    with _client() as c:
        r = c.get("/progress")
    return json.dumps(r.json(), indent=2)


@mcp.resource("deepseek-ocr://model-info")
def get_model_info() -> str:
    """Model configuration including name, device, and cache location."""
    with _client() as c:
        r = c.get("/model_info")
    return json.dumps(r.json(), indent=2)


@mcp.resource("deepseek-ocr://queue/status")
def get_queue_status() -> str:
    """Full queue status including all items and their states."""
    with _client() as c:
        r = c.get("/queue/status")
    return json.dumps(r.json(), indent=2)


@mcp.resource("deepseek-ocr://diagnostics")
def get_diagnostics() -> str:
    """Runtime diagnostics snapshot from the backend."""
    with _client() as c:
        r = c.get("/diagnostics")
    return json.dumps(r.json(), indent=2)


@mcp.resource("deepseek-ocr://retention-policy")
def get_retention_policy() -> str:
    """Current retention policy settings."""
    return json.dumps(_read_retention_policy(), indent=2)


@mcp.resource("deepseek-ocr://preflight")
def get_preflight() -> str:
    """Preflight check report including disk space and setup status."""
    report = {
        "app_data_dir": str(_app_data),
        "app_data_exists": _app_data.is_dir(),
        "retention_policy_exists": RETENTION_POLICY_PATH.is_file(),
        "platform": platform.system(),
        "architecture": platform.machine(),
    }

    # Check disk space
    try:
        usage = shutil.disk_usage(str(_app_data.parent))
        report["disk_free_bytes"] = usage.free
        report["disk_free_human"] = f"{usage.free / (1024**3):.1f} GB"
        report["disk_total_bytes"] = usage.total
    except OSError:
        report["disk_free_bytes"] = None

    # Check Flask backend
    report["flask_reachable"] = _flask_available()

    # If Flask is up, get model status
    if report["flask_reachable"]:
        try:
            with _client() as c:
                health = c.get("/health").json()
            report["model_loaded"] = health.get("model_loaded", False)
            report["preferred_device"] = health.get("preferred_device", "unknown")
            report["device_state"] = health.get("device_state", "unknown")
        except Exception:
            report["model_loaded"] = None

    return json.dumps(report, indent=2)


# ---------------------------------------------------------------------------
# PROMPTS — workflow templates
# ---------------------------------------------------------------------------


@mcp.prompt()
def ocr_single_file(file_path: str, quality: str = "balanced") -> str:
    """Process a single image or PDF file with DeepSeek OCR.

    Args:
        file_path: Absolute path to the file to process.
        quality: One of "fast", "balanced", "quality".
    """
    quality_presets = {
        "fast": {"base_size": 512, "image_size": 384, "crop_mode": True},
        "balanced": {"base_size": 1024, "image_size": 640, "crop_mode": True},
        "quality": {"base_size": 2048, "image_size": 1280, "crop_mode": False},
    }
    preset = quality_presets.get(quality, quality_presets["balanced"])

    return f"""Process this file with DeepSeek OCR:
File: {file_path}
Quality: {quality} (base_size={preset['base_size']}, image_size={preset['image_size']}, crop_mode={preset['crop_mode']})

Steps:
1. Read the deepseek-ocr://health resource to check if the backend is running
2. If model is not loaded, call model_load and poll deepseek-ocr://progress until loaded
3. Call ocr_process_file with file_path="{file_path}" and the quality preset parameters
4. Return the OCR result text
"""


@mcp.prompt()
def ocr_batch_folder(folder_path: str, quality: str = "balanced") -> str:
    """Process all images and PDFs in a folder using the queue system.

    Args:
        folder_path: Absolute path to the folder to process.
        quality: One of "fast", "balanced", "quality".
    """
    return f"""Process all images and PDFs in this folder:
Folder: {folder_path}
Quality: {quality}

Steps:
1. Read the deepseek-ocr://health resource to check backend status
2. If model is not loaded, call model_load and wait for it
3. List all image and PDF files in {folder_path}
4. Call queue_add_files with all file paths
5. Call queue_start_processing to begin
6. Call queue_wait_for_completion to wait for results
7. Read deepseek-ocr://queue/status for final results
8. Report summary: how many files processed, any failures
"""


@mcp.prompt()
def system_check() -> str:
    """Run a full system diagnostic check."""
    return """Run a comprehensive system check on the DeepSeek OCR application:

Steps:
1. Read deepseek-ocr://preflight - check disk space and setup status
2. Read deepseek-ocr://health - check backend health and device status
3. Read deepseek-ocr://model-info - check model configuration
4. Read deepseek-ocr://diagnostics - get full runtime diagnostics
5. Read deepseek-ocr://queue/status - check queue state
6. Read deepseek-ocr://retention-policy - check cleanup settings

Report a summary covering:
- Backend status (running/stopped)
- Model status (loaded/not loaded, device in use)
- Hardware (GPU available, device type)
- Disk space (free space, model cache size)
- Queue state (items pending/processing/completed/failed)
- Retention policy settings
- Any issues or warnings
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
