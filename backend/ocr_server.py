#!/usr/bin/env python3
"""
DeepSeek OCR Backend Server
Handles model loading, caching, and OCR inference
"""
import os
import sys
import logging
import re
import math
import shutil
import platform
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tempfile
import time
import json
import functools
from collections import deque
from datetime import datetime
from threading import Thread, Lock, Event
from pathlib import Path

# Linux CUDA fragmentation mitigation.
# Must be configured before importing torch.
if platform.system() == 'Linux':
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

TORCH_IMPORT_ERROR = None
try:
    import torch
except Exception as exc:  # pragma: no cover - depends on runtime platform/env.
    torch = None
    TORCH_IMPORT_ERROR = str(exc)

TRANSFORMERS_IMPORT_ERROR = None
try:
    from transformers import AutoModel, AutoTokenizer
except Exception as exc:  # pragma: no cover - depends on runtime platform/env.
    AutoModel = None
    AutoTokenizer = None
    TRANSFORMERS_IMPORT_ERROR = str(exc)

MLX_IMPORT_ERROR = None
try:
    from mlx_vlm import load as mlx_load, generate as mlx_generate, stream_generate as mlx_stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template as mlx_apply_chat_template
except Exception as exc:  # pragma: no cover - depends on runtime platform/env.
    mlx_load = None
    mlx_generate = None
    mlx_stream_generate = None
    mlx_apply_chat_template = None
    MLX_IMPORT_ERROR = str(exc)

# Reduce noisy tokenizers fork warnings from child process startup/log streaming.
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def parse_env_flag(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


# Linux default: skip torch.compile unless explicitly enabled, to avoid
# platform-specific mixed-precision runtime issues.
TORCH_COMPILE_ENABLED = parse_env_flag(
    os.environ.get('DEEPSEEK_OCR_ENABLE_TORCH_COMPILE'),
    default=(platform.system() != 'Linux')
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress HTTP request logs from werkzeug
logging.getLogger('werkzeug').setLevel(logging.ERROR)

BACKEND_LOG_BUFFER = deque(maxlen=1200)


class InMemoryLogHandler(logging.Handler):
    """Capture recent backend logs for diagnostics export."""

    def emit(self, record):
        try:
            BACKEND_LOG_BUFFER.append(self.format(record))
        except Exception:
            # Logging must never break request processing.
            pass


_log_handler = InMemoryLogHandler()
_log_handler.setLevel(logging.INFO)
_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(_log_handler)

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = 'cpu'
dtype = torch.float32 if torch is not None else 'float32'


def is_device_available(device_name):
    normalized = str(device_name or '').lower()
    if normalized == 'mlx':
        return is_mlx_available()
    if normalized == 'cuda':
        if not is_torch_available():
            return False
        return torch.cuda.is_available()
    if normalized == 'mps':
        if is_mlx_available():
            return True
        if not is_torch_available():
            return False
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    return normalized == 'cpu'


def detect_best_available_device():
    if is_mlx_available():
        return 'mps'
    if not is_torch_available():
        return 'cpu'
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def map_gpu_target_to_device(gpu_target):
    normalized = str(gpu_target or '').strip().lower()
    if normalized == 'mlx':
        return 'mps'
    if normalized.startswith('cuda-'):
        return 'cuda'
    if normalized == 'mps':
        return 'mps'
    return 'cpu'


def is_torch_available():
    return torch is not None


def is_mlx_available():
    return mlx_load is not None and mlx_generate is not None and mlx_apply_chat_template is not None


def resolve_runtime_backend():
    forced_backend = str(os.environ.get('DEEPSEEK_OCR_BACKEND', '')).strip().lower()
    gpu_target_hint = str(os.environ.get('DEEPSEEK_OCR_GPU_TARGET', '')).strip().lower()
    prefer_mlx = gpu_target_hint == 'mlx'

    if forced_backend:
        if forced_backend not in ('torch', 'mlx'):
            logger.warning(
                "Unsupported DEEPSEEK_OCR_BACKEND=%r; expected 'torch' or 'mlx'",
                forced_backend
            )
        elif forced_backend == 'mlx':
            if is_mlx_available():
                return 'mlx'
            logger.warning("MLX backend requested but unavailable: %s", MLX_IMPORT_ERROR or 'mlx-vlm import failed')
        elif forced_backend == 'torch':
            if is_torch_available():
                return 'torch'
            logger.warning("Torch backend requested but unavailable: %s", TORCH_IMPORT_ERROR or 'torch import failed')

    if prefer_mlx:
        if is_mlx_available():
            return 'mlx'
        logger.warning(
            "GPU target hint requested MLX, but mlx-vlm is unavailable: %s. Falling back.",
            MLX_IMPORT_ERROR or 'mlx-vlm import failed'
        )

    if is_torch_available():
        return 'torch'
    if is_mlx_available():
        return 'mlx'
    return 'unavailable'


RUNTIME_BACKEND = resolve_runtime_backend()


def get_preferred_device():
    """Select runtime device based on explicit overrides, then detected availability."""
    if RUNTIME_BACKEND == 'mlx':
        return 'mps'

    if 'DEVICE' in os.environ:
        forced_device = str(os.environ['DEVICE']).strip().lower()
        if forced_device in ('cpu', 'cuda', 'mps', 'mlx'):
            if forced_device == 'mlx':
                forced_device = 'mps'
            if is_device_available(forced_device):
                return forced_device
            fallback = detect_best_available_device()
            logger.warning(
                "DEVICE override %s is unavailable; falling back to %s",
                forced_device,
                fallback
            )
            return fallback

        fallback = detect_best_available_device()
        logger.warning(
            "Unsupported DEVICE override %r; expected cpu/cuda/mps/mlx. Falling back to %s",
            forced_device,
            fallback
        )
        return fallback

    gpu_target_hint = os.environ.get('DEEPSEEK_OCR_GPU_TARGET')
    if gpu_target_hint:
        hinted_device = map_gpu_target_to_device(gpu_target_hint)
        if is_device_available(hinted_device):
            return hinted_device
        fallback = detect_best_available_device()
        logger.warning(
            "GPU target hint %s mapped to %s, but that device is unavailable. Falling back to %s",
            gpu_target_hint,
            hinted_device,
            fallback
        )
        return fallback

    return detect_best_available_device()


def get_preferred_model_name():
    """Select default model based on active inference backend/device."""
    if 'MODEL_NAME' in os.environ:
        return os.environ['MODEL_NAME']
    if RUNTIME_BACKEND == 'mlx':
        return 'mlx-community/DeepSeek-OCR-2-8bit'
    return 'deepseek-ai/DeepSeek-OCR-2'


MODEL_NAME = get_preferred_model_name()
logger.info(
    "Runtime backend selected: %s (gpu_target_hint=%s, model=%s)",
    RUNTIME_BACKEND,
    os.environ.get('DEEPSEEK_OCR_GPU_TARGET'),
    MODEL_NAME
)

# Queue processing state
processing_queue = []
queue_lock = Lock()
current_queue_id = None
queue_results = {}
queue_next_id = 0
queue_paused = False
queue_cancel_requested = False
queue_processing_active = False

# Use a writable cache directory (override via env for packaged apps)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get('DEEPSEEK_OCR_CACHE_DIR')
if CACHE_DIR:
    CACHE_DIR = os.path.abspath(CACHE_DIR)
else:
    CACHE_DIR = os.path.join(SCRIPT_DIR, '..', 'cache')
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, 'models')
model_cache_override = os.environ.get('DEEPSEEK_OCR_MODEL_CACHE_DIR')
if model_cache_override:
    MODEL_CACHE_DIR = os.path.abspath(model_cache_override)
OUTPUT_DIR = os.path.join(CACHE_DIR, 'outputs')

# Keep HuggingFace internals aligned with the explicit model cache directory.
os.environ.setdefault('HF_HOME', MODEL_CACHE_DIR)
os.environ.setdefault('HF_HUB_CACHE', MODEL_CACHE_DIR)
os.environ.setdefault('TRANSFORMERS_CACHE', MODEL_CACHE_DIR)

# Progress tracking
progress_data = {
    'status': 'idle',  # idle, loading, loaded, error
    'stage': '',       # tokenizer, model
    'message': '',
    'progress_percent': 0,  # 0-100
    'chars_generated': 0,  # For OCR character counting
    'raw_token_stream': '',  # Accumulated raw tokens during OCR
    'current_page_image': '',  # Path to current PDF page image being processed
    'timestamp': time.time()
}
progress_lock = Lock()
loading_thread = None
inference_lock = Lock()
progress_log_state = {
    'status': None,
    'stage': None,
    'progress_percent': -1,
    'chars_bucket': -1,
    'logged_at': 0.0
}

def update_progress(status, stage='', message='', progress_percent=0, chars_generated=0, raw_token_stream='', current_page_image=''):
    """Update the global progress data"""
    global progress_data, progress_log_state
    with progress_lock:
        progress_data['status'] = status
        progress_data['stage'] = stage
        progress_data['message'] = message
        progress_data['progress_percent'] = progress_percent
        progress_data['chars_generated'] = chars_generated
        progress_data['raw_token_stream'] = raw_token_stream
        if current_page_image:
            progress_data['current_page_image'] = current_page_image
        elif stage != 'ocr' or status != 'processing':
            progress_data['current_page_image'] = ''
        progress_data['timestamp'] = time.time()
        # Throttle verbose progress logs to reduce overhead while streaming.
        now = progress_data['timestamp']
        chars_bucket = int(chars_generated / 250) if chars_generated > 0 else 0
        should_log = (
            status != progress_log_state['status'] or
            stage != progress_log_state['stage'] or
            abs(int(progress_percent) - int(progress_log_state['progress_percent'])) >= 3 or
            chars_bucket != progress_log_state['chars_bucket'] or
            (now - progress_log_state['logged_at']) >= 2.0
        )
        if should_log:
            progress_log_state = {
                'status': status,
                'stage': stage,
                'progress_percent': int(progress_percent),
                'chars_bucket': chars_bucket,
                'logged_at': now
            }
            if chars_generated > 0:
                logger.info(
                    f"Progress: {status} - {stage} - {message} ({progress_percent}%) - {chars_generated} chars"
                )
            else:
                logger.info(f"Progress: {status} - {stage} - {message} ({progress_percent}%)")

def log_selected_device(selected_device):
    if RUNTIME_BACKEND == 'mlx':
        logger.info("Using Apple Silicon GPU (MLX)")
        return
    if selected_device == 'cuda':
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif selected_device == 'mps':
        logger.info("Using Apple Silicon GPU (MPS)")
    else:
        logger.warning("Using CPU backend (this will be slower)")

def get_cache_dir_size(directory):
    """Get total size of files in directory in bytes"""
    total = 0
    try:
        for entry in os.scandir(directory):
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_cache_dir_size(entry.path)
    except (PermissionError, FileNotFoundError):
        pass
    return total


def is_loading_in_progress():
    return loading_thread is not None and loading_thread.is_alive()


def is_model_ready():
    return (model is not None) and (tokenizer is not None) and (not is_loading_in_progress())


def resolve_max_new_tokens_cap(explicit_cap=None):
    """Resolve an effective max_new_tokens cap."""
    if explicit_cap is not None:
        try:
            return max(128, int(explicit_cap))
        except (TypeError, ValueError):
            logger.warning(f"Invalid max_new_tokens cap {explicit_cap!r}, using device default")

    if device == 'mps':
        return DEFAULT_MAX_NEW_TOKENS_MPS
    return DEFAULT_MAX_NEW_TOKENS_OTHER


def invoke_infer_with_generation_cap(infer_call, max_new_tokens_cap):
    """Temporarily patch model.generate so hardcoded model defaults can't exceed cap."""
    global model
    if model is None or max_new_tokens_cap is None:
        return infer_call()

    generate_fn = getattr(model, 'generate', None)
    if not callable(generate_fn):
        return infer_call()

    original_generate = generate_fn
    patched = False

    cap_log_state = {'logged': False}

    @functools.wraps(original_generate)
    def capped_generate(*args, **gen_kwargs):
        requested = gen_kwargs.get('max_new_tokens')
        try:
            requested_int = int(requested) if requested is not None else None
        except (TypeError, ValueError):
            requested_int = None

        if requested_int is None or requested_int > max_new_tokens_cap:
            gen_kwargs['max_new_tokens'] = max_new_tokens_cap
        if not cap_log_state['logged']:
            cap_log_state['logged'] = True
            logger.info(
                "Applying generation cap: requested_max_new_tokens=%s effective_max_new_tokens=%s",
                requested_int if requested_int is not None else 'unset',
                gen_kwargs.get('max_new_tokens')
            )
        return original_generate(*args, **gen_kwargs)

    try:
        setattr(model, 'generate', capped_generate)
        patched = True
    except Exception as exc:
        logger.warning(f"Unable to patch model.generate for token cap: {exc}")

    try:
        return infer_call()
    finally:
        if patched:
            try:
                setattr(model, 'generate', original_generate)
            except Exception:
                pass


def to_mlx_prompt(raw_prompt):
    prompt = (raw_prompt or '').strip()
    if not prompt:
        return "Convert this image to markdown."

    # Keep <|grounding|> so DeepSeek-OCR-2 emits <|ref|>/<|det|> tokens for overlay boxes.
    prompt = prompt.replace('<image>', '').strip()
    if not prompt:
        return "Convert this image to markdown."
    return prompt


def extract_text_from_mlx_output(generated):
    if generated is None:
        return ''
    if isinstance(generated, str):
        return generated
    text_value = getattr(generated, 'text', None)
    if isinstance(text_value, str):
        return text_value
    return str(generated)


GROUNDING_TOKEN_PATTERN = re.compile(
    r'<\|ref\|>([^<]*)<\|/ref\|><\|det\|>\[\[([^\]]+)\]\]<\|/det\|>',
    re.MULTILINE
)


def contains_grounding_tokens(text):
    if not text:
        return False
    return bool(GROUNDING_TOKEN_PATTERN.search(text))


def extract_grounding_ref_text(token_text):
    if not token_text:
        return ''
    refs = [match.group(1).strip() for match in GROUNDING_TOKEN_PATTERN.finditer(token_text)]
    refs = [entry for entry in refs if entry]
    return '\n'.join(refs).strip()


def strip_grounding_tokens(token_text):
    if not token_text:
        return token_text
    stripped = GROUNDING_TOKEN_PATTERN.sub('', token_text)
    return stripped.strip()


def normalize_mlx_output(prompt_type, generated_text):
    normalized_prompt_type = str(prompt_type or '').strip().lower()
    raw_token_text = generated_text if contains_grounding_tokens(generated_text) else None
    cleaned_text = (generated_text or '').strip()

    if raw_token_text:
        if normalized_prompt_type == 'ocr':
            extracted = extract_grounding_ref_text(raw_token_text)
            if extracted:
                cleaned_text = extracted
        elif normalized_prompt_type == 'document':
            cleaned_text = strip_grounding_tokens(raw_token_text)

    return cleaned_text, raw_token_text


def run_model_infer_mlx(**kwargs):
    global model, tokenizer
    max_new_tokens_cap = resolve_max_new_tokens_cap(kwargs.pop('max_new_tokens_cap', None))
    progress_callback = kwargs.pop('progress_callback', None)
    image_file = kwargs.get('image_file')
    if not image_file:
        raise ValueError('Missing image file path for MLX inference')

    raw_prompt = kwargs.get('prompt')
    user_prompt = to_mlx_prompt(raw_prompt)
    logger.info(
        "MLX inference settings: max_new_tokens_cap=%s base_size=%s image_size=%s crop_mode=%s",
        max_new_tokens_cap,
        kwargs.get('base_size'),
        kwargs.get('image_size'),
        kwargs.get('crop_mode')
    )
    try:
        formatted_prompt = mlx_apply_chat_template(
            tokenizer,
            config=model.config,
            prompt=user_prompt,
            num_images=1
        )
    except TypeError:
        formatted_prompt = mlx_apply_chat_template(tokenizer, prompt=user_prompt, num_images=1)
    with inference_lock:
        if mlx_stream_generate is not None and callable(progress_callback):
            streamed_chunks = []
            try:
                for partial in mlx_stream_generate(
                    model,
                    tokenizer,
                    formatted_prompt,
                    image=image_file,
                    max_tokens=max_new_tokens_cap,
                    temperature=0.0
                ):
                    text_chunk = getattr(partial, 'text', '')
                    if not text_chunk:
                        continue
                    streamed_chunks.append(text_chunk)
                    try:
                        progress_callback(''.join(streamed_chunks))
                    except Exception as callback_exc:
                        logger.debug(f"Ignoring MLX progress callback error: {callback_exc}")
                if streamed_chunks:
                    return ''.join(streamed_chunks)
                logger.warning("MLX stream generation produced no text chunks; retrying non-streaming generation")
            except Exception as stream_exc:
                logger.warning(f"MLX stream generation failed, retrying non-streaming generation: {stream_exc}")

        generated = mlx_generate(
            model=model,
            processor=tokenizer,
            image=image_file,
            prompt=formatted_prompt,
            max_tokens=max_new_tokens_cap,
            temperature=0.0,
            verbose=False
        )
    return extract_text_from_mlx_output(generated)


def is_cuda_oom_error(exc):
    text = str(exc or '').lower()
    return 'cuda out of memory' in text or ('out of memory' in text and 'cuda' in text)


def is_cuda_dtype_mismatch_error(exc):
    text = str(exc or '').lower()
    return (
        'masked_scatter_' in text and
        'same dtypes' in text and
        'half' in text and
        'float' in text
    )


# ---------------------------------------------------------------------------
# Pre-Ampere CUDA vision-encoder dtype fix
# ---------------------------------------------------------------------------
# The upstream DeepSeek-OCR-2 model hardcodes
#   torch.autocast("cuda", dtype=torch.bfloat16)
# inside its infer() method.  On pre-Ampere GPUs (Compute < 8.0) that lack
# native bfloat16 support, autocast silently falls back to float32 for many
# vision-encoder ops while the language model embeddings stay in float16.
# This causes a dtype mismatch at the masked_scatter_ call inside the model's
# forward(), which requires self and source to share a dtype.
#
# The fix: during the model's inner forward pass, temporarily replace
# torch.Tensor.masked_scatter_ with a wrapper that casts the source tensor
# to match the target dtype.  The wrapper is only active for the duration of
# the forward call and is protected by inference_lock, so it is thread-safe.
# This patch is applied on any OS when CUDA compute capability < 8.0.
# On Ampere+ GPUs (Compute >= 8.0) bfloat16 is natively supported and the
# upstream autocast works correctly, so no patch is needed.
# ---------------------------------------------------------------------------

_CUDA_DTYPE_FIX_APPLIED = False


def _apply_cuda_vision_dtype_fix(model_obj):
    """Monkey-patch model forward to fix vision-encoder dtype mismatch on pre-Ampere CUDA.

    Returns True if the patch was applied, False otherwise.
    """
    global _CUDA_DTYPE_FIX_APPLIED
    if _CUDA_DTYPE_FIX_APPLIED:
        return True

    inner = getattr(model_obj, 'model', None)
    if inner is None:
        logger.warning("Cannot apply CUDA vision dtype fix: model.model not found")
        return False

    original_forward = inner.forward  # bound method

    @functools.wraps(original_forward)
    def _patched_forward(*args, **kwargs):
        _orig_masked_scatter = torch.Tensor.masked_scatter_

        def _dtype_safe_masked_scatter(self_tensor, mask, source):
            if source.dtype != self_tensor.dtype:
                source = source.to(self_tensor.dtype)
            return _orig_masked_scatter(self_tensor, mask, source)

        torch.Tensor.masked_scatter_ = _dtype_safe_masked_scatter
        try:
            return original_forward(*args, **kwargs)
        finally:
            torch.Tensor.masked_scatter_ = _orig_masked_scatter

    inner.forward = _patched_forward
    _CUDA_DTYPE_FIX_APPLIED = True
    logger.info(
        "Applied pre-Ampere CUDA vision dtype fix "
        "(auto-cast masked_scatter_ source dtype during forward pass)"
    )
    return True


def get_cuda_memory_snapshot():
    if torch is None or device != 'cuda' or not torch.cuda.is_available():
        return None
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return {
            'free_bytes': int(free_bytes),
            'total_bytes': int(total_bytes),
            'allocated_bytes': int(torch.cuda.memory_allocated()),
            'reserved_bytes': int(torch.cuda.memory_reserved())
        }
    except Exception:
        return None


def format_cuda_memory_snapshot(snapshot):
    if not snapshot:
        return 'unavailable'
    mib = 1024 * 1024
    return (
        f"free={snapshot['free_bytes'] / mib:.1f} MiB, "
        f"total={snapshot['total_bytes'] / mib:.1f} MiB, "
        f"allocated={snapshot['allocated_bytes'] / mib:.1f} MiB, "
        f"reserved={snapshot['reserved_bytes'] / mib:.1f} MiB"
    )


def run_model_infer_torch(**kwargs):
    """Run torch-backed model inference with device arguments when supported by the model."""
    global model, tokenizer, device, dtype
    max_new_tokens_cap = resolve_max_new_tokens_cap(kwargs.pop('max_new_tokens_cap', None))
    host_os = platform.system()

    def invoke_infer(infer_kwargs, token_cap, prefer_explicit_device_dtype=False):
        logger.info(
            "Inference settings: device=%s eval_mode=%s max_new_tokens_cap=%s base_size=%s image_size=%s crop_mode=%s",
            device,
            bool(infer_kwargs.get('eval_mode', False)),
            token_cap,
            infer_kwargs.get('base_size'),
            infer_kwargs.get('image_size'),
            infer_kwargs.get('crop_mode')
        )
        if device == 'cuda' and not prefer_explicit_device_dtype:
            return model.infer(tokenizer, **infer_kwargs)
        try:
            return model.infer(tokenizer, device=torch.device(device), dtype=dtype, **infer_kwargs)
        except TypeError:
            logger.warning("Model infer() does not accept explicit device/dtype, retrying default call")
            return model.infer(tokenizer, **infer_kwargs)

    def invoke_with_guards(infer_kwargs, token_cap, prefer_explicit_device_dtype=False):
        with inference_lock:
            return invoke_infer_with_generation_cap(
                lambda: invoke_infer(infer_kwargs, token_cap, prefer_explicit_device_dtype),
                token_cap
            )

    try:
        return invoke_with_guards(kwargs, max_new_tokens_cap)
    except RuntimeError as exc:
        # Guard against a known MPS race where inference starts before the model
        # has finished transitioning to the MPS device.
        is_mps_placeholder_error = (
            device == 'mps' and
            'Placeholder storage has not been allocated on MPS device' in str(exc)
        )
        if is_mps_placeholder_error:
            logger.warning(
                "Encountered MPS placeholder storage error; waiting for model readiness and retrying once"
            )
            ready, load_error = wait_for_model_ready(timeout_seconds=240, poll_seconds=0.25)
            if not ready:
                logger.warning(f"Model did not become ready before retry: {load_error}")
                raise

            logger.info("Retrying inference after model readiness confirmed on MPS")
            return invoke_with_guards(kwargs, max_new_tokens_cap)

        is_linux_cuda_dtype_mismatch = (
            host_os == 'Linux' and
            device == 'cuda' and
            is_cuda_dtype_mismatch_error(exc)
        )
        if is_linux_cuda_dtype_mismatch:
            logger.warning("Linux CUDA dtype mismatch detected: %s", exc)
            logger.warning(
                "Retrying Linux CUDA inference with explicit device/dtype binding "
                "(dtype=%s) to align tensor types",
                dtype
            )
            clear_cuda_cache()
            try:
                return invoke_with_guards(
                    kwargs,
                    max_new_tokens_cap,
                    prefer_explicit_device_dtype=True
                )
            except RuntimeError as dtype_retry_exc:
                if is_cuda_dtype_mismatch_error(dtype_retry_exc):
                    raise RuntimeError(
                        "Linux CUDA dtype mismatch persisted after retry. "
                        "Set DEEPSEEK_OCR_ENABLE_TORCH_COMPILE=0, lower Base/Size, "
                        "or run with DEVICE=cpu."
                    ) from dtype_retry_exc
                raise

        # Linux-only CUDA OOM recovery path.
        # Keep Windows/macOS behavior unchanged.
        is_linux_cuda_oom = (
            host_os == 'Linux' and
            device == 'cuda' and
            is_cuda_oom_error(exc)
        )
        if not is_linux_cuda_oom:
            raise

        logger.warning("Linux CUDA OOM detected: %s", exc)
        logger.warning("CUDA memory snapshot before recovery: %s", format_cuda_memory_snapshot(get_cuda_memory_snapshot()))
        clear_cuda_cache()

        if not LINUX_CUDA_OOM_RETRY_ENABLED:
            raise RuntimeError(
                "Linux CUDA out of memory. Set smaller Base/Size, disable Crop, "
                "or set DEVICE=cpu for this run."
            ) from exc

        retry_kwargs = dict(kwargs)
        retry_kwargs['base_size'] = min(
            max(256, int(retry_kwargs.get('base_size', 1024))),
            LINUX_CUDA_OOM_RETRY_BASE_SIZE_MAX
        )
        retry_kwargs['image_size'] = min(
            max(256, int(retry_kwargs.get('image_size', 640))),
            LINUX_CUDA_OOM_RETRY_IMAGE_SIZE_MAX
        )
        retry_kwargs['crop_mode'] = False
        retry_cap = min(max_new_tokens_cap, LINUX_CUDA_OOM_RETRY_MAX_NEW_TOKENS)

        logger.warning(
            "Retrying Linux CUDA inference with conservative settings: "
            "base_size=%s image_size=%s crop_mode=%s max_new_tokens_cap=%s",
            retry_kwargs['base_size'],
            retry_kwargs['image_size'],
            retry_kwargs['crop_mode'],
            retry_cap
        )

        try:
            return invoke_with_guards(retry_kwargs, retry_cap)
        except RuntimeError as retry_exc:
            if is_cuda_oom_error(retry_exc):
                clear_cuda_cache()
                snapshot = format_cuda_memory_snapshot(get_cuda_memory_snapshot())
                raise RuntimeError(
                    "Linux CUDA out of memory after automatic retry. "
                    f"Current CUDA memory snapshot: {snapshot}. "
                    "Close other GPU processes, lower Base/Size, or run with DEVICE=cpu."
                ) from retry_exc
            raise


def run_model_infer(**kwargs):
    if RUNTIME_BACKEND == 'mlx':
        return run_model_infer_mlx(**kwargs)
    return run_model_infer_torch(**kwargs)


def load_model_background_mlx():
    """Load MLX model/processor for Apple Silicon."""
    global model, tokenizer, device, dtype

    update_progress('loading', 'init', 'Initializing MLX model loading...', 0)
    logger.info(f"Loading MLX DeepSeek OCR model from {MODEL_NAME}")
    logger.info(f"Model cache directory: {MODEL_CACHE_DIR}")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    device = 'mps'
    dtype = 'float16'
    log_selected_device(device)

    update_progress('loading', 'tokenizer', 'Loading MLX processor...', 15)
    update_progress('loading', 'model', 'Loading MLX model (first run downloads files)...', 35)
    model, tokenizer = mlx_load(MODEL_NAME)

    update_progress('loading', 'optimize', 'Finalizing MLX runtime...', 90)
    logger.info("MLX model loaded successfully")
    update_progress('loaded', 'complete', 'Model ready!', 100)


def load_model_background():
    """Background thread function to load the model"""
    global model, tokenizer, device, dtype

    try:
        if RUNTIME_BACKEND == 'mlx':
            if not is_mlx_available():
                raise RuntimeError(f"MLX backend unavailable: {MLX_IMPORT_ERROR or 'mlx-vlm import failed'}")
            load_model_background_mlx()
            return

        if not is_torch_available():
            raise RuntimeError(f"Torch backend unavailable: {TORCH_IMPORT_ERROR or 'torch import failed'}")
        if AutoModel is None or AutoTokenizer is None:
            raise RuntimeError(
                f"Transformers unavailable: {TRANSFORMERS_IMPORT_ERROR or 'transformers import failed'}"
            )

        update_progress('loading', 'init', 'Initializing model loading...', 0)
        logger.info(f"Loading DeepSeek OCR model from {MODEL_NAME}...")
        logger.info(f"Model will be cached in: {MODEL_CACHE_DIR}")

        # Create cache directory if it doesn't exist
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        # Select and report runtime device
        device = get_preferred_device()
        log_selected_device(device)

        # Load tokenizer (10% progress)
        update_progress('loading', 'tokenizer', 'Loading tokenizer...', 10)
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR
        )
        update_progress('loading', 'tokenizer', 'Tokenizer loaded', 20)

        # Check if model is already cached
        initial_cache_size = get_cache_dir_size(MODEL_CACHE_DIR)
        is_cached = initial_cache_size > 100 * 1024 * 1024  # More than 100 MB suggests model is cached

        if is_cached:
            # Model is cached, just loading from disk
            update_progress('loading', 'model', 'Loading model from cache...', 25)
            logger.info("Loading model from cache...")
        else:
            # Model needs to be downloaded
            update_progress('loading', 'model', 'Downloading model files (this will take several minutes)...', 25)
            logger.info("Downloading model (this may take a while on first run)...")

        # Start a thread to monitor download progress (only if downloading)
        download_monitor_active = [True]  # Use list for mutable access in nested function
        def monitor_download():
            last_size = initial_cache_size
            stall_count = 0
            progress = 25

            while download_monitor_active[0] and progress < 75:
                time.sleep(2)  # Check every 2 seconds
                current_size = get_cache_dir_size(MODEL_CACHE_DIR)

                if current_size > last_size:
                    # Download is progressing
                    stall_count = 0
                    # Increment progress (max 75%)
                    progress = min(progress + 2, 75)
                    size_mb = current_size / (1024 * 1024)
                    update_progress('loading', 'model', f'Downloading model files... ({size_mb:.1f} MB downloaded)', progress)
                    last_size = current_size
                else:
                    # No change in size
                    stall_count += 1
                    if stall_count < 5:  # Still show activity for first 10 seconds
                        if is_cached:
                            update_progress('loading', 'model', 'Loading model from cache...', progress)
                        else:
                            update_progress('loading', 'model', 'Downloading model files...', progress)

        monitor_thread = Thread(target=monitor_download)
        monitor_thread.daemon = True
        monitor_thread.start()

        # Try to use flash attention if available, otherwise fallback
        try:
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                use_safetensors=True,
                cache_dir=MODEL_CACHE_DIR
            )
            logger.info("Using flash attention 2")
        except Exception as e:
            logger.warning(f"Flash attention not available: {e}, using default attention")
            model = AutoModel.from_pretrained(
                MODEL_NAME,
                trust_remote_code=True,
                use_safetensors=True,
                cache_dir=MODEL_CACHE_DIR
            )

        # Stop download monitor and wait for it to finish
        download_monitor_active[0] = False
        monitor_thread.join(timeout=5)  # Wait up to 5 seconds for thread to finish

        # Set to eval mode (80% progress)
        update_progress('loading', 'gpu', 'Moving model to GPU...', 80)
        model = model.eval()

        # Move model to selected device (85% progress)
        update_progress('loading', 'gpu', 'Optimizing model on GPU...', 85)
        if device == 'cuda':
            compute_cap = torch.cuda.get_device_capability()
            if compute_cap[0] >= 8:  # Ampere or newer (RTX 30/40/50 series)
                dtype = torch.bfloat16
            else:  # Pascal/Turing (GTX 10/16 series, RTX 20 series)
                dtype = torch.float16
            model = model.cuda().to(dtype)
            logger.info(f"Model loaded on CUDA with dtype={dtype} (Compute {compute_cap[0]}.{compute_cap[1]})")

            # The upstream model hardcodes torch.autocast("cuda",
            # dtype=torch.bfloat16) which causes a Half/Float dtype mismatch
            # on pre-Ampere GPUs.  Patch the forward pass to auto-cast.
            if compute_cap[0] < 8:
                _apply_cuda_vision_dtype_fix(model)
        elif device == 'mps':
            dtype = torch.float16
            model = model.to(torch.device('mps')).to(dtype)
            logger.info("Model loaded on MPS with float16")
        else:
            dtype = torch.float32
            model = model.to(torch.device('cpu')).to(dtype)
            logger.info("Model loaded on CPU with float32")

        # Apply torch.compile on CUDA when available (95% progress).
        update_progress('loading', 'optimize', 'Optimizing model runtime...', 95)
        try:
            if hasattr(torch, 'compile') and device == 'cuda' and TORCH_COMPILE_ENABLED:
                logger.info("Applying torch.compile for faster inference...")
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled successfully for CUDA")
            elif device == 'cuda' and not TORCH_COMPILE_ENABLED:
                logger.info(
                    "Skipping torch.compile on CUDA "
                    "(set DEEPSEEK_OCR_ENABLE_TORCH_COMPILE=1 to enable)"
                )
            else:
                logger.info("torch.compile unavailable or unsupported for current device, skipping compilation")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}, using uncompiled model")

        # Warmup inference to initialize graphs.
        # MPS warmup can stall on some systems, so it's disabled by default unless explicitly enabled.
        warmup_override = os.environ.get('DEEPSEEK_OCR_ENABLE_WARMUP')
        if warmup_override is None:
            should_warmup = (device == 'cuda')
        else:
            should_warmup = str(warmup_override).strip().lower() in ('1', 'true', 'yes', 'on')

        if should_warmup:
            update_progress('loading', 'warmup', 'Running warmup inference...', 98)
            logger.info("Running warmup inference...")
            try:
                # Create a small dummy image for warmup
                import numpy as np
                from PIL import Image
                dummy_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    dummy_img.save(tmp.name)
                with tempfile.TemporaryDirectory() as warmup_output:
                    run_model_infer(
                        prompt='<image>\nFree OCR. ',
                        image_file=tmp.name,
                        output_path=warmup_output,
                        base_size=512,
                        image_size=512,
                        crop_mode=False,
                        save_results=False,
                        test_compress=False
                    )
                    logger.info("Warmup inference complete")
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
            except Exception as e:
                logger.warning(f"Warmup inference failed: {e}")
        else:
            logger.info(f"Skipping warmup inference on {device}. Set DEEPSEEK_OCR_ENABLE_WARMUP=1 to enable.")

        logger.info("Model loaded successfully!")
        update_progress('loaded', 'complete', 'Model ready!', 100)

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        update_progress('error', 'failed', str(e), 0)
        import traceback
        traceback.print_exc()

def load_model():
    """Load the DeepSeek OCR model and tokenizer

    Model size configurations:
    - Tiny: base_size=512, image_size=512, crop_mode=False
    - Small: base_size=640, image_size=640, crop_mode=False
    - Base: base_size=1024, image_size=1024, crop_mode=False
    - Large: base_size=1280, image_size=1280, crop_mode=False
    - Gundam (recommended): base_size=1024, image_size=640, crop_mode=True
    """
    global model, tokenizer, loading_thread

    if model is not None and tokenizer is not None:
        logger.info("Model already loaded")
        update_progress('loaded', 'complete', 'Model already loaded', 100)
        return True

    # Check if already loading
    if loading_thread is not None and loading_thread.is_alive():
        logger.info("Model loading already in progress")
        return True

    # Start loading in background thread
    loading_thread = Thread(target=load_model_background)
    loading_thread.daemon = True
    loading_thread.start()

    return True


def wait_for_model_ready(timeout_seconds=600, poll_seconds=0.5):
    """Wait until the model is ready or loading fails."""
    started = time.time()
    while time.time() - started < timeout_seconds:
        with progress_lock:
            status = progress_data.get('status')
            message = progress_data.get('message', '')

        if is_model_ready():
            return True, ''
        if status == 'error':
            return False, message or 'Model loading failed'
        time.sleep(poll_seconds)

    return False, f'Model did not finish loading within {timeout_seconds} seconds'

def clear_cuda_cache():
    """Clear CUDA cache to free memory between processing"""
    if torch is not None and device == 'cuda' and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as exc:
            logger.debug(f"Unable to clear CUDA cache cleanly: {exc}")


def run_model_infer_with_heartbeat(
    infer_kwargs,
    status,
    stage,
    message_builder,
    progress_builder=None,
    heartbeat_seconds=2.0
):
    """Run inference in a worker thread and emit heartbeat progress while waiting."""
    result_holder = {'error': None, 'result': None}
    finished = Event()
    started_at = time.time()

    def worker():
        try:
            result_holder['result'] = run_model_infer(**infer_kwargs)
        except Exception as exc:
            result_holder['error'] = exc
        finally:
            finished.set()

    thread = Thread(target=worker, daemon=True)
    thread.start()

    while not finished.wait(heartbeat_seconds):
        elapsed = max(1, int(time.time() - started_at))
        message = message_builder(elapsed)
        progress_percent = progress_builder(elapsed) if callable(progress_builder) else None
        chars_generated = 0
        raw_token_stream = ''
        current_page_image = ''
        if progress_percent is None:
            with progress_lock:
                progress_percent = progress_data.get('progress_percent', 0)
                chars_generated = progress_data.get('chars_generated', 0)
                raw_token_stream = progress_data.get('raw_token_stream', '')
                current_page_image = progress_data.get('current_page_image', '')
        else:
            with progress_lock:
                chars_generated = progress_data.get('chars_generated', 0)
                raw_token_stream = progress_data.get('raw_token_stream', '')
                current_page_image = progress_data.get('current_page_image', '')
        update_progress(
            status,
            stage,
            message,
            int(progress_percent),
            chars_generated,
            raw_token_stream,
            current_page_image=current_page_image
        )

    if result_holder['error'] is not None:
        raise result_holder['error']
    return result_holder['result']

def create_queue_output_folder():
    """Create a timestamped folder for queue processing results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    queue_folder = os.path.join(OUTPUT_DIR, f"queue_{timestamp}")
    os.makedirs(queue_folder, exist_ok=True)
    return queue_folder

def get_prompt_for_type(prompt_type):
    """Get prompt text for given type"""
    prompts = {
        'document': '<image>\n<|grounding|>Convert the document to markdown. ',
        'ocr': '<image>\n<|grounding|>OCR this image. ',
        'free': '<image>\nFree OCR. ',
        'figure': '<image>\nParse the figure. ',
        'describe': '<image>\nDescribe this image in detail. '
    }
    return prompts.get(prompt_type, prompts['document'])

def get_result_filename(prompt_type):
    """Get expected result filename for prompt type"""
    if prompt_type == 'document':
        return 'result.mmd'
    return 'result.txt'


SUPPORTED_INPUT_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.pdf'}


def get_env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


PREFERRED_DEVICE_HINT = get_preferred_device()
PDF_RENDER_SCALE = max(0.4, float(os.environ.get('DEEPSEEK_OCR_PDF_RENDER_SCALE', '1.0')))
PDF_MIN_RENDER_SIDE = max(
    600,
    int(os.environ.get('DEEPSEEK_OCR_PDF_MIN_SIDE', '800' if PREFERRED_DEVICE_HINT == 'mps' else '1000'))
)
PDF_MAX_RENDER_SIDE = max(
    PDF_MIN_RENDER_SIDE,
    int(os.environ.get('DEEPSEEK_OCR_PDF_MAX_SIDE', '1200' if PREFERRED_DEVICE_HINT == 'mps' else '1400'))
)
PDF_TARGET_SIDE_MULTIPLIER = max(
    1.0,
    float(os.environ.get('DEEPSEEK_OCR_PDF_TARGET_SIDE_MULTIPLIER', '1.1' if PREFERRED_DEVICE_HINT == 'mps' else '1.25'))
)
PDF_FAST_MODE = get_env_flag('DEEPSEEK_OCR_PDF_FAST_MODE', True)
PDF_FAST_BASE_SIZE_MAX = max(
    512,
    int(os.environ.get('DEEPSEEK_OCR_PDF_FAST_BASE_MAX', '640' if PREFERRED_DEVICE_HINT == 'mps' else '768'))
)
PDF_FAST_IMAGE_SIZE_MAX = max(
    320,
    int(os.environ.get('DEEPSEEK_OCR_PDF_FAST_IMAGE_MAX', '384' if PREFERRED_DEVICE_HINT == 'mps' else '448'))
)
PDF_FAST_DISABLE_CROP = get_env_flag('DEEPSEEK_OCR_PDF_FAST_DISABLE_CROP', True)
PDF_EVAL_MODE = get_env_flag('DEEPSEEK_OCR_PDF_EVAL_MODE', True)
PDF_MAX_NEW_TOKENS = max(
    128,
    int(os.environ.get('DEEPSEEK_OCR_PDF_MAX_NEW_TOKENS', '384' if PREFERRED_DEVICE_HINT == 'mps' else '1024'))
)

DEFAULT_MAX_NEW_TOKENS_MPS = max(256, int(os.environ.get('DEEPSEEK_OCR_MAX_NEW_TOKENS_MPS', '768')))
DEFAULT_MAX_NEW_TOKENS_OTHER = max(256, int(os.environ.get('DEEPSEEK_OCR_MAX_NEW_TOKENS_OTHER', '2048')))
LINUX_CUDA_OOM_RETRY_ENABLED = get_env_flag('DEEPSEEK_OCR_LINUX_CUDA_OOM_RETRY', True)
LINUX_CUDA_OOM_RETRY_BASE_SIZE_MAX = max(
    384,
    int(os.environ.get('DEEPSEEK_OCR_LINUX_CUDA_OOM_RETRY_BASE_MAX', '640'))
)
LINUX_CUDA_OOM_RETRY_IMAGE_SIZE_MAX = max(
    320,
    int(os.environ.get('DEEPSEEK_OCR_LINUX_CUDA_OOM_RETRY_IMAGE_MAX', '512'))
)
LINUX_CUDA_OOM_RETRY_MAX_NEW_TOKENS = max(
    128,
    int(os.environ.get('DEEPSEEK_OCR_LINUX_CUDA_OOM_RETRY_MAX_NEW_TOKENS', '768'))
)


def get_input_suffix(filename):
    suffix = Path(filename or '').suffix.lower()
    if suffix in SUPPORTED_INPUT_EXTENSIONS:
        return suffix
    return '.jpg'


def is_pdf_input(input_path):
    return Path(input_path).suffix.lower() == '.pdf'


def build_pdf_inference_profile(base_size, image_size, crop_mode):
    tuned_base = max(256, int(base_size))
    tuned_image = max(256, int(image_size))
    tuned_crop = bool(crop_mode)
    notes = []

    if PDF_FAST_MODE:
        if tuned_base > PDF_FAST_BASE_SIZE_MAX:
            notes.append(f'base_size {tuned_base} -> {PDF_FAST_BASE_SIZE_MAX}')
            tuned_base = PDF_FAST_BASE_SIZE_MAX
        if tuned_image > PDF_FAST_IMAGE_SIZE_MAX:
            notes.append(f'image_size {tuned_image} -> {PDF_FAST_IMAGE_SIZE_MAX}')
            tuned_image = PDF_FAST_IMAGE_SIZE_MAX
        if tuned_image > tuned_base:
            notes.append(f'image_size {tuned_image} -> {tuned_base}')
            tuned_image = tuned_base
        if tuned_crop and PDF_FAST_DISABLE_CROP:
            notes.append('crop_mode true -> false')
            tuned_crop = False

    target_max_side = max(
        PDF_MIN_RENDER_SIDE,
        min(
            PDF_MAX_RENDER_SIDE,
            int(max(tuned_base, tuned_image) * PDF_TARGET_SIDE_MULTIPLIER)
        )
    )

    return {
        'base_size': tuned_base,
        'image_size': tuned_image,
        'crop_mode': tuned_crop,
        'target_max_side': target_max_side,
        'adjusted': len(notes) > 0,
        'notes': notes
    }


def clear_output_artifacts(output_dir):
    for filename in ('result.txt', 'result.mmd', 'result_with_boxes.jpg'):
        target = os.path.join(output_dir, filename)
        if os.path.exists(target):
            os.remove(target)
    images_dir = os.path.join(output_dir, 'images')
    if os.path.isdir(images_dir):
        for entry in os.scandir(images_dir):
            if entry.is_file():
                os.remove(entry.path)


def read_result_text(output_dir, expected_output_file):
    result_path = os.path.join(output_dir, expected_output_file)
    if os.path.exists(result_path):
        with open(result_path, 'r', encoding='utf-8') as f:
            return f.read()

    for filename in os.listdir(output_dir):
        if filename.endswith(('.txt', '.mmd', '.md')):
            with open(os.path.join(output_dir, filename), 'r', encoding='utf-8') as f:
                return f.read()
    return None


def strip_markdown_image_refs(markdown_text):
    if not markdown_text:
        return markdown_text
    return re.sub(r'!\[[^\]]*\]\(images\/[^)]+\)', '', markdown_text)


def combine_pdf_results(page_results, prompt_type, page_labels=None):
    if not page_results:
        return ''

    if page_labels is None:
        page_labels = list(range(1, len(page_results) + 1))

    if prompt_type == 'document':
        chunks = []
        for idx, text in enumerate(page_results):
            page_num = page_labels[idx] if idx < len(page_labels) else idx + 1
            cleaned = (text or '').strip() or '(No text extracted)'
            chunks.append(f'## Page {page_num}\n\n{cleaned}')
        return '\n\n'.join(chunks).strip()

    chunks = []
    for idx, text in enumerate(page_results):
        page_num = page_labels[idx] if idx < len(page_labels) else idx + 1
        cleaned = (text or '').strip()
        chunks.append(f'--- Page {page_num} ---\n{cleaned}')
    return '\n\n'.join(chunks).strip()


def parse_pdf_page_range(page_range_text, page_count):
    """Parse page-range syntax like '1-3,5,7-9' into sorted unique page numbers."""
    if page_count <= 0:
        return []

    text = (page_range_text or '').strip()
    if not text:
        return list(range(1, page_count + 1))

    selected = set()
    for token in text.split(','):
        part = token.strip()
        if not part:
            continue

        if '-' in part:
            bounds = [value.strip() for value in part.split('-', 1)]
            if len(bounds) != 2:
                raise ValueError(f'Invalid page range token: "{part}"')
            start_text, end_text = bounds
            start = int(start_text) if start_text else 1
            end = int(end_text) if end_text else page_count
            if start > end:
                raise ValueError(f'Invalid page range token: "{part}"')
            for page_num in range(start, end + 1):
                if page_num < 1 or page_num > page_count:
                    raise ValueError(f'Page {page_num} is out of range (1-{page_count})')
                selected.add(page_num)
        else:
            page_num = int(part)
            if page_num < 1 or page_num > page_count:
                raise ValueError(f'Page {page_num} is out of range (1-{page_count})')
            selected.add(page_num)

    if not selected:
        raise ValueError('No pages selected from page range.')
    return sorted(selected)


def validate_page_range_syntax(page_range_text):
    """Validate page range syntax before enqueueing PDF work."""
    text = (page_range_text or '').strip()
    if not text:
        return

    for token in text.split(','):
        part = token.strip()
        if not part:
            continue
        if '-' in part:
            start_text, end_text = [value.strip() for value in part.split('-', 1)]
            if not start_text and not end_text:
                raise ValueError(f'Invalid page range token: "{part}"')
            if start_text and (not start_text.isdigit() or int(start_text) < 1):
                raise ValueError(f'Invalid page number: "{start_text}"')
            if end_text and (not end_text.isdigit() or int(end_text) < 1):
                raise ValueError(f'Invalid page number: "{end_text}"')
            if start_text and end_text and int(start_text) > int(end_text):
                raise ValueError(f'Invalid page range token: "{part}"')
        else:
            if not part.isdigit() or int(part) < 1:
                raise ValueError(f'Invalid page number: "{part}"')


def render_pdf_to_images(pdf_path, output_dir, page_range_text=None, target_max_side=None):
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError('PDF support requires pypdfium2. Please reinstall dependencies.') from exc

    os.makedirs(output_dir, exist_ok=True)
    max_side = max(PDF_MIN_RENDER_SIDE, int(target_max_side or PDF_MAX_RENDER_SIDE))
    logger.info(
        f"Rendering PDF pages with base_scale={PDF_RENDER_SCALE:.2f}, max_side={max_side}"
    )
    pdf = pdfium.PdfDocument(pdf_path)
    page_paths = []
    selected_pages = []
    try:
        selected_pages = parse_pdf_page_range(page_range_text, len(pdf))
        for page_num in selected_pages:
            page_idx = page_num - 1
            page = pdf[page_idx]
            effective_scale = PDF_RENDER_SCALE
            page_width = 0
            page_height = 0
            try:
                page_width, page_height = page.get_size()
            except Exception:
                page_width, page_height = (0, 0)

            page_longest = max(page_width, page_height)
            if page_longest > 0:
                projected_longest = page_longest * PDF_RENDER_SCALE
                if projected_longest > max_side:
                    effective_scale = max(0.5, max_side / float(page_longest))
            if abs(effective_scale - PDF_RENDER_SCALE) > 0.01:
                logger.info(
                    f"Adjusted render scale for PDF page {page_num}: "
                    f"{PDF_RENDER_SCALE:.2f} -> {effective_scale:.2f}"
                )

            pil_image = page.render(scale=effective_scale).to_pil()
            original_size = pil_image.size
            longest_side = max(original_size) if original_size else 0
            if longest_side > max_side:
                scale = max_side / float(longest_side)
                target_size = (
                    max(1, int(round(original_size[0] * scale))),
                    max(1, int(round(original_size[1] * scale)))
                )
                try:
                    from PIL import Image as PILImage
                    try:
                        resample_filter = PILImage.Resampling.LANCZOS
                    except AttributeError:
                        resample_filter = PILImage.LANCZOS
                    pil_image = pil_image.resize(target_size, resample=resample_filter)
                except Exception:
                    pil_image = pil_image.resize(target_size)
                logger.info(
                    f"Downscaled PDF page {page_num} from {original_size[0]}x{original_size[1]} "
                    f"to {target_size[0]}x{target_size[1]}"
                )
            page_path = os.path.join(output_dir, f'page_{page_num:04d}.jpg')
            pil_image.save(page_path, format='JPEG', quality=95)
            page_paths.append(page_path)
    finally:
        try:
            pdf.close()
        except Exception:
            pass

    if not page_paths:
        raise RuntimeError('The PDF does not contain any renderable pages.')
    return page_paths, selected_pages


def extract_raw_token_section(accumulated_text):
    if not accumulated_text:
        return None
    parts = accumulated_text.split('=' * 20)
    if len(parts) < 3:
        return None
    return parts[2].strip().lstrip('=').strip()


def estimate_stream_progress(chars_generated):
    """Estimate stream progress with a smooth curve that avoids long stalls."""
    if chars_generated <= 0:
        return 8
    return min(92, 8 + int(84 * (1 - math.exp(-chars_generated / 900.0))))


def estimate_pdf_page_progress(page_start_percent, total_pages, elapsed_seconds):
    page_span = max(1, int(95 / max(total_pages, 1)))
    headroom = max(1, page_span - 2)
    within_page = int(headroom * (1 - math.exp(-max(0, elapsed_seconds) / 45.0)))
    return min(94, page_start_percent + within_page)


def format_queue_progress_detail(current_page=None, total_pages=None, paused=False):
    if current_page and total_pages:
        base = f'Page {current_page}/{total_pages}'
    else:
        base = 'Page 1/1'
    return f'{base} (Paused)' if paused else base


def queue_should_stop():
    with queue_lock:
        return queue_cancel_requested


def queue_wait_if_paused(item=None):
    """Pause processing while queue is paused; return False if cancel requested."""
    while True:
        with queue_lock:
            paused = queue_paused
            cancel_requested = queue_cancel_requested

            if item is not None and item.get('status') == 'processing':
                detail = item.get('progress_detail') or 'Page 1/1'
                if paused and '(Paused)' not in detail:
                    item['progress_detail'] = f'{detail} (Paused)'
                if (not paused) and '(Paused)' in detail:
                    item['progress_detail'] = detail.replace(' (Paused)', '')

        if cancel_requested:
            return False
        if not paused:
            return True
        update_progress(
            'processing',
            'queue',
            'Queue paused - waiting to resume...',
            progress_data.get('progress_percent', 0),
            progress_data.get('chars_generated', 0),
            progress_data.get('raw_token_stream', '')
        )
        time.sleep(0.25)


def get_recent_backend_logs(limit=250):
    if limit <= 0:
        return []
    return list(BACKEND_LOG_BUFFER)[-limit:]


def get_queue_summary():
    with queue_lock:
        current_item = next((item for item in processing_queue if item['status'] == 'processing'), None)
        summary = {
            'total': len(processing_queue),
            'pending': sum(1 for item in processing_queue if item['status'] == 'pending'),
            'processing': sum(1 for item in processing_queue if item['status'] == 'processing'),
            'completed': sum(1 for item in processing_queue if item['status'] == 'completed'),
            'failed': sum(1 for item in processing_queue if item['status'] == 'failed'),
            'paused': queue_paused,
            'cancel_requested': queue_cancel_requested,
            'is_processing': queue_processing_active,
            'can_retry_failed': any(item['status'] == 'failed' for item in processing_queue),
            'items': [{
                'id': item['id'],
                'filename': item['filename'],
                'status': item['status'],
                'progress': item['progress'],
                'progress_detail': item.get('progress_detail'),
                'error': item['error']
            } for item in processing_queue],
            'current_file': {
                'id': current_item['id'],
                'filename': current_item['filename'],
                'image_path': current_item.get('current_image_path'),
                'progress': current_item['progress'],
                'progress_detail': current_item.get('progress_detail')
            } if current_item else None
        }
    return summary


def collect_runtime_diagnostics():
    with progress_lock:
        progress_snapshot = progress_data.copy()

    preferred_device = get_preferred_device()
    gpu_target_hint = os.environ.get('DEEPSEEK_OCR_GPU_TARGET')
    model_cache_bytes = get_cache_dir_size(MODEL_CACHE_DIR)
    output_cache_bytes = get_cache_dir_size(OUTPUT_DIR)
    total_cache_bytes = get_cache_dir_size(CACHE_DIR)
    disk_usage = shutil.disk_usage(CACHE_DIR if os.path.exists(CACHE_DIR) else os.path.expanduser('~'))
    cuda_memory = get_cuda_memory_snapshot()

    return {
        'timestamp': datetime.now().astimezone().isoformat(),
        'runtime': {
            'python_version': sys.version,
            'platform': platform.platform(),
            'pid': os.getpid(),
            'backend': RUNTIME_BACKEND,
            'torch_compile_enabled': TORCH_COMPILE_ENABLED,
            'torch_available': is_torch_available(),
            'mlx_available': is_mlx_available(),
            'torch_import_error': TORCH_IMPORT_ERROR,
            'mlx_import_error': MLX_IMPORT_ERROR
        },
        'model': {
            'model_name': MODEL_NAME,
            'loaded': is_model_ready(),
            'initialized': model is not None and tokenizer is not None,
            'preferred_device': preferred_device,
            'device': device,
            'dtype': str(dtype),
            'runtime_backend': RUNTIME_BACKEND,
            'gpu_target_hint': gpu_target_hint,
            'gpu_target_device_hint': map_gpu_target_to_device(gpu_target_hint) if gpu_target_hint else None,
            'cuda_memory': cuda_memory
        },
        'cache': {
            'cache_dir': CACHE_DIR,
            'model_cache_dir': MODEL_CACHE_DIR,
            'output_dir': OUTPUT_DIR,
            'model_cache_bytes': model_cache_bytes,
            'output_cache_bytes': output_cache_bytes,
            'total_cache_bytes': total_cache_bytes,
            'disk_total_bytes': disk_usage.total,
            'disk_used_bytes': disk_usage.used,
            'disk_free_bytes': disk_usage.free
        },
        'queue': get_queue_summary(),
        'progress': progress_snapshot,
        'logs_tail': get_recent_backend_logs()
    }

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    gpu_target_hint = os.environ.get('DEEPSEEK_OCR_GPU_TARGET')
    preferred_device = get_preferred_device()
    model_ready = is_model_ready()
    cuda_memory = get_cuda_memory_snapshot()
    return jsonify({
        'status': 'ok',
        'runtime_backend': RUNTIME_BACKEND,
        'torch_compile_enabled': TORCH_COMPILE_ENABLED,
        'torch_available': is_torch_available(),
        'mlx_available': is_mlx_available(),
        'model_loaded': model_ready,
        'model_initialized': model is not None and tokenizer is not None,
        'loading_in_progress': is_loading_in_progress(),
        'gpu_available': device in ('cuda', 'mps'),
        'preferred_gpu_available': preferred_device in ('cuda', 'mps'),
        'gpu_target_hint': gpu_target_hint,
        'gpu_target_device_hint': map_gpu_target_to_device(gpu_target_hint) if gpu_target_hint else None,
        'preferred_device': preferred_device,
        'device_state': device,
        'cuda_memory': cuda_memory
    })

@app.route('/progress', methods=['GET'])
def get_progress():
    """Get current model loading progress"""
    with progress_lock:
        current_data = progress_data.copy()
    return jsonify(current_data)

@app.route('/load_model', methods=['POST'])
def load_model_endpoint():
    """Endpoint to trigger model loading"""
    success = load_model()
    if success:
        return jsonify({'status': 'success', 'message': 'Model loaded successfully'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to load model'}), 500

@app.route('/ocr', methods=['POST'])
def perform_ocr():
    """Perform OCR on an uploaded image or PDF"""
    global model, tokenizer, device, dtype

    temp_input_path = None
    try:
        # Check if model is ready (fully loaded and not mid-load)
        if not is_model_ready():
            logger.info("Model not ready, loading/waiting now...")
            if not load_model():
                return jsonify({'status': 'error', 'message': 'Failed to load model'}), 500
            ready, load_error = wait_for_model_ready(timeout_seconds=600, poll_seconds=0.5)
            if not ready:
                return jsonify({
                    'status': 'error',
                    'message': load_error
                }), 500

        # Get file from request
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'}), 400

        input_file = request.files['image']
        input_suffix = get_input_suffix(input_file.filename)

        # Get optional parameters
        prompt_type = request.form.get('prompt_type', 'document')
        base_size = int(request.form.get('base_size', 1024))
        image_size = int(request.form.get('image_size', 640))
        crop_mode = request.form.get('crop_mode', 'true').lower() == 'true'
        pdf_page_range = request.form.get('pdf_page_range', '').strip() or None
        if pdf_page_range:
            validate_page_range_syntax(pdf_page_range)
        prompt = get_prompt_for_type(prompt_type)
        expected_output_file = get_result_filename(prompt_type)

        logger.info(f"Processing OCR request with prompt type: {prompt_type}")

        # Save input temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix) as tmp_file:
            input_file.save(tmp_file.name)
            temp_input_path = tmp_file.name

        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        logger.info("Running OCR inference...")
        logger.info(f"Saving results to: {OUTPUT_DIR}")

        # Reset progress
        update_progress('processing', 'ocr', 'Starting OCR...', 0, 0)

        if is_pdf_input(temp_input_path):
            logger.info("PDF input detected - rendering pages before OCR")
            pdf_profile = build_pdf_inference_profile(base_size, image_size, crop_mode)
            if pdf_profile['adjusted']:
                logger.info(
                    f"Applied PDF fast profile ({'; '.join(pdf_profile['notes'])}), "
                    f"render max side={pdf_profile['target_max_side']}"
                )
                update_progress(
                    'processing',
                    'ocr',
                    'Optimizing PDF settings for faster OCR...',
                    1,
                    0
                )
            with tempfile.TemporaryDirectory(prefix='ocr_pdf_pages_') as page_dir:
                page_images, selected_pages = render_pdf_to_images(
                    temp_input_path,
                    page_dir,
                    page_range_text=pdf_page_range,
                    target_max_side=pdf_profile['target_max_side']
                )
                total_pages = len(page_images)
                page_results = []

                for page_idx, page_image in enumerate(page_images, start=1):
                    page_label = selected_pages[page_idx - 1] if page_idx - 1 < len(selected_pages) else page_idx
                    logger.info(f"Starting PDF page {page_label} ({page_idx}/{total_pages})")
                    page_started_at = time.time()
                    update_progress(
                        'processing',
                        'ocr',
                        f'Processing PDF page {page_label} ({page_idx}/{total_pages})...',
                        int(((page_idx - 1) / total_pages) * 95),
                        0,
                        current_page_image=page_image
                    )
                    clear_output_artifacts(OUTPUT_DIR)
                    page_progress = int(((page_idx - 1) / total_pages) * 95)
                    page_infer_result = run_model_infer_with_heartbeat(
                        infer_kwargs={
                            'prompt': prompt,
                            'image_file': page_image,
                            'output_path': OUTPUT_DIR,
                            'base_size': pdf_profile['base_size'],
                            'image_size': pdf_profile['image_size'],
                            'crop_mode': pdf_profile['crop_mode'],
                            'save_results': not PDF_EVAL_MODE,
                            'test_compress': False,
                            'eval_mode': PDF_EVAL_MODE,
                            'max_new_tokens_cap': PDF_MAX_NEW_TOKENS
                        },
                        status='processing',
                        stage='ocr',
                        message_builder=lambda elapsed, page_label=page_label, page_idx=page_idx, total_pages=total_pages:
                            f'Processing PDF page {page_label} ({page_idx}/{total_pages})... {elapsed}s elapsed',
                        progress_builder=lambda elapsed, page_progress=page_progress, total_pages=total_pages:
                            estimate_pdf_page_progress(page_progress, total_pages, elapsed),
                        heartbeat_seconds=2.0
                    )
                    page_text = page_infer_result if isinstance(page_infer_result, str) else ''
                    if RUNTIME_BACKEND == 'mlx' and page_text:
                        page_text, _ = normalize_mlx_output(prompt_type, page_text)
                    if not page_text:
                        page_text = read_result_text(OUTPUT_DIR, expected_output_file) or ''
                    if prompt_type == 'document':
                        page_text = strip_markdown_image_refs(page_text)
                    page_results.append(page_text)
                    logger.info(
                        f"Completed PDF page {page_label} ({page_idx}/{total_pages}) in "
                        f"{time.time() - page_started_at:.1f}s"
                    )

                result_text = combine_pdf_results(page_results, prompt_type, page_labels=selected_pages)

            return jsonify({
                'status': 'success',
                'result': result_text or 'OCR completed but no text was generated',
                'boxes_image_path': None,
                'prompt_type': prompt_type,
                'raw_tokens': None,
                'is_pdf': True,
                'page_count': len(page_results),
                'page_labels': selected_pages,
                'applied_settings': {
                    'base_size': pdf_profile['base_size'],
                    'image_size': pdf_profile['image_size'],
                    'crop_mode': pdf_profile['crop_mode'],
                    'render_max_side': pdf_profile['target_max_side']
                }
            })

        # Image flow
        clear_output_artifacts(OUTPUT_DIR)
        infer_result = None
        char_stream = None

        if RUNTIME_BACKEND == 'mlx':
            def mlx_live_progress(raw_token_text):
                chars_generated = len(raw_token_text or '')
                if chars_generated <= 0:
                    return
                update_progress(
                    'processing',
                    'ocr',
                    'Generating OCR...',
                    estimate_stream_progress(chars_generated),
                    chars_generated,
                    raw_token_text
                )

            infer_result = run_model_infer_with_heartbeat(
                infer_kwargs={
                    'prompt': prompt,
                    'image_file': temp_input_path,
                    'output_path': OUTPUT_DIR,
                    'base_size': base_size,
                    'image_size': image_size,
                    'crop_mode': crop_mode,
                    'save_results': True,
                    'test_compress': False,
                    'progress_callback': mlx_live_progress
                },
                status='processing',
                stage='ocr',
                message_builder=lambda elapsed: f'Running MLX OCR... {elapsed}s elapsed',
                progress_builder=lambda elapsed: min(95, 25 + (elapsed * 2)),
                heartbeat_seconds=1.5
            )
        else:
            old_stdout = sys.stdout
            char_count = [0]

            class CharCountingStream:
                def __init__(self, original_stdout):
                    self.original = original_stdout
                    self.accumulated_text = ''

                def write(self, text):
                    try:
                        self.original.write(text)
                    except UnicodeEncodeError:
                        try:
                            safe_text = text.encode(
                                self.original.encoding or 'utf-8',
                                errors='replace'
                            ).decode(self.original.encoding or 'utf-8')
                            self.original.write(safe_text)
                        except Exception:
                            pass

                    self.accumulated_text += text
                    raw_token_text = extract_raw_token_section(self.accumulated_text)
                    if raw_token_text:
                        char_count[0] = len(raw_token_text)
                        update_progress(
                            'processing',
                            'ocr',
                            'Generating OCR...',
                            estimate_stream_progress(char_count[0]),
                            char_count[0],
                            raw_token_text
                        )

                def flush(self):
                    self.original.flush()

            char_stream = CharCountingStream(old_stdout)
            sys.stdout = char_stream

            try:
                infer_result = run_model_infer(
                    prompt=prompt,
                    image_file=temp_input_path,
                    output_path=OUTPUT_DIR,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    save_results=True,
                    test_compress=False
                )
            finally:
                sys.stdout = old_stdout

        logger.info("OCR inference completed successfully")
        raw_token_text = extract_raw_token_section(char_stream.accumulated_text) if char_stream else None
        result_text = infer_result if isinstance(infer_result, str) else None
        if RUNTIME_BACKEND == 'mlx' and isinstance(infer_result, str):
            normalized_text, mlx_raw_tokens = normalize_mlx_output(prompt_type, infer_result)
            result_text = normalized_text or result_text
            if mlx_raw_tokens:
                raw_token_text = mlx_raw_tokens
        if not result_text:
            result_text = read_result_text(OUTPUT_DIR, expected_output_file)
        if result_text is None:
            result_text = "OCR completed but no text file was generated"
            logger.warning("No result file found in output directory")

        boxes_image_path = os.path.join(OUTPUT_DIR, 'result_with_boxes.jpg')
        has_boxes_image = os.path.exists(boxes_image_path)

        return jsonify({
            'status': 'success',
            'result': result_text,
            'boxes_image_path': 'result_with_boxes.jpg' if has_boxes_image else None,
            'prompt_type': prompt_type,
            'raw_tokens': raw_token_text
        })

    except ValueError as e:
        logger.error(f"Invalid OCR request: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error during OCR: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        update_progress('idle', '', '', 0, 0)

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the model"""
    gpu_target_hint = os.environ.get('DEEPSEEK_OCR_GPU_TARGET')
    cuda_memory = get_cuda_memory_snapshot()
    return jsonify({
        'runtime_backend': RUNTIME_BACKEND,
        'torch_compile_enabled': TORCH_COMPILE_ENABLED,
        'torch_available': is_torch_available(),
        'mlx_available': is_mlx_available(),
        'model_name': MODEL_NAME,
        'cache_dir': MODEL_CACHE_DIR,
        'model_loaded': is_model_ready(),
        'model_initialized': model is not None and tokenizer is not None,
        'gpu_available': device in ('cuda', 'mps'),
        'gpu_target_hint': gpu_target_hint,
        'gpu_target_device_hint': map_gpu_target_to_device(gpu_target_hint) if gpu_target_hint else None,
        'preferred_device': get_preferred_device(),
        'device_state': device,
        'gpu_name': torch.cuda.get_device_name(0) if (torch is not None and device == 'cuda' and torch.cuda.is_available()) else None,
        'cuda_memory': cuda_memory
    })

@app.route('/queue/add', methods=['POST'])
def add_to_queue():
    """Add files to the processing queue"""
    global processing_queue, queue_next_id
    
    try:
        # Get files from request
        files = request.files.getlist('files')
        if not files:
            return jsonify({'status': 'error', 'message': 'No files provided'}), 400
        
        # Get processing parameters
        prompt_type = request.form.get('prompt_type', 'document')
        base_size = int(request.form.get('base_size', 1024))
        image_size = int(request.form.get('image_size', 640))
        crop_mode = request.form.get('crop_mode', 'true').lower() == 'true'
        pdf_page_range = request.form.get('pdf_page_range', '').strip() or None
        
        added_files = []
        with queue_lock:
            for file in files:
                if file.filename:
                    raw_suffix = Path(file.filename).suffix.lower()
                    if raw_suffix and raw_suffix not in SUPPORTED_INPUT_EXTENSIONS:
                        logger.warning(f"Skipping unsupported file type: {file.filename}")
                        continue
                    input_suffix = raw_suffix if raw_suffix in SUPPORTED_INPUT_EXTENSIONS else '.jpg'

                    # Save file temporarily
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=input_suffix)
                    file.save(temp_file.name)
                    temp_file.close()
                    
                    queue_item = {
                        'id': queue_next_id,
                        'filename': file.filename,
                        'temp_path': temp_file.name,
                        'input_type': 'pdf' if input_suffix == '.pdf' else 'image',
                        'prompt_type': prompt_type,
                        'base_size': base_size,
                        'image_size': image_size,
                        'crop_mode': crop_mode,
                        'pdf_page_range': pdf_page_range,
                        'status': 'pending',
                        'progress': 0,
                        'progress_detail': None,
                        'result': None,
                        'error': None
                    }
                    queue_next_id += 1
                    processing_queue.append(queue_item)
                    added_files.append({'id': queue_item['id'], 'filename': file.filename})

        if not added_files:
            return jsonify({'status': 'error', 'message': 'No supported image/PDF files were provided'}), 400
        
        return jsonify({
            'status': 'success',
            'added': len(added_files),
            'files': added_files,
            'queue_length': len(processing_queue)
        })
    
    except Exception as e:
        logger.error(f"Error adding to queue: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/queue/status', methods=['GET'])
def get_queue_status():
    """Get current queue status with current file info for preview"""
    return jsonify(get_queue_summary())

@app.route('/queue/process', methods=['POST'])
def process_queue():
    """Start processing the queue sequentially"""
    global model, tokenizer, current_queue_id, device, dtype
    global queue_processing_active, queue_paused, queue_cancel_requested

    try:
        with queue_lock:
            if queue_processing_active:
                return jsonify({'status': 'error', 'message': 'Queue is already processing'}), 409
            queue_processing_active = True
            queue_paused = False
            queue_cancel_requested = False

        # Check if model is ready, load/wait if needed.
        if not is_model_ready():
            logger.info("Model not ready, loading now before processing queue...")
            load_model()

            max_wait = 300  # 5 minutes max
            start_time = time.time()
            while (time.time() - start_time) < max_wait:
                with progress_lock:
                    status = progress_data['status']

                if status == 'loaded' and is_model_ready():
                    logger.info("Model loaded successfully, starting queue processing")
                    break
                if status == 'error':
                    error_msg = progress_data.get('message', 'Unknown error')
                    logger.error(f"Model loading failed: {error_msg}")
                    with queue_lock:
                        queue_processing_active = False
                        queue_paused = False
                        queue_cancel_requested = False
                    return jsonify({'status': 'error', 'message': f"Model loading failed: {error_msg}"}), 500
                time.sleep(2)

            if not is_model_ready():
                logger.error("Model failed to load within timeout")
                with queue_lock:
                    queue_processing_active = False
                    queue_paused = False
                    queue_cancel_requested = False
                return jsonify({
                    'status': 'error',
                    'message': 'Model failed to load within 5 minutes. Please try loading model manually first.'
                }), 500
            logger.info("Waiting for model to be fully ready...")
            time.sleep(2)

        queue_folder = create_queue_output_folder()
        logger.info(f"Processing queue - output folder: {queue_folder}")

        with queue_lock:
            items_to_process = [item for item in processing_queue if item['status'] == 'pending']

        if not items_to_process:
            with queue_lock:
                queue_processing_active = False
            update_progress('idle', '', '', 0, 0)
            return jsonify({
                'status': 'success',
                'queue_folder': queue_folder,
                'completed': 0,
                'failed': 0,
                'total': 0,
                'canceled': False,
                'results': []
            })

        total_items = max(len(items_to_process), 1)
        results_summary = []
        canceled = False

        for idx, item in enumerate(items_to_process):
            if queue_should_stop():
                canceled = True
                break
            if not queue_wait_if_paused(item):
                canceled = True
                break

            try:
                with queue_lock:
                    item['status'] = 'processing'
                    item['progress'] = 0
                    item['error'] = None
                    item['progress_detail'] = format_queue_progress_detail(1, 1, paused=False)
                    item['current_image_path'] = None
                    current_queue_id = item['id']

                progress_msg = f"[{idx + 1}/{len(items_to_process)}] Processing {item['filename']}"
                logger.info(f"=== {progress_msg} ===")

                file_output_dir = os.path.join(queue_folder, f"file_{item['id']:03d}_{Path(item['filename']).stem}")
                os.makedirs(file_output_dir, exist_ok=True)
                update_progress('processing', 'queue', progress_msg, int((idx / len(items_to_process)) * 100), 0)

                prompt = get_prompt_for_type(item['prompt_type'])
                result_file = get_result_filename(item['prompt_type'])
                result_text = None
                page_count = 1
                page_labels = [1]
                effective_base_size = item['base_size']
                effective_image_size = item['image_size']
                effective_crop_mode = item['crop_mode']

                if is_pdf_input(item['temp_path']):
                    pdf_profile = build_pdf_inference_profile(
                        item['base_size'],
                        item['image_size'],
                        item['crop_mode']
                    )
                    effective_base_size = pdf_profile['base_size']
                    effective_image_size = pdf_profile['image_size']
                    effective_crop_mode = pdf_profile['crop_mode']
                    if pdf_profile['adjusted']:
                        logger.info(
                            f"[Queue {idx + 1}/{len(items_to_process)}] Applied PDF fast profile "
                            f"for {item['filename']}: {'; '.join(pdf_profile['notes'])} "
                            f"(render max side={pdf_profile['target_max_side']})"
                        )

                    with tempfile.TemporaryDirectory(prefix='queue_pdf_pages_') as page_dir:
                        page_images, page_labels = render_pdf_to_images(
                            item['temp_path'],
                            page_dir,
                            page_range_text=item.get('pdf_page_range'),
                            target_max_side=pdf_profile['target_max_side']
                        )
                        page_count = len(page_images)
                        page_results = []

                        for page_idx, page_image in enumerate(page_images, start=1):
                            page_label = page_labels[page_idx - 1] if page_idx - 1 < len(page_labels) else page_idx
                            logger.info(
                                f"[Queue {idx + 1}/{len(items_to_process)}] Starting page "
                                f"{page_label} ({page_idx}/{page_count}) for {item['filename']}"
                            )
                            page_started_at = time.time()
                            if queue_should_stop():
                                canceled = True
                                break
                            if not queue_wait_if_paused(item):
                                canceled = True
                                break

                            with queue_lock:
                                item['current_image_path'] = page_image
                                item['progress'] = int(((page_idx - 1) / max(page_count, 1)) * 100)
                                item['progress_detail'] = format_queue_progress_detail(page_idx, page_count, paused=False)

                            page_progress = int(((idx + ((page_idx - 1) / max(page_count, 1))) / total_items) * 100)
                            update_progress(
                                'processing',
                                'queue',
                                f"[{idx + 1}/{len(items_to_process)}] {item['filename']} (page {page_label}, {page_idx}/{page_count})",
                                page_progress,
                                0
                            )

                            page_output_dir = os.path.join(file_output_dir, f'page_{page_idx:04d}')
                            os.makedirs(page_output_dir, exist_ok=True)
                            clear_output_artifacts(page_output_dir)

                            page_infer_result = run_model_infer_with_heartbeat(
                                infer_kwargs={
                                    'prompt': prompt,
                                    'image_file': page_image,
                                    'output_path': page_output_dir,
                                    'base_size': effective_base_size,
                                    'image_size': effective_image_size,
                                    'crop_mode': effective_crop_mode,
                                    'save_results': not PDF_EVAL_MODE,
                                    'test_compress': False,
                                    'eval_mode': PDF_EVAL_MODE,
                                    'max_new_tokens_cap': PDF_MAX_NEW_TOKENS
                                },
                                status='processing',
                                stage='queue',
                                message_builder=lambda elapsed, idx=idx, total=len(items_to_process), filename=item['filename'], page_label=page_label, page_idx=page_idx, page_count=page_count:
                                    f"[{idx + 1}/{total}] {filename} (page {page_label}, {page_idx}/{page_count})... {elapsed}s elapsed",
                                progress_builder=lambda elapsed, page_progress=page_progress, page_count=page_count:
                                    estimate_pdf_page_progress(page_progress, page_count, elapsed),
                                heartbeat_seconds=2.0
                            )

                            page_text = page_infer_result if isinstance(page_infer_result, str) else ''
                            if RUNTIME_BACKEND == 'mlx' and page_text:
                                page_text, _ = normalize_mlx_output(item['prompt_type'], page_text)
                            if not page_text:
                                page_text = read_result_text(page_output_dir, result_file) or ''
                            if item['prompt_type'] == 'document':
                                page_text = strip_markdown_image_refs(page_text)
                            page_results.append(page_text)
                            logger.info(
                                f"[Queue {idx + 1}/{len(items_to_process)}] Completed page "
                                f"{page_label} ({page_idx}/{page_count}) for {item['filename']} "
                                f"in {time.time() - page_started_at:.1f}s"
                            )

                        if canceled:
                            with queue_lock:
                                item['status'] = 'pending'
                                item['progress'] = 0
                                item['progress_detail'] = None
                                item['current_image_path'] = None
                            break

                        result_text = combine_pdf_results(
                            page_results,
                            item['prompt_type'],
                            page_labels=page_labels
                        )
                        with open(os.path.join(file_output_dir, result_file), 'w', encoding='utf-8') as f:
                            f.write(result_text)
                else:
                    if not queue_wait_if_paused(item):
                        canceled = True
                        with queue_lock:
                            item['status'] = 'pending'
                            item['progress'] = 0
                            item['progress_detail'] = None
                            item['current_image_path'] = None
                        break

                    with queue_lock:
                        item['current_image_path'] = item['temp_path']
                        item['progress_detail'] = format_queue_progress_detail(1, 1, paused=False)

                    infer_result = None
                    if RUNTIME_BACKEND == 'mlx':
                        def mlx_progress(elapsed, idx=idx, total_items=total_items, item=item):
                            file_progress = min(95, max(1, int(elapsed * 3)))
                            with queue_lock:
                                is_paused = queue_paused
                                item['progress'] = file_progress
                                item['progress_detail'] = format_queue_progress_detail(1, 1, paused=is_paused)
                            return int(((idx + (file_progress / 100.0)) / total_items) * 100)

                        clear_output_artifacts(file_output_dir)
                        infer_result = run_model_infer_with_heartbeat(
                            infer_kwargs={
                                'prompt': prompt,
                                'image_file': item['temp_path'],
                                'output_path': file_output_dir,
                                'base_size': effective_base_size,
                                'image_size': effective_image_size,
                                'crop_mode': effective_crop_mode,
                                'save_results': True,
                                'test_compress': False
                            },
                            status='processing',
                            stage='queue',
                            message_builder=lambda elapsed, idx=idx, total=len(items_to_process), filename=item['filename']:
                                f"[{idx + 1}/{total}] {filename}... {elapsed}s elapsed",
                            progress_builder=mlx_progress,
                            heartbeat_seconds=1.5
                        )
                    else:
                        old_stdout = sys.stdout
                        char_count = [0]

                        class CharCountingStream:
                            def __init__(self):
                                self.accumulated_text = ''

                            def write(self, text):
                                self.accumulated_text += text
                                raw_token_text = extract_raw_token_section(self.accumulated_text)
                                if raw_token_text:
                                    char_count[0] = len(raw_token_text)
                                    file_progress = estimate_stream_progress(char_count[0])
                                    overall_progress = int(((idx + (file_progress / 100.0)) / total_items) * 100)
                                    with queue_lock:
                                        is_paused = queue_paused
                                        item['progress'] = file_progress
                                        item['progress_detail'] = format_queue_progress_detail(1, 1, paused=is_paused)
                                    update_progress(
                                        'processing',
                                        'queue',
                                        f"[{idx + 1}/{len(items_to_process)}] {item['filename']} (page 1/1)",
                                        overall_progress,
                                        char_count[0],
                                        raw_token_text
                                    )

                            def flush(self):
                                pass

                        char_stream = CharCountingStream()
                        sys.stdout = char_stream
                        try:
                            clear_output_artifacts(file_output_dir)
                            infer_result = run_model_infer(
                                prompt=prompt,
                                image_file=item['temp_path'],
                                output_path=file_output_dir,
                                base_size=effective_base_size,
                                image_size=effective_image_size,
                                crop_mode=effective_crop_mode,
                                save_results=True,
                                test_compress=False
                            )
                        finally:
                            sys.stdout = old_stdout

                    if queue_should_stop():
                        canceled = True
                        with queue_lock:
                            item['status'] = 'pending'
                            item['progress'] = 0
                            item['progress_detail'] = None
                            item['current_image_path'] = None
                        break

                    if isinstance(infer_result, str) and infer_result:
                        result_text = infer_result
                        if RUNTIME_BACKEND == 'mlx':
                            normalized_text, _ = normalize_mlx_output(item['prompt_type'], result_text)
                            result_text = normalized_text or result_text
                    else:
                        result_text = read_result_text(file_output_dir, result_file)

                if result_text is None:
                    result_text = ''

                metadata = {
                    'filename': item['filename'],
                    'input_type': item.get('input_type', 'image'),
                    'page_count': page_count,
                    'page_labels': page_labels,
                    'pdf_page_range': item.get('pdf_page_range'),
                    'prompt_type': item['prompt_type'],
                    'base_size': item['base_size'],
                    'image_size': item['image_size'],
                    'crop_mode': item['crop_mode'],
                    'applied_base_size': effective_base_size,
                    'applied_image_size': effective_image_size,
                    'applied_crop_mode': effective_crop_mode,
                    'processed_at': datetime.now().astimezone().isoformat(),
                    'status': 'completed'
                }
                with open(os.path.join(file_output_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)

                with queue_lock:
                    item['status'] = 'completed'
                    item['progress'] = 100
                    item['progress_detail'] = None
                    item['result'] = result_text
                    item['current_image_path'] = None

                logger.info(f"✓ Completed: {item['filename']} -> {file_output_dir}")
                results_summary.append({
                    'id': item['id'],
                    'filename': item['filename'],
                    'status': 'completed',
                    'output_dir': file_output_dir
                })

            except Exception as e:
                logger.error(f"Error processing {item['filename']}: {e}")
                with queue_lock:
                    item['status'] = 'failed'
                    item['error'] = str(e)
                    item['progress_detail'] = None
                    item['current_image_path'] = None
                results_summary.append({
                    'id': item['id'],
                    'filename': item['filename'],
                    'status': 'failed',
                    'error': str(e)
                })
            finally:
                if (
                    item.get('status') in ('completed', 'failed') and
                    os.path.exists(item['temp_path'])
                ):
                    os.remove(item['temp_path'])
                clear_cuda_cache()

        summary_path = os.path.join(queue_folder, 'queue_summary.json')
        completed_count = sum(1 for result in results_summary if result['status'] == 'completed')
        failed_count = sum(1 for result in results_summary if result['status'] == 'failed')

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'processed_at': datetime.now().astimezone().isoformat(),
                'total_files': len(results_summary),
                'completed': completed_count,
                'failed': failed_count,
                'canceled': canceled,
                'output_folder': queue_folder,
                'results': results_summary
            }, f, indent=2)

        logger.info("")
        logger.info("=" * 60)
        logger.info("QUEUE PROCESSING COMPLETE!")
        logger.info(f"Total: {len(results_summary)} | Completed: {completed_count} | Failed: {failed_count}")
        logger.info(f"Canceled: {canceled}")
        logger.info(f"Results saved to: {queue_folder}")
        logger.info("=" * 60)
        logger.info("")

        with queue_lock:
            queue_processing_active = False
            queue_paused = False
            queue_cancel_requested = False

        update_progress('idle', '', '', 0, 0)
        current_queue_id = None

        return jsonify({
            'status': 'success',
            'queue_folder': queue_folder,
            'completed': completed_count,
            'failed': failed_count,
            'total': len(results_summary),
            'canceled': canceled,
            'results': results_summary
        })

    except Exception as e:
        logger.error(f"Error processing queue: {e}")
        with queue_lock:
            queue_processing_active = False
            queue_paused = False
            queue_cancel_requested = False
        update_progress('idle', '', '', 0, 0)
        current_queue_id = None
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/queue/clear', methods=['POST'])
def clear_queue():
    """Clear all items from the queue"""
    global processing_queue, queue_paused, queue_cancel_requested

    with queue_lock:
        if queue_processing_active:
            return jsonify({'status': 'error', 'message': 'Cannot clear queue while processing'}), 409

        # Clean up temp files
        for item in processing_queue:
            if 'temp_path' in item and os.path.exists(item['temp_path']):
                try:
                    os.remove(item['temp_path'])
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {item['temp_path']}: {e}")

        processing_queue.clear()
        queue_paused = False
        queue_cancel_requested = False

    return jsonify({'status': 'success', 'message': 'Queue cleared'})


@app.route('/queue/pause', methods=['POST'])
def pause_queue():
    """Pause active queue processing."""
    global queue_paused
    with queue_lock:
        if not queue_processing_active:
            return jsonify({'status': 'error', 'message': 'Queue is not processing'}), 409
        queue_paused = True
    return jsonify({'status': 'success', 'message': 'Queue paused'})


@app.route('/queue/resume', methods=['POST'])
def resume_queue():
    """Resume paused queue processing."""
    global queue_paused
    with queue_lock:
        if not queue_processing_active:
            return jsonify({'status': 'error', 'message': 'Queue is not processing'}), 409
        queue_paused = False
    return jsonify({'status': 'success', 'message': 'Queue resumed'})


@app.route('/queue/cancel', methods=['POST'])
def cancel_queue():
    """Request cancellation for active queue processing."""
    global queue_cancel_requested, queue_paused
    with queue_lock:
        if not queue_processing_active:
            return jsonify({'status': 'error', 'message': 'Queue is not processing'}), 409
        queue_cancel_requested = True
        queue_paused = False
    return jsonify({'status': 'success', 'message': 'Queue cancellation requested'})


@app.route('/queue/retry_failed', methods=['POST'])
def retry_failed_queue_items():
    """Move failed queue items back to pending for retry."""
    retried = 0
    with queue_lock:
        if queue_processing_active:
            return jsonify({'status': 'error', 'message': 'Cannot retry while queue is processing'}), 409
        for item in processing_queue:
            if item['status'] == 'failed':
                item['status'] = 'pending'
                item['error'] = None
                item['progress'] = 0
                item['progress_detail'] = None
                item['result'] = None
                item['current_image_path'] = None
                retried += 1
    return jsonify({'status': 'success', 'retried': retried})

@app.route('/queue/remove/<int:item_id>', methods=['DELETE'])
def remove_from_queue(item_id):
    """Remove a specific item from the queue"""
    global processing_queue
    
    with queue_lock:
        for i, item in enumerate(processing_queue):
            if item['id'] == item_id:
                if queue_processing_active and item['status'] == 'processing':
                    return jsonify({'status': 'error', 'message': 'Cannot remove item currently processing'}), 409
                # Clean up temp file
                if 'temp_path' in item and os.path.exists(item['temp_path']):
                    try:
                        os.remove(item['temp_path'])
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file: {e}")
                
                processing_queue.pop(i)
                return jsonify({'status': 'success', 'message': f'Item {item_id} removed'})
        
        return jsonify({'status': 'error', 'message': 'Item not found'}), 404


@app.route('/diagnostics', methods=['GET'])
def diagnostics():
    """Return backend diagnostics payload for support bundles."""
    return jsonify(collect_runtime_diagnostics())

@app.route('/current_page_image', methods=['GET'])
def serve_current_page_image():
    """Serve the current PDF page image being processed"""
    with progress_lock:
        image_path = progress_data.get('current_page_image', '')
    if not image_path or not os.path.isfile(image_path):
        return '', 404
    directory = os.path.dirname(os.path.abspath(image_path))
    filename = os.path.basename(image_path)
    return send_from_directory(directory, filename)

@app.route('/outputs/<path:filename>', methods=['GET'])
def serve_output_file(filename):
    """Serve files from the outputs directory"""
    return send_from_directory(OUTPUT_DIR, filename)

if __name__ == '__main__':
    # Load model on startup
    logger.info("Starting DeepSeek OCR Server...")
    logger.info("Model will be automatically downloaded on first use")

    # Suppress Flask's default request logging
    import logging as log
    log.getLogger('werkzeug').disabled = True

    # Run Flask server without request logging
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
