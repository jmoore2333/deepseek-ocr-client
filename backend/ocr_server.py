#!/usr/bin/env python3
"""
DeepSeek OCR Backend Server
Handles model loading, caching, and OCR inference
"""
import os
import sys
import logging
import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from transformers import AutoModel, AutoTokenizer
import tempfile
import time
import json
from datetime import datetime
from threading import Thread, Lock
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress HTTP request logs from werkzeug
logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None
device = 'cpu'
dtype = torch.float32


def get_preferred_device():
    """Select runtime device based on availability or override."""
    if 'DEVICE' in os.environ:
        return os.environ['DEVICE'].lower()
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def get_preferred_model_name():
    """Use upstream model for CUDA, MPS/CPU-compatible fork otherwise."""
    if 'MODEL_NAME' in os.environ:
        return os.environ['MODEL_NAME']
    if get_preferred_device() == 'cuda':
        return 'deepseek-ai/DeepSeek-OCR'
    return 'Dogacel/DeepSeek-OCR-Metal-MPS'


MODEL_NAME = get_preferred_model_name()

# Queue processing state
processing_queue = []
queue_lock = Lock()
current_queue_id = None
queue_results = {}

# Use a writable cache directory (override via env for packaged apps)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.environ.get('DEEPSEEK_OCR_CACHE_DIR')
if CACHE_DIR:
    CACHE_DIR = os.path.abspath(CACHE_DIR)
else:
    CACHE_DIR = os.path.join(SCRIPT_DIR, '..', 'cache')
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, 'models')
OUTPUT_DIR = os.path.join(CACHE_DIR, 'outputs')

# Progress tracking
progress_data = {
    'status': 'idle',  # idle, loading, loaded, error
    'stage': '',       # tokenizer, model
    'message': '',
    'progress_percent': 0,  # 0-100
    'chars_generated': 0,  # For OCR character counting
    'raw_token_stream': '',  # Accumulated raw tokens during OCR
    'timestamp': time.time()
}
progress_lock = Lock()
loading_thread = None

def update_progress(status, stage='', message='', progress_percent=0, chars_generated=0, raw_token_stream=''):
    """Update the global progress data"""
    global progress_data
    with progress_lock:
        progress_data['status'] = status
        progress_data['stage'] = stage
        progress_data['message'] = message
        progress_data['progress_percent'] = progress_percent
        progress_data['chars_generated'] = chars_generated
        progress_data['raw_token_stream'] = raw_token_stream
        progress_data['timestamp'] = time.time()
        if chars_generated > 0:
            logger.info(f"Progress: {status} - {stage} - {message} ({progress_percent}%) - {chars_generated} chars")
        else:
            logger.info(f"Progress: {status} - {stage} - {message} ({progress_percent}%)")

def log_selected_device(selected_device):
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


def run_model_infer(**kwargs):
    """Run model inference with device arguments when supported by the model."""
    global model, tokenizer, device, dtype
    if device == 'cuda':
        return model.infer(tokenizer, **kwargs)
    try:
        return model.infer(tokenizer, device=torch.device(device), dtype=dtype, **kwargs)
    except TypeError:
        logger.warning("Model infer() does not accept explicit device/dtype, retrying default call")
        return model.infer(tokenizer, **kwargs)

def load_model_background():
    """Background thread function to load the model"""
    global model, tokenizer, device, dtype

    try:
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
        elif device == 'mps':
            dtype = torch.float16
            model = model.to(torch.device('mps')).to(dtype)
            logger.info("Model loaded on MPS with float16")
        else:
            dtype = torch.float32
            model = model.to(torch.device('cpu')).to(dtype)
            logger.info("Model loaded on CPU with float32")

        # Apply torch.compile for ~30% inference speedup (PyTorch 2.0+) (95% progress)
        update_progress('loading', 'optimize', 'Compiling model with torch.compile...', 95)
        try:
            if hasattr(torch, 'compile') and device == 'cuda':
                logger.info("Applying torch.compile for faster inference...")
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled successfully (expect ~30% speedup)")
            else:
                logger.info("torch.compile unavailable or unsupported for current device, skipping compilation")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}, using uncompiled model")

        # Warmup inference to initialize graphs
        if device in ('cuda', 'mps'):
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

def clear_cuda_cache():
    """Clear CUDA cache to free memory between processing"""
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

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
PDF_RENDER_SCALE = 2.0


def get_input_suffix(filename):
    suffix = Path(filename or '').suffix.lower()
    if suffix in SUPPORTED_INPUT_EXTENSIONS:
        return suffix
    return '.jpg'


def is_pdf_input(input_path):
    return Path(input_path).suffix.lower() == '.pdf'


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


def combine_pdf_results(page_results, prompt_type):
    if not page_results:
        return ''

    if prompt_type == 'document':
        chunks = []
        for page_num, text in enumerate(page_results, start=1):
            cleaned = (text or '').strip() or '(No text extracted)'
            chunks.append(f'## Page {page_num}\n\n{cleaned}')
        return '\n\n'.join(chunks).strip()

    chunks = []
    for page_num, text in enumerate(page_results, start=1):
        cleaned = (text or '').strip()
        chunks.append(f'--- Page {page_num} ---\n{cleaned}')
    return '\n\n'.join(chunks).strip()


def render_pdf_to_images(pdf_path, output_dir):
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError('PDF support requires pypdfium2. Please reinstall dependencies.') from exc

    os.makedirs(output_dir, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_path)
    page_paths = []
    try:
        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            pil_image = page.render(scale=PDF_RENDER_SCALE).to_pil()
            page_path = os.path.join(output_dir, f'page_{page_idx + 1:04d}.jpg')
            pil_image.save(page_path, format='JPEG', quality=95)
            page_paths.append(page_path)
    finally:
        try:
            pdf.close()
        except Exception:
            pass

    if not page_paths:
        raise RuntimeError('The PDF does not contain any renderable pages.')
    return page_paths


def extract_raw_token_section(accumulated_text):
    if not accumulated_text:
        return None
    parts = accumulated_text.split('=' * 20)
    if len(parts) < 3:
        return None
    return parts[2].strip().lstrip('=').strip()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'gpu_available': device in ('cuda', 'mps'),
        'device_state': device
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
        # Check if model is loaded
        if model is None or tokenizer is None:
            logger.info("Model not loaded, loading now...")
            if not load_model():
                return jsonify({'status': 'error', 'message': 'Failed to load model'}), 500

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
            with tempfile.TemporaryDirectory(prefix='ocr_pdf_pages_') as page_dir:
                page_images = render_pdf_to_images(temp_input_path, page_dir)
                total_pages = len(page_images)
                page_results = []

                for page_idx, page_image in enumerate(page_images, start=1):
                    update_progress(
                        'processing',
                        'ocr',
                        f'Processing PDF page {page_idx}/{total_pages}...',
                        int(((page_idx - 1) / total_pages) * 95),
                        0
                    )
                    clear_output_artifacts(OUTPUT_DIR)
                    run_model_infer(
                        prompt=prompt,
                        image_file=page_image,
                        output_path=OUTPUT_DIR,
                        base_size=base_size,
                        image_size=image_size,
                        crop_mode=crop_mode,
                        save_results=True,
                        test_compress=True
                    )
                    page_text = read_result_text(OUTPUT_DIR, expected_output_file) or ''
                    if prompt_type == 'document':
                        page_text = strip_markdown_image_refs(page_text)
                    page_results.append(page_text)

                result_text = combine_pdf_results(page_results, prompt_type)

            return jsonify({
                'status': 'success',
                'result': result_text or 'OCR completed but no text was generated',
                'boxes_image_path': None,
                'prompt_type': prompt_type,
                'raw_tokens': None,
                'is_pdf': True,
                'page_count': len(page_results)
            })

        # Image flow with token streaming progress
        clear_output_artifacts(OUTPUT_DIR)
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
                    update_progress('processing', 'ocr', 'Generating OCR...', 50, char_count[0], raw_token_text)

            def flush(self):
                self.original.flush()

        char_stream = CharCountingStream(old_stdout)
        sys.stdout = char_stream

        try:
            run_model_infer(
                prompt=prompt,
                image_file=temp_input_path,
                output_path=OUTPUT_DIR,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=True,
                test_compress=True
            )
        finally:
            sys.stdout = old_stdout

        logger.info("OCR inference completed successfully")
        result_text = read_result_text(OUTPUT_DIR, expected_output_file)
        if result_text is None:
            result_text = "OCR completed but no text file was generated"
            logger.warning("No result file found in output directory")

        boxes_image_path = os.path.join(OUTPUT_DIR, 'result_with_boxes.jpg')
        has_boxes_image = os.path.exists(boxes_image_path)
        raw_token_text = extract_raw_token_section(char_stream.accumulated_text)

        return jsonify({
            'status': 'success',
            'result': result_text,
            'boxes_image_path': 'result_with_boxes.jpg' if has_boxes_image else None,
            'prompt_type': prompt_type,
            'raw_tokens': raw_token_text
        })

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
    return jsonify({
        'model_name': MODEL_NAME,
        'cache_dir': MODEL_CACHE_DIR,
        'model_loaded': model is not None,
        'gpu_available': device in ('cuda', 'mps'),
        'device_state': device,
        'gpu_name': torch.cuda.get_device_name(0) if (device == 'cuda' and torch.cuda.is_available()) else None
    })

@app.route('/queue/add', methods=['POST'])
def add_to_queue():
    """Add files to the processing queue"""
    global processing_queue
    
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
                        'id': len(processing_queue),
                        'filename': file.filename,
                        'temp_path': temp_file.name,
                        'input_type': 'pdf' if input_suffix == '.pdf' else 'image',
                        'prompt_type': prompt_type,
                        'base_size': base_size,
                        'image_size': image_size,
                        'crop_mode': crop_mode,
                        'status': 'pending',
                        'progress': 0,
                        'result': None,
                        'error': None
                    }
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
    with queue_lock:
        # Find currently processing item
        current_item = next((item for item in processing_queue if item['status'] == 'processing'), None)
        
        queue_summary = {
            'total': len(processing_queue),
            'pending': sum(1 for item in processing_queue if item['status'] == 'pending'),
            'processing': sum(1 for item in processing_queue if item['status'] == 'processing'),
            'completed': sum(1 for item in processing_queue if item['status'] == 'completed'),
            'failed': sum(1 for item in processing_queue if item['status'] == 'failed'),
            'items': [{
                'id': item['id'],
                'filename': item['filename'],
                'status': item['status'],
                'progress': item['progress'],
                'error': item['error']
            } for item in processing_queue],
            'current_file': {
                'id': current_item['id'],
                'filename': current_item['filename'],
                'image_path': current_item.get('current_image_path'),
                'progress': current_item['progress']
            } if current_item else None
        }
    
    return jsonify(queue_summary)

@app.route('/queue/process', methods=['POST'])
def process_queue():
    """Start processing the queue sequentially"""
    global model, tokenizer, current_queue_id, device, dtype
    
    try:
        # Check if model is loaded, load it if not
        if model is None or tokenizer is None:
            logger.info("Model not loaded, loading now before processing queue...")
            load_model()
            
            # Wait for model to load (poll until loaded or error)
            max_wait = 300  # 5 minutes max
            start_time = time.time()
            while (time.time() - start_time) < max_wait:
                # Check progress status
                with progress_lock:
                    status = progress_data['status']
                    
                if status == 'loaded' and model is not None and tokenizer is not None:
                    logger.info("Model loaded successfully, starting queue processing")
                    break
                elif status == 'error':
                    error_msg = progress_data.get('message', 'Unknown error')
                    logger.error(f"Model loading failed: {error_msg}")
                    return jsonify({'status': 'error', 'message': f"Model loading failed: {error_msg}"}), 500
                
                time.sleep(2)  # Check every 2 seconds
            
            # Final check after timeout
            if model is None or tokenizer is None:
                logger.error("Model failed to load within timeout")
                return jsonify({'status': 'error', 'message': 'Model failed to load within 5 minutes. Please try loading model manually first.'}), 500
            
            # Extra safety: wait 2 more seconds for model to be fully ready
            logger.info("Waiting for model to be fully ready...")
            time.sleep(2)
        
        # Create output folder for this queue
        queue_folder = create_queue_output_folder()
        logger.info(f"Processing queue - output folder: {queue_folder}")
        
        # Process queue sequentially
        results_summary = []
        
        with queue_lock:
            items_to_process = [item for item in processing_queue if item['status'] == 'pending']
        
        for idx, item in enumerate(items_to_process):
            try:
                with queue_lock:
                    item['status'] = 'processing'
                    current_queue_id = item['id']
                
                progress_msg = f"[{idx + 1}/{len(items_to_process)}] Processing {item['filename']}"
                logger.info(f"=== {progress_msg} ===")
                
                # Create output subfolder for this file
                file_output_dir = os.path.join(queue_folder, f"file_{item['id']:03d}_{Path(item['filename']).stem}")
                os.makedirs(file_output_dir, exist_ok=True)
                
                # Update progress
                update_progress('processing', 'queue', progress_msg, int((idx / len(items_to_process)) * 100), 0)
                
                # Perform OCR with progress tracking (similar to single file)
                try:
                    prompt = get_prompt_for_type(item['prompt_type'])
                    result_file = get_result_filename(item['prompt_type'])
                    result_text = None
                    page_count = 1

                    if is_pdf_input(item['temp_path']):
                        with tempfile.TemporaryDirectory(prefix='queue_pdf_pages_') as page_dir:
                            page_images = render_pdf_to_images(item['temp_path'], page_dir)
                            page_count = len(page_images)
                            page_results = []

                            for page_idx, page_image in enumerate(page_images, start=1):
                                with queue_lock:
                                    item['current_image_path'] = page_image
                                    item['progress'] = int(((page_idx - 1) / page_count) * 100)

                                page_progress = int(
                                    ((idx + ((page_idx - 1) / page_count)) / max(len(items_to_process), 1)) * 100
                                )
                                update_progress(
                                    'processing',
                                    'queue',
                                    f"[{idx + 1}/{len(items_to_process)}] {item['filename']} (page {page_idx}/{page_count})",
                                    page_progress,
                                    0
                                )

                                page_output_dir = os.path.join(file_output_dir, f'page_{page_idx:04d}')
                                os.makedirs(page_output_dir, exist_ok=True)
                                clear_output_artifacts(page_output_dir)

                                run_model_infer(
                                    prompt=prompt,
                                    image_file=page_image,
                                    output_path=page_output_dir,
                                    base_size=item['base_size'],
                                    image_size=item['image_size'],
                                    crop_mode=item['crop_mode'],
                                    save_results=True,
                                    test_compress=True
                                )

                                page_text = read_result_text(page_output_dir, result_file) or ''
                                if item['prompt_type'] == 'document':
                                    page_text = strip_markdown_image_refs(page_text)
                                page_results.append(page_text)

                            result_text = combine_pdf_results(page_results, item['prompt_type'])
                            with open(os.path.join(file_output_dir, result_file), 'w', encoding='utf-8') as f:
                                f.write(result_text)
                    else:
                        with queue_lock:
                            item['current_image_path'] = item['temp_path']

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
                                    progress_msg = f"[{idx + 1}/{len(items_to_process)}] {item['filename']}"
                                    update_progress(
                                        'processing',
                                        'queue',
                                        progress_msg,
                                        int((idx / len(items_to_process)) * 100),
                                        char_count[0],
                                        raw_token_text
                                    )
                                    with queue_lock:
                                        item['progress'] = min(int((char_count[0] / 1000) * 100), 90)

                            def flush(self):
                                pass

                        char_stream = CharCountingStream()
                        sys.stdout = char_stream
                        try:
                            clear_output_artifacts(file_output_dir)
                            run_model_infer(
                                prompt=prompt,
                                image_file=item['temp_path'],
                                output_path=file_output_dir,
                                base_size=item['base_size'],
                                image_size=item['image_size'],
                                crop_mode=item['crop_mode'],
                                save_results=True,
                                test_compress=True
                            )
                        finally:
                            sys.stdout = old_stdout

                        result_text = read_result_text(file_output_dir, result_file)

                    if result_text is None:
                        result_text = ''
                    
                    # Save metadata
                    metadata = {
                        'filename': item['filename'],
                        'input_type': item.get('input_type', 'image'),
                        'page_count': page_count,
                        'prompt_type': item['prompt_type'],
                        'base_size': item['base_size'],
                        'image_size': item['image_size'],
                        'crop_mode': item['crop_mode'],
                        'processed_at': datetime.now().isoformat(),
                        'status': 'completed'
                    }
                    with open(os.path.join(file_output_dir, 'metadata.json'), 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    with queue_lock:
                        item['status'] = 'completed'
                        item['progress'] = 100
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
                        item['current_image_path'] = None
                    
                    results_summary.append({
                        'id': item['id'],
                        'filename': item['filename'],
                        'status': 'failed',
                        'error': str(e)
                    })
                
                finally:
                    # Clean up temp file
                    if os.path.exists(item['temp_path']):
                        os.remove(item['temp_path'])
                    
                    # Clear CUDA cache between items
                    clear_cuda_cache()
            
            except Exception as e:
                logger.error(f"Critical error processing queue item: {e}")
                with queue_lock:
                    item['status'] = 'failed'
                    item['error'] = str(e)
        
        # Save queue summary
        summary_path = os.path.join(queue_folder, 'queue_summary.json')
        completed_count = sum(1 for r in results_summary if r['status'] == 'completed')
        failed_count = sum(1 for r in results_summary if r['status'] == 'failed')
        
        with open(summary_path, 'w') as f:
            json.dump({
                'processed_at': datetime.now().isoformat(),
                'total_files': len(results_summary),
                'completed': completed_count,
                'failed': failed_count,
                'output_folder': queue_folder,
                'results': results_summary
            }, f, indent=2)
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"QUEUE PROCESSING COMPLETE!")
        logger.info(f"Total: {len(results_summary)} | Completed: {completed_count} | Failed: {failed_count}")
        logger.info(f"Results saved to: {queue_folder}")
        logger.info("=" * 60)
        logger.info("")
        
        update_progress('idle', '', '', 0, 0)
        current_queue_id = None
        
        return jsonify({
            'status': 'success',
            'queue_folder': queue_folder,
            'completed': completed_count,
            'failed': failed_count,
            'total': len(results_summary),
            'results': results_summary
        })
    
    except Exception as e:
        logger.error(f"Error processing queue: {e}")
        update_progress('idle', '', '', 0, 0)
        current_queue_id = None
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/queue/clear', methods=['POST'])
def clear_queue():
    """Clear all items from the queue"""
    global processing_queue
    
    with queue_lock:
        # Clean up temp files
        for item in processing_queue:
            if 'temp_path' in item and os.path.exists(item['temp_path']):
                try:
                    os.remove(item['temp_path'])
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {item['temp_path']}: {e}")
        
        processing_queue.clear()
    
    return jsonify({'status': 'success', 'message': 'Queue cleared'})

@app.route('/queue/remove/<int:item_id>', methods=['DELETE'])
def remove_from_queue(item_id):
    """Remove a specific item from the queue"""
    global processing_queue
    
    with queue_lock:
        for i, item in enumerate(processing_queue):
            if item['id'] == item_id:
                # Clean up temp file
                if 'temp_path' in item and os.path.exists(item['temp_path']):
                    try:
                        os.remove(item['temp_path'])
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file: {e}")
                
                processing_queue.pop(i)
                return jsonify({'status': 'success', 'message': f'Item {item_id} removed'})
        
        return jsonify({'status': 'error', 'message': 'Item not found'}), 404

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
