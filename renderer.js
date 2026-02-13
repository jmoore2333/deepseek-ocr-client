const api = window.appAPI;

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const selectBtn = document.getElementById('select-btn');
const clearBtn = document.getElementById('clear-btn');
const viewBoxesBtn = document.getElementById('view-boxes-btn');
const viewTokensBtn = document.getElementById('view-tokens-btn');
const downloadZipBtn = document.getElementById('download-zip-btn');
const ocrBtn = document.getElementById('ocr-btn');
const ocrBtnText = document.getElementById('ocr-btn-text');
const loadModelBtn = document.getElementById('load-model-btn');
const copyBtn = document.getElementById('copy-btn');
const previewSection = document.getElementById('preview-section');
const imagePreview = document.getElementById('image-preview');
const selectedFileLabel = document.getElementById('selected-file-label');
const resultsContent = document.getElementById('results-content');
const ocrPreviewImage = document.getElementById('ocr-preview-image');
const ocrBoxesOverlay = document.getElementById('ocr-boxes-overlay');
const progressInline = document.getElementById('progress-inline');
const progressStatus = document.getElementById('progress-status');
const preflightBtn = document.getElementById('preflight-btn');
const diagnosticsBtn = document.getElementById('diagnostics-btn');
const preflightPanel = document.getElementById('preflight-panel');
const preflightSummary = document.getElementById('preflight-summary');
const preflightDetails = document.getElementById('preflight-details');
const messageBanner = document.getElementById('message-banner');
const uiModeBasicBtn = document.getElementById('ui-mode-basic-btn');
const uiModeAdvancedBtn = document.getElementById('ui-mode-advanced-btn');

// Lightbox elements
const lightbox = document.getElementById('lightbox');
const lightboxImage = document.getElementById('lightbox-image');
const lightboxText = document.getElementById('lightbox-text');
const lightboxClose = document.querySelector('.lightbox-close');

// Status elements
const serverStatus = document.getElementById('server-status');
const modelStatus = document.getElementById('model-status');
const gpuStatus = document.getElementById('gpu-status');
const startupPanel = document.getElementById('startup-panel');
const startupPhase = document.getElementById('startup-phase');
const startupMessage = document.getElementById('startup-message');
const startupPercent = document.getElementById('startup-percent');
const startupProgressBar = document.getElementById('startup-progress-bar');

// Form elements
const promptType = document.getElementById('prompt-type');
const baseSize = document.getElementById('base-size');
const imageSize = document.getElementById('image-size');
const cropMode = document.getElementById('crop-mode');
const pdfPageRange = document.getElementById('pdf-page-range');

// Queue elements
const queueCount = document.getElementById('queue-count');
const addFilesBtn = document.getElementById('add-files-btn');
const addFolderBtn = document.getElementById('add-folder-btn');
const processQueueBtn = document.getElementById('process-queue-btn');
const pauseQueueBtn = document.getElementById('pause-queue-btn');
const resumeQueueBtn = document.getElementById('resume-queue-btn');
const cancelQueueBtn = document.getElementById('cancel-queue-btn');
const retryFailedBtn = document.getElementById('retry-failed-btn');
const clearQueueBtn = document.getElementById('clear-queue-btn');
const openOutputBtn = document.getElementById('open-output-btn');
const queueList = document.getElementById('queue-list');
const queueProgressText = document.getElementById('queue-progress-text');
const queueProgressBar = document.getElementById('queue-progress-bar');
const retentionOutputDays = document.getElementById('retention-output-days');
const retentionMaxRuns = document.getElementById('retention-max-runs');
const retentionCacheDays = document.getElementById('retention-cache-days');
const retentionCleanupStartup = document.getElementById('retention-cleanup-startup');
const saveRetentionBtn = document.getElementById('save-retention-btn');
const runCleanupBtn = document.getElementById('run-cleanup-btn');
const retentionStatus = document.getElementById('retention-status');

// Constants
const DEEPSEEK_COORD_MAX = 999;
const STARTUP_PHASE_LABELS = {
    booting: 'Starting',
    'setup-check': 'Checking Python environment',
    'setup-ready': 'Using existing environment',
    'setup-init': 'Initializing setup',
    'setup-detect-hardware': 'Detecting hardware',
    'setup-hardware-change': 'Refreshing for hardware',
    'setup-install-python': 'Installing Python',
    'setup-create-venv': 'Creating virtual environment',
    'setup-install-deps': 'Installing dependencies',
    'setup-verify': 'Verifying environment',
    'setup-complete': 'Setup complete',
    cleanup: 'Cleaning cache',
    'backend-init': 'Preparing backend',
    'dev-env': 'Using development environment',
    'backend-launch': 'Launching backend',
    ready: 'Ready',
    error: 'Startup error'
};

const KNOWN_TYPES = ['title', 'sub_title', 'text', 'table', 'image', 'image_caption', 'figure', 'caption', 'formula', 'list'];

const TYPE_COLORS = {
    'title': '#8B5CF6',
    'sub_title': '#A78BFA',
    'text': '#3B82F6',
    'table': '#F59E0B',
    'image': '#EC4899',
    'figure': '#06B6D4',
    'caption': '#10B981',
    'image_caption': '#4EC483',
    'formula': '#EF4444',
    'list': '#6366F1'
};

// State
let currentImagePath = null;
let currentResultText = null;
let currentRawTokens = null;
let currentPromptType = null;
let isProcessing = false;
let lastBoxCount = 0;

// Queue state
let queueItems = [];
let isQueueProcessing = false;
let currentQueueFolder = null;
let lastProcessedFileId = null;
let lastQueuePreviewImagePath = null;
let startupHideTimer = null;
let uiMode = 'advanced';
const progressUpdateTimes = new WeakMap();
let messageHideTimer = null;

function isPdfPath(filePath) {
    return api.extname(filePath || '') === '.pdf';
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function clampPercent(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
        return 0;
    }
    return Math.max(0, Math.min(100, Math.round(numeric)));
}

function setProgressBarWidth(barEl, percent) {
    if (!barEl) return;
    const next = clampPercent(percent);
    const now = performance.now();
    const lastUpdate = progressUpdateTimes.get(barEl) || 0;
    if (now - lastUpdate < 80) {
        return;
    }
    progressUpdateTimes.set(barEl, now);
    barEl.style.width = `${next}%`;
}

function formatBytes(bytes) {
    const numeric = Number(bytes);
    if (!Number.isFinite(numeric) || numeric < 0) {
        return 'Unknown';
    }
    if (numeric === 0) {
        return '0 B';
    }
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = numeric;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024;
        unitIndex += 1;
    }
    return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function sanitizePageRangeInput(value) {
    const text = (value || '').trim();
    if (!text) {
        return '';
    }
    if (!/^[0-9,\-\s]+$/.test(text)) {
        throw new Error('PDF page range can only include digits, commas, spaces, and hyphens');
    }
    return text;
}

async function waitForModelLoadCompletion(timeoutMs = 1200000, pollMs = 500) {
    const startedAt = Date.now();
    let lastPercent = 0;
    let loopCount = 0;

    while (Date.now() - startedAt < timeoutMs) {
        loopCount += 1;
        try {
            const response = await fetch('http://127.0.0.1:5000/progress');
            const data = await response.json();

            if (data.status === 'loading') {
                lastPercent = Math.max(lastPercent, clampPercent(data.progress_percent));
                const stageText = data.message || data.stage || 'Preparing model...';
                progressStatus.textContent = `Loading ${lastPercent}% - ${stageText}`;
            } else if (data.status === 'loaded') {
                progressStatus.textContent = 'Model loaded successfully!';
                return data;
            } else if (data.status === 'error') {
                const message = data.message ? `Error loading model: ${data.message}` : 'Error loading model';
                progressStatus.textContent = message;
                return data;
            }
        } catch (error) {
            // Transient polling failures are expected while backend state changes.
        }

        // Some backends can report model loaded before progress leaves warmup.
        if (loopCount % 3 === 0) {
            try {
                const statusResult = await api.checkServerStatus();
                if (statusResult.success && statusResult.data?.model_loaded) {
                    progressStatus.textContent = 'Model loaded successfully!';
                    return { status: 'loaded', message: 'Model loaded (health endpoint confirmed)' };
                }
            } catch (error) {
                // Ignore transient health-check failures while backend starts.
            }
        }

        await sleep(pollMs);
    }

    return { status: 'timeout', message: 'Timed out waiting for model loading progress' };
}

async function ensureModelLoadedForRun() {
    const statusResult = await api.checkServerStatus();
    if (!statusResult.success) {
        throw new Error(`Backend unavailable: ${statusResult.error || 'Unable to reach backend service'}`);
    }

    if (statusResult.data?.model_loaded) {
        return;
    }

    progressInline.style.display = 'flex';
    progressStatus.textContent = 'Model not loaded. Starting automatic load...';

    const loadResult = await api.loadModel();
    if (!loadResult.success) {
        throw new Error(`Failed to start model loading: ${loadResult.error || 'Unknown error'}`);
    }

    const finalProgress = await waitForModelLoadCompletion(1200000, 600);
    if (finalProgress.status === 'loaded') {
        progressStatus.textContent = 'Model loaded. Starting OCR...';
        return;
    }

    if (finalProgress.status === 'error') {
        throw new Error(finalProgress.message || 'Model loading failed');
    }

    // If progress polling timed out, do one final readiness check.
    try {
        const finalStatus = await api.checkServerStatus();
        if (finalStatus.success && finalStatus.data?.model_loaded) {
            progressStatus.textContent = 'Model loaded. Starting OCR...';
            return;
        }
    } catch (error) {
        // Fall through to timeout error.
    }

    throw new Error('Model loading timed out. Please check diagnostics/logs and retry.');
}

window.addEventListener('DOMContentLoaded', () => {
    initializeUiMode();
    initializeRetentionPolicy();
    initializeStartupStatus();
    checkServerStatus();
    setupEventListeners();
    updateQueueDisplay();
    setInterval(checkServerStatus, 5000);
});

function formatStartupPhase(phase) {
    if (!phase) {
        return 'Starting';
    }
    if (STARTUP_PHASE_LABELS[phase]) {
        return STARTUP_PHASE_LABELS[phase];
    }
    return phase
        .split('-')
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(' ');
}

function renderStartupStatus(status) {
    if (!startupPanel || !status) {
        return;
    }

    const state = status.state || 'running';
    const numericProgress = Number.isFinite(Number(status.progress)) ? Number(status.progress) : 0;
    const progress = Math.max(0, Math.min(100, Math.round(numericProgress)));

    if (startupHideTimer) {
        clearTimeout(startupHideTimer);
        startupHideTimer = null;
    }

    startupPanel.style.display = 'block';
    startupPanel.setAttribute('data-state', state);
    startupPhase.textContent = formatStartupPhase(status.phase);
    startupMessage.textContent = status.message || 'Preparing application...';
    startupPercent.textContent = `${progress}%`;
    setProgressBarWidth(startupProgressBar, progress);

    if (state === 'ready') {
        startupHideTimer = setTimeout(() => {
            startupPanel.style.display = 'none';
        }, 1600);
    }
}

async function initializeStartupStatus() {
    api.onStartupStatus((status) => {
        renderStartupStatus(status);
    });

    try {
        const status = await api.getStartupStatus();
        renderStartupStatus(status);
    } catch (error) {
        console.error('Failed to get startup status:', error);
    }
}

function setupEventListeners() {
    // Image selection
    selectBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering dropZone click
        selectImage();
    });
    clearBtn.addEventListener('click', clearImage);
    viewBoxesBtn.addEventListener('click', viewBoxesImage);
    viewTokensBtn.addEventListener('click', viewRawTokens);
    imagePreview.addEventListener('click', viewOriginalImage);

    // Make entire drop zone clickable
    dropZone.addEventListener('click', selectImage);

    // Drag and drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('is-drag-over');
    });

    dropZone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropZone.classList.remove('is-drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('is-drag-over');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const file = files[0];
            const droppedPath = file.path || '';
            if (file.type.startsWith('image/') || file.type === 'application/pdf' || isPdfPath(droppedPath)) {
                loadImage(file.path);
            } else {
                showMessage('Please drop an image or PDF file', 'error');
            }
        }
    });

    // OCR
    ocrBtn.addEventListener('click', performOCR);

    // Load model
    loadModelBtn.addEventListener('click', loadModel);

    // Copy results
    copyBtn.addEventListener('click', copyResults);

    // Download zip
    downloadZipBtn.addEventListener('click', downloadZip);

    // Lightbox
    lightboxClose.addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', (e) => {
        if (e.target === lightbox) {
            closeLightbox();
        }
    });

    // Queue operations
    addFilesBtn.addEventListener('click', addFilesToQueue);
    addFolderBtn.addEventListener('click', addFolderToQueue);
    processQueueBtn.addEventListener('click', processQueue);
    pauseQueueBtn.addEventListener('click', pauseQueue);
    resumeQueueBtn.addEventListener('click', resumeQueue);
    cancelQueueBtn.addEventListener('click', cancelQueue);
    retryFailedBtn.addEventListener('click', retryFailedQueueItems);
    clearQueueBtn.addEventListener('click', clearQueue);
    openOutputBtn.addEventListener('click', openOutputFolder);
    preflightBtn.addEventListener('click', runPreflightCheck);
    diagnosticsBtn.addEventListener('click', exportDiagnosticsBundle);
    uiModeBasicBtn.addEventListener('click', () => applyUiMode('basic'));
    uiModeAdvancedBtn.addEventListener('click', () => applyUiMode('advanced'));
    saveRetentionBtn.addEventListener('click', saveRetentionPolicy);
    runCleanupBtn.addEventListener('click', runRetentionCleanup);
}

function applyUiMode(nextMode) {
    uiMode = nextMode === 'basic' ? 'basic' : 'advanced';
    document.body.setAttribute('data-ui-mode', uiMode);
    uiModeBasicBtn.classList.toggle('active', uiMode === 'basic');
    uiModeAdvancedBtn.classList.toggle('active', uiMode === 'advanced');
    localStorage.setItem('deepseek-ocr-ui-mode', uiMode);

    if (uiMode === 'basic') {
        viewBoxesBtn.style.display = 'none';
        viewTokensBtn.style.display = 'none';
    } else if (currentRawTokens) {
        viewTokensBtn.style.display = 'inline-block';
        viewBoxesBtn.style.display = 'inline-block';
    }
}

function initializeUiMode() {
    const storedMode = localStorage.getItem('deepseek-ocr-ui-mode');
    applyUiMode(storedMode || 'advanced');
}

async function initializeRetentionPolicy() {
    try {
        const result = await api.getRetentionPolicy();
        if (!result.success) {
            return;
        }
        const policy = result.data || {};
        retentionOutputDays.value = String(policy.outputRetentionDays ?? 30);
        retentionMaxRuns.value = String(policy.maxQueueRuns ?? 40);
        retentionCacheDays.value = String(policy.downloadCacheRetentionDays ?? 30);
        retentionCleanupStartup.checked = Boolean(policy.cleanupOnStartup);
    } catch (error) {
        console.error('Failed to load retention policy:', error);
    }
}

function collectRetentionPolicyInput() {
    return {
        outputRetentionDays: parseInt(retentionOutputDays.value, 10),
        maxQueueRuns: parseInt(retentionMaxRuns.value, 10),
        downloadCacheRetentionDays: parseInt(retentionCacheDays.value, 10),
        cleanupOnStartup: retentionCleanupStartup.checked
    };
}

async function saveRetentionPolicy() {
    retentionStatus.textContent = 'Saving...';
    const result = await api.updateRetentionPolicy(collectRetentionPolicyInput());
    if (result.success) {
        retentionStatus.textContent = 'Policy saved';
    } else {
        retentionStatus.textContent = `Save failed: ${result.error}`;
    }
}

async function runRetentionCleanup() {
    retentionStatus.textContent = 'Running cleanup...';
    const result = await api.applyRetentionCleanup();
    if (!result.success) {
        retentionStatus.textContent = `Cleanup failed: ${result.error}`;
        return;
    }

    const report = result.data || {};
    retentionStatus.textContent = `Removed ${report.removedQueueRuns || 0} queue runs, ${report.removedCacheFiles || 0} cache files`;
}

async function runPreflightCheck() {
    preflightSummary.textContent = 'Checking...';
    preflightDetails.textContent = '';
    preflightPanel.style.display = 'block';

    const result = await api.runPreflightCheck();
    if (!result.success) {
        preflightSummary.textContent = 'Failed';
        preflightDetails.textContent = result.error || 'Preflight check failed';
        return;
    }

    const data = result.data || {};
    const free = data.freeDiskBytes;
    const diskOk = data.diskOk;
    preflightSummary.textContent = diskOk === null ? 'Disk check unavailable' : (diskOk ? 'Disk OK' : 'Low disk');
    preflightDetails.textContent = [
        `Setup complete: ${data.setupComplete ? 'Yes' : 'No'}`,
        `Model cache present: ${data.modelCachePresent ? 'Yes' : 'No'}`,
        `Expected download: ${formatBytes(data.expectedDownloadBytes)}`,
        `Estimated required space: ${formatBytes(data.expectedRequiredBytes)}`,
        `Free disk space: ${free === null ? 'Unknown' : formatBytes(free)}`,
        `Estimated time @100 Mbps: ${data.estimatedMinutes?.fast_100mbps ?? 0} min`,
        `Estimated time @30 Mbps: ${data.estimatedMinutes?.typical_30mbps ?? 0} min`,
        `Estimated time @10 Mbps: ${data.estimatedMinutes?.slow_10mbps ?? 0} min`
    ].join('\n');
}

async function exportDiagnosticsBundle() {
    diagnosticsBtn.disabled = true;
    const original = diagnosticsBtn.textContent;
    diagnosticsBtn.textContent = 'Exporting...';
    try {
        const result = await api.exportDiagnostics();
        if (result.success) {
            diagnosticsBtn.textContent = 'Diagnostics Saved';
            showMessage(`Diagnostics exported: ${result.filePath}`, 'success');
        } else if (result.canceled) {
            diagnosticsBtn.textContent = original;
        } else {
            diagnosticsBtn.textContent = original;
            showMessage(`Diagnostics export failed: ${result.error}`, 'error');
        }
    } catch (error) {
        diagnosticsBtn.textContent = original;
        showMessage(`Diagnostics export failed: ${error.message}`, 'error');
    } finally {
        setTimeout(() => {
            diagnosticsBtn.textContent = original;
            diagnosticsBtn.disabled = false;
        }, 1200);
    }
}

async function checkServerStatus() {
    try {
        const result = await api.checkServerStatus();

        if (result.success) {
            serverStatus.textContent = 'Connected';
            serverStatus.className = 'status-value success';

            const modelLoaded = result.data.model_loaded;
            modelStatus.textContent = modelLoaded ? 'Loaded' : 'Not loaded';
            modelStatus.className = `status-value ${modelLoaded ? 'success' : 'warning'}`;

            const preferredDevice = String(result.data.preferred_device || result.data.device_state || 'cpu').toLowerCase();
            const activeDevice = String(result.data.device_state || 'cpu').toLowerCase();
            const effectiveDevice = modelLoaded ? activeDevice : preferredDevice;
            const hasGpuReady = effectiveDevice === 'cuda' || effectiveDevice === 'mps';

            if (modelLoaded && activeDevice === 'cpu' && (preferredDevice === 'cuda' || preferredDevice === 'mps')) {
                gpuStatus.textContent = 'CPU (fallback)';
                gpuStatus.className = 'status-value warning';
            } else if (effectiveDevice === 'cuda') {
                gpuStatus.textContent = modelLoaded ? 'CUDA Active' : 'CUDA Ready';
                gpuStatus.className = 'status-value success';
            } else if (effectiveDevice === 'mps') {
                gpuStatus.textContent = modelLoaded ? 'Apple MPS Active' : 'Apple MPS Ready';
                gpuStatus.className = 'status-value success';
            } else {
                gpuStatus.textContent = hasGpuReady ? 'Available' : 'CPU Only';
                gpuStatus.className = `status-value ${hasGpuReady ? 'success' : 'warning'}`;
            }

            // Update load model button state (but don't change if currently processing)
            if (!isProcessing) {
                if (modelLoaded) {
                    loadModelBtn.disabled = true;
                    loadModelBtn.textContent = 'Model Loaded ✓';
                    loadModelBtn.classList.add('btn-loaded');
                } else {
                    loadModelBtn.disabled = false;
                    loadModelBtn.textContent = 'Load Model';
                    loadModelBtn.classList.remove('btn-loaded');
                }
            }

            // Update OCR button state - allow run with a selected input; model can auto-load on run.
            if (!isProcessing) {
                if (currentImagePath) {
                    ocrBtn.disabled = false;
                } else {
                    ocrBtn.disabled = true;
                }
            }
        } else {
            serverStatus.textContent = 'Disconnected';
            serverStatus.className = 'status-value error';
            modelStatus.textContent = 'Unknown';
            modelStatus.className = 'status-value';
            gpuStatus.textContent = 'Unknown';
            gpuStatus.className = 'status-value';

            // Disable OCR if server disconnected
            ocrBtn.disabled = true;
        }
    } catch (error) {
        console.error('Status check error:', error);
    }
}

async function selectImage() {
    const result = await api.selectImage();

    if (result.success) {
        loadImage(result.filePath);
    }
}

async function loadImage(filePath) {
    currentImagePath = filePath;
    dropZone.classList.remove('is-drag-over');
    const isPdf = isPdfPath(filePath);
    const fileName = api.basename(filePath);

    if (isPdf) {
        imagePreview.src = '';
        imagePreview.style.display = 'none';
        if (selectedFileLabel) {
            selectedFileLabel.textContent = `PDF: ${fileName}`;
            selectedFileLabel.style.display = 'block';
        }
    } else {
        imagePreview.src = filePath;
        imagePreview.style.display = 'block';
        if (selectedFileLabel) {
            selectedFileLabel.textContent = fileName;
            selectedFileLabel.style.display = 'block';
        }
    }

    dropZone.style.display = 'none';
    previewSection.style.display = 'block';

    // Clear previous results
    ocrPreviewImage.removeAttribute('src');
    ocrPreviewImage.style.display = 'none';
    resultsContent.innerHTML = '';
    progressInline.style.display = 'none';
    copyBtn.style.display = 'none';
    downloadZipBtn.style.display = 'none';
    viewBoxesBtn.style.display = 'none';
    viewTokensBtn.style.display = 'none';

    // Clear overlay boxes
    ocrBoxesOverlay.innerHTML = '';
    ocrBoxesOverlay.removeAttribute('viewBox');
    lastBoxCount = 0;

    // Check server status to update OCR button state
    await checkServerStatus();
}

function clearImage() {
    currentImagePath = null;
    dropZone.classList.remove('is-drag-over');
    currentResultText = null;
    currentRawTokens = null;
    currentPromptType = null;
    imagePreview.src = '';
    imagePreview.style.display = 'block';
    if (selectedFileLabel) {
        selectedFileLabel.textContent = '';
        selectedFileLabel.style.display = 'none';
    }

    dropZone.style.display = 'block';
    previewSection.style.display = 'none';
    ocrBtn.disabled = true;
    viewBoxesBtn.style.display = 'none';
    viewTokensBtn.style.display = 'none';

    // Clear results and progress
    ocrPreviewImage.removeAttribute('src');
    ocrPreviewImage.style.display = 'none';
    resultsContent.innerHTML = '';
    progressInline.style.display = 'none';
    copyBtn.style.display = 'none';
    downloadZipBtn.style.display = 'none';

    // Clear overlay boxes
    ocrBoxesOverlay.innerHTML = '';
    ocrBoxesOverlay.removeAttribute('viewBox');
    lastBoxCount = 0;
}

function openLightbox(imageSrc) {
    lightboxImage.src = imageSrc;
    lightboxImage.style.display = 'block';
    lightboxText.style.display = 'none';
    lightbox.style.display = 'block';
}

function openLightboxWithText(text) {
    lightboxText.textContent = text;
    lightboxText.style.display = 'block';
    lightboxImage.style.display = 'none';
    lightbox.style.display = 'block';
}

function closeLightbox() {
    lightbox.style.display = 'none';
}

function viewOriginalImage() {
    if (currentImagePath) {
        if (isPdfPath(currentImagePath)) {
            showMessage('Original preview is not available for PDF files', 'warning');
            return;
        }
        openLightbox(currentImagePath);
    }
}

async function viewBoxesImage() {
    if (!currentImagePath) return;
    if (isPdfPath(currentImagePath)) {
        showMessage('Boxes image view is only available for image files', 'warning');
        return;
    }

    // Create a canvas to render the image with boxes
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Load the original image
    const img = new Image();
    img.src = currentImagePath;

    await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
    });

    // Set canvas size to match image
    canvas.width = img.width;
    canvas.height = img.height;

    // Draw the original image
    ctx.drawImage(img, 0, 0);

    // Parse boxes from current raw tokens
    if (currentRawTokens) {
        const boxes = parseBoxesFromTokens(currentRawTokens, true); // OCR is complete when viewing boxes

        // Helper to convert hex to rgba
        const hexToRgba = (hex, alpha) => {
            const r = parseInt(hex.slice(1, 3), 16);
            const g = parseInt(hex.slice(3, 5), 16);
            const b = parseInt(hex.slice(5, 7), 16);
            return `rgba(${r}, ${g}, ${b}, ${alpha})`;
        };

        // Draw each box
        boxes.forEach((box) => {
            const x1 = (box.x1 / DEEPSEEK_COORD_MAX) * img.width;
            const y1 = (box.y1 / DEEPSEEK_COORD_MAX) * img.height;
            const x2 = (box.x2 / DEEPSEEK_COORD_MAX) * img.width;
            const y2 = (box.y2 / DEEPSEEK_COORD_MAX) * img.height;

            const color = TYPE_COLORS[box.type] || '#FF1493';
            const isUnknownType = !TYPE_COLORS[box.type];

            // Draw semi-transparent fill
            ctx.fillStyle = isUnknownType ? 'rgba(0, 255, 0, 0.3)' : hexToRgba(color, 0.1);
            ctx.fillRect(x1, y1, x2 - x1, y2 - y1);

            // Draw border
            ctx.strokeStyle = color;
            ctx.lineWidth = isUnknownType ? 3 : 2;
            ctx.globalAlpha = 0.9;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.globalAlpha = 1.0;

            // Draw label
            const labelPadding = 4;
            const labelHeight = 18;
            const displayText = box.isType ? box.type : (box.content.length > 30 ? box.content.substring(0, 30) + '...' : box.content);
            ctx.font = isUnknownType ? 'bold 12px system-ui' : '500 12px system-ui';
            const labelWidth = ctx.measureText(displayText).width + labelPadding * 2;
            const labelY = Math.max(0, y1 - labelHeight);

            // Label background
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.95;
            ctx.fillRect(x1, labelY, labelWidth, labelHeight);
            ctx.globalAlpha = 1.0;

            // Label text
            ctx.fillStyle = isUnknownType ? '#00FF00' : 'white';
            ctx.fillText(displayText, x1 + labelPadding, labelY + 13);
        });
    }

    // Convert canvas to image and show in lightbox
    const imageUrl = canvas.toDataURL('image/png');
    openLightbox(imageUrl);
}

function viewRawTokens() {
    if (currentRawTokens) {
        openLightboxWithText(currentRawTokens);
    }
}

function parseBoxesFromTokens(tokenText, isOcrComplete = false) {
    // Extract all bounding boxes from token format: <|ref|>CONTENT<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>
    const boxes = [];
    const refDetRegex = /<\|ref\|>([^<]+)<\|\/ref\|><\|det\|>\[\[([^\]]+)\]\]<\|\/det\|>/g;
    let match;
    const matches = [];

    // First, collect all matches with their positions
    while ((match = refDetRegex.exec(tokenText)) !== null) {
        matches.push({
            content: match[1].trim(),
            coords: match[2],
            matchStart: match.index,
            matchEnd: match.index + match[0].length
        });
    }

    // Now process each match and determine if it's complete
    for (let i = 0; i < matches.length; i++) {
        try {
            const matchData = matches[i];
            const content = matchData.content;

            // Parse the coordinate string "x1,\ny1,\nx2,\ny2" or "x1, y1, x2, y2"
            const coords = matchData.coords.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
            if (coords.length === 4) {
                // Determine if this is a type label or actual text content
                const isType = KNOWN_TYPES.includes(content);

                // Extract the actual text content that comes after this box (for Document mode)
                let textContent = '';
                let isComplete = false;

                if (i < matches.length - 1) {
                    // Not the last box - extract content between this box and the next
                    textContent = tokenText.substring(matchData.matchEnd, matches[i + 1].matchStart).trim();
                    isComplete = textContent.length > 0;
                } else {
                    // Last box - extract everything after it
                    textContent = tokenText.substring(matchData.matchEnd).trim();
                    isComplete = isOcrComplete && textContent.length > 0;
                }

                boxes.push({
                    content: content,
                    textContent: textContent,  // The actual text to copy in Document mode
                    isType: isType,
                    type: isType ? content : 'text',  // Use 'text' as default type for OCR content
                    x1: coords[0],
                    y1: coords[1],
                    x2: coords[2],
                    y2: coords[3],
                    isComplete: isComplete  // Add completion status for Document mode
                });
            }
        } catch (e) {
            console.error('Error parsing box coordinates:', e);
        }
    }

    return boxes;
}

function extractTextFromTokens(tokenText) {
    // Extract just the text content (non-type labels) from tokens
    const boxes = parseBoxesFromTokens(tokenText);
    const textPieces = boxes
        .filter(box => !box.isType)  // Only non-type content
        .map(box => box.content);
    return textPieces.join('\n');  // Join with newlines for readability
}

function renderBoxes(boxes, imageWidth, imageHeight, promptType) {
    if (!imageWidth || !imageHeight || boxes.length === 0) {
        return;
    }

    // Set SVG viewBox to match image dimensions (only once)
    if (!ocrBoxesOverlay.hasAttribute('viewBox')) {
        ocrBoxesOverlay.setAttribute('viewBox', `0 0 ${imageWidth} ${imageHeight}`);
        ocrBoxesOverlay.setAttribute('preserveAspectRatio', 'none');
    }

    // OCR Text and Document modes have interactive boxes
    const isInteractive = promptType === 'ocr' || promptType === 'document';

    // Only add new boxes that haven't been rendered yet
    const newBoxes = boxes.slice(lastBoxCount);

    newBoxes.forEach((box) => {
        // Scale coordinates from 0-999 normalized space to actual image dimensions
        const scaledX1 = (box.x1 / DEEPSEEK_COORD_MAX) * imageWidth;
        const scaledY1 = (box.y1 / DEEPSEEK_COORD_MAX) * imageHeight;
        const scaledX2 = (box.x2 / DEEPSEEK_COORD_MAX) * imageWidth;
        const scaledY2 = (box.y2 / DEEPSEEK_COORD_MAX) * imageHeight;

        // Get color for this box type - use bright pink/green if unknown
        const color = TYPE_COLORS[box.type] || '#FF1493';  // Hot pink for unknown types
        const isUnknownType = !TYPE_COLORS[box.type];

        // Create group for box and label
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('class', 'ocr-box-group');
        group.style.cursor = box.isType ? 'default' : 'pointer';

        // Create semi-transparent fill rectangle
        const fillRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        fillRect.setAttribute('x', scaledX1);
        fillRect.setAttribute('y', scaledY1);
        fillRect.setAttribute('width', scaledX2 - scaledX1);
        fillRect.setAttribute('height', scaledY2 - scaledY1);
        fillRect.setAttribute('fill', isUnknownType ? '#00FF00' : color);  // Lime green for unknown
        fillRect.setAttribute('opacity', isUnknownType ? '0.3' : '0.1');
        fillRect.setAttribute('class', 'ocr-box-fill');

        // Create border rectangle
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', scaledX1);
        rect.setAttribute('y', scaledY1);
        rect.setAttribute('width', scaledX2 - scaledX1);
        rect.setAttribute('height', scaledY2 - scaledY1);
        rect.setAttribute('fill', 'none');
        rect.setAttribute('stroke', color);  // Hot pink border for unknown
        rect.setAttribute('stroke-width', isUnknownType ? '3' : '2');
        rect.setAttribute('opacity', '0.9');
        rect.setAttribute('class', 'ocr-box-border');

        // Create label background
        const labelBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        const labelPadding = 4;
        const labelHeight = 18;
        const displayText = box.isType ? box.type : (box.content.length > 30 ? box.content.substring(0, 30) + '...' : box.content);
        const labelWidth = displayText.length * 7 + labelPadding * 2;

        labelBg.setAttribute('x', scaledX1);
        labelBg.setAttribute('y', Math.max(0, scaledY1 - labelHeight));
        labelBg.setAttribute('width', labelWidth);
        labelBg.setAttribute('height', labelHeight);
        labelBg.setAttribute('fill', color);  // Hot pink background for unknown types
        labelBg.setAttribute('opacity', '0.95');
        labelBg.setAttribute('class', 'ocr-box-label-bg');

        // Create label text
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', scaledX1 + labelPadding);
        label.setAttribute('y', Math.max(0, scaledY1 - labelHeight) + 13);
        label.setAttribute('fill', isUnknownType ? '#00FF00' : 'white');  // Lime green text for unknown
        label.setAttribute('font-size', '12');
        label.setAttribute('font-family', 'system-ui, -apple-system, sans-serif');
        label.setAttribute('font-weight', isUnknownType ? '700' : '500');
        label.setAttribute('class', 'ocr-box-label-text');
        label.textContent = displayText;

        // Add hover and click interactions
        // OCR mode: text content boxes (not type labels) are clickable
        // Document mode: type label boxes with complete text content are clickable
        let isClickable = false;
        let copyText = '';

        if (promptType === 'ocr') {
            // OCR mode: clickable if not a type label
            isClickable = !box.isType && isInteractive;
            copyText = box.content;
        } else if (promptType === 'document') {
            // Document mode: clickable if it's a type label with complete text content
            isClickable = box.isType && box.isComplete && box.textContent && isInteractive;
            copyText = box.textContent;
        }

        if (isClickable) {
            // Enable pointer events
            group.style.pointerEvents = 'all';
            group.style.cursor = 'pointer';

            group.addEventListener('mouseenter', (e) => {
                fillRect.setAttribute('opacity', '0.3');
                rect.setAttribute('stroke-width', isUnknownType ? '4' : '3');
                labelBg.setAttribute('opacity', '1');
                e.stopPropagation();
            });

            group.addEventListener('mouseleave', (e) => {
                fillRect.setAttribute('opacity', isUnknownType ? '0.3' : '0.1');
                rect.setAttribute('stroke-width', isUnknownType ? '3' : '2');
                labelBg.setAttribute('opacity', '0.95');
                e.stopPropagation();
            });

            group.addEventListener('click', async (e) => {
                e.stopPropagation();
                try {
                    await navigator.clipboard.writeText(copyText);
                    // Visual feedback - flash the label
                    const originalBg = labelBg.getAttribute('fill');
                    const originalText = label.textContent;
                    labelBg.setAttribute('fill', '#10B981');  // Green
                    label.textContent = '✓ Copied!';
                    console.log('Copied text:', copyText);
                    setTimeout(() => {
                        labelBg.setAttribute('fill', originalBg);
                        label.textContent = originalText;
                    }, 1000);
                } catch (err) {
                    console.error('Failed to copy text:', err);
                }
            });

            // Add title for tooltip
            const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
            const previewText = copyText.length > 50 ? copyText.substring(0, 50) + '...' : copyText;
            title.textContent = `Click to copy: ${previewText}`;
            group.appendChild(title);
        } else if (box.isType && promptType === 'document' && !box.isComplete) {
            // Document mode incomplete boxes - show as non-clickable but with visual feedback
            group.style.pointerEvents = 'none';
            group.style.cursor = 'default';
            group.style.opacity = '0.6';  // Dimmed to show it's not ready yet
        } else {
            // Non-interactive boxes
            group.style.pointerEvents = 'none';
        }

        // Add animation for new boxes
        group.style.animation = 'fadeIn 0.3s ease-in';

        group.appendChild(fillRect);
        group.appendChild(rect);
        group.appendChild(labelBg);
        group.appendChild(label);

        ocrBoxesOverlay.appendChild(group);
    });

    // Update the count of rendered boxes
    lastBoxCount = boxes.length;
}

async function loadModel() {
    if (isProcessing) return;

    try {
        isProcessing = true;
        loadModelBtn.disabled = true;
        loadModelBtn.textContent = 'Loading Model...';

        // Show inline progress indicator
        progressInline.style.display = 'flex';
        progressStatus.textContent = 'Loading model...';

        // Trigger model loading
        const result = await api.loadModel();
        if (!result.success) {
            progressInline.style.display = 'none';
            showMessage(`Failed to load model: ${result.error}`, 'error');
            await checkServerStatus();
            return;
        }

        // Wait for completion with a single polling loop (reduces duplicate requests).
        const finalProgress = await waitForModelLoadCompletion();

        // Hide progress indicator
        progressInline.style.display = 'none';

        if (finalProgress.status === 'loaded') {
            showMessage('Model loaded successfully!', 'success');
        } else if (finalProgress.status === 'error') {
            showMessage(`Model loading failed: ${finalProgress.message || 'Unknown error'}`, 'error');
        } else if (finalProgress.status === 'timeout') {
            showMessage('Model loading is taking longer than expected. Check logs and try again.', 'warning');
        } else {
            showMessage('Model loading state is unknown. Please check server status.', 'warning');
        }
        await checkServerStatus();
    } catch (error) {
        progressInline.style.display = 'none';
        showMessage(`Error: ${error.message}`, 'error');
        await checkServerStatus(); // Update button state even on error
    } finally {
        isProcessing = false;
        // Don't reset button state here - let checkServerStatus() handle it
    }
}

async function performOCR() {
    if (!currentImagePath || isProcessing) return;

    let tokenPollInterval = null;
    let imageNaturalWidth = 0;
    let imageNaturalHeight = 0;

    try {
        isProcessing = true;
        ocrBtn.disabled = true;
        ocrBtnText.textContent = 'Preparing...';

        // Store current prompt type
        currentPromptType = promptType.value;

        // Show progress in header
        progressInline.style.display = 'flex';
        progressStatus.textContent = 'Preparing OCR...';

        // Auto-load model when needed before OCR request starts.
        await ensureModelLoadedForRun();

        ocrBtnText.textContent = 'Processing...';
        progressStatus.textContent = 'Starting OCR...';
        const ocrStartedAt = Date.now();

        // Clear panels
        resultsContent.innerHTML = '';
        copyBtn.style.display = 'none';

        // Reset box tracking
        lastBoxCount = 0;
        ocrBoxesOverlay.innerHTML = '';
        ocrBoxesOverlay.removeAttribute('viewBox');

        // Load image into preview and get dimensions
        ocrPreviewImage.style.display = 'none';
        await new Promise((resolve) => {
            let settled = false;
            const finalize = () => {
                imageNaturalWidth = ocrPreviewImage.naturalWidth;
                imageNaturalHeight = ocrPreviewImage.naturalHeight;
                console.log(`Image dimensions: ${imageNaturalWidth}×${imageNaturalHeight}`);
                if (imageNaturalWidth > 0 && imageNaturalHeight > 0) {
                    ocrPreviewImage.style.display = 'block';
                }
                if (!settled) {
                    settled = true;
                    resolve();
                }
            };
            ocrPreviewImage.onload = finalize;
            ocrPreviewImage.onerror = finalize;
            ocrPreviewImage.src = currentImagePath;
            setTimeout(finalize, 2000);
        });

        // Poll for token count and raw token stream updates
        tokenPollInterval = setInterval(async () => {
            try {
                const response = await fetch('http://127.0.0.1:5000/progress');
                const data = await response.json();

                if (data.status === 'loading') {
                    const percent = clampPercent(data.progress_percent || 0);
                    const stageText = data.message || data.stage || 'Loading model...';
                    progressStatus.textContent = `Loading model ${percent}% - ${stageText}`;
                    return;
                }

                if (data.status === 'processing') {
                    const elapsedSeconds = Math.max(1, Math.floor((Date.now() - ocrStartedAt) / 1000));
                    if (data.chars_generated > 0) {
                        progressStatus.textContent = `${data.chars_generated} characters generated (${elapsedSeconds}s)`;
                    } else if (data.message) {
                        const hasElapsedInMessage = /\belapsed\b/i.test(data.message);
                        progressStatus.textContent = hasElapsedInMessage
                            ? data.message
                            : `${data.message} (${elapsedSeconds}s)`;
                    } else {
                        progressStatus.textContent = `Processing OCR... (${elapsedSeconds}s)`;
                    }

                    // Parse and render boxes from raw token stream
                    if (data.raw_token_stream) {
                        const boxes = parseBoxesFromTokens(data.raw_token_stream, false); // Still streaming, not complete
                        renderBoxes(boxes, imageNaturalWidth, imageNaturalHeight, currentPromptType);

                        // Update text panel in real-time
                        if (currentPromptType === 'ocr') {
                            // OCR mode: show extracted text
                            const extractedText = extractTextFromTokens(data.raw_token_stream);
                            if (extractedText) {
                                resultsContent.textContent = extractedText;
                            }
                        } else if (currentPromptType === 'document') {
                            // Document mode: show raw markdown (will be rendered later)
                            resultsContent.textContent = data.raw_token_stream;
                        } else {
                            // Free OCR, Figure, Describe modes: show raw tokens streaming
                            resultsContent.textContent = data.raw_token_stream;
                        }
                    }
                }
            } catch (error) {
                const elapsedSeconds = Math.max(1, Math.floor((Date.now() - ocrStartedAt) / 1000));
                progressStatus.textContent = `Processing OCR... (${elapsedSeconds}s)`;
            }
        }, 700); // Poll less frequently to reduce backend contention during long OCR runs

        const pageRange = isPdfPath(currentImagePath) ? sanitizePageRangeInput(pdfPageRange?.value || '') : '';
        const result = await api.performOCR({
            imagePath: currentImagePath,
            promptType: promptType.value,
            baseSize: parseInt(baseSize.value),
            imageSize: parseInt(imageSize.value),
            cropMode: cropMode.checked,
            pdfPageRange: pageRange
        });

        // Stop polling
        if (tokenPollInterval) {
            clearInterval(tokenPollInterval);
            tokenPollInterval = null;
        }

        if (result.success) {
            // Hide progress spinner
            progressInline.style.display = 'none';

            // Store raw tokens
            currentRawTokens = result.data.raw_tokens;

            // Do a final render of all boxes with the complete token stream
            // This ensures any boxes that arrived after polling stopped are rendered
            if (currentRawTokens && imageNaturalWidth && imageNaturalHeight) {
                const boxes = parseBoxesFromTokens(currentRawTokens, true); // OCR is complete
                // Reset lastBoxCount to 0 to force re-render of all boxes
                lastBoxCount = 0;
                ocrBoxesOverlay.innerHTML = '';
                renderBoxes(boxes, imageNaturalWidth, imageNaturalHeight, currentPromptType);
            }

            // Display results based on mode
            if (result.data.prompt_type === 'ocr') {
                // OCR Text mode: extract and show just the text
                const extractedText = currentRawTokens ? extractTextFromTokens(currentRawTokens) : result.data.result;
                resultsContent.textContent = extractedText;
                currentResultText = extractedText;
            } else if (result.data.prompt_type === 'document') {
                // Document mode: render markdown
                displayResults(result.data.result, result.data.prompt_type);
                currentResultText = result.data.result;
            } else {
                // Free OCR, Figure, Describe modes: show raw tokens
                const rawText = currentRawTokens || result.data.result;
                resultsContent.textContent = rawText;
                currentResultText = rawText;
            }

            // Always show copy button when we have results
            copyBtn.style.display = 'inline-block';

            // Show download zip button only for document mode
            if (currentPromptType === 'document') {
                downloadZipBtn.style.display = 'inline-block';
            } else {
                downloadZipBtn.style.display = 'none';
            }

            // Show raw tokens button and boxes button if raw tokens exist
            if (currentRawTokens && uiMode === 'advanced') {
                viewTokensBtn.style.display = 'inline-block';
                viewBoxesBtn.style.display = 'inline-block';
            } else {
                viewTokensBtn.style.display = 'none';
                viewBoxesBtn.style.display = 'none';
            }

            showMessage('OCR completed successfully!', 'success');
        } else {
            // Error handling
            ocrBoxesOverlay.innerHTML = '';
            ocrBoxesOverlay.removeAttribute('viewBox');
            lastBoxCount = 0;

            ocrPreviewImage.removeAttribute('src');
            ocrPreviewImage.style.display = 'none';
            progressInline.style.display = 'none';
            resultsContent.innerHTML = `<p class="error">Error: ${result.error}</p>`;
            copyBtn.style.display = 'none';
            downloadZipBtn.style.display = 'none';
            viewBoxesBtn.style.display = 'none';
            viewTokensBtn.style.display = 'none';
            showMessage(`OCR failed: ${result.error}`, 'error');
        }
    } catch (error) {
        if (tokenPollInterval) {
            clearInterval(tokenPollInterval);
        }
        ocrBoxesOverlay.innerHTML = '';
        ocrBoxesOverlay.removeAttribute('viewBox');
        lastBoxCount = 0;
        ocrPreviewImage.removeAttribute('src');
        ocrPreviewImage.style.display = 'none';
        progressInline.style.display = 'none';
        resultsContent.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        copyBtn.style.display = 'none';
        downloadZipBtn.style.display = 'none';
        viewBoxesBtn.style.display = 'none';
        viewTokensBtn.style.display = 'none';
        showMessage(`Error: ${error.message}`, 'error');
    } finally {
        if (tokenPollInterval) {
            clearInterval(tokenPollInterval);
        }
        isProcessing = false;
        ocrBtnText.textContent = 'Run OCR';
        // Check server status to properly set button state based on model loaded status
        await checkServerStatus();
    }
}

function displayResults(result, promptType) {
    // Format the result nicely
    let formattedResult = '';

    if (typeof result === 'string') {
        formattedResult = result;
    } else if (typeof result === 'object') {
        formattedResult = JSON.stringify(result, null, 2);
    } else {
        formattedResult = String(result);
    }

    // Store original text for copying (with relative paths)
    currentResultText = formattedResult;

    // Render markdown for document mode
    if (promptType === 'document') {
        const cacheBuster = Date.now();
        const renderedMarkdown = formattedResult.replace(
            /!\[([^\]]*)\]\(images\/([^)]+)\)/g,
            `![$1](http://127.0.0.1:5000/outputs/images/$2?t=${cacheBuster})`
        );
        resultsContent.innerHTML = api.renderMarkdown(renderedMarkdown);
    } else {
        resultsContent.textContent = formattedResult;
    }
}

function copyResults() {
    // Use the original text (markdown) instead of rendered HTML
    const text = currentResultText || resultsContent.textContent;

    navigator.clipboard.writeText(text).then(() => {
        const originalText = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        setTimeout(() => {
            copyBtn.textContent = originalText;
        }, 2000);
    }).catch(err => {
        showMessage('Failed to copy to clipboard', 'error');
    });
}

async function downloadZip() {
    if (!currentResultText || currentPromptType !== 'document') {
        showMessage('No document to download', 'error');
        return;
    }

    try {
        const originalText = downloadZipBtn.textContent;
        downloadZipBtn.textContent = 'Saving ZIP...';
        downloadZipBtn.disabled = true;

        const result = await api.saveDocumentZip(currentResultText);
        if (result.success) {
            downloadZipBtn.textContent = 'Saved';
            showMessage('ZIP file saved successfully', 'success');
        } else if (result.canceled) {
            downloadZipBtn.textContent = originalText;
        } else {
            downloadZipBtn.textContent = originalText;
            showMessage(`Failed to save ZIP: ${result.error || 'Unknown error'}`, 'error');
        }

        setTimeout(() => {
            downloadZipBtn.textContent = originalText;
            downloadZipBtn.disabled = false;
        }, 1200);
    } catch (error) {
        console.error('Error saving ZIP:', error);
        showMessage('Failed to save ZIP file', 'error');
        downloadZipBtn.textContent = 'Download ZIP';
        downloadZipBtn.disabled = false;
    }
}

function showMessage(message, type = 'info') {
    console.log(`[${type.toUpperCase()}] ${message}`);
    if (!messageBanner) {
        return;
    }

    if (messageHideTimer) {
        clearTimeout(messageHideTimer);
        messageHideTimer = null;
    }

    messageBanner.hidden = false;
    messageBanner.className = `message-banner ${type}`;
    messageBanner.textContent = message;

    messageHideTimer = setTimeout(() => {
        messageBanner.hidden = true;
        messageBanner.textContent = '';
    }, type === 'error' ? 6500 : 4200);
}

// ============= Queue Management Functions =============

async function addFilesToQueue() {
    const result = await api.selectImages();
    
    if (result.success && result.filePaths.length > 0) {
        await addToQueue(result.filePaths);
    }
}

async function addFolderToQueue() {
    const result = await api.selectFolder();
    
    if (result.success) {
        const listResult = await api.listFolderInputs(result.folderPath);
        if (!listResult.success) {
            showMessage(`Error reading folder: ${listResult.error}`, 'error');
            return;
        }

        if (listResult.filePaths.length > 0) {
            await addToQueue(listResult.filePaths);
        } else {
            showMessage('No image/PDF files found in folder', 'warning');
        }
    }
}

async function addToQueue(filePaths) {
    try {
        const pageRange = sanitizePageRangeInput(pdfPageRange?.value || '');
        const result = await api.addToQueue({
            filePaths,
            promptType: promptType.value,
            baseSize: parseInt(baseSize.value),
            imageSize: parseInt(imageSize.value),
            cropMode: cropMode.checked,
            pdfPageRange: pageRange
        });
        
        if (result.success) {
            showMessage(`Added ${result.data.added} file(s) to queue`, 'success');
            await updateQueueDisplay();
        } else {
            showMessage(`Failed to add files to queue: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error adding files to queue: ${error.message}`, 'error');
    }
}

async function updateQueueDisplay() {
    try {
        const result = await api.getQueueStatus();
        
        if (result.success) {
            const queueData = result.data;
            queueItems = queueData.items || [];
            const queueIsActive = Boolean(queueData.is_processing || queueData.processing > 0);
            const queuePaused = Boolean(queueData.paused);
            if (queueData.cancel_requested) {
                progressStatus.textContent = 'Cancellation requested...';
            }
            
            // Update queue count
            queueCount.textContent = queueData.total;
            
            // Update queue list
            if (queueItems.length === 0) {
                queueList.innerHTML = '<p class="queue-empty">Queue is empty. Add files to begin.</p>';
                processQueueBtn.disabled = true;
                clearQueueBtn.disabled = true;
            } else {
                queueList.innerHTML = '';
                queueItems.forEach(item => {
                    const itemEl = createQueueItemElement(item);
                    queueList.appendChild(itemEl);
                });
                processQueueBtn.disabled = queueIsActive || queueData.pending === 0;
                clearQueueBtn.disabled = queueIsActive;
            }

            pauseQueueBtn.disabled = !queueIsActive || queuePaused;
            resumeQueueBtn.disabled = !queueIsActive || !queuePaused;
            cancelQueueBtn.disabled = !queueIsActive;
            retryFailedBtn.disabled = queueIsActive || !queueData.can_retry_failed;
            resumeQueueBtn.style.display = queuePaused ? 'inline-block' : 'none';
            pauseQueueBtn.style.display = queuePaused ? 'none' : 'inline-block';
            addFilesBtn.disabled = queueIsActive;
            addFolderBtn.disabled = queueIsActive;
            
            // Update progress
            const totalFiles = queueData.total;
            const completedFiles = queueData.completed + queueData.failed;
            const processingProgress = queueItems
                .filter((item) => item.status === 'processing')
                .reduce((sum, item) => sum + (clampPercent(item.progress) / 100), 0);
            const weightedCompleted = completedFiles + processingProgress;
            const activeItem = queueItems.find((item) => item.status === 'processing');
            const activeText = activeItem ? ` (active ${clampPercent(activeItem.progress)}%)` : '';
            const pausedText = queuePaused ? ' (paused)' : '';
            queueProgressText.textContent = `${completedFiles}/${totalFiles}${activeText}${pausedText}`;
            
            if (totalFiles > 0) {
                const progressPercent = (weightedCompleted / totalFiles) * 100;
                setProgressBarWidth(queueProgressBar, progressPercent);
            } else {
                setProgressBarWidth(queueProgressBar, 0);
            }
        }
    } catch (error) {
        console.error('Error updating queue display:', error);
    }
}

function createQueueItemElement(item) {
    const div = document.createElement('div');
    div.className = `queue-item ${item.status}`;
    div.setAttribute('data-id', item.id);
    
    const info = document.createElement('div');
    info.className = 'queue-item-info';
    
    const filename = document.createElement('div');
    filename.className = 'queue-item-filename';
    filename.textContent = item.filename;
    
    const status = document.createElement('div');
    status.className = 'queue-item-status';
    
    let statusText = '';
    if (item.status === 'pending') {
        statusText = 'Waiting...';
    } else if (item.status === 'processing') {
        const detail = item.progress_detail ? ` (${item.progress_detail})` : '';
        statusText = `Processing... ${item.progress}%${detail}`;
    } else if (item.status === 'completed') {
        statusText = 'Completed ✓';
    } else if (item.status === 'failed') {
        statusText = `Failed: ${item.error || 'Unknown error'}`;
    }
    
    status.innerHTML = `<span class="status-badge ${item.status}">${statusText}</span>`;
    
    info.appendChild(filename);
    info.appendChild(status);
    
    const actions = document.createElement('div');
    actions.className = 'queue-item-actions';
    
    if (item.status === 'pending') {
        const removeBtn = document.createElement('button');
        removeBtn.className = 'queue-item-remove';
        removeBtn.textContent = 'Remove';
        removeBtn.onclick = () => removeFromQueue(item.id);
        actions.appendChild(removeBtn);
    }
    
    div.appendChild(info);
    div.appendChild(actions);
    
    return div;
}

async function processQueue() {
    if (isQueueProcessing) return;
    
    try {
        isQueueProcessing = true;
        processQueueBtn.disabled = true;
        processQueueBtn.textContent = 'Processing...';
        pauseQueueBtn.disabled = false;
        cancelQueueBtn.disabled = false;
        
        // Reset preview for queue processing
        ocrPreviewImage.removeAttribute('src');
        ocrPreviewImage.style.display = 'none';
        ocrBoxesOverlay.innerHTML = '';
        ocrBoxesOverlay.removeAttribute('viewBox');
        lastBoxCount = 0;
        lastProcessedFileId = null;  // Reset file tracking
        lastQueuePreviewImagePath = null;
        resultsContent.innerHTML = '<p>Queue processing... watch here for real-time preview</p>';
        
        // Start polling for queue status updates
        startQueuePolling();
        
        // Show progress
        progressInline.style.display = 'flex';
        progressStatus.textContent = 'Starting queue processing (model will auto-load if needed)...';
        
        // Start queue processing
        const result = await api.processQueue();
        
        // Stop polling
        stopQueuePolling();
        
        if (result.success) {
            currentQueueFolder = result.data.queue_folder;
            const canceledNotice = result.data.canceled
                ? '<p style="margin: 5px 0; color: #9a3412;"><strong>Canceled:</strong> Yes (remaining pending items were left in queue)</p>'
                : '';
            
            // Show completion message in results panel
            const completedMsg = `
                <div style="padding: 20px; background: #f0fdf4; border: 2px solid #10b981; border-radius: 8px; margin: 20px 0;">
                    <h3 style="color: #065f46; margin: 0 0 10px 0;">✓ Queue Processing Complete!</h3>
                    <p style="margin: 5px 0;"><strong>Total Files:</strong> ${result.data.total}</p>
                    <p style="margin: 5px 0;"><strong>Completed:</strong> ${result.data.completed}</p>
                    <p style="margin: 5px 0;"><strong>Failed:</strong> ${result.data.failed}</p>
                    ${canceledNotice}
                    <p style="margin: 15px 0 5px 0;"><strong>Results Location:</strong></p>
                    <code style="background: #dcfce7; padding: 8px; border-radius: 4px; display: block; word-break: break-all; font-size: 12px;">${currentQueueFolder}</code>
                    <p style="margin: 15px 0 0 0; font-size: 14px;">Click "Open Output Folder" button to view all results →</p>
                </div>
            `;
            resultsContent.innerHTML = completedMsg;
            
            // Show open folder button
            openOutputBtn.style.display = 'inline-block';
            progressInline.style.display = 'none';
            
            // Update display
            await updateQueueDisplay();
        } else {
            progressInline.style.display = 'none';
            showMessage(`Queue processing failed: ${result.error}`, 'error');
        }
    } catch (error) {
        progressInline.style.display = 'none';
        stopQueuePolling();
        showMessage(`Error processing queue: ${error.message}`, 'error');
    } finally {
        isQueueProcessing = false;
        processQueueBtn.textContent = 'Process Queue';
        pauseQueueBtn.disabled = true;
        resumeQueueBtn.disabled = true;
        cancelQueueBtn.disabled = true;
        await updateQueueDisplay();
    }
}

// Poll queue status during processing
let queueStatusInterval = null;

function startQueuePolling() {
    if (queueStatusInterval) return;
    
    queueStatusInterval = setInterval(async () => {
        if (isQueueProcessing) {
            await updateQueueDisplay();
            
            // Update progress status and preview for current file
            try {
                const progressResult = await fetch('http://127.0.0.1:5000/progress');
                const progressData = await progressResult.json();
                
                if (progressData.status === 'loading') {
                    // Model is being loaded
                    const percent = progressData.progress_percent || 0;
                    progressStatus.textContent = `Loading model ${percent}% before processing queue...`;
                } else if (progressData.status === 'processing') {
                    // Update preview with current file if available
                    const queueStatusResult = await api.getQueueStatus();
                    if (!queueStatusResult.success) {
                        throw new Error(queueStatusResult.error || 'Failed to fetch queue status');
                    }
                    const queueData = queueStatusResult.data;

                    if (progressData.message) {
                        progressStatus.textContent = progressData.message;
                    } else if (queueData.current_file) {
                        const detail = queueData.current_file.progress_detail ? ` (${queueData.current_file.progress_detail})` : '';
                        progressStatus.textContent = `Processing ${queueData.current_file.filename}${detail}`;
                    } else {
                        progressStatus.textContent = 'Processing...';
                    }
                    
                    if (queueData.current_file && queueData.current_file.image_path) {
                        const previewPath = queueData.current_file.image_path;
                        const fileChanged = lastProcessedFileId !== queueData.current_file.id;
                        const pageChanged = lastQueuePreviewImagePath !== previewPath;

                        // Check if we switched to a new file
                        if (fileChanged || pageChanged) {
                            lastProcessedFileId = queueData.current_file.id;
                            lastQueuePreviewImagePath = previewPath;
                            
                            // Reset boxes for new file
                            ocrBoxesOverlay.innerHTML = '';
                            ocrBoxesOverlay.removeAttribute('viewBox');
                            lastBoxCount = 0;
                            const detail = queueData.current_file.progress_detail ? ` (${queueData.current_file.progress_detail})` : '';
                            resultsContent.innerHTML = `<p>Processing ${queueData.current_file.filename}${detail}...</p>`;
                            
                            // Load new file image
                            ocrPreviewImage.style.display = 'none';

                            // Wait for image to load
                            await new Promise((resolve) => {
                                let settled = false;
                                const finalize = () => {
                                    if (ocrPreviewImage.naturalWidth > 0 && ocrPreviewImage.naturalHeight > 0) {
                                        ocrPreviewImage.style.display = 'block';
                                    }
                                    if (!settled) {
                                        settled = true;
                                        resolve();
                                    }
                                };
                                ocrPreviewImage.onload = finalize;
                                ocrPreviewImage.onerror = finalize;
                                ocrPreviewImage.src = previewPath;
                                setTimeout(finalize, 1000);
                            });
                        }
                        
                        // Render boxes in real-time if we have raw tokens
                        if (progressData.raw_token_stream && ocrPreviewImage.naturalWidth > 0) {
                            const boxes = parseBoxesFromTokens(progressData.raw_token_stream, false);
                            renderBoxes(boxes, ocrPreviewImage.naturalWidth, ocrPreviewImage.naturalHeight, promptType.value);
                            
                            // Also show text in real-time
                            if (promptType.value === 'ocr') {
                                const extractedText = extractTextFromTokens(progressData.raw_token_stream);
                                if (extractedText) {
                                    resultsContent.textContent = extractedText;
                                }
                            } else if (promptType.value === 'document') {
                                resultsContent.textContent = progressData.raw_token_stream;
                            } else {
                                resultsContent.textContent = progressData.raw_token_stream;
                            }
                        }
                    }
                }
            } catch (err) {
                // Ignore polling errors
                console.error('Polling error:', err);
            }
        } else {
            stopQueuePolling();
        }
    }, 500); // Poll every 500ms for smoother updates
}

function stopQueuePolling() {
    if (queueStatusInterval) {
        clearInterval(queueStatusInterval);
        queueStatusInterval = null;
    }
}

async function pauseQueue() {
    const result = await api.pauseQueue();
    if (!result.success) {
        showMessage(`Failed to pause queue: ${result.error}`, 'error');
        return;
    }
    progressStatus.textContent = 'Queue paused';
    await updateQueueDisplay();
}

async function resumeQueue() {
    const result = await api.resumeQueue();
    if (!result.success) {
        showMessage(`Failed to resume queue: ${result.error}`, 'error');
        return;
    }
    progressStatus.textContent = 'Queue resumed';
    await updateQueueDisplay();
}

async function cancelQueue() {
    if (!confirm('Cancel queue processing after the current step?')) {
        return;
    }
    const result = await api.cancelQueue();
    if (!result.success) {
        showMessage(`Failed to cancel queue: ${result.error}`, 'error');
        return;
    }
    progressStatus.textContent = 'Queue cancellation requested...';
    await updateQueueDisplay();
}

async function retryFailedQueueItems() {
    const result = await api.retryFailedQueue();
    if (!result.success) {
        showMessage(`Retry failed: ${result.error}`, 'error');
        return;
    }
    const retriedCount = Number(result.data?.retried || 0);
    showMessage(`Moved ${retriedCount} failed item(s) back to pending`, 'success');
    await updateQueueDisplay();
}

async function clearQueue() {
    if (isQueueProcessing) return;
    
    if (!confirm('Are you sure you want to clear the entire queue?')) {
        return;
    }
    
    try {
        const result = await api.clearQueue();
        
        if (result.success) {
            showMessage('Queue cleared', 'success');
            await updateQueueDisplay();
        } else {
            showMessage(`Failed to clear queue: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error clearing queue: ${error.message}`, 'error');
    }
}

async function removeFromQueue(itemId) {
    try {
        const result = await api.removeFromQueue(itemId);
        
        if (result.success) {
            await updateQueueDisplay();
        } else {
            showMessage(`Failed to remove item: ${result.error}`, 'error');
        }
    } catch (error) {
        showMessage(`Error removing item: ${error.message}`, 'error');
    }
}

async function openOutputFolder() {
    if (currentQueueFolder) {
        await api.openFolder(currentQueueFolder);
    }
}

// Poll queue status periodically when not processing
setInterval(() => {
    if (!isQueueProcessing && queueItems.length > 0) {
        updateQueueDisplay();
    }
}, 5000); // Check every 5 seconds
