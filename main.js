const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const crypto = require('crypto');
const axios = require('axios');

let mainWindow;
let pythonProcess;
const PYTHON_SERVER_PORT = 5000;
const PYTHON_SERVER_URL = `http://127.0.0.1:${PYTHON_SERVER_PORT}`;
const TORCH_PRIMARY_PACKAGES = ['torch>=2.6.0', 'torchvision>=0.21.0', 'torchaudio>=2.6.0'];
const TORCH_FALLBACK_PACKAGES = ['torch>=2.4.0', 'torchvision>=0.19.0', 'torchaudio>=2.4.0'];
const INPUT_MIME_TYPES = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.bmp': 'image/bmp',
  '.webp': 'image/webp',
  '.pdf': 'application/pdf'
};
let startupStatus = {
  phase: 'booting',
  message: 'Starting application...',
  progress: 0,
  state: 'running',
  updatedAt: Date.now()
};

function emitStartupStatus() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('startup-status', startupStatus);
  }
}

function setStartupStatus(phase, message, progress, state = 'running') {
  startupStatus = {
    phase,
    message,
    progress,
    state,
    updatedAt: Date.now()
  };
  console.log(`[startup] ${phase} ${progress}% ${message}`);
  emitStartupStatus();
}

function getVenvPythonPath(venvDir) {
  if (process.platform === 'win32') {
    return path.join(venvDir, 'Scripts', 'python.exe');
  }
  return path.join(venvDir, 'bin', 'python');
}

function getMimeTypeForFile(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  return INPUT_MIME_TYPES[ext] || 'application/octet-stream';
}

function getAppPaths() {
  const runtimeRoot = app.isPackaged ? process.resourcesPath : __dirname;
  const userDataRoot = app.getPath('userData');
  const pythonEnvRoot = path.join(userDataRoot, 'python_env');
  const cacheRoot = path.join(userDataRoot, 'cache');
  const huggingFaceRoot = path.join(pythonEnvRoot, 'huggingface');

  const venvDir = path.join(pythonEnvRoot, 'venv');

  return {
    runtimeRoot,
    backendScript: path.join(runtimeRoot, 'backend', 'ocr_server.py'),
    uvBinary: path.join(runtimeRoot, 'runtime', process.platform === 'win32' ? 'uv.exe' : 'uv'),
    requirementsFile: path.join(runtimeRoot, 'requirements.txt'),
    pythonEnvRoot,
    pythonInstallDir: path.join(pythonEnvRoot, 'python'),
    venvDir,
    venvPython: getVenvPythonPath(venvDir),
    setupMarkerPath: path.join(pythonEnvRoot, '.setup-complete.json'),
    uvCacheDir: path.join(pythonEnvRoot, 'uv-cache'),
    pipCacheDir: path.join(pythonEnvRoot, 'pip-cache'),
    cacheRoot,
    huggingFaceRoot
  };
}

function ensureDirectory(dirPath) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function readJsonFile(filePath) {
  if (!fs.existsSync(filePath)) {
    return null;
  }
  try {
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  } catch (error) {
    return null;
  }
}

function writeJsonFile(filePath, value) {
  ensureDirectory(path.dirname(filePath));
  fs.writeFileSync(filePath, JSON.stringify(value, null, 2), 'utf-8');
}

function sha256File(filePath) {
  const hash = crypto.createHash('sha256');
  hash.update(fs.readFileSync(filePath));
  return hash.digest('hex');
}

function parseRequirementsFile(requirementsFile) {
  const lines = fs.readFileSync(requirementsFile, 'utf-8').split(/\r?\n/);
  const requirements = [];
  for (const line of lines) {
    const stripped = line.split('#', 1)[0].trim();
    if (stripped) {
      requirements.push(stripped);
    }
  }
  return requirements;
}

function removeTorchRequirements(requirements) {
  return requirements.filter((value) => {
    const lower = value.toLowerCase();
    return !lower.startsWith('torch') && !lower.startsWith('torchvision') && !lower.startsWith('torchaudio');
  });
}

function getUvSetupEnv(paths) {
  return {
    ...process.env,
    UV_CACHE_DIR: paths.uvCacheDir,
    PIP_CACHE_DIR: paths.pipCacheDir
  };
}

function resolveUvBinaryPath(paths) {
  if (!fs.existsSync(paths.uvBinary)) {
    throw new Error(
      `Bundled uv binary not found at ${paths.uvBinary}. Run scripts/download-uv for this platform before building.`
    );
  }
  if (process.platform !== 'win32') {
    try {
      fs.chmodSync(paths.uvBinary, 0o755);
    } catch (error) {
      // Best effort: if chmod fails, spawn may still work depending on file mode.
    }
  }
  return paths.uvBinary;
}

function runCommand(command, args, options = {}) {
  const { env = process.env, cwd = undefined, logPrefix = '' } = options;
  const maxCapture = 20000;
  const appendCaptured = (current, chunk) => {
    const combined = current + chunk;
    if (combined.length <= maxCapture) {
      return combined;
    }
    return combined.slice(combined.length - maxCapture);
  };

  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      env,
      cwd,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      const text = data.toString();
      stdout = appendCaptured(stdout, text);
      if (logPrefix) {
        for (const line of text.split(/\r?\n/)) {
          if (line) {
            console.log(`${logPrefix}${line}`);
          }
        }
      } else {
        process.stdout.write(text);
      }
    });

    child.stderr.on('data', (data) => {
      const text = data.toString();
      stderr = appendCaptured(stderr, text);
      if (logPrefix) {
        for (const line of text.split(/\r?\n/)) {
          if (line) {
            console.log(`${logPrefix}${line}`);
          }
        }
      } else {
        process.stderr.write(text);
      }
    });

    child.on('error', (error) => {
      reject(new Error(`Command failed to start: ${command} ${args.join(' ')} (${error.message})`));
    });

    child.on('close', (code) => {
      if (code !== 0) {
        reject(
          new Error(
            `Command failed (${code}): ${command} ${args.join(' ')}\n` +
            `${stderr || stdout}`.trim()
          )
        );
        return;
      }
      resolve({ stdout, stderr });
    });
  });
}

function findStandalonePython(pythonInstallDir) {
  if (!fs.existsSync(pythonInstallDir)) {
    throw new Error(`Python install directory not found: ${pythonInstallDir}`);
  }

  const candidates = fs
    .readdirSync(pythonInstallDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && entry.name.startsWith('cpython-'))
    .map((entry) => path.join(pythonInstallDir, entry.name))
    .sort()
    .reverse();

  for (const installRoot of candidates) {
    const binaryCandidates = process.platform === 'win32'
      ? [path.join(installRoot, 'python.exe')]
      : [path.join(installRoot, 'bin', 'python3'), path.join(installRoot, 'bin', 'python')];

    for (const binaryPath of binaryCandidates) {
      if (fs.existsSync(binaryPath)) {
        return binaryPath;
      }
    }
  }

  throw new Error(`Could not locate standalone Python under ${pythonInstallDir}`);
}

function detectGpuTarget() {
  if (process.platform === 'darwin' && process.arch === 'arm64') {
    return {
      id: 'mps',
      displayName: 'Apple Silicon (MPS)',
      torchIndexUrl: null
    };
  }

  const nvidiaSmi = spawnSync(
    'nvidia-smi',
    ['--query-gpu=name,compute_cap', '--format=csv,noheader'],
    { windowsHide: true, encoding: 'utf-8' }
  );

  if (nvidiaSmi.status === 0 && nvidiaSmi.stdout) {
    const firstLine = nvidiaSmi.stdout
      .split(/\r?\n/)
      .map((line) => line.trim())
      .find((line) => line.length > 0);

    if (firstLine) {
      const [gpuNameRaw, computeCapRaw] = firstLine.split(',').map((value) => value.trim());
      const gpuName = gpuNameRaw || 'NVIDIA GPU';
      const major = Number.parseInt((computeCapRaw || '').split('.', 1)[0], 10);
      const cudaVariant = Number.isFinite(major) && major < 5 ? 'cu118' : 'cu124';
      return {
        id: `cuda-${cudaVariant}`,
        displayName: `${gpuName} (${cudaVariant})`,
        torchIndexUrl: `https://download.pytorch.org/whl/${cudaVariant}`
      };
    }
  }

  if (process.platform === 'darwin') {
    return {
      id: 'cpu',
      displayName: 'CPU (macOS)',
      torchIndexUrl: null
    };
  }

  return {
    id: 'cpu',
    displayName: 'CPU',
    torchIndexUrl: 'https://download.pytorch.org/whl/cpu'
  };
}

async function installTorchWithFallback(uvBinary, venvPython, gpuTarget, setupEnv) {
  const baseArgs = ['pip', 'install', '--python', venvPython];

  const primaryArgs = [...baseArgs, ...TORCH_PRIMARY_PACKAGES];
  if (gpuTarget.torchIndexUrl) {
    primaryArgs.push('--index-url', gpuTarget.torchIndexUrl);
  }

  try {
    await runCommand(uvBinary, primaryArgs, { env: setupEnv, logPrefix: '[setup] ' });
    return;
  } catch (error) {
    console.log('[setup] Primary torch install failed, trying fallback versions');
  }

  const fallbackArgs = [...baseArgs, ...TORCH_FALLBACK_PACKAGES];
  if (gpuTarget.torchIndexUrl) {
    fallbackArgs.push('--index-url', gpuTarget.torchIndexUrl);
  }
  await runCommand(uvBinary, fallbackArgs, { env: setupEnv, logPrefix: '[setup] ' });
}

async function installNonTorchDependencies(uvBinary, venvPython, requirementsFile, pythonEnvRoot, setupEnv) {
  const requirements = parseRequirementsFile(requirementsFile);
  const filteredRequirements = removeTorchRequirements(requirements);
  if (!filteredRequirements.length) {
    return;
  }

  const tempReqFile = path.join(pythonEnvRoot, '.requirements.non_torch.tmp');
  fs.writeFileSync(tempReqFile, `${filteredRequirements.join(os.EOL)}${os.EOL}`, 'utf-8');
  try {
    await runCommand(
      uvBinary,
      ['pip', 'install', '--python', venvPython, '-r', tempReqFile],
      { env: setupEnv, logPrefix: '[setup] ' }
    );
  } finally {
    fs.rmSync(tempReqFile, { force: true });
  }
}

async function setupPythonEnvironmentWithUv(paths) {
  setStartupStatus('setup-check', 'Checking local Python environment...', 10);

  if (!fs.existsSync(paths.requirementsFile)) {
    throw new Error(`requirements.txt not found: ${paths.requirementsFile}`);
  }

  const uvBinary = resolveUvBinaryPath(paths);
  const requirementsHash = sha256File(paths.requirementsFile);
  const existingMarker = readJsonFile(paths.setupMarkerPath);

  if (
    existingMarker &&
    existingMarker.app_version === app.getVersion() &&
    existingMarker.requirements_hash === requirementsHash &&
    fs.existsSync(paths.venvPython)
  ) {
    console.log('[setup] Existing Python environment is valid');
    setStartupStatus('setup-ready', 'Using existing Python environment', 35);
    return paths.venvPython;
  }

  console.log('[setup] Initializing Python environment with bundled uv');
  setStartupStatus('setup-init', 'Initializing first-run Python setup...', 20);
  ensureDirectory(paths.pythonEnvRoot);
  ensureDirectory(paths.pythonInstallDir);
  ensureDirectory(paths.uvCacheDir);
  ensureDirectory(paths.pipCacheDir);

  const setupEnv = getUvSetupEnv(paths);
  const gpuTarget = detectGpuTarget();
  console.log(`[setup] Detected hardware target: ${gpuTarget.displayName}`);
  setStartupStatus('setup-detect-hardware', `Detected ${gpuTarget.displayName}`, 25);

  if (fs.existsSync(paths.venvDir)) {
    fs.rmSync(paths.venvDir, { recursive: true, force: true });
  }

  setStartupStatus('setup-install-python', 'Installing standalone Python runtime...', 35);
  await runCommand(
    uvBinary,
    ['python', 'install', '3.11', '--install-dir', paths.pythonInstallDir],
    { env: setupEnv, logPrefix: '[setup] ' }
  );

  const standalonePython = findStandalonePython(paths.pythonInstallDir);
  setStartupStatus('setup-create-venv', 'Creating virtual environment...', 50);
  await runCommand(
    uvBinary,
    ['venv', paths.venvDir, '--python', standalonePython],
    { env: setupEnv, logPrefix: '[setup] ' }
  );

  setStartupStatus('setup-install-deps', 'Installing Python dependencies...', 65);
  await installTorchWithFallback(uvBinary, paths.venvPython, gpuTarget, setupEnv);
  await installNonTorchDependencies(
    uvBinary,
    paths.venvPython,
    paths.requirementsFile,
    paths.pythonEnvRoot,
    setupEnv
  );

  setStartupStatus('setup-verify', 'Verifying runtime environment...', 90);
  await runCommand(
    paths.venvPython,
    ['-c', 'import flask, flask_cors, PIL, transformers, torch, pypdfium2; print(f"torch={torch.__version__}")'],
    { env: setupEnv, logPrefix: '[setup] ' }
  );

  const uvVersionResult = spawnSync(uvBinary, ['--version'], {
    windowsHide: true,
    encoding: 'utf-8'
  });
  const uvVersion = uvVersionResult.status === 0 ? uvVersionResult.stdout.trim() : 'unknown';

  writeJsonFile(paths.setupMarkerPath, {
    app_version: app.getVersion(),
    requirements_hash: requirementsHash,
    gpu_target: gpuTarget.id,
    gpu_display: gpuTarget.displayName,
    uv_version: uvVersion,
    python_executable: paths.venvPython,
    updated_at: new Date().toISOString()
  });
  console.log('[setup] Python environment setup complete');
  setStartupStatus('setup-complete', 'Python environment ready', 95);

  return paths.venvPython;
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false
    },
    icon: path.join(__dirname, 'assets', 'icon.png')
  });

  mainWindow.loadFile('index.html');
  mainWindow.webContents.on('did-finish-load', () => {
    emitStartupStatus();
  });

  // Open external links in browser instead of Electron window
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  // Handle navigation to external URLs
  mainWindow.webContents.on('will-navigate', (event, url) => {
    // Allow navigation to local files and the Python server
    if (url.startsWith('file://') || url.startsWith(PYTHON_SERVER_URL)) {
      return;
    }
    // Open external URLs in browser
    event.preventDefault();
    shell.openExternal(url);
  });

  // Open DevTools in development mode
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', function () {
    mainWindow = null;
  });
}

async function startPythonServer() {
  console.log('Starting Python OCR server...');
  setStartupStatus('backend-init', 'Preparing backend startup...', 5);
  const paths = getAppPaths();

  // Check if backend script exists
  if (!fs.existsSync(paths.backendScript)) {
    throw new Error(`Python script not found: ${paths.backendScript}`);
  }

  let pythonExecutable;

  // Packaged builds bootstrap a managed venv with bundled uv.
  if (app.isPackaged) {
    pythonExecutable = await setupPythonEnvironmentWithUv(paths);
  } else {
    setStartupStatus('dev-env', 'Using development Python environment', 20);
    // Development mode: prefer local venv and fall back to system Python.
    if (process.platform === 'win32') {
      const venvPython = path.join(__dirname, 'venv', 'Scripts', 'python.exe');
      pythonExecutable = fs.existsSync(venvPython) ? venvPython : 'python';
    } else {
      const venvPython = path.join(__dirname, 'venv', 'bin', 'python3');
      pythonExecutable = fs.existsSync(venvPython) ? venvPython : 'python3';
    }
  }

  console.log(`Using Python: ${pythonExecutable}`);
  setStartupStatus('backend-launch', 'Launching OCR backend service...', 96);

  ensureDirectory(paths.pythonEnvRoot);
  ensureDirectory(paths.cacheRoot);
  ensureDirectory(paths.huggingFaceRoot);
  ensureDirectory(path.join(paths.huggingFaceRoot, 'hub'));

  const pythonEnv = {
    ...process.env,
    PYTHONUNBUFFERED: '1',
    DEEPSEEK_OCR_CACHE_DIR: paths.cacheRoot,
    HF_HOME: paths.huggingFaceRoot,
    HF_HUB_CACHE: path.join(paths.huggingFaceRoot, 'hub'),
    TRANSFORMERS_CACHE: path.join(paths.huggingFaceRoot, 'hub'),
    PIP_CACHE_DIR: path.join(paths.pythonEnvRoot, 'pip-cache')
  };

  return new Promise((resolve, reject) => {
    pythonProcess = spawn(pythonExecutable, [paths.backendScript], {
      windowsHide: true,
      env: pythonEnv
    });

    let resolved = false;

    const markAsReady = () => {
      if (!resolved) {
        resolved = true;
        console.log('Python server is ready!');
        setStartupStatus('ready', 'Application is ready', 100, 'ready');
        resolve();
      }
    };

    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python: ${data.toString()}`);

      // Check if server is ready
      if (data.toString().includes('Running on')) {
        markAsReady();
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      // Flask logs to stderr by default, even for INFO messages
      console.log(`Python: ${data.toString()}`);

      // Flask logs to stderr, so also check here for server ready message
      if (data.toString().includes('Running on')) {
        markAsReady();
      }
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
    });

    pythonProcess.on('error', (error) => {
      if (!resolved) {
        resolved = true;
        setStartupStatus('error', `Backend process failed: ${error.message}`, 100, 'error');
        reject(new Error(`Failed to start Python backend process: ${error.message}`));
      }
    });

    // Wait for server to start with retry logic (timeout after 30 seconds)
    const startTime = Date.now();
    const maxWaitTime = 30000; // 30 seconds
    const checkInterval = 1000; // Check every 1 second

    const checkWithRetry = async () => {
      if (resolved) return;

      const elapsed = Date.now() - startTime;
      if (elapsed >= maxWaitTime) {
        if (!resolved) {
          resolved = true;
          setStartupStatus('error', 'Backend startup timed out', 100, 'error');
          reject(new Error('Python server failed to start within timeout'));
        }
        return;
      }

      try {
        await checkServerHealth();
        markAsReady();
      } catch (error) {
        // Retry after interval
        setTimeout(checkWithRetry, checkInterval);
      }
    };

    // Start checking after initial delay
    setTimeout(checkWithRetry, 2000);
  });
}

async function checkServerHealth() {
  try {
    const response = await axios.get(`${PYTHON_SERVER_URL}/health`);
    return response.data.status === 'ok';
  } catch (error) {
    throw error;
  }
}

function stopPythonServer() {
  if (pythonProcess) {
    console.log('Stopping Python server...');
    pythonProcess.kill();
    pythonProcess = null;
  }
}

// IPC Handlers
ipcMain.handle('check-server-status', async () => {
  try {
    const response = await axios.get(`${PYTHON_SERVER_URL}/health`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-startup-status', async () => startupStatus);

ipcMain.handle('load-model', async () => {
  try {
    const response = await axios.post(`${PYTHON_SERVER_URL}/load_model`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-model-info', async () => {
  try {
    const response = await axios.get(`${PYTHON_SERVER_URL}/model_info`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('select-image', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Images and PDFs', extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'pdf'] }
    ]
  });

  if (!result.canceled && result.filePaths.length > 0) {
    return { success: true, filePath: result.filePaths[0] };
  }
  return { success: false };
});

ipcMain.handle('select-images', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Images and PDFs', extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'pdf'] }
    ]
  });

  if (!result.canceled && result.filePaths.length > 0) {
    return { success: true, filePaths: result.filePaths };
  }
  return { success: false };
});

ipcMain.handle('select-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory']
  });

  if (!result.canceled && result.filePaths.length > 0) {
    return { success: true, folderPath: result.filePaths[0] };
  }
  return { success: false };
});

ipcMain.handle('open-folder', async (event, folderPath) => {
  try {
    shell.openPath(folderPath);
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('perform-ocr', async (event, { imagePath, promptType, baseSize, imageSize, cropMode }) => {
  try {
    const FormData = require('form-data');
    const formData = new FormData();

    // Read image file and append to form data
    const imageBuffer = fs.readFileSync(imagePath);
    formData.append('image', imageBuffer, {
      filename: path.basename(imagePath),
      contentType: getMimeTypeForFile(imagePath)
    });

    formData.append('prompt_type', promptType || 'document');
    formData.append('base_size', baseSize || 1024);
    formData.append('image_size', imageSize || 640);
    formData.append('crop_mode', cropMode ? 'true' : 'false');

    const response = await axios.post(`${PYTHON_SERVER_URL}/ocr`, formData, {
      headers: formData.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });

    return { success: true, data: response.data };
  } catch (error) {
    console.error('OCR Error:', error);
    return {
      success: false,
      error: error.response?.data?.message || error.message
    };
  }
});

// Queue operations
ipcMain.handle('add-to-queue', async (event, { filePaths, promptType, baseSize, imageSize, cropMode }) => {
  try {
    const FormData = require('form-data');
    const formData = new FormData();

    // Add all files to form data
    for (const filePath of filePaths) {
      const imageBuffer = fs.readFileSync(filePath);
      formData.append('files', imageBuffer, {
        filename: path.basename(filePath),
        contentType: getMimeTypeForFile(filePath)
      });
    }

    formData.append('prompt_type', promptType || 'document');
    formData.append('base_size', baseSize || 1024);
    formData.append('image_size', imageSize || 640);
    formData.append('crop_mode', cropMode ? 'true' : 'false');

    const response = await axios.post(`${PYTHON_SERVER_URL}/queue/add`, formData, {
      headers: formData.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });

    return { success: true, data: response.data };
  } catch (error) {
    console.error('Add to queue error:', error);
    return {
      success: false,
      error: error.response?.data?.message || error.message
    };
  }
});

ipcMain.handle('get-queue-status', async () => {
  try {
    const response = await axios.get(`${PYTHON_SERVER_URL}/queue/status`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('process-queue', async () => {
  try {
    const response = await axios.post(`${PYTHON_SERVER_URL}/queue/process`, {}, {
      timeout: 0 // No timeout for queue processing
    });
    return { success: true, data: response.data };
  } catch (error) {
    console.error('Process queue error:', error);
    return {
      success: false,
      error: error.response?.data?.message || error.message
    };
  }
});

ipcMain.handle('clear-queue', async () => {
  try {
    const response = await axios.post(`${PYTHON_SERVER_URL}/queue/clear`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('remove-from-queue', async (event, itemId) => {
  try {
    const response = await axios.delete(`${PYTHON_SERVER_URL}/queue/remove/${itemId}`);
    return { success: true, data: response.data };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

// App lifecycle
app.whenReady().then(async () => {
  createWindow();

  try {
    await startPythonServer();
  } catch (error) {
    console.error('Failed to start Python server:', error);
    setStartupStatus('error', `Startup failed: ${error.message}`, 100, 'error');
    dialog.showErrorBox(
      'Startup Error',
      `Failed to start Python server: ${error.message}\n\n` +
      'Ensure internet access is available for first-run dependency setup and try again.\n' +
      'The app will remain open so you can review setup status.'
    );
  }

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', function () {
  stopPythonServer();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  stopPythonServer();
});
