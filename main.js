const { app, BrowserWindow, ipcMain, dialog, shell, session } = require('electron');
const path = require('path');
const { spawn, spawnSync } = require('child_process');
const fs = require('fs');
const os = require('os');
const crypto = require('crypto');
const axios = require('axios');
const JSZip = require('jszip');

let mainWindow;
let pythonProcess;
const PYTHON_SERVER_PORT = 5000;
const PYTHON_SERVER_URL = `http://127.0.0.1:${PYTHON_SERVER_PORT}`;
const TORCH_PRIMARY_PACKAGES = ['torch>=2.6.0', 'torchvision>=0.21.0', 'torchaudio>=2.6.0'];
const TORCH_FALLBACK_PACKAGES = ['torch>=2.4.0', 'torchvision>=0.19.0', 'torchaudio>=2.4.0'];
const MLX_PACKAGES = ['mlx', 'mlx-vlm>=0.3.0'];
const INPUT_MIME_TYPES = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.bmp': 'image/bmp',
  '.webp': 'image/webp',
  '.pdf': 'application/pdf'
};
const RETENTION_POLICY_FILE = 'retention-policy.json';
const DEFAULT_RETENTION_POLICY = {
  outputRetentionDays: 30,
  maxQueueRuns: 40,
  downloadCacheRetentionDays: 30,
  cleanupOnStartup: true
};
const PREFLIGHT_ESTIMATES = {
  pythonRuntimeBytes: 1600 * 1024 * 1024,
  dependencyBytes: 1800 * 1024 * 1024,
  modelBytes: 7800 * 1024 * 1024,
  scratchBytes: 2200 * 1024 * 1024
};

const MAIN_LOG_BUFFER_MAX = 1800;
const mainLogBuffer = [];

function formatLocalTimestamp(date = new Date()) {
  const pad = (value, size = 2) => String(value).padStart(size, '0');
  const year = date.getFullYear();
  const month = pad(date.getMonth() + 1);
  const day = pad(date.getDate());
  const hours = pad(date.getHours());
  const minutes = pad(date.getMinutes());
  const seconds = pad(date.getSeconds());
  const milliseconds = pad(date.getMilliseconds(), 3);
  const offsetMinutes = -date.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? '+' : '-';
  const absOffset = Math.abs(offsetMinutes);
  const offsetHours = pad(Math.floor(absOffset / 60));
  const offsetRemainder = pad(absOffset % 60);
  return `${year}-${month}-${day}T${hours}:${minutes}:${seconds}.${milliseconds}${sign}${offsetHours}:${offsetRemainder}`;
}

function getLocalTimezone() {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || 'local';
  } catch (error) {
    return 'local';
  }
}

function getZipEntryLocalDate() {
  const now = new Date();
  return new Date(now.getTime() - (now.getTimezoneOffset() * 60 * 1000));
}

function appendMainLog(level, source, message) {
  const entry = {
    timestamp: formatLocalTimestamp(),
    level,
    source,
    message: String(message)
  };
  mainLogBuffer.push(entry);
  if (mainLogBuffer.length > MAIN_LOG_BUFFER_MAX) {
    mainLogBuffer.splice(0, mainLogBuffer.length - MAIN_LOG_BUFFER_MAX);
  }
}

function getMainLogTail(limit = 300) {
  return mainLogBuffer.slice(Math.max(0, mainLogBuffer.length - limit));
}
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
  appendMainLog('info', 'startup', `${phase} ${progress}% ${message}`);
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
  const modelCacheDir = path.join(cacheRoot, 'models');

  const venvDir = path.join(pythonEnvRoot, 'venv');

  return {
    runtimeRoot,
    backendScript: path.join(runtimeRoot, 'backend', 'ocr_server.py'),
    mockBackendScript: path.join(runtimeRoot, 'tests', 'e2e', 'mock_backend.js'),
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
    modelCacheDir
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

function safeNumber(value, fallback, min = Number.MIN_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER) {
  const numeric = Number.parseInt(value, 10);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, numeric));
}

function getDirectorySize(dirPath) {
  let total = 0;
  if (!fs.existsSync(dirPath)) {
    return total;
  }
  const stack = [dirPath];
  while (stack.length > 0) {
    const current = stack.pop();
    let entries = [];
    try {
      entries = fs.readdirSync(current, { withFileTypes: true });
    } catch (error) {
      continue;
    }
    for (const entry of entries) {
      const fullPath = path.join(current, entry.name);
      try {
        if (entry.isDirectory()) {
          stack.push(fullPath);
        } else if (entry.isFile()) {
          total += fs.statSync(fullPath).size;
        }
      } catch (error) {
        // Best effort: ignore transient files.
      }
    }
  }
  return total;
}

function statDiskFreeBytes(targetPath) {
  try {
    const statfs = fs.statfsSync(targetPath);
    return Number(statfs.bavail) * Number(statfs.bsize);
  } catch (error) {
    return null;
  }
}

function getRetentionPolicyPath() {
  return path.join(app.getPath('userData'), RETENTION_POLICY_FILE);
}

function normalizeRetentionPolicy(input = {}) {
  return {
    outputRetentionDays: safeNumber(input.outputRetentionDays, DEFAULT_RETENTION_POLICY.outputRetentionDays, 0, 3650),
    maxQueueRuns: safeNumber(input.maxQueueRuns, DEFAULT_RETENTION_POLICY.maxQueueRuns, 1, 1000),
    downloadCacheRetentionDays: safeNumber(
      input.downloadCacheRetentionDays,
      DEFAULT_RETENTION_POLICY.downloadCacheRetentionDays,
      0,
      3650
    ),
    cleanupOnStartup: input.cleanupOnStartup !== undefined
      ? Boolean(input.cleanupOnStartup)
      : DEFAULT_RETENTION_POLICY.cleanupOnStartup
  };
}

function loadRetentionPolicy() {
  const existing = readJsonFile(getRetentionPolicyPath());
  if (!existing) {
    return { ...DEFAULT_RETENTION_POLICY };
  }
  return normalizeRetentionPolicy(existing);
}

function saveRetentionPolicy(policy) {
  const normalized = normalizeRetentionPolicy(policy);
  writeJsonFile(getRetentionPolicyPath(), normalized);
  return normalized;
}

function cleanupFilesOlderThan(rootPath, cutoffMs) {
  let removedFiles = 0;
  let removedBytes = 0;
  if (!fs.existsSync(rootPath)) {
    return { removedFiles, removedBytes };
  }

  const walk = (dirPath) => {
    let entries = [];
    try {
      entries = fs.readdirSync(dirPath, { withFileTypes: true });
    } catch (error) {
      return;
    }

    for (const entry of entries) {
      const fullPath = path.join(dirPath, entry.name);
      if (entry.isDirectory()) {
        walk(fullPath);
        try {
          const remaining = fs.readdirSync(fullPath);
          if (remaining.length === 0) {
            fs.rmdirSync(fullPath);
          }
        } catch (error) {
          // Ignore non-empty or inaccessible directories.
        }
      } else if (entry.isFile()) {
        try {
          const stat = fs.statSync(fullPath);
          if (stat.mtimeMs < cutoffMs) {
            removedBytes += stat.size;
            removedFiles += 1;
            fs.rmSync(fullPath, { force: true });
          }
        } catch (error) {
          // Ignore files that disappear during cleanup.
        }
      }
    }
  };

  walk(rootPath);
  return { removedFiles, removedBytes };
}

function applyRetentionPolicy(paths, policyInput) {
  const policy = normalizeRetentionPolicy(policyInput || loadRetentionPolicy());
  const nowMs = Date.now();
  const report = {
    policy,
    removedQueueRuns: 0,
    removedQueueBytes: 0,
    removedCacheFiles: 0,
    removedCacheBytes: 0,
    scannedAt: formatLocalTimestamp()
  };

  const outputsRoot = path.join(paths.cacheRoot, 'outputs');
  if (fs.existsSync(outputsRoot)) {
    let queueDirs = [];
    try {
      queueDirs = fs.readdirSync(outputsRoot, { withFileTypes: true })
        .filter((entry) => entry.isDirectory() && entry.name.startsWith('queue_'))
        .map((entry) => {
          const fullPath = path.join(outputsRoot, entry.name);
          const stat = fs.statSync(fullPath);
          return { name: entry.name, fullPath, mtimeMs: stat.mtimeMs };
        });
    } catch (error) {
      queueDirs = [];
    }

    if (policy.outputRetentionDays > 0) {
      const cutoffMs = nowMs - (policy.outputRetentionDays * 24 * 60 * 60 * 1000);
      for (const dir of queueDirs.filter((entry) => entry.mtimeMs < cutoffMs)) {
        const removedSize = getDirectorySize(dir.fullPath);
        fs.rmSync(dir.fullPath, { recursive: true, force: true });
        report.removedQueueRuns += 1;
        report.removedQueueBytes += removedSize;
      }
    }

    if (policy.maxQueueRuns > 0) {
      let activeQueueDirs = [];
      try {
        activeQueueDirs = fs.readdirSync(outputsRoot, { withFileTypes: true })
          .filter((entry) => entry.isDirectory() && entry.name.startsWith('queue_'))
          .map((entry) => {
            const fullPath = path.join(outputsRoot, entry.name);
            const stat = fs.statSync(fullPath);
            return { name: entry.name, fullPath, mtimeMs: stat.mtimeMs };
          })
          .sort((a, b) => b.mtimeMs - a.mtimeMs);
      } catch (error) {
        activeQueueDirs = [];
      }

      const stale = activeQueueDirs.slice(policy.maxQueueRuns);
      for (const dir of stale) {
        const removedSize = getDirectorySize(dir.fullPath);
        fs.rmSync(dir.fullPath, { recursive: true, force: true });
        report.removedQueueRuns += 1;
        report.removedQueueBytes += removedSize;
      }
    }
  }

  if (policy.downloadCacheRetentionDays > 0) {
    const cutoffMs = nowMs - (policy.downloadCacheRetentionDays * 24 * 60 * 60 * 1000);
    const uvCleanup = cleanupFilesOlderThan(paths.uvCacheDir, cutoffMs);
    const pipCleanup = cleanupFilesOlderThan(paths.pipCacheDir, cutoffMs);
    report.removedCacheFiles = uvCleanup.removedFiles + pipCleanup.removedFiles;
    report.removedCacheBytes = uvCleanup.removedBytes + pipCleanup.removedBytes;
  }

  appendMainLog('info', 'retention', `Applied retention policy: ${JSON.stringify(report)}`);
  return report;
}

function buildPreflightReport(paths) {
  const setupMarker = readJsonFile(paths.setupMarkerPath);
  const hasManagedEnv = Boolean(setupMarker && fs.existsSync(paths.venvPython));
  const pythonEnvBytes = getDirectorySize(paths.pythonEnvRoot);
  const modelCacheBytes = getDirectorySize(paths.modelCacheDir);
  const modelLikelyPresent = modelCacheBytes > (800 * 1024 * 1024);
  const freeBytes = statDiskFreeBytes(app.getPath('userData'));

  const pythonSetupBytes = PREFLIGHT_ESTIMATES.pythonRuntimeBytes + PREFLIGHT_ESTIMATES.dependencyBytes;
  const expectedDownloadBytes = (hasManagedEnv ? 0 : pythonSetupBytes) + (modelLikelyPresent ? 0 : PREFLIGHT_ESTIMATES.modelBytes);
  const expectedRequiredBytes = expectedDownloadBytes + PREFLIGHT_ESTIMATES.scratchBytes;
  const diskOk = freeBytes === null ? null : freeBytes >= expectedRequiredBytes;

  const estimateMinutesAtMbps = (mbps) => {
    if (expectedDownloadBytes <= 0 || mbps <= 0) {
      return 0;
    }
    const bits = expectedDownloadBytes * 8;
    const seconds = bits / (mbps * 1_000_000);
    return Math.round(seconds / 60);
  };

  return {
    checkedAt: formatLocalTimestamp(),
    setupComplete: hasManagedEnv,
    modelCachePresent: modelLikelyPresent,
    pythonEnvBytes,
    modelCacheBytes,
    expectedDownloadBytes,
    expectedRequiredBytes,
    freeDiskBytes: freeBytes,
    diskOk,
    estimatedMinutes: {
      fast_100mbps: estimateMinutesAtMbps(100),
      typical_30mbps: estimateMinutesAtMbps(30),
      slow_10mbps: estimateMinutesAtMbps(10)
    }
  };
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
            appendMainLog('info', 'setup', `${logPrefix}${line}`);
          }
        }
      } else {
        process.stdout.write(text);
        for (const line of text.split(/\r?\n/)) {
          if (line) {
            appendMainLog('info', 'process', line);
          }
        }
      }
    });

    child.stderr.on('data', (data) => {
      const text = data.toString();
      stderr = appendCaptured(stderr, text);
      if (logPrefix) {
        for (const line of text.split(/\r?\n/)) {
          if (line) {
            console.log(`${logPrefix}${line}`);
            appendMainLog('warn', 'setup', `${logPrefix}${line}`);
          }
        }
      } else {
        process.stderr.write(text);
        for (const line of text.split(/\r?\n/)) {
          if (line) {
            appendMainLog('warn', 'process', line);
          }
        }
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
      id: 'mlx',
      displayName: 'Apple Silicon (MLX)',
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
      displayName: 'CPU (macOS Intel)',
      torchIndexUrl: 'https://download.pytorch.org/whl/cpu'
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

async function installMlxDependencies(uvBinary, venvPython, setupEnv) {
  await runCommand(
    uvBinary,
    ['pip', 'install', '--python', venvPython, ...MLX_PACKAGES],
    { env: setupEnv, logPrefix: '[setup] ' }
  );
}

async function installModelFrameworkDependencies(uvBinary, venvPython, gpuTarget, setupEnv) {
  if (gpuTarget.id === 'mlx') {
    await installMlxDependencies(uvBinary, venvPython, setupEnv);
    return;
  }
  await installTorchWithFallback(uvBinary, venvPython, gpuTarget, setupEnv);
}

function getRuntimeVerifySnippet(gpuTarget) {
  if (gpuTarget.id === 'mlx') {
    return [
      'import flask, flask_cors, PIL, transformers, pypdfium2, mlx_vlm',
      'print("runtime=mlx")'
    ].join('; ');
  }
  return [
    'import flask, flask_cors, PIL, transformers, torch, pypdfium2',
    'print(f"runtime=torch torch={torch.__version__}")'
  ].join('; ');
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

  const gpuTarget = detectGpuTarget();
  console.log(`[setup] Detected hardware target: ${gpuTarget.displayName}`);
  setStartupStatus('setup-detect-hardware', `Detected ${gpuTarget.displayName}`, 18);

  const uvBinary = resolveUvBinaryPath(paths);
  const requirementsHash = sha256File(paths.requirementsFile);
  const existingMarker = readJsonFile(paths.setupMarkerPath);
  const markerCoreMatch = Boolean(
    existingMarker &&
    existingMarker.app_version === app.getVersion() &&
    existingMarker.requirements_hash === requirementsHash &&
    fs.existsSync(paths.venvPython)
  );
  const markerTargetMatch = markerCoreMatch && existingMarker.gpu_target === gpuTarget.id;

  if (markerTargetMatch) {
    console.log(`[setup] Existing Python environment is valid for ${gpuTarget.displayName}`);
    setStartupStatus('setup-ready', `Using existing ${gpuTarget.displayName} environment`, 35);
    return paths.venvPython;
  }

  if (markerCoreMatch) {
    const previousTarget = existingMarker.gpu_display || existingMarker.gpu_target || 'unknown target';
    console.log(`[setup] Hardware target changed (${previousTarget} -> ${gpuTarget.displayName}), refreshing runtime`);
    setStartupStatus(
      'setup-hardware-change',
      `Hardware changed (${previousTarget} -> ${gpuTarget.displayName}), refreshing runtime...`,
      22
    );
  } else {
    console.log('[setup] Initializing Python environment with bundled uv');
    setStartupStatus('setup-init', 'Initializing first-run Python setup...', 20);
  }

  ensureDirectory(paths.pythonEnvRoot);
  ensureDirectory(paths.pythonInstallDir);
  ensureDirectory(paths.uvCacheDir);
  ensureDirectory(paths.pipCacheDir);

  const setupEnv = getUvSetupEnv(paths);

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
  await installModelFrameworkDependencies(uvBinary, paths.venvPython, gpuTarget, setupEnv);
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
    ['-c', getRuntimeVerifySnippet(gpuTarget)],
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
    runtime_backend: gpuTarget.id === 'mlx' ? 'mlx' : 'torch',
    uv_version: uvVersion,
    python_executable: paths.venvPython,
    updated_at: formatLocalTimestamp()
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
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: false
    },
    icon: path.join(__dirname, 'assets', 'icon.png')
  });

  // Set Content-Security-Policy via response headers (instead of HTML meta tag)
  // to avoid false-positive AV heuristic matches on the HTML file.
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; " +
          `img-src 'self' data: blob: ${PYTHON_SERVER_URL}; ` +
          `connect-src 'self' ${PYTHON_SERVER_URL};`
        ]
      }
    });
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
  const useMockBackend = process.env.DEEPSEEK_MOCK_BACKEND === '1';

  try {
    const retentionPolicy = loadRetentionPolicy();
    if (retentionPolicy.cleanupOnStartup) {
      setStartupStatus('cleanup', 'Applying cache retention policy...', 8);
      const cleanupReport = applyRetentionPolicy(paths, retentionPolicy);
      console.log(`[cleanup] Removed ${cleanupReport.removedQueueRuns} queue runs`);
    }
  } catch (error) {
    appendMainLog('warn', 'retention', `Retention cleanup failed: ${error.message}`);
  }

  let backendCommand;
  let backendArgs = [];
  let backendEnv;
  let backendLogLabel;

  if (useMockBackend) {
    if (!fs.existsSync(paths.mockBackendScript)) {
      throw new Error(`Mock backend script not found: ${paths.mockBackendScript}`);
    }
    backendCommand = process.execPath;
    backendArgs = [paths.mockBackendScript];
    backendEnv = {
      ...process.env,
      MOCK_BACKEND_PORT: String(PYTHON_SERVER_PORT)
    };
    backendLogLabel = 'MockBackend';
    setStartupStatus('backend-launch', 'Launching mock backend service...', 96);
  } else {
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
    const runtimeGpuTarget = detectGpuTarget();
    console.log(`[backend] Runtime hardware target: ${runtimeGpuTarget.displayName}`);

    ensureDirectory(paths.pythonEnvRoot);
    ensureDirectory(paths.cacheRoot);
    ensureDirectory(paths.modelCacheDir);

    backendCommand = pythonExecutable;
    backendArgs = [paths.backendScript];
    backendEnv = {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      DEEPSEEK_OCR_CACHE_DIR: paths.cacheRoot,
      DEEPSEEK_OCR_MODEL_CACHE_DIR: paths.modelCacheDir,
      DEEPSEEK_OCR_GPU_TARGET: runtimeGpuTarget.id,
      HF_HOME: paths.modelCacheDir,
      HF_HUB_CACHE: paths.modelCacheDir,
      TRANSFORMERS_CACHE: paths.modelCacheDir,
      PIP_CACHE_DIR: path.join(paths.pythonEnvRoot, 'pip-cache')
    };
    backendLogLabel = 'Python';
  }

  return new Promise((resolve, reject) => {
    pythonProcess = spawn(backendCommand, backendArgs, {
      windowsHide: true,
      env: backendEnv
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
      const text = data.toString();
      console.log(`${backendLogLabel}: ${text}`);
      for (const line of text.split(/\r?\n/)) {
        if (line) {
          appendMainLog('info', backendLogLabel, line);
        }
      }

      // Check if server is ready
      if (text.includes('Running on')) {
        markAsReady();
      }
    });

    pythonProcess.stderr.on('data', (data) => {
      const text = data.toString();
      // Flask logs to stderr by default, even for INFO messages
      console.log(`${backendLogLabel}: ${text}`);
      for (const line of text.split(/\r?\n/)) {
        if (line) {
          appendMainLog('warn', backendLogLabel, line);
        }
      }

      // Flask logs to stderr, so also check here for server ready message
      if (text.includes('Running on')) {
        markAsReady();
      }
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
      appendMainLog('warn', 'backend', `Backend process exited with code ${code}`);
    });

    pythonProcess.on('error', (error) => {
      if (!resolved) {
        resolved = true;
        appendMainLog('error', 'backend', `Backend process error: ${error.message}`);
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
    const response = await axios.get(`${PYTHON_SERVER_URL}/health`, { timeout: 2000 });
    return response.data.status === 'ok';
  } catch (error) {
    throw error;
  }
}

function stopPythonServer() {
  if (pythonProcess) {
    console.log('Stopping Python server...');
    appendMainLog('info', 'backend', 'Stopping backend process');
    pythonProcess.kill();
    pythonProcess = null;
  }
}

function parseMarkdownImageNames(markdownText) {
  const imageRegex = /!\[[^\]]*\]\(images\/([^)]+)\)/g;
  const imageNames = new Set();
  let match = imageRegex.exec(markdownText || '');
  while (match) {
    const safeName = path.basename(match[1]);
    if (safeName) {
      imageNames.add(safeName);
    }
    match = imageRegex.exec(markdownText || '');
  }
  return Array.from(imageNames);
}

async function exportDiagnosticsBundle() {
  const paths = getAppPaths();
  const preflight = buildPreflightReport(paths);
  const retention = loadRetentionPolicy();
  const setupMarker = readJsonFile(paths.setupMarkerPath);

  let backendDiagnostics = null;
  let backendError = null;
  try {
    const response = await axios.get(`${PYTHON_SERVER_URL}/diagnostics`, { timeout: 8000 });
    backendDiagnostics = response.data;
  } catch (error) {
    backendError = error.message;
  }

  const diagnostics = {
    capturedAt: formatLocalTimestamp(),
    timezone: getLocalTimezone(),
    app: {
      version: app.getVersion(),
      isPackaged: app.isPackaged,
      electron: process.versions.electron,
      node: process.versions.node,
      chrome: process.versions.chrome,
      platform: process.platform,
      arch: process.arch
    },
    startupStatus,
    paths: {
      userData: app.getPath('userData'),
      pythonEnvRoot: paths.pythonEnvRoot,
      cacheRoot: paths.cacheRoot,
      modelCacheDir: paths.modelCacheDir
    },
    setupMarker,
    preflight,
    retention,
    backendDiagnostics,
    backendDiagnosticsError: backendError
  };

  const defaultPath = path.join(app.getPath('downloads'), `deepseek-ocr-diagnostics-${Date.now()}.zip`);
  const saveResult = await dialog.showSaveDialog(mainWindow, {
    title: 'Save Diagnostics Bundle',
    defaultPath,
    filters: [{ name: 'ZIP Archive', extensions: ['zip'] }]
  });

  if (saveResult.canceled || !saveResult.filePath) {
    return { success: false, canceled: true };
  }

  const zip = new JSZip();
  const zipEntryDate = getZipEntryLocalDate();
  zip.file('diagnostics.json', JSON.stringify(diagnostics, null, 2), { date: zipEntryDate });
  zip.file('main-logs.json', JSON.stringify(getMainLogTail(600), null, 2), { date: zipEntryDate });
  if (backendDiagnostics && Array.isArray(backendDiagnostics.logs_tail)) {
    zip.file('backend-logs.txt', backendDiagnostics.logs_tail.join(os.EOL), { date: zipEntryDate });
  }

  const buffer = await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE', compressionOptions: { level: 6 } });
  fs.writeFileSync(saveResult.filePath, buffer);
  appendMainLog('info', 'diagnostics', `Saved diagnostics bundle to ${saveResult.filePath}`);
  return { success: true, filePath: saveResult.filePath };
}

async function saveDocumentZip(markdownText) {
  const paths = getAppPaths();
  const outputsDir = path.join(paths.cacheRoot, 'outputs');
  const imagesDir = path.join(outputsDir, 'images');
  const zip = new JSZip();
  const zipEntryDate = getZipEntryLocalDate();
  zip.file('output.md', markdownText || '', { date: zipEntryDate });

  const imageNames = parseMarkdownImageNames(markdownText || '');
  if (imageNames.length > 0) {
    const imageFolder = zip.folder('images');
    for (const imageName of imageNames) {
      const imagePath = path.join(imagesDir, imageName);
      if (fs.existsSync(imagePath) && fs.statSync(imagePath).isFile()) {
        imageFolder.file(imageName, fs.readFileSync(imagePath), { date: zipEntryDate });
      }
    }
  }

  const saveResult = await dialog.showSaveDialog(mainWindow, {
    title: 'Save OCR ZIP',
    defaultPath: path.join(app.getPath('downloads'), `ocr-output-${Date.now()}.zip`),
    filters: [{ name: 'ZIP Archive', extensions: ['zip'] }]
  });
  if (saveResult.canceled || !saveResult.filePath) {
    return { success: false, canceled: true };
  }

  const buffer = await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE', compressionOptions: { level: 6 } });
  fs.writeFileSync(saveResult.filePath, buffer);
  return { success: true, filePath: saveResult.filePath, imageCount: imageNames.length };
}

// IPC Handlers
ipcMain.handle('check-server-status', async () => {
  try {
    const response = await axios.get(`${PYTHON_SERVER_URL}/health`, { timeout: 2000 });
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

ipcMain.handle('list-folder-inputs', async (event, folderPath) => {
  try {
    if (!folderPath || !fs.existsSync(folderPath)) {
      return { success: false, error: 'Folder does not exist' };
    }
    const entries = fs.readdirSync(folderPath, { withFileTypes: true });
    const files = entries
      .filter((entry) => entry.isFile())
      .map((entry) => entry.name)
      .filter((name) => Object.prototype.hasOwnProperty.call(INPUT_MIME_TYPES, path.extname(name).toLowerCase()))
      .sort((a, b) => a.localeCompare(b))
      .map((name) => path.join(folderPath, name));
    return { success: true, filePaths: files };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('open-folder', async (event, folderPath) => {
  try {
    shell.openPath(folderPath);
    return { success: true };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('perform-ocr', async (event, {
  imagePath,
  promptType,
  baseSize,
  imageSize,
  cropMode,
  pdfPageRange
}) => {
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
    if (pdfPageRange) {
      formData.append('pdf_page_range', String(pdfPageRange));
    }

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
ipcMain.handle('add-to-queue', async (event, {
  filePaths,
  promptType,
  baseSize,
  imageSize,
  cropMode,
  pdfPageRange
}) => {
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
    if (pdfPageRange) {
      formData.append('pdf_page_range', String(pdfPageRange));
    }

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

ipcMain.handle('pause-queue', async () => {
  try {
    const response = await axios.post(`${PYTHON_SERVER_URL}/queue/pause`);
    return { success: true, data: response.data };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.message || error.message
    };
  }
});

ipcMain.handle('resume-queue', async () => {
  try {
    const response = await axios.post(`${PYTHON_SERVER_URL}/queue/resume`);
    return { success: true, data: response.data };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.message || error.message
    };
  }
});

ipcMain.handle('cancel-queue', async () => {
  try {
    const response = await axios.post(`${PYTHON_SERVER_URL}/queue/cancel`);
    return { success: true, data: response.data };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.message || error.message
    };
  }
});

ipcMain.handle('retry-failed-queue', async () => {
  try {
    const response = await axios.post(`${PYTHON_SERVER_URL}/queue/retry_failed`);
    return { success: true, data: response.data };
  } catch (error) {
    return {
      success: false,
      error: error.response?.data?.message || error.message
    };
  }
});

ipcMain.handle('run-preflight-check', async () => {
  try {
    const paths = getAppPaths();
    const report = buildPreflightReport(paths);
    return {
      success: true,
      data: {
        ...report,
        retentionPolicy: loadRetentionPolicy()
      }
    };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('export-diagnostics', async () => {
  try {
    return await exportDiagnosticsBundle();
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('save-document-zip', async (event, { markdownText }) => {
  try {
    return await saveDocumentZip(markdownText || '');
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-retention-policy', async () => {
  try {
    return { success: true, data: loadRetentionPolicy() };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('update-retention-policy', async (event, policy) => {
  try {
    const savedPolicy = saveRetentionPolicy(policy || {});
    return { success: true, data: savedPolicy };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('apply-retention-cleanup', async () => {
  try {
    const paths = getAppPaths();
    const policy = loadRetentionPolicy();
    const report = applyRetentionPolicy(paths, policy);
    return { success: true, data: report };
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
