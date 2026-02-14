const fs = require('fs');
const os = require('os');
const path = require('path');
const { test, expect } = require('@playwright/test');
const { _electron: electron } = require('playwright');

const APP_ROOT = path.resolve(__dirname, '..', '..');

function createFixtures() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'deepseek-ocr-e2e-'));
  const imagePath = path.join(dir, 'sample.png');
  const pdfPath = path.join(dir, 'sample.pdf');

  // 1x1 transparent PNG
  fs.writeFileSync(
    imagePath,
    Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAFhQJ/lQ1xXwAAAABJRU5ErkJggg==',
      'base64'
    )
  );

  // Minimal valid PDF
  const minimalPdf = [
    '%PDF-1.1',
    '1 0 obj',
    '<< /Type /Catalog /Pages 2 0 R >>',
    'endobj',
    '2 0 obj',
    '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
    'endobj',
    '3 0 obj',
    '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << >> >>',
    'endobj',
    '4 0 obj',
    '<< /Length 44 >>',
    'stream',
    'BT /F1 12 Tf 72 120 Td (Mock PDF Fixture) Tj ET',
    'endstream',
    'endobj',
    'xref',
    '0 5',
    '0000000000 65535 f ',
    '0000000010 00000 n ',
    '0000000060 00000 n ',
    '0000000117 00000 n ',
    '0000000222 00000 n ',
    'trailer',
    '<< /Root 1 0 R /Size 5 >>',
    'startxref',
    '315',
    '%%EOF'
  ].join('\n');
  fs.writeFileSync(pdfPath, minimalPdf, 'utf-8');

  return { dir, imagePath, pdfPath };
}

function removeDirQuietly(dirPath) {
  try {
    fs.rmSync(dirPath, { recursive: true, force: true });
  } catch (error) {
    // Best-effort cleanup only.
  }
}

async function invokeIpc(window, channel, payload) {
  const channelToMethod = {
    'load-model': 'loadModel',
    'perform-ocr': 'performOCR',
    'add-to-queue': 'addToQueue',
    'process-queue': 'processQueue',
    'get-queue-status': 'getQueueStatus'
  };
  const methodName = channelToMethod[channel];
  if (!methodName) {
    throw new Error(`Unsupported channel in E2E helper: ${channel}`);
  }

  return window.evaluate(
    async ({ apiMethod, channelPayload }) => {
      if (!window.appAPI || typeof window.appAPI[apiMethod] !== 'function') {
        throw new Error(`appAPI method not found: ${apiMethod}`);
      }
      return window.appAPI[apiMethod](channelPayload);
    },
    { apiMethod: methodName, channelPayload: payload }
  );
}

async function waitForConnected(window) {
  await expect
    .poll(() =>
      window.evaluate(() => document.getElementById('server-status')?.textContent || '')
    )
    .toContain('Connected');
}

test.describe('DeepSeek OCR Electron E2E', () => {
  let electronApp;
  let window;
  let fixtures;

  test.beforeEach(async () => {
    fixtures = createFixtures();
    const launchArgs = [APP_ROOT];
    if (process.platform === 'linux') {
      // CI runners can have sandbox restrictions that prevent Electron from booting.
      launchArgs.push('--no-sandbox', '--disable-setuid-sandbox');
    }
    const launchEnv = {
      ...process.env,
      DEEPSEEK_MOCK_BACKEND: '1',
      ELECTRON_DISABLE_SECURITY_WARNINGS: 'true'
    };
    // Some CI environments leak this variable, which makes Electron exit as Node.
    delete launchEnv.ELECTRON_RUN_AS_NODE;

    electronApp = await electron.launch({
      args: launchArgs,
      cwd: APP_ROOT,
      env: launchEnv
    });
    window = await electronApp.firstWindow();
    await waitForConnected(window);
  });

  test.afterEach(async () => {
    if (electronApp) {
      await electronApp.close();
    }
    if (fixtures) {
      removeDirQuietly(fixtures.dir);
    }
  });

  test('loads model from UI and reflects status', async () => {
    await window.locator('#load-model-btn').click();
    await expect
      .poll(() =>
        window.evaluate(() => document.getElementById('model-status')?.textContent || '')
      )
      .toContain('Loaded');
  });

  test('runs OCR operations for image and PDF through IPC', async () => {
    const modelResult = await invokeIpc(window, 'load-model');
    expect(modelResult.success).toBeTruthy();

    const imageResult = await invokeIpc(window, 'perform-ocr', {
      imagePath: fixtures.imagePath,
      promptType: 'document',
      baseSize: 1024,
      imageSize: 640,
      cropMode: true
    });
    expect(imageResult.success).toBeTruthy();
    expect(imageResult.data.status).toBe('success');
    expect(imageResult.data.result).toContain('Mock');

    const pdfResult = await invokeIpc(window, 'perform-ocr', {
      imagePath: fixtures.pdfPath,
      promptType: 'document',
      baseSize: 1024,
      imageSize: 640,
      cropMode: true
    });
    expect(pdfResult.success).toBeTruthy();
    expect(pdfResult.data.status).toBe('success');
  });

  test('processes mixed queue and reports page progress detail', async () => {
    const addResult = await invokeIpc(window, 'add-to-queue', {
      filePaths: [fixtures.imagePath, fixtures.pdfPath],
      promptType: 'document',
      baseSize: 1024,
      imageSize: 640,
      cropMode: true
    });
    expect(addResult.success).toBeTruthy();
    expect(addResult.data.added).toBe(2);

    const processPromise = invokeIpc(window, 'process-queue');

    await expect
      .poll(async () => {
        const status = await invokeIpc(window, 'get-queue-status');
        if (!status.success) {
          return '';
        }
        const active = status.data.items.find((item) => item.status === 'processing');
        return active?.progress_detail || '';
      })
      .toMatch(/Page \d+\/\d+/);

    const processResult = await processPromise;
    expect(processResult.success).toBeTruthy();
    expect(processResult.data.total).toBe(2);

    const finalStatus = await invokeIpc(window, 'get-queue-status');
    expect(finalStatus.success).toBeTruthy();
    expect(finalStatus.data.completed).toBe(2);
  });
});
