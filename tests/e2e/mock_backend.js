#!/usr/bin/env node
/* eslint-disable no-console */
const http = require('http');
const os = require('os');
const path = require('path');
const { URL } = require('url');

const PORT = Number(process.env.MOCK_BACKEND_PORT || process.env.PORT || 5000);
const MOCK_QUEUE_FOLDER = path.join(os.tmpdir(), 'deepseek-ocr-mock-queue');

let modelLoaded = false;
let modelLoading = false;
let queueProcessing = false;
let queueItems = [];
let nextQueueId = 0;

let progressData = {
  status: 'idle',
  stage: '',
  message: '',
  progress_percent: 0,
  chars_generated: 0,
  raw_token_stream: '',
  timestamp: Date.now() / 1000
};

function nowSeconds() {
  return Date.now() / 1000;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function updateProgress(status, stage = '', message = '', progressPercent = 0, charsGenerated = 0, rawTokenStream = '') {
  progressData = {
    status,
    stage,
    message,
    progress_percent: Math.max(0, Math.min(100, Math.round(progressPercent))),
    chars_generated: charsGenerated,
    raw_token_stream: rawTokenStream,
    timestamp: nowSeconds()
  };
}

function parseMultipartField(bodyText, fieldName, fallback = '') {
  const pattern = new RegExp(`name="${fieldName}"\\r\\n\\r\\n([^\\r\\n]*)`, 'i');
  const match = pattern.exec(bodyText);
  return match ? match[1].trim() : fallback;
}

function parseMultipartFilenames(bodyText) {
  const names = [];
  const regex = /filename="([^"]+)"/gi;
  let match = regex.exec(bodyText);
  while (match) {
    names.push(match[1]);
    match = regex.exec(bodyText);
  }
  return names;
}

function buildMockRawTokens() {
  return [
    '<|ref|>title<|/ref|><|det|>[[70, 70, 930, 180]]<|/det|>DeepSeek OCR Mock',
    '<|ref|>text<|/ref|><|det|>[[90, 230, 910, 380]]<|/det|>This is a synthetic OCR response for end-to-end testing.'
  ].join('\n');
}

function queueSummary() {
  const currentItem = queueItems.find((item) => item.status === 'processing');
  return {
    total: queueItems.length,
    pending: queueItems.filter((item) => item.status === 'pending').length,
    processing: queueItems.filter((item) => item.status === 'processing').length,
    completed: queueItems.filter((item) => item.status === 'completed').length,
    failed: queueItems.filter((item) => item.status === 'failed').length,
    items: queueItems.map((item) => ({
      id: item.id,
      filename: item.filename,
      status: item.status,
      progress: item.progress,
      progress_detail: item.progress_detail,
      error: item.error
    })),
    current_file: currentItem
      ? {
          id: currentItem.id,
          filename: currentItem.filename,
          image_path: currentItem.current_image_path,
          progress: currentItem.progress,
          progress_detail: currentItem.progress_detail
        }
      : null
  };
}

function jsonHeaders(statusCode = 200) {
  return {
    statusCode,
    headers: {
      'Content-Type': 'application/json; charset=utf-8',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET,POST,DELETE,OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization'
    }
  };
}

function sendJson(res, payload, statusCode = 200) {
  const meta = jsonHeaders(statusCode);
  res.writeHead(meta.statusCode, meta.headers);
  res.end(JSON.stringify(payload));
}

function sendNoContent(res) {
  res.writeHead(204, {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET,POST,DELETE,OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
  });
  res.end();
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    req.on('data', (chunk) => chunks.push(chunk));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

async function kickModelLoading() {
  if (modelLoaded || modelLoading) {
    return;
  }
  modelLoading = true;
  const stages = [
    { stage: 'tokenizer', message: 'Loading tokenizer...', percent: 20 },
    { stage: 'model', message: 'Downloading model files...', percent: 45 },
    { stage: 'gpu', message: 'Optimizing model on GPU...', percent: 70 },
    { stage: 'warmup', message: 'Running warmup inference...', percent: 92 }
  ];

  for (const item of stages) {
    updateProgress('loading', item.stage, item.message, item.percent);
    await sleep(120);
  }

  modelLoaded = true;
  modelLoading = false;
  updateProgress('loaded', 'complete', 'Model ready!', 100);
}

async function ensureModelLoaded() {
  if (modelLoaded) {
    return;
  }
  await kickModelLoading();
}

async function processMockQueue() {
  if (queueProcessing) {
    return null;
  }

  queueProcessing = true;
  const pendingItems = queueItems.filter((item) => item.status === 'pending');
  const results = [];
  const totalItems = Math.max(pendingItems.length, 1);
  const rawTokens = buildMockRawTokens();

  for (let itemIdx = 0; itemIdx < pendingItems.length; itemIdx += 1) {
    const item = pendingItems[itemIdx];
    const pageCount = item.filename.toLowerCase().endsWith('.pdf') ? 3 : 1;

    item.status = 'processing';
    item.error = null;

    for (let pageNum = 1; pageNum <= pageCount; pageNum += 1) {
      item.progress_detail = `Page ${pageNum}/${pageCount}`;
      item.current_image_path = null;

      for (let step = 1; step <= 5; step += 1) {
        const pageStepProgress = Math.round((((pageNum - 1) + step / 5) / pageCount) * 100);
        item.progress = Math.min(pageStepProgress, 95);

        const overallProgress = Math.round(((itemIdx + item.progress / 100) / totalItems) * 100);
        const charsGenerated = Math.round(rawTokens.length * (step / 5));

        updateProgress(
          'processing',
          'queue',
          `[${itemIdx + 1}/${pendingItems.length}] ${item.filename} (${item.progress_detail.toLowerCase()})`,
          overallProgress,
          charsGenerated,
          rawTokens.slice(0, charsGenerated)
        );

        await sleep(90);
      }
    }

    item.status = 'completed';
    item.progress = 100;
    item.progress_detail = null;
    item.result = `Mock queue OCR output for ${item.filename}`;
    item.current_image_path = null;

    results.push({
      id: item.id,
      filename: item.filename,
      status: 'completed',
      output_dir: path.join(MOCK_QUEUE_FOLDER, `item_${item.id}`)
    });
  }

  updateProgress('idle', '', '', 0, 0, '');
  queueProcessing = false;
  return results;
}

const server = http.createServer(async (req, res) => {
  try {
    const parsed = new URL(req.url, `http://127.0.0.1:${PORT}`);
    const pathname = parsed.pathname;

    if (req.method === 'OPTIONS') {
      sendNoContent(res);
      return;
    }

    if (req.method === 'GET' && pathname === '/health') {
      sendJson(res, {
        status: 'ok',
        model_loaded: modelLoaded,
        gpu_available: false,
        device_state: 'cpu'
      });
      return;
    }

    if (req.method === 'GET' && pathname === '/progress') {
      sendJson(res, progressData);
      return;
    }

    if (req.method === 'GET' && pathname === '/model_info') {
      sendJson(res, {
        model_name: 'mock/deepseek-ocr',
        cache_dir: path.join(os.tmpdir(), 'deepseek-ocr-mock-cache'),
        model_loaded: modelLoaded,
        gpu_available: false,
        device_state: 'cpu',
        gpu_name: null
      });
      return;
    }

    if (req.method === 'POST' && pathname === '/load_model') {
      kickModelLoading().catch((error) => {
        modelLoading = false;
        updateProgress('error', 'failed', error.message, 0);
      });
      sendJson(res, { status: 'success', message: 'Model loading started' });
      return;
    }

    if (req.method === 'POST' && pathname === '/ocr') {
      await ensureModelLoaded();
      const body = await readBody(req);
      const bodyText = body.toString('latin1');
      const promptType = parseMultipartField(bodyText, 'prompt_type', 'document');
      const filename = parseMultipartFilenames(bodyText)[0] || 'input.jpg';
      const isPdf = filename.toLowerCase().endsWith('.pdf');
      const pageCount = isPdf ? 3 : 1;
      const rawTokens = buildMockRawTokens();

      for (let step = 1; step <= 4; step += 1) {
        const chars = Math.round(rawTokens.length * (step / 4));
        updateProgress('processing', 'ocr', `Processing ${filename}...`, 15 + step * 20, chars, rawTokens.slice(0, chars));
        await sleep(70);
      }

      let resultText;
      if (promptType === 'document') {
        resultText = isPdf
          ? '## Page 1\n\nMock PDF OCR output.\n\n## Page 2\n\nMock PDF OCR output.'
          : '# Mock OCR Output\n\nThis is a mock markdown OCR result.';
      } else {
        resultText = isPdf
          ? 'Mock plain-text OCR output for PDF input.'
          : 'Mock plain-text OCR output for image input.';
      }

      updateProgress('idle', '', '', 0, 0, '');
      sendJson(res, {
        status: 'success',
        result: resultText,
        boxes_image_path: null,
        prompt_type: promptType,
        raw_tokens: rawTokens,
        is_pdf: isPdf,
        page_count: pageCount
      });
      return;
    }

    if (req.method === 'POST' && pathname === '/queue/add') {
      const body = await readBody(req);
      const bodyText = body.toString('latin1');
      const filenames = parseMultipartFilenames(bodyText);
      const promptType = parseMultipartField(bodyText, 'prompt_type', 'document');
      const baseSize = Number(parseMultipartField(bodyText, 'base_size', '1024'));
      const imageSize = Number(parseMultipartField(bodyText, 'image_size', '640'));
      const cropMode = parseMultipartField(bodyText, 'crop_mode', 'true').toLowerCase() === 'true';

      if (!filenames.length) {
        sendJson(res, { status: 'error', message: 'No files provided' }, 400);
        return;
      }

      const added = [];
      for (const filename of filenames) {
        const item = {
          id: nextQueueId,
          filename,
          status: 'pending',
          progress: 0,
          progress_detail: null,
          error: null,
          current_image_path: null,
          prompt_type: promptType,
          base_size: baseSize,
          image_size: imageSize,
          crop_mode: cropMode
        };
        nextQueueId += 1;
        queueItems.push(item);
        added.push({ id: item.id, filename: item.filename });
      }

      sendJson(res, {
        status: 'success',
        added: added.length,
        files: added,
        queue_length: queueItems.length
      });
      return;
    }

    if (req.method === 'GET' && pathname === '/queue/status') {
      sendJson(res, queueSummary());
      return;
    }

    if (req.method === 'POST' && pathname === '/queue/process') {
      await ensureModelLoaded();
      const results = await processMockQueue();
      if (results === null) {
        sendJson(res, { status: 'error', message: 'Queue already processing' }, 409);
        return;
      }
      const completed = results.filter((item) => item.status === 'completed').length;
      const failed = results.filter((item) => item.status === 'failed').length;
      sendJson(res, {
        status: 'success',
        queue_folder: MOCK_QUEUE_FOLDER,
        completed,
        failed,
        total: results.length,
        results
      });
      return;
    }

    if (req.method === 'POST' && pathname === '/queue/clear') {
      queueItems = [];
      updateProgress('idle', '', '', 0, 0, '');
      sendJson(res, { status: 'success', message: 'Queue cleared' });
      return;
    }

    if (req.method === 'DELETE' && pathname.startsWith('/queue/remove/')) {
      const idText = pathname.slice('/queue/remove/'.length);
      const itemId = Number(idText);
      const idx = queueItems.findIndex((item) => item.id === itemId);
      if (idx === -1) {
        sendJson(res, { status: 'error', message: 'Item not found' }, 404);
        return;
      }
      queueItems.splice(idx, 1);
      sendJson(res, { status: 'success', message: `Item ${itemId} removed` });
      return;
    }

    sendJson(res, { status: 'error', message: 'Not found' }, 404);
  } catch (error) {
    console.error('[mock-backend] request error:', error);
    sendJson(res, { status: 'error', message: error.message }, 500);
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Mock backend listening on http://127.0.0.1:${PORT}`);
});

process.on('SIGTERM', () => {
  server.close(() => process.exit(0));
});

process.on('SIGINT', () => {
  server.close(() => process.exit(0));
});
