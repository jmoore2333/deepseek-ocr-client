const { contextBridge, ipcRenderer } = require('electron');
const path = require('path');
const { marked } = require('marked');

marked.setOptions({
  mangle: false,
  headerIds: false,
  breaks: true
});

contextBridge.exposeInMainWorld('appAPI', {
  onStartupStatus: (callback) => {
    if (typeof callback !== 'function') {
      return () => {};
    }
    const listener = (_event, status) => callback(status);
    ipcRenderer.on('startup-status', listener);
    return () => ipcRenderer.removeListener('startup-status', listener);
  },

  getStartupStatus: () => ipcRenderer.invoke('get-startup-status'),
  checkServerStatus: () => ipcRenderer.invoke('check-server-status'),
  loadModel: () => ipcRenderer.invoke('load-model'),
  getModelInfo: () => ipcRenderer.invoke('get-model-info'),

  selectImage: () => ipcRenderer.invoke('select-image'),
  selectImages: () => ipcRenderer.invoke('select-images'),
  selectFolder: () => ipcRenderer.invoke('select-folder'),
  listFolderInputs: (folderPath) => ipcRenderer.invoke('list-folder-inputs', folderPath),
  openFolder: (folderPath) => ipcRenderer.invoke('open-folder', folderPath),

  performOCR: (payload) => ipcRenderer.invoke('perform-ocr', payload),

  addToQueue: (payload) => ipcRenderer.invoke('add-to-queue', payload),
  getQueueStatus: () => ipcRenderer.invoke('get-queue-status'),
  processQueue: () => ipcRenderer.invoke('process-queue'),
  clearQueue: () => ipcRenderer.invoke('clear-queue'),
  removeFromQueue: (itemId) => ipcRenderer.invoke('remove-from-queue', itemId),
  pauseQueue: () => ipcRenderer.invoke('pause-queue'),
  resumeQueue: () => ipcRenderer.invoke('resume-queue'),
  cancelQueue: () => ipcRenderer.invoke('cancel-queue'),
  retryFailedQueue: () => ipcRenderer.invoke('retry-failed-queue'),

  runPreflightCheck: () => ipcRenderer.invoke('run-preflight-check'),
  exportDiagnostics: () => ipcRenderer.invoke('export-diagnostics'),
  saveDocumentZip: (markdownText) => ipcRenderer.invoke('save-document-zip', { markdownText }),

  getRetentionPolicy: () => ipcRenderer.invoke('get-retention-policy'),
  updateRetentionPolicy: (policy) => ipcRenderer.invoke('update-retention-policy', policy),
  applyRetentionCleanup: () => ipcRenderer.invoke('apply-retention-cleanup'),

  renderMarkdown: (markdownText) => marked.parse(markdownText || ''),
  basename: (filePath) => path.basename(filePath || ''),
  extname: (filePath) => path.extname(filePath || '').toLowerCase()
});
