#!/usr/bin/env node
/* eslint-disable no-console */
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const projectRoot = path.resolve(__dirname, '..');
const distDir = path.join(projectRoot, 'dist');

function runNpmScript(scriptName) {
  const npmCommand = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  const result = spawnSync(npmCommand, ['run', scriptName], {
    cwd: projectRoot,
    stdio: 'inherit',
    env: {
      ...process.env,
      CSC_IDENTITY_AUTO_DISCOVERY: 'false'
    }
  });
  if (result.status !== 0) {
    process.exit(result.status || 1);
  }
}

function listDistFiles() {
  if (!fs.existsSync(distDir)) {
    return [];
  }
  const files = [];
  for (const entry of fs.readdirSync(distDir, { withFileTypes: true })) {
    if (entry.isFile()) {
      files.push(entry.name);
    }
  }
  return files;
}

function assertHostArtifacts() {
  const files = listDistFiles();
  if (!files.length) {
    throw new Error('dist/ is empty after host distribution build');
  }

  let expected;
  if (process.platform === 'win32') {
    expected = files.find((name) => name.endsWith('.exe'));
  } else if (process.platform === 'darwin') {
    expected = files.find((name) => name.endsWith('.dmg') || name.endsWith('.zip'));
  } else {
    expected = files.find((name) => name.endsWith('.AppImage') || name.endsWith('.deb'));
  }

  if (!expected) {
    throw new Error(`No expected distribution artifact found in dist/. Files: ${files.join(', ')}`);
  }

  console.log(`[dist-host] Found distribution artifact: dist/${expected}`);
}

function getHostDistScript() {
  if (process.platform === 'win32') return 'dist:win';
  if (process.platform === 'darwin') return 'dist:mac';
  return 'dist:linux';
}

const script = getHostDistScript();
console.log(`[dist-host] Running ${script}...`);
runNpmScript(script);
assertHostArtifacts();
console.log('[dist-host] Host distribution build passed.');
