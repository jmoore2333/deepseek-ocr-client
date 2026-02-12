# DeepSeek-OCR Client

A real-time Electron-based desktop GUI for [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)

**Unaffiliated with [DeepSeek](https://www.deepseek.com/)**

## Features

- Drag-and-drop image upload
- Real-time OCR processing
- **NEW: Queue processing** - Process multiple files or entire folders automatically
- **NEW: CUDA optimizations** - Up to 6x faster with GPU acceleration

<img src="docs/images/document.gif" width="1000">

- Click regions to copy 
- Export results as ZIP with markdown images
- GPU acceleration (CUDA) with torch.compile optimization
- **NEW: Auto-save** - Queue results organized in timestamped folders

<img src="docs/images/document2.png" width="1000">

## Requirements

- Windows 10/11, other OS are experimental
- Node.js 18+ ([download](https://nodejs.org/))
- Python 3.10-3.12 ([download](https://www.python.org/)) - required for source/dev mode
- NVIDIA GPU with CUDA (optional but recommended for 6x speedup)

## Packaged Installer Runtime

Packaged builds now bundle `uv` and perform deferred first-run setup:

- Detect hardware target (`Apple Silicon MPS`, `NVIDIA CUDA`, or `CPU`)
- Install standalone Python into app data
- Create app-managed virtual environment
- Install the matching PyTorch variant and remaining dependencies

This means end users do not need a preinstalled Python when using installers.

## Build Installers

Windows (NSIS):

- `powershell -ExecutionPolicy Bypass -File .\scripts\build-release.ps1`

macOS/Linux:

- `bash ./scripts/build-release.sh`

Manual commands:

- `npm run dist:win`
- `npm run dist:mac`
- `npm run dist:linux`

The installer is generated in `dist/` as an `.exe` file.
On first app launch, the packaged build creates a Python environment in app data
and installs dependencies automatically.

## Quick Start (Windows)

1. **Install Python 3.10-3.12** if not already installed ([Python 3.10 recommended](https://www.python.org/ftp/python/3.10.14/python-3.10.14-amd64.exe))
   - ⚠️ **Important**: Python 3.13+ is not supported (PyTorch limitation)
   - Source launcher scripts detect compatible Python versions automatically
2. **Extract** the [ZIP file](https://github.com/ihatecsv/deepseek-ocr-client/archive/refs/heads/main.zip)
3. **Run** `start-client.bat`
   - First run will automatically:
     - Create a Python virtual environment
     - Install PyTorch with CUDA support
     - Install all dependencies
   - Subsequent runs will start quicker
4. **Load Model** - Click the "Load Model" button in the app (downloads model on first run)
5. **Drop an image** or click the drop zone to select one
6. **Run OCR** - Click "Run OCR" to process

Note: if you have issues processing images but the model loads properly, please close and re-open the app and try with the default resolution for "base" and "size". This is a [known issue](https://github.com/ihatecsv/deepseek-ocr-client/issues/2), if you can help to fix it I would appreciate it!

## Linux/macOS

Use `start-client.sh` for source mode.

For packaged builds, run `bash ./scripts/build-release.sh` on Linux/macOS to generate platform installers (`AppImage`/`deb` on Linux, `dmg` on macOS).

## Links

- [Model HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [Model Blog Post](https://deepseek.ai/blog/deepseek-ocr-context-compression)
- [Model GitHub](https://github.com/deepseek-ai/DeepSeek-OCR)

## Future goals (PRs welcome!)

- [ ] Code cleanup needed (quickly put together)
- [ ] TypeScript
- [ ] Updater from GitHub releases
- [ ] PDF support
- [ ] Batch processing
- [ ] CPU support?
- [ ] Web version (so you can run the server on a different machine)
- [ ] Better progress bar algo
- [ ] ???

## License

MIT
