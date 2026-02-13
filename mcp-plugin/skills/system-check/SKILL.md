---
name: system-check
description: Run a full diagnostic check on the DeepSeek OCR system. Use when troubleshooting issues, checking setup status, or verifying the application is ready.
---

Run a comprehensive diagnostic check on the DeepSeek OCR application:

1. Read `deepseek-ocr://preflight` — check disk space and setup status
2. Read `deepseek-ocr://health` — check backend health and device status
3. Read `deepseek-ocr://model-info` — check model configuration
4. Read `deepseek-ocr://diagnostics` — get full runtime diagnostics
5. Read `deepseek-ocr://queue/status` — check queue state
6. Read `deepseek-ocr://retention-policy` — check cleanup settings

Report a clear summary covering:
- **Backend**: running or stopped
- **Model**: loaded or not, which model, which device (CUDA/MPS/CPU)
- **Hardware**: GPU availability, device type, GPU name if CUDA
- **Disk**: free space, model cache location
- **Queue**: current state (idle, processing, items pending/completed/failed)
- **Retention**: current cleanup policy settings
- **Issues**: any warnings or problems detected

If backend is unreachable, advise the user to start the DeepSeek OCR Client application.
