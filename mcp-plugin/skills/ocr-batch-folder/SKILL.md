---
name: ocr-batch-folder
description: Process multiple images and PDFs in a folder using the queue system. Use when a user wants to OCR many files at once or process an entire directory.
---

When batch processing files with DeepSeek OCR:

1. Read `deepseek-ocr://health` to check if the backend is running
   - If unreachable, tell the user to start the DeepSeek OCR Client app first
2. If model is not loaded, call `model_load` and poll `deepseek-ocr://progress` until loaded
3. List all supported image and PDF files in the target folder (`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.tif`, `.webp`, `.pdf`)
4. Call `queue_add_files` with the list of file paths
5. Call `queue_start_processing` to begin (this returns immediately)
6. Poll with `queue_wait_for_completion` — use `timeout_seconds=1800` for large batches
7. Read `deepseek-ocr://queue/status` for the final results
8. Report summary:
   - Total files processed
   - Successful / failed count
   - Output folder location
   - Any errors encountered

If the user wants to pause, resume, or cancel during processing, use the corresponding queue tools.
