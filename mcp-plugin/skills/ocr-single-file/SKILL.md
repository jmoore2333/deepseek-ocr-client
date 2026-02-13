---
name: ocr-single-file
description: Process a single image or PDF file with DeepSeek OCR. Use when a user asks to extract text from a document, image, or PDF.
---

When processing a single file with DeepSeek OCR:

1. Read `deepseek-ocr://health` to check if the backend is running
   - If unreachable, tell the user to start the DeepSeek OCR Client app first
2. If model is not loaded, call `model_load` and poll `deepseek-ocr://progress` until status is "loaded"
3. Determine quality based on user request:
   - **Fast**: `base_size=512, image_size=384, crop_mode=true` — quick scan, lower accuracy
   - **Balanced** (default): `base_size=1024, image_size=640, crop_mode=true` — good accuracy/speed tradeoff
   - **Quality**: `base_size=2048, image_size=1280, crop_mode=false` — highest accuracy, slower
4. Call `ocr_process_file` with the file path and chosen parameters
5. For PDFs, ask if the user wants specific pages (e.g., `pdf_page_range="1-3,5"`) or all pages
6. Return the extracted text to the user
