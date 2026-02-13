# Apple Silicon (MPS) Model Evaluation

**Date:** 2026-02-13
**Current Issue:** MPS inference is significantly slower than CUDA despite appearing functional

---

## Current Setup

- **Model:** `Dogacel/DeepSeek-OCR-Metal-MPS` (HuggingFace)
- **Approach:** PyTorch MPS backend with float16
- **Modifications:** Uses eager attention (`_attn_implementation='eager'`) instead of flash attention
- **Parameters:** 3B, ~5GB memory footprint
- **Downloads:** ~351 on HuggingFace (low adoption)

### Observed Behavior
- Model loads and produces correct OCR output on Apple Silicon
- Inference speed is notably slow compared to CUDA
- PDF processing with MPS defaults was tightened in commit `3528eef` (generation token cap, reduced logging)
- The performance gap suggests the model/approach may not be optimal for Apple's unified memory architecture

---

## Candidate Alternatives

### 1. Official `deepseek-ai/DeepSeek-OCR` with Native MPS Support

**What changed:** A community PR adding MPS backend support was merged directly into the official DeepSeek-OCR model repository. This means the canonical model may now work on MPS without needing the Dogacel fork at all.

**Source:** [MPS support commit](https://huggingface.co/deepseek-ai/DeepSeek-OCR/commit/1e3401a3d4603e9e71ea0ec850bfead602191ec4), [Discussion #20](https://huggingface.co/deepseek-ai/DeepSeek-OCR/discussions/20)

**Pros:**
- Official model, well-maintained
- Single model for all devices (simplifies code)
- Potentially better optimized than third-party fork

**Cons:**
- MPS performance not independently benchmarked
- May have similar speed characteristics to Dogacel fork
- Needs testing to confirm it works without modification

**Effort:** Low - change model name in `ocr_server.py`, test

**Recommendation:** Test first. If it works and performs similarly or better than the Dogacel fork, switch to it to reduce the model selection complexity.

---

### 2. `Dogacel/Universal-DeepSeek-OCR-2`

**What is it:** The same author (Dogacel) has ported the newer DeepSeek-OCR-2 model to support CPU and MPS inference. DeepSeek-OCR-2 is a significantly improved version of the original model (released January 2026).

**Source:** [HuggingFace](https://huggingface.co/Dogacel/Universal-DeepSeek-OCR-2), [GitHub](https://github.com/Dogacel/Universal-DeepSeek-OCR-2)

**Pros:**
- Newer, more capable model (OCR-2 vs OCR-1)
- Same author's adaptation, familiar approach
- CPU + MPS support

**Cons:**
- Larger model (OCR-2 is bigger than OCR-1)
- May require different inference code
- Limited community testing

**Effort:** Medium - may need inference pipeline changes

**Recommendation:** Evaluate after testing option 1. If OCR-2 provides significantly better accuracy or speed, it's worth the integration work.

---

### 3. MLX-Based: `mlx-community/DeepSeek-OCR-8bit`

**What is it:** A quantized version of DeepSeek-OCR using Apple's MLX framework. MLX is Apple's native machine learning framework purpose-built for Apple Silicon's unified memory architecture.

**Source:** [MLX implementation discussion](https://github.com/deepseek-ai/DeepSeek-OCR/issues/145)

**Pros:**
- **Natively optimized for Apple Silicon** (MLX, not PyTorch MPS)
- 8-bit quantization: ~5.5GB memory (vs ~10GB full precision)
- Better performance expected since MLX understands unified memory
- Quality comparable to full precision at significantly lower memory

**Cons:**
- Requires replacing PyTorch inference pipeline with `mlx-vlm` on macOS
- Two separate inference paths (PyTorch for CUDA/CPU, MLX for macOS)
- Additional dependency: `mlx`, `mlx-vlm`
- Less mature ecosystem than PyTorch

**Effort:** High - new inference backend, platform-specific code paths

**Recommendation:** Long-term option. Best potential performance, but significant engineering effort. Consider after establishing performance baselines with options 1 and 2.

---

### 4. `matica0902/MLX-Video-OCR-DeepSeek-Apple-Silicon`

**What is it:** A full MLX-based application for OCR on Apple Silicon, built on DeepSeek-OCR + MLX ecosystem.

**Source:** [HuggingFace](https://huggingface.co/matica0902/MLX-Video-OCR-DeepSeek-Apple-Silicon)

**Pros:**
- Complete working solution for Apple Silicon
- Can reference implementation patterns

**Cons:**
- Separate application, not a drop-in model
- Would need to extract and adapt inference code
- Unknown maintenance status

**Effort:** Medium-high - extract patterns, adapt to our backend

**Recommendation:** Reference for patterns if we pursue MLX path (option 3), not a direct integration candidate.

---

## Evaluation Plan

### Phase 1: Quick Wins (Low Effort)

1. **Test official model on MPS:**
   ```python
   # In ocr_server.py, try using deepseek-ai/deepseek-ocr-base with MPS
   # instead of Dogacel/DeepSeek-OCR-Metal-MPS
   ```
   - Measure: load time, single image latency, memory usage
   - Compare against current Dogacel fork numbers
   - If better or equal: switch to official model for all devices

2. **Profile current MPS bottleneck:**
   ```python
   # Add timing instrumentation to ocr_server.py
   import torch.profiler
   # Profile: model loading, tokenization, generation, post-processing
   ```
   - Identify whether bottleneck is in attention, generation loop, or data transfer
   - May reveal quick optimization opportunities

### Phase 2: Model Upgrade (Medium Effort)

3. **Test Dogacel/Universal-DeepSeek-OCR-2:**
   - Download and test with existing inference pipeline
   - May need transformers version update
   - Compare accuracy and speed against v1

### Phase 3: Native Backend (High Effort, Best Potential)

4. **Prototype MLX inference path:**
   - Install `mlx` and `mlx-vlm` on macOS dev machine
   - Test `mlx-community/DeepSeek-OCR-8bit` inference standalone
   - Measure performance vs PyTorch MPS
   - If >2x faster: build MLX backend in ocr_server.py with runtime detection

---

## Benchmark Protocol

For each candidate, measure on the same Apple Silicon hardware:

| Metric | How |
|--------|-----|
| Model load time | Time from `load_model()` call to model ready |
| Single image (simple) | Time for `fixtures/images/simple_text.png` |
| Single image (complex) | Time for `fixtures/images/complex_layout.png` |
| PDF (5 pages) | Total time and per-page time |
| Peak memory | `torch.mps.current_allocated_memory()` or Activity Monitor |
| Accuracy | Compare OCR output against ground truth for test images |

Report results in a comparison table to make the decision clear.

---

## Implementation Changes Required

### For options 1 & 2 (model swap):
- Modify model selection logic in `backend/ocr_server.py` (~10-20 lines)
- Make model name configurable (environment variable or settings)
- Update `requirements.txt` if transformers version needs bumping

### For option 3 (MLX backend):
- Add `mlx>=0.18.0` and `mlx-vlm>=0.18.0` to requirements (macOS only)
- New inference function in `ocr_server.py` for MLX path
- Runtime detection: `if platform.system() == "Darwin" and mlx_available: use_mlx()`
- Shared pre/post-processing between PyTorch and MLX paths
- ~100-200 lines of new code

### Model selection refactor:
```python
# Current (hardcoded):
if device == "cuda":
    model_name = "deepseek-ai/deepseek-ocr-base"
else:
    model_name = "Dogacel/DeepSeek-OCR-Metal-MPS"

# Proposed (configurable):
MODEL_REGISTRY = {
    "default-cuda": "deepseek-ai/deepseek-ocr-base",
    "default-mps": "deepseek-ai/deepseek-ocr-base",  # or Dogacel variant
    "default-cpu": "Dogacel/DeepSeek-OCR-Metal-MPS",
    "dogacel-v2": "Dogacel/Universal-DeepSeek-OCR-2",
    "mlx-8bit": "mlx-community/DeepSeek-OCR-8bit",
}
model_name = os.environ.get("DEEPSEEK_OCR_MODEL") or MODEL_REGISTRY.get(f"default-{device}")
```
