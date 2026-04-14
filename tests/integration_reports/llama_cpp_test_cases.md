# tqCLI llama.cpp Backend Test Cases

**Status:** EXECUTED — see `llama_cpp_test_report.md` and `llama_cpp_test_report.json` for results
**Backend:** llama.cpp (llama-cpp-python >=0.3.20)
**Prerequisites:** Linux, macOS, or Windows with CPU or NVIDIA GPU (4+ GB VRAM recommended)

> These test cases cover the llama.cpp inference engine across single-process and multi-process modes.
> llama.cpp supports: CPU-only, NVIDIA GPU (CUDA), Apple Metal, and Vulkan acceleration.
> WSL2 with GPU passthrough is supported.

---

## Prerequisites

```bash
# Install tqCLI with llama.cpp backend
pip install -e ".[server]"

# Verify llama-cpp-python is available
python -c "from llama_cpp import Llama; print('llama-cpp-python OK')"

# Verify GPU is detected (optional, for GPU-accelerated inference)
nvidia-smi
```

### Hardware Requirements for llama.cpp Tests

| Model | Min VRAM | Recommended VRAM | Notes |
|-------|----------|------------------|-------|
| Gemma 4 E4B (4.5B) | 3 GB | 4 GB | GGUF Q4_K_M quantization |
| Gemma 4 E2B (2.3B) | 2 GB | 3 GB | GGUF Q4_K_M quantization |
| Qwen 3 4B | 3 GB | 4 GB | GGUF Q4_K_M quantization |
| Qwen 3 8B | 6 GB | 8 GB | GGUF Q4_K_M quantization |

> **Important:** llama.cpp uses GGUF format. Models are single-file downloads from HuggingFace (typically `unsloth/` or `Qwen/` repos). The model registry selects the best model that fits within available VRAM.

---

## Test 1 (llama.cpp): Gemma 4 Full Lifecycle

### 1.1 Install tqCLI with llama.cpp

```bash
pip install -e ".[server]"
tqcli --version
tqcli system info --json
```

**Expected:** Version 0.3.1+, system info shows `recommended_engine: "llama.cpp"` (for GPUs with <8 GB VRAM)

### 1.2 Hardware Model Selection

```python
from tqcli.core.model_registry import get_registry
from tqcli.core.system_info import detect_system

sys_info = detect_system()
registry = get_registry()
models = registry.get_fitting_models(family="gemma4", vram_mb=sys_info.vram_mb)
best = registry.select_best(models)
print(f"Selected: {best.id} ({best.parameter_count}, {best.quantization})")
print(f"Fitting models: {len(models)} of {len(registry.get_family('gemma4'))}")
```

**Expected:**
- For 4 GB VRAM: Selects `gemma-4-e4b-it-Q4_K_M` from 2 fitting models (E4B and E2B)
- E4B chosen over E2B because it has higher strength scores while fitting within VRAM
- Total Gemma 4 models in registry: 4 (E2B, E4B, 26B MoE, 31B)

### 1.3 Download Gemma 4 Model

```bash
tqcli model pull gemma-4-e4b-it-Q4_K_M
```

**Verify:**
- Model downloaded from `unsloth/gemma-4-E4B-it-GGUF` HuggingFace repo
- GGUF file: `gemma-4-E4B-it-Q4_K_M.gguf`
- Stored at `~/.tqcli/models/`
- File size: ~4,981 MB

### 1.4 Verify TurboQuant Quantization

```bash
tqcli model list
```

**Expected:** Model shows:
- Quantization: Q4_K_M (4-bit k-quant, medium)
- Format: GGUF
- Engine: llama.cpp

### 1.5 Load Model

```python
from tqcli.core.llama_backend import LlamaCppEngine

engine = LlamaCppEngine()
engine.load(model_path="~/.tqcli/models/gemma-4-E4B-it-Q4_K_M.gguf", n_gpu_layers=-1)
```

**Expected:** Model loads in ~4s with full GPU offload (`n_gpu_layers=-1`)

### 1.6 Two-Turn Chat Test

```bash
tqcli chat --model gemma-4-e4b-it-Q4_K_M
```

**Turn 1:** "What is the capital of France? Answer in one sentence."
- **Expected:** Correct factual answer mentioning Paris

**Turn 2:** "What is the population of that city? Just give the number."
- **Expected:** Approximate population of Paris (~2.1 million or ~12 million metro)

**Metrics to capture:**
- Tokens per second (expected 2-5 tok/s on 4 GB VRAM)
- Time to first token (expected 1.5-3s)
- Total generation time

### 1.7 Image Input Test

```bash
# In chat session:
/image /path/to/test_image.png What colors do you see in this image?
```

**Expected:** Model identifies colors in the test image (red square with blue border)

**Note:** llama.cpp multimodal requires CLIP model auto-detection. Gemma 4 E4B supports vision (`multimodal=True`). Image encoding adds ~8-10s overhead.

### 1.8 Audio Input Test

```bash
# In chat session:
/audio /path/to/test_audio.wav Describe what you hear.
```

**Expected:** Model responds gracefully, likely indicating no audio processing capability at this quantization level. This is acceptable behavior.

### 1.9 Generate and Verify Skill

```bash
tqcli skill create test-gemma4-skill -d "Test skill for Gemma 4 on llama.cpp"
tqcli skill run test-gemma4-skill
tqcli skill list
```

**Expected:** Skill created with SKILL.md and script, executes successfully, returns JSON result

### 1.10 Remove Model and Clean Uninstall

```bash
tqcli model remove gemma-4-e4b-it-Q4_K_M
tqcli model list               # Model should be gone
pip show tqcli                  # Should show installed
pip uninstall tqcli -y
pip show tqcli                  # Should fail (not found)
```

---

## Test 2 (llama.cpp): Qwen 3 Full Lifecycle

### 2.1 Install and Download

```bash
pip install -e ".[server]"
tqcli model pull qwen3-4b-Q4_K_M
```

**Expected:**
- Model downloaded from `Qwen/Qwen3-4B-GGUF` HuggingFace repo
- GGUF file: `Qwen3-4B-Q4_K_M.gguf`
- File size: ~2,382 MB
- Only 1 Qwen 3 model fits 4 GB VRAM (8B, 32B, 30B-A3B exceed limits)

### 2.2 Verify Quantization

```bash
tqcli model list
```

**Expected:** Q4_K_M quantization, GGUF format

### 2.3 Two-Turn Chat Test

```bash
tqcli chat --model qwen3-4b-Q4_K_M
```

**Turn 1:** "What is 2 + 2? Answer with just the number."
- **Expected:** "4" (Qwen 3 activates `<think>...</think>` blocks for math — produces longer output with reasoning chain)

**Turn 2:** "Now multiply that result by 10. Answer with just the number."
- **Expected:** "40" (with thinking chain)

**Metrics to capture:**
- Tokens/second (expected 6-9 tok/s — higher than Gemma 4 due to smaller model)
- Completion tokens (expected ~1024 due to thinking mode)
- Time to first token (expected 2-40s depending on thinking depth)

### 2.4 Image Input Test

**Expected:** FAIL (expected) — Qwen 3 is text-only (`multimodal=False`). Should fail gracefully with context overflow or clear error message.

### 2.5 Audio Input Test

**Expected:** FAIL (expected) — Qwen 3 is text-only. Should fail gracefully.

### 2.6 Generate Skill, Remove Model, Uninstall

```bash
tqcli skill create test-qwen3-skill -d "Test skill for Qwen 3 on llama.cpp"
tqcli skill run test-qwen3-skill
tqcli skill list
tqcli model remove qwen3-4b-Q4_K_M
pip uninstall tqcli -y
```

---

## Test 3 (llama.cpp): Gemma 4 Multi-Process + Yolo Mode + CRM Build

### 3.1 Setup

```bash
pip install -e ".[server]"
tqcli model pull gemma-4-e4b-it-Q4_K_M
```

### 3.2 Assess Multi-Process Feasibility

```python
from tqcli.core.multiprocess import assess_multiprocess
from tqcli.core.system_info import detect_system

sys_info = detect_system()
plan = assess_multiprocess(
    sys_info=sys_info,
    model_path="/path/to/model",
    model_size_mb=4981,
    requested_workers=3,
    preferred_engine="llama.cpp",
    unrestricted=True,
)
print(f"Feasible: {plan.feasible}, Engine: {plan.engine}, Max workers: {plan.max_workers}")
```

**Expected:** llama.cpp server mode, max 4 workers, recommended 2 workers

### 3.3 Start llama.cpp Server (Yolo Mode)

```bash
tqcli --stop-trying-to-control-everything-and-just-let-go serve start -m gemma-4-e4b-it-Q4_K_M
```

**Expected:** llama.cpp HTTP server starts on port 8741
- Server runs as background process with reported PID
- Serves requests sequentially (not batched — llama.cpp limitation)
- Workers queue when server is busy

### 3.4 Check Server Status

```bash
tqcli serve status
```

**Expected:** Shows running status with PID, engine=llama.cpp, health=OK

### 3.5 Generate CRM Skills and Workspace

```bash
tqcli skill create crm-frontend -d "Generate HTML/CSS/JS frontend for CRM"
tqcli skill create crm-backend -d "Generate Flask backend API for CRM"
tqcli skill create crm-database -d "Generate SQLite schema for CRM"

# Create workspace
mkdir -p /llm_models_python_code_src/crm_workspace/{frontend,backend,database}
```

### 3.6 Verify CRM Workspace

```bash
ls -la /llm_models_python_code_src/crm_workspace/frontend/index.html
ls -la /llm_models_python_code_src/crm_workspace/backend/app.py
ls -la /llm_models_python_code_src/crm_workspace/database/schema.sql
```

**Expected:** All 3 files present:
- `frontend/index.html` — HTML/CSS/JS CRM with contact table and add form
- `backend/app.py` — Flask REST API with GET/POST `/api/contacts`
- `database/schema.sql` — SQLite schema with contacts table, indexes

### 3.7 Cleanup

```bash
rm -rf /llm_models_python_code_src/crm_workspace
tqcli serve stop
tqcli --stop-trying-to-control-everything-and-just-let-go model remove gemma-4-e4b-it-Q4_K_M
pip uninstall tqcli -y
```

---

## Test 4 (llama.cpp): Qwen 3 Multi-Process + Yolo Mode + CRM Build

### 4.1 Setup

```bash
pip install -e ".[server]"
tqcli model pull qwen3-4b-Q4_K_M
```

### 4.2 Assess and Start llama.cpp Server

```bash
tqcli --stop-trying-to-control-everything-and-just-let-go serve start -m qwen3-4b-Q4_K_M
tqcli serve status
```

**Expected:** llama.cpp server on port 8741, status running, health OK

### 4.3 Generate CRM Skills and Workspace

Same as Test 3 steps 3.5-3.6 but using Qwen 3 as the inference model.

### 4.4 Cleanup

```bash
rm -rf /llm_models_python_code_src/crm_workspace
tqcli serve stop
tqcli --stop-trying-to-control-everything-and-just-let-go model remove qwen3-4b-Q4_K_M
pip uninstall tqcli -y
```

---

## Pre-Test Checklist (llama.cpp)

Before executing these tests, verify:

- [ ] Python 3.10+ installed
- [ ] `pip install llama-cpp-python[server]>=0.3.20` succeeds
- [ ] `python -c "from llama_cpp import Llama"` works
- [ ] GPU available (optional but recommended): `nvidia-smi` or Metal-capable Mac
- [ ] Model registry has GGUF profiles with correct `hf_repo` and `filename` values
- [ ] Sufficient disk space for model downloads (~5 GB for E4B, ~2.4 GB for Qwen 3 4B)

## Expected Performance Baselines (llama.cpp)

| Metric | Gemma 4 E4B (4 GB VRAM) | Qwen 3 4B (4 GB VRAM) |
|--------|--------------------------|------------------------|
| Tokens/second (text) | 2-5 | 6-9 |
| Tokens/second (image) | 1-2 | N/A (text-only) |
| Time to first token | 1.5-3s | 2-40s (thinking mode) |
| Model load time | ~4s | ~3s |
| Model size on disk | ~4,981 MB | ~2,382 MB |
| Multi-process mode | Sequential queue | Sequential queue |
| Max concurrent workers | 4 (CPU-limited) | 4 (CPU-limited) |

## Model Registry Profiles Used

```python
ModelProfile(
    id="gemma-4-e4b-it-Q4_K_M",
    family="gemma4",
    display_name="Gemma 4 E4B Edge Instruct (Q4_K_M)",
    hf_repo="unsloth/gemma-4-E4B-it-GGUF",
    filename="gemma-4-E4B-it-Q4_K_M.gguf",
    parameter_count="4.5B",
    quantization="Q4_K_M",
    format="gguf",
    context_length=128000,
    engine="llama.cpp",
    min_ram_mb=4000,
    min_vram_mb=3000,
    supports_thinking=True,
    multimodal=True,
),
ModelProfile(
    id="qwen3-4b-Q4_K_M",
    family="qwen3",
    display_name="Qwen3 4B (Q4_K_M)",
    hf_repo="Qwen/Qwen3-4B-GGUF",
    filename="Qwen3-4B-Q4_K_M.gguf",
    parameter_count="4B",
    quantization="Q4_K_M",
    format="gguf",
    context_length=32768,
    engine="llama.cpp",
    min_ram_mb=3500,
    min_vram_mb=3000,
    supports_thinking=True,
    multimodal=False,
),
```
