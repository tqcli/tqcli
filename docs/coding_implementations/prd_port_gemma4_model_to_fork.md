# Product Requirements Document (PRD): Port Mainline Gemma 4 Model to vLLM-TurboQuant Fork

## 1. Introduction
**Port Mainline Gemma 4 Model to vLLM-TurboQuant Fork** is a targeted code port that copies the dedicated Gemma 4 model implementation from mainline vLLM (`vllm-project/vllm`) into the TurboQuant fork (`ithllc/vllm-turboquant`). The fork currently routes Gemma 4 through a generic Transformers-based backend that assumes uniform head sizes across all attention layers, causing a fatal shape mismatch on Gemma 4's mixed `head_dim=256` / `global_head_dim=512` architecture. The mainline has a dedicated model file that handles per-layer head dimension selection, which this port brings into the fork.

## 2. Target Audience
- **tqCLI developers:** Need Gemma 4 E2B running on vLLM with TurboQuant KV cache compression
- **Low-VRAM users (4 GB GPUs):** Benefit from CPU offloading + BNB INT4 + turboquant35 KV to run Gemma 4 on constrained hardware
- **vLLM-TurboQuant fork maintainers:** Need the fork to stay compatible with new model architectures without full rebase

## 3. Scope & Constraints
- **Source repo:** `vllm-project/vllm` (main branch)
- **Target repo:** `ithllc/vllm-turboquant` (built from `mitkox/vllm-turboquant`)
- **Fork base version:** vLLM 0.1.dev5 (CUDA 12.8, SM86)
- **Hardware:** NVIDIA RTX A2000 Laptop GPU (4 GB VRAM), 32 GB RAM, WSL2
- **CUDA:** 12.8

**In Scope:**
- Port dedicated `gemma4.py` and `gemma4_mm.py` model files from mainline
- Port `gemma4_rope.py` rotary embedding if required by the model
- Add `Gemma4ModelArchConfigConvertor` with `max(head_dim, global_head_dim)` override
- Register Gemma 4 in the vLLM model registry so it bypasses the generic Transformers backend
- Include buffer loading fix for `Gemma4ClippableLinear` (audio/vision tower buffers)
- Include BNB loader fix for `None` weight mapping
- Verify TurboQuant turboquant35 KV cache works with the ported model
- Rebuild fork from source and install
- E2E test: Gemma 4 E2B with BNB_INT4 + cpu_offload 2.1 GB + turboquant35 KV

**Out of Scope:**
- Rebasing the entire fork onto a newer mainline vLLM version
- Multimodal (image/video/audio) inference testing — text-only verification
- Porting Gemma 4 reasoning parser or tool parser (nice-to-have, not blocking)
- Supporting Gemma 4 31B or 26B-A4B MoE variants
- Changes to the tqCLI pipeline logic (already works correctly)

## 4. Key Features

### 4.1. Dedicated Gemma 4 Model File (`gemma4.py`)
- Per-layer head dimension selection based on `config.layer_types[layer_idx]`
- Sliding attention layers use `head_dim=256`, full attention layers use `global_head_dim=512`
- Direct per-layer input projection (avoids O(seq*vocab*hidden) reverse embedding OOM)
- QKV projections sized correctly per layer type
- **User Interaction:** None — internal vLLM model routing

### 4.2. Multimodal Wrapper (`gemma4_mm.py`)
- Wraps the text model with vision and audio encoder towers
- Handles multimodal input merging (even though we only test text-only)
- Required for model initialization since the HF config declares `Gemma4ForConditionalGeneration`
- **AI/Models:** google/gemma-4-E2B (2.6B effective parameters)

### 4.3. Model Architecture Config Convertor
- `Gemma4ModelArchConfigConvertor.get_head_size()` returns `max(head_dim, global_head_dim) = 512`
- Ensures KV cache buffers are allocated large enough for full attention layers
- `get_total_num_kv_heads()` handles Gemma 4's GQA with 1 KV head per layer

### 4.4. Model Registry Wiring
- Register `Gemma4ForConditionalGeneration` → `gemma4_mm.py` in the model registry
- Register `Gemma4ForCausalLM` → `gemma4.py`
- Ensures vLLM uses the dedicated model file instead of the generic Transformers backend

### 4.5. Dependency Fixes (from #20 investigation)
- Buffer loading in `utils.py` for registered buffers (Gemma4ClippableLinear)
- BNB loader `None` mapping skip in `bitsandbytes_loader.py`
- TurboQuant metadata generation for Gemma 4 (35 layers, `{i}.attn` or model-specific naming)

## 5. User Stories
1. *As a tqCLI user with a 4 GB GPU, I want Gemma 4 E2B to load on vLLM so I can use the TurboQuant KV compression path for longer context.*
2. *As a fork maintainer, I want Gemma 4 support added via isolated model files so it doesn't risk breaking existing Qwen3 AWQ inference.*
3. *As a developer, I want the fork to handle variable head dimensions so future models with similar architectures work without additional patches.*
4. *As a CI maintainer, I want all integration tests passing after the port so I know the fork is stable.*

## 6. Technical Requirements
- **Runtime:** Python 3.10+, CUDA 12.8, NVIDIA driver 570+
- **Build:** `CUDA_HOME=/usr/local/cuda-12.8 VLLM_TARGET_DEVICE=cuda MAX_JOBS=2 pip install . --no-build-isolation`
- **Packages:** transformers>=5.5.0, bitsandbytes>=0.49.2
- **VRAM budget:** 4,096 MB total, 2,536 MB available for model after CUDA/runtime overhead
- **CPU offload:** 2.1 GB to system RAM for Gemma 4 E2B INT4
- **KV cache:** turboquant35 dtype (4.6x compression)
- **Privacy:** All inference local, no data leaves the machine

## 7. Success Metrics
- **Model load:** Gemma 4 E2B loads on vLLM with BNB_INT4 + cpu_offload_gb=2.1 + turboquant35 KV
- **Inference quality:** Multi-turn chat produces coherent, on-topic responses
- **Regression:** Qwen3 AWQ Tests 5-7 on vLLM all PASS
- **Integration suite:** All tests PASS
- **Performance:** Benchmark captured (tok/s with Gemma 4 E2B on vLLM)
- **Fork updated:** Changes pushed to `ithllc/vllm-turboquant`

---

**Next steps:** Run `/technical-planner` to generate the phased implementation plan from this PRD.
