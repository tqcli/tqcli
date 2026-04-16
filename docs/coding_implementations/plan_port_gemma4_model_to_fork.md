# Technical Implementation Plan: Port Mainline Gemma 4 Model to vLLM-TurboQuant Fork

**PRD:** `docs/coding_implementations/prd_port_gemma4_model_to_fork.md`
**Issue:** ithllc/tqCLI#21 (depends on #20)
**Date:** 2026-04-15

---

## Overview

Port 6-8 files from mainline vLLM (`vllm-project/vllm`) into the TurboQuant fork (`ithllc/vllm-turboquant`) to add a dedicated Gemma 4 model implementation. The fork currently has `gemma.py`, `gemma2.py`, `gemma3.py`, `gemma3_mm.py`, `gemma3n.py`, `gemma3n_mm.py` — but no `gemma4*` files. Gemma 4 falls through to the generic Transformers backend which crashes on the variable head dimension architecture.

## Architecture

```
                    Model Registry Lookup
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     gemma3.py       gemma4.py ← NEW   transformers/base.py
     (Gemma 3)       (Gemma 4)         (generic fallback)
            │              │
            ▼              ▼
     gemma3_mm.py    gemma4_mm.py ← NEW
     (MM wrapper)    (MM wrapper)
```

**Critical path:** Phase 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10

---

## Phase 1: Fetch Mainline Gemma 4 Files

### Objective
Download the dedicated Gemma 4 model files from mainline vLLM via `gh api`.

### Implementation Steps
1. Fetch each file from `vllm-project/vllm` main branch:
   ```bash
   gh api repos/vllm-project/vllm/contents/PATH --jq '.content' | base64 -d > /tmp/gemma4_port/FILENAME
   ```
2. Files to fetch:
   - `vllm/model_executor/models/gemma4.py` → text model with per-layer head_dim
   - `vllm/model_executor/models/gemma4_mm.py` → multimodal wrapper
   - `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py` → Gemma 4 RoPE
   - `vllm/model_executor/models/config.py` → check for Gemma 4 config entries
   - `vllm/transformers_utils/model_arch_config_convertor.py` → extract Gemma4 convertor class
   - `vllm/model_executor/models/registry.py` → extract Gemma 4 registration entries
   - `vllm/reasoning/gemma4_reasoning_parser.py` → optional, for thinking mode
   - `vllm/reasoning/gemma4_utils.py` → optional, shared reasoning utils

3. Save all files to `/tmp/gemma4_port/` for review before integration

### Files
- Source: mainline `vllm-project/vllm` (read-only)
- Destination: `/tmp/gemma4_port/` (staging)

### Dependencies
- None (first phase)

---

## Phase 2: Adapt Files for Fork Codebase

### Objective
Resolve import differences between mainline and fork, ensure compatibility with fork's v0.1.dev5 codebase.

### Implementation Steps
1. **Diff mainline vs fork API surfaces:**
   - Compare `from vllm.xxx import yyy` statements in fetched files against fork's available modules
   - Identify any imports that don't exist in the fork (added after v0.1.dev5)
   - Map each missing import to either: (a) equivalent fork import, (b) inline implementation, or (c) additional file to port

2. **Adapt `gemma4.py`:**
   - Verify `Attention`, `QKVParallelLinear`, `RowParallelLinear`, `MergedColumnParallelLinear` exist in fork
   - Verify `VllmConfig`, `QuantizationConfig` interfaces match
   - Check that `select_kv_quant` / TurboQuant KV dtype handling works with per-layer head sizes

3. **Adapt `gemma4_mm.py`:**
   - Verify vision/audio encoder imports exist in fork
   - Check multimodal processing pipeline compatibility
   - Ensure `SupportsMultiModal` interface is compatible

4. **Adapt `gemma4_rope.py`:**
   - Verify base RoPE class exists in fork's `rotary_embedding/`
   - Check `RotaryEmbedding` registration mechanism

5. **Wire TurboQuant KV into the dedicated model:**
   - The dedicated model's attention layers construct their own `Attention` instances
   - Ensure `kv_cache_dtype=turboquant35` is passed through correctly
   - Verify TurboQuant metadata lookup works with the dedicated model's layer naming convention
   - The layer names in the dedicated model will likely be `model.layers.{i}.self_attn` (not `{i}.attn` like the Transformers backend)

### Files
- `/tmp/gemma4_port/gemma4.py` → modify in place
- `/tmp/gemma4_port/gemma4_mm.py` → modify in place
- `/tmp/gemma4_port/gemma4_rope.py` → modify in place

### Dependencies
- Phase 1 complete

### Risk
- **Import mismatches:** Mainline may use APIs added after v0.1.dev5. Mitigation: map to fork equivalents or port the specific utility function.
- **Attention backend differences:** Fork's TurboQuant attention backend may need adjustments for variable head sizes. The metadata `head_size` field currently stores a single value — need to confirm if `max(256, 512) = 512` works for the KV cache buffer allocation.

---

## Phase 3: Add Gemma4ModelArchConfigConvertor

### Objective
Add the config convertor that returns `max(head_dim, global_head_dim)` for global buffer allocation.

### Implementation Steps
1. Extract the `Gemma4ModelArchConfigConvertor` class from mainline's `model_arch_config_convertor.py`
2. Add it to the fork's `vllm/transformers_utils/model_arch_config_convertor.py`
3. Key methods to include:
   ```python
   class Gemma4ModelArchConfigConvertor(ModelArchConfigConvertorBase):
       def get_head_size(self) -> int:
           head_dim = getattr(self.hf_text_config, "head_dim", 0)
           global_head_dim = getattr(self.hf_text_config, "global_head_dim", 0)
           return max(head_dim, global_head_dim) or super().get_head_size()

       def get_total_num_kv_heads(self) -> int:
           return getattr(self.hf_text_config, "num_key_value_heads", 1)
   ```
4. Register the convertor in the convertor registry/factory

### Files
- `vllm/transformers_utils/model_arch_config_convertor.py` — add class + registration

### Dependencies
- Phase 1 complete (need to see mainline's exact implementation)

---

## Phase 4: Register Gemma 4 in Model Registry

### Objective
Wire `Gemma4ForConditionalGeneration` and `Gemma4ForCausalLM` to the dedicated model files instead of the Transformers fallback.

### Implementation Steps
1. Edit `vllm/model_executor/models/registry.py`:
   ```python
   # In _TEXT_GENERATION_MODELS:
   "Gemma4ForCausalLM": ("gemma4", "Gemma4ForCausalLM"),

   # In _MULTIMODAL_MODELS:
   "Gemma4ForConditionalGeneration": ("gemma4_mm", "Gemma4ForConditionalGeneration"),
   ```
2. Add `__init__.py` exports if needed
3. Verify that with these entries, vLLM no longer routes Gemma 4 to the Transformers backend

### Files
- `vllm/model_executor/models/registry.py` — add 2 entries
- `vllm/model_executor/models/__init__.py` — add exports if pattern requires it

### Dependencies
- Phases 2, 3 complete (model files and config convertor must exist)

---

## Phase 5: Apply Dependency Fixes

### Objective
Include the buffer loading and BNB loader fixes discovered during #20 investigation.

### Implementation Steps
1. **Buffer loading** — edit `vllm/model_executor/models/utils.py`:
   - In `_add_loadable_non_param_tensors`, add registered buffer handling:
     ```python
     for buf_name, buf_tensor in module.named_buffers(recurse=False):
         if buf_name not in child_params:
             child_params[buf_name] = buf_tensor
     ```

2. **BNB loader None handling** — edit `vllm/model_executor/model_loader/bitsandbytes_loader.py`:
   - In `_hf_weight_iter`, add None check:
     ```python
     mapped_name = self.weight_mapper(org_name)
     if mapped_name is None:
         continue
     ```

3. **Revert Transformers patch** — the `get_per_layer_inputs` patch in Transformers is no longer needed since the dedicated model handles per-layer inputs directly

### Files
- `vllm/model_executor/models/utils.py` — buffer loading
- `vllm/model_executor/model_loader/bitsandbytes_loader.py` — None mapping

### Dependencies
- None (can be done in parallel with Phases 2-4)

---

## Phase 6: Rebuild Fork from Source

### Objective
Build the modified vLLM-TurboQuant fork with CUDA 12.8 and install it.

### Implementation Steps
1. Clone the fork (or use existing clone):
   ```bash
   cd /tmp
   git clone https://github.com/ithllc/vllm-turboquant.git
   cd vllm-turboquant
   ```
2. Copy adapted files into the clone:
   ```bash
   cp /tmp/gemma4_port/gemma4.py vllm/model_executor/models/
   cp /tmp/gemma4_port/gemma4_mm.py vllm/model_executor/models/
   cp /tmp/gemma4_port/gemma4_rope.py vllm/model_executor/layers/rotary_embedding/
   ```
3. Apply registry, config convertor, utils, and BNB loader edits
4. Build:
   ```bash
   CUDA_HOME=/usr/local/cuda-12.8 VLLM_TARGET_DEVICE=cuda MAX_JOBS=2 pip install . --no-build-isolation
   ```
   Note: `MAX_JOBS=2` to stay within 32 GB RAM during compilation.
5. Verify version:
   ```bash
   pip show vllm | grep Version
   python3 -c "import vllm; print(vllm.__version__)"
   ```

### Files
- Fork repo at `/tmp/vllm-turboquant/` — multiple files modified
- No tqCLI code changes

### Dependencies
- Phases 2, 3, 4, 5 all complete

### Risk
- Build takes ~25-30 min. If build fails, check cmake/CUDA compatibility (documented in #15).
- `MAX_JOBS=2` is critical — `MAX_JOBS=4` may OOM on 32 GB RAM.

---

## Phase 7: Generate TurboQuant Metadata

### Objective
Create `turboquant_kv.json` for Gemma 4 E2B with layer names matching the dedicated model's naming convention.

### Implementation Steps
1. Determine layer naming convention in the dedicated model:
   - Likely `model.layers.{i}.self_attn` (standard Transformers convention)
   - Differs from `{i}.attn` used by the generic Transformers backend
2. Generate metadata using `build_default_turboquant_metadata`:
   ```python
   from vllm.v1.attention.ops.turboquant_metadata import (
       build_default_turboquant_metadata, save_turboquant_metadata
   )
   metadata = build_default_turboquant_metadata(
       recipe="turboquant35",
       head_size=512,  # max(256, 512) — global buffer allocation
       num_kv_heads=1,
       layer_names=[f"model.layers.{i}.self_attn" for i in range(35)],
       model_name="gemma-4-E2B-it",
   )
   save_turboquant_metadata(metadata, "/root/.tqcli/models/gemma-4-e2b-it-vllm/turboquant_kv.json")
   ```
3. **Important:** `head_size` in metadata must match the global `get_head_size()` return value (512), because the TurboQuant metadata validation checks `metadata.head_size == layer.head_size`.
   - For sliding layers (head_dim=256), the metadata will have too many high-precision indices — but the default metadata uses sequential indices which are valid for any count
   - For full attention layers (head_dim=512), the metadata matches exactly

### Files
- `/root/.tqcli/models/gemma-4-e2b-it-vllm/turboquant_kv.json` — regenerate

### Dependencies
- Phase 6 complete (need rebuilt fork to determine exact layer naming)

### Risk
- **head_size mismatch:** If the dedicated model uses per-layer head sizes in TurboQuant validation, the single `head_size=512` may fail for sliding layers. Need to test and potentially generate per-layer metadata.
- **Mitigation:** Check the fork's TurboQuant attention backend for the exact validation logic.

---

## Phase 8: Test Gemma 4 E2B Load + Chat

### Objective
Verify Gemma 4 E2B loads and runs on vLLM with the full pipeline.

### Implementation Steps
1. Run the dedicated CPU offload test:
   ```bash
   python3 tests/test_gemma4_vllm_cpu_offload.py
   ```
   Expected: All stages PASS including model load + chat turns

2. If load fails, run the pipeline diagnostic:
   ```bash
   python3 tests/test_gemma4_vllm_pipeline_diagnostic.py
   ```

3. Verify expected config:
   ```
   feasible=True
   cpu_offload_gb=2.1
   quantization=bitsandbytes
   kv_cache_dtype=turboquant35
   max_model_len=2048
   ```

### Files
- `tests/test_gemma4_vllm_cpu_offload.py` (execution)
- `tests/integration_reports/gemma4_vllm_cpu_offload_report.*` (output)

### Dependencies
- Phases 6, 7 complete

---

## Phase 9: Regression Test Qwen3 AWQ

### Objective
Confirm the fork rebuild didn't break Qwen3 AWQ + turboquant35.

### Implementation Steps
1. Run vLLM tests 5-7:
   ```bash
   python3 tests/test_integration_combined.py --engine vllm
   ```
   Expected: All Qwen3 steps PASS

### Files
- `tests/test_integration_combined.py` (execution)

### Dependencies
- Phase 6 complete (rebuilt fork must be installed)

---

## Phase 10: Full Integration Suite + Benchmark

### Objective
Run all tests and capture Gemma 4 E2B vLLM performance numbers.

### Implementation Steps
1. Run full suite:
   ```bash
   python3 tests/test_integration_combined.py
   ```
   Expected: All tests PASS on both engines

2. Run `/tq-benchmark` for Gemma 4 E2B on vLLM:
   - Config: BNB_INT4 + cpu_offload 2.1 GB + turboquant35 KV
   - Metrics: tok/s, time-to-first-token
   - Compare against llama.cpp baseline (7.33 tok/s turbo3)

3. Update reports:
   - `tests/integration_reports/turboquant_kv_comparison_report.md`
   - `tests/integration_reports/gemma4_vllm_cpu_offload_report.md`

4. Push changes to `ithllc/vllm-turboquant`:
   ```bash
   cd /tmp/vllm-turboquant
   git add -A
   git commit -m "Add dedicated Gemma 4 model with per-layer head_dim support"
   git push origin main
   ```

5. Close issue #21 with implementation comment
6. Update issue #20 with final status

### Files
- `tests/integration_reports/*` (update)
- Fork repo (push)

### Dependencies
- Phases 8, 9 PASS

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Import mismatches between mainline and fork | High | Medium | Map to fork equivalents; port missing utilities inline |
| TurboQuant metadata head_size mismatch for sliding layers | Medium | Medium | Test with max(256,512); fall back to per-layer metadata |
| Build failure after adding new files | Low | Medium | Follow #15 build pattern; check cmake compatibility |
| Performance regression from dedicated model overhead | Low | Low | Benchmark will reveal; acceptable if >3 tok/s |
| Qwen3 AWQ regression from utils/BNB patches | Low | High | Regression gate at Phase 9; patches are additive |

## Success Criteria

1. `python3 -c "from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM"` succeeds
2. Gemma 4 E2B loads on vLLM with BNB_INT4 + cpu_offload + turboquant35
3. Multi-turn chat produces coherent output
4. Qwen3 AWQ Tests 5-7 all PASS (no regression)
5. Full integration suite PASS
6. Fork pushed to `ithllc/vllm-turboquant`
7. Issues #20 and #21 closed with implementation comments

---

**Next step:** Execute Phase 1 — fetch mainline Gemma 4 files via `gh api`.
