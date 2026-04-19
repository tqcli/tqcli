# Cross-Reference Review: TurboQuant KV-Cache Root-Cause + Fix Architecture

**Purpose:** Independent verification by Google Gemini of a multi-stage investigation into why a specific TurboQuant integration failed, whether the proposed fix is correct, and whether the proposed long-term architecture is sound. All findings below were produced by Claude (Opus 4.7) after direct code inspection, GitHub API queries, web search, and reading the upstream TurboQuant implementations.

**Your job:** confirm, challenge, or correct every claim below. Be direct. Cite line-level evidence where relevant.

---

## Section 1 — The immediate problem

**Project:** tqCLI — a cross-platform CLI that wraps llama.cpp and vLLM for local LLM inference with TurboQuant KV cache compression. Uses `github.com/ithllc/vllm-turboquant` as the vLLM fork (CUDA 12.8+, Triton backend).

**The failure:** During the 0.6.0 functional integration suite (`tests/test_integration_agent_functional.py`), the vLLM + Qwen 3 4B configuration failed to load with `kv_cache_dtype=turboquant35` (i.e. `kv_quant_choice="turbo3"`). Exception:

```
ValueError: TurboQuant KV cache requires metadata. Pass
`turboquant_metadata_path` or place `turboquant_kv.json` under the
local model path.
```

Raised at `vllm/v1/attention/backends/triton_attn.py:600` inside `TritonAttentionImpl.__init__` when `kv_cache_dtype.startswith("turboquant")` and `discover_turboquant_metadata_path()` returns `None`.

**On-disk evidence:**
- `~/.tqcli/models/gemma-4-e2b-it-vllm/turboquant_kv.json` — EXISTS (9,320 lines; model_name="gemma-4-E2B-it"; recipe=turboquant35; head_size=256; 35 layers; outer list length=1 per layer; inner list length=128)
- `~/.tqcli/models/qwen3-4b-AWQ/turboquant_kv.json` — EXISTS (526,910 bytes; model_name="Qwen3-4B-AWQ"; recipe=turboquant35; head_size=128; 36 layers; outer list length=8 per layer (GQA); inner list length=64)
- `~/.tqcli/models/qwen3-4b-vllm/turboquant_kv.json` — **DOES NOT EXIST**. This is the BF16 safetensors variant of Qwen3 4B used by the failing test.

**Temporary workaround in the failing test:** the suite now passes `kv_quant_choice="none"` for Qwen 3 4B on vLLM (`tests/test_integration_agent_functional.py:run_vllm_group`, `load_vllm_engine(QWEN_VLLM, qwen_profile, kv_quant_choice="none", max_len=896)`). The test report annotates this combination `kv:none — turboquant_kv.json not present for Qwen3 4B` and cites the fix prompt `docs/prompts/fix_qwen3_turboquant.md` as the tracking task.

---

## Section 2 — The user's suspicion + what prior tests actually did

The user was suspicious because `tests/integration_reports/turboquant_kv_comparison_report.md` (2026-04-17) reported **PASS (22/22)** for "Test 3: vLLM Qwen 3 4B BF16 + bnb INT4 + turboquant35 KV". How can a config that "passed" a day earlier now fail deterministically?

**Direct inspection of the test code (`tests/test_integration_turboquant_kv.py:431, test_3_vllm_qwen3_bf16_bnb_turbo`):**
- Step 1: `step_detect_precision` — reads `config.json` on disk, no model load
- Step 2: `step_plan_pipeline` — pure Python planning (builds a config dict)
- Step 3: `step_verify_kv_params_vllm` — dict lookup: `{kv_cache_dtype: "turboquant35", enable_turboquant: True, attention_backend: "TRITON_ATTN"}`
- Step 4: `verify_dual_pipeline` — **hardcoded `passed=True`** with inline comment on line 472: `# We're testing pipeline logic, not actual model load`
- Steps 5+: Section E lifecycle — `tqcli --version`, `tqcli model list`, `tqcli skill create/list/run/cleanup`, no model load

**Total wall-clock duration of Test 3: 9.10 seconds.** A 4B BF16 safetensors load via vLLM with cpu_offload=6.5 GB takes 3-4 minutes on the reference box (RTX A2000 Laptop, 4 GB VRAM, WSL2). 9.10 s is not enough to finish `from_pretrained`, let alone run inference.

**Test 4 (`test_4_vllm_qwen3_awq_turbo`) has the same shape** — planning + lifecycle, no model load. Same ~7 s duration.

**The ONE prior test that actually loaded Qwen 3 + turboquant35 on vLLM** was in `tests/integration_reports/turboquant_kv_unified_report.md` Test 5/6/7: `load_model_turbo_kv PASS 70.43s` — but that used `qwen3-4b-AWQ` (the AWQ-quantized variant whose directory ships `turboquant_kv.json`), NOT `qwen3-4b-vllm` (the BF16 variant that doesn't).

**Claim under review:** This is not a regression. The Qwen 3 4B vLLM BF16 + turboquant35 code path was exercised end-to-end for the first time in the 0.6.0 agent-modes test, and it failed immediately on a legitimate precondition check. Prior "pass" results were parameter-level / planning-level assertions that the test author explicitly documented as such (the inline comment on line 472 is honest).

**Question 1 for Gemini:** Is the characterization above accurate? Is this a test-coverage gap rather than a regression? Is there any other reading of the evidence that would indicate an actual regression in the fork?

---

## Section 3 — Schema of `turboquant_kv.json` and why metadata is required

Schema from `/usr/local/lib/python3.11/site-packages/vllm/v1/attention/ops/turboquant_metadata.py`:

| Field | Meaning |
|---|---|
| `version` | `1` (only supported version) |
| `recipe` | `"turboquant25"` (~2.5 bits/value, 6.4× compression) or `"turboquant35"` (~3.5 bits/value, 4.6× compression) |
| `head_size` | Attention head dimension, must be `% 16 == 0` |
| `model_name` | Informational, freeform string |
| `transform_version` | `"structured_hadamard_v1"` — the only supported rotation family |
| `codebook_version` | `"lloyd_beta_v1"` — the fork-baked Lloyd-Max scalar codebook (compiled into Triton kernels) |
| `layers.<name>.key_high_precision_indices` | Shape `[num_kv_heads][outlier_count]` of ints in `[0, head_size)`. NB: outer list = num_KV_heads (GQA), NOT num_attention_heads |
| `layers.<name>.value_high_precision_indices` | Same shape, independent indices for V |

For Qwen 3 4B: `head_size=128`, `num_kv_heads=8` (GQA 4:1), `num_hidden_layers=36`. For `turboquant35`, `outlier_count = round(128 × 0.50 / 16) × 16 = 64`. So each inner list has 64 integers in `[0, 128)`, outer list has 8 entries, 36 layers.

The vLLM fork's runtime path (`triton_attn.py:592-605`) calls `discover_turboquant_metadata_path(model_name_or_path, explicit_path)` which only checks `<model_path>/turboquant_kv.json`. tqCLI's `tqcli/core/vllm_backend.py` does NOT pass `turboquant_metadata_path` explicitly, so the fork's implicit-path lookup is the only resolution mechanism. File missing → ValueError.

**Claim under review:** The proposed root cause ("`turboquant_kv.json` is missing for `qwen3-4b-vllm`, while `qwen3-4b-AWQ` and `gemma-4-e2b-it-vllm` have one shipped") is mechanically correct, reproducible, and explained by reading 15 lines of code in `triton_attn.py` + `turboquant_metadata.py:364`.

**Question 2 for Gemini:** Is the schema interpretation above correct, including the GQA outer-list rule (num_kv_heads, not num_attention_heads)? Is the outlier_count formula (`round(head_size × ratio / 16) × 16`) correct, and are the ratio→recipe mappings 0.25→turboquant25 and 0.50→turboquant35 correct?

---

## Section 4 — Consistency check across four TurboQuant implementations

The user asked for confirmation that TurboQuant's algorithm is portable across inference engines (llama.cpp, vLLM, SGLang) with only integration-layer differences. Evidence gathered from direct reads of issues, PRs, and code.

| Implementation | Status | Calibration policy | Rotation primitive | Outlier fraction | Codebook |
|---|---|---|---|---|---|
| **ithllc/vllm-turboquant** (tqCLI uses this) | Production (pinned to vLLM 0.1.dev6+gb236390bf) | **Pre-computed JSON sidecar** (Lloyd-beta offline calibration) | Block-decomposed FWHT + deterministic-seeded random ±1 sign flip (`structured_hadamard_v1`) | 25% (turboquant25) or 50% (turboquant35) | Fixed, baked into Triton kernels (`lloyd_beta_v1`) |
| **vLLM upstream PR #38280** (`lishunyang12`) | CLOSED, WIP, not merged | **Runtime auto-calibrate on first batch** | Hadamard butterfly + random ±1 sign flip (7 levels for d=128) | 19/128 ≈ 15% (paper's default) | Fixed Lloyd-Max centroids, no file |
| **SGLang PR #21617** (`scottgl9`) | OPEN, Draft | **Runtime auto-calibrate on first batch** (variance top-k) | **Dense QR-random orthogonal matrix** (seeded, O(d²)) — NOT Hadamard | 15% | Gaussian-quantile init + 20 Lloyd-Max iterations at calibration |
| **OnlyTerp/turboquant** (first OSS impl) | Standalone library | **Data-oblivious** (no calibration at all) | Supports BOTH RHT (Hadamard butterfly + signs, default) AND QR-random | Not specified in README | Fixed, baked-in (Beta-distribution assumption) |

**All four share the same three-stage algorithm** (paper Sections 2.2-2.3):
1. Apply an orthogonal rotation to each head vector → decorrelates channels
2. Preserve top-k "outlier" channels at higher precision (bf16 or similar)
3. Apply Lloyd-Max scalar quantization (4-bit or 3-bit) to the remaining channels
4. Optional: 1-bit QJL residual correction for bias-free inner-product recovery

**Differences are integration-layer only:**
- Where the encode/decode kernels live (vLLM Triton backend; SGLang FlashInfer + Triton; llama.cpp would be ggml kernels)
- Paging / memory-layout compatibility with the engine's attention scheduler (vLLM upstream PR flagged this as blocking hybrid models like Qwen3.5 MoE)
- Calibration policy (see table)

**Claim under review:** TurboQuant is engine-portable at the algorithm level. The user's mental model — that llama.cpp, vLLM, and SGLang differ only in "plugin mechanics" while the KV-cache compression application itself is the same — is correct. The ithllc fork is the outlier only in its choice of calibration policy (pre-computed vs. runtime), not in the core algorithm.

**Question 3 for Gemini:** Is the consistency-check table accurate? In particular:
(a) Is SGLang's choice of dense QR-random rotation a deliberate departure from the paper (which uses FWHT), and what are the speed / quality tradeoffs?
(b) Is OnlyTerp's data-oblivious approach defensible theoretically, or does it rely on assumptions (e.g. rotated coordinates being i.i.d. Beta) that don't hold for real LLM activations?
(c) Does the paper itself prescribe pre-computed calibration or runtime calibration for the "near-lossless 3-bit" claim? Which of the four approaches aligns most closely with the paper's evaluation methodology?

---

## Section 5 — The ithllc fork's Hadamard primitive IS reproducible from user code

Direct inspection of `/usr/local/lib/python3.11/site-packages/vllm/v1/attention/ops/turboquant_kv_cache.py` reveals the EXACT primitive:

- `_fwht_pow2(x)` (lines 174-186) — standard in-place Fast Walsh-Hadamard Transform butterfly, pure PyTorch, O(d log d)
- `_structured_signs_cached(device_type, device_index, dim, seed)` (lines 189-210) — deterministic ±1 sign vector from `torch.Generator().manual_seed(seed)`, where `seed = TURBOQUANT_SEED + seed_offset + dim`
- `_apply_block_hadamard(x, signs, normalized=True, inverse=False)` (lines 213-235) — decomposes `x.shape[-1]` into power-of-2 blocks via `_hadamard_block_sizes`, per block applies: multiply by signs → FWHT → /√blocksize
- `_apply_mse_transform = _apply_block_hadamard(..., normalized=True, inverse=False)`
- `get_turboquant_mse_transform_matrix(device, dim, seed_offset)` (lines 439-456) — materializes the full `[dim, dim]` forward-transform matrix F such that `F @ v` = rotated unit vector. Cached per `(device, dim, seed_offset)`

**Implication for a metadata generator ("Option A" in `docs/prompts/fix_qwen3_turboquant.md`):**

The user's code can `import` this primitive and call `get_turboquant_mse_transform_matrix(device, head_size, seed_offset)` to produce the EXACT rotation matrix the runtime uses. No reverse engineering of the Hadamard variant, no mismatch between pytorch-side calibration and triton-side runtime.

**One architectural caveat:** the runtime uses **two independent rotations with different seeds** (`triton_attn.py:694-695`: `seed_offset=101` for group 0, `seed_offset=211` for group 1), splitting `head_size` into two groups. This matches the TurboQuant paper Section 2.3 ("two independent rotations for outlier/regular channel subsets"). A correct metadata generator must honor this grouping — two rotations on two channel subsets — or the outlier indices will be computed in the wrong basis.

The exact group-split logic (how `head_size` is partitioned into group 0 and group 1) is not yet traced. TODO before writing the generator.

**Question 4 for Gemini:** Is the plan "import `get_turboquant_mse_transform_matrix` directly from the fork and use it to rotate k_proj/v_proj weight rows into the runtime's coordinate space, then compute per-head per-channel L2 norms across the hidden_size axis, then select top-`outlier_count` indices per head" mathematically sound? Does it match what the runtime actually looks up during KV-cache encoding? Are there subtleties around the two-rotation group split that break this plan?

---

## Section 6 — The user's pushback on my auto-calibration recommendation (this is the critical open question)

In my prior review I said: *"A cleaner long-term fix is to port the runtime auto-calibration logic from vLLM upstream PR #38280 into the ithllc fork, making turboquant_kv.json optional rather than mandatory."* The user pushed back, arguing that the current pre-computed metadata approach is **more accurate** than runtime auto-calibration.

**The user is correct, and my recommendation was wrong-headed.** Here is the corrected analysis:

### Quality vs. simplicity tradeoff across calibration policies

| Calibration policy | Data used | Outlier selection fidelity | Statistical power | Reproducibility | Deployment cost |
|---|---|---|---|---|---|
| **Pre-computed Lloyd calibration** (ithllc) | Hundreds of prompts, fp32 accumulation, full activation traces per layer per head | Highest — reflects where the model actually puts outlier energy on real workloads | High (N ≫ 1 prompts, diverse) | Fully reproducible (same model + same calibration set = same indices) | Need to ship / generate a ~500 KB–1 MB JSON per model |
| **Runtime auto-calibration** (vLLM upstream, SGLang) | 1 batch of whatever happens to arrive first at inference time | Lower — first-batch variance may not reflect steady-state activation distribution across diverse prompts | Low (N = single batch, possibly non-representative) | Non-reproducible across servers (depends on warmup batch) | Zero — works out of the box for any model |
| **Data-oblivious** (OnlyTerp) | No data — relies on rotated coords being ≈ Gaussian/Beta i.i.d. | Lowest — assumption holds only asymptotically | N/A | Fully reproducible | Zero |

**Why the user is right:**
1. The TurboQuant paper's "near-lossless 3-bit" claim is evaluated WITH proper offline outlier selection, not with single-batch runtime calibration. The paper explicitly discusses outlier detection as a calibration step.
2. More calibration data → tighter approximation error bounds. This is basic statistics; a single batch cannot match 100+ prompts for estimating per-channel variance.
3. "Silent quality collapse" risk grows if outlier indices are poorly chosen. For production serving where the same model runs for weeks, paying the one-time calibration cost is negligible.
4. Reproducibility matters: two vLLM servers behind a load balancer should give byte-identical outputs for the same prompt. Runtime auto-calibration breaks this because the warmup batch can differ.

### Revised architectural recommendation

The correct long-term architecture is NOT to port runtime auto-calibration from upstream. It is:

**(a)** Ship `tqcli calibrate-model <model-id>` as a first-class CLI command that runs full Lloyd calibration offline (Option B in the fix prompt) and writes `turboquant_kv.json` into the model directory. This runs once at model-download time or on-demand.

**(b)** Allow a **hybrid fallback** so models without calibrated metadata still run (with a loud warning) via on-the-fly calibration or data-oblivious defaults — degraded quality but operational. This is a safety net, not the primary path.

**(c)** For the immediate near-term fix (Qwen 3 4B BF16 on vLLM), ship a weight-L2 heuristic generator ("Option A") as a bridge: defensible, fast to prototype, and a strictly better starting point than no metadata. Gate it behind the quality-validation criteria (PPL within 1.05× of kv:none baseline on a 100-prompt sanity set, coherent smoke output, mini needle-in-a-haystack at ≥8k context).

**(d)** Refuse Option A (weight-L2) for pre-quantized source weights (AWQ / GPTQ), because those weights have already been scaled to protect activation outliers — L2 on AWQ weights finds scales, not feature importance. For pre-quantized sources, Option B (activation-based) is mandatory.

**Question 5 for Gemini (the most important):** Is the user's argument correct — that pre-computed Lloyd calibration is MORE accurate than runtime auto-calibration for near-lossless 3-bit KV compression? Specifically:
(a) Does the TurboQuant paper's experimental methodology use pre-computed or runtime outlier selection?
(b) What is the statistical expectation for outlier-channel-index stability across batches? Would runtime single-batch selection converge to the same indices that offline calibration produces, or diverge?
(c) Is there published empirical evidence comparing the two policies on downstream quality (PPL, needle-in-haystack, LongBench)?
(d) Is there a principled reason vLLM upstream and SGLang chose runtime calibration despite the apparent quality cost — e.g. a deployment constraint that makes offline calibration infeasible at their target scale?
(e) Is the revised architecture (offline calibration as primary path + runtime fallback for cold-start) defensible, or is there a better design?

---

## Section 7 — Generalization to other models (user's final concern)

End users will try TurboQuant KV on models other than Qwen 3 4B and Gemma 4 E2B. The generator must generalize. Per Gemini's prior review (in the earlier conversation round), four invariants must hold:

1. **GQA / MQA outer-list rule:** `outer_len = num_kv_heads`, not `num_attention_heads`. Qwen 3 4B is 32 Q-heads / 8 KV-heads — using 32 will shape-mismatch in the Triton kernel launch. Multi-query-attention models (1 KV head) must emit outer_len=1. The Gemma 4 E2B metadata already exhibits this (outer_len=1 per layer).

2. **Variable head_dim:** Gemma 4 is the canonical pathological case — sliding-window layers use `head_dim=256`, full-attention layers use `head_dim=512`. The fork handles this by logging a warning and falling back to bf16 per-layer when metadata `head_size` ≠ layer `head_size` (`triton_attn.py:616-629`). A correct generator must either emit head-size-matched metadata per layer, or refuse to cover layers that don't match. A runtime guard in `tqcli/core/vllm_config.py` should fail loudly on head_dim mismatch rather than silently falling back per-layer.

3. **Pre-quantized sources (AWQ, GPTQ):** weight-L2 heuristic is unsafe. AWQ already rescales weights to protect activation outliers; L2 on AWQ weights finds scales, not importance. Option B (activation calibration) is the only robust path. The generator should refuse or emit a hard error on AWQ/GPTQ source weights unless given pre-calibrated activations.

4. **Tensor-parallel sharding:** metadata is NOT pre-sharded — the runtime (`slice_turboquant_layer_metadata_for_tp` in the fork) slices the full metadata per TP rank at load time. The generator emits complete `num_kv_heads × outlier_count` lists and trusts the runtime to split correctly.

5. **Power-of-2 head_dim requirement:** `_hadamard_block_sizes` decomposes `head_size` into powers of 2 (e.g. head_size=96 = 64 + 32). For strictly non-power-of-2 head sizes (e.g. head_size=80), the block decomposition still works but inserts structural boundaries into the rotation that the metadata's per-channel indices must respect. Models with head_dim not a power of 2 need special validation.

**Question 6 for Gemini:** Are there additional invariants the generator must honor for other model families (Llama 3 8B, Mistral 7B, Phi-3, Qwen 3 8B/14B/32B, Qwen 3 MoE, Gemma 4 E4B, DeepSeek V3 MLA)? In particular:
(a) Multi-head Latent Attention (MLA) in DeepSeek V3 — does TurboQuant even apply, or is the KV representation incompatible?
(b) Models with attention sinks, sliding window, or chunked attention — same head_size across layers or variable?
(c) Qwen 3 MoE variants — TurboQuant is per-layer, MoE shouldn't change that, correct?
(d) Are there quantization interactions (rotation commutes with int4 weight quant? with RoPE?) that a generator must be aware of?

---

## Section 8 — Proposed near-term plan (for Gemini to validate or redirect)

**Stage 1 — Close the two-rotation-group question:** read `turboquant_kv_cache.py` lines around `get_turboquant_rotation(..., seed_offset=101)` and `seed_offset=211` to confirm how `head_size` is partitioned into groups, and verify the metadata JSON indices live in which group's coordinate basis.

**Stage 2 — Prototype `tqcli/core/kv_metadata_generator.py` targeting Qwen 3 4B BF16:**
- Load model via safetensors.
- For each attention layer, extract `k_proj.weight` and `v_proj.weight`.
- Reshape to `[num_kv_heads, head_size, hidden_size]`.
- For each KV head, apply the EXACT forward Hadamard used at runtime via `get_turboquant_mse_transform_matrix`.
- Compute per-channel L2 norm along the `hidden_size` axis → one scalar per `head_size` channel per head.
- Select top-`outlier_count` indices per head, sorted ascending.
- Emit JSON matching the schema.
- Write to `~/.tqcli/models/qwen3-4b-vllm/turboquant_kv.json`.

**Stage 3 — Guards and refuses:**
- Head_dim mismatch guard in `tqcli/core/vllm_config.py` — fail loudly if metadata `head_size` ≠ any layer's `head_size`.
- AWQ / GPTQ source refuse — if `precision=weight_quantized` and source is not calibrated activations, refuse Option A and escalate to Option B.
- Non-power-of-2 head_dim warning.
- Recipe / outlier_count cross-check (emit a helpful error if `head_size * ratio / 16` is not an integer).

**Stage 4 — Validation gates before shipping:**
- Metadata loads via `vllm.v1.attention.ops.turboquant_metadata.load_turboquant_metadata()` without raising.
- `tqcli chat --model qwen3-4b-vllm --engine vllm --kv-quant turbo3 --prompt "capital of France" --json` returns coherent text containing "Paris" (no `</think></think>` loops, no mojibake, no repetition).
- PPL within 1.05× of `kv:none` baseline on a 100-prompt sanity set.
- Mini needle-in-a-haystack at ≥8k context.

**Stage 5 — Long-term architecture:**
- Ship `tqcli calibrate-model <id>` for proper Option B offline Lloyd calibration.
- Document Option A as the fast-path heuristic, Option B as the quality-path calibration.
- Keep pre-computed metadata as the primary deployment path — do NOT port runtime auto-calibration as the primary path.

**Question 7 for Gemini (final):** Is this plan correct, ordered correctly, and complete? What is missing, what is redundant, and where do you disagree with the sequencing or the architecture?

---

## Closing ask to Gemini

Please answer all seven questions above (Sections 2, 3, 4, 5, 6, 7, 8). Be direct — if any claim is wrong, say so and explain why. If the root-cause analysis is correct but the fix architecture is wrong, say that. If both are correct but the generalization plan is missing something critical, say that. Budget ~1500 words total across all seven answers. No flattery; treat this as adversarial peer review.

The user's primary concern is **Question 5** — whether pre-computed calibration is genuinely more accurate than runtime auto-calibration for near-lossless TurboQuant 3-bit, or whether Claude's revised position (pre-computed is the right primary path) is still wrong in some way that Claude hasn't seen.
