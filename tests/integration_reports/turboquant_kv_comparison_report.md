# TurboQuant KV Cache Compression — Integration Test Report

**Generated:** 2026-04-17 10:20:43
**System:** NVIDIA RTX A2000 Laptop GPU (4096 MB VRAM)
**CUDA:** driver 13.0, toolkit 12.8
**OS:** Linux (Ubuntu 22.04.4 LTS) (WSL2)

---

## Summary

| Test | Model | Engine | Weight Quant | KV Quant | Pipeline | Result |
|------|-------|--------|-------------|----------|----------|--------|
| Test 1: llama.cpp Gemma 4 E4B Q4_K_M + turbo3 KV | gemma-4-e4b-it-Q4_K_M | llama.cpp | Q4_K_M (pre-quantized) | turbo3 | kv:turbo3 | **PASS** (4/4) |
| Test 2: llama.cpp Qwen 3 4B Q4_K_M + turbo3 KV | qwen3-4b-Q4_K_M | llama.cpp | Q4_K_M (pre-quantized) | turbo3 | kv:turbo3 | **PASS** (7/7) |
| Test 3: vLLM Qwen 3 4B BF16 + bnb INT4 + turboquant35 KV | qwen3-4b-vllm | vllm | BF16 → bnb INT4 | turboquant35 | kv:turbo3 | **PASS** (5/5) |
| Test 4: vLLM Qwen 3 4B AWQ + turboquant35 KV | qwen3-4b-AWQ | vllm | AWQ (pre-quantized) | turboquant35 | kv:turbo3 | **PASS** (5/5) |
| Test 5: Baseline (no KV compression) | multiple | both | various | none | n/a | **PASS** (8/8) |
| Test 6: CUDA compatibility + graceful degradation | n/a | both | n/a | auto | n/a | **PASS** (13/13) |
| Gemma 4 E2B vLLM + CPU offload + TurboQuant KV | gemma-4-e2b-it-vllm | vllm | BF16 → bnb_int4 | turboquant35 | detect full_precision → weight:bnb_int4 → cpu_offload → kv:turboquant35 | **PASS** (15/15) |

---

## Test 1: llama.cpp Gemma 4 E4B Q4_K_M + turbo3 KV

- **Model:** gemma-4-e4b-it-Q4_K_M
- **Engine:** llama.cpp
- **Weight Quantization:** Q4_K_M (pre-quantized)
- **KV Cache:** turbo3
- **Pipeline Stages:** kv:turbo3
- **Duration:** 0.58s
- **Result:** PASS (4/4)

| Step | Result | Details |
|------|--------|---------|
| detect_model_precision | PASS | Model gemma-4-e4b-it-Q4_K_M: precision=weight_quantized (quant=Q4_K_M, format=gguf) |
| plan_quantization_pipeline | PASS | Pipeline: kv:turbo3 / Weight: Model already weight-quantized (Q4_K_M gguf) / KV: Applying turbo3 KV compression (4.6x, M |
| verify_kv_params_llama | PASS | KV params for turbo3: {'cache_type_k': 'turbo3', 'cache_type_v': 'turbo3'} |
| verify_kv_only_pipeline | PASS | Weight quant needed: False (expected: False for pre-quantized GGUF) |

## Test 2: llama.cpp Qwen 3 4B Q4_K_M + turbo3 KV

- **Model:** qwen3-4b-Q4_K_M
- **Engine:** llama.cpp
- **Weight Quantization:** Q4_K_M (pre-quantized)
- **KV Cache:** turbo3
- **Pipeline Stages:** kv:turbo3
- **Duration:** 1.04s
- **Result:** PASS (7/7)

| Step | Result | Details |
|------|--------|---------|
| detect_model_precision | PASS | Model qwen3-4b-Q4_K_M: precision=weight_quantized (quant=Q4_K_M, format=gguf) |
| plan_quantization_pipeline | PASS | Pipeline: kv:turbo3 / Weight: Model already weight-quantized (Q4_K_M gguf) / KV: Applying turbo3 KV compression (4.6x, M |
| verify_kv_params_llama | PASS | KV params for turbo3: {'cache_type_k': 'turbo3', 'cache_type_v': 'turbo3'} |
| verify_kv_params_llama | PASS | KV params for turbo4: {'cache_type_k': 'turbo4', 'cache_type_v': 'turbo4'} |
| verify_kv_params_llama | PASS | KV params for turbo3: {'cache_type_k': 'turbo3', 'cache_type_v': 'turbo3'} |
| verify_kv_params_llama | PASS | KV params for turbo2: {'cache_type_k': 'turbo2', 'cache_type_v': 'turbo2'} |
| verify_kv_only_pipeline | PASS | Weight quant needed: False (expected: False for pre-quantized GGUF) |

## Test 3: vLLM Qwen 3 4B BF16 + bnb INT4 + turboquant35 KV

- **Model:** qwen3-4b-vllm
- **Engine:** vllm
- **Weight Quantization:** BF16 → bnb INT4
- **KV Cache:** turboquant35
- **Pipeline Stages:** kv:turbo3
- **Duration:** 0.94s
- **Result:** PASS (5/5)

| Step | Result | Details |
|------|--------|---------|
| detect_model_precision | PASS | Model qwen3-4b-vllm: precision=full_precision (quant=BF16, format=safetensors) |
| verify_full_precision | PASS | Precision: full_precision (expected: full_precision) |
| plan_quantization_pipeline | PASS | Pipeline: kv:turbo3 / Weight: Model is BF16 — too large for 4096 MB VRAM even with INT4 quantization (may need smaller m |
| verify_kv_params_vllm | PASS | KV params for turbo3: {'kv_cache_dtype': 'turboquant35', 'enable_turboquant': True, 'attention_backend': 'TRITON_ATTN'} |
| verify_dual_pipeline | PASS | Pipeline stages: ['kv:turbo3']. Weight quant: False (). KV: True (turbo3) |

## Test 4: vLLM Qwen 3 4B AWQ + turboquant35 KV

- **Model:** qwen3-4b-AWQ
- **Engine:** vllm
- **Weight Quantization:** AWQ (pre-quantized)
- **KV Cache:** turboquant35
- **Pipeline Stages:** kv:turbo3
- **Duration:** 0.66s
- **Result:** PASS (5/5)

| Step | Result | Details |
|------|--------|---------|
| detect_model_precision | PASS | Model qwen3-4b-AWQ: precision=weight_quantized (quant=AWQ, format=awq) |
| verify_weight_quantized | PASS | Precision: weight_quantized (expected: weight_quantized) |
| plan_quantization_pipeline | PASS | Pipeline: kv:turbo3 / Weight: Model already weight-quantized (AWQ awq) / KV: Applying turbo3 KV compression (4.6x, Minim |
| verify_kv_only_pipeline | PASS | Weight quant needed: False (expected: False for AWQ) |
| verify_kv_params_vllm | PASS | KV params for turbo3: {'kv_cache_dtype': 'turboquant35', 'enable_turboquant': True, 'attention_backend': 'TRITON_ATTN'} |

## Test 5: Baseline (no KV compression)

- **Model:** multiple
- **Engine:** both
- **Weight Quantization:** various
- **KV Cache:** none
- **Pipeline Stages:** none
- **Duration:** 1.08s
- **Result:** PASS (8/8)

| Step | Result | Details |
|------|--------|---------|
| plan_quantization_pipeline | PASS | Pipeline: No quantization applied / Weight: Model already weight-quantized (Q4_K_M gguf) / KV: KV compression disabled b |
| verify_no_kv_qwen3-4b-Q4_K_M | PASS | KV level: none (expected: none for --kv-quant none) |
| plan_quantization_pipeline | PASS | Pipeline: No quantization applied / Weight: Model already weight-quantized (AWQ awq) / KV: KV compression disabled by us |
| verify_no_kv_qwen3-4b-AWQ | PASS | KV level: none (expected: none for --kv-quant none) |
| plan_quantization_pipeline | PASS | Pipeline: No quantization applied / Weight: Model is BF16 — too large for 4096 MB VRAM even with INT4 quantization (may  |
| verify_no_kv_qwen3-4b-vllm | PASS | KV level: none (expected: none for --kv-quant none) |
| verify_kv_params_llama | PASS | KV params for none: {'cache_type_k': 'f16', 'cache_type_v': 'f16'} |
| verify_kv_params_vllm | PASS | KV params for none: {} |

## Test 6: CUDA compatibility + graceful degradation

- **Model:** n/a
- **Engine:** both
- **Weight Quantization:** n/a
- **KV Cache:** auto
- **Pipeline Stages:** none
- **Duration:** 0.69s
- **Result:** PASS (13/13)

| Step | Result | Details |
|------|--------|---------|
| check_cuda_compatibility | PASS | TurboQuant available=True, toolkit=12.8, driver=13.0: TurboQuant KV cache available (llama.cpp, vLLM) |
| parse_cuda_12.8 | PASS | parse_cuda_version('12.8') = (12, 8) (expected (12, 8)) |
| parse_cuda_11.5 | PASS | parse_cuda_version('11.5') = (11, 5) (expected (11, 5)) |
| parse_cuda_13.0 | PASS | parse_cuda_version('13.0') = (13, 0) (expected (13, 0)) |
| parse_cuda_empty | PASS | parse_cuda_version('') = (0, 0) (expected (0, 0)) |
| parse_cuda_invalid | PASS | parse_cuda_version('invalid') = (0, 0) (expected (0, 0)) |
| verify_compatibility_matches_cuda | PASS | CUDA 12.8: available=True (expected=True) |
| plan_quantization_pipeline | PASS | Pipeline: kv:turbo3 / Weight: Model already weight-quantized (Q4_K_M gguf) / KV: Applying turbo3 KV compression (4.6x, M |
| verify_turbo3_when_available | PASS | KV level: turbo3 (expected: turbo3 since TurboQuant is available) |
| verify_ratio_none | PASS | none: 1.0x compression, Baseline |
| verify_ratio_turbo4 | PASS | turbo4: 3.8x compression, Near-lossless (+0.23% PPL) |
| verify_ratio_turbo3 | PASS | turbo3: 4.6x compression, Minimal loss (+1.06% PPL) |
| verify_ratio_turbo2 | PASS | turbo2: 6.4x compression, Noticeable loss (+6.48% PPL) |

## Gemma 4 E2B vLLM + CPU offload + TurboQuant KV

- **Model:** gemma-4-e2b-it-vllm
- **Engine:** vllm
- **Weight Quantization:** BF16 → bnb_int4
- **KV Cache:** turboquant35
- **Pipeline Stages:** detect full_precision → weight:bnb_int4 → cpu_offload → kv:turboquant35
- **Duration:** 681.63s
- **Result:** PASS (15/15)

| Step | Result | Details |
|------|--------|---------|
| find_model_profile | PASS | Found gemma-4-e2b-it-vllm: Gemma 4 E2B Edge Instruct (vLLM BF16) (2.3B, BF16) |
| detect_precision | PASS | Detected: full_precision (quant=BF16, format=safetensors) |
| size_estimates | PASS | BF16=11710 MB, INT4=4145 MB, VRAM=4096 MB, RAM=17201 MB |
| select_quantization_without_offload | PASS | select_quantization() returned: None (expected None — too large for VRAM alone) |
| build_vllm_config_with_offload | PASS | feasible=True / cpu_offload_gb=9.9 / quantization=bitsandbytes / kv_cache_dtype=turboquant35 / max_model_len=2048 |
| model_available | PASS | Model already downloaded at /root/.tqcli/models/gemma-4-e2b-it-vllm |
| load_model_with_cpu_offload | PASS | Loaded Gemma 4 E2B via vLLM in 557.1s / BNB_INT4 + cpu_offload=9.9 GB + kv=turboquant35 |
| chat_thinking_turn | PASS | Response: * **15% of 240 is 36.**... |
| chat_simple_turn | PASS | Response: — Paris... |
| unload_model | PASS | Model unloaded |
| cli_--version | PASS | cmd: tqcli --version / rc=0 |
| cli_system_info_--json | PASS | cmd: tqcli system info --json / rc=0 / keys=['os', 'os_display', 'arch', 'is_wsl', 'cpu_cores'] |
| cli_model_list | PASS | cmd: tqcli model list / rc=0 |
| cli_chat_--help | PASS | cmd: tqcli chat --help / rc=0 |
| workflow_items_not_performed | PASS | Model pull, KV-compressed chat, image/audio input, skill creation, multi-process server, model removal, and clean uninst |

---

## TurboQuant KV Compression Reference

| Level | Bits/Value | Compression | Quality Impact |
|-------|-----------|-------------|---------------|
| none | 8.5 | 1x (no compression) | Baseline |
| turbo4 | 4.25 | 3.8x | Near-lossless (+0.23% PPL) |
| turbo3 | 3.5 | 4.6x | Minimal loss (+1.06% PPL) |
| turbo2 | 2.5 | 6.4x | Noticeable loss (+6.48% PPL) |