# Gemma 4 E2B vLLM + CPU Offloading + BNB_INT4 + TurboQuant KV

**Generated:** 2026-04-16T19:20:45
**Test:** Gemma 4 E2B vLLM + CPU Offloading + BNB_INT4 + TurboQuant KV
**Result:** PASS (10/10 steps)
**Duration:** 428.96s

## System
| Property | Value |
|----------|-------|
| os | Linux (Ubuntu 22.04.4 LTS) (WSL2) |
| gpu | NVIDIA RTX A2000 Laptop GPU |
| vram_mb | 4096 |
| ram_total_mb | 31956 |
| ram_available_mb | 24739 |
| cuda_toolkit | 12.8 |
| compute_capability | 8.6 |
| is_wsl | True |

## Pipeline Configuration
**Path:** `detect full_precision → weight:bnb_int4 → cpu_offload → kv:turboquant35`

| Setting | Value |
|---------|-------|
| weight_quantization | bnb_int4 |
| cpu_offload_gb | 9.9 |
| kv_compression | turboquant35 |
| max_model_len | 2048 |

## Step Results

| # | Step | Result | Duration | Details |
|---|------|--------|----------|---------|
| 1 | find_model_profile | PASS | - | Found gemma-4-e2b-it-vllm: Gemma 4 E2B Edge Instruct (vLLM BF16) (2.3B, BF16) |
| 2 | detect_precision | PASS | - | Detected: full_precision (quant=BF16, format=safetensors) |
| 3 | size_estimates | PASS | - | BF16=11710 MB, INT4=4145 MB, VRAM=4096 MB, RAM=24417 MB |
| 4 | select_quantization_without_offload | PASS | - | select_quantization() returned: None (expected None — too large for VRAM alone) |
| 5 | build_vllm_config_with_offload | PASS | 0.00s | feasible=True \| cpu_offload_gb=9.9 \| quantization=bitsandbytes \| kv_cache_dtype=turboquant35 \| max_m |
| 6 | model_available | PASS | - | Model already downloaded at /root/.tqcli/models/gemma-4-e2b-it-vllm |
| 7 | load_model_with_cpu_offload | PASS | 375.32s | Loaded Gemma 4 E2B via vLLM in 375.3s \| BNB_INT4 + cpu_offload=9.9 GB + kv=turboquant35 |
| 8 | chat_thinking_turn | PASS | 45.26s | Response: * **15% of 240 is 36.**... |
| 9 | chat_simple_turn | PASS | 8.38s | Response: — Paris... |
| 10 | unload_model | PASS | - | Model unloaded |
