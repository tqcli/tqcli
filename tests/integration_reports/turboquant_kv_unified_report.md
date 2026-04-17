# tqCLI Unified Integration Test Report — TurboQuant KV

**Generated:** 2026-04-16 12:41:05
**tqCLI Version:** 0.5.0
**Scope:** Tests 5-7 (Thinking + Tool Calling + Combined) with TurboQuant KV
**Engines:** llama.cpp (0.3.20), vLLM (0.1.dev6+gb236390bf)

## System Information

| Property | Value |
|----------|-------|
| os | Linux (Ubuntu 22.04.4 LTS) (WSL2) |
| arch | x86_64 |
| cpu_cores | 16 |
| cpu_physical | 8 |
| ram_total_mb | 31956 |
| ram_available_mb | 24591 |
| gpu | NVIDIA RTX A2000 Laptop GPU |
| vram_mb | 4096 |
| recommended_engine | llama.cpp |
| recommended_quant | Q3_K_M |
| max_model_gb | 3.4 |
| is_wsl | True |

## Quantization Pipeline Validation

The unified quantization pipeline detects model precision and applies the appropriate stages:

| Model Type | Weight Quantization | KV Cache Compression | Pipeline Path |
|------------|--------------------|--------------------|---------------|
| GGUF Q4_K_M (llama.cpp) | SKIP (pre-quantized) | turbo3 (4.6x) | KV-only |
| AWQ INT4 (vLLM) | SKIP (pre-quantized) | turboquant35 | KV-only |
| BF16 safetensors (vLLM) | BNB_INT4 (on-the-fly) | turboquant35 | Full pipeline |

## Overall Summary

| Metric | Value |
|--------|-------|
| Total Tests | 3 |
| Total Steps | 33 |
| Passed | 30 |
| Failed | 3 |
| Pass Rate | 90.9% |

---

## vLLM (TurboQuant fork)
**Duration:** 4891.3s

### Test 5: Thinking Mode + turboquant35 KV (vLLM)

**Model:** `qwen3-4b-AWQ` | **Engine:** vLLM (TurboQuant fork) | **Result:** **FAIL** (12/13 steps) | **Duration:** 3430.5s

| # | Step | Result | Duration | Details |
|---|------|--------|----------|---------|
| 1 | turboquant_compatibility | PASS | 0.00s | TurboQuant KV cache available (llama.cpp, vLLM) |
| 2 | download_model | PASS | 0.01s | Already downloaded at /root/.tqcli/models/qwen3-4b-AWQ (2558 MB) |
| 3 | quantization_pipeline | PASS | 0.00s | Pipeline: kv:turbo3 \| Precision: weight_quantized \| Weight quant needed: False \| KV compression: tur |
| 4 | verify_kv_only_pipeline | PASS | 0.00s | Weight quant needed: False (expected: False for pre-quantized) \| KV compression: True (expected: Tru |
| 5 | load_model_turbo_kv | PASS | 70.43s | Loaded Qwen3 4B (AWQ INT4, vLLM) via vLLM with kv_cache_dtype=turboquant35 in 70.4s |
| 6 | qwen3_thinking_turn | PASS | 72.72s | Thinking: NO (0 chars) \| Response: To calculate 15% of 240, I can follow the:  15. First, convert th |
| 7 | qwen3_no_think_turn | PASS | 81.29s | Thinking: 0 chars (expected minimal) \| Response: </think>  </think>  To find 10% of 5 5 500, I can m |
| 8 | qwen3_reasoning_turn | PASS | 178.71s | Thinking: NO (0 chars) \| Response: The ball costs 5 cents.   **Explanation  The problem is a classic |
| 9 | download_model | PASS | 0.00s | Already downloaded at /root/.tqcli/models/gemma-4-e2b-it-vllm (9803 MB) |
| 10 | quantization_pipeline | PASS | 0.00s | Pipeline: kv:turbo3 \| Precision: full_precision \| Weight quant needed: False \| KV compression: turbo |
| 11 | verify_full_pipeline | PASS | 0.00s | Weight quant needed: False \| Weight method:  \| KV compression: True \| KV level: turbo3 \| Precision:  |
| 12 | load_model_turbo_kv | FAIL | 3027.36s | Failed to load with TurboQuant KV: Engine core initialization failed. See root cause above. Failed c |
| 13 | gemma4_hw_limitation | PASS | 0.00s | Gemma 4 E2B BF16 + BNB_INT4 may not fit 4096 MB VRAM. Expected on < 6 GB hardware. |

#### Pipeline Decision
- **Precision:** weight_quantized | **Weight quant:** False () | **KV level:** turbo3 | **KV-only:** True
- **Precision:** full_precision | **Weight quant:** False () | **KV level:** turbo3 | **KV-only:** True

#### Performance
| Step | tok/s | Tokens | Thinking |
|------|-------|--------|----------|
| qwen3_thinking_turn | 1.54 | 112 | NO |
| qwen3_no_think_turn | 0.65 | 53 | NO |
| qwen3_reasoning_turn | 0.62 | 111 | NO |

### Test 6: Tool Calling + turboquant35 KV (vLLM)

**Model:** `qwen3-4b-AWQ` | **Engine:** vLLM (TurboQuant fork) | **Result:** **FAIL** (10/11 steps) | **Duration:** 806.7s

| # | Step | Result | Duration | Details |
|---|------|--------|----------|---------|
| 1 | download_model | PASS | 0.00s | Already downloaded at /root/.tqcli/models/qwen3-4b-AWQ (2558 MB) |
| 2 | quantization_pipeline | PASS | 0.01s | Pipeline: kv:turbo3 \| Precision: weight_quantized \| Weight quant needed: False \| KV compression: tur |
| 3 | verify_kv_only_pipeline | PASS | 0.00s | Weight quant needed: False (expected: False for pre-quantized) \| KV compression: True (expected: Tru |
| 4 | load_model_turbo_kv | PASS | 151.62s | Loaded Qwen3 4B (AWQ INT4, vLLM) via vLLM with kv_cache_dtype=turboquant35 in 151.6s |
| 5 | qwen3_tool_call | PASS | 110.31s | Tool call: YES \| JSON valid: NO \| Response: <think> Okay, the user is asking for the weather in Pari |
| 6 | qwen3_no_tool_turn | PASS | 63.93s | Direct answer: YES \| Response: <think> Okay The user is asking for the sum of 2 and 2. Since 2 is a  |
| 7 | download_model | PASS | 0.00s | Already downloaded at /root/.tqcli/models/gemma-4-e2b-it-vllm (9803 MB) |
| 8 | quantization_pipeline | PASS | 0.00s | Pipeline: kv:turbo3 \| Precision: full_precision \| Weight quant needed: False \| KV compression: turbo |
| 9 | verify_full_pipeline | PASS | 0.00s | Weight quant needed: False \| Weight method:  \| KV compression: True \| KV level: turbo3 \| Precision:  |
| 10 | load_model_turbo_kv | FAIL | 480.81s | Failed to load with TurboQuant KV: Engine core initialization failed. See root cause above. Failed c |
| 11 | gemma4_hw_limitation | PASS | 0.00s | Gemma 4 E2B may not fit 4096 MB VRAM. |

#### Pipeline Decision
- **Precision:** weight_quantized | **Weight quant:** False () | **KV level:** turbo3 | **KV-only:** True
- **Precision:** full_precision | **Weight quant:** False () | **KV level:** turbo3 | **KV-only:** True

#### Performance
| Step | tok/s | Tokens | Thinking |
|------|-------|--------|----------|
| qwen3_tool_call | 1.12 | 124 | N/A |
| qwen3_no_tool_turn | 0.89 | 57 | N/A |

### Test 7: Combined Thinking + Tool Calling + turboquant35 KV (vLLM)

**Model:** `qwen3-4b-AWQ` | **Engine:** vLLM (TurboQuant fork) | **Result:** **FAIL** (8/9 steps) | **Duration:** 639.5s

| # | Step | Result | Duration | Details |
|---|------|--------|----------|---------|
| 1 | download_model | PASS | 0.00s | Already downloaded at /root/.tqcli/models/qwen3-4b-AWQ (2558 MB) |
| 2 | quantization_pipeline | PASS | 0.00s | Pipeline: kv:turbo3 \| Precision: weight_quantized \| Weight quant needed: False \| KV compression: tur |
| 3 | load_model_turbo_kv | PASS | 115.80s | Loaded Qwen3 4B (AWQ INT4, vLLM) via vLLM with kv_cache_dtype=turboquant35 in 115.8s |
| 4 | qwen3_combined_think | PASS | 169.15s | Thinking: YES (519 chars) \| Response: <tool_call> {"name": "get_weather", "arguments": {"city":{"cit |
| 5 | qwen3_combined_tool_awareness | PASS | 0.00s | Tool/weather referenced in response: YES |
| 6 | download_model | PASS | 0.00s | Already downloaded at /root/.tqcli/models/gemma-4-e2b-it-vllm (9803 MB) |
| 7 | quantization_pipeline | PASS | 0.00s | Pipeline: kv:turbo3 \| Precision: full_precision \| Weight quant needed: False \| KV compression: turbo |
| 8 | load_model_turbo_kv | FAIL | 354.59s | Failed to load with TurboQuant KV: Engine core initialization failed. See root cause above. Failed c |
| 9 | gemma4_hw_limitation | PASS | 0.00s | Gemma 4 E2B may not fit 4096 MB VRAM. |

#### Pipeline Decision
- **Precision:** weight_quantized | **Weight quant:** False () | **KV level:** turbo3 | **KV-only:** True
- **Precision:** full_precision | **Weight quant:** False () | **KV level:** turbo3 | **KV-only:** True

#### Performance
| Step | tok/s | Tokens | Thinking |
|------|-------|--------|----------|
| qwen3_combined_think | 0.98 | 166 | YES |

---
