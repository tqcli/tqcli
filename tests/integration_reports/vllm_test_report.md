# tqCLI Integration Test Report -- vLLM Backend

**Date:** 2026-04-14
**tqCLI Version:** 0.3.2
**Backend:** vLLM 0.19.0
**Test Runner:** Automated Python integration tests (`tests/test_integration_vllm.py`)

## System Information

| Property | Value |
|----------|-------|
| os | Linux (Ubuntu 22.04.4 LTS) (WSL2) |
| arch | x86_64 |
| cpu_cores | 16 |
| cpu_physical | 8 |
| ram_total_mb | 31956 |
| ram_available_mb | 24686 |
| gpu | NVIDIA RTX A2000 Laptop GPU |
| vram_mb | 4096 |
| recommended_engine | llama.cpp |
| recommended_quant | Q3_K_M |
| max_model_gb | 3.4 |
| is_wsl | True |
| vllm_available | True |

## Overall Summary

| Metric | Value |
|--------|-------|
| Total Tests | 4 |
| Total Steps Executed | 31 |
| Steps Passed | 31 |
| Steps Failed | 0 |
| Pass Rate | 100.0% |

---

## Test 1: Qwen 3 4B AWQ + vLLM Full Lifecycle

**Model:** `qwen3-4b-AWQ` | **Engine:** vllm | **Result:** **PASS** (10/10 steps)

### Step Results

| # | Step | Result | Duration | Details |
|---|------|--------|----------|---------|
| 1 | hardware_model_selection | PASS | 0.00s | Selected qwen3-4b-AWQ (4B, AWQ) from 1 fitting vLLM models |
| 2 | verify_vllm_quantization | PASS | 0.00s | Quantization: AWQ, Format: awq, Engine: vllm |
| 3 | download_model | PASS | 83.01s | Downloaded to /root/.tqcli/models/qwen3-4b-AWQ (2558 MB) in 83.0s |
| 4 | load_model | PASS | 164.40s | Loaded Qwen3 4B (AWQ INT4, vLLM) via vLLM in 164.4s |
| 5 | chat_turn_1 | PASS | 13.12s | Response (381 chars): <think> Okay, the user is asking what 2 + 2 is and wants the answer just with  |
| 6 | chat_turn_2 | PASS | 16.85s | Response (440 chars): <think> Okay, the user is asking, "What is 4 times 10? Answer with just the nu |
| 7 | generate_skill | PASS | 0.33s | Created skill at /root/.tqcli/skills/test-qwen3-vllm-skill: Skill created: /root/.tqcli/skills/test- |
| 8 | verify_skill | PASS | 0.45s | Output: Running run_test_qwen3_vllm_skill.py... {   "skill": "test-qwen3-vllm-skill",   "status": "c |
| 9 | remove_model | PASS | 1.02s | Removed model directory: /root/.tqcli/models/qwen3-4b-AWQ |
| 10 | clean_uninstall_check | PASS | 7.06s | Package is installed and can be cleanly uninstalled via 'pip3 uninstall tqcli' |

### Performance Metrics

| Step | Tokens/s | Completion Tokens | Total Time (s) |
|------|----------|-------------------|----------------|
| chat_turn_1 | 7.56 | 99 | 13.1 |
| chat_turn_2 | 7.48 | 126 | 16.85 |

---

## Test 2: Gemma 4 E2B BF16 + vLLM Full Lifecycle

**Model:** `` | **Engine:** vllm | **Result:** **PASS** (2/2 steps)

### Step Results

| # | Step | Result | Duration | Details |
|---|------|--------|----------|---------|
| 1 | hardware_model_selection | PASS | 0.00s | No vLLM gemma4 model fits hardware (RAM=23916MB, VRAM=4096MB) [EXPECTED: Gemma 4 BF16 needs >= 6 GB  |
| 2 | hw_limitation_note | PASS | 0.00s | Gemma 4 vLLM models require >= 6 GB VRAM. System has 4096 MB. This is a hardware limitation, not a b |

---

## Test 3: Qwen 3 Multi-Process + Yolo Mode CRM Build (vLLM)

**Model:** `qwen3-4b-AWQ` | **Engine:** vllm (server) | **Result:** **PASS** (12/12 steps)

### Step Results

| # | Step | Result | Duration | Details |
|---|------|--------|----------|---------|
| 1 | hardware_model_selection | PASS | 0.00s | Selected qwen3-4b-AWQ (4B, AWQ) from 1 fitting vLLM models |
| 2 | download_model | PASS | 65.11s | Downloaded to /root/.tqcli/models/qwen3-4b-AWQ (2558 MB) in 65.1s |
| 3 | multiprocess_assessment_yolo | PASS | 0.00s | Engine: vllm, Max workers: 1, Recommended: 1 |
| 4 | multiprocess_serve_start | PASS | 59.71s | Output: ┌─────────────────────────────────────────────────────────────────┐ │  stop-trying-to-contro |
| 5 | multiprocess_serve_status | PASS | 0.72s | Output: Inference Server Status    Status:  running   PID:     120477   Engine:  vllm   Model:   /ro |
| 6 | generate_crm_skills | PASS | 1.66s | Created 3/3 CRM skills |
| 7 | create_crm_workspace | PASS | 0.00s | Created CRM workspace at /llm_models_python_code_src/crm_workspace_vllm |
| 8 | verify_crm_workspace | PASS | 0.00s | Files: {'frontend/index.html': True, 'backend/app.py': True, 'database/schema.sql': True} |
| 9 | delete_crm_workspace | PASS | 0.00s | Deleted workspace at /llm_models_python_code_src/crm_workspace_vllm |
| 10 | multiprocess_serve_stop | PASS | 3.77s | Output: Stopping inference server... Server stopped. |
| 11 | remove_model | PASS | 1.44s | Removed model directory: /root/.tqcli/models/qwen3-4b-AWQ |
| 12 | clean_uninstall_check | PASS | 8.70s | Package is installed and can be cleanly uninstalled via 'pip3 uninstall tqcli' |

---

## Test 4: Gemma 4 Multi-Process + Yolo Mode CRM Build (vLLM)

**Model:** `` | **Engine:** vllm (server) | **Result:** **PASS** (7/7 steps)

### Step Results

| # | Step | Result | Duration | Details |
|---|------|--------|----------|---------|
| 1 | hardware_model_selection | PASS | 0.00s | No vLLM gemma4 model fits hardware (RAM=23829MB, VRAM=4096MB) [EXPECTED: Gemma 4 BF16 needs >= 6 GB  |
| 2 | hw_limitation_note | PASS | 0.00s | Gemma 4 vLLM needs >= 6 GB VRAM. System has 4096 MB. Running CRM skills/workspace only. |
| 3 | generate_crm_skills | PASS | 1.68s | Created 3/3 CRM skills |
| 4 | create_crm_workspace | PASS | 0.00s | Created CRM workspace at /llm_models_python_code_src/crm_workspace_vllm |
| 5 | verify_crm_workspace | PASS | 0.00s | Files: {'frontend/index.html': True, 'backend/app.py': True, 'database/schema.sql': True} |
| 6 | delete_crm_workspace | PASS | 0.00s | Deleted workspace at /llm_models_python_code_src/crm_workspace_vllm |
| 7 | clean_uninstall_check | PASS | 9.03s | Package is installed and can be cleanly uninstalled via 'pip3 uninstall tqcli' |

---

## Performance Comparison (vLLM vs llama.cpp)

| Metric | llama.cpp (prior test) | vLLM (this test) |
|--------|-----------------------|------------------|
| Qwen 3 4B tok/s | 6-9 (Q4_K_M) | See results above (AWQ) |
| Multi-process mode | Sequential queue | Continuous batching |
| KV Cache | Per-request | PagedAttention (shared) |
| Quantization | GGUF Q4_K_M | AWQ INT4 / BF16 |
