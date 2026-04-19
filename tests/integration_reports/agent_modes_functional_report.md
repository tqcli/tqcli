# Agent Modes — FUNCTIONAL Integration Test Report

**Generated:** 2026-04-19T14:24:35
**System:** NVIDIA RTX A2000 Laptop GPU (4096 MB VRAM)
**CUDA:** driver 13.0, toolkit 12.8
**OS:** Linux (Ubuntu 22.04.4 LTS) (WSL2)

Exercises the full parse→execute→observation→live-inference loop of `tqcli/core/agent_orchestrator.py` with concrete assertions — spy fidelity, history integrity, filesystem side-effects, and secret-word ingestion into the live KV cache. Zero-shot tests (T0*) are DATA POINTS capturing real-model tag-emission compliance; they do not gate the suite.

---

## Summary

| Test | Engine | Model | Mode | KV Quant | Kind | Result |
|------|--------|-------|------|----------|------|--------|
| T0a | llama.cpp | gemma-4-e2b-it-Q4_K_M | ai_tinkering | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | data-point | **PASS** (1/1) |
| T1_lg | llama.cpp | gemma-4-e2b-it-Q4_K_M | ai_tinkering | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (6/6) |
| T2 | llama.cpp | gemma-4-e2b-it-Q4_K_M | ai_tinkering | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (4/4) |
| T3 | llama.cpp | gemma-4-e2b-it-Q4_K_M | ai_tinkering | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (3/3) |
| T4_lg | llama.cpp | gemma-4-e2b-it-Q4_K_M | unrestricted | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (6/6) |
| T5 | llama.cpp | gemma-4-e2b-it-Q4_K_M | unrestricted | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (3/3) |
| T0b | llama.cpp | qwen3-4b-Q4_K_M | ai_tinkering | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | data-point | **PASS** (1/1) |
| T1_lq | llama.cpp | qwen3-4b-Q4_K_M | ai_tinkering | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (6/6) |
| T4_lq | llama.cpp | qwen3-4b-Q4_K_M | unrestricted | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (6/6) |
| T0c | vllm | gemma-4-e2b-it-vllm | unrestricted | turboquant35 (quant=bitsandbytes, cpu_offload_gb=9.9) | data-point | **PASS** (1/1) |
| T1_vg | vllm | gemma-4-e2b-it-vllm | ai_tinkering | turboquant35 (quant=bitsandbytes, cpu_offload_gb=9.9) | functional | **PASS** (6/6) |
| T4_vg | vllm | gemma-4-e2b-it-vllm | unrestricted | turboquant35 (quant=bitsandbytes, cpu_offload_gb=9.9) | functional | **PASS** (6/6) |
| T0d | vllm | qwen3-4b-vllm | unrestricted | turboquant35 (quant=bitsandbytes, cpu_offload_gb=6.5) | data-point | **PASS** (1/1) |
| T1_vq | vllm | qwen3-4b-vllm | ai_tinkering | turboquant35 (quant=bitsandbytes, cpu_offload_gb=6.5) | functional | **PASS** (6/6) |
| T4_vq | vllm | qwen3-4b-vllm | unrestricted | turboquant35 (quant=bitsandbytes, cpu_offload_gb=6.5) | functional | **PASS** (6/6) |

---

## T0a: T0 zero-shot (ai_tinkering, llama.cpp gemma-4-e2b-it-Q4_K_M)

- **Engine:** llama.cpp
- **Model:** gemma-4-e2b-it-Q4_K_M
- **Mode:** ai_tinkering
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** DATA POINT (no pass gate)
- **Started / Finished:** 2026-04-19T13:52:12 → 2026-04-19T13:52:21
- **Duration:** 8.19s
- **Result:** PASS (1/1)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| zero_shot_data_point | PASS | 8.19s | emitted_staged_tool_call=True / chars=117 / head=<staged_tool_call>{ "name":"tq-file-read", "arguments":{ "path":"/tmp/tqcli_agent_fixture.txt" } }</staged_tool_call> |

## T1_lg: T1 approve actionable (llama.cpp gemma-4-e2b-it-Q4_K_M, ai_tinkering)

- **Engine:** llama.cpp
- **Model:** gemma-4-e2b-it-Q4_K_M
- **Mode:** ai_tinkering
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-19T13:52:21 → 2026-04-19T13:52:27
- **Duration:** 6.70s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| spy_file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| actionable_tool_confirmed_once | PASS | 0.00s | confirm_fn_calls=[('tq-file-read', {'path': '/tmp/tqcli_agent_fixture.txt'}, 'actionable')] |
| observation_appended | PASS | 0.00s | hist_len=5 / hist_before=2 / obs_count=1 / obs_head=Observation:
secret_word=ALPHACHARLIE-2c299d6aefe2
 |
| real_engine_invoked_after_observation | PASS | 0.00s | pb.call_count=2 / scripted=1 / real=1 |
| secret_word_in_real_followup | PASS | 6.70s | secret=ALPHACHARLIE-2c299d6aefe2 / found_in_final=True / final_len=134 / final_head=Thank you for providing the observation.  The secret word is: **ALPHACHARLIE-2c299d6aefe2**  How can I help you with this information? |
| turboquant_kv_active | PASS | 0.00s | kv_params={'cache_type_k': 'turbo3', 'cache_type_v': 'turbo3'} |

## T2: T2 deny actionable (llama.cpp Gemma 4, ai_tinkering)

- **Engine:** llama.cpp
- **Model:** gemma-4-e2b-it-Q4_K_M
- **Mode:** ai_tinkering
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-19T13:52:27 → 2026-04-19T13:52:27
- **Duration:** 0.00s
- **Result:** PASS (4/4)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| confirm_fn_called_once | PASS | 0.00s | confirm_calls=1 |
| terminal_exec_never_called | PASS | 0.00s | calls=[] |
| observation_is_denial | PASS | 0.00s | obs_head=Observation:
User denied execution. Request alternatives. |
| loop_terminated_after_denial | PASS | 0.00s | scripted=1 real=0 |

## T3: T3 edit actionable (llama.cpp Gemma 4, ai_tinkering)

- **Engine:** llama.cpp
- **Model:** gemma-4-e2b-it-Q4_K_M
- **Mode:** ai_tinkering
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-19T13:52:27 → 2026-04-19T13:52:31
- **Duration:** 3.71s
- **Result:** PASS (3/3)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| exec_called_once_with_edited_args | PASS | 0.00s | calls=[{'command': 'touch /tmp/tq_t3_mark'}] |
| edited_mark_exists | PASS | 0.00s | path=/tmp/tq_t3_mark exists=True |
| original_mark_never_created | PASS | 3.71s | path=/tmp/tq_t3_ORIGINAL exists=False |

## T4_lg: T4 unrestricted ReAct (llama.cpp gemma-4-e2b-it-Q4_K_M)

- **Engine:** llama.cpp
- **Model:** gemma-4-e2b-it-Q4_K_M
- **Mode:** unrestricted
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-19T13:52:31 → 2026-04-19T13:52:35
- **Duration:** 3.86s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| confirm_fn_never_called | PASS | 0.00s | confirm_calls=0 |
| file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| observation_carries_secret | PASS | 0.00s | obs_head=Observation:
secret_word=ALPHACHARLIE-98ead8448b43
 |
| real_engine_followup_invoked | PASS | 0.00s | real=1 scripted=1 |
| secret_word_in_real_followup | PASS | 3.86s | secret=ALPHACHARLIE-98ead8448b43 / found_in_final=True / final_head=Okay, I have read the observation.  The secret word is: **ALPHACHARLIE-98ead8448b43** |
| turboquant_kv_active | PASS | 0.00s | kv_params={'cache_type_k': 'turbo3', 'cache_type_v': 'turbo3'} |

## T5: T5 dead tool name (llama.cpp Gemma 4, unrestricted)

- **Engine:** llama.cpp
- **Model:** gemma-4-e2b-it-Q4_K_M
- **Mode:** unrestricted
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-19T13:52:35 → 2026-04-19T13:52:46
- **Duration:** 11.43s
- **Result:** PASS (3/3)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| no_real_tool_invoked | PASS | 0.00s | total_spy_calls=0 |
| observation_is_error | PASS | 0.00s | obs_head=Observation:
ERROR: unknown tool 'nonexistent_tool' |
| orchestrator_did_not_crash | PASS | 11.43s | ran to completion |

## T0b: T0 zero-shot (ai_tinkering, llama.cpp qwen3-4b-Q4_K_M)

- **Engine:** llama.cpp
- **Model:** qwen3-4b-Q4_K_M
- **Mode:** ai_tinkering
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** DATA POINT (no pass gate)
- **Started / Finished:** 2026-04-19T13:52:55 → 2026-04-19T13:53:24
- **Duration:** 28.55s
- **Result:** PASS (1/1)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| zero_shot_data_point | PASS | 28.55s | emitted_staged_tool_call=False / chars=1121 / head=<think> Okay, the user wants me to use the tq-file-read tool to read the file at /tmp/tqcli_agent_fixture.txt. They specified that the response should be exactl |

## T1_lq: T1 approve actionable (llama.cpp qwen3-4b-Q4_K_M, ai_tinkering)

- **Engine:** llama.cpp
- **Model:** qwen3-4b-Q4_K_M
- **Mode:** ai_tinkering
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-19T13:53:24 → 2026-04-19T13:53:49
- **Duration:** 25.43s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| spy_file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| actionable_tool_confirmed_once | PASS | 0.00s | confirm_fn_calls=[('tq-file-read', {'path': '/tmp/tqcli_agent_fixture.txt'}, 'actionable')] |
| observation_appended | PASS | 0.00s | hist_len=5 / hist_before=2 / obs_count=1 / obs_head=Observation:
secret_word=ALPHACHARLIE-a30479282136
 |
| real_engine_invoked_after_observation | PASS | 0.00s | pb.call_count=2 / scripted=1 / real=1 |
| secret_word_in_real_followup | PASS | 25.43s | secret=ALPHACHARLIE-a30479282136 / found_in_final=True / final_len=628 / final_head=<think> Okay, the user asked me to read the fixture, and I used the tool to read the file. The observation shows that the secret word is |
| turboquant_kv_active | PASS | 0.00s | kv_params={'cache_type_k': 'turbo3', 'cache_type_v': 'turbo3'} |

## T4_lq: T4 unrestricted ReAct (llama.cpp qwen3-4b-Q4_K_M)

- **Engine:** llama.cpp
- **Model:** qwen3-4b-Q4_K_M
- **Mode:** unrestricted
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-19T13:53:49 → 2026-04-19T13:54:14
- **Duration:** 24.63s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| confirm_fn_never_called | PASS | 0.00s | confirm_calls=0 |
| file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| observation_carries_secret | PASS | 0.00s | obs_head=Observation:
secret_word=ALPHACHARLIE-2f8df3b6f91b
 |
| real_engine_followup_invoked | PASS | 0.00s | real=1 scripted=1 |
| secret_word_in_real_followup | PASS | 24.63s | secret=ALPHACHARLIE-2f8df3b6f91b / found_in_final=True / final_head=<think> Okay, the user asked me to "read it" after I provided an observation that included a variable called secret_word with a specific value. Let me b |
| turboquant_kv_active | PASS | 0.00s | kv_params={'cache_type_k': 'turbo3', 'cache_type_v': 'turbo3'} |

## T0c: T0 zero-shot (unrestricted, vllm gemma-4-e2b-it-vllm)

- **Engine:** vllm
- **Model:** gemma-4-e2b-it-vllm
- **Mode:** unrestricted
- **KV Quant:** turboquant35 (quant=bitsandbytes, cpu_offload_gb=9.9)
- **Kind:** DATA POINT (no pass gate)
- **Started / Finished:** 2026-04-19T13:57:12 → 2026-04-19T13:58:05
- **Duration:** 52.76s
- **Result:** PASS (1/1)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| zero_shot_data_point | PASS | 52.76s | emitted_tool_call=True / chars=102 / head=    <tool_call>{"name":"tq-file-read","arguments":{"path":"/tmp/tqcli_agent_fixture.txt"}}</tool_call> |

## T1_vg: T1 approve actionable (vllm gemma-4-e2b-it-vllm, ai_tinkering)

- **Engine:** vllm
- **Model:** gemma-4-e2b-it-vllm
- **Mode:** ai_tinkering
- **KV Quant:** turboquant35 (quant=bitsandbytes, cpu_offload_gb=9.9)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-19T13:58:05 → 2026-04-19T14:00:43
- **Duration:** 158.71s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| spy_file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| actionable_tool_confirmed_once | PASS | 0.00s | confirm_fn_calls=[('tq-file-read', {'path': '/tmp/tqcli_agent_fixture.txt'}, 'actionable')] |
| observation_appended | PASS | 0.00s | hist_len=5 / hist_before=2 / obs_count=1 / obs_head=Observation:
secret_word=ALPHACHARLIE-199bbc8a829f
 |
| real_engine_invoked_after_observation | PASS | 0.00s | pb.call_count=2 / scripted=1 / real=1 |
| secret_word_in_real_followup | PASS | 158.71s | secret=ALPHACHARLIE-199bbc8a829f / found_in_final=True / final_len=508 / final_head=<thought>The user asked me to read the fixture. I executed a tool call to read a file named `/tmp/tqcli_agent_fixture.txt`. The observat |
| turboquant_kv_active | PASS | 0.00s | tune.kv_cache_dtype=turboquant35 |

## T4_vg: T4 unrestricted ReAct (vllm gemma-4-e2b-it-vllm)

- **Engine:** vllm
- **Model:** gemma-4-e2b-it-vllm
- **Mode:** unrestricted
- **KV Quant:** turboquant35 (quant=bitsandbytes, cpu_offload_gb=9.9)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-19T14:00:43 → 2026-04-19T14:01:24
- **Duration:** 40.33s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| confirm_fn_never_called | PASS | 0.00s | confirm_calls=0 |
| file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| observation_carries_secret | PASS | 0.00s | obs_head=Observation:
secret_word=ALPHACHARLIE-ad2efa474311
 |
| real_engine_followup_invoked | PASS | 0.00s | real=1 scripted=1 |
| secret_word_in_real_followup | PASS | 40.33s | secret=ALPHACHARLIE-ad2efa474311 / found_in_final=True / final_head=I have read the text.  The content is: ``` secret_word=ALPHACHARLIE-ad2efa474311 ``` |
| turboquant_kv_active | PASS | 0.00s | tune.kv_cache_dtype=turboquant35 |

## T0d: T0 zero-shot (unrestricted, vllm qwen3-4b-vllm)

- **Engine:** vllm
- **Model:** qwen3-4b-vllm
- **Mode:** unrestricted
- **KV Quant:** turboquant35 (quant=bitsandbytes, cpu_offload_gb=6.5)
- **Kind:** DATA POINT (no pass gate)
- **Started / Finished:** 2026-04-19T14:04:54 → 2026-04-19T14:10:05
- **Duration:** 311.50s
- **Result:** PASS (1/1)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| zero_shot_data_point | PASS | 311.50s | emitted_tool_call=True / chars=339 / head=<think>      </think>                                 <tool_call>  <tool_call>  </think>            <tool_call>{"{"{"{"{"{"{""name"arguments"""的"的""的""的"的"""的"的 |

## T1_vq: T1 approve actionable (vllm qwen3-4b-vllm, ai_tinkering)

- **Engine:** vllm
- **Model:** qwen3-4b-vllm
- **Mode:** ai_tinkering
- **KV Quant:** turboquant35 (quant=bitsandbytes, cpu_offload_gb=6.5)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-19T14:10:05 → 2026-04-19T14:18:34
- **Duration:** 508.63s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| spy_file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| actionable_tool_confirmed_once | PASS | 0.00s | confirm_fn_calls=[('tq-file-read', {'path': '/tmp/tqcli_agent_fixture.txt'}, 'actionable')] |
| observation_appended | PASS | 0.00s | hist_len=5 / hist_before=2 / obs_count=1 / obs_head=Observation:
secret_word=ALPHACHARLIE-a893599e62fd
 |
| real_engine_invoked_after_observation | PASS | 0.00s | pb.call_count=2 / scripted=1 / real=1 |
| secret_word_in_real_followup | PASS | 508.63s | secret=ALPHACHARLIE-a893599e62fd / found_in_final=True / final_len=444 / final_head=<think>      </think>    The secret word is is a明月宝相似空明，但根据提供的信息，`的的的观察是“秘密词”为 "ALPHACHARLIE-a8935999e62fd"。不过，这似乎与实际的“秘密词”不符，因为实际的“秘密词” |
| turboquant_kv_active | PASS | 0.00s | tune.kv_cache_dtype=turboquant35 |

## T4_vq: T4 unrestricted ReAct (vllm qwen3-4b-vllm)

- **Engine:** vllm
- **Model:** qwen3-4b-vllm
- **Mode:** unrestricted
- **KV Quant:** turboquant35 (quant=bitsandbytes, cpu_offload_gb=6.5)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-19T14:18:34 → 2026-04-19T14:24:34
- **Duration:** 359.63s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| confirm_fn_never_called | PASS | 0.00s | confirm_calls=0 |
| file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| observation_carries_secret | PASS | 0.00s | obs_head=Observation:
secret_word=ALPHACHARLIE-c37aff72c685
 |
| real_engine_followup_invoked | PASS | 0.00s | real=1 scripted=1 |
| secret_word_in_real_followup | PASS | 359.63s | secret=ALPHACHARLIE-c37aff72c685 / found_in_final=True / final_head=<think> <think>     </think>        The secret word is is the secret word.. The secret word is "AL法CHAR"奉" $"char" and the secret word is is "ALPHACHARL |
| turboquant_kv_active | PASS | 0.00s | tune.kv_cache_dtype=turboquant35 |
