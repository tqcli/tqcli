# Resume Prompt: TurboQuant KV Cache Integration

**Last Session:** 2026-04-14 15:10 EDT
**GitHub Issue:** ithllc/tqCLI#13
**Last Commit:** 7259cf1 (WIP: KV quantizer module + llama.cpp turbo cache types)

---

## Copy-paste this prompt to resume:

```
Resume the TurboQuant KV cache integration for GitHub issue ithllc/tqCLI#13. The session was paused at 3:10 PM on April 14.

Check your memory file at /root/.claude/projects/-llm-models-python-code-src-tqCLI/memory/project_turboquant_kv_progress.md for full status.

Summary of where we stopped:
1. llama-cpp-turboquant fork is cloned at /tmp/llama-cpp-turboquant/ (CPU-only build works, turbo3 KV types verified)
2. BLOCKER: System nvcc is CUDA 11.5 — too old. Need CUDA toolkit 12.x for GPU compilation.
3. tqcli/core/kv_quantizer.py is written and committed
4. --kv-quant CLI flag is wired into tqcli chat
5. LlamaBackend has cache_type_k/cache_type_v params

What needs to be done (in order):

1. **Install CUDA 12.8 toolkit** (the full toolkit with nvcc, not just runtime):
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   dpkg -i cuda-keyring_1.1-1_all.deb
   apt-get update && apt-get install -y cuda-toolkit-12-8
   export PATH=/usr/local/cuda-12.8/bin:$PATH
   ```

2. **Rebuild llama-cpp-turboquant with CUDA** (SM86 for RTX A2000):
   ```bash
   cd /tmp/llama-cpp-turboquant
   rm -rf build
   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86" -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j$(nproc)
   ```

3. **Test turbo3 with GPU offload** — should now run at reasonable speed:
   ```bash
   /tmp/llama-cpp-turboquant/build/bin/llama-cli \
     -m /root/.tqcli/models/Qwen3-4B-Q4_K_M.gguf \
     --cache-type-k turbo3 --cache-type-v turbo3 \
     -ngl 99 -n 32 -c 2048 \
     -p "What is the capital of France? Answer in one sentence."
   ```

4. **Build vLLM TurboQuant** (Prompt 2 from docs/prompts/run_turboquant_kv_integration_tests.md):
   Clone mitkox/vllm-turboquant, build from source with CUDA 12.8

5. **Wire VllmBackend** for turboquant KV dtype (not yet done)

6. **Write test_integration_turboquant_kv.py** (Prompt 4)

7. **Run all 4 integration tests** (Prompt 4)

8. **Post-integration tasks** (Prompt 5): review issues, update docs, close issues, commit v0.5.0, push

If you encounter issues, file them via gh issue create on ithllc/tqCLI, fix them, and comment solutions. Use your issue-manager and project-manager skill sets as needed.
```
