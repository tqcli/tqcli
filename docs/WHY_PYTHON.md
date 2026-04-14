# Why Python Only?

A fair question — most popular model harness CLIs use Node.js:

| CLI | Language | What It Does |
|-----|----------|-------------|
| Claude Code | TypeScript/Node | API client to Anthropic servers |
| Gemini CLI | Node.js | API client to Google servers |
| Aider | Python | API client + local git operations |
| **tqCLI** | **Python** | **Local inference engine** |

The first two are **API clients**. They format prompts, send HTTP requests to a cloud endpoint, and render the streamed response. The heavy computation (running the model) happens on Anthropic's or Google's servers. Node.js is a great fit for that — excellent async I/O, fast terminal rendering, mature CLI tooling (ink, commander, etc.).

tqCLI is fundamentally different. **It loads multi-gigabyte model files into local memory and runs inference on your hardware.** The model runs in-process, not on a remote server.

## The Binding Reality

The two inference engines tqCLI supports have their primary, maintained bindings in Python:

- **llama-cpp-python** — official Python bindings for llama.cpp. Maintained by the llama.cpp community. Supports Metal (macOS), CUDA (Linux/Windows), and CPU fallback.
- **vLLM** — Python-native from the ground up. No official bindings for any other language.

Node.js alternatives exist but have significant gaps:

| Feature | Python bindings | Node.js bindings |
|---------|----------------|-----------------|
| llama.cpp support | `llama-cpp-python` (official) | `node-llama-cpp` (community) |
| vLLM support | Native | None |
| HuggingFace Hub | `huggingface_hub` (official) | `@huggingface/hub` (limited) |
| GPU detection (CUDA) | `torch`, `pynvml`, `psutil` | `systeminformation` (less reliable) |
| Model conversion | `transformers`, `safetensors` | None |

Using Node.js for tqCLI would mean either:
1. Shelling out to Python for everything that matters (inference, model management, GPU detection)
2. Using less mature, community-maintained bindings that lag behind official releases
3. Losing vLLM support entirely

## The Ecosystem Argument

The ML/AI tooling ecosystem is overwhelmingly Python:

- Model conversion: `convert_hf_to_gguf.py` (part of llama.cpp)
- Quantization: `auto-gptq`, `autoawq`, llama.cpp's `quantize` tool (called from Python)
- Benchmarking: `lm-eval`, `vllm.entrypoints.openai`
- Model hosting: HuggingFace (Python SDK is the primary interface)

Contributors working on quantization, model support, or inference optimization will already be Python developers. Requiring them to also write TypeScript would shrink the contributor pool.

## What About a Hybrid?

A Node.js frontend + Python backend is a valid architecture:

```
Node.js CLI (terminal UI, rich rendering)
    ↕ HTTP/IPC
Python inference server (llama.cpp, vLLM)
```

This would give the best terminal UI experience (Node's `ink` library is excellent) while keeping inference in Python. The tradeoff:

- More complex setup (two runtimes required)
- IPC overhead for every token (adds latency)
- Two dependency ecosystems to manage
- Harder for contributors to understand the full stack

For v0.1, the simplicity of a single-language stack outweighs the UI advantages of a hybrid. This is a conscious tradeoff that may be revisited as the project matures.

## Summary

| Factor | Python | Node.js |
|--------|--------|---------|
| Inference bindings | Native, official | Community, incomplete |
| vLLM support | Yes | No |
| GPU/hardware detection | Excellent | Adequate |
| ML ecosystem | Home turf | Foreign territory |
| CLI/terminal UI | Good (Rich, Click) | Excellent (ink, commander) |
| Contributor pool for ML | Large | Small |
| Setup complexity | Single runtime | Dual runtime (if hybrid) |

Python is the right choice for a tool whose primary job is running ML models on local hardware. If tqCLI were an API client, Node.js would be the right choice.
