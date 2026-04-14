# tqCLI (TurboQuant CLI)

A cross-platform CLI for **local LLM inference** using quantized models, with smart routing, real-time performance monitoring, and automatic handoff to frontier model CLIs when local inference falls below acceptable thresholds.

Built with [TurboQuant](https://arxiv.org/abs/2504.19874) methodologies — applying quantization best practices from Google Research's ICLR 2026 paper on lossless 3-bit KV cache compression.

## What It Does

- **Run quantized LLMs locally** on Mac, Linux, or Windows via [llama.cpp](https://github.com/ggerganov/llama.cpp) or [vLLM](https://github.com/vllm-project/vllm)
- **Smart routing** automatically dispatches prompts to the best model for the task (coding, reasoning, general)
- **Performance monitoring** tracks tokens/second in real-time with live stats display
- **Automatic handoff** generates context files to transfer work to Claude Code, Gemini CLI, or Aider when local inference is too slow
- **Security isolation** via Python venvs, resource guards, and audit logging
- **Skills system** with discoverable, extensible skills (like Claude Code's skill architecture)

## Supported Models

| Model Family | Strengths | Best For |
|-------------|-----------|----------|
| **Google Gemma 4** (12B, 27B) | General reasoning, multilingual, instruction following | Q&A, analysis, general tasks |
| **Qwen2.5-Coder** (7B, 32B) | Code generation, debugging, code review | Coding, code explanation, debugging |
| **Qwen2.5-Instruct** (7B, 32B) | Instruction following, conversation | Chat, summarization, translation |

All models available as GGUF quantized files (Q2_K through Q8_0) for llama.cpp, or AWQ/GPTQ for vLLM.

## Platform Support

| Platform | llama.cpp | vLLM | Notes |
|----------|-----------|------|-------|
| macOS (Apple Silicon) | Full (Metal GPU) | No | Unified memory enables large models |
| macOS (Intel) | CPU only | No | |
| Linux + NVIDIA GPU | Full (CUDA) | Full | Best vLLM experience |
| Linux (CPU only) | Full | No | |
| Windows (WSL2 + NVIDIA) | Full (CUDA) | Full | Recommended for Windows |
| Windows (Native) | CPU only | No | WSL2 recommended |

## Quick Start

### Install

```bash
# Clone the repository
git clone https://github.com/ithllc/tqCLI.git
cd tqCLI

# Install with llama.cpp backend (most platforms)
pip install -e ".[llama]"

# Or with vLLM backend (Linux/WSL2 + NVIDIA GPU)
pip install -e ".[vllm]"

# Or both
pip install -e ".[all]"
```

### Initialize

```bash
# Set up configuration and directories
tqcli config init

# Check your system capabilities
tqcli system info
```

### Download a Model

```bash
# List available models
tqcli model list

# Download based on your hardware (see system info output)
tqcli model pull gemma-4-12b-it-Q4_K_M        # 12B general-purpose (needs ~8GB)
tqcli model pull qwen2.5-coder-7b-instruct-Q4_K_M  # 7B coding (needs ~6GB)
tqcli model pull qwen2.5-7b-instruct-Q4_K_M        # 7B instruction (needs ~6GB)
```

### Chat

```bash
# Interactive chat (auto-selects best model via router)
tqcli

# Chat with a specific model
tqcli chat --model qwen2.5-coder-7b-instruct-Q4_K_M

# In-session commands:
#   /stats    — Show performance statistics
#   /handoff  — Generate handoff file for frontier CLI
#   /help     — Show help
#   /quit     — Exit
```

## Smart Routing

When multiple models are installed, tqCLI automatically routes prompts to the best model:

```
User: "Write a Python function to parse JSON"
  -> Routes to Qwen2.5-Coder (coding: 0.95 score)

User: "Explain the trade-offs between TCP and UDP"
  -> Routes to Gemma 4 (reasoning: 0.90 score)

User: "Summarize this article for me"
  -> Routes to Qwen2.5-Instruct (instruction: 0.92 score)
```

The router classifies prompts using keyword and pattern heuristics, then ranks available models by their domain-specific strength scores. If only one model is loaded, classification is skipped.

## Performance Monitoring & Handoff

tqCLI tracks tokens/second during every inference:

```
  32.8 tok/s | 128 tokens | 3.90s
```

When performance drops below threshold (default: 5 tok/s), tqCLI generates a **handoff file** — a self-contained markdown document with:
- Task description and conversation history
- Performance stats explaining why the handoff occurred  
- CLI-specific instructions for continuing with Claude Code, Gemini CLI, or Aider

```bash
# Manual handoff
tqcli handoff -t "Implement authentication middleware" --target claude-code

# Auto-handoff (enable in config)
# Set performance.auto_handoff: true in ~/.tqcli/config.yaml
```

## Security

tqCLI runs in an isolated environment by default:

- **Virtual environment** — Separate Python venv at `~/.tqcli/venv`
- **Resource guards** — Memory and GPU limits prevent system overload
- **Audit logging** — All model loads, downloads, and security events logged
- **Environment detection** — Identifies WSL2, containers, bare-metal

```bash
# Run security audit
tqcli security audit

# Auto-fix safe issues
tqcli security audit --fix
```

## Skills

tqCLI includes a skills system inspired by Claude Code:

| Skill | Description |
|-------|-------------|
| `tq-system-info` | Detect OS, hardware, recommend engine and quantization |
| `tq-model-manager` | Download, list, and remove quantized models |
| `tq-benchmark` | Benchmark models for tokens/second performance |
| `tq-security-audit` | Audit environment isolation and security posture |
| `tq-handoff-generator` | Generate handoff files for frontier model CLIs |

```bash
tqcli skills  # List all available skills
```

## Configuration

Configuration lives at `~/.tqcli/config.yaml`:

```yaml
models_dir: ~/.tqcli/models
preferred_engine: auto          # auto, llama.cpp, vllm
default_quantization: Q4_K_M
context_length: 4096
n_gpu_layers: -1                # -1 = all layers to GPU

performance:
  min_tokens_per_second: 5.0    # Below this = handoff trigger
  warning_tokens_per_second: 10.0
  auto_handoff: false

security:
  use_venv: true
  sandbox_enabled: true
  audit_log: true
  max_memory_percent: 80.0
  max_gpu_memory_percent: 90.0

router:
  enabled: true
```

## TurboQuant Reference

This project applies methodologies from [Google's TurboQuant research](https://arxiv.org/abs/2504.19874):

- **PolarQuant**: Polar coordinate transformation for lossless KV cache compression
- **QJL**: Quantized Johnson-Lindenstrauss sign-bit error correction
- **Result**: 6x+ KV cache compression at 3-bit with zero accuracy loss

The quantization decision tree in tqCLI (engine selection, quant level recommendation, model sizing) is informed by the comparative analysis in the [GoogleTurboQuant](https://github.com/ithllc/GoogleTurboQuant) workspace.

## License

MIT License. See [LICENSE](LICENSE) for details.
