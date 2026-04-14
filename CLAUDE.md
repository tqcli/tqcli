# tqCLI — TurboQuant CLI

## Project Purpose
Cross-platform CLI for local LLM inference using quantized models (Gemma 4 series, Qwen 3, Qwen3-Coder) with smart routing, Qwen 3 thinking mode, TurboQuant KV cache compression, unified quantization pipeline, performance monitoring, and automatic handoff to frontier model CLIs when local inference is too slow. Built on llama.cpp and vLLM inference engines with TurboQuant methodologies from the GoogleTurboQuant workspace.

## Workspace Structure
```
tqcli/                     — Python package (pip-installable)
  cli.py                   — Click-based CLI entry point
  config.py                — YAML configuration management
  core/
    system_info.py         — OS + hardware detection (cross-platform)
    engine.py              — Abstract inference engine interface
    llama_backend.py       — llama.cpp backend (Mac/Linux/Windows)
    vllm_backend.py        — vLLM backend (Linux/WSL2 + NVIDIA)
    model_registry.py      — Model catalog with capability profiles
    kv_quantizer.py        — TurboQuant KV cache compression + unified pipeline
    quantizer.py           — Weight quantization selection (bnb INT4, AWQ, GGUF)
    router.py              — Smart prompt → model routing
    performance.py         — Token/s tracking and threshold detection
    handoff.py             — Frontier model CLI handoff generator
    vllm_config.py         — Hardware-aware vLLM configuration builder
    security.py            — Venv isolation, audit logging, resource guards
    server.py              — Inference server management (llama.cpp/vLLM HTTP server)
    server_client.py       — HTTP client backend (connects workers to server)
    multiprocess.py        — Multi-process coordinator and resource assessment
    unrestricted.py        — stop-trying-to-control-everything-and-just-let-go mode
  skills/
    loader.py              — Skill discovery from SKILL.md files
    base.py                — Base class for builtin skills
    builtin/               — Python-implemented skills
  ui/
    console.py             — Rich console output
    interactive.py         — Interactive chat session
tests/                     — Test suite
  integration_reports/     — Test reports (llama.cpp and vLLM)
.claude/skills/            — Claude Code skills (tq-* are tqCLI-specific)
```

## Key Skills (in .claude/skills/)
### tqCLI-Specific
- **tq-system-info** — Detect OS, hardware, recommend engine/quant (`/tq-system-info`)
- **tq-model-manager** — Download, list, remove models (`/tq-model-manager`)
- **tq-benchmark** — Benchmark models for tok/s performance (`/tq-benchmark`)
- **tq-security-audit** — Audit environment isolation and security (`/tq-security-audit`)
- **tq-handoff-generator** — Generate handoff files for frontier CLIs (`/tq-handoff-generator`)
- **tq-multi-process** — Multi-process inference orchestration (`/tq-multi-process`)
- **tq-model-updater** — Research and update model registry with latest HF repos (`/tq-model-updater`)

### Inherited from CLI_Skills
- **project-manager** — Parallel task orchestration (`/project-manager`)
- **prd-generator** — PRD creation (`/prd-generator`)
- **technical-planner** — Implementation planning (`/technical-planner`)
- **feedback-helper** — Feedback capture (`/feedback-helper`)
- **skill-evolution-manager** — Skill improvement (`/skill-evolution-manager`)

## Conventions
- Python 3.10+ required
- CLI uses Click for commands, Rich for output
- Config at ~/.tqcli/config.yaml
- Models stored at ~/.tqcli/models/
- All inference goes through the abstract InferenceEngine interface
- Router classifies prompts using keyword heuristics, ranks models by strength_scores
- Performance threshold: 5 tok/s minimum, 10 tok/s warning
- Security: venv isolation, audit logging, resource guards
- Multi-process: shared HTTP server + N worker processes
- Unrestricted mode: --stop-trying-to-control-everything-and-just-let-go bypasses guards
- TurboQuant KV: runtime detection, graceful degradation — ONE tqcli version for all CUDA versions
- Unified quantization pipeline: full precision → weight quant + KV; pre-quantized → KV only
- TurboQuant forks: ithllc/llama-cpp-turboquant, ithllc/vllm-turboquant (CUDA 12.8+)

## Quick Commands
```bash
tqcli                          # Start interactive chat (single-process)
tqcli system info              # Show hardware + TurboQuant KV status
tqcli model list               # List models
tqcli model pull <id>          # Download a model
tqcli benchmark                # Run benchmarks
tqcli security audit           # Security check
tqcli handoff -t "task desc"   # Generate handoff to frontier CLI
tqcli skills                   # List available skills
tqcli config init              # Initialize configuration

# TurboQuant KV cache compression
tqcli chat --kv-quant auto     # Auto-select KV compression (default)
tqcli chat --kv-quant turbo3   # Force turbo3 (4.6x, +1% PPL)
tqcli chat --kv-quant turbo4   # Force turbo4 (3.8x, near-lossless)
tqcli chat --kv-quant turbo2   # Force turbo2 (6.4x, noticeable)
tqcli chat --kv-quant none     # Disable KV compression

# Multi-process
tqcli serve start              # Start shared inference server
tqcli chat --engine server     # Connect as a worker
tqcli workers spawn 3          # Spawn 3 worker processes
tqcli serve status             # Check server status
tqcli serve stop               # Stop server + workers

# Unrestricted mode
tqcli --stop-trying-to-control-everything-and-just-let-go chat
tqcli --stop-trying-to-control-everything-and-just-let-go serve start
```

## Reference
- GoogleTurboQuant workspace: ../GoogleTurboQuant/
- TurboQuant paper: arxiv.org/abs/2504.19874
- PolarQuant paper: arxiv.org/abs/2502.02617
- llama.cpp TurboQuant fork: github.com/ithllc/llama-cpp-turboquant
- vLLM TurboQuant fork: github.com/ithllc/vllm-turboquant
- GitHub Issue: ithllc/tqCLI#13
