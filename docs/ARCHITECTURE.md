# tqCLI Architecture

This document describes the system design of tqCLI for contributors who need to understand the codebase before making changes.

## High-Level Architecture

```
User
  │
  ▼
┌──────────┐
│  CLI     │  tqcli/cli.py — Click command groups
│  (Click) │  Commands: chat, system, model, benchmark, security, skills, handoff, config
└────┬─────┘
     │
     ▼
┌──────────────────────────────────────────────────┐
│                    Core Layer                     │
│                                                  │
│  ┌─────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ Router  │  │ Registry │  │ Performance    │  │
│  │         │──│          │  │ Monitor        │  │
│  │classify │  │ profiles │  │ tok/s tracking │  │
│  │& rank   │  │ & scan   │  │ & thresholds   │  │
│  └────┬────┘  └────┬─────┘  └───────┬────────┘  │
│       │            │                │            │
│       ▼            ▼                ▼            │
│  ┌─────────────────────────────────────────────┐ │
│  │           Inference Engine (ABC)            │ │
│  │                                             │ │
│  │  ┌──────────────┐  ┌────────────────────┐   │ │
│  │  │ LlamaBackend │  │   VllmBackend      │   │ │
│  │  │ (llama.cpp)  │  │   (vLLM)           │   │ │
│  │  │ Mac/Lin/Win  │  │   Linux+NVIDIA     │   │ │
│  │  └──────────────┘  └────────────────────┘   │ │
│  └─────────────────────────────────────────────┘ │
│                                                  │
│  ┌────────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ SystemInfo │  │ Security │  │  Handoff    │  │
│  │ OS/HW     │  │ venv,    │  │  generator  │  │
│  │ detection  │  │ audit,   │  │  for Claude │  │
│  │           │  │ guards   │  │  /Gemini    │  │
│  └────────────┘  └──────────┘  └─────────────┘  │
└──────────────────────────────────────────────────┘
     │
     ▼
┌──────────┐
│    UI    │  tqcli/ui/ — Rich console output, interactive chat
└──────────┘
```

## Module Responsibilities

### `tqcli/cli.py` — Entry Point

The Click-based CLI. Each subcommand (`chat`, `model list`, `system info`, etc.) is defined here. Commands import core modules lazily to keep startup fast.

**Key design decision:** The `chat` command is the default when `tqcli` is invoked with no subcommand.

### `tqcli/config.py` — Configuration

Dataclass-based configuration loaded from `~/.tqcli/config.yaml`. Nested configs for performance thresholds, security settings, and router preferences.

### `tqcli/core/engine.py` — Inference Abstraction

Abstract base class `InferenceEngine` that defines the interface all backends must implement:

- `load_model(path)` / `unload_model()`
- `chat(messages)` — full response
- `chat_stream(messages)` — streaming generator yielding `(text_chunk, final_stats)`
- `complete(prompt)` — raw completion

Both backends return `CompletionResult` with `InferenceStats` (tokens, timing, tok/s).

### `tqcli/core/llama_backend.py` / `vllm_backend.py` — Backends

Concrete implementations of `InferenceEngine`. Each wraps its respective library and normalizes the output into `CompletionResult`.

**llama_backend** supports streaming natively. **vllm_backend** does full generation then yields chunks (vLLM's streaming requires the async engine which is not used in CLI mode).

### `tqcli/core/model_registry.py` — Model Catalog

`ModelProfile` dataclass stores everything about a model: HF repo, filename, parameter count, quantization, strengths, minimum hardware requirements.

`BUILTIN_PROFILES` is the hardcoded catalog of known models. `ModelRegistry` manages discovery (scanning disk for downloaded models) and querying (filter by domain, check hardware fit).

**Strength scores** are `0.0-1.0` floats per task domain, derived from public benchmarks. These drive routing decisions.

### `tqcli/core/router.py` — Smart Routing

Two-phase routing:

1. **Classify** — regex-based keyword matching assigns a `TaskDomain` (coding, math, reasoning, creative, general, instruction) with a confidence score.
2. **Rank** — available models are sorted by their `strength_score` for that domain, filtered by hardware constraints, and the top model is selected.

Single-model bypass: if only one model is loaded, skip classification entirely.

### `tqcli/core/performance.py` — Performance Monitor

Maintains a rolling window of `PerfSample` (timestamp, tokens, elapsed). Computes:

- Current/average/rolling tok/s
- Whether below threshold (handoff trigger)
- Whether in warning zone
- Slow inference ratio (what fraction of inferences were below threshold)

### `tqcli/core/handoff.py` — Frontier CLI Handoff

Generates self-contained markdown files with conversation context, performance stats, and CLI-specific instructions for Claude Code, Gemini CLI, Aider, or OpenAI.

### `tqcli/core/security.py` — Security Layer

- `EnvironmentDetector` — identifies WSL2, container, venv, or bare-metal
- `VenvManager` — creates and manages Python virtual environments
- `ResourceGuard` — checks RAM and GPU memory before model loads
- `AuditLogger` — append-only JSON-lines log of security events
- `SecurityManager` — coordinates all of the above

### `tqcli/core/system_info.py` — Hardware Detection

Cross-platform detection of OS, CPU, RAM, GPU (via `nvidia-smi`), Apple Silicon Metal, WSL2. Returns a `SystemInfo` dataclass with computed properties like `recommended_engine` and `max_model_size_estimate_gb`.

### `tqcli/skills/` — Skills System

Mirrors Claude Code's skill architecture:

- `loader.py` — discovers skills by scanning for `SKILL.md` files in `.claude/skills/`
- `base.py` — `BuiltinSkill` ABC for Python-implemented skills
- `builtin/` — concrete skill implementations

### `tqcli/ui/` — User Interface

- `console.py` — Rich-based output (tables, panels, colored stats)
- `interactive.py` — `InteractiveSession` manages the chat loop with streaming, routing, and performance monitoring

## Data Flow: Chat Request

```
1. User types prompt
2. InteractiveSession receives input
3. Router classifies prompt → TaskDomain
4. Router ranks available models by strength_score for that domain
5. If best model differs from loaded model → unload/load
6. Engine.chat_stream() called
7. Tokens stream to UI via Rich Live display
8. Final stats recorded by PerformanceMonitor
9. If below threshold → warning or handoff
```

## Adding a New Backend

1. Create `tqcli/core/new_backend.py`
2. Subclass `InferenceEngine`
3. Implement all abstract methods
4. Add engine selection logic in `cli.py` `chat` command
5. Add optional dependency in `pyproject.toml`
6. Add tests

## Adding a New Model Family

1. Add `ModelProfile` entries to `BUILTIN_PROFILES` in `model_registry.py`
2. Set `strength_scores` based on published benchmarks (cite sources in comments)
3. Set `min_ram_mb` and `min_vram_mb` based on quantized file sizes + overhead
4. Add tests for the new profiles
