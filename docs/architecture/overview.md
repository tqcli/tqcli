# Architecture Overview

## Module map

```mermaid
flowchart LR
    classDef core fill:#e3f2fd,stroke:#1976d2
    classDef backend fill:#fff3e0,stroke:#f57c00
    classDef ui fill:#f3e5f5,stroke:#7b1fa2
    classDef infra fill:#e8f5e9,stroke:#388e3c

    subgraph entry[Entry]
        cli[cli.py<br/>Click commands]:::ui
    end

    subgraph corelayer[Core]
        config[config.py]:::core
        system[system_info.py]:::core
        registry[model_registry.py]:::core
        router[router.py]:::core
        thinking[thinking.py]:::core
        quant[quantizer.py]:::core
        kvq[kv_quantizer.py]:::core
        perf[performance.py]:::core
        engine[engine.py<br/>ABC]:::core
        llama[llama_backend.py]:::backend
        vllm[vllm_backend.py]:::backend
        vllm_cfg[vllm_config.py]:::backend
        server[server.py]:::core
        client[server_client.py]:::backend
        mp[multiprocess.py]:::core
        handoff[handoff.py]:::core
        security[security.py]:::infra
        unres[unrestricted.py]:::infra
    end

    subgraph uilayer[UI]
        console[ui/console.py]:::ui
        interactive[ui/interactive.py]:::ui
    end

    subgraph skills[Skills]
        loader[skills/loader.py]:::core
        base[skills/base.py]:::core
        builtin[skills/builtin/*]:::core
    end

    cli --> config
    cli --> system
    cli --> registry
    cli --> router
    cli --> thinking
    cli --> quant
    cli --> kvq
    cli --> engine
    cli --> server
    cli --> mp
    cli --> handoff
    cli --> security
    cli --> unres
    cli --> loader
    cli --> interactive
    interactive --> console
    interactive --> engine
    interactive --> perf
    router --> engine
    engine --> llama
    engine --> vllm
    engine --> client
    vllm --> vllm_cfg
    mp --> server
    mp --> client
    loader --> base
    base --> builtin
```

## Runtime boundaries

- **Startup fast path:** `cli.py` imports are lazy; `system info` and
  `model list` avoid importing llama-cpp or vllm.
- **Inference hot path:** `InteractiveSession` → `InferenceEngine.chat_stream`
  → backend → tokens streamed to `ui/console.py`.
- **Async boundaries:** none. tqCLI is synchronous. Server mode uses
  blocking HTTP via `requests` and streaming via SSE iterator.

## Data flow — single-process chat

```mermaid
sequenceDiagram
    participant U as User
    participant I as InteractiveSession
    participant R as Router
    participant M as ModelRegistry
    participant E as InferenceEngine
    participant P as PerformanceMonitor
    participant C as Rich Console

    U->>I: prompt
    I->>R: classify(prompt)
    R-->>I: TaskDomain + score
    I->>M: rank(domain, hardware)
    M-->>I: best_model
    alt best_model != loaded_model
        I->>E: unload_model()
        I->>E: load_model(best_model.path)
    end
    I->>E: chat_stream(messages, images?, audio?)
    loop per token chunk
        E-->>C: chunk
    end
    E-->>I: final stats
    I->>P: record(tokens, elapsed)
    P-->>I: tok/s, threshold status
    I->>C: render_stats
    opt below threshold
        I->>I: offer handoff
    end
```

## Data flow — multi-process chat

```mermaid
sequenceDiagram
    participant Op as Operator
    participant MP as MultiProcessCoordinator
    participant SRV as InferenceServer
    participant W1 as Worker 1
    participant W2 as Worker 2
    participant ENG as In-server Engine

    Op->>MP: tqcli serve start -m MODEL
    MP->>SRV: spawn (llama.cpp or vLLM OpenAI-compatible)
    SRV->>ENG: load_model
    Note over SRV,ENG: Server holds the model;<br/>workers hold only ChatHistory.

    Op->>W1: tqcli chat --engine server
    Op->>W2: tqcli chat --engine server
    par Worker 1
        W1->>SRV: POST /v1/chat/completions (SSE)
        SRV-->>W1: tokens
    and Worker 2
        W2->>SRV: POST /v1/chat/completions (SSE)
        SRV-->>W2: tokens
    end
    Note over SRV: vLLM: continuous batching<br/>llama.cpp: sequential queue

    Op->>MP: tqcli serve stop
    MP->>SRV: SIGTERM
```

## Configuration surface

All runtime knobs come from `~/.tqcli/config.yaml` loaded by
`tqcli/config.py` into a `TqConfig` dataclass. Nested groups:

- `performance` — tok/s thresholds + handoff triggers
- `security` — venv, sandbox, audit log, resource percentages
- `router` — router enable + preferences
- `multiprocess` — host/port, max_workers, auto_start_server

CLI flags override config at invocation time
(e.g. `tqcli --stop-trying-to-control-everything-and-just-let-go …`).

## Versioning

`tqcli/__init__.py` resolves `__version__` from installed package metadata
via `importlib.metadata.version("tqcli")`, falling back to a hard-coded
string for source-of-truth development. `pyproject.toml` is the single
source of truth for the published version.

```mermaid
flowchart LR
    pyproject[pyproject.toml<br/>version = 0.5.0] -->|setuptools| meta[.dist-info METADATA]
    meta -->|importlib.metadata.version| init[tqcli/__init__.py<br/>__version__]
    init --> cli[cli.py<br/>click.version_option]
    init --> banner[ui/console.py<br/>banner]
```
