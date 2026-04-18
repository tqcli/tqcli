# Multi-Process Mode

tqCLI supports a single shared inference server + N worker processes. The
server holds the model; workers keep only the conversation history and
stream tokens over HTTP (SSE).

## Components

```mermaid
graph TB
    subgraph serverbox[Server process]
        serverpy[server.py<br/>manages subprocess PID + health]
        subproc[llama.cpp OR vllm<br/>OpenAI-compatible HTTP]
        model[Loaded model<br/>in memory]
    end

    subgraph workers[Worker processes]
        w1[Worker 1<br/>tqcli chat --engine server]
        w2[Worker 2]
        w3[Worker 3]
    end

    subgraph coord[Coordinator]
        mp[multiprocess.py<br/>MultiProcessCoordinator]
        assess[assess_multiprocess<br/>feasibility analysis]
    end

    mp --> serverpy
    mp --> w1
    mp --> w2
    mp --> w3
    serverpy --> subproc
    subproc --> model
    w1 -->|HTTP + SSE| subproc
    w2 -->|HTTP + SSE| subproc
    w3 -->|HTTP + SSE| subproc
    assess --> mp
```

## Engine-specific concurrency

| Engine | Concurrency model | Throughput shape |
|--------|-------------------|------------------|
| llama.cpp | Sequential queue (server processes one request at a time) | Saturates at ~1 request in flight |
| vLLM | Continuous batching + PagedAttention | Multiple requests share GPU, throughput rises with concurrency up to KV budget |

This is why vLLM is the recommended server for commercial multi-tenant
workloads. The `--engine server` flag inside `tqcli chat` does not need
to know which server engine is running — it just speaks
OpenAI-compatible HTTP via `ServerClientBackend`.

## `assess_multiprocess`

`tqcli/core/multiprocess.py::assess_multiprocess` is pure Python — it
evaluates hardware feasibility for N workers and returns a plan object:

```mermaid
flowchart LR
    input[sys_info + model_size_mb +<br/>requested_workers + preferred_engine] --> vram{VRAM sufficient<br/>for model?}
    vram -- no --> infeasible[feasible=False]
    vram -- yes --> engine{preferred_engine}
    engine -- llama.cpp --> seq[max_workers = min requested, 4<br/>mode = sequential queue]
    engine -- vllm --> vllm_branch{VRAM - model >= KV budget × workers?}
    vllm_branch -- yes --> batched[max_workers = requested<br/>mode = continuous batching]
    vllm_branch -- no --> reduced[max_workers = floor available VRAM / per-worker KV<br/>mode = continuous batching]
    seq --> out[Return AssessmentPlan]
    batched --> out
    reduced --> out
    infeasible --> out
```

Unrestricted mode (`--stop-trying-to-control-everything-and-just-let-go`)
passes `unrestricted=True` and skips the VRAM feasibility check — the
coordinator still reports its findings, but will not refuse to start.

## Lifecycle commands

| Command | What it does |
|---------|--------------|
| `tqcli serve start -m <model-id> [-e llama.cpp|vllm]` | Start the server subprocess (port 8741 by default) |
| `tqcli serve status` | Report PID, engine, health, uptime |
| `tqcli serve stop` | Send SIGTERM to the server subprocess and wait for clean exit |
| `tqcli workers spawn 3` | Spawn three worker processes pointing at the server |
| `tqcli chat --engine server` | Interactive chat as a single worker |

All of these are verified-available in the integration test lifecycle
(`tests/integration_lifecycle.py::step_serve_lifecycle` — run
`TQCLI_TEST_SERVER=1` to exercise the real start/stop cycle).

## Server = OpenAI-compatible endpoint

Both `llama-cpp-python[server]` and vLLM's OpenAI API server expose
`/v1/chat/completions`. `ServerClientBackend` is a thin wrapper over
`requests`:

```mermaid
sequenceDiagram
    participant W as Worker
    participant SC as ServerClientBackend
    participant HTTP as HTTP transport
    participant S as Server

    W->>SC: chat_stream(messages)
    SC->>HTTP: POST /v1/chat/completions<br/>stream=True
    HTTP->>S: request
    loop SSE chunks
        S-->>HTTP: data: {delta...}
        HTTP-->>SC: bytes
        SC-->>W: ChatCompletionChunk
    end
    SC-->>W: final InferenceStats
```

## Failure modes

- **Server not running** → `tqcli serve status` reports clearly; workers
  raise on first request with a human-readable error.
- **Server crashed mid-stream** → worker receives an incomplete SSE
  stream; `PerformanceMonitor` flags the stats as incomplete; handoff
  offered.
- **Port conflict** → `tqcli serve start` refuses to start; pass
  `--port 8742` to override.
