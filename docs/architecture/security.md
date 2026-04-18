# Security Layer

Everything under `tqcli/core/security.py` + `tqcli/core/unrestricted.py`.
Inspired by Claude Code's permission model — **sandbox by default, allow
explicit override for experienced users**.

## Components

```mermaid
classDiagram
    class SecurityManager {
        -env: EnvironmentDetector
        -guard: ResourceGuard
        -venv: VenvManager
        -audit: AuditLogger
        +audit_log(event, **kwargs)
        +check_load_allowed(model_path) bool
        +ensure_venv() Path
    }

    class EnvironmentDetector {
        +is_wsl: bool
        +is_container: bool
        +is_venv: bool
        +is_bare_metal: bool
        +classify() str
    }

    class VenvManager {
        +venv_path: Path
        +exists() bool
        +create()
        +python_executable() Path
    }

    class ResourceGuard {
        +max_memory_percent: float
        +max_gpu_memory_percent: float
        +check_ram(needed_mb) bool
        +check_vram(needed_mb) bool
    }

    class AuditLogger {
        +log_path: Path
        +append(event: dict)
        +rotate_if_needed()
    }

    SecurityManager *-- EnvironmentDetector
    SecurityManager *-- VenvManager
    SecurityManager *-- ResourceGuard
    SecurityManager *-- AuditLogger
```

## Audit log

Every model load, download, skill execution, server start/stop, and
unrestricted-mode invocation is appended as a JSON object to
`~/.tqcli/audit.log` (JSON Lines). The log is append-only — tqCLI never
edits or truncates historical records. Tooling can tail it for SIEM
integration.

```mermaid
sequenceDiagram
    participant CLI as CLI command
    participant SM as SecurityManager
    participant RG as ResourceGuard
    participant AL as AuditLogger
    participant FS as audit.log

    CLI->>SM: audit_log("model.load", model_id=...)
    SM->>RG: check_vram(model_size_mb)
    RG-->>SM: allowed / denied
    alt allowed
        SM->>AL: append({ts, event, user, model_id, outcome: "allowed"})
        SM-->>CLI: True
    else denied
        SM->>AL: append({ts, event, outcome: "denied", reason})
        SM-->>CLI: False (CLI aborts)
    end
    AL->>FS: atomic write
```

## Resource guards

Hard thresholds from `~/.tqcli/config.yaml::security`:

- `max_memory_percent` — default 80%; prevents OOM by refusing to load a
  model whose peak estimated RAM exceeds this fraction.
- `max_gpu_memory_percent` — default 90%; same concept for VRAM.
- `sandbox_enabled` — if True, blocks loading models outside
  `~/.tqcli/models/`.
- `use_venv` — if True, `SecurityManager.ensure_venv()` creates
  `~/.tqcli/venv` on first run.

Resource guards kick in before heavy IO starts. The integration tests
use them via the normal CLI — no special test hooks.

## Unrestricted mode

The `--stop-trying-to-control-everything-and-just-let-go` flag (aka
"yolo mode") bypasses:

- VRAM / RAM feasibility checks
- Confirmation prompts
- Worker-count caps

It does **not** bypass:

- Audit logging (every unrestricted invocation is still recorded)
- Download integrity checks
- Unsafe path / file permission checks

```mermaid
flowchart LR
    cli[CLI invocation] --> opt{--stop-trying...?}
    opt -- yes --> unrestricted[UnrestrictedContext<br/>pass-through]
    opt -- no --> guarded[Guarded path<br/>ResourceGuard.check_*]
    unrestricted --> audit1[AuditLogger.append unrestricted=True]
    guarded --> audit2[AuditLogger.append unrestricted=False]
    audit1 --> run[Command body]
    audit2 --> run
```

Equivalent to Claude Code's `--dangerously-skip-permissions` and Gemini
CLI's `--yolo`.

## Environment detection

`EnvironmentDetector` classifies the host as `wsl2`, `container` (Docker,
Kubernetes), `venv` (pip-installed inside an activated venv), or
`bare-metal`. This drives:

- **WSL2 quirks** — vLLM uses `spawn` multiprocess start method (NVML is
  not fork-compatible on WSL2); `pin_memory=False` to avoid slowdowns.
- **Container checks** — if no cgroup limits are set, resource guards
  downgrade to RAM-visible estimates instead of container-limited ones.
- **Recommended engine** — WSL2 + NVIDIA → vLLM; macOS Apple Silicon →
  llama.cpp with Metal.

## Running a security audit

```bash
tqcli security audit              # read-only report
tqcli security audit --fix        # auto-fix safe issues (create venv, etc.)
```

Covered by the `tq-security-audit` skill and integration tests — see
`.claude/skills/tq-security-audit/`.
