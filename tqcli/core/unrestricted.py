"""stop-trying-to-control-everything-and-just-let-go mode.

Equivalent to Claude Code's --dangerously-skip-permissions and
Gemini CLI's --yolo.

When active:
- Resource guard checks are bypassed (memory, GPU limits)
- Confirmation prompts are suppressed (model remove, etc.)
- Multi-process spawns without resource feasibility checks
- Auto-handoff is disabled (you chose to run locally, so run locally)
- Security audit warnings are noted but don't block operations

What it does NOT bypass:
- Audit logging (always on — you need the forensic trail)
- Model download integrity (still verifies files)
- Network binding (server still binds to localhost only)

This flag exists because experienced users with unusual hardware
configurations (NVLink multi-GPU, 512GB RAM servers, custom CUDA
builds) know their system better than our heuristics do. The resource
guards are conservative by design — this lets you override them.
"""

from __future__ import annotations

_UNRESTRICTED_WARNING = """
┌─────────────────────────────────────────────────────────────────┐
│  stop-trying-to-control-everything-and-just-let-go mode ACTIVE │
│                                                                 │
│  Resource guards:     BYPASSED                                  │
│  Confirmation prompts: SUPPRESSED                               │
│  Auto-handoff:         DISABLED                                 │
│  Audit logging:        STILL ON (always)                        │
│                                                                 │
│  You are responsible for monitoring system resources.            │
│  If the system becomes unresponsive, kill the tqcli process.    │
│                                                                 │
│  This is the equivalent of:                                     │
│    Claude Code:  --dangerously-skip-permissions                 │
│    Gemini CLI:   --yolo                                         │
└─────────────────────────────────────────────────────────────────┘
"""

_UNRESTRICTED_SHORT = "[bold red]UNRESTRICTED[/bold red]"


def show_unrestricted_warning(console) -> None:
    """Display the unrestricted mode warning banner."""
    console.print(_UNRESTRICTED_WARNING, style="red")


def is_unrestricted(ctx_or_config) -> bool:
    """Check if unrestricted mode is active from a Click context or config."""
    if hasattr(ctx_or_config, "obj") and isinstance(ctx_or_config.obj, dict):
        return ctx_or_config.obj.get("unrestricted", False)
    if hasattr(ctx_or_config, "unrestricted"):
        return ctx_or_config.unrestricted
    return False
