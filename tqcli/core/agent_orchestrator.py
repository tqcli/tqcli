"""Agent orchestration middleware for tqCLI tri-state autonomy.

Modes
-----
- ``manual``:        no tool schema is injected; streamed output is returned as-is.
- ``ai_tinkering``:  tool schemas are injected. The LLM emits
                     ``<staged_tool_call>`` blocks. The orchestrator halts the
                     stream, presents the name + JSON args, and asks
                     ``[Y/n/Edit]``. On approval the tool runs and an
                     ``Observation:`` is fed back as a new user turn.
- ``unrestricted``:  tool schemas are injected. ``<tool_call>`` blocks execute
                     immediately, observations are threaded back into the
                     context, and the ReAct loop continues until the model
                     emits a plain-text reply (no tool tag) or ``max_steps``
                     is reached.

The orchestrator is transport-agnostic: it consumes the existing
``engine.chat_stream()`` generator and only looks at the text. It never
touches backend-specific internals (llama.cpp, vLLM, server).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from tqcli.core.agent_tools import AgentTool, default_tools, is_safe_tool
from tqcli.core.engine import ChatMessage, InferenceEngine

MODE_MANUAL = "manual"
MODE_TINKERING = "ai_tinkering"
MODE_UNRESTRICTED = "unrestricted"

# Non-greedy wildcard so nested JSON braces in arguments don't truncate the
# payload. The JSON parser validates structure.
_TAG_STAGED = re.compile(r"<staged_tool_call>\s*(.*?)\s*</staged_tool_call>", re.DOTALL)
_TAG_LIVE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

_OBS_TRUNCATE = 1000  # per TP Risk Assessment


@dataclass
class ToolInvocation:
    """Parsed tool request from the LLM stream."""
    raw: str
    name: str
    args: dict
    kind: str  # "staged" or "live"


@dataclass
class OrchestratorConfig:
    mode: str = MODE_MANUAL
    max_steps: int = 10
    # Prompt hook — allows tests to feed a deterministic answer.
    # Signature: (tool_name, args, tool_safety) -> "y" | "n" | "edit" + edited args.
    confirm_fn: Callable[[str, dict, str], tuple[str, dict]] | None = None
    tools: list[AgentTool] = field(default_factory=default_tools)


def parse_tool_calls(text: str) -> list[ToolInvocation]:
    """Extract every recognized tool-call tag from a block of text."""
    calls: list[ToolInvocation] = []
    for m in _TAG_STAGED.finditer(text):
        calls.append(_mk_invocation(m.group(0), m.group(1), "staged"))
    for m in _TAG_LIVE.finditer(text):
        calls.append(_mk_invocation(m.group(0), m.group(1), "live"))
    return calls


def _mk_invocation(raw: str, payload: str, kind: str) -> ToolInvocation:
    try:
        obj = json.loads(payload)
    except json.JSONDecodeError:
        return ToolInvocation(raw=raw, name="", args={}, kind=kind)
    if not isinstance(obj, dict):
        # Model emitted a list / string / number inside the tag — reject.
        return ToolInvocation(raw=raw, name="", args={}, kind=kind)
    args = obj.get("arguments") or obj.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    return ToolInvocation(
        raw=raw,
        name=str(obj.get("name", "")),
        args=args,
        kind=kind,
    )


def build_tool_system_prompt(tools: list[AgentTool], mode: str) -> str:
    """Inject tool schemas + protocol hint into the system prompt."""
    if mode == MODE_MANUAL:
        return ""
    tag = "staged_tool_call" if mode == MODE_TINKERING else "tool_call"
    schemas = [t.to_tool_schema() for t in tools]
    lines = [
        "You have access to the following tools. "
        f"To invoke a tool, emit EXACTLY one <{tag}> block containing JSON:",
        f'<{tag}>{{"name": "<tool>", "arguments": {{...}}}}</{tag}>',
        "Wait for an Observation: reply before emitting the next call. "
        "When the task is complete, reply with plain text and no tool tag.",
        "",
        "Available tools (JSON Schema):",
        json.dumps(schemas, indent=2),
    ]
    return "\n".join(lines)


def truncate_observation(text: str, limit: int = _OBS_TRUNCATE) -> str:
    """Keep head + tail. Errors usually land at the end, so head-only loses them."""
    if len(text) <= limit:
        return text
    half = max(limit // 2, 1)
    dropped = len(text) - 2 * half
    return text[:half] + f"\n...[truncated {dropped} chars]...\n" + text[-half:]


class AgentOrchestrator:
    """Runs the chat turn through manual / tinkering / ReAct pipelines."""

    def __init__(self, engine: InferenceEngine, config: OrchestratorConfig):
        self.engine = engine
        self.config = config
        self.tools: dict[str, AgentTool] = {t.name: t for t in config.tools}

    @property
    def injected_tool_schemas(self) -> list[dict]:
        """Empty list in manual mode — tests assert this invariant."""
        if self.config.mode == MODE_MANUAL:
            return []
        return [t.to_tool_schema() for t in self.config.tools]

    def run_turn(
        self,
        history: list[ChatMessage],
        max_tokens: int | None = None,
    ) -> tuple[str, list[ChatMessage]]:
        """Execute one user turn. Returns (final_visible_text, updated_history).

        Callers supply `history` with the final message being the user turn.
        On return the history has any assistant + observation turns appended.
        """
        mode = self.config.mode
        if mode == MODE_MANUAL:
            text = self._drain_stream(history, max_tokens)
            history.append(ChatMessage(role="assistant", content=text))
            return text, history

        if mode == MODE_TINKERING:
            return self._run_tinkering(history, max_tokens)

        if mode == MODE_UNRESTRICTED:
            return self._run_react(history, max_tokens)

        raise ValueError(f"unknown mode: {mode}")

    def _drain_stream(
        self, history: list[ChatMessage], max_tokens: int | None
    ) -> str:
        buf = ""
        for chunk, stats in self.engine.chat_stream(history, max_tokens=max_tokens):
            if stats:
                break
            buf += chunk
        return buf

    def _run_tinkering(
        self, history: list[ChatMessage], max_tokens: int | None
    ) -> tuple[str, list[ChatMessage]]:
        """Loop up to max_steps so the model can chain staged tool calls.

        Each iteration still gates actionable tools through confirm_fn, so the
        user stays in the loop — but once approved, the model sees the
        observation and can plan the next step without requiring a fresh
        user prompt.
        """
        last_text = ""
        for _step in range(self.config.max_steps):
            text = self._drain_stream(history, max_tokens)
            last_text = text
            history.append(ChatMessage(role="assistant", content=text))

            calls = [c for c in parse_tool_calls(text) if c.kind == "staged"]
            if not calls:
                return text, history

            any_denied = False
            for call in calls:
                observation = self._confirm_and_execute(call)
                if observation.startswith("User denied"):
                    any_denied = True
                history.append(
                    ChatMessage(role="user", content=f"Observation:\n{observation}")
                )
            if any_denied:
                # Stop auto-looping after a denial — return control to the user.
                return last_text, history
        return last_text, history

    def _run_react(
        self, history: list[ChatMessage], max_tokens: int | None
    ) -> tuple[str, list[ChatMessage]]:
        last_text = ""
        for step in range(self.config.max_steps):
            text = self._drain_stream(history, max_tokens)
            last_text = text
            history.append(ChatMessage(role="assistant", content=text))

            calls = [c for c in parse_tool_calls(text) if c.kind == "live"]
            if not calls:
                return text, history  # plain reply = task complete

            for call in calls:
                observation = self._execute(call)
                history.append(
                    ChatMessage(role="user", content=f"Observation:\n{observation}")
                )
        return last_text, history

    def _confirm_and_execute(self, call: ToolInvocation) -> str:
        if not call.name:
            return "ERROR: malformed tool call"
        tool = self.tools.get(call.name)
        if tool is None:
            return f"ERROR: unknown tool '{call.name}'"

        if is_safe_tool(call.name, self.tools):
            return truncate_observation(self._safe_execute(tool, call.args))

        choice, edited_args = self._ask_confirm(call)
        if choice == "n":
            return "User denied execution. Request alternatives."
        return truncate_observation(self._safe_execute(tool, edited_args))

    def _execute(self, call: ToolInvocation) -> str:
        if not call.name:
            return "ERROR: malformed tool call"
        tool = self.tools.get(call.name)
        if tool is None:
            return f"ERROR: unknown tool '{call.name}'"
        return truncate_observation(self._safe_execute(tool, call.args))

    @staticmethod
    def _safe_execute(tool: AgentTool, args: dict) -> str:
        try:
            return tool.execute(args)
        except Exception as e:  # surface failure to LLM, do not crash loop
            return f"ERROR: {type(e).__name__}: {e}"

    def _ask_confirm(self, call: ToolInvocation) -> tuple[str, dict]:
        if self.config.confirm_fn is not None:
            tool = self.tools.get(call.name)
            safety = tool.safety if tool else "actionable"
            return self.config.confirm_fn(call.name, call.args, safety)
        return _default_confirm(call)


def _default_confirm(call: ToolInvocation) -> tuple[str, dict]:
    """Interactive rich-prompt confirmation for tinkering mode."""
    from rich.prompt import Prompt

    from tqcli.ui.console import console

    console.print(
        f"[yellow]Agent wants to run [bold]{call.name}[/bold] with:[/yellow]"
    )
    console.print_json(data=call.args)
    choice = Prompt.ask("Proceed?", choices=["y", "n", "edit"], default="y")
    if choice != "edit":
        return choice, call.args

    # Minimal edit path: re-prompt the JSON payload as a single string.
    edited = Prompt.ask("Edit JSON arguments", default=json.dumps(call.args))
    try:
        return "y", json.loads(edited)
    except json.JSONDecodeError:
        console.print("[red]Invalid JSON, aborting tool call.[/red]")
        return "n", call.args


def make_orchestrator(
    engine: InferenceEngine,
    mode: str,
    max_steps: int = 10,
    confirm_fn: Callable | None = None,
) -> AgentOrchestrator:
    cfg = OrchestratorConfig(
        mode=mode,
        max_steps=max_steps,
        confirm_fn=confirm_fn,
        tools=default_tools(),
    )
    return AgentOrchestrator(engine=engine, config=cfg)
