"""Phase 4 rubrics for the agent orchestrator.

Covers:
1. Tinkering Enforcement — a 'n' denial must NOT execute the tool and must
   surface the denial to the LLM as an Observation.
2. Yolo Unrestricted Loop — a <tool_call> fires immediately, the observation
   is appended as a user turn, and engine.chat_stream is invoked a second time
   without human IO.
3. Manual Default Silence — the orchestrator's injected_tool_schemas is an
   empty list in manual mode (i.e., no tools leaked into the system prompt).
"""

from __future__ import annotations

from typing import Iterator

import pytest

from tqcli.core.agent_orchestrator import (
    MODE_MANUAL,
    MODE_TINKERING,
    MODE_UNRESTRICTED,
    AgentOrchestrator,
    OrchestratorConfig,
    parse_tool_calls,
)
from tqcli.core.agent_tools import AgentTool, default_tools
from tqcli.core.engine import ChatMessage, InferenceEngine, InferenceStats


class ScriptedEngine(InferenceEngine):
    """Deterministic engine that replays pre-scripted responses turn by turn."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.call_count = 0

    @property
    def engine_name(self) -> str:
        return "scripted"

    @property
    def is_available(self) -> bool:
        return True

    @property
    def is_loaded(self) -> bool:
        return True

    def load_model(self, model_path: str, **_) -> None: ...
    def unload_model(self) -> None: ...

    def chat(self, messages, **_):
        raise NotImplementedError

    def complete(self, prompt: str, **_):
        raise NotImplementedError

    def chat_stream(self, messages, **_) -> Iterator[tuple[str, InferenceStats | None]]:
        self.call_count += 1
        reply = self._responses.pop(0) if self._responses else ""
        yield reply, None
        yield "", InferenceStats(
            prompt_tokens=1, completion_tokens=1, total_tokens=2, total_time_s=0.01
        )


class SpyTool(AgentTool):
    """Captures calls so tests can assert execution vs. skip."""

    def __init__(self, name="tq-terminal-exec", safety="actionable"):
        self._name = name
        self._safety = safety
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "spy"

    @property
    def safety(self) -> str:
        return self._safety

    @property
    def arg_schema(self) -> dict:
        return {"type": "object", "properties": {"command": {"type": "string"}}}

    def execute(self, args: dict) -> str:
        self.calls.append(args)
        return f"ran:{args.get('command', '')}"


def test_manual_mode_injects_no_tools():
    """Rubric #3: manual mode must not inject any tool schemas."""
    cfg = OrchestratorConfig(mode=MODE_MANUAL, tools=default_tools())
    orch = AgentOrchestrator(engine=ScriptedEngine(["hi"]), config=cfg)
    assert orch.injected_tool_schemas == []


def test_tinkering_denial_blocks_execution():
    """Rubric #1: providing 'n' to the confirm prompt must prevent execution."""
    spy = SpyTool()
    staged_reply = (
        '<staged_tool_call>{"name": "tq-terminal-exec", '
        '"arguments": {"command": "rm -rf /"}}</staged_tool_call>'
    )
    engine = ScriptedEngine([staged_reply])

    def deny_confirm(name, args, safety):
        return "n", args

    cfg = OrchestratorConfig(
        mode=MODE_TINKERING,
        tools=[spy],
        confirm_fn=deny_confirm,
    )
    orch = AgentOrchestrator(engine=engine, config=cfg)

    history = [ChatMessage(role="system", content="sys"),
               ChatMessage(role="user", content="do the thing")]
    _, updated = orch.run_turn(history)

    assert spy.calls == [], "tool must not execute after denial"
    assert any(
        "denied" in m.content.lower() for m in updated if m.role == "user"
    ), "LLM should see a denial observation"


def test_unrestricted_loop_executes_without_prompt():
    """Rubric #2: yolo mode auto-executes and re-enters engine.chat_stream."""
    spy = SpyTool()
    engine = ScriptedEngine([
        '<tool_call>{"name": "tq-terminal-exec", "arguments": {"command": "echo hi"}}</tool_call>',
        "Done.",
    ])
    cfg = OrchestratorConfig(
        mode=MODE_UNRESTRICTED,
        max_steps=5,
        tools=[spy],
    )
    orch = AgentOrchestrator(engine=engine, config=cfg)

    history = [ChatMessage(role="system", content="sys"),
               ChatMessage(role="user", content="go")]
    final, updated = orch.run_turn(history)

    assert spy.calls == [{"command": "echo hi"}], "tool should fire once"
    assert engine.call_count == 2, "engine must be re-invoked after tool output"
    assert final == "Done."
    # Observation was threaded back as a user role.
    assert any(
        m.role == "user" and m.content.startswith("Observation:")
        for m in updated
    )


def test_tinkering_safe_tool_auto_runs_without_prompt():
    """Safe-marked tools (e.g., file-read) should not require confirmation."""
    spy = SpyTool(name="tq-file-read", safety="safe")
    staged_reply = (
        '<staged_tool_call>{"name": "tq-file-read", '
        '"arguments": {"path": "/etc/hostname"}}</staged_tool_call>'
    )
    engine = ScriptedEngine([staged_reply])

    def should_never_be_called(*_a, **_kw):
        raise AssertionError("safe tool must not prompt")

    cfg = OrchestratorConfig(
        mode=MODE_TINKERING,
        tools=[spy],
        confirm_fn=should_never_be_called,
    )
    orch = AgentOrchestrator(engine=engine, config=cfg)

    history = [ChatMessage(role="system", content="sys"),
               ChatMessage(role="user", content="read it")]
    _, _ = orch.run_turn(history)

    assert spy.calls == [{"path": "/etc/hostname"}]


def test_parse_tool_calls_handles_both_tags():
    staged = '<staged_tool_call>{"name": "a", "arguments": {}}</staged_tool_call>'
    live = '<tool_call>{"name": "b", "arguments": {}}</tool_call>'
    parsed = parse_tool_calls(f"text {staged} more {live} end")
    assert [p.kind for p in parsed] == ["staged", "live"]
    assert [p.name for p in parsed] == ["a", "b"]


def test_parse_tool_calls_handles_nested_json_args():
    """Regression for gemini-flagged regex bug: nested {} in args must parse."""
    nested = (
        '<tool_call>{"name": "tq-file-write", '
        '"arguments": {"path": "/tmp/x", "meta": {"k": "v"}}}</tool_call>'
    )
    parsed = parse_tool_calls(nested)
    assert len(parsed) == 1
    assert parsed[0].name == "tq-file-write"
    assert parsed[0].args == {"path": "/tmp/x", "meta": {"k": "v"}}


def test_tinkering_chains_multiple_steps_after_approval():
    """Tinkering must loop so the model can act on its observation."""
    spy = SpyTool(name="tq-file-read", safety="safe")
    engine = ScriptedEngine([
        '<staged_tool_call>{"name": "tq-file-read", "arguments": {"path": "/a"}}</staged_tool_call>',
        '<staged_tool_call>{"name": "tq-file-read", "arguments": {"path": "/b"}}</staged_tool_call>',
        "All done.",
    ])
    cfg = OrchestratorConfig(mode=MODE_TINKERING, max_steps=5, tools=[spy])
    orch = AgentOrchestrator(engine=engine, config=cfg)
    history = [ChatMessage(role="user", content="read both")]
    final, _ = orch.run_turn(history)
    assert engine.call_count == 3
    assert [c["path"] for c in spy.calls] == ["/a", "/b"]
    assert final == "All done."


def test_unrestricted_honors_max_steps():
    """A buggy model that keeps emitting tool calls must be bounded."""
    spy = SpyTool()
    endless = '<tool_call>{"name": "tq-terminal-exec", "arguments": {"command": "x"}}</tool_call>'
    engine = ScriptedEngine([endless] * 20)
    cfg = OrchestratorConfig(mode=MODE_UNRESTRICTED, max_steps=3, tools=[spy])
    orch = AgentOrchestrator(engine=engine, config=cfg)
    history = [ChatMessage(role="user", content="go")]
    orch.run_turn(history)
    assert engine.call_count == 3
