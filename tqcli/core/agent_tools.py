"""Agent-facing tools for AI Tinkering / Unrestricted (ReAct) modes.

Each tool exposes:
- `name` — LLM-visible identifier.
- `description` — natural-language hint used in the injected tool schema.
- `safety` — "safe" (read-only / user-gated) or "actionable" (side-effects).
- `arg_schema` — OpenAI/Anthropic-style JSON Schema describing parameters.
- `execute(args: dict) -> str` — synchronous handler returning stdout.

Tools are deliberately small and side-effect-scoped. Destructive operations
(file write, shell exec) are marked `actionable` so the orchestrator can route
them through the Shared Determinism confirmation prompt in ai_tinkering mode.
"""

from __future__ import annotations

import getpass
import os
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path


class AgentTool(ABC):
    """Base class for an LLM-callable tool in agentic modes."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def safety(self) -> str:
        """'safe' or 'actionable'."""

    @property
    @abstractmethod
    def arg_schema(self) -> dict: ...

    @abstractmethod
    def execute(self, args: dict) -> str: ...

    def to_tool_schema(self) -> dict:
        """Emit OpenAI-compatible tool definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.arg_schema,
            },
        }


class FileReadTool(AgentTool):
    name = "tq-file-read"
    description = (
        "Read a UTF-8 text file from disk and return its contents. "
        "Use for inspecting source, config, or log files."
    )
    # Marked actionable in ai_tinkering mode because a read can exfiltrate
    # secrets (~/.ssh, .env, /etc/shadow) even though it does not mutate state.
    # Unrestricted mode still auto-executes; the user opted in with --yolo.
    safety = "actionable"
    arg_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Absolute or cwd-relative file path."},
            "max_bytes": {
                "type": "integer",
                "description": "Optional cap on bytes read (default 200000).",
                "default": 200000,
            },
        },
        "required": ["path"],
    }

    def execute(self, args: dict) -> str:
        path = Path(args["path"]).expanduser()
        if not path.is_file():
            return f"ERROR: not a file: {path}"
        max_bytes = int(args.get("max_bytes") or 200000)
        try:
            data = path.read_bytes()[:max_bytes]
            return data.decode("utf-8", errors="replace")
        except OSError as e:
            return f"ERROR: {e}"


class FileWriteTool(AgentTool):
    name = "tq-file-write"
    description = (
        "Write (or overwrite) a UTF-8 text file at the given path. "
        "Creates parent directories as needed. Actionable — gated in ai_tinkering mode."
    )
    safety = "actionable"
    arg_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Destination file path."},
            "content": {"type": "string", "description": "Full file contents to write."},
        },
        "required": ["path", "content"],
    }

    def execute(self, args: dict) -> str:
        path = Path(args["path"]).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"], encoding="utf-8")
        return f"OK: wrote {len(args['content'])} chars to {path}"


class TerminalExecTool(AgentTool):
    name = "tq-terminal-exec"
    description = (
        "Execute a shell command via /bin/sh and capture combined stdout+stderr. "
        "Actionable — gated in ai_tinkering mode. Timeout 120s, output capped."
    )
    safety = "actionable"
    arg_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command line to run."},
            "cwd": {"type": "string", "description": "Working directory (optional)."},
            "timeout_s": {"type": "integer", "description": "Timeout seconds (default 120)."},
        },
        "required": ["command"],
    }

    def execute(self, args: dict) -> str:
        cmd = args["command"]
        cwd = args.get("cwd") or None
        timeout_s = int(args.get("timeout_s") or 120)
        try:
            r = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return f"ERROR: timeout after {timeout_s}s"
        out = (r.stdout or "") + (r.stderr or "")
        if len(out) > 4000:
            out = out[:1000] + f"\n...[truncated {len(out) - 2000} chars]...\n" + out[-1000:]
        return f"exit={r.returncode}\n{out}"


class InteractivePromptTool(AgentTool):
    name = "tq-interactive-prompt"
    description = (
        "Pause the agent loop and ask the user for input (e.g., a password, API "
        "token, or a yes/no confirmation). Returns the user's reply as a string. "
        "For secret_mode=true, input is read without echo."
    )
    safety = "safe"
    arg_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Prompt shown to the user."},
            "secret_mode": {
                "type": "boolean",
                "description": "If true, read without echoing input (passwords/tokens).",
                "default": False,
            },
        },
        "required": ["question"],
    }

    def execute(self, args: dict) -> str:
        question = args["question"]
        prompt = f"[agent asks] {question}\n> "
        if args.get("secret_mode"):
            return getpass.getpass(prompt)
        # Write to stderr so it does not pollute any JSON stdout stream.
        sys.stderr.write(prompt)
        sys.stderr.flush()
        return sys.stdin.readline().rstrip("\n")


def default_tools() -> list[AgentTool]:
    return [FileReadTool(), FileWriteTool(), TerminalExecTool(), InteractivePromptTool()]


def build_tool_registry(tools: list[AgentTool] | None = None) -> dict[str, AgentTool]:
    tools = tools if tools is not None else default_tools()
    return {t.name: t for t in tools}


def is_safe_tool(name: str, registry: dict[str, AgentTool]) -> bool:
    t = registry.get(name)
    return bool(t and t.safety == "safe")
