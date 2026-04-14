"""Handoff system — generates prompt files to transfer work to frontier model CLIs
when local inference performance is too slow."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

from tqcli.core.performance import PerformanceMonitor


def generate_handoff(
    monitor: PerformanceMonitor,
    conversation_history: list[dict],
    task_description: str,
    output_dir: Path,
    target_cli: str = "auto",
) -> Path:
    """Generate a handoff markdown file with context for a frontier model CLI.

    Args:
        monitor: Performance monitor with current stats.
        conversation_history: List of {"role": ..., "content": ...} messages.
        task_description: What the user was trying to accomplish.
        output_dir: Directory to write the handoff file.
        target_cli: Which CLI to target ("claude-code", "gemini-cli", "auto").

    Returns:
        Path to the generated handoff file.
    """
    stats = monitor.get_stats_display()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pick target
    if target_cli == "auto":
        target_cli = "claude-code"  # default recommendation

    filename = f"tqcli_handoff_{timestamp}.md"
    filepath = output_dir / filename

    # Build conversation context (last 10 messages)
    recent = conversation_history[-10:]
    conv_section = ""
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        conv_section += f"**{role}:** {content}\n\n"

    cli_instructions = _get_cli_instructions(target_cli)

    content = dedent(f"""\
        ---
        title: tqCLI Handoff — Task Transfer to Frontier Model
        generated_by: tqCLI v0.1.0
        generated_at: {datetime.now(timezone.utc).isoformat()}
        reason: Local inference performance below threshold
        target_cli: {target_cli}
        ---

        # tqCLI Performance Handoff

        Local inference performance has dropped below the acceptable threshold.
        This file contains the context needed to continue this task using a
        frontier model via **{target_cli}**.

        ## Performance Stats at Handoff

        | Metric | Value |
        |--------|-------|
        | Current tok/s | {stats['current_tps']} |
        | Rolling avg tok/s | {stats['rolling_tps']} |
        | Session avg tok/s | {stats['average_tps']} |
        | Threshold tok/s | {stats['threshold_tps']} |
        | Slow inference ratio | {stats['slow_ratio']}% |
        | Total tokens generated | {stats['session_tokens']} |

        ## Task Description

        {task_description}

        ## Conversation Context

        {conv_section}

        ## How to Continue

        {cli_instructions}

        ## Raw Context (for programmatic use)

        ```json
        {json.dumps({"task": task_description, "history": recent, "stats": stats}, indent=2)}
        ```
    """)

    filepath.write_text(content)
    return filepath


def _get_cli_instructions(target: str) -> str:
    instructions = {
        "claude-code": dedent("""\
            ### Using Claude Code

            1. Open a terminal in the project directory.
            2. Run: `claude`
            3. Paste or reference this handoff file:
               ```
               @tqcli_handoff_<timestamp>.md Continue this task.
               ```
            4. Claude Code will pick up the context and continue where tqCLI left off.
        """),
        "gemini-cli": dedent("""\
            ### Using Gemini CLI

            1. Open a terminal in the project directory.
            2. Run: `gemini`
            3. Provide the context from this handoff file.
            4. Reference the task description and conversation history above.
        """),
        "aider": dedent("""\
            ### Using Aider

            1. Open a terminal in the project directory.
            2. Run: `aider --message-file tqcli_handoff_<timestamp>.md`
            3. Aider will use the handoff context to continue the task.
        """),
        "openai": dedent("""\
            ### Using OpenAI CLI / ChatGPT

            1. Copy the task description and conversation context above.
            2. Paste into ChatGPT or the OpenAI API.
            3. Ask it to continue the task.
        """),
    }
    return instructions.get(target, instructions["claude-code"])
