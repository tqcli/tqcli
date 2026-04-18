"""Headless-chat unit tests using a fake inference engine.

Keeps CI fast — the heavy model-backed paths are exercised in
integration_lifecycle.py and test_integration_turboquant_kv.py.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Generator

import pytest

from tqcli.config import TqConfig
from tqcli.core.engine import (
    ChatMessage,
    CompletionResult,
    InferenceEngine,
    InferenceStats,
)
from tqcli.core.performance import PerformanceMonitor
from tqcli.ui.interactive import InteractiveSession


class FakeEngine(InferenceEngine):
    """Minimal engine that echoes a canned reply and records what it got."""

    def __init__(self, reply: str = "hi from fake engine"):
        self._reply = reply
        self._loaded = False
        self.received: list[list[ChatMessage]] = []

    @property
    def engine_name(self) -> str:
        return "fake"

    @property
    def is_available(self) -> bool:
        return True

    def load_model(self, model_path: str, **kwargs) -> None:
        self._loaded = True

    def unload_model(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _mk_stats(self, text: str) -> InferenceStats:
        return InferenceStats(
            prompt_tokens=5,
            completion_tokens=len(text),
            total_tokens=5 + len(text),
            completion_time_s=0.1,
            tokens_per_second=float(len(text)) / 0.1,
            total_time_s=0.1,
        )

    def chat(self, messages, **kwargs) -> CompletionResult:
        self.received.append(list(messages))
        return CompletionResult(
            text=self._reply,
            stats=self._mk_stats(self._reply),
            model_id="fake",
        )

    def chat_stream(
        self, messages, **kwargs
    ) -> Generator[tuple[str, InferenceStats | None], None, None]:
        self.received.append(list(messages))
        yield self._reply, None
        yield "", self._mk_stats(self._reply)

    def complete(self, prompt: str, **kwargs) -> CompletionResult:
        return CompletionResult(
            text=self._reply,
            stats=self._mk_stats(self._reply),
            model_id="fake",
        )


def _session(reply="42"):
    cfg = TqConfig()
    eng = FakeEngine(reply=reply)
    mon = PerformanceMonitor(cfg.performance)
    return InteractiveSession(cfg, eng, None, mon, model_family=""), eng


def test_headless_chat_turn_records_response_and_stats():
    sess, eng = _session(reply="42")
    out = sess.chat_turn("what is 6 times 7?", show_ui=False)
    assert out == "42"
    assert sess.last_response == "42"
    assert sess.last_stats is not None
    assert sess.last_stats.completion_tokens == 2
    assert len(eng.received) == 1


def test_headless_chat_turn_passes_images():
    sess, eng = _session(reply="a red square")
    sess.chat_turn("describe", images=["/tmp/x.png"], show_ui=False)
    msgs = eng.received[0]
    # Last user message is the one we just appended
    user = [m for m in msgs if m.role == "user"][-1]
    assert user.images == ["/tmp/x.png"]


def test_headless_chat_cli_help_advertises_flags():
    """The chat --help output must list the new headless + multimodal flags."""
    result = subprocess.run(
        [sys.executable, "-m", "tqcli", "chat", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--prompt" in result.stdout
    assert "--image" in result.stdout
    assert "--audio" in result.stdout
    assert "--messages" in result.stdout
    assert "--json" in result.stdout
    assert "--max-tokens" in result.stdout
