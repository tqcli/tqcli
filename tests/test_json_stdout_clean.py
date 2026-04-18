"""Verify --json stdout is parseable by strict consumers (e.g. `jq -e`).

Issue #25 regression guard: tqCLI's own Rich console is already routed to
stderr under --json, but third-party library loggers (vllm, torch, tqdm)
were polluting stdout. These tests assert stdout is now parseable.
"""

from __future__ import annotations

import io
import json
import logging
import sys

import pytest

from tqcli.ui.console import setup_json_logging


def test_setup_json_logging_sets_env_vars(monkeypatch):
    monkeypatch.delenv("VLLM_CONFIGURE_LOGGING", raising=False)
    monkeypatch.delenv("VLLM_LOGGING_LEVEL", raising=False)
    setup_json_logging()
    import os
    assert os.environ["VLLM_CONFIGURE_LOGGING"] == "0"
    assert os.environ["VLLM_LOGGING_LEVEL"] == "ERROR"
    # TQDM_DISABLE is intentionally NOT set — vLLM reads tqdm.format_dict
    # which returns None when disabled, triggering ZeroDivisionError inside
    # vLLM. The file=sys.stderr monkey-patch is enough.


def test_setup_json_logging_reroutes_existing_vllm_loggers(caplog):
    # Pretend vllm already initialized a logger with a stdout handler.
    vllm_log = logging.getLogger("vllm.engine.test")
    vllm_log.addHandler(logging.StreamHandler(stream=sys.stdout))
    vllm_log.setLevel(logging.INFO)
    assert any(getattr(h, "stream", None) is sys.stdout for h in vllm_log.handlers)

    setup_json_logging()

    # After setup, existing handlers should be cleared off the vllm logger.
    vllm_log_after = logging.getLogger("vllm.engine.test")
    assert vllm_log_after.handlers == []
    assert vllm_log_after.propagate is True


def test_setup_json_logging_patches_tqdm_to_stderr():
    try:
        import tqdm
    except ImportError:
        pytest.skip("tqdm not installed")

    setup_json_logging()
    # After the patch, tqdm.tqdm should default to file=sys.stderr.
    # We instantiate it with an iterable and peek at the default kwargs.
    import functools

    assert isinstance(tqdm.tqdm, functools.partial)
    assert tqdm.tqdm.keywords.get("file") is sys.stderr
