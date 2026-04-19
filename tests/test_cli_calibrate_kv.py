"""CLI tests for `tqcli model calibrate-kv`.

All assertions derive from real CliRunner invocations; no hardcoded pass.
The expensive model-load-and-calibrate path is exercised separately in
`tests/test_integration_agent_functional.py` (already covered by the
auto-calibrate-on-load integration test). This file exercises everything
*around* that: help, unknown model, not-installed, wrong-engine refuse,
precondition refuse, existing-metadata-without-force skip.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from tqcli.cli import main


class TestCalibrateKvCli(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_help_exposes_command(self) -> None:
        result = self.runner.invoke(main, ["model", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("calibrate-kv", result.output)

    def test_calibrate_kv_help_lists_flags(self) -> None:
        result = self.runner.invoke(main, ["model", "calibrate-kv", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        for expected in ("--recipe", "--force", "MODEL_ID"):
            self.assertIn(expected, result.output, f"Missing in help: {expected}")

    def test_unknown_model_exits_nonzero(self) -> None:
        result = self.runner.invoke(
            main, ["model", "calibrate-kv", "___definitely_not_a_real_model___"]
        )
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("Unknown model", result.output)

    def test_not_installed_model_exits_nonzero(self) -> None:
        """A known-but-uninstalled vllm model should exit 2 with a helpful message."""
        # qwen3-8b-AWQ is in the registry but typically not installed on the
        # reference box. If someone installs it locally, the test becomes a
        # no-op skip rather than a false pass.
        awq_8b = Path.home() / ".tqcli/models/qwen3-8b-AWQ"
        if awq_8b.exists():
            self.skipTest(f"qwen3-8b-AWQ is installed at {awq_8b}; test assumes it is NOT.")
        result = self.runner.invoke(
            main, ["model", "calibrate-kv", "qwen3-8b-AWQ"]
        )
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertIn("not installed locally", result.output)

    def test_gguf_model_refused(self) -> None:
        """llama.cpp GGUF models should be refused (engine mismatch)."""
        result = self.runner.invoke(
            main, ["model", "calibrate-kv", "qwen3-4b-Q4_K_M"]
        )
        # If installed locally, exits 2 with engine message. If not installed,
        # exits 2 with not-installed message. Either is acceptable — both are
        # exit 2 refuses with an informative message.
        self.assertEqual(result.exit_code, 2, result.output)
        self.assertTrue(
            "not installed locally" in result.output
            or "vllm" in result.output.lower(),
            f"Expected engine-mismatch or not-installed message. Got: {result.output}",
        )

    def test_existing_metadata_without_force_skips(self) -> None:
        """Pre-existing turboquant_kv.json should cause a no-op exit 0 unless --force."""
        # Requires qwen3-4b-vllm to be installed with metadata present.
        metadata_path = Path.home() / ".tqcli/models/qwen3-4b-vllm/turboquant_kv.json"
        if not metadata_path.is_file():
            self.skipTest(f"Skipped: {metadata_path} not present on this host.")
        result = self.runner.invoke(
            main, ["model", "calibrate-kv", "qwen3-4b-vllm"]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("already exists", result.output)
        self.assertIn("--force", result.output)

    def test_recipe_flag_validated(self) -> None:
        """--recipe must be one of the allowed values."""
        result = self.runner.invoke(
            main,
            ["model", "calibrate-kv", "qwen3-4b-vllm", "--recipe", "bogus-recipe"],
        )
        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertIn("bogus-recipe", result.output)

    def test_awq_model_refused_on_preconditions(self) -> None:
        """AWQ source weights must be refused with a specific reason."""
        # Only runs if qwen3-4b-AWQ is installed locally.
        awq_dir = Path.home() / ".tqcli/models/qwen3-4b-AWQ"
        if not awq_dir.is_file() and not awq_dir.is_dir():
            self.skipTest("qwen3-4b-AWQ not installed; skipping precondition test.")
        # Delete any existing metadata to reach the precondition check.
        existing = awq_dir / "turboquant_kv.json"
        had_existing = existing.is_file()
        backup_text: str | None = None
        if had_existing:
            backup_text = existing.read_text()
            existing.unlink()
        try:
            result = self.runner.invoke(
                main, ["model", "calibrate-kv", "qwen3-4b-AWQ", "--force"]
            )
            self.assertEqual(result.exit_code, 3, result.output)
            self.assertIn("already quantized", result.output)
        finally:
            if backup_text is not None:
                existing.write_text(backup_text)


if __name__ == "__main__":
    unittest.main()
