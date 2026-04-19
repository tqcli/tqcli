"""Validate the default TurboQuant KV calibration corpus.

These tests enforce the invariants that keep the calibration corpus
statistically powered for activation-variance estimation. All assertions
derive from real tokenization / file presence — no hardcoded pass.
"""

from __future__ import annotations

import unittest

from tqcli.core.kv_metadata_generator import (
    DEFAULT_CALIBRATION_PROMPTS,
    MIN_OBSERVED_TOKENS,
)


class TestCalibrationCorpus(unittest.TestCase):
    def test_corpus_has_thirty_prompts(self) -> None:
        self.assertEqual(
            len(DEFAULT_CALIBRATION_PROMPTS),
            30,
            f"Expected 30 prompts; got {len(DEFAULT_CALIBRATION_PROMPTS)}.",
        )

    def test_no_duplicates(self) -> None:
        self.assertEqual(
            len(set(DEFAULT_CALIBRATION_PROMPTS)),
            len(DEFAULT_CALIBRATION_PROMPTS),
            "Calibration corpus must not contain duplicate prompts.",
        )

    def test_min_prompt_length(self) -> None:
        short = [
            (i, p[:60])
            for i, p in enumerate(DEFAULT_CALIBRATION_PROMPTS)
            if len(p.split()) < 60
        ]
        self.assertEqual(
            short,
            [],
            f"All corpus prompts should be paragraph-length (>= 60 words). "
            f"Too-short prompts: {short}",
        )

    def test_corpus_tokenized_size_meets_threshold_qwen3(self) -> None:
        """With Qwen3 tokenizer (dense vocab), the corpus must exceed the minimum.

        Skips gracefully if the Qwen3 model dir isn't present, so this test
        stays useful in minimal CI environments. When a Qwen3 dir exists we
        enforce the threshold end-to-end.
        """
        from pathlib import Path

        candidate_dirs = [
            Path.home() / ".tqcli/models/qwen3-4b-vllm",
            Path.home() / ".tqcli/models/qwen3-4b-AWQ",
        ]
        model_dir = next((d for d in candidate_dirs if (d / "tokenizer.json").is_file()), None)
        if model_dir is None:
            self.skipTest(
                "No local Qwen3 tokenizer found; run `tqcli model pull qwen3-4b-vllm` first."
            )

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

        # Emulate the generator's tokenization: truncate at max_seq_len=1024.
        observed = 0
        per_prompt: list[int] = []
        for prompt in DEFAULT_CALIBRATION_PROMPTS:
            ids = tokenizer(prompt, truncation=True, max_length=1024)["input_ids"]
            per_prompt.append(len(ids))
            observed += len(ids)

        avg = observed / len(DEFAULT_CALIBRATION_PROMPTS)
        self.assertGreaterEqual(
            observed,
            MIN_OBSERVED_TOKENS,
            f"Corpus tokenized size {observed} < MIN_OBSERVED_TOKENS "
            f"{MIN_OBSERVED_TOKENS}. Per-prompt: {per_prompt} (avg {avg:.1f}).",
        )
        # Guard against one very long prompt carrying the total.
        self.assertGreater(
            min(per_prompt),
            50,
            f"Shortest prompt tokenizes to only {min(per_prompt)} tokens; "
            f"corpus diversity is broken. Per-prompt: {per_prompt}.",
        )


if __name__ == "__main__":
    unittest.main()
