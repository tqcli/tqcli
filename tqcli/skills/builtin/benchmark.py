"""Built-in skill: quick benchmark."""

from __future__ import annotations

from tqcli.skills.base import BuiltinSkill


class BenchmarkSkill(BuiltinSkill):
    @property
    def name(self) -> str:
        return "benchmark"

    @property
    def description(self) -> str:
        return "Run performance benchmarks on loaded models"

    def execute(self, args: list[str], context: dict) -> str:
        engine = context.get("engine")
        if not engine or not engine.is_loaded:
            return "No model loaded. Use 'tqcli model pull <id>' then 'tqcli chat' first."

        from tqcli.core.engine import ChatMessage
        import time

        prompts = [
            ("Short Generation", "What is 2+2? Answer in one word."),
            ("Code Generation", "Write a Python function to reverse a string."),
            ("Reasoning", "If all cats are animals and some animals are dogs, can we conclude cats are dogs? Explain."),
        ]

        lines = ["=== Quick Benchmark ===\n"]
        lines.append(f"{'Test':<25} {'Tok/s':>8} {'Tokens':>8} {'Time':>8}")
        lines.append("-" * 55)

        total_tps = 0.0
        for name, prompt in prompts:
            msg = ChatMessage(role="user", content=prompt)
            result = engine.chat([msg], max_tokens=128, temperature=0.1)
            tps = result.stats.tokens_per_second
            total_tps += tps
            lines.append(
                f"{name:<25} {tps:>8.1f} {result.stats.completion_tokens:>8} "
                f"{result.stats.total_time_s:>7.2f}s"
            )

        avg = total_tps / len(prompts)
        lines.append("-" * 55)
        lines.append(f"{'Average':<25} {avg:>8.1f}")
        return "\n".join(lines)
