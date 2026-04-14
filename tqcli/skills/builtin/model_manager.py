"""Built-in skill: model management."""

from __future__ import annotations

from pathlib import Path

from tqcli.core.model_registry import ModelRegistry
from tqcli.skills.base import BuiltinSkill


class ModelManagerSkill(BuiltinSkill):
    def __init__(self, models_dir: Path):
        self._models_dir = models_dir
        self._registry = ModelRegistry(models_dir)

    @property
    def name(self) -> str:
        return "model-manager"

    @property
    def description(self) -> str:
        return "List, download, and manage quantized models"

    def execute(self, args: list[str], context: dict) -> str:
        self._registry.scan_local_models()
        available = self._registry.get_available_models()
        all_profiles = self._registry.get_all_profiles()

        lines = [f"=== Model Registry ({len(available)} installed / {len(all_profiles)} known) ===\n"]

        if available:
            lines.append("Installed:")
            for m in available:
                strengths = ", ".join(s.value for s in m.strengths)
                lines.append(f"  * {m.id} — {strengths}")
        else:
            lines.append("No models installed.")

        not_installed = [p for p in all_profiles if p.local_path is None]
        if not_installed:
            lines.append("\nAvailable for download:")
            for m in not_installed:
                lines.append(f"  - {m.id} ({m.parameter_count}, {m.quantization})")
                lines.append(f"    Pull: tqcli model pull {m.id}")

        return "\n".join(lines)
