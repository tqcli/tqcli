"""Built-in skill: system info display."""

from __future__ import annotations

from tqcli.core.system_info import detect_system
from tqcli.skills.base import BuiltinSkill


class SystemInfoSkill(BuiltinSkill):
    @property
    def name(self) -> str:
        return "system-info"

    @property
    def description(self) -> str:
        return "Display OS, hardware, and inference engine information"

    def execute(self, args: list[str], context: dict) -> str:
        info = detect_system()
        lines = [
            "=== tqCLI System Report ===",
            f"OS:          {info.os_display}",
            f"Arch:        {info.arch}",
            f"CPU:         {info.cpu_name} ({info.cpu_cores_physical}c / {info.cpu_cores_logical}t)",
            f"RAM:         {info.ram_total_mb:,} MB total / {info.ram_available_mb:,} MB available",
        ]
        if info.gpus:
            for gpu in info.gpus:
                lines.append(f"GPU:         {gpu.name} ({gpu.vram_total_mb:,} MB)")
        elif info.has_metal:
            lines.append("GPU:         Apple Silicon (Metal)")
        else:
            lines.append("GPU:         None detected")

        lines.append(f"Engine:      {info.recommended_engine} (recommended)")
        lines.append(f"Max Model:   ~{info.max_model_size_estimate_gb} GB")
        lines.append(f"Quant:       {info.recommended_quant} recommended")
        if info.is_wsl:
            lines.append(f"Environment: WSL{info.wsl_version}")
        return "\n".join(lines)
