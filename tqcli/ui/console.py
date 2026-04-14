"""Rich console output for tqCLI."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tqcli.core.engine import InferenceStats
from tqcli.core.model_registry import ModelProfile
from tqcli.core.performance import PerformanceMonitor
from tqcli.core.router import RouteDecision

console = Console()


def print_banner():
    banner = Text()
    banner.append("tqCLI", style="bold cyan")
    banner.append(" v0.1.0", style="dim")
    banner.append(" — TurboQuant Local Inference", style="white")
    console.print(Panel(banner, border_style="cyan"))


def print_system_info(info):
    table = Table(title="System Info", border_style="cyan", show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("OS", info.os_display)
    table.add_row("CPU", f"{info.cpu_name} ({info.cpu_cores_physical}c/{info.cpu_cores_logical}t)")
    table.add_row("RAM", f"{info.ram_total_mb:,} MB total / {info.ram_available_mb:,} MB available")

    if info.gpus:
        for gpu in info.gpus:
            table.add_row("GPU", f"{gpu.name} ({gpu.vram_total_mb:,} MB VRAM)")
    elif info.has_metal:
        table.add_row("GPU", "Apple Silicon (Metal)")
    else:
        table.add_row("GPU", "None detected")

    table.add_row("Engine", f"{info.recommended_engine} (recommended)")
    table.add_row("Max Model", f"~{info.max_model_size_estimate_gb} GB")
    table.add_row("Quant", f"{info.recommended_quant} recommended")

    if info.is_wsl:
        table.add_row("Environment", f"WSL{info.wsl_version}")

    console.print(table)


def print_route_decision(decision: RouteDecision):
    console.print(f"  [dim]Router:[/dim] {decision.reason}", highlight=False)


def print_stats_bar(stats: InferenceStats):
    tps = stats.tokens_per_second
    if tps >= 20:
        color = "green"
    elif tps >= 10:
        color = "yellow"
    else:
        color = "red"

    console.print(
        f"\n  [{color}]{tps:.1f} tok/s[/{color}] | "
        f"{stats.completion_tokens} tokens | "
        f"{stats.total_time_s:.2f}s",
        highlight=False,
    )


def print_performance_warning(monitor: PerformanceMonitor):
    stats = monitor.get_stats_display()
    console.print(
        Panel(
            f"[yellow]Performance Warning[/yellow]\n"
            f"Rolling avg: {stats['rolling_tps']} tok/s (threshold: {stats['threshold_tps']})\n"
            f"Consider: tqcli handoff --target claude-code",
            border_style="yellow",
        )
    )


def print_handoff_alert(filepath):
    console.print(
        Panel(
            f"[red]Performance Below Threshold[/red]\n\n"
            f"Handoff file generated:\n"
            f"  {filepath}\n\n"
            f"To continue with a frontier model:\n"
            f"  claude  (then reference the handoff file)",
            border_style="red",
        )
    )


def print_model_list(models: list[ModelProfile], title: str = "Models"):
    table = Table(title=title, border_style="cyan")
    table.add_column("ID", style="bold")
    table.add_column("Params")
    table.add_column("Quant")
    table.add_column("Engine")
    table.add_column("Strengths")
    table.add_column("Status")

    for m in models:
        strengths = ", ".join(s.value for s in m.strengths)
        status = "[green]installed[/green]" if m.local_path else "[dim]available[/dim]"
        table.add_row(m.id, m.parameter_count, m.quantization, m.engine, strengths, status)

    console.print(table)


def print_skill_list(skills):
    table = Table(title="Available Skills", border_style="cyan")
    table.add_column("Skill", style="bold")
    table.add_column("Description")
    table.add_column("Scripts", justify="center")

    for skill in skills:
        scripts = str(len(skill.scripts)) if skill.has_scripts else "-"
        table.add_row(skill.name, skill.description[:80], scripts)

    console.print(table)
