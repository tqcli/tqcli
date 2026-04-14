"""tqCLI — TurboQuant CLI main entry point.

Usage:
    tqcli                     # Interactive chat (auto-selects model)
    tqcli chat                # Interactive chat mode
    tqcli chat --model <id>   # Chat with specific model
    tqcli system info         # Show system hardware and capabilities
    tqcli model list          # List available/installed models
    tqcli model pull <id>     # Download a model from HuggingFace
    tqcli model remove <id>   # Remove a downloaded model
    tqcli benchmark           # Benchmark loaded models
    tqcli security audit      # Run security checks
    tqcli skills              # List available skills
    tqcli handoff             # Generate handoff file for frontier CLI
    tqcli config show         # Show current configuration
    tqcli config init         # Initialize default configuration
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from tqcli import __version__


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="tqcli")
@click.pass_context
def main(ctx):
    """tqCLI — TurboQuant CLI for local LLM inference with smart routing."""
    ctx.ensure_object(dict)
    if ctx.invoked_subcommand is None:
        # Default: launch interactive chat
        ctx.invoke(chat)


# ── Chat ──────────────────────────────────────────────────────────────


@main.command()
@click.option("--model", "-m", default=None, help="Model ID to use (bypasses router)")
@click.option("--engine", "-e", type=click.Choice(["auto", "llama.cpp", "vllm"]), default="auto")
@click.option("--context-length", "-c", type=int, default=None)
def chat(model, engine, context_length):
    """Start an interactive chat session."""
    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.core.performance import PerformanceMonitor
    from tqcli.core.router import ModelRouter
    from tqcli.core.security import SecurityManager
    from tqcli.core.system_info import detect_system
    from tqcli.ui.console import console, print_banner
    from tqcli.ui.interactive import InteractiveSession

    print_banner()
    config = TqConfig.load()
    config.ensure_dirs()

    if context_length:
        config.context_length = context_length

    # Security initialization
    sec = SecurityManager(config.security)
    for msg in sec.initialize():
        console.print(f"  [dim]{msg}[/dim]")

    # System detection
    sys_info = detect_system()
    console.print(f"  [dim]System: {sys_info.os_display} | RAM: {sys_info.ram_available_mb:,} MB[/dim]")
    if sys_info.gpus:
        console.print(f"  [dim]GPU: {sys_info.gpus[0].name} ({sys_info.total_vram_mb:,} MB VRAM)[/dim]")

    # Model registry
    registry = ModelRegistry(config.models_dir)
    registry.scan_local_models()
    available = registry.get_available_models()

    if not available:
        console.print("\n[yellow]No models installed.[/yellow]")
        console.print("Download one with:")
        for p in registry.get_all_profiles()[:3]:
            console.print(f"  tqcli model pull {p.id}")
        return

    # Router
    router = ModelRouter(registry, sys_info.ram_available_mb, sys_info.total_vram_mb)
    if model:
        router.set_override(model)

    # Engine selection
    selected_engine = engine if engine != "auto" else config.preferred_engine
    if selected_engine == "auto":
        selected_engine = sys_info.recommended_engine

    if selected_engine == "vllm":
        from tqcli.core.vllm_backend import VllmBackend
        eng = VllmBackend(max_model_len=config.context_length)
    else:
        from tqcli.core.llama_backend import LlamaBackend
        eng = LlamaBackend(
            n_ctx=config.context_length,
            n_gpu_layers=config.n_gpu_layers,
            n_threads=config.threads,
        )

    if not eng.is_available:
        console.print(f"\n[red]Engine '{selected_engine}' not installed.[/red]")
        if selected_engine == "vllm":
            console.print("Install with: pip install vllm")
        else:
            console.print("Install with: pip install llama-cpp-python")
        return

    # Load the first available model (router will switch as needed)
    target = available[0]
    if model:
        target = registry.get_profile(model) or target
    if target.local_path:
        console.print(f"\n  Loading [bold]{target.display_name}[/bold]...")
        sec.log_event("model_load", {"model": target.id})
        eng.load_model(str(target.local_path))
    else:
        console.print(f"\n[red]Model {target.id} not found locally.[/red]")
        return

    # Performance monitor
    monitor = PerformanceMonitor(config.performance)

    # Launch interactive session
    session = InteractiveSession(config, eng, router, monitor)
    try:
        session.run()
    finally:
        eng.unload_model()
        sec.log_event("session_end")


# ── System ────────────────────────────────────────────────────────────


@main.group()
def system():
    """System information and diagnostics."""
    pass


@system.command("info")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def system_info(as_json):
    """Display OS, hardware, and inference engine capabilities."""
    import json as json_mod

    from tqcli.core.system_info import detect_system
    from tqcli.ui.console import print_system_info

    info = detect_system()
    if as_json:
        data = {
            "os": info.os_name,
            "os_display": info.os_display,
            "arch": info.arch,
            "is_wsl": info.is_wsl,
            "cpu_cores": info.cpu_cores_logical,
            "ram_total_mb": info.ram_total_mb,
            "ram_available_mb": info.ram_available_mb,
            "gpus": [{"name": g.name, "vram_mb": g.vram_total_mb} for g in info.gpus],
            "recommended_engine": info.recommended_engine,
            "recommended_quant": info.recommended_quant,
            "max_model_gb": info.max_model_size_estimate_gb,
        }
        click.echo(json_mod.dumps(data, indent=2))
    else:
        print_system_info(info)


# ── Model ─────────────────────────────────────────────────────────────


@main.group()
def model():
    """Model management — list, pull, remove."""
    pass


@model.command("list")
def model_list():
    """List all known and installed models."""
    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.ui.console import print_model_list

    config = TqConfig.load()
    registry = ModelRegistry(config.models_dir)
    registry.scan_local_models()
    print_model_list(registry.get_all_profiles())


@model.command("pull")
@click.argument("model_id")
def model_pull(model_id):
    """Download a model from HuggingFace Hub."""
    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.core.security import SecurityManager
    from tqcli.ui.console import console

    config = TqConfig.load()
    config.ensure_dirs()
    registry = ModelRegistry(config.models_dir)
    profile = registry.get_profile(model_id)

    if not profile:
        console.print(f"[red]Unknown model: {model_id}[/red]")
        console.print("Available models:")
        for p in registry.get_all_profiles():
            console.print(f"  {p.id}")
        return

    # Security check
    sec = SecurityManager(config.security)
    ok, issues = sec.check_before_load(model_id)
    if not ok:
        console.print("[red]Resource check failed:[/red]")
        for issue in issues:
            console.print(f"  {issue}")
        return

    console.print(f"Downloading [bold]{profile.display_name}[/bold]...")
    console.print(f"  From: {profile.hf_repo}")
    console.print(f"  File: {profile.filename}")
    console.print(f"  To:   {config.models_dir}/")

    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            repo_id=profile.hf_repo,
            filename=profile.filename,
            local_dir=str(config.models_dir),
            local_dir_use_symlinks=False,
        )
        sec.log_event("model_downloaded", {"model": model_id, "path": path})
        console.print(f"\n[green]Downloaded:[/green] {path}")
    except Exception as e:
        console.print(f"\n[red]Download failed:[/red] {e}")


@model.command("remove")
@click.argument("model_id")
def model_remove(model_id):
    """Remove a downloaded model."""
    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.ui.console import console

    config = TqConfig.load()
    registry = ModelRegistry(config.models_dir)
    registry.scan_local_models()
    profile = registry.get_profile(model_id)

    if not profile or not profile.local_path:
        console.print(f"[red]Model not found locally: {model_id}[/red]")
        return

    path = profile.local_path
    size_mb = path.stat().st_size / (1024 * 1024)
    if click.confirm(f"Remove {profile.display_name} ({size_mb:.0f} MB)?"):
        path.unlink()
        console.print(f"[green]Removed:[/green] {path}")


# ── Benchmark ─────────────────────────────────────────────────────────


@main.command()
@click.option("--model", "-m", default=None, help="Model ID to benchmark")
@click.option("--all-models", is_flag=True, help="Benchmark all installed models")
def benchmark(model, all_models):
    """Run performance benchmarks on installed models."""
    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.ui.console import console

    config = TqConfig.load()
    registry = ModelRegistry(config.models_dir)
    registry.scan_local_models()
    available = registry.get_available_models()

    if not available:
        console.print("[yellow]No models installed. Use 'tqcli model pull <id>' first.[/yellow]")
        return

    console.print("[bold]Running benchmarks...[/bold]\n")
    # Delegate to benchmark script
    import subprocess

    script = Path(__file__).parent.parent / ".claude" / "skills" / "tq-benchmark" / "scripts" / "run_benchmark.py"
    cmd = [sys.executable, str(script)]
    if model:
        cmd.extend(["--model", model])
    elif all_models:
        cmd.append("--all")
    subprocess.run(cmd)


# ── Security ──────────────────────────────────────────────────────────


@main.group()
def security():
    """Security tools and audit."""
    pass


@security.command("audit")
@click.option("--json", "as_json", is_flag=True)
@click.option("--fix", is_flag=True, help="Auto-fix safe issues")
def security_audit(as_json, fix):
    """Run security audit on the tqCLI environment."""
    import subprocess

    script = Path(__file__).parent.parent / ".claude" / "skills" / "tq-security-audit" / "scripts" / "run_audit.py"
    cmd = [sys.executable, str(script)]
    if as_json:
        cmd.append("--json")
    if fix:
        cmd.append("--fix")
    subprocess.run(cmd)


# ── Skills ────────────────────────────────────────────────────────────


@main.command("skills")
def list_skills():
    """List available tqCLI skills."""
    from tqcli.skills.loader import SkillLoader
    from tqcli.ui.console import print_skill_list

    project_skills = Path(__file__).parent.parent / ".claude" / "skills"
    loader = SkillLoader([project_skills])
    skills = loader.get_tq_skills()
    print_skill_list(skills)


# ── Handoff ───────────────────────────────────────────────────────────


@main.command()
@click.option("--task", "-t", required=True, help="Task description")
@click.option(
    "--target",
    type=click.Choice(["auto", "claude-code", "gemini-cli", "aider", "openai"]),
    default="auto",
)
def handoff(task, target):
    """Generate a handoff file to transfer work to a frontier model CLI."""
    from tqcli.config import TqConfig
    from tqcli.core.handoff import generate_handoff
    from tqcli.core.performance import PerformanceMonitor
    from tqcli.ui.console import console

    config = TqConfig.load()
    monitor = PerformanceMonitor(config.performance)
    # Simulate slow performance for handoff context
    monitor.record(tokens=20, elapsed_s=10.0)
    monitor.record(tokens=15, elapsed_s=8.0)

    output_dir = Path.home() / ".tqcli" / "handoffs"
    filepath = generate_handoff(
        monitor=monitor,
        conversation_history=[{"role": "user", "content": task}],
        task_description=task,
        output_dir=output_dir,
        target_cli=target,
    )
    console.print(f"[green]Handoff file generated:[/green] {filepath}")
    console.print(f"\nTo continue with {target}:")
    if target in ("auto", "claude-code"):
        console.print("  claude")
        console.print(f"  Then reference: @{filepath.name}")


# ── Config ────────────────────────────────────────────────────────────


@main.group()
def config():
    """Configuration management."""
    pass


@config.command("show")
def config_show():
    """Show current tqCLI configuration."""
    from tqcli.config import TqConfig
    from tqcli.ui.console import console

    cfg = TqConfig.load()
    console.print("[bold]tqCLI Configuration[/bold]\n")
    console.print(f"  Models dir:       {cfg.models_dir}")
    console.print(f"  Preferred engine: {cfg.preferred_engine}")
    console.print(f"  Default quant:    {cfg.default_quantization}")
    console.print(f"  Context length:   {cfg.context_length}")
    console.print(f"  GPU layers:       {cfg.n_gpu_layers}")
    console.print(f"  Threads:          {cfg.threads or 'auto'}")
    console.print(f"\n  [bold]Performance[/bold]")
    console.print(f"  Min tok/s:        {cfg.performance.min_tokens_per_second}")
    console.print(f"  Warning tok/s:    {cfg.performance.warning_tokens_per_second}")
    console.print(f"  Auto handoff:     {cfg.performance.auto_handoff}")
    console.print(f"\n  [bold]Security[/bold]")
    console.print(f"  Venv:             {cfg.security.use_venv}")
    console.print(f"  Sandbox:          {cfg.security.sandbox_enabled}")
    console.print(f"  Audit log:        {cfg.security.audit_log}")
    console.print(f"  Max memory:       {cfg.security.max_memory_percent}%")
    console.print(f"  Max GPU memory:   {cfg.security.max_gpu_memory_percent}%")


@config.command("init")
def config_init():
    """Initialize default tqCLI configuration."""
    from tqcli.config import TqConfig
    from tqcli.ui.console import console

    cfg = TqConfig()
    cfg.save()
    cfg.ensure_dirs()
    console.print("[green]Configuration initialized at ~/.tqcli/config.yaml[/green]")
    console.print("Edit the file to customize settings, or use 'tqcli config show' to review.")
