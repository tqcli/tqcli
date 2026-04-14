"""Basic tests for tqCLI core modules."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


def test_version():
    from tqcli import __version__
    assert __version__ == "0.2.0"


def test_config_defaults():
    from tqcli.config import TqConfig
    config = TqConfig()
    assert config.preferred_engine == "auto"
    assert config.default_quantization == "Q4_K_M"
    assert config.context_length == 4096
    assert config.performance.min_tokens_per_second == 5.0
    assert config.security.use_venv is True


def test_config_save_load():
    from tqcli.config import TqConfig
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.yaml"
        config = TqConfig()
        config.context_length = 8192
        config.save(path)

        loaded = TqConfig.load(path)
        assert loaded.context_length == 8192
        assert loaded.preferred_engine == "auto"


def test_system_info():
    from tqcli.core.system_info import detect_system
    info = detect_system()
    assert info.os_name in ("linux", "darwin", "windows")
    assert info.ram_total_mb > 0
    assert info.cpu_cores_logical > 0
    assert info.recommended_engine in ("llama.cpp", "vllm")
    assert info.recommended_quant in ("Q2_K", "Q3_K_M", "Q4_K_M", "Q6_K", "Q8_0")


def test_model_registry():
    from tqcli.core.model_registry import BUILTIN_PROFILES, ModelRegistry, TaskDomain
    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(Path(tmpdir))
        profiles = registry.get_all_profiles()
        assert len(profiles) == len(BUILTIN_PROFILES)

        # Check that all profiles have strength scores
        for p in profiles:
            assert "coding" in p.strength_scores
            assert "general" in p.strength_scores
            assert 0.0 <= p.strength_scores["coding"] <= 1.0

        # Check domain ranking
        coders = registry.get_models_for_domain(TaskDomain.CODING)
        # Qwen Coder should rank first for coding (but won't appear since none are "available")
        # Just verify the method doesn't error
        assert isinstance(coders, list)


def test_router_classification():
    from tqcli.core.router import TaskDomain, classify_prompt

    domain, conf = classify_prompt("Write a Python function to sort a list")
    assert domain == TaskDomain.CODING

    domain, conf = classify_prompt("Calculate the integral of x^2 from 0 to 5")
    assert domain == TaskDomain.MATH

    domain, conf = classify_prompt("Analyze the pros and cons of microservices")
    assert domain == TaskDomain.REASONING

    domain, conf = classify_prompt("Hello, how are you?")
    assert domain == TaskDomain.GENERAL


def test_performance_monitor():
    from tqcli.config import PerformanceConfig
    from tqcli.core.performance import PerformanceMonitor

    config = PerformanceConfig(min_tokens_per_second=10.0, warning_tokens_per_second=20.0)
    monitor = PerformanceMonitor(config)

    # Record fast inference
    monitor.record(tokens=100, elapsed_s=2.0)
    assert monitor.current_tps == 50.0
    assert not monitor.is_below_threshold

    # Record slow inferences to trigger threshold
    for _ in range(5):
        monitor.record(tokens=10, elapsed_s=5.0)
    assert monitor.rolling_tps < 10.0
    assert monitor.is_below_threshold


def test_handoff_generation():
    from tqcli.config import PerformanceConfig
    from tqcli.core.handoff import generate_handoff
    from tqcli.core.performance import PerformanceMonitor

    with tempfile.TemporaryDirectory() as tmpdir:
        config = PerformanceConfig()
        monitor = PerformanceMonitor(config)
        monitor.record(tokens=10, elapsed_s=5.0)

        filepath = generate_handoff(
            monitor=monitor,
            conversation_history=[{"role": "user", "content": "Hello"}],
            task_description="Test task",
            output_dir=Path(tmpdir),
            target_cli="claude-code",
        )

        assert filepath.exists()
        content = filepath.read_text()
        assert "tqCLI Handoff" in content
        assert "claude-code" in content
        assert "Test task" in content


def test_skill_loader():
    from tqcli.skills.loader import SkillLoader
    # Use the actual .claude/skills directory in the project
    skills_dir = Path(__file__).parent.parent / ".claude" / "skills"
    if skills_dir.exists():
        loader = SkillLoader([skills_dir])
        skills = loader.discover()
        tq_skills = loader.get_tq_skills()
        assert len(tq_skills) >= 5  # tq-system-info, tq-model-manager, etc.
        for skill in tq_skills:
            assert skill.name.startswith("tq-")


def test_security_environment_detector():
    from tqcli.core.security import EnvironmentDetector
    env_type = EnvironmentDetector.get_environment_type()
    assert env_type in ("bare-metal", "wsl2", "container", "venv")


def test_security_resource_guard():
    from tqcli.core.security import ResourceGuard
    guard = ResourceGuard(max_memory_percent=99.0, max_gpu_memory_percent=99.0)
    ok, msg = guard.check_memory()
    assert ok  # 99% threshold should always pass
