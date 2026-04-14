"""Configuration management for tqCLI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _default_models_dir() -> Path:
    return Path.home() / ".tqcli" / "models"


@dataclass
class PerformanceConfig:
    min_tokens_per_second: float = 5.0
    warning_tokens_per_second: float = 10.0
    measurement_window_tokens: int = 50
    auto_handoff: bool = False


@dataclass
class SecurityConfig:
    use_venv: bool = True
    venv_path: Path = field(default_factory=lambda: Path.home() / ".tqcli" / "venv")
    sandbox_enabled: bool = True
    audit_log: bool = True
    audit_log_path: Path = field(default_factory=lambda: Path.home() / ".tqcli" / "audit.log")
    max_memory_percent: float = 80.0
    max_gpu_memory_percent: float = 90.0


@dataclass
class MultiProcessConfig:
    enabled: bool = False
    max_workers: int = 3
    server_port: int = 8741
    server_host: str = "127.0.0.1"
    auto_start_server: bool = True


@dataclass
class RouterConfig:
    enabled: bool = True
    default_model: str | None = None
    coding_model: str | None = None
    reasoning_model: str | None = None
    general_model: str | None = None


@dataclass
class TqConfig:
    models_dir: Path = field(default_factory=_default_models_dir)
    preferred_engine: str = "auto"  # "auto", "llama.cpp", "vllm"
    default_quantization: str = "Q4_K_M"
    context_length: int = 4096
    n_gpu_layers: int = -1  # -1 = all layers
    threads: int = 0  # 0 = auto-detect
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    multiprocess: MultiProcessConfig = field(default_factory=MultiProcessConfig)
    unrestricted: bool = False  # stop-trying-to-control-everything-and-just-let-go mode
    skills_dir: Path = field(default_factory=lambda: Path.home() / ".tqcli" / "skills")
    memory_dir: Path = field(default_factory=lambda: Path.home() / ".tqcli" / "memory")

    @classmethod
    def load(cls, path: Path | None = None) -> TqConfig:
        if path is None:
            path = Path.home() / ".tqcli" / "config.yaml"
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> TqConfig:
        perf = PerformanceConfig(**data.pop("performance", {}))
        sec_raw = data.pop("security", {})
        for key in ("venv_path", "audit_log_path"):
            if key in sec_raw:
                sec_raw[key] = Path(sec_raw[key])
        sec = SecurityConfig(**sec_raw)
        router = RouterConfig(**data.pop("router", {}))
        mp = MultiProcessConfig(**data.pop("multiprocess", {}))
        for key in ("models_dir", "skills_dir", "memory_dir"):
            if key in data:
                data[key] = Path(data[key])
        return cls(performance=perf, security=sec, router=router, multiprocess=mp, **data)

    def save(self, path: Path | None = None) -> None:
        if path is None:
            path = Path.home() / ".tqcli" / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)

        def _serialize(obj):
            if isinstance(obj, Path):
                return str(obj)
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _serialize(v) for k, v in obj.__dict__.items()}
            return obj

        with open(path, "w") as f:
            yaml.dump(_serialize(self), f, default_flow_style=False, sort_keys=False)

    def ensure_dirs(self) -> None:
        for d in (self.models_dir, self.skills_dir, self.memory_dir):
            d.mkdir(parents=True, exist_ok=True)
        if self.security.audit_log:
            self.security.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
