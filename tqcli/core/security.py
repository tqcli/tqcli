"""Security and virtual environment isolation.

Provides:
- Python venv creation and activation
- Environment detection (WSL2, containers, bare-metal)
- Resource limit enforcement
- Audit logging for all model operations
- Permission checks before executing skills
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import venv
from datetime import datetime, timezone
from pathlib import Path

from tqcli.config import SecurityConfig

logger = logging.getLogger("tqcli.security")


class AuditLogger:
    """Append-only audit log for security-relevant events."""

    def __init__(self, log_path: Path):
        self._path = log_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, details: dict | None = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "pid": os.getpid(),
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        }
        if details:
            entry["details"] = details
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")


class EnvironmentDetector:
    """Detects the execution environment for security decisions."""

    @staticmethod
    def is_wsl() -> bool:
        if platform.system() != "Linux":
            return False
        try:
            with open("/proc/version") as f:
                return "microsoft" in f.read().lower()
        except FileNotFoundError:
            return False

    @staticmethod
    def is_container() -> bool:
        try:
            with open("/proc/1/cgroup") as f:
                content = f.read()
                return "docker" in content or "containerd" in content
        except FileNotFoundError:
            return False
        return Path("/.dockerenv").exists()

    @staticmethod
    def is_virtual_env() -> bool:
        return sys.prefix != sys.base_prefix

    @staticmethod
    def get_environment_type() -> str:
        if EnvironmentDetector.is_container():
            return "container"
        if EnvironmentDetector.is_wsl():
            return "wsl2"
        if EnvironmentDetector.is_virtual_env():
            return "venv"
        return "bare-metal"


class VenvManager:
    """Manages Python virtual environment for isolated tqCLI operation."""

    def __init__(self, venv_path: Path):
        self.venv_path = venv_path

    @property
    def exists(self) -> bool:
        return (self.venv_path / "bin" / "python").exists() or (
            self.venv_path / "Scripts" / "python.exe"
        ).exists()

    @property
    def python_path(self) -> Path:
        if platform.system() == "Windows":
            return self.venv_path / "Scripts" / "python.exe"
        return self.venv_path / "bin" / "python"

    @property
    def pip_path(self) -> Path:
        if platform.system() == "Windows":
            return self.venv_path / "Scripts" / "pip.exe"
        return self.venv_path / "bin" / "pip"

    def create(self, with_pip: bool = True) -> None:
        logger.info(f"Creating virtual environment at {self.venv_path}")
        self.venv_path.parent.mkdir(parents=True, exist_ok=True)
        venv.create(str(self.venv_path), with_pip=with_pip, clear=False)

    def install_packages(self, packages: list[str]) -> subprocess.CompletedProcess:
        if not self.exists:
            raise RuntimeError(f"Virtual environment not found at {self.venv_path}")
        return subprocess.run(
            [str(self.pip_path), "install", *packages],
            capture_output=True,
            text=True,
            timeout=300,
        )

    def run_in_venv(
        self, args: list[str], timeout: int = 120
    ) -> subprocess.CompletedProcess:
        if not self.exists:
            raise RuntimeError(f"Virtual environment not found at {self.venv_path}")
        return subprocess.run(
            [str(self.python_path), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )


class ResourceGuard:
    """Enforces resource limits to prevent system overload."""

    def __init__(self, max_memory_percent: float = 80.0, max_gpu_memory_percent: float = 90.0):
        self.max_memory_percent = max_memory_percent
        self.max_gpu_memory_percent = max_gpu_memory_percent

    def check_memory(self) -> tuple[bool, str]:
        import psutil

        mem = psutil.virtual_memory()
        if mem.percent > self.max_memory_percent:
            return False, (
                f"System memory usage ({mem.percent:.1f}%) exceeds limit "
                f"({self.max_memory_percent}%). Free memory before loading models."
            )
        return True, f"Memory OK: {mem.percent:.1f}% used"

    def check_gpu_memory(self) -> tuple[bool, str]:
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return True, "GPU memory check skipped (nvidia-smi unavailable)"

            for line in result.stdout.strip().split("\n"):
                parts = line.split(",")
                if len(parts) == 2:
                    used = float(parts[0].strip())
                    total = float(parts[1].strip())
                    pct = (used / total) * 100 if total > 0 else 0
                    if pct > self.max_gpu_memory_percent:
                        return False, (
                            f"GPU memory usage ({pct:.1f}%) exceeds limit "
                            f"({self.max_gpu_memory_percent}%). Unload models or use smaller quantization."
                        )
            return True, "GPU memory OK"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return True, "GPU memory check skipped (no NVIDIA GPU)"

    def pre_load_check(self) -> tuple[bool, list[str]]:
        issues = []
        mem_ok, mem_msg = self.check_memory()
        if not mem_ok:
            issues.append(mem_msg)
        gpu_ok, gpu_msg = self.check_gpu_memory()
        if not gpu_ok:
            issues.append(gpu_msg)
        return len(issues) == 0, issues


class SecurityManager:
    """Central security coordinator."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.audit = AuditLogger(config.audit_log_path) if config.audit_log else None
        self.env = EnvironmentDetector()
        self.venv = VenvManager(config.venv_path) if config.use_venv else None
        self.guard = ResourceGuard(config.max_memory_percent, config.max_gpu_memory_percent)

    def initialize(self) -> list[str]:
        """Set up security infrastructure. Returns list of status messages."""
        messages = []
        env_type = self.env.get_environment_type()
        messages.append(f"Environment: {env_type}")

        if self.audit:
            self.audit.log("session_start", {"environment": env_type})
            messages.append(f"Audit logging to: {self.config.audit_log_path}")

        if self.venv and not self.venv.exists:
            self.venv.create()
            messages.append(f"Created venv: {self.config.venv_path}")
        elif self.venv:
            messages.append(f"Using existing venv: {self.config.venv_path}")

        if self.config.sandbox_enabled:
            messages.append("Sandbox mode: enabled")

        return messages

    def log_event(self, event: str, details: dict | None = None) -> None:
        if self.audit:
            self.audit.log(event, details)

    def check_before_load(self, model_id: str) -> tuple[bool, list[str]]:
        ok, issues = self.guard.pre_load_check()
        self.log_event("model_load_check", {"model": model_id, "ok": ok, "issues": issues})
        return ok, issues
