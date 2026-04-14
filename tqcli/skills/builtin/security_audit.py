"""Built-in skill: security audit."""

from __future__ import annotations

from tqcli.config import TqConfig
from tqcli.core.security import EnvironmentDetector, ResourceGuard
from tqcli.skills.base import BuiltinSkill


class SecurityAuditSkill(BuiltinSkill):
    def __init__(self, config: TqConfig):
        self._config = config

    @property
    def name(self) -> str:
        return "security-audit"

    @property
    def description(self) -> str:
        return "Run security checks on the tqCLI environment"

    def execute(self, args: list[str], context: dict) -> str:
        checks = []
        env_type = EnvironmentDetector.get_environment_type()
        is_isolated = env_type in ("wsl2", "container", "venv")
        checks.append(("Environment", is_isolated, env_type))
        checks.append(("Venv Active", EnvironmentDetector.is_virtual_env(), ""))

        guard = ResourceGuard(
            self._config.security.max_memory_percent,
            self._config.security.max_gpu_memory_percent,
        )
        mem_ok, mem_msg = guard.check_memory()
        checks.append(("Memory", mem_ok, mem_msg))
        gpu_ok, gpu_msg = guard.check_gpu_memory()
        checks.append(("GPU Memory", gpu_ok, gpu_msg))

        audit_on = self._config.security.audit_log
        checks.append(("Audit Log", audit_on, str(self._config.security.audit_log_path)))

        lines = ["=== Security Audit ===\n"]
        passes = 0
        for name, ok, detail in checks:
            status = "PASS" if ok else "WARN"
            if ok:
                passes += 1
            lines.append(f"[{status}] {name}: {detail}")

        lines.append(f"\n{passes}/{len(checks)} checks passed")
        return "\n".join(lines)
