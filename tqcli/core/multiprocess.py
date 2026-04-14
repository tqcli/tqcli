"""Multi-process coordinator — manages worker processes that connect to a
shared inference server.

Workflow:
1. Assess hardware → determine max workers and engine
2. Start inference server (llama.cpp or vLLM) as background process
3. Spawn N worker processes, each running a tqCLI chat session
4. Monitor resource usage, kill workers if limits exceeded
5. Graceful shutdown: stop workers first, then server
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from tqcli.config import TqConfig
from tqcli.core.server import InferenceServer, ServerConfig, estimate_server_resources
from tqcli.core.system_info import SystemInfo, detect_system


@dataclass
class WorkerInfo:
    id: int
    pid: int
    started: float
    status: str = "running"  # "running", "stopped", "failed"


@dataclass
class MultiProcessPlan:
    """Resource assessment and plan for multi-process operation."""

    feasible: bool
    engine: str
    max_workers: int
    recommended_workers: int
    model_id: str
    model_path: str
    resource_estimate: dict
    warnings: list[str] = field(default_factory=list)
    reason: str = ""


def assess_multiprocess(
    sys_info: SystemInfo, model_path: str, model_size_mb: int, requested_workers: int,
    preferred_engine: str = "auto", unrestricted: bool = False,
) -> MultiProcessPlan:
    """Assess whether multi-process mode is feasible and plan resources.

    Args:
        sys_info: System hardware info.
        model_path: Path to the model file.
        model_size_mb: Approximate model size in MB.
        requested_workers: How many workers the user wants.
        preferred_engine: "auto", "llama.cpp", or "vllm".
        unrestricted: If True, skip resource checks (stop-trying-to-control-everything-and-just-let-go).
    """
    warnings = []

    # Determine engine
    if preferred_engine == "auto":
        if sys_info.has_nvidia_gpu and sys_info.total_vram_mb >= 8000 and sys_info.vllm_available:
            engine = "vllm"
        else:
            engine = "llama.cpp"
    else:
        engine = preferred_engine

    # Estimate resources
    estimate = estimate_server_resources(
        engine=engine,
        model_size_mb=model_size_mb,
        n_workers=requested_workers,
        vram_mb=sys_info.total_vram_mb,
        ram_mb=sys_info.ram_available_mb,
    )

    # Calculate max workers based on available resources
    if engine == "vllm":
        # vLLM: limited by VRAM. Each worker adds ~120MB KV cache (with PagedAttention)
        if sys_info.total_vram_mb > 0:
            usable_vram = sys_info.total_vram_mb * 0.9 - model_size_mb - 500
            max_workers = max(1, int(usable_vram / 120))
        else:
            max_workers = 1
    else:
        # llama.cpp: sequential queue, so workers don't multiply memory much
        # but more workers = more waiting. Practical limit based on CPU cores.
        max_workers = max(1, sys_info.cpu_cores_physical // 2)
        # Also check RAM can hold the model + overhead
        usable_ram = sys_info.ram_available_mb * 0.8 - 2000  # leave 2GB for OS
        if model_size_mb > usable_ram:
            max_workers = 0

    recommended = min(requested_workers, max_workers, 4)  # cap at 4

    if engine == "llama.cpp" and requested_workers > 1:
        warnings.append(
            "llama.cpp serves requests sequentially. Multiple workers will queue, "
            "not run in parallel. This is useful for task isolation but won't increase throughput."
        )

    if engine == "vllm" and sys_info.total_vram_mb < 8000:
        warnings.append(
            f"vLLM multi-process needs 8+ GB VRAM for a 7B model. "
            f"You have {sys_info.total_vram_mb} MB. Consider llama.cpp instead."
        )

    if not estimate["fits"] and not unrestricted:
        return MultiProcessPlan(
            feasible=False,
            engine=engine,
            max_workers=max_workers,
            recommended_workers=0,
            model_id="",
            model_path=model_path,
            resource_estimate=estimate,
            warnings=warnings,
            reason=f"Insufficient resources: {estimate}",
        )

    if not estimate["fits"] and unrestricted:
        warnings.append(
            "Resource limits exceeded but proceeding anyway "
            "(stop-trying-to-control-everything-and-just-let-go mode). "
            "System may become unresponsive."
        )

    return MultiProcessPlan(
        feasible=True,
        engine=engine,
        max_workers=max_workers,
        recommended_workers=recommended,
        model_id="",
        model_path=model_path,
        resource_estimate=estimate,
        warnings=warnings,
    )


class MultiProcessCoordinator:
    """Manages the inference server and worker processes."""

    def __init__(self, config: TqConfig, plan: MultiProcessPlan):
        self.config = config
        self.plan = plan
        self._server: InferenceServer | None = None
        self._workers: list[WorkerInfo] = []
        self._shutdown_requested = False

    def start_server(self) -> None:
        """Start the shared inference server."""
        server_config = ServerConfig(
            engine=self.plan.engine,
            model_path=self.plan.model_path,
            host="127.0.0.1",
            port=8741,
            context_length=self.config.context_length,
            n_gpu_layers=self.config.n_gpu_layers,
            threads=self.config.threads,
            gpu_memory_utilization=self.config.security.max_gpu_memory_percent / 100.0,
        )
        self._server = InferenceServer(server_config)
        self._server.start()

    def stop_server(self) -> None:
        """Stop the inference server."""
        if self._server:
            self._server.stop()

    def spawn_worker(self, worker_id: int, extra_args: list[str] | None = None) -> WorkerInfo:
        """Spawn a tqCLI chat worker that connects to the server."""
        cmd = [
            sys.executable, "-m", "tqcli",
            "chat",
            "--engine", "server",
            "--server-url", f"http://127.0.0.1:8741",
        ]
        if extra_args:
            cmd.extend(extra_args)

        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
        )
        worker = WorkerInfo(
            id=worker_id,
            pid=proc.pid,
            started=time.time(),
        )
        self._workers.append(worker)
        return worker

    def stop_worker(self, worker_id: int) -> None:
        """Stop a specific worker."""
        for w in self._workers:
            if w.id == worker_id and w.status == "running":
                try:
                    os.kill(w.pid, signal.SIGTERM)
                    w.status = "stopped"
                except OSError:
                    w.status = "failed"

    def stop_all(self) -> None:
        """Graceful shutdown: stop all workers, then the server."""
        self._shutdown_requested = True
        for w in self._workers:
            if w.status == "running":
                self.stop_worker(w.id)
        # Wait briefly for workers to exit
        time.sleep(1)
        self.stop_server()

    def get_workers(self) -> list[WorkerInfo]:
        """Get current worker status, updating stale entries."""
        for w in self._workers:
            if w.status == "running":
                try:
                    os.kill(w.pid, 0)
                except OSError:
                    w.status = "stopped"
        return self._workers

    def get_resource_usage(self) -> dict:
        """Get current resource usage across server and workers."""
        pids = []
        if self._server and self._server.is_running():
            status = self._server.status()
            if status.pid:
                pids.append(status.pid)
        for w in self._workers:
            if w.status == "running":
                pids.append(w.pid)

        total_rss_mb = 0
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                total_rss_mb += proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        mem = psutil.virtual_memory()
        return {
            "managed_processes": len(pids),
            "total_memory_mb": round(total_rss_mb, 1),
            "system_memory_percent": round(mem.percent, 1),
            "active_workers": sum(1 for w in self._workers if w.status == "running"),
        }
