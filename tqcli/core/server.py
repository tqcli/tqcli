"""Inference server management — starts llama.cpp or vLLM as a shared HTTP server
that multiple tqCLI worker processes connect to.

Architecture:
    ┌────────────┐
    │ Coordinator │  manages workers, monitors resources
    └─────┬──────┘
          │ spawns
    ┌─────┴──────┐
    │  Workers   │  tqCLI chat clients (N processes)
    └─────┬──────┘
          │ HTTP (OpenAI-compatible API)
    ┌─────┴──────┐
    │  Server    │  single llama-server or vllm serve process
    │  (1 model) │  handles concurrent requests
    └────────────┘

llama.cpp server: sequential request queue, cross-platform
vLLM server: continuous batching + PagedAttention, Linux+NVIDIA only
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass
class ServerConfig:
    engine: str  # "llama.cpp" or "vllm"
    model_path: str
    host: str = "127.0.0.1"
    port: int = 8741  # tqCLI default port
    context_length: int = 4096
    n_gpu_layers: int = -1
    threads: int = 0
    gpu_memory_utilization: float = 0.85
    quantization: str | None = None
    kv_cache_dtype: str = "auto"
    tensor_parallel_size: int = 1


@dataclass
class ServerStatus:
    running: bool
    pid: int | None = None
    engine: str = ""
    model: str = ""
    host: str = ""
    port: int = 0
    uptime_s: float = 0.0
    requests_served: int = 0


class InferenceServer:
    """Manages a background inference server process."""

    def __init__(self, config: ServerConfig):
        self.config = config
        self._process: subprocess.Popen | None = None
        self._start_time: float = 0.0
        self._pid_file = Path.home() / ".tqcli" / "server.pid"
        self._log_file = Path.home() / ".tqcli" / "server.log"

    @property
    def base_url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}"

    @property
    def api_url(self) -> str:
        return f"{self.base_url}/v1"

    def start(self) -> ServerStatus:
        """Start the inference server as a background process."""
        if self.is_running():
            return self.status()

        self._log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(self._log_file, "w")

        if self.config.engine == "vllm":
            cmd = self._build_vllm_cmd()
        else:
            cmd = self._build_llama_cmd()

        self._process = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detach from parent
        )
        self._start_time = time.time()

        # Write PID file
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._pid_file.write_text(
            json.dumps({
                "pid": self._process.pid,
                "engine": self.config.engine,
                "model": self.config.model_path,
                "host": self.config.host,
                "port": self.config.port,
                "started": self._start_time,
            })
        )

        # Wait for server to become ready
        if not self._wait_for_ready(timeout=120):
            self.stop()
            raise RuntimeError(
                f"Server failed to start within 120s. Check logs: {self._log_file}"
            )

        return self.status()

    def stop(self) -> None:
        """Stop the inference server."""
        pid = self._get_running_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                # Wait up to 10s for graceful shutdown
                for _ in range(20):
                    try:
                        os.kill(pid, 0)  # check if still alive
                        time.sleep(0.5)
                    except OSError:
                        break
                else:
                    os.kill(pid, signal.SIGKILL)
            except OSError:
                pass

        if self._pid_file.exists():
            self._pid_file.unlink()
        self._process = None

    def is_running(self) -> bool:
        pid = self._get_running_pid()
        if not pid:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            # Stale PID file
            if self._pid_file.exists():
                self._pid_file.unlink()
            return False

    def status(self) -> ServerStatus:
        pid = self._get_running_pid()
        if not pid or not self.is_running():
            return ServerStatus(running=False)

        info = self._read_pid_file()
        uptime = time.time() - info.get("started", time.time())
        return ServerStatus(
            running=True,
            pid=pid,
            engine=info.get("engine", ""),
            model=info.get("model", ""),
            host=info.get("host", "127.0.0.1"),
            port=info.get("port", 8741),
            uptime_s=uptime,
        )

    def health_check(self) -> bool:
        """Check if the server is responding to requests."""
        import urllib.request
        import urllib.error

        try:
            url = f"{self.base_url}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            # Try the models endpoint as fallback (vLLM uses this)
            try:
                url = f"{self.api_url}/models"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    return resp.status == 200
            except (urllib.error.URLError, OSError, TimeoutError):
                return False

    def _wait_for_ready(self, timeout: int = 120) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._process and self._process.poll() is not None:
                return False  # process exited
            if self.health_check():
                return True
            time.sleep(2)
        return False

    def _build_llama_cmd(self) -> list[str]:
        cmd = [
            sys.executable, "-m", "llama_cpp.server",
            "--model", self.config.model_path,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--n_ctx", str(self.config.context_length),
        ]
        if self.config.n_gpu_layers != 0:
            cmd.extend(["--n_gpu_layers", str(self.config.n_gpu_layers)])
        if self.config.threads > 0:
            cmd.extend(["--n_threads", str(self.config.threads)])
        return cmd

    def _build_vllm_cmd(self) -> list[str]:
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_path,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--max-model-len", str(self.config.context_length),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
        ]
        if self.config.quantization:
            cmd.extend(["--quantization", self.config.quantization])
        if self.config.kv_cache_dtype != "auto":
            cmd.extend(["--kv-cache-dtype", self.config.kv_cache_dtype])
        if self.config.tensor_parallel_size > 1:
            cmd.extend(["--tensor-parallel-size", str(self.config.tensor_parallel_size)])
        return cmd

    def _get_running_pid(self) -> int | None:
        if self._process and self._process.poll() is None:
            return self._process.pid
        info = self._read_pid_file()
        return info.get("pid")

    def _read_pid_file(self) -> dict:
        if not self._pid_file.exists():
            return {}
        try:
            return json.loads(self._pid_file.read_text())
        except (json.JSONDecodeError, OSError):
            return {}


def estimate_server_resources(
    engine: str, model_size_mb: int, n_workers: int, vram_mb: int, ram_mb: int
) -> dict:
    """Estimate resource usage for a server with N concurrent workers.

    Returns a dict with estimated memory usage and whether it fits.
    """
    engine_overhead_mb = 500
    per_worker_kv_cache_mb = 200  # conservative estimate per concurrent request

    if engine == "vllm":
        # vLLM uses PagedAttention — KV cache is shared and managed efficiently
        # Overhead scales sub-linearly with workers
        kv_total_mb = per_worker_kv_cache_mb * n_workers * 0.6  # PagedAttention savings
        total_vram_mb = model_size_mb + kv_total_mb + engine_overhead_mb
        fits_vram = total_vram_mb <= vram_mb * 0.9 if vram_mb > 0 else False
        return {
            "engine": "vllm",
            "model_mb": model_size_mb,
            "kv_cache_mb": int(kv_total_mb),
            "overhead_mb": engine_overhead_mb,
            "total_vram_needed_mb": int(total_vram_mb),
            "available_vram_mb": vram_mb,
            "fits": fits_vram,
            "note": "vLLM uses PagedAttention for efficient KV cache sharing across workers",
        }
    else:
        # llama.cpp server — sequential queue, one KV cache active at a time
        kv_total_mb = per_worker_kv_cache_mb  # only 1 active at a time
        total_gpu_mb = min(model_size_mb, vram_mb * 0.85) if vram_mb > 0 else 0
        total_ram_mb = (model_size_mb - total_gpu_mb) + kv_total_mb + engine_overhead_mb
        fits = total_ram_mb <= ram_mb * 0.8
        return {
            "engine": "llama.cpp",
            "model_mb": model_size_mb,
            "kv_cache_mb": int(kv_total_mb),
            "overhead_mb": engine_overhead_mb,
            "gpu_offload_mb": int(total_gpu_mb),
            "total_ram_needed_mb": int(total_ram_mb),
            "available_ram_mb": ram_mb,
            "fits": fits,
            "note": "llama.cpp queues requests sequentially — workers wait in line",
        }
