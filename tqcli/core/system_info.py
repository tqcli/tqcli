"""OS and hardware detection for cross-platform operation."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

import psutil


@dataclass
class GPUInfo:
    name: str
    vram_total_mb: int
    vram_free_mb: int
    compute_capability: str = ""
    driver_version: str = ""
    cuda_version: str = ""


@dataclass
class SystemInfo:
    os_name: str  # "linux", "darwin", "windows"
    os_version: str
    os_display: str  # "Linux (Ubuntu 22.04)", "macOS 14.2", "Windows 11 Pro"
    arch: str  # "x86_64", "arm64"
    is_wsl: bool = False
    wsl_version: str = ""

    cpu_name: str = ""
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0

    ram_total_mb: int = 0
    ram_available_mb: int = 0

    gpus: list[GPUInfo] = field(default_factory=list)
    has_nvidia_gpu: bool = False
    has_metal: bool = False  # Apple Silicon Metal
    total_vram_mb: int = 0

    llama_cpp_available: bool = False
    vllm_available: bool = False

    @property
    def recommended_engine(self) -> str:
        if self.has_nvidia_gpu and self.total_vram_mb >= 8000 and self.vllm_available:
            return "vllm"
        if self.llama_cpp_available:
            return "llama.cpp"
        if self.has_nvidia_gpu and self.total_vram_mb >= 8000:
            return "vllm"
        return "llama.cpp"

    @property
    def max_model_size_estimate_gb(self) -> float:
        if self.has_nvidia_gpu and self.total_vram_mb > 0:
            usable_vram = self.total_vram_mb * 0.85
            return round(usable_vram / 1024, 1)
        usable_ram = self.ram_available_mb * 0.70
        return round(usable_ram / 1024, 1)

    @property
    def recommended_quant(self) -> str:
        budget = self.max_model_size_estimate_gb
        if budget >= 14:
            return "Q8_0"
        if budget >= 8:
            return "Q6_K"
        if budget >= 5:
            return "Q4_K_M"
        if budget >= 3:
            return "Q3_K_M"
        return "Q2_K"


def _detect_wsl() -> tuple[bool, str]:
    if platform.system() != "Linux":
        return False, ""
    try:
        with open("/proc/version", "r") as f:
            version_str = f.read().lower()
        if "microsoft" in version_str or "wsl" in version_str:
            ver = "2" if "wsl2" in version_str else "1"
            return True, ver
    except FileNotFoundError:
        pass
    return False, ""


def _detect_gpus() -> list[GPUInfo]:
    gpus = []
    if not shutil.which("nvidia-smi"):
        return gpus
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append(
                        GPUInfo(
                            name=parts[0],
                            vram_total_mb=int(float(parts[1])),
                            vram_free_mb=int(float(parts[2])),
                            driver_version=parts[3],
                        )
                    )
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    # Try to get CUDA version
    if gpus:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                caps = result.stdout.strip().split("\n")
                for i, cap in enumerate(caps):
                    if i < len(gpus):
                        gpus[i].compute_capability = cap.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return gpus


def _detect_os_display() -> str:
    system = platform.system()
    if system == "Darwin":
        ver = platform.mac_ver()[0]
        return f"macOS {ver}" if ver else "macOS"
    if system == "Windows":
        ver = platform.version()
        edition = platform.win32_edition() if hasattr(platform, "win32_edition") else ""
        return f"Windows {platform.release()} {edition}".strip()
    if system == "Linux":
        try:
            result = subprocess.run(
                ["lsb_release", "-ds"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                return f"Linux ({result.stdout.strip()})"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        name = line.split("=", 1)[1].strip().strip('"')
                        return f"Linux ({name})"
        except FileNotFoundError:
            pass
        return f"Linux {platform.release()}"
    return f"{system} {platform.release()}"


def _check_python_package(name: str) -> bool:
    try:
        __import__(name.replace("-", "_").replace(".", "_"))
        return True
    except ImportError:
        return False


def detect_system() -> SystemInfo:
    system = platform.system().lower()
    is_wsl, wsl_ver = _detect_wsl()
    gpus = _detect_gpus()
    mem = psutil.virtual_memory()

    has_metal = system == "darwin" and platform.machine() == "arm64"

    info = SystemInfo(
        os_name=system,
        os_version=platform.release(),
        os_display=_detect_os_display(),
        arch=platform.machine(),
        is_wsl=is_wsl,
        wsl_version=wsl_ver,
        cpu_name=platform.processor() or "Unknown",
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        ram_total_mb=int(mem.total / (1024 * 1024)),
        ram_available_mb=int(mem.available / (1024 * 1024)),
        gpus=gpus,
        has_nvidia_gpu=len(gpus) > 0,
        has_metal=has_metal,
        total_vram_mb=sum(g.vram_total_mb for g in gpus),
        llama_cpp_available=_check_python_package("llama_cpp"),
        vllm_available=_check_python_package("vllm"),
    )

    if is_wsl:
        info.os_display += f" (WSL{wsl_ver})"

    return info
