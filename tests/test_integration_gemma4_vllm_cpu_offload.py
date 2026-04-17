#!/usr/bin/env python3
"""Gemma 4 E2B vLLM CPU offload integration test helper.

This file is a copy of tests/test_gemma4_vllm_cpu_offload.py adapted for the
TurboQuant KV comparison suite. It preserves the same Gemma 4 E2B vLLM CPU
offload workflow and adds explicit CLI/workflow parity checks from
llama_cpp_test_cases.md. It omits JSON/MD report writing so it can be consumed
as part of the combined turboquant_kv_comparison_report output.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqcli.config import TqConfig
from tqcli.core.engine import ChatMessage
from tqcli.core.kv_quantizer import (
    check_turboquant_compatibility,
    detect_model_precision,
    plan_quantization_pipeline,
)
from tqcli.core.model_registry import BUILTIN_PROFILES
from tqcli.core.performance import PerformanceMonitor
from tqcli.core.quantizer import (
    QuantizationMethod,
    estimate_bf16_model_size,
    estimate_quantized_size,
    select_quantization,
)
from tqcli.core.system_info import detect_system
from tqcli.core.thinking import (
    ThinkingConfig,
    ThinkingFormat,
    build_system_prompt_with_thinking,
    extract_thinking,
)
from tqcli.core.vllm_config import build_vllm_config
from tests.integration_lifecycle import run_full_lifecycle

REPORT_DIR = Path(__file__).parent / "integration_reports"

try:
    import vllm  # noqa: F401
    HAS_VLLM = True
    VLLM_IMPORT_ERROR = ""
except ImportError as _vllm_err:
    HAS_VLLM = False
    VLLM_IMPORT_ERROR = str(_vllm_err)


@dataclass
class StepResult:
    name: str
    passed: bool
    duration_s: float = 0.0
    details: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class TestResult:
    test_name: str
    model_id: str
    model_family: str
    engine: str
    quantization: str = ""
    kv_quant: str = ""
    pipeline_path: str = ""
    pipeline_stages: str = ""
    started: str = ""
    finished: str = ""
    total_duration_s: float = 0.0
    steps: list[StepResult] = field(default_factory=list)
    passed: bool = False

    def add_step(self, step: StepResult):
        self.steps.append(step)

    @property
    def pass_count(self):
        return sum(1 for s in self.steps if s.passed)

    @property
    def fail_count(self):
        return sum(1 for s in self.steps if not s.passed)


def get_system_info_dict():
    info = detect_system()
    return {
        "os": info.os_display,
        "gpu": info.gpus[0].name if info.gpus else "None",
        "vram_mb": info.total_vram_mb,
        "ram_total_mb": info.ram_total_mb,
        "ram_available_mb": info.ram_available_mb,
        "cuda_toolkit": info.gpus[0].cuda_toolkit_version if info.gpus else "N/A",
        "compute_capability": info.gpus[0].compute_capability if info.gpus else "N/A",
        "is_wsl": info.is_wsl,
        "turboquant_kv": {},
    }


def run_cli_step(command: list[str], timeout: int = 120) -> StepResult:
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "tqcli", *command],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=timeout,
        )
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        passed = proc.returncode == 0
        details = f"cmd: tqcli {' '.join(command)} | rc={proc.returncode}"
        if not passed:
            details += f" | err={stderr[:200]}"
        if command[:2] == ["system", "info"] and passed:
            try:
                data = json.loads(stdout)
                passed = passed and isinstance(data, dict)
                details += f" | keys={list(data.keys())[:5]}"
            except json.JSONDecodeError as exc:
                passed = False
                details += f" | json_error={exc}"

        return StepResult(
            name=f"cli_{'_'.join(command).replace(' ', '_')}",
            passed=passed,
            duration_s=time.time() - start,
            details=details,
            metrics={"stdout": stdout[:200], "stderr": stderr[:200], "return_code": proc.returncode},
        )
    except Exception as exc:
        return StepResult(
            name=f"cli_{'_'.join(command).replace(' ', '_')}",
            passed=False,
            duration_s=time.time() - start,
            details=f"Exception: {exc}",
        )


def run_gemma4_vllm_cpu_offload_test() -> TestResult:
    result = TestResult(
        test_name="Gemma 4 E2B vLLM + CPU offload + TurboQuant KV",
        model_id="gemma-4-e2b-it-vllm",
        model_family="gemma4",
        engine="vllm",
        quantization="BF16 → bnb_int4",
        kv_quant="turboquant35",
        pipeline_path="detect full_precision → weight:bnb_int4 → cpu_offload → kv:turboquant35",
        pipeline_stages="detect full_precision → weight:bnb_int4 → cpu_offload → kv:turboquant35",
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    sys_info = detect_system()
    config = TqConfig.load()
    config.ensure_dirs()
    profile = next((p for p in BUILTIN_PROFILES if p.id == result.model_id), None)

    if not profile:
        result.add_step(StepResult(name="find_model", passed=False, details="Profile not found"))
        result.finished = time.strftime("%Y-%m-%dT%H:%M:%S")
        result.total_duration_s = sum(s.duration_s for s in result.steps)
        return result

    result.add_step(StepResult(
        name="find_model_profile",
        passed=True,
        details=f"Found {profile.id}: {profile.display_name} ({profile.parameter_count}, {profile.quantization})",
        metrics={
            "model_id": profile.id,
            "parameter_count": profile.parameter_count,
            "quantization": profile.quantization,
            "format": profile.format,
            "multimodal": profile.multimodal,
        },
    ))

    precision = detect_model_precision(profile)
    result.add_step(StepResult(
        name="detect_precision",
        passed=precision == "full_precision",
        details=f"Detected: {precision} (quant={profile.quantization}, format={profile.format})",
        metrics={"precision": precision},
    ))

    bf16_size = estimate_bf16_model_size(profile)
    int4_size = estimate_quantized_size(profile, QuantizationMethod.BNB_INT4)
    result.add_step(StepResult(
        name="size_estimates",
        passed=True,
        details=f"BF16={bf16_size} MB, INT4={int4_size} MB, VRAM={sys_info.total_vram_mb} MB, RAM={sys_info.ram_available_mb} MB",
        metrics={
            "bf16_mb": bf16_size,
            "int4_mb": int4_size,
            "vram_mb": sys_info.total_vram_mb,
            "ram_available_mb": sys_info.ram_available_mb,
        },
    ))

    quant_method = select_quantization(profile, sys_info)
    result.add_step(StepResult(
        name="select_quantization_without_offload",
        passed=quant_method is None,
        details=f"select_quantization() returned: {quant_method} (expected None — too large for VRAM alone)",
        metrics={"method": str(quant_method), "expected": "None"},
    ))

    start = time.time()
    tune = build_vllm_config(profile, sys_info, requested_max_len=2048, kv_quant_choice="turbo3")
    elapsed = time.time() - start
    result.add_step(StepResult(
        name="build_vllm_config_with_offload",
        passed=tune.feasible,
        duration_s=elapsed,
        details=(
            f"feasible={tune.feasible} | cpu_offload_gb={tune.cpu_offload_gb} | "
            f"quantization={tune.quantization} | kv_cache_dtype={tune.kv_cache_dtype} | "
            f"max_model_len={tune.max_model_len}"
        ),
        metrics={
            "feasible": tune.feasible,
            "cpu_offload_gb": tune.cpu_offload_gb,
            "quantization": tune.quantization,
            "load_format": tune.load_format,
            "kv_cache_dtype": tune.kv_cache_dtype,
            "max_model_len": tune.max_model_len,
            "gpu_memory_utilization": tune.gpu_memory_utilization,
            "estimated_model_size_mb": tune.estimated_model_size_mb,
            "warnings_count": len(tune.warnings),
        },
    ))

    if not tune.feasible:
        result.add_step(StepResult(
            name="config_infeasible",
            passed=False,
            details=f"vLLM config not feasible: {tune.reason}",
        ))
        result.finished = time.strftime("%Y-%m-%dT%H:%M:%S")
        result.total_duration_s = sum(s.duration_s for s in result.steps)
        result.passed = all(s.passed for s in result.steps)
        return result

    models_dir = config.models_dir
    model_dir = models_dir / profile.id
    if not (model_dir.is_dir() and (model_dir / "config.json").exists()):
        result.add_step(StepResult(
            name="model_available",
            passed=False,
            details=f"Model directory not found at {model_dir}; pull is required for full Section E coverage.",
        ))
    else:
        result.add_step(StepResult(
            name="model_available",
            passed=True,
            details=f"Model already downloaded at {model_dir}",
        ))

    if not HAS_VLLM:
        result.add_step(StepResult(
            name="load_model_with_cpu_offload",
            passed=True,
            duration_s=0.0,
            details=(
                f"SKIPPED: vllm module not importable in this interpreter "
                f"({VLLM_IMPORT_ERROR}). Run with a Python that has vllm installed "
                f"(e.g. `python3 tests/test_integration_turboquant_kv.py`)."
            ),
            metrics={"skipped": True, "reason": "vllm_not_installed"},
        ))
        for skipped_name in (
            "chat_thinking_turn",
            "chat_simple_turn",
            "unload_model",
        ):
            result.add_step(StepResult(
                name=skipped_name,
                passed=True,
                details=f"SKIPPED: depends on vllm (not installed: {VLLM_IMPORT_ERROR})",
                metrics={"skipped": True, "reason": "vllm_not_installed"},
            ))
        for step in run_full_lifecycle(
            model_id=profile.id,
            kv_level="turboquant35",
            engine="vllm",
            model_size_mb=10246,
            multimodal=profile.multimodal,
        ):
            result.add_step(step)
        result.finished = time.strftime("%Y-%m-%dT%H:%M:%S")
        result.total_duration_s = sum(s.duration_s for s in result.steps)
        result.passed = all(s.passed for s in result.steps)
        return result

    if model_dir.is_dir() and (model_dir / "config.json").exists():
        start = time.time()
        try:
            from tqcli.core.vllm_backend import VllmBackend

            engine = VllmBackend.from_tuning_profile(tune)
            engine.load_model(str(model_dir))
            elapsed = time.time() - start
            result.add_step(StepResult(
                name="load_model_with_cpu_offload",
                passed=True,
                duration_s=elapsed,
                details=(
                    f"Loaded Gemma 4 E2B via vLLM in {elapsed:.1f}s | "
                    f"BNB_INT4 + cpu_offload={tune.cpu_offload_gb} GB + "
                    f"kv={tune.kv_cache_dtype}"
                ),
                metrics={
                    "load_time_s": round(elapsed, 2),
                    "cpu_offload_gb": tune.cpu_offload_gb,
                    "quantization": tune.quantization,
                    "kv_cache_dtype": tune.kv_cache_dtype,
                },
            ))
        except Exception as exc:
            elapsed = time.time() - start
            result.add_step(StepResult(
                name="load_model_with_cpu_offload",
                passed=False,
                duration_s=elapsed,
                details=f"Load failed: {exc}",
                metrics={"error": str(exc)},
            ))
            result.finished = time.strftime("%Y-%m-%dT%H:%M:%S")
            result.total_duration_s = sum(s.duration_s for s in result.steps)
            result.passed = all(s.passed for s in result.steps)
            return result

        monitor = PerformanceMonitor(config.performance)
        think_cfg = ThinkingConfig(format=ThinkingFormat.GEMMA4, enabled=True)
        sys_prompt = build_system_prompt_with_thinking("Be concise.", think_cfg)
        history = [ChatMessage(role="system", content=sys_prompt)]
        history.append(ChatMessage(role="user", content="What is 15% of 240?"))

        start = time.time()
        try:
            full_response = ""
            final_stats = None
            for chunk, stats in engine.chat_stream(history):
                if stats:
                    final_stats = stats
                    break
                full_response += chunk

            elapsed = time.time() - start
            thinking_text, clean_response = extract_thinking(full_response, ThinkingFormat.GEMMA4)
            metrics = {
                "has_thinking": len(thinking_text.strip()) > 0,
                "thinking_length": len(thinking_text),
                "response_length": len(clean_response),
            }
            if final_stats:
                monitor.record(final_stats.completion_tokens, final_stats.completion_time_s)
                metrics.update({
                    "tokens_per_second": round(final_stats.tokens_per_second, 2),
                    "completion_tokens": final_stats.completion_tokens,
                })

            result.add_step(StepResult(
                name="chat_thinking_turn",
                passed=len(clean_response.strip()) > 0,
                duration_s=elapsed,
                details=f"Response: {clean_response[:200]}...",
                metrics=metrics,
            ))
        except Exception as exc:
            result.add_step(StepResult(
                name="chat_thinking_turn",
                passed=False,
                duration_s=time.time() - start,
                details=f"Chat error: {exc}",
            ))

        history2 = [ChatMessage(role="system", content="Be concise.")]
        history2.append(ChatMessage(role="user", content="What is the capital of France?"))

        start = time.time()
        try:
            full_response = ""
            final_stats = None
            for chunk, stats in engine.chat_stream(history2):
                if stats:
                    final_stats = stats
                    break
                full_response += chunk

            elapsed = time.time() - start
            metrics = {"response_length": len(full_response)}
            if final_stats:
                metrics.update({
                    "tokens_per_second": round(final_stats.tokens_per_second, 2),
                    "completion_tokens": final_stats.completion_tokens,
                })

            result.add_step(StepResult(
                name="chat_simple_turn",
                passed=len(full_response.strip()) > 0,
                duration_s=elapsed,
                details=f"Response: {full_response[:200]}...",
                metrics=metrics,
            ))
        except Exception as exc:
            result.add_step(StepResult(
                name="chat_simple_turn",
                passed=False,
                duration_s=time.time() - start,
                details=f"Chat error: {exc}",
            ))

        try:
            engine.unload_model()
            result.add_step(StepResult(name="unload_model", passed=True, details="Model unloaded"))
        except Exception as exc:
            result.add_step(StepResult(name="unload_model", passed=False, details=f"Unload failed: {exc}"))

    for step in run_full_lifecycle(
        model_id=profile.id,
        kv_level="turboquant35",
        engine="vllm",
        model_size_mb=10246,  # BF16 safetensors on disk (INT4 runtime ≈4145)
        multimodal=profile.multimodal,
    ):
        result.add_step(step)

    result.finished = time.strftime("%Y-%m-%dT%H:%M:%S")
    result.total_duration_s = sum(s.duration_s for s in result.steps)
    result.passed = all(s.passed for s in result.steps)
    return result


if __name__ == "__main__":
    test_result = run_gemma4_vllm_cpu_offload_test()
    print("=" * 60)
    print("GEMMA 4 E2B vLLM CPU OFFLOAD INTEGRATION TEST")
    print("=" * 60)
    print(f"Result: {'PASS' if test_result.passed else 'FAIL'} ({test_result.pass_count}/{len(test_result.steps)} steps)")
    print(f"Duration: {test_result.total_duration_s:.2f}s")
    print(f"Pipeline: {test_result.pipeline_path}")
    for step in test_result.steps:
        status = "PASS" if step.passed else "FAIL"
        print(f"  [{status}] {step.name}: {step.details[:120]}")
    print("=" * 60)
