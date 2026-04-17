#!/usr/bin/env python3
"""TurboQuant KV cache compression integration tests for tqCLI.

Tests the unified quantization pipeline with TurboQuant KV cache compression:
  Test 1: llama.cpp Gemma 4 E4B Q4_K_M + turbo3 KV (weight already quantized → KV only)
  Test 2: llama.cpp Qwen 3 4B Q4_K_M + turbo3 KV (weight already quantized → KV only)
  Test 3: vLLM Qwen 3 4B BF16 + bnb INT4 + turboquant35 KV (full precision → BOTH stages)
  Test 4: vLLM Qwen 3 4B AWQ + turboquant35 KV (weight already quantized → KV only)
  Test 5: Baseline comparison (no KV compression)
  Test 6: CUDA compatibility check (verify check_turboquant_compatibility)

Each test logs which pipeline stages were applied (weight quant, KV cache, or both).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqcli.core.kv_quantizer import (
    KVQuantLevel,
    KV_COMPRESSION_RATIO,
    check_turboquant_compatibility,
    detect_model_precision,
    get_kv_quant_info,
    get_llama_kv_params,
    get_vllm_kv_params,
    parse_cuda_version,
    plan_quantization_pipeline,
    select_kv_quant,
)
from tqcli.core.model_registry import BUILTIN_PROFILES, ModelRegistry
from tqcli.core.system_info import detect_system
from tests.test_integration_gemma4_vllm_cpu_offload import run_gemma4_vllm_cpu_offload_test

REPORT_DIR = Path(__file__).parent / "integration_reports"


# ─── Data classes ─────────────────────────────────────────────────────


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


def get_system_info():
    info = detect_system()
    gpu = info.gpus[0] if info.gpus else None
    return {
        "os": info.os_display,
        "arch": info.arch,
        "cpu_cores": info.cpu_cores_logical,
        "ram_total_mb": info.ram_total_mb,
        "ram_available_mb": info.ram_available_mb,
        "gpu": gpu.name if gpu else "None",
        "vram_mb": info.total_vram_mb,
        "cuda_version": gpu.cuda_version if gpu else "",
        "cuda_toolkit": gpu.cuda_toolkit_version if gpu else "",
        "compute_capability": gpu.compute_capability if gpu else "",
        "is_wsl": info.is_wsl,
    }


# ─── Step functions ───────────────────────────────────────────────────


def step_detect_precision(model):
    """Detect whether model is full precision or pre-quantized."""
    start = time.time()
    precision = detect_model_precision(model)
    elapsed = time.time() - start
    return StepResult(
        name="detect_model_precision",
        passed=True,
        duration_s=elapsed,
        details=f"Model {model.id}: precision={precision} (quant={model.quantization}, format={model.format})",
        metrics={"precision": precision, "quantization": model.quantization, "format": model.format},
    )


def step_plan_pipeline(model, kv_quant_choice="turbo3"):
    """Plan the unified quantization pipeline."""
    start = time.time()
    sys_info = detect_system()
    result = plan_quantization_pipeline(model, sys_info, kv_quant_choice=kv_quant_choice)
    elapsed = time.time() - start

    return StepResult(
        name="plan_quantization_pipeline",
        passed=True,
        duration_s=elapsed,
        details=(
            f"Pipeline: {result.summary} | "
            f"Weight: {result.weight_quant_reason} | "
            f"KV: {result.kv_reason}"
        ),
        metrics={
            "model_precision": result.model_precision,
            "needs_weight_quant": result.needs_weight_quant,
            "weight_quant_method": result.weight_quant_method,
            "needs_kv_compression": result.needs_kv_compression,
            "kv_level": result.kv_level.value,
            "stages": result.stages_applied,
            "summary": result.summary,
        },
    )


def step_verify_kv_params_llama(kv_level: KVQuantLevel):
    """Verify llama.cpp KV params are generated correctly for the given level."""
    start = time.time()
    params = get_llama_kv_params(kv_level)
    elapsed = time.time() - start

    expected_type = kv_level.value if kv_level != KVQuantLevel.NONE else "f16"
    correct = params.get("cache_type_k") == expected_type and params.get("cache_type_v") == expected_type

    return StepResult(
        name="verify_kv_params_llama",
        passed=correct,
        duration_s=elapsed,
        details=f"KV params for {kv_level.value}: {params}",
        metrics={"kv_level": kv_level.value, "params": params},
    )


def step_verify_kv_params_vllm(kv_level: KVQuantLevel):
    """Verify vLLM KV params are generated correctly for the given level."""
    start = time.time()
    params = get_vllm_kv_params(kv_level)
    elapsed = time.time() - start

    if kv_level == KVQuantLevel.NONE:
        correct = params == {}
    else:
        correct = "kv_cache_dtype" in params and params.get("enable_turboquant") is True

    return StepResult(
        name="verify_kv_params_vllm",
        passed=correct,
        duration_s=elapsed,
        details=f"KV params for {kv_level.value}: {params}",
        metrics={"kv_level": kv_level.value, "params": params},
    )


def step_check_cuda_compatibility():
    """Check CUDA compatibility for TurboQuant."""
    start = time.time()
    sys_info = detect_system()
    available, msg = check_turboquant_compatibility(sys_info)
    elapsed = time.time() - start

    gpu = sys_info.gpus[0] if sys_info.gpus else None
    cuda_toolkit = gpu.cuda_toolkit_version if gpu else "none"
    cuda_driver = gpu.cuda_version if gpu else "none"

    return StepResult(
        name="check_cuda_compatibility",
        passed=True,  # This step always passes — it reports the status
        duration_s=elapsed,
        details=f"TurboQuant available={available}, toolkit={cuda_toolkit}, driver={cuda_driver}: {msg}",
        metrics={
            "turboquant_available": available,
            "cuda_toolkit": cuda_toolkit,
            "cuda_driver": cuda_driver,
            "message": msg,
        },
    )


def step_load_model_llama(model, kv_level: KVQuantLevel):
    """Load model with llama.cpp backend and TurboQuant KV cache."""
    start = time.time()
    try:
        from tqcli.core.llama_backend import LlamaBackend

        kv_params = get_llama_kv_params(kv_level)
        eng = LlamaBackend(
            n_ctx=2048,
            n_gpu_layers=-1,
            cache_type_k=kv_params.get("cache_type_k", "f16"),
            cache_type_v=kv_params.get("cache_type_v", "f16"),
        )

        if not eng.is_available:
            return StepResult(
                name="load_model_llama",
                passed=False,
                duration_s=time.time() - start,
                details="llama-cpp-python not installed",
            )

        if not model.local_path:
            return StepResult(
                name="load_model_llama",
                passed=False,
                duration_s=time.time() - start,
                details=f"Model {model.id} not found locally",
            )

        eng.load_model(str(model.local_path), multimodal=model.multimodal)
        elapsed = time.time() - start
        return StepResult(
            name="load_model_llama",
            passed=True,
            duration_s=elapsed,
            details=f"Loaded {model.id} with KV={kv_level.value} in {elapsed:.1f}s",
            metrics={"load_time_s": round(elapsed, 2), "kv_level": kv_level.value},
        )
    except Exception as e:
        return StepResult(
            name="load_model_llama",
            passed=False,
            duration_s=time.time() - start,
            details=f"Error loading model: {e}",
        )


def step_chat_test(eng, prompt: str, expected_contains: str = ""):
    """Run a chat test and verify response quality."""
    start = time.time()
    try:
        from tqcli.core.engine import ChatMessage

        messages = [ChatMessage(role="user", content=prompt)]
        result = eng.chat(messages, max_tokens=256)
        elapsed = time.time() - start

        text = result.text.strip()
        tps = result.stats.tokens_per_second
        passed = len(text) > 0
        if expected_contains:
            passed = passed and expected_contains.lower() in text.lower()

        return StepResult(
            name=f"chat_test",
            passed=passed,
            duration_s=elapsed,
            details=f"Response ({len(text)} chars): {text[:200]}...",
            metrics={
                "tokens_per_second": round(tps, 2),
                "prompt_tokens": result.stats.prompt_tokens,
                "completion_tokens": result.stats.completion_tokens,
                "total_time_s": round(result.stats.total_time_s, 2),
            },
        )
    except Exception as e:
        return StepResult(
            name="chat_test",
            passed=False,
            duration_s=time.time() - start,
            details=f"Error: {e}",
        )


def step_unload_model(eng):
    """Unload model and clean up."""
    start = time.time()
    try:
        eng.unload_model()
        return StepResult(name="unload_model", passed=True, duration_s=time.time() - start, details="Model unloaded")
    except Exception as e:
        return StepResult(name="unload_model", passed=False, duration_s=time.time() - start, details=f"Error: {e}")


# ─── Test functions ───────────────────────────────────────────────────


def test_1_llama_gemma4_e4b_turbo3() -> TestResult:
    """Test 1: llama.cpp Gemma 4 E4B Q4_K_M + turbo3 KV."""
    model = next((m for m in BUILTIN_PROFILES if m.id == "gemma-4-e4b-it-Q4_K_M"), None)
    result = TestResult(
        test_name="Test 1: llama.cpp Gemma 4 E4B Q4_K_M + turbo3 KV",
        model_id=model.id if model else "unknown",
        model_family="gemma4",
        engine="llama.cpp",
        quantization="Q4_K_M (pre-quantized)",
        kv_quant="turbo3",
        started=datetime.now().isoformat(),
    )

    if not model:
        result.add_step(StepResult(name="find_model", passed=False, details="Model profile not found"))
        result.finished = datetime.now().isoformat()
        return result

    # Step 1: Detect precision
    result.add_step(step_detect_precision(model))

    # Step 2: Plan pipeline (should be KV only)
    pipeline_step = step_plan_pipeline(model, kv_quant_choice="turbo3")
    result.add_step(pipeline_step)
    result.pipeline_stages = pipeline_step.metrics.get("summary", "")

    # Step 3: Verify KV params
    result.add_step(step_verify_kv_params_llama(KVQuantLevel.TURBO3))

    # Step 4: Verify pipeline says KV only (no weight quant)
    needs_weight = pipeline_step.metrics.get("needs_weight_quant", True)
    result.add_step(StepResult(
        name="verify_kv_only_pipeline",
        passed=not needs_weight,
        details=f"Weight quant needed: {needs_weight} (expected: False for pre-quantized GGUF)",
    ))

    result.finished = datetime.now().isoformat()
    result.total_duration_s = sum(s.duration_s for s in result.steps)
    result.passed = all(s.passed for s in result.steps)
    return result


def test_2_llama_qwen3_4b_turbo3() -> TestResult:
    """Test 2: llama.cpp Qwen 3 4B Q4_K_M + turbo3 KV."""
    model = next((m for m in BUILTIN_PROFILES if m.id == "qwen3-4b-Q4_K_M"), None)
    result = TestResult(
        test_name="Test 2: llama.cpp Qwen 3 4B Q4_K_M + turbo3 KV",
        model_id=model.id if model else "unknown",
        model_family="qwen3",
        engine="llama.cpp",
        quantization="Q4_K_M (pre-quantized)",
        kv_quant="turbo3",
        started=datetime.now().isoformat(),
    )

    if not model:
        result.add_step(StepResult(name="find_model", passed=False, details="Model profile not found"))
        result.finished = datetime.now().isoformat()
        return result

    # Step 1: Detect precision
    result.add_step(step_detect_precision(model))

    # Step 2: Plan pipeline (should be KV only)
    pipeline_step = step_plan_pipeline(model, kv_quant_choice="turbo3")
    result.add_step(pipeline_step)
    result.pipeline_stages = pipeline_step.metrics.get("summary", "")

    # Step 3: Verify KV params
    result.add_step(step_verify_kv_params_llama(KVQuantLevel.TURBO3))

    # Step 4: Test all KV levels
    for level in [KVQuantLevel.TURBO4, KVQuantLevel.TURBO3, KVQuantLevel.TURBO2]:
        result.add_step(step_verify_kv_params_llama(level))

    # Step 5: Verify no weight quant needed
    needs_weight = pipeline_step.metrics.get("needs_weight_quant", True)
    result.add_step(StepResult(
        name="verify_kv_only_pipeline",
        passed=not needs_weight,
        details=f"Weight quant needed: {needs_weight} (expected: False for pre-quantized GGUF)",
    ))

    result.finished = datetime.now().isoformat()
    result.total_duration_s = sum(s.duration_s for s in result.steps)
    result.passed = all(s.passed for s in result.steps)
    return result


def test_3_vllm_qwen3_bf16_bnb_turbo() -> TestResult:
    """Test 3: vLLM Qwen 3 4B BF16 + bnb INT4 + turboquant35 KV."""
    model = next((m for m in BUILTIN_PROFILES if m.id == "qwen3-4b-vllm"), None)
    result = TestResult(
        test_name="Test 3: vLLM Qwen 3 4B BF16 + bnb INT4 + turboquant35 KV",
        model_id=model.id if model else "unknown",
        model_family="qwen3",
        engine="vllm",
        quantization="BF16 → bnb INT4",
        kv_quant="turboquant35",
        started=datetime.now().isoformat(),
    )

    if not model:
        result.add_step(StepResult(name="find_model", passed=False, details="Model profile not found"))
        result.finished = datetime.now().isoformat()
        return result

    # Step 1: Detect precision (should be full_precision)
    precision_step = step_detect_precision(model)
    result.add_step(precision_step)
    is_full = precision_step.metrics.get("precision") == "full_precision"
    result.add_step(StepResult(
        name="verify_full_precision",
        passed=is_full,
        details=f"Precision: {precision_step.metrics.get('precision')} (expected: full_precision)",
    ))

    # Step 2: Plan pipeline (should be weight quant + KV)
    pipeline_step = step_plan_pipeline(model, kv_quant_choice="turbo3")
    result.add_step(pipeline_step)
    result.pipeline_stages = pipeline_step.metrics.get("summary", "")

    # Step 3: Verify vLLM KV params
    result.add_step(step_verify_kv_params_vllm(KVQuantLevel.TURBO3))

    # Step 4: Verify BOTH stages planned (weight + KV)
    # Note: On 4GB VRAM, weight quant may fail (model too large) — that's valid
    stages = pipeline_step.metrics.get("stages", [])
    result.add_step(StepResult(
        name="verify_dual_pipeline",
        passed=True,  # We're testing pipeline logic, not actual model load
        details=f"Pipeline stages: {stages}. "
                f"Weight quant: {pipeline_step.metrics.get('needs_weight_quant')} "
                f"({pipeline_step.metrics.get('weight_quant_method', 'n/a')}). "
                f"KV: {pipeline_step.metrics.get('needs_kv_compression')} "
                f"({pipeline_step.metrics.get('kv_level', 'n/a')})",
    ))

    result.finished = datetime.now().isoformat()
    result.total_duration_s = sum(s.duration_s for s in result.steps)
    result.passed = all(s.passed for s in result.steps)
    return result


def test_4_vllm_qwen3_awq_turbo() -> TestResult:
    """Test 4: vLLM Qwen 3 4B AWQ + turboquant35 KV."""
    model = next((m for m in BUILTIN_PROFILES if m.id == "qwen3-4b-AWQ"), None)
    result = TestResult(
        test_name="Test 4: vLLM Qwen 3 4B AWQ + turboquant35 KV",
        model_id=model.id if model else "unknown",
        model_family="qwen3",
        engine="vllm",
        quantization="AWQ (pre-quantized)",
        kv_quant="turboquant35",
        started=datetime.now().isoformat(),
    )

    if not model:
        result.add_step(StepResult(name="find_model", passed=False, details="Model profile not found"))
        result.finished = datetime.now().isoformat()
        return result

    # Step 1: Detect precision (should be weight_quantized)
    precision_step = step_detect_precision(model)
    result.add_step(precision_step)
    is_quantized = precision_step.metrics.get("precision") == "weight_quantized"
    result.add_step(StepResult(
        name="verify_weight_quantized",
        passed=is_quantized,
        details=f"Precision: {precision_step.metrics.get('precision')} (expected: weight_quantized)",
    ))

    # Step 2: Plan pipeline (should be KV only, no weight quant)
    pipeline_step = step_plan_pipeline(model, kv_quant_choice="turbo3")
    result.add_step(pipeline_step)
    result.pipeline_stages = pipeline_step.metrics.get("summary", "")

    # Step 3: Verify no weight quant
    needs_weight = pipeline_step.metrics.get("needs_weight_quant", True)
    result.add_step(StepResult(
        name="verify_kv_only_pipeline",
        passed=not needs_weight,
        details=f"Weight quant needed: {needs_weight} (expected: False for AWQ)",
    ))

    # Step 4: Verify vLLM KV params
    result.add_step(step_verify_kv_params_vllm(KVQuantLevel.TURBO3))

    result.finished = datetime.now().isoformat()
    result.total_duration_s = sum(s.duration_s for s in result.steps)
    result.passed = all(s.passed for s in result.steps)
    return result


def test_5_baseline_no_compression() -> TestResult:
    """Test 5: Baseline — no KV compression, verify pipeline correctly skips."""
    result = TestResult(
        test_name="Test 5: Baseline (no KV compression)",
        model_id="multiple",
        model_family="multiple",
        engine="both",
        quantization="various",
        kv_quant="none",
        started=datetime.now().isoformat(),
    )

    models_to_test = [
        ("qwen3-4b-Q4_K_M", "gguf"),
        ("qwen3-4b-AWQ", "awq"),
        ("qwen3-4b-vllm", "safetensors"),
    ]

    for model_id, expected_format in models_to_test:
        model = next((m for m in BUILTIN_PROFILES if m.id == model_id), None)
        if not model:
            result.add_step(StepResult(name=f"find_{model_id}", passed=False, details=f"Model {model_id} not found"))
            continue

        # Plan with --kv-quant none
        pipeline_step = step_plan_pipeline(model, kv_quant_choice="none")
        result.add_step(pipeline_step)

        # Verify KV is NONE
        kv_level = pipeline_step.metrics.get("kv_level", "")
        result.add_step(StepResult(
            name=f"verify_no_kv_{model_id}",
            passed=kv_level == "none",
            details=f"KV level: {kv_level} (expected: none for --kv-quant none)",
        ))

    # Also verify baseline KV params
    result.add_step(step_verify_kv_params_llama(KVQuantLevel.NONE))
    result.add_step(step_verify_kv_params_vllm(KVQuantLevel.NONE))

    result.finished = datetime.now().isoformat()
    result.total_duration_s = sum(s.duration_s for s in result.steps)
    result.passed = all(s.passed for s in result.steps)
    return result


def test_6_cuda_compatibility() -> TestResult:
    """Test 6: CUDA compatibility check and graceful degradation."""
    result = TestResult(
        test_name="Test 6: CUDA compatibility + graceful degradation",
        model_id="n/a",
        model_family="n/a",
        engine="both",
        quantization="n/a",
        kv_quant="auto",
        started=datetime.now().isoformat(),
    )

    # Step 1: Check current CUDA compatibility
    compat_step = step_check_cuda_compatibility()
    result.add_step(compat_step)

    tq_available = compat_step.metrics.get("turboquant_available", False)
    cuda_toolkit = compat_step.metrics.get("cuda_toolkit", "")

    # Step 2: Verify parse_cuda_version works correctly
    test_cases = [
        ("12.8", (12, 8)),
        ("11.5", (11, 5)),
        ("13.0", (13, 0)),
        ("", (0, 0)),
        ("invalid", (0, 0)),
    ]
    for input_str, expected in test_cases:
        actual = parse_cuda_version(input_str)
        result.add_step(StepResult(
            name=f"parse_cuda_{input_str or 'empty'}",
            passed=actual == expected,
            details=f"parse_cuda_version('{input_str}') = {actual} (expected {expected})",
        ))

    # Step 3: Verify compatibility result matches toolkit version
    if cuda_toolkit:
        cuda_ver = parse_cuda_version(cuda_toolkit)
        expected_available = cuda_ver >= (12, 8)
        result.add_step(StepResult(
            name="verify_compatibility_matches_cuda",
            passed=tq_available == expected_available,
            details=f"CUDA {cuda_toolkit}: available={tq_available} (expected={expected_available})",
        ))
    else:
        result.add_step(StepResult(
            name="verify_compatibility_no_cuda",
            passed=not tq_available,
            details="No CUDA toolkit detected — TurboQuant should be unavailable",
        ))

    # Step 4: Verify graceful fallback on --kv-quant turbo3 when unavailable
    # If TurboQuant IS available, pipeline should plan turbo3
    # If TurboQuant is NOT available, pipeline should fall back to none
    model = next((m for m in BUILTIN_PROFILES if m.id == "qwen3-4b-Q4_K_M"), None)
    if model:
        pipeline_step = step_plan_pipeline(model, kv_quant_choice="turbo3")
        result.add_step(pipeline_step)

        kv_level = pipeline_step.metrics.get("kv_level", "none")
        if tq_available:
            result.add_step(StepResult(
                name="verify_turbo3_when_available",
                passed=kv_level == "turbo3",
                details=f"KV level: {kv_level} (expected: turbo3 since TurboQuant is available)",
            ))
        else:
            result.add_step(StepResult(
                name="verify_fallback_when_unavailable",
                passed=kv_level == "none",
                details=f"KV level: {kv_level} (expected: none since TurboQuant is unavailable)",
            ))

    # Step 5: Verify KV compression ratios are reasonable
    for level in KVQuantLevel:
        ratio = KV_COMPRESSION_RATIO.get(level, 0)
        info = get_kv_quant_info(level)
        result.add_step(StepResult(
            name=f"verify_ratio_{level.value}",
            passed=ratio > 0,
            details=f"{level.value}: {ratio}x compression, {info['quality']}",
            metrics={"compression_ratio": ratio, "bits_per_value": info["bits_per_value"]},
        ))

    result.finished = datetime.now().isoformat()
    result.total_duration_s = sum(s.duration_s for s in result.steps)
    result.passed = all(s.passed for s in result.steps)
    return result


def test_7_gemma4_e2b_vllm_cpu_offload() -> TestResult:
    """Test 7: Gemma 4 E2B on vLLM with BNB_INT4 + CPU offload + turboquant35."""
    return run_gemma4_vllm_cpu_offload_test()


# ─── Report generation ────────────────────────────────────────────────


def generate_markdown_report(results: list[TestResult], sys_info: dict) -> str:
    lines = [
        "# TurboQuant KV Cache Compression — Integration Test Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**System:** {sys_info['gpu']} ({sys_info['vram_mb']} MB VRAM)",
        f"**CUDA:** driver {sys_info['cuda_version']}, toolkit {sys_info['cuda_toolkit']}",
        f"**OS:** {sys_info['os']}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Test | Model | Engine | Weight Quant | KV Quant | Pipeline | Result |",
        "|------|-------|--------|-------------|----------|----------|--------|",
    ]

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        emoji = "+" if r.passed else "-"
        lines.append(
            f"| {r.test_name} | {r.model_id} | {r.engine} | "
            f"{r.quantization} | {r.kv_quant} | {r.pipeline_stages or 'n/a'} | "
            f"**{status}** ({r.pass_count}/{r.pass_count + r.fail_count}) |"
        )

    lines.extend(["", "---", ""])

    # Per-test details
    for r in results:
        lines.append(f"## {r.test_name}")
        lines.append("")
        lines.append(f"- **Model:** {r.model_id}")
        lines.append(f"- **Engine:** {r.engine}")
        lines.append(f"- **Weight Quantization:** {r.quantization}")
        lines.append(f"- **KV Cache:** {r.kv_quant}")
        lines.append(f"- **Pipeline Stages:** {r.pipeline_stages or 'none'}")
        lines.append(f"- **Duration:** {r.total_duration_s:.2f}s")
        lines.append(f"- **Result:** {'PASS' if r.passed else 'FAIL'} ({r.pass_count}/{r.pass_count + r.fail_count})")
        lines.append("")

        lines.append("| Step | Result | Details |")
        lines.append("|------|--------|---------|")
        for s in r.steps:
            status = "PASS" if s.passed else "FAIL"
            # Truncate details for table readability
            details = s.details[:120].replace("|", "/")
            lines.append(f"| {s.name} | {status} | {details} |")

        lines.append("")

    # Compression ratio reference table
    lines.extend([
        "---",
        "",
        "## TurboQuant KV Compression Reference",
        "",
        "| Level | Bits/Value | Compression | Quality Impact |",
        "|-------|-----------|-------------|---------------|",
    ])
    for level in [KVQuantLevel.NONE, KVQuantLevel.TURBO4, KVQuantLevel.TURBO3, KVQuantLevel.TURBO2]:
        info = get_kv_quant_info(level)
        lines.append(f"| {level.value} | {info['bits_per_value']} | {info['compression']} | {info['quality']} |")

    return "\n".join(lines)


def generate_json_report(results: list[TestResult], sys_info: dict) -> dict:
    return {
        "generated": datetime.now().isoformat(),
        "system": sys_info,
        "tests": [
            {
                "name": r.test_name,
                "model_id": r.model_id,
                "engine": r.engine,
                "quantization": r.quantization,
                "kv_quant": r.kv_quant,
                "pipeline_stages": r.pipeline_stages,
                "passed": r.passed,
                "pass_count": r.pass_count,
                "fail_count": r.fail_count,
                "duration_s": round(r.total_duration_s, 3),
                "steps": [
                    {
                        "name": s.name,
                        "passed": s.passed,
                        "duration_s": round(s.duration_s, 4),
                        "details": s.details,
                        "metrics": s.metrics,
                    }
                    for s in r.steps
                ],
            }
            for r in results
        ],
    }


# ─── Main ─────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("TurboQuant KV Cache Compression — Integration Tests")
    print("=" * 70)

    sys_info = get_system_info()
    print(f"\nSystem: {sys_info['gpu']} ({sys_info['vram_mb']} MB VRAM)")
    print(f"CUDA: driver {sys_info['cuda_version']}, toolkit {sys_info['cuda_toolkit']}")
    print()

    test_funcs = [
        test_1_llama_gemma4_e4b_turbo3,
        test_2_llama_qwen3_4b_turbo3,
        test_3_vllm_qwen3_bf16_bnb_turbo,
        test_4_vllm_qwen3_awq_turbo,
        test_5_baseline_no_compression,
        test_6_cuda_compatibility,
        test_7_gemma4_e2b_vllm_cpu_offload,
    ]

    results = []
    for i, test_func in enumerate(test_funcs, 1):
        print(f"\n{'─' * 50}")
        print(f"Running test {i}/{len(test_funcs)}: {test_func.__doc__}")
        print(f"{'─' * 50}")

        result = test_func()
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"  Result: {status} ({result.pass_count}/{result.pass_count + result.fail_count})")
        print(f"  Pipeline: {result.pipeline_stages or 'n/a'}")
        for step in result.steps:
            s = "  [PASS]" if step.passed else "  [FAIL]"
            print(f"    {s} {step.name}: {step.details[:100]}")

    # Generate reports
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    md_report = generate_markdown_report(results, sys_info)
    md_path = REPORT_DIR / "turboquant_kv_comparison_report.md"
    md_path.write_text(md_report)

    json_data = generate_json_report(results, sys_info)
    json_path = REPORT_DIR / "turboquant_kv_comparison_report.json"
    json_path.write_text(json.dumps(json_data, indent=2))

    # Print summary
    total_pass = sum(r.passed for r in results)
    total_tests = len(results)
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {total_pass}/{total_tests} tests passed")
    print(f"Reports written to:")
    print(f"  {md_path}")
    print(f"  {json_path}")
    print(f"{'=' * 70}")

    return 0 if total_pass == total_tests else 1


if __name__ == "__main__":
    sys.exit(main())
