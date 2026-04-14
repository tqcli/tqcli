"""Hardware-aware vLLM configuration builder.

Automatically tunes vLLM parameters based on detected GPU VRAM, model size,
and model capabilities.  Based on the official vLLM Gemma 4 recipe:
https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tqcli.core.model_registry import ModelProfile
from tqcli.core.system_info import SystemInfo


@dataclass
class VllmTuningProfile:
    """Tuned vLLM parameters for a specific model + hardware combination."""

    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    enforce_eager: bool = False
    quantization: str | None = None
    kv_cache_dtype: str = "auto"
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True
    # Multimodal settings (Gemma 4)
    mm_processor_kwargs: dict = field(default_factory=dict)
    limit_mm_per_prompt: dict = field(default_factory=dict)
    # Gemma 4 thinking mode
    reasoning_parser: str = ""
    warnings: list[str] = field(default_factory=list)
    feasible: bool = True
    reason: str = ""


# CUDA context overhead per GPU in MB.  Measured from vLLM's reported
# free memory vs total: on WSL2 ~810 MB, on bare-metal Linux ~400 MB.
_CUDA_CONTEXT_OVERHEAD_MB = 400

# Minimum KV cache memory in MB to be usable (enough for ~128 tokens)
_MIN_KV_CACHE_MB = 30


def build_vllm_config(
    model: ModelProfile,
    sys_info: SystemInfo,
    requested_max_len: int | None = None,
) -> VllmTuningProfile:
    """Build a hardware-tuned vLLM configuration for the given model.

    Args:
        model: The model profile to configure for.
        sys_info: Detected system hardware info.
        requested_max_len: Desired context length (auto-tuned if None).

    Returns:
        A VllmTuningProfile with all parameters set.
    """
    profile = VllmTuningProfile()
    total_vram_mb = sys_info.total_vram_mb
    warnings = []

    if total_vram_mb <= 0:
        profile.feasible = False
        profile.reason = "No GPU detected; vLLM requires NVIDIA GPU"
        return profile

    # ── Step 1: Estimate model weight memory ──────────────────────────
    # Use min_vram_mb from registry as a proxy for model weight footprint
    model_weight_mb = model.min_vram_mb * 0.85  # weights are ~85% of min_vram

    # ── Step 2: Calculate usable VRAM after CUDA context ──────────────
    overhead_mb = _CUDA_CONTEXT_OVERHEAD_MB
    if sys_info.is_wsl:
        overhead_mb = 810  # WSL2 takes more (measured: ~807 MB on RTX A2000)
    usable_vram_mb = total_vram_mb - overhead_mb

    # ── Step 3: Determine gpu_memory_utilization ──────────────────────
    # vLLM checks: util * total_vram <= free_vram_after_cuda_init
    if usable_vram_mb <= 0:
        profile.feasible = False
        profile.reason = f"Insufficient VRAM: {total_vram_mb} MB total, ~{overhead_mb} MB CUDA overhead"
        return profile

    max_util = usable_vram_mb / total_vram_mb
    profile.gpu_memory_utilization = min(round(max_util, 2), 0.95)

    # ── Step 4: Determine if we need enforce_eager ────────────────────
    # torch.compile + CUDA graphs add 500 MB - 2 GB overhead.
    # On GPUs < 8 GB, this often pushes us OOM.
    if total_vram_mb < 8000:
        profile.enforce_eager = True
        warnings.append("enforce_eager=True (saves ~1 GB VRAM on small GPUs)")

    # ── Step 5: Estimate available KV cache memory ────────────────────
    # vLLM runtime overhead includes activation memory, NCCL buffers,
    # weight unpacking buffers, and internal allocations (~500-700 MB).
    vllm_runtime_overhead_mb = 600
    model_loaded_mb = model_weight_mb + vllm_runtime_overhead_mb
    if not profile.enforce_eager:
        model_loaded_mb += 500  # torch.compile + CUDA graph overhead
    available_for_kv_mb = (profile.gpu_memory_utilization * total_vram_mb) - model_loaded_mb

    if available_for_kv_mb < _MIN_KV_CACHE_MB:
        profile.feasible = False
        profile.reason = (
            f"Insufficient VRAM for KV cache: model needs ~{model_loaded_mb:.0f} MB, "
            f"only {profile.gpu_memory_utilization * total_vram_mb:.0f} MB usable "
            f"({total_vram_mb} MB total, {overhead_mb} MB overhead)"
        )
        return profile

    # ── Step 6: Use FP8 KV cache if beneficial ───────────────────────
    # FP8 KV cache halves KV memory, doubling effective context length.
    # Only available on Ampere+ GPUs (compute capability >= 8.0).
    gpu_cc = ""
    if sys_info.gpus:
        gpu_cc = sys_info.gpus[0].compute_capability
    cc_major = 0
    if gpu_cc:
        try:
            cc_major = int(gpu_cc.split(".")[0])
        except (ValueError, IndexError):
            pass

    if cc_major >= 9 and available_for_kv_mb < 500:  # FP8 native on Hopper+ (cc >= 9.0)
        profile.kv_cache_dtype = "fp8"
        available_for_kv_mb *= 2  # FP8 doubles effective KV capacity
        warnings.append("kv_cache_dtype=fp8 (doubles KV cache capacity)")

    # ── Step 7: Calculate max_model_len ───────────────────────────────
    # Rough estimate: each token of KV cache uses ~0.14 MB per billion params
    # for a dense transformer with FP16 KV.
    param_billions = _parse_param_count(model.parameter_count)
    kv_per_token_mb = 0.14 * param_billions
    if profile.kv_cache_dtype == "fp8":
        kv_per_token_mb *= 0.5

    max_tokens_from_kv = int(available_for_kv_mb / max(kv_per_token_mb, 0.01))
    max_tokens_from_kv = max(max_tokens_from_kv, 128)  # absolute minimum

    if requested_max_len:
        profile.max_model_len = min(requested_max_len, max_tokens_from_kv, model.context_length)
    else:
        # Auto-select: use available KV budget, capped at model max
        profile.max_model_len = min(max_tokens_from_kv, model.context_length, 8192)

    if profile.max_model_len < 256:
        warnings.append(f"Very short context ({profile.max_model_len} tokens) due to limited VRAM")

    # ── Step 8: Quantization ──────────────────────────────────────────
    if model.quantization == "AWQ":
        profile.quantization = "awq_marlin"
    elif model.quantization in ("GPTQ", "FP8"):
        profile.quantization = model.quantization.lower()

    # ── Step 9: Multimodal configuration (Gemma 4) ────────────────────
    if model.multimodal and model.family == "gemma4":
        # Gemma 4 SigLIP vision: 70/140/280/560/1120 tokens per image
        # Use smaller budget on low-VRAM
        if total_vram_mb < 12000:
            vision_budget = 70
        elif total_vram_mb < 24000:
            vision_budget = 280
        else:
            vision_budget = 560

        profile.mm_processor_kwargs = {"max_soft_tokens": vision_budget}
        profile.limit_mm_per_prompt = {"image": 2, "audio": 1}
        if total_vram_mb < 8000:
            profile.limit_mm_per_prompt = {"image": 1, "audio": 0}
            warnings.append("Multimodal limited to 1 image, no audio (low VRAM)")

    # ── Step 10: Thinking mode ────────────────────────────────────────
    if model.supports_thinking and model.family == "gemma4":
        profile.reasoning_parser = "gemma4"

    profile.warnings = warnings
    return profile


def _parse_param_count(param_str: str) -> float:
    """Parse '4.5B', '31B', etc. into float billions."""
    s = param_str.upper().replace("B", "").strip()
    try:
        return float(s)
    except ValueError:
        return 4.0  # default assumption
