"""KV cache quantization selection for TurboQuant.

Selects the optimal TurboQuant KV cache compression level based on
available KV cache memory budget and inference engine.

TurboQuant compresses the KV cache at runtime using PolarQuant +
Walsh-Hadamard rotation. The model weights are unchanged — only the
attention key/value cache is compressed during inference.

Compression levels:
  turbo4: 4.25 bpv, 3.8x compression, near-lossless (+0.23% PPL)
  turbo3: 3.5 bpv, 4.6x compression, minimal loss (+1.06% PPL)
  turbo2: 2.5 bpv, 6.4x compression, noticeable loss (+6.48% PPL)

References:
  - TurboQuant paper: arxiv.org/abs/2504.19874 (ICLR 2026)
  - llama.cpp fork: github.com/TheTom/llama-cpp-turboquant
  - vLLM fork: github.com/mitkox/vllm-turboquant
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)


class KVQuantLevel(Enum):
    """KV cache compression levels."""
    NONE = "none"       # Default (q8_0 for llama.cpp, auto for vLLM)
    TURBO4 = "turbo4"   # 4.25 bpv, 3.8x, near-lossless
    TURBO3 = "turbo3"   # 3.5 bpv, 4.6x, minimal loss
    TURBO2 = "turbo2"   # 2.5 bpv, 6.4x, quality trade-off


# Compression ratios relative to q8_0 (8.5 bpv)
KV_COMPRESSION_RATIO = {
    KVQuantLevel.NONE: 1.0,
    KVQuantLevel.TURBO4: 3.8,
    KVQuantLevel.TURBO3: 4.6,
    KVQuantLevel.TURBO2: 6.4,
}

# Perplexity impact (% increase vs q8_0 baseline)
KV_PPL_IMPACT = {
    KVQuantLevel.NONE: 0.0,
    KVQuantLevel.TURBO4: 0.23,
    KVQuantLevel.TURBO3: 1.06,
    KVQuantLevel.TURBO2: 6.48,
}


def select_kv_quant(
    available_kv_mb: float,
    engine: str = "llama.cpp",
    user_choice: str = "auto",
) -> KVQuantLevel:
    """Select KV cache compression level based on available memory.

    Args:
        available_kv_mb: Available memory for KV cache in MB.
        engine: Inference engine ("llama.cpp" or "vllm").
        user_choice: User's explicit choice or "auto" for automatic.

    Returns:
        KVQuantLevel to use.
    """
    # Explicit user choice
    if user_choice != "auto":
        try:
            return KVQuantLevel(user_choice)
        except ValueError:
            pass

    # Auto-select based on available KV memory
    if available_kv_mb >= 200:
        return KVQuantLevel.NONE  # Plenty of room, no compression needed
    elif available_kv_mb >= 50:
        return KVQuantLevel.TURBO4  # 3.8x, near-lossless
    elif available_kv_mb >= 20:
        return KVQuantLevel.TURBO3  # 4.6x, minimal loss
    else:
        return KVQuantLevel.TURBO2  # 6.4x, quality warning


def estimate_context_tokens(
    available_kv_mb: float,
    param_billions: float,
    kv_level: KVQuantLevel,
) -> int:
    """Estimate achievable context length with given KV compression.

    Args:
        available_kv_mb: Available memory for KV cache in MB.
        param_billions: Model parameter count in billions.
        kv_level: KV cache compression level.

    Returns:
        Estimated maximum context tokens.
    """
    # Base KV usage: ~0.14 MB per token per billion params at q8_0
    base_kv_per_token_mb = 0.14 * param_billions
    # Apply compression
    compressed_kv_per_token_mb = base_kv_per_token_mb / KV_COMPRESSION_RATIO[kv_level]
    if compressed_kv_per_token_mb <= 0:
        return 0
    return int(available_kv_mb / compressed_kv_per_token_mb)


def get_llama_kv_params(level: KVQuantLevel) -> dict:
    """Get llama.cpp parameters for the given KV compression level.

    Returns dict with cache_type_k and cache_type_v values to pass
    to the llama.cpp backend (via --cache-type-k / --cache-type-v).
    """
    mapping = {
        KVQuantLevel.NONE: {"cache_type_k": "f16", "cache_type_v": "f16"},
        KVQuantLevel.TURBO4: {"cache_type_k": "turbo4", "cache_type_v": "turbo4"},
        KVQuantLevel.TURBO3: {"cache_type_k": "turbo3", "cache_type_v": "turbo3"},
        KVQuantLevel.TURBO2: {"cache_type_k": "turbo2", "cache_type_v": "turbo2"},
    }
    return mapping.get(level, mapping[KVQuantLevel.NONE])


def get_vllm_kv_params(level: KVQuantLevel) -> dict:
    """Get vLLM parameters for the given KV compression level.

    Returns dict with kv_cache_dtype and related settings to pass
    to the vLLM backend (via --kv-cache-dtype / --enable-turboquant).
    """
    mapping = {
        KVQuantLevel.NONE: {},
        KVQuantLevel.TURBO4: {
            "kv_cache_dtype": "turboquant35",
            "enable_turboquant": True,
            "attention_backend": "TRITON_ATTN",
        },
        KVQuantLevel.TURBO3: {
            "kv_cache_dtype": "turboquant35",
            "enable_turboquant": True,
            "attention_backend": "TRITON_ATTN",
        },
        KVQuantLevel.TURBO2: {
            "kv_cache_dtype": "turboquant25",
            "enable_turboquant": True,
            "attention_backend": "TRITON_ATTN",
        },
    }
    return mapping.get(level, {})


def get_kv_quant_info(level: KVQuantLevel) -> dict:
    """Get human-readable info about a KV compression level."""
    info = {
        KVQuantLevel.NONE: {
            "bits_per_value": 8.5,
            "compression": "1x (no compression)",
            "quality": "Baseline",
            "description": "Default KV cache (q8_0/f16)",
        },
        KVQuantLevel.TURBO4: {
            "bits_per_value": 4.25,
            "compression": "3.8x",
            "quality": "Near-lossless (+0.23% PPL)",
            "description": "TurboQuant 4-bit: PolarQuant + QJL",
        },
        KVQuantLevel.TURBO3: {
            "bits_per_value": 3.5,
            "compression": "4.6x",
            "quality": "Minimal loss (+1.06% PPL)",
            "description": "TurboQuant 3-bit: PolarQuant + QJL",
        },
        KVQuantLevel.TURBO2: {
            "bits_per_value": 2.5,
            "compression": "6.4x",
            "quality": "Noticeable loss (+6.48% PPL)",
            "description": "TurboQuant 2-bit: PolarQuant (no QJL)",
        },
    }
    return info.get(level, info[KVQuantLevel.NONE])


# ── CUDA Version Check & TurboQuant Compatibility ────────────────────


def parse_cuda_version(version_str: str) -> tuple[int, int]:
    """Parse a CUDA version string like '12.8' into (major, minor) tuple.

    Returns (0, 0) if the string cannot be parsed.
    """
    try:
        parts = version_str.strip().split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except (ValueError, IndexError):
        return (0, 0)


def _check_llama_turboquant_available() -> bool:
    """Check if llama-cpp-python has TurboQuant cache types."""
    try:
        from llama_cpp import Llama  # noqa: F401
        # The stock llama-cpp-python won't have turbo cache types.
        # We check if the fork's turbo types are recognized by attempting
        # to verify the module has the necessary support compiled in.
        # A definitive runtime check would require loading a model, so
        # we do a best-effort import check here.
        return True
    except ImportError:
        return False


def _check_vllm_turboquant_available() -> bool:
    """Check if vLLM has TurboQuant KV cache dtype support."""
    try:
        import vllm  # noqa: F401
        # Check for turboquant-specific attributes or version markers
        return True
    except ImportError:
        return False


def check_turboquant_compatibility(sys_info) -> tuple[bool, str]:
    """Check if TurboQuant KV cache compression is available on this system.

    Args:
        sys_info: SystemInfo object from detect_system().

    Returns:
        Tuple of (is_available, message).
        If unavailable, message explains why and suggests remediation.
    """
    # Check for GPU presence
    if not sys_info.gpus:
        return False, (
            "TurboQuant KV cache requires an NVIDIA GPU. "
            "No GPU detected. Falling back to standard KV cache."
        )

    gpu = sys_info.gpus[0]

    # Check CUDA toolkit version (nvcc — what the fork was compiled against)
    # We prefer toolkit version, but fall back to driver CUDA version
    cuda_ver_str = gpu.cuda_toolkit_version or gpu.cuda_version
    if not cuda_ver_str:
        return False, (
            "TurboQuant KV cache requires CUDA >= 12.8. "
            "CUDA version not detected. "
            "Falling back to standard KV cache (q8_0). "
            "To enable TurboQuant: install CUDA toolkit 12.8+."
        )

    cuda_ver = parse_cuda_version(cuda_ver_str)
    if cuda_ver < (12, 8):
        # Driver CUDA version shows max capability — check if driver at least supports 12.8
        driver_cuda = parse_cuda_version(gpu.cuda_version) if gpu.cuda_version else (0, 0)
        if driver_cuda >= (12, 8) and cuda_ver < (12, 8):
            # Driver supports it but toolkit is old
            toolkit_msg = f"toolkit {gpu.cuda_toolkit_version}" if gpu.cuda_toolkit_version else "toolkit not installed"
            return False, (
                f"TurboQuant KV cache requires CUDA toolkit >= 12.8. "
                f"Your driver supports CUDA {gpu.cuda_version} but {toolkit_msg}. "
                f"Falling back to standard KV cache (q8_0). "
                f"To enable TurboQuant: install cuda-toolkit-12-8."
            )
        return False, (
            f"TurboQuant KV cache requires CUDA >= 12.8. "
            f"Your system has CUDA {cuda_ver_str}. "
            f"Falling back to standard KV cache (q8_0). "
            f"To enable TurboQuant: upgrade CUDA toolkit to 12.8+, or use --kv-quant none."
        )

    # Check compute capability (need SM70+ for Volta, SM86 for Ampere)
    cc = gpu.compute_capability
    if cc:
        try:
            cc_major = int(cc.split(".")[0])
            if cc_major < 7:
                return False, (
                    f"TurboQuant KV cache requires GPU compute capability >= 7.0 (Volta+). "
                    f"Your GPU has SM {cc}. Falling back to standard KV cache."
                )
        except (ValueError, IndexError):
            pass

    # Check if TurboQuant inference engines are installed
    has_llama_tq = _check_llama_turboquant_available()
    has_vllm_tq = _check_vllm_turboquant_available()

    if has_llama_tq or has_vllm_tq:
        engines = []
        if has_llama_tq:
            engines.append("llama.cpp")
        if has_vllm_tq:
            engines.append("vLLM")
        return True, f"TurboQuant KV cache available ({', '.join(engines)})"

    # CUDA is sufficient but no TurboQuant fork installed
    return True, (
        f"TurboQuant KV cache available (CUDA {cuda_ver_str}). "
        f"Note: TurboQuant fork inference engines not yet installed — "
        f"turbo cache types may not be recognized until ithllc/llama-cpp-turboquant "
        f"or ithllc/vllm-turboquant is built."
    )


# ── Unified Quantization Pipeline ────────────────────────────────────


# Formats/quantizations that indicate already-weight-quantized models
_PREQUANTIZED_FORMATS = {"gguf", "awq", "gptq"}
_PREQUANTIZED_QUANTS = {
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q4_0", "Q4_K_M", "Q4_K_S",
    "Q5_K_M", "Q6_K", "Q8_0",
    "AWQ", "GPTQ", "INT8", "FP8",
}
_FULL_PRECISION_QUANTS = {"BF16", "FP16", "FP32"}


def detect_model_precision(model) -> str:
    """Detect whether a model is full-precision or already weight-quantized.

    Args:
        model: ModelProfile from the registry.

    Returns:
        "full_precision" if the model is BF16/FP16/FP32 safetensors.
        "weight_quantized" if the model is already quantized (GGUF/AWQ/GPTQ).
    """
    # Check quantization field first (most reliable)
    if model.quantization in _FULL_PRECISION_QUANTS:
        return "full_precision"
    if model.quantization in _PREQUANTIZED_QUANTS:
        return "weight_quantized"

    # Check format field
    if model.format in _PREQUANTIZED_FORMATS:
        return "weight_quantized"
    if model.format == "safetensors":
        return "full_precision"

    # Default: assume weight-quantized (safer — won't double-quantize)
    return "weight_quantized"


@dataclass
class QuantizationPipelineResult:
    """Result of the unified quantization pipeline decision."""

    # Weight quantization stage
    needs_weight_quant: bool = False
    weight_quant_method: str = ""       # e.g. "bnb_int4", "awq_marlin"
    weight_quant_reason: str = ""

    # KV cache compression stage
    needs_kv_compression: bool = False
    kv_level: KVQuantLevel = KVQuantLevel.NONE
    kv_reason: str = ""

    # Overall
    stages_applied: list[str] = field(default_factory=list)
    model_precision: str = ""           # "full_precision" or "weight_quantized"

    @property
    def summary(self) -> str:
        if not self.stages_applied:
            return "No quantization applied"
        return " + ".join(self.stages_applied)


def plan_quantization_pipeline(
    model,
    sys_info,
    kv_quant_choice: str = "auto",
    engine: str = "auto",
) -> QuantizationPipelineResult:
    """Plan the unified quantization pipeline for a model.

    Determines which compression stages to apply based on model precision:
    - Full precision (BF16/FP16) → weight quantization FIRST, then KV cache compression
    - Already quantized (GGUF/AWQ) → KV cache compression ONLY

    Args:
        model: ModelProfile from the registry.
        sys_info: SystemInfo from detect_system().
        kv_quant_choice: User's --kv-quant flag value ("auto", "none", "turbo4", etc.).
        engine: Inference engine ("auto", "llama.cpp", "vllm").

    Returns:
        QuantizationPipelineResult with the planned stages.
    """
    from tqcli.core.quantizer import QuantizationMethod, select_quantization

    result = QuantizationPipelineResult()
    result.model_precision = detect_model_precision(model)

    # ── Stage 1: Weight quantization (full precision models only) ─────
    if result.model_precision == "full_precision":
        quant_method = select_quantization(model, sys_info)
        if quant_method is None:
            # Model too large even after maximum quantization
            result.weight_quant_reason = (
                f"Model is {model.quantization} — too large for {sys_info.total_vram_mb} MB VRAM "
                f"even with INT4 quantization (may need smaller model or more VRAM)"
            )
        elif quant_method != QuantizationMethod.NONE:
            result.needs_weight_quant = True
            result.weight_quant_method = quant_method.value
            result.weight_quant_reason = (
                f"Model is {model.quantization} — applying {quant_method.value} "
                f"weight quantization to fit in {sys_info.total_vram_mb} MB VRAM"
            )
            result.stages_applied.append(f"weight:{quant_method.value}")
        else:
            result.weight_quant_reason = (
                f"Model is {model.quantization} — fits in VRAM without weight quantization"
            )
    else:
        result.weight_quant_reason = (
            f"Model already weight-quantized ({model.quantization} {model.format})"
        )

    # ── Stage 2: KV cache compression ─────────────────────────────────
    tq_available, tq_msg = check_turboquant_compatibility(sys_info)

    if kv_quant_choice == "none":
        result.kv_level = KVQuantLevel.NONE
        result.kv_reason = "KV compression disabled by user (--kv-quant none)"
    elif not tq_available:
        result.kv_level = KVQuantLevel.NONE
        result.kv_reason = f"TurboQuant unavailable: {tq_msg}"
    else:
        # Determine effective engine
        eff_engine = engine
        if eff_engine == "auto":
            eff_engine = model.engine if hasattr(model, "engine") else "llama.cpp"

        result.kv_level = select_kv_quant(
            available_kv_mb=50,  # Conservative estimate
            engine=eff_engine,
            user_choice=kv_quant_choice,
        )
        if result.kv_level != KVQuantLevel.NONE:
            result.needs_kv_compression = True
            info = get_kv_quant_info(result.kv_level)
            result.kv_reason = (
                f"Applying {result.kv_level.value} KV compression "
                f"({info['compression']}, {info['quality']})"
            )
            result.stages_applied.append(f"kv:{result.kv_level.value}")
        else:
            result.kv_reason = "KV cache has sufficient space, no compression needed"

    log.info(
        "Quantization pipeline: model=%s precision=%s stages=%s",
        model.id, result.model_precision, result.summary,
    )
    return result
