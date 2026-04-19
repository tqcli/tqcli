"""Generate turboquant_kv.json metadata via activation-based calibration.

Mirrors the vLLM-turboquant fork's own reference selector,
`vllm.v1.attention.ops.turboquant_kv_cache.build_turboquant_outlier_masks`,
which scores each channel by the mean-squared activation across tokens and
picks the top-`outlier_count` indices per KV head. This module accumulates the
same statistic online over a calibration corpus and writes the resulting
per-layer, per-head outlier indices to `turboquant_kv.json` in the model
directory.

Design rationale:
    Pre-computed metadata is higher quality than runtime first-batch auto
    calibration (more prompts → tighter variance estimate, reproducible
    across servers). We capture POST-RoPE K (since the KV cache stores
    post-RoPE keys) and V (no RoPE on values) at bf16 precision with CPU
    offload to fit small-VRAM hosts.

Refused cases:
    - Pre-quantized source weights (AWQ, GPTQ, bnb): activation statistics
      from quantized weights are biased by the weight quantization, not the
      true FP distribution the runtime will see.
    - Variable head_dim within a model (Gemma 4 style sliding/global).
    - head_dim not a multiple of 16.
    - Architectures without a registered capture wrapper.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from vllm.v1.attention.ops.turboquant_kv_cache import (
    TURBOQUANT_GROUP_ALIGNMENT,
    TURBOQUANT_OUTLIER_RATIOS,
    get_turboquant_outlier_count,
)

DEFAULT_CALIBRATION_PROMPTS: list[str] = [
    "The capital of France is Paris, a city of roughly 2.1 million residents on the Seine.",
    "Explain the theory of general relativity in two short paragraphs aimed at a physics undergraduate.",
    "Write a short poem about autumn leaves falling from maple trees in October.",
    "List three common causes of database deadlocks in PostgreSQL and how to mitigate each.",
    "Summarize the plot of Dostoevsky's 'Crime and Punishment' in one paragraph.",
    "Describe how a transformer-based language model computes self-attention during inference.",
    "Write a Python function that inverts a binary tree using an iterative depth-first traversal.",
    "What are the differences between TCP and UDP at the transport layer? Give two examples of each.",
    "Translate the following sentence to Spanish: 'The train to Madrid departs at seven fifteen in the morning.'",
    "Give me a quick recipe for making sourdough bread without a starter.",
    "Explain why compiler inlining can sometimes make code slower rather than faster.",
    "What is the Monty Hall problem, and why is switching doors the correct strategy?",
    "Describe the life cycle of a monarch butterfly from egg to adult.",
    "Compare and contrast the ZFS and Btrfs filesystems in three bullet points.",
    "Write a Haiku about the smell of rain on hot asphalt in summer.",
    "Give a step-by-step derivation of the quadratic formula from completing the square.",
    "What is the difference between a virus and a bacterium? Keep the answer under 150 words.",
    "Write a short dialogue between two software engineers debating whether to adopt Rust for a new microservice.",
    "Summarize the key economic conditions that led to the 2008 financial crisis.",
    "Explain how CRISPR-Cas9 gene editing works at the molecular level to a smart high schooler.",
    "List five common indoor plants that tolerate low-light conditions, and how often to water each.",
    "Describe the role of mitochondria in eukaryotic cells and the endosymbiotic theory of their origin.",
    "Write a letter to a friend describing a weekend hiking trip through the Grand Canyon.",
    "What are the main trade-offs between REST and GraphQL APIs for a small-team startup?",
    "Give a beginner-friendly explanation of how the SHA-256 hash function works internally.",
    "Describe three cognitive biases that commonly affect investment decisions in individual retail traders.",
    "Write a single-paragraph summary of Newton's three laws of motion.",
    "Explain how MRI machines produce images of soft tissue using nuclear magnetic resonance.",
    "What are the primary differences between classical, operant, and observational learning in psychology?",
    "Write a short science-fiction vignette set aboard a generation ship 300 years into its voyage.",
]


# ---------------------------------------------------------------------------
# Architecture registry — each entry patches a specific model's attention
# forward to capture post-RoPE K and V as second-moment accumulators.
# ---------------------------------------------------------------------------


@dataclass
class _CaptureHandle:
    """Lifetime-manager for a monkey-patched attention forward."""

    restore: Callable[[], None]
    scores_k: dict[int, torch.Tensor]
    scores_v: dict[int, torch.Tensor]
    token_counts: dict[int, int]


def _install_qwen3_capture() -> _CaptureHandle:
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3.modeling_qwen3 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    scores_k: dict[int, torch.Tensor] = {}
    scores_v: dict[int, torch.Tensor] = {}
    token_counts: dict[int, int] = {}
    original = modeling_qwen3.Qwen3Attention.forward

    def patched_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        with torch.no_grad():
            k32 = key_states.detach().to(torch.float32)
            v32 = value_states.detach().to(torch.float32)
            bsz, n_kv, seq, hdim = k32.shape
            k_flat = k32.permute(0, 2, 1, 3).reshape(-1, n_kv, hdim)
            v_flat = v32.permute(0, 2, 1, 3).reshape(-1, n_kv, hdim)
            k_sum = k_flat.square().sum(dim=0).to(torch.float64).cpu()
            v_sum = v_flat.square().sum(dim=0).to(torch.float64).cpu()
            if self.layer_idx in scores_k:
                scores_k[self.layer_idx] += k_sum
                scores_v[self.layer_idx] += v_sum
                token_counts[self.layer_idx] += bsz * seq
            else:
                scores_k[self.layer_idx] = k_sum
                scores_v[self.layer_idx] = v_sum
                token_counts[self.layer_idx] = bsz * seq

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attn_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    modeling_qwen3.Qwen3Attention.forward = patched_forward

    def restore():
        modeling_qwen3.Qwen3Attention.forward = original

    return _CaptureHandle(restore=restore, scores_k=scores_k, scores_v=scores_v, token_counts=token_counts)


# Registry maps HF model architecture strings to capture installers.
# Add new entries as other families are validated.
_CAPTURE_INSTALLERS: dict[str, Callable[[], _CaptureHandle]] = {
    "Qwen3ForCausalLM": _install_qwen3_capture,
}


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------


def _extract_architecture_params(config: dict) -> tuple[str, int, int, int]:
    """Return (architecture, head_dim, num_kv_heads, num_hidden_layers).

    Handles nested text_config (Gemma-style multimodal) when values aren't at
    the top level.
    """
    arch = (config.get("architectures") or ["unknown"])[0]
    head_dim = config.get("head_dim")
    num_kv_heads = config.get("num_key_value_heads")
    num_layers = config.get("num_hidden_layers")

    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        head_dim = head_dim if head_dim is not None else text_config.get("head_dim")
        num_kv_heads = num_kv_heads if num_kv_heads is not None else text_config.get("num_key_value_heads")
        num_layers = num_layers if num_layers is not None else text_config.get("num_hidden_layers")

    return arch, head_dim, num_kv_heads, num_layers


def check_calibration_preconditions(
    model_dir: Path | str,
    kv_cache_dtype: str,
) -> tuple[bool, str]:
    """Return (ok, reason). Reason always carries enough info to act on."""
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False, f"No config.json at {model_dir}"

    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        return False, f"config.json malformed: {exc}"

    # Refuse pre-quantized source weights — activation stats would be biased.
    if config.get("quantization_config"):
        method = config["quantization_config"].get("quant_method", "unknown")
        return False, (
            f"Source weights are already quantized ({method}). Activation calibration "
            f"on pre-quantized weights biases the variance estimate. Requires dequantized "
            f"source or a different calibration path (out of scope)."
        )

    arch, head_dim, num_kv_heads, num_layers = _extract_architecture_params(config)

    if head_dim is None or num_kv_heads is None or num_layers is None:
        return False, (
            f"Could not resolve head_dim / num_kv_heads / num_hidden_layers from "
            f"config.json (got {head_dim} / {num_kv_heads} / {num_layers})."
        )

    if head_dim % TURBOQUANT_GROUP_ALIGNMENT != 0:
        return False, f"head_dim {head_dim} is not a multiple of {TURBOQUANT_GROUP_ALIGNMENT}."

    text_config = config.get("text_config") or {}
    global_head_dim = text_config.get("global_head_dim")
    if global_head_dim is not None and global_head_dim != head_dim:
        return False, (
            f"Variable head_dim detected (head_dim={head_dim}, global_head_dim={global_head_dim}). "
            f"Mixed-head-dim architectures need per-layer metadata, not supported."
        )

    if arch not in _CAPTURE_INSTALLERS:
        return False, (
            f"Architecture {arch!r} has no capture wrapper registered. "
            f"Supported: {sorted(_CAPTURE_INSTALLERS)}. Add a handler in "
            f"_CAPTURE_INSTALLERS to enable."
        )

    if kv_cache_dtype not in TURBOQUANT_OUTLIER_RATIOS:
        return False, (
            f"Unknown kv_cache_dtype {kv_cache_dtype!r}. Supported: "
            f"{sorted(TURBOQUANT_OUTLIER_RATIOS)}."
        )

    try:
        outlier_count = get_turboquant_outlier_count(head_dim, kv_cache_dtype)
    except ValueError as exc:
        return False, str(exc)

    return True, (
        f"OK: arch={arch}, head_dim={head_dim}, num_kv_heads={num_kv_heads}, "
        f"num_layers={num_layers}, outlier_count={outlier_count}"
    )


# ---------------------------------------------------------------------------
# Calibration entry point
# ---------------------------------------------------------------------------


def generate_turboquant_metadata(
    model_dir: Path | str,
    kv_cache_dtype: str,
    calibration_prompts: list[str] | None = None,
    max_seq_len: int = 512,
    output_path: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Calibrate a model and write turboquant_kv.json to its directory.

    Returns the output path. Raises ValueError on precondition failure.
    """
    model_dir = Path(model_dir)
    ok, reason = check_calibration_preconditions(model_dir, kv_cache_dtype)
    if not ok:
        raise ValueError(f"Cannot calibrate {model_dir}: {reason}")

    if calibration_prompts is None:
        calibration_prompts = DEFAULT_CALIBRATION_PROMPTS

    output_path = Path(output_path) if output_path else (model_dir / "turboquant_kv.json")

    def _log(msg: str) -> None:
        if progress is not None:
            progress(msg)
        else:
            print(msg, flush=True)

    config = json.loads((model_dir / "config.json").read_text())
    arch, head_dim, num_kv_heads, num_layers = _extract_architecture_params(config)
    outlier_count = get_turboquant_outlier_count(head_dim, kv_cache_dtype)

    _log(
        f"[calibrate] model={model_dir.name} arch={arch} head_dim={head_dim} "
        f"num_kv_heads={num_kv_heads} num_layers={num_layers} recipe={kv_cache_dtype} "
        f"outlier_count={outlier_count}"
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    handle = _CAPTURE_INSTALLERS[arch]()

    model = None
    total_tokens_processed = 0
    start = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        _log(f"[calibrate] loading model (bf16, device_map=auto)...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()

        with torch.no_grad():
            for idx, prompt in enumerate(calibration_prompts, start=1):
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                seq_len = int(inputs["input_ids"].shape[-1])
                model(**inputs, use_cache=False)
                total_tokens_processed += seq_len
                _log(
                    f"[calibrate]   prompt {idx}/{len(calibration_prompts)}: "
                    f"{seq_len} tokens (total {total_tokens_processed})"
                )

        elapsed = time.perf_counter() - start

        metadata = {
            "version": 1,
            "recipe": kv_cache_dtype,
            "head_size": head_dim,
            "model_name": config.get("_name_or_path") or model_dir.name,
            "transform_version": "structured_hadamard_v1",
            "codebook_version": "lloyd_beta_v1",
            "calibration": {
                "method": "activation_second_moment_top_k",
                "objective": "per_kv_head_per_channel",
                "num_prompts": len(calibration_prompts),
                "max_seq_len": max_seq_len,
                "num_observed_tokens": total_tokens_processed,
                "dtype": "bfloat16",
                "device": str(model.device),
            },
            "layers": {},
        }

        for layer_idx in range(num_layers):
            if layer_idx not in handle.scores_k:
                raise RuntimeError(
                    f"Layer {layer_idx} produced no activations during calibration — "
                    f"check that the capture wrapper is wired correctly."
                )
            k_score = handle.scores_k[layer_idx]  # [num_kv_heads, head_dim]
            v_score = handle.scores_v[layer_idx]

            if k_score.shape != (num_kv_heads, head_dim):
                raise RuntimeError(
                    f"Layer {layer_idx} K score has shape {k_score.shape}, "
                    f"expected {(num_kv_heads, head_dim)}."
                )

            k_top = torch.sort(
                torch.topk(k_score, k=outlier_count, dim=-1).indices, dim=-1
            ).values.tolist()
            v_top = torch.sort(
                torch.topk(v_score, k=outlier_count, dim=-1).indices, dim=-1
            ).values.tolist()

            metadata["layers"][f"model.layers.{layer_idx}.self_attn"] = {
                "key_high_precision_indices": k_top,
                "value_high_precision_indices": v_top,
            }

        output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        size_kb = output_path.stat().st_size / 1024.0
        _log(
            f"[calibrate] wrote {output_path} ({size_kb:.1f} KB, {num_layers} layers, "
            f"{total_tokens_processed} tokens, elapsed {elapsed:.1f}s)"
        )

    finally:
        handle.restore()
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return output_path


def ensure_turboquant_metadata(
    model_dir: Path | str,
    kv_cache_dtype: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, bool]:
    """If metadata is missing and prerequisites hold, generate it.

    Returns (path, generated_bool). If metadata exists, skips; if
    precondition check refuses, raises ValueError with the reason.
    """
    model_dir = Path(model_dir)
    metadata_path = model_dir / "turboquant_kv.json"
    if metadata_path.is_file():
        return metadata_path, False

    generated_path = generate_turboquant_metadata(
        model_dir=model_dir,
        kv_cache_dtype=kv_cache_dtype,
        progress=progress,
    )
    return generated_path, True
