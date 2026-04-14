"""Model registry: catalog of supported models and their capability profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TaskDomain(Enum):
    CODING = "coding"
    REASONING = "reasoning"
    GENERAL = "general"
    MATH = "math"
    CREATIVE = "creative"
    INSTRUCTION = "instruction"


@dataclass
class ModelProfile:
    """Describes a model's identity and strengths for routing decisions."""

    id: str  # e.g. "gemma-4-12b-Q4_K_M"
    family: str  # e.g. "gemma", "qwen-coder", "qwen-instruct"
    display_name: str
    hf_repo: str  # HuggingFace repo for download
    filename: str  # GGUF filename or model dir name
    parameter_count: str  # "12B", "7B", etc.
    quantization: str  # "Q4_K_M", "AWQ", etc.
    format: str  # "gguf", "awq", "gptq", "safetensors"
    context_length: int = 8192
    strengths: list[TaskDomain] = field(default_factory=list)
    strength_scores: dict[str, float] = field(default_factory=dict)
    min_ram_mb: int = 4000
    min_vram_mb: int = 0
    engine: str = "llama.cpp"  # "llama.cpp" or "vllm"
    local_path: Path | None = None


# Predefined profiles for supported models.
# strength_scores: 0.0 (worst) to 1.0 (best) — used by the router to rank models per task.
# Scores are derived from public benchmark leaderboards and model card claims:
#  - Gemma 4: Strong general + multilingual + reasoning; Google reports top-tier
#    instruction following and factual grounding.
#  - Qwen2.5-Coder: Specifically trained on code; dominates HumanEval, MBPP, and
#    LiveCodeBench at its size class.
#  - Qwen2.5-Instruct: Trained for conversational instruction following; strong
#    on MT-Bench, IFEval, and general NLP tasks but less specialized for code.

BUILTIN_PROFILES: list[ModelProfile] = [
    # --- Google Gemma 4 ---
    ModelProfile(
        id="gemma-4-27b-it-Q4_K_M",
        family="gemma",
        display_name="Gemma 4 27B Instruct (Q4_K_M)",
        hf_repo="google/gemma-4-27b-it-GGUF",
        filename="gemma-4-27b-it-Q4_K_M.gguf",
        parameter_count="27B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=128000,
        strengths=[TaskDomain.GENERAL, TaskDomain.REASONING, TaskDomain.INSTRUCTION],
        strength_scores={
            "coding": 0.75,
            "reasoning": 0.90,
            "general": 0.92,
            "math": 0.80,
            "creative": 0.85,
            "instruction": 0.90,
        },
        min_ram_mb=18000,
        min_vram_mb=16000,
        engine="llama.cpp",
    ),
    ModelProfile(
        id="gemma-4-12b-it-Q4_K_M",
        family="gemma",
        display_name="Gemma 4 12B Instruct (Q4_K_M)",
        hf_repo="google/gemma-4-12b-it-GGUF",
        filename="gemma-4-12b-it-Q4_K_M.gguf",
        parameter_count="12B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=128000,
        strengths=[TaskDomain.GENERAL, TaskDomain.REASONING, TaskDomain.INSTRUCTION],
        strength_scores={
            "coding": 0.65,
            "reasoning": 0.80,
            "general": 0.85,
            "math": 0.70,
            "creative": 0.80,
            "instruction": 0.82,
        },
        min_ram_mb=8000,
        min_vram_mb=7000,
        engine="llama.cpp",
    ),
    # --- Qwen2.5-Coder ---
    ModelProfile(
        id="qwen2.5-coder-32b-instruct-Q4_K_M",
        family="qwen-coder",
        display_name="Qwen2.5 Coder 32B Instruct (Q4_K_M)",
        hf_repo="Qwen/Qwen2.5-Coder-32B-Instruct-GGUF",
        filename="qwen2.5-coder-32b-instruct-q4_k_m.gguf",
        parameter_count="32B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=32768,
        strengths=[TaskDomain.CODING, TaskDomain.MATH, TaskDomain.REASONING],
        strength_scores={
            "coding": 0.95,
            "reasoning": 0.78,
            "general": 0.65,
            "math": 0.88,
            "creative": 0.50,
            "instruction": 0.70,
        },
        min_ram_mb=20000,
        min_vram_mb=18000,
        engine="llama.cpp",
    ),
    ModelProfile(
        id="qwen2.5-coder-7b-instruct-Q4_K_M",
        family="qwen-coder",
        display_name="Qwen2.5 Coder 7B Instruct (Q4_K_M)",
        hf_repo="Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
        filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        parameter_count="7B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=32768,
        strengths=[TaskDomain.CODING, TaskDomain.MATH],
        strength_scores={
            "coding": 0.85,
            "reasoning": 0.60,
            "general": 0.55,
            "math": 0.75,
            "creative": 0.40,
            "instruction": 0.58,
        },
        min_ram_mb=6000,
        min_vram_mb=5000,
        engine="llama.cpp",
    ),
    # --- Qwen2.5-Instruct ---
    ModelProfile(
        id="qwen2.5-32b-instruct-Q4_K_M",
        family="qwen-instruct",
        display_name="Qwen2.5 32B Instruct (Q4_K_M)",
        hf_repo="Qwen/Qwen2.5-32B-Instruct-GGUF",
        filename="qwen2.5-32b-instruct-q4_k_m.gguf",
        parameter_count="32B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=32768,
        strengths=[TaskDomain.INSTRUCTION, TaskDomain.REASONING, TaskDomain.GENERAL],
        strength_scores={
            "coding": 0.72,
            "reasoning": 0.88,
            "general": 0.90,
            "math": 0.82,
            "creative": 0.80,
            "instruction": 0.92,
        },
        min_ram_mb=20000,
        min_vram_mb=18000,
        engine="llama.cpp",
    ),
    ModelProfile(
        id="qwen2.5-7b-instruct-Q4_K_M",
        family="qwen-instruct",
        display_name="Qwen2.5 7B Instruct (Q4_K_M)",
        hf_repo="Qwen/Qwen2.5-7B-Instruct-GGUF",
        filename="qwen2.5-7b-instruct-q4_k_m.gguf",
        parameter_count="7B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=32768,
        strengths=[TaskDomain.INSTRUCTION, TaskDomain.GENERAL],
        strength_scores={
            "coding": 0.58,
            "reasoning": 0.70,
            "general": 0.75,
            "math": 0.65,
            "creative": 0.70,
            "instruction": 0.78,
        },
        min_ram_mb=6000,
        min_vram_mb=5000,
        engine="llama.cpp",
    ),
]


class ModelRegistry:
    """Discovers, registers, and queries available models."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._profiles: dict[str, ModelProfile] = {}
        for p in BUILTIN_PROFILES:
            self._profiles[p.id] = p

    def scan_local_models(self) -> list[ModelProfile]:
        found = []
        if not self.models_dir.exists():
            return found
        for f in self.models_dir.rglob("*.gguf"):
            for profile in self._profiles.values():
                if f.name.lower() == profile.filename.lower():
                    profile.local_path = f
                    found.append(profile)
        return found

    def get_available_models(self) -> list[ModelProfile]:
        return [p for p in self._profiles.values() if p.local_path is not None]

    def get_all_profiles(self) -> list[ModelProfile]:
        return list(self._profiles.values())

    def get_profile(self, model_id: str) -> ModelProfile | None:
        return self._profiles.get(model_id)

    def get_models_for_domain(self, domain: TaskDomain) -> list[ModelProfile]:
        available = self.get_available_models()
        scored = []
        for m in available:
            score = m.strength_scores.get(domain.value, 0.0)
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def register_model(self, profile: ModelProfile) -> None:
        self._profiles[profile.id] = profile

    def fits_hardware(self, profile: ModelProfile, ram_mb: int, vram_mb: int) -> bool:
        if vram_mb > 0 and profile.min_vram_mb > 0:
            return profile.min_vram_mb <= vram_mb
        return profile.min_ram_mb <= ram_mb
