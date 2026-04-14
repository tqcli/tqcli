"""vLLM backend for GPU-accelerated serving. Linux/WSL2 + NVIDIA GPU only."""

from __future__ import annotations

import time
from typing import Generator

from tqcli.core.engine import (
    ChatMessage,
    CompletionResult,
    InferenceEngine,
    InferenceStats,
)


class VllmBackend(InferenceEngine):
    def __init__(
        self,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.85,
        quantization: str | None = None,
        tensor_parallel_size: int = 1,
        kv_cache_dtype: str = "auto",
    ):
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._quantization = quantization
        self._tensor_parallel_size = tensor_parallel_size
        self._kv_cache_dtype = kv_cache_dtype
        self._llm = None
        self._model_name: str = ""
        self._tokenizer = None

    @property
    def engine_name(self) -> str:
        return "vllm"

    @property
    def is_available(self) -> bool:
        try:
            import vllm  # noqa: F401

            return True
        except ImportError:
            return False

    def load_model(self, model_path: str, **kwargs) -> None:
        from vllm import LLM

        params = {
            "model": model_path,
            "max_model_len": kwargs.get("max_model_len", self._max_model_len),
            "gpu_memory_utilization": kwargs.get(
                "gpu_memory_utilization", self._gpu_memory_utilization
            ),
            "tensor_parallel_size": kwargs.get(
                "tensor_parallel_size", self._tensor_parallel_size
            ),
        }
        if self._quantization:
            params["quantization"] = self._quantization
        if self._kv_cache_dtype != "auto":
            params["kv_cache_dtype"] = self._kv_cache_dtype

        self._llm = LLM(**params)
        self._model_name = model_path

    def unload_model(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._model_name = ""

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def _format_chat_prompt(self, messages: list[ChatMessage]) -> str:
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"<|system|>\n{msg.content}")
            elif msg.role == "user":
                parts.append(f"<|user|>\n{msg.content}")
            elif msg.role == "assistant":
                parts.append(f"<|assistant|>\n{msg.content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def chat(self, messages: list[ChatMessage], **kwargs) -> CompletionResult:
        if not self._llm:
            raise RuntimeError("No model loaded. Call load_model() first.")

        from vllm import SamplingParams

        prompt = self._format_chat_prompt(messages)
        sampling = SamplingParams(
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            stop=kwargs.get("stop"),
        )

        start = time.perf_counter()
        outputs = self._llm.generate([prompt], sampling)
        elapsed = time.perf_counter() - start

        output = outputs[0]
        text = output.outputs[0].text
        completion_tokens = len(output.outputs[0].token_ids)
        prompt_tokens = len(output.prompt_token_ids)

        stats = InferenceStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            completion_time_s=elapsed,
            tokens_per_second=self._compute_tps(completion_tokens, elapsed),
            total_time_s=elapsed,
        )

        return CompletionResult(
            text=text,
            stats=stats,
            model_id=self._model_name,
            finish_reason=output.outputs[0].finish_reason or "stop",
        )

    def chat_stream(
        self, messages: list[ChatMessage], **kwargs
    ) -> Generator[tuple[str, InferenceStats | None], None, None]:
        # vLLM streaming requires the async engine; for the CLI we do a full
        # generation and then yield the text in chunks to keep the UI responsive.
        result = self.chat(messages, **kwargs)
        chunk_size = 4
        text = result.text
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            yield chunk, None
        yield "", result.stats

    def complete(self, prompt: str, **kwargs) -> CompletionResult:
        if not self._llm:
            raise RuntimeError("No model loaded. Call load_model() first.")

        from vllm import SamplingParams

        sampling = SamplingParams(
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            stop=kwargs.get("stop"),
        )

        start = time.perf_counter()
        outputs = self._llm.generate([prompt], sampling)
        elapsed = time.perf_counter() - start

        output = outputs[0]
        text = output.outputs[0].text
        completion_tokens = len(output.outputs[0].token_ids)
        prompt_tokens = len(output.prompt_token_ids)

        stats = InferenceStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            completion_time_s=elapsed,
            tokens_per_second=self._compute_tps(completion_tokens, elapsed),
            total_time_s=elapsed,
        )

        return CompletionResult(
            text=text,
            stats=stats,
            model_id=self._model_name,
            finish_reason=output.outputs[0].finish_reason or "stop",
        )
