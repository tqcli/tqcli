"""llama.cpp backend via llama-cpp-python. Cross-platform: macOS, Linux, Windows."""

from __future__ import annotations

import time
from typing import Generator

from tqcli.core.engine import (
    ChatMessage,
    CompletionResult,
    InferenceEngine,
    InferenceStats,
)


class LlamaBackend(InferenceEngine):
    def __init__(
        self,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int = 0,
        verbose: bool = False,
    ):
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads
        self._verbose = verbose
        self._model = None
        self._model_path: str = ""

    @property
    def engine_name(self) -> str:
        return "llama.cpp"

    @property
    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401

            return True
        except ImportError:
            return False

    def load_model(self, model_path: str, **kwargs) -> None:
        from llama_cpp import Llama

        params = {
            "model_path": model_path,
            "n_ctx": kwargs.get("n_ctx", self._n_ctx),
            "n_gpu_layers": kwargs.get("n_gpu_layers", self._n_gpu_layers),
            "verbose": self._verbose,
        }
        if self._n_threads > 0:
            params["n_threads"] = self._n_threads
        self._model = Llama(**params)
        self._model_path = model_path

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._model_path = ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def chat(self, messages: list[ChatMessage], **kwargs) -> CompletionResult:
        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")

        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        start = time.perf_counter()

        response = self._model.create_chat_completion(
            messages=msg_dicts,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            stop=kwargs.get("stop"),
        )

        elapsed = time.perf_counter() - start
        text = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        stats = InferenceStats(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", 0),
            completion_time_s=elapsed,
            tokens_per_second=self._compute_tps(completion_tokens, elapsed),
            total_time_s=elapsed,
        )

        return CompletionResult(
            text=text,
            stats=stats,
            model_id=self._model_path,
            finish_reason=response["choices"][0].get("finish_reason", "stop"),
        )

    def chat_stream(
        self, messages: list[ChatMessage], **kwargs
    ) -> Generator[tuple[str, InferenceStats | None], None, None]:
        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")

        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]
        start = time.perf_counter()
        first_token_time = None
        token_count = 0

        stream = self._model.create_chat_completion(
            messages=msg_dicts,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            stream=True,
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1
                yield content, None

        elapsed = time.perf_counter() - start
        ttft = (first_token_time - start) if first_token_time else elapsed

        final_stats = InferenceStats(
            completion_tokens=token_count,
            completion_time_s=elapsed,
            tokens_per_second=self._compute_tps(token_count, elapsed),
            time_to_first_token_s=ttft,
            total_time_s=elapsed,
        )
        yield "", final_stats

    def complete(self, prompt: str, **kwargs) -> CompletionResult:
        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")

        start = time.perf_counter()
        response = self._model(
            prompt,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            stop=kwargs.get("stop"),
            echo=False,
        )
        elapsed = time.perf_counter() - start
        text = response["choices"][0]["text"]
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        stats = InferenceStats(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", 0),
            completion_time_s=elapsed,
            tokens_per_second=self._compute_tps(completion_tokens, elapsed),
            total_time_s=elapsed,
        )

        return CompletionResult(
            text=text,
            stats=stats,
            model_id=self._model_path,
            finish_reason=response["choices"][0].get("finish_reason", "stop"),
        )
