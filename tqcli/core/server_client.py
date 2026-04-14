"""HTTP client backend — connects to a running inference server instead of
loading the model in-process. Used by multi-process workers.

Compatible with both llama.cpp server and vLLM server (OpenAI API format).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Generator

from tqcli.core.engine import (
    ChatMessage,
    CompletionResult,
    InferenceEngine,
    InferenceStats,
)


class ServerClientBackend(InferenceEngine):
    """Inference engine that delegates to a running HTTP server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8741", model_name: str = "local"):
        self._base_url = base_url.rstrip("/")
        self._model_name = model_name
        self._connected = False

    @property
    def engine_name(self) -> str:
        return "server-client"

    @property
    def is_available(self) -> bool:
        return True  # always available — just needs a running server

    def load_model(self, model_path: str, **kwargs) -> None:
        """For the server client, 'loading' means verifying the server is reachable."""
        self._model_name = model_path
        self._connected = self._health_check()
        if not self._connected:
            raise RuntimeError(
                f"Cannot connect to inference server at {self._base_url}. "
                f"Start one with: tqcli serve start"
            )

    def unload_model(self) -> None:
        self._connected = False

    @property
    def is_loaded(self) -> bool:
        return self._connected

    def chat(self, messages: list[ChatMessage], **kwargs) -> CompletionResult:
        payload = {
            "model": self._model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False,
        }
        if kwargs.get("stop"):
            payload["stop"] = kwargs["stop"]

        start = time.perf_counter()
        data = self._post("/v1/chat/completions", payload)
        elapsed = time.perf_counter() - start

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        stats = InferenceStats(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", 0),
            completion_time_s=elapsed,
            tokens_per_second=completion_tokens / elapsed if elapsed > 0 else 0,
            total_time_s=elapsed,
        )

        return CompletionResult(
            text=text,
            stats=stats,
            model_id=self._model_name,
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
        )

    def chat_stream(
        self, messages: list[ChatMessage], **kwargs
    ) -> Generator[tuple[str, InferenceStats | None], None, None]:
        payload = {
            "model": self._model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True,
        }
        if kwargs.get("stop"):
            payload["stop"] = kwargs["stop"]

        url = f"{self._base_url}/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        start = time.perf_counter()
        first_token_time = None
        token_count = 0
        full_text = ""

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                buffer = ""
                for raw_chunk in _iter_lines(resp):
                    line = raw_chunk.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # strip "data: " prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk_data["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
                        full_text += content
                        yield content, None
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"Server connection lost: {e}")

        elapsed = time.perf_counter() - start
        ttft = (first_token_time - start) if first_token_time else elapsed

        final_stats = InferenceStats(
            completion_tokens=token_count,
            completion_time_s=elapsed,
            tokens_per_second=token_count / elapsed if elapsed > 0 else 0,
            time_to_first_token_s=ttft,
            total_time_s=elapsed,
        )
        yield "", final_stats

    def complete(self, prompt: str, **kwargs) -> CompletionResult:
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False,
        }

        start = time.perf_counter()
        data = self._post("/v1/completions", payload)
        elapsed = time.perf_counter() - start

        text = data["choices"][0]["text"]
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        stats = InferenceStats(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", 0),
            completion_time_s=elapsed,
            tokens_per_second=completion_tokens / elapsed if elapsed > 0 else 0,
            total_time_s=elapsed,
        )

        return CompletionResult(
            text=text, stats=stats, model_id=self._model_name,
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
        )

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self._base_url}{path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode() if e.fp else ""
            raise RuntimeError(f"Server error {e.code}: {body}")
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach server at {self._base_url}: {e.reason}. "
                f"Is it running? Start with: tqcli serve start"
            )

    def _health_check(self) -> bool:
        try:
            url = f"{self._base_url}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            try:
                url = f"{self._base_url}/health"
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5):
                    return True
            except (urllib.error.URLError, OSError):
                return False


def _iter_lines(resp) -> Generator[str, None, None]:
    """Iterate over lines from a streaming HTTP response."""
    buffer = ""
    while True:
        chunk = resp.read(1024)
        if not chunk:
            if buffer:
                yield buffer
            break
        buffer += chunk.decode("utf-8", errors="replace")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line
