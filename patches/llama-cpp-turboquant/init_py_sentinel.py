# ---------------------------------------------------------------------------
# TurboQuant runtime sentinels.
#
# Append to `src/llama_cpp/__init__.py` of the (forthcoming)
# `tqcli/llama-cpp-python-turboquant` fork, after the upstream re-exports.
#
# `tqcli/core/engine_auditor.py` reads `TURBOQUANT_BUILD` to distinguish this
# fork from upstream `llama-cpp-python` at runtime. `TURBOQUANT_KV_TYPES`
# enumerates the cache-quantization labels the C++ engine accepts via
# `LlamaContextParams.kv_quant_type` (or the equivalent wrapper kwarg the
# Python bindings expose).
#
# DO NOT promote these to a class or a function — the auditor uses a plain
# `getattr(llama_cpp, "TURBOQUANT_BUILD", False) is True` check so absence
# (upstream) and presence (this fork) both behave correctly.
# ---------------------------------------------------------------------------
TURBOQUANT_BUILD: bool = True
TURBOQUANT_KV_TYPES: tuple[str, ...] = ("turbo2", "turbo3", "turbo4")
