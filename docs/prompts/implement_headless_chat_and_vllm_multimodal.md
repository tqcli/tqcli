# Implementation Prompt: Headless Chat + vLLM Multimodal Support

**Status:** Shipped in v0.5.0 ([#24](https://github.com/ithllc/tqCLI/issues/24)). `tqcli chat` now accepts `--prompt`, `--image`, `--audio`, `--messages`, `--json`, `--max-tokens`; `VllmBackend` passes images through `multi_modal_data={"image": [...]}` while preserving TurboQuant KV + CPU offload + `enforce_eager`.

**Why this prompt exists:** The tqCLI v0.5.0 integration suite (`tests/test_integration_turboquant_kv.py`) can only exercise vLLM CPU-offload text chat because:

1. `tqcli chat` is interactive-only — no `--prompt`, `--image`, or `--messages` flags. CI cannot drive multimodal runs.
2. `tqcli/core/vllm_backend.py::_messages_to_dicts` strips `ChatMessage.images` / `ChatMessage.audio` (see lines 134–136) — even if a message carries image paths, vLLM never sees them. The llama.cpp backend is fine (`tqcli/core/llama_backend.py:104`–`129` wires multimodal into llama-cpp-python's chat handler).

Both gaps have to close before we can verify vLLM image input for Gemma 4 E2B and multi-process CRM flows for Qwen 3 4B AWQ + Gemma 4 E2B on vLLM.

---

## Scope (two landing items)

### 1. Headless chat (single-shot + multi-turn)

Extend `tqcli/cli.py` `chat` command with:

| Flag | Type | Purpose |
|------|------|---------|
| `--prompt TEXT` | single string | Single-turn user input. Presence of this flag disables the interactive REPL. |
| `--image PATH` | repeatable | Path(s) to image files, appended to the final user message. |
| `--audio PATH` | repeatable | Path(s) to audio files, appended to the final user message. |
| `--messages FILE` | JSON path | List of `{role, content, images?, audio?}` dicts → prior history. If combined with `--prompt`, the prompt becomes the final user turn. |
| `--json` | flag | Emit the result as a JSON object on stdout instead of Rich output. Forces non-interactive. Sets exit code: 0 success, non-zero on engine/load/file-path failure. |
| `--max-tokens INT` | int (default 1024) | Generation budget. |

All existing flags (`--model`, `--engine`, `--kv-quant`, `--server-url`, `--context-length`) continue to work.

**JSON output schema (when `--json` is set):**

```json
{
  "model": "gemma-4-e2b-it-vllm",
  "engine": "vllm",
  "response": "The image shows a red square with a blue border.",
  "thinking": "Step 1 ...",
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 18,
    "total_tokens": 60
  },
  "performance": {
    "tokens_per_second": 0.22,
    "total_time_s": 81.7
  },
  "metadata": {
    "kv_quant": "turboquant35",
    "images": ["tests/fixtures/test_image.png"],
    "audio": []
  }
}
```

- If generation fails before producing tokens, emit `{"error": "...", "error_type": "..."}` and exit non-zero. `thinking` is `null` when the model produces no `<think>` / `<thinking>` blocks.
- Route all non-JSON chatter (progress bars, warnings) to stderr so stdout stays parseable.

**Where to plug in:**
- `tqcli/cli.py` — add Click options + a top-level branch: if `--prompt` or `--messages` provided, call a new `tqcli/ui/interactive.py::headless_turn()` instead of instantiating `InteractiveSession.run_loop()`.
- `tqcli/ui/interactive.py::InteractiveSession.chat_turn(...)` already accepts `images` / `audio` lists — reuse it directly from the new headless entry.
- `tqcli/core/thinking.py::extract_thinking()` already separates reasoning from the final answer; feed the full response through it before serializing.

### 2. vLLM multimodal pass-through

`tqcli/core/vllm_backend.py` has to carry `ChatMessage.images` / `.audio` through to vLLM's generation API.

**API to call:** vLLM's multimodal path uses the `multi_modal_data` kwarg on `LLM.generate(...)` (or `llm.chat(...)` in newer vLLM). For Gemma 4 E2B with `kv_cache_dtype=turboquant35`, the relevant payload is:

```python
from PIL import Image

prompt = self._apply_chat_template(messages)
multi_modal_data = {"image": [Image.open(p).convert("RGB") for m in messages for p in (m.images or [])]}
outputs = self._llm.generate(
    prompts=[{"prompt": prompt, "multi_modal_data": multi_modal_data}],
    sampling_params=sampling,
)
```

Implementation notes:
1. Update `_messages_to_dicts` to pass `images` + `audio` fields through when any message carries them. vLLM's chat template expects image placeholder tokens — for Gemma 4 that is `<start_of_image>` injected by the tokenizer's `apply_chat_template` when the message content is a list containing image dicts: `[{"type": "image"}, {"type": "text", "text": "..."}]`.
2. Rewrite `_apply_chat_template` to emit that structured form when images are present, so the tokenizer inserts the correct placeholder.
3. Audio for vLLM: there is no Gemma 4 vLLM audio path today; record audio inputs in the message and pass through — the model will respond with a graceful "no audio capability" text (matching the expected §E.5 behavior).
4. Preserve CPU offload + TurboQuant KV: no changes to `cpu_offload_gb`, `kv_cache_dtype`, `enable_turboquant`, `kv_cache_memory_bytes`, `enforce_eager` wiring.
5. Preserve streaming: update `chat_stream` in tandem so the same multimodal wiring is used.

---

## Skills to invoke when executing this prompt

| Skill | Why |
|-------|-----|
| `architecture-doc-review` | After the change, re-run to pick up the new `tqcli/cli.py` options + multimodal wiring in `docs/architecture/inference_engines.md`. |
| `tq-model-manager` | Verify `gemma-4-e2b-it-vllm` is present (downloaded + `turboquant_kv.json` head_dim=256). Re-pull if missing. |
| `tq-system-info` | Re-confirm 4 GB VRAM + CUDA 12.8 + WSL2 UVA offloader path before running image tests (same hardware envelope as §C.2). |
| `tq-multi-process` | Drive §G vLLM multi-process CRM — start `tqcli serve start -m gemma-4-e2b-it-vllm --engine vllm`, spawn workers, assert continuous batching. |

Non-skill references:
- GitHub issues #3 (multimodal image/audio), #9 (hardware-aware vLLM config), #20 (Gemma 4 vLLM CPU offload), #22 (page-size fix) — all closed; do not regress them.

---

## Tests to land alongside the code

1. `tests/test_integration_turboquant_kv.py` — unchanged. Still 7/7.
2. `tests/test_headless_chat.py` — new. Cases:
   - `--prompt "2+2"` against `qwen3-4b-Q4_K_M` (llama.cpp) with `--kv-quant turbo3` → JSON `response` contains `"4"`, exit 0.
   - `--prompt ... --image tests/fixtures/test_image.png` against `gemma-4-e2b-it-vllm` (vLLM + CPU offload) → JSON `response` mentions the image colors, exit 0.
   - `--prompt "hi" --messages history.json` → history applied before prompt.
   - Bad image path → non-zero exit, JSON `error` payload.
   - `--prompt "describe this audio" --audio tests/fixtures/test_audio.wav` against Gemma 4 → graceful "no audio capability" text, exit 0.
3. `tests/integration_lifecycle.py` — flip the placeholder helpers (added in this PR's sibling commit):
   - `lifecycle_F_vllm_image_input_gemma4` → actually run headless chat with image, assert response length > 0.
   - `lifecycle_G_vllm_multiprocess_crm` → start server, spawn 2 workers, each worker generates one CRM module (frontend / backend / schema) via headless chat, verify files exist under a tmpdir.
4. `tests/test_integration_turboquant_kv.py::test_7_gemma4_e2b_vllm_cpu_offload` — once headless lands, replace the interactive `chat_thinking_turn` / `chat_simple_turn` with `headless_turn(...)` calls so the test is non-interactive.

---

## Regressions to watch

- `test_1`–`test_6` must stay green (pipeline-logic only).
- `test_7` on the same 4 GB VRAM WSL2 host: Gemma 4 E2B load ≈ 500–625 s, tokens/s ≈ 0.2, KV cache size 4,368 tokens, max concurrency 4.21× at 2,048 ctx. Any deviation of more than 10% on load time indicates a regression.
- Interactive REPL (`tqcli chat` with no flags) must still drop into `InteractiveSession.run_loop()` — no behavior change for existing users.
- `ChatMessage` dataclass shape stays backward compatible (new optional fields only).

---

## Acceptance criteria

- [ ] `tqcli chat --prompt "hello" --json` returns `{"response": "...", ...}` on stdout, exit 0.
- [ ] `tqcli chat --model gemma-4-e2b-it-vllm --prompt "What colors?" --image tests/fixtures/test_image.png --kv-quant turbo3 --json` produces an image-grounded answer, exit 0.
- [ ] `tqcli serve start -m qwen3-4b-AWQ --engine vllm` + 2 workers generate a three-file CRM under a tmp workspace; all `skill run` assertions in lifecycle §G pass.
- [ ] `tests/test_headless_chat.py` 5/5 pass.
- [ ] `tests/test_integration_turboquant_kv.py` still 7/7 (137/137 step assertions green).
- [ ] `docs/architecture/inference_engines.md` mermaid sequence updated to show the new multimodal path for vLLM.
