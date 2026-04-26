# README banner for `tqcli/llama-cpp-python-turboquant`

Insert at the very top of the fork's `README.md`, above the upstream content.

```markdown
> **TurboQuant fork notice.** This is a fork of
> [`abetlen/llama-cpp-python`](https://github.com/abetlen/llama-cpp-python)
> with TurboQuant KV cache compression baked into the bundled
> [`llama.cpp`](https://github.com/ggml-org/llama.cpp) build. The Python
> bindings layer is unchanged from upstream; the C++ vendor submodule points
> at [`tqcli/llama-cpp-turboquant`](https://github.com/tqcli/llama-cpp-turboquant)
> instead of `ggml-org/llama.cpp`.
>
> Install:
>
> ```bash
> pip install llama-cpp-python-turboquant
> ```
>
> The runtime sentinel `llama_cpp.TURBOQUANT_BUILD` is `True` on this fork
> and absent on upstream — `tqcli`'s engine auditor uses this to detect
> mismatched installs. **Not affiliated with the upstream `llama-cpp-python`
> project.** See [NOTICE](./NOTICE) for research-attribution details
> (TurboQuant ICLR 2026, PolarQuant AISTATS 2026, QJL AAAI 2025).

---
```
