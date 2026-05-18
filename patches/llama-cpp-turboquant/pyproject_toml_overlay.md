# `pyproject.toml` overlay for `tqcli/llama-cpp-python-turboquant`

Diff-shaped overlay against upstream `abetlen/llama-cpp-python/pyproject.toml`.
The `[build-system]` / `[tool.scikit-build]` blocks should match upstream
verbatim — only the `[project]` name + `[tool.cibuildwheel]` block are
TurboQuant-specific.

```toml
[project]
# WAS: name = "llama_cpp_python"
name = "llama-cpp-python-turboquant"
description = "TurboQuant fork of llama-cpp-python with KV cache compression (turbo2/turbo3/turbo4 cache types). Not affiliated with the upstream llama-cpp-python project."
# Keep version aligned with upstream + a -tqN suffix:
#   version = "0.3.0.post1+tq1"   ← PEP 440 local version segment, OR
#   version = "0.3.0"             ← match upstream and tag the *git* ref as v0.3.0-tq1
# The prompt asks for a v0.3.0-tq1 git tag, so the project version stays "0.3.0";
# `+tq1` would block PyPI publish (PyPI strips local segments). Use the tag.
version = "0.3.0"
requires-python = ">=3.10"
license = { text = "MIT" }   # inherited from abetlen/llama-cpp-python — keep
authors = [{ name = "tqcli" }, { name = "Andrei Betlen", email = "abetlen@gmail.com" }]
urls = { Homepage = "https://github.com/tqcli/llama-cpp-python-turboquant" }
# ... rest of the upstream [project] table stays unchanged ...

[tool.cibuildwheel]
# Project-level fallback for `cibuildwheel --output-dir wheelhouse` from a
# developer's local box; full matrix logic lives in wheels.yml.
build = "cp310-* cp311-* cp312-*"
skip = "*-musllinux* *-manylinux_i686 pp*"
test-command = 'python -c "import llama_cpp; assert llama_cpp.TURBOQUANT_BUILD is True; print(llama_cpp.TURBOQUANT_KV_TYPES)"'

[tool.cibuildwheel.linux]
archs = ["x86_64"]

[tool.cibuildwheel.windows]
archs = ["AMD64"]

[tool.cibuildwheel.macos]
# arm64 vs x86_64 selected by runner OS in wheels.yml
```

## Notes

- **Keep the `MIT` license** — inherited from upstream `abetlen/llama-cpp-python`.
  The umbrella `tqcli/tqcli` package switched to Apache-2.0 (Section 0.B), but
  forks keep their inherited licenses (memory: `project_license_apache_2_0`).
- **Distribution name uses hyphens** (`llama-cpp-python-turboquant`); import
  name stays `llama_cpp` (no underscore, no `_turboquant` suffix). This
  matches the dateutil pattern (memory: `project_pypi_distribution_name`).
- **Version `0.3.0`** matches the git tag `v0.3.0-tq1` semantically while
  staying PyPI-publishable. The `-tq1` part lives in the git ref only; PyPI
  rejects PEP 440 local-version segments on upload.
