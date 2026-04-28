# Product Requirements Document: TurboQuant Fork Wheel Distribution

> **Status update 2026-04-27 (late UTC) — W-A 0.3.1-tq2 fast-follow Path A shipping:** Five workflow iterations (each fixing a real bug masked by the prior one) revealed the actual W-A blocker: cibuildwheel for Linux runs builds inside a sandboxed manylinux Docker container, and the Jimver-on-host CUDA install (the original fast-follow plan) is invisible to nvcc inside that sandbox.
>
> **Path A (0.7.0, shipping in iteration #6):** drop the Linux Jimver step entirely; switch `CIBW_MANYLINUX_X86_64_IMAGE` to `pytorch/manylinux-builder:cuda12.8` (PyTorch's own CUDA-enabled manylinux2014 image, verified real on Docker Hub, 2025-02-25). nvcc + libcublas + libcurand at `/usr/local/cuda`. Windows path keeps Jimver-on-host (Windows cibuildwheel runs on the host, no sandbox).
>
> **Path B (0.7.1, follow-up):** switch to canonical PyPA `quay.io/pypa/manylinux_2_28_x86_64` and install CUDA inside the container at build time via `CIBW_BEFORE_ALL_LINUX` (DNF — manylinux_2_28 is AlmaLinux 8). Removes our dependency on a third-party CUDA image staying current. Trade-off: +5 min × 3 cells (~15 min added per build). Tracked at **[tqcli/llama-cpp-python-turboquant#3](https://github.com/tqcli/llama-cpp-python-turboquant/issues/3)**.
>
> Other side-effect fixes baked into 0.3.1-tq2: `python -m pip` for Windows pip self-upgrade, `non-cuda-sub-packages` for NVIDIA's CUDA-12 lib namespace, `\|`-block + quoted CMAKE_ARGS, macos-13 native runners replaced by macos-14 cross-build for x86_64 with `-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON` SIMD baseline. Full iteration table in `docs/prompts/ship_turboquant_wheels.md` Section 4.6.

> **Status update 2026-04-27 (early UTC) — build phase in progress:** Workstreams tagged + builds kicked off. W-A on GitHub Actions, W-B on GCP VM-self-driving tmux. Three real build bugs surfaced and patched on the VM (must be backported to fork before next release):
> 1. `setuptools_scm` rejects `v0.7.0-tq1` git tag (non-PEP-440) — fix: `SETUPTOOLS_SCM_PRETEND_VERSION=0.7.0.post20260426` env var
> 2. vLLM `setup.py` auto-throttles to `-j=1` under RAM pressure (32 GB n2-standard-8) — fix: 16 GB swap + explicit `MAX_JOBS=8`/`NVCC_THREADS=2`
> 3. `_build_one_wheel.sh` `--no-isolation` build-deps list missing `setuptools-scm`, `packaging`, `cmake`, `jinja2`, `numpy`, plus version pins — fix: install full `[build-system].requires` verbatim
>
> Build run history this cycle: run #1 (Sun 13:42 UTC) failed on bug 3; run #2 (Sun 15:39 UTC) failed on bug 1 again (pyproject.toml hack didn't fix it because setup.py calls setuptools_scm directly); run #3 (Mon 01:58 UTC) addresses bugs 1+2+3 and is the live run as of this update. Sunk cost on previous attempts: ~$5 of GCP compute. Build now VM-self-driving in detached tmux — survives local WSL2 reboots.
>
> W-A status: tag `v0.3.0-tq1` pushed; cibuildwheel matrix succeeded for CPU + sdist + Metal cells; **all CUDA cells failed** at `Install CUDA 12.8 toolkit` step using `Jimver/cuda-toolkit@v0.2.16` (action-version compatibility issue). Fast-follow: 0.3.1-tq2 with `@v0.2.19+`. CPU/Metal wheels can publish from current run; CUDA users fall back to source build until 0.3.1.
>
> Monitoring (no Claude session needed): `gsutil ls gs://tqcli-wheel-build/0.7.0-tq1/_status/` shows per-wheel progress sentinels.

> **Status update 2026-04-26 (afternoon):** First `/project-manager` orchestration run completed. Surfaced critical findings; release is **PAUSED** at multi-workstream prep:
> - **Workstream A — BLOCKED on fork-target mismatch.** Worker found that `tqcli/llama-cpp-turboquant` is a fork of `ggml-org/llama.cpp` (the C++ engine), NOT `abetlen/llama-cpp-python` (Python bindings). Publishing the C++ fork as `llama-cpp-python-turboquant` would ship helper scripts under a name that promises Python bindings — breaks `from llama_cpp import Llama`. Resolution path: create `tqcli/llama-cpp-python-turboquant` (fork from `abetlen/llama-cpp-python`), point its `vendor/llama.cpp` submodule at the existing C++ TurboQuant fork, re-register PyPI Pending Publisher (current 0.C reg keyed to wrong repo). 6 patches staged at `patches/llama-cpp-turboquant/`.
> - **Workstream B — STAGED, build paused.** Worker authored sentinels, runtime arch check, sequential GCP build script, RELEASING runbook, release body, RunPod verification commands. Did NOT execute the cross-repo work (correctly — needs maintainer to apply patches to fork, identify golden commit, run verification on real GPU). 7+ artifacts at `patches/vllm-turboquant/wheel_distribution_v0.7.0-tq1/`. **GCP build deliberately not kicked off until Workstream A is also resolved (Option 3 — cohesive launch).**
> - **Workstream C — DONE, pin-blocked.** engine_auditor.py (15 tests pass), pyproject.toml extras updated, FUNDING.yml routing to `ithllc`, community_verify.sh, all docs, CHANGELOG block, release drafts (uses `tq-linkedin` skill). Final pin `vllm-turboquant==0.7.0.postYYYYMMDD` blocked on Workstream B's actual publish.
> - **New skills created (release-engineering toolkit):** `tq-pre-release-verify` (would have caught the W-A fork-target mismatch), `tq-cross-repo-prep` (encodes the W-A+B prep pattern), `tq-wheel-orchestrator` (wheel matrix design + GCP build for big wheels), `tq-release-conductor` (umbrella coordinator). All four at `.claude/skills/tq-*/`.
> - **`.claude/settings.json` created** with Write/Edit permissions under `.claude/skills/worktrees/**` for future orchestrator runs.

> **Status update 2026-04-26:** Sections 0.C + 0.D of the playbook closed; wheel-naming, GCP build strategy, and verification provider locked:
> - **0.C complete** — PyPI Pending Publisher registered for `llama-cpp-python-turboquant` (owner `tqcli`, repo `llama-cpp-turboquant`, workflow `wheels.yml`, environment Any). Auto-promotes Pending → Active on first run.
> - **0.D GCP** — project `tqcli-wheel-build` provisioned against billing account `01124B-E52669-78A9D0`. APIs enabled, $50 budget alert at 50/90/100%, bucket `gs://tqcli-wheel-build/` ready in us-central1. **Build path: sequential single n2-standard-8 VM** (no quota request); six wheels back-to-back, ~30h wall time, ~$12.
> - **0.D verification** — Vast.ai + Lambda Labs replaced by **RunPod** (`runpodctl`). V6 (sm_121, GB10) stays on the user's owned ASUS Ascent GX10 (RunPod does not carry GB10).
> - **vllm wheel split (Decision #1, locked)** — wheel splits into **`vllm-turboquant`** (Ampere/Ada/Hopper: sm_8.0/8.6/8.9/9.0) and **`vllm-turboquant-blackwell`** (Blackwell DC + consumer + DGX Spark + Rubin PTX hedge: sm_10.0/12.0/12.1+PTX). Runtime sentinel `vllm.TURBOQUANT_BUILD_ARCH` distinguishes the variant; importing the wrong wheel raises a clear RuntimeError. README install table maps GPU → install command. Section 6 Key Feature 6 ("BNB_INT4 + CPU offload + turboquant35 Gemma 4 path") still holds; the pinned commit applies to both flavors.
> - **Architecture list** — `TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0 12.1+PTX"` is now split across the two wheels (4 arches in `vllm-turboquant`, 3 arches + PTX hedge in `vllm-turboquant-blackwell`). CUDA 13.0 toolkit required (12.8 cannot compile sm_121).
> - **`FUNDING.yml`** — `github: ithllc` only. `ithllc` is "Ivey Technology Holdings LLC" (the user's business; tqCLI is a product of ithllc until spinout). Sponsor revenue routes to the LLC's business account.
>
> Sections 3 ("Out of Scope" item: "Publishing `vllm-turboquant` to PyPI"), 4 (Key Features 6: single wheel), and 8 (Release & Documentation Requirements) need body updates to reflect the split + RunPod + sequential build. Use this preamble as authoritative until those edits land.

> **Status update 2026-04-25:** Sections 0.A and 0.B of the playbook
> (`docs/prompts/ship_turboquant_wheels.md`) are complete:
> - GitHub org `tqcli` is live; repos transferred from `ithllc/`.
> - PyPI distribution name for the umbrella package is **`turboquant-cli`** (not
>   `tqcli`, which is taken by an unrelated project, TranQuant). Import name
>   remains `tqcli` (dateutil pattern). All `pip install tqcli[...]` examples
>   below should be read as `pip install turboquant-cli[...]` until Workstream C
>   step 1 performs the global rename.
> - License switched from MIT to **Apache-2.0** for the umbrella `tqcli`
>   package. Forks retain their distinct upstream licenses (see Section 3
>   Constraints — `License (forks)` line). New artifacts at repo root: `LICENSE`,
>   `NOTICE`, `CITATION.cff`.
> - 0.0.0 placeholder published live at https://pypi.org/project/turboquant-cli/.
> - PyPI Trusted Publishing via OIDC is wired up at
>   `tqcli/tqcli:.github/workflows/publish-pypi.yml` — same pattern required for
>   `llama-cpp-python-turboquant` per Section 0.C.

## 1. Introduction
**Product/Feature Name:** TurboQuant Fork Wheel Distribution (tqCLI 0.7.0)

**Elevator Pitch:** Make `pip install tqcli[llama-tq]` and `pip install tqcli[vllm-tq]` deliver the TurboQuant-capable forks of llama.cpp and vLLM as pre-built wheels so every public user gets working TurboQuant KV cache compression out of the box — no source builds, no silent upstream fallback, no "why doesn't `--kv-quant turbo3` do anything" bug reports.

**Release coupling note (updated 2026-04-19):** The tri-state agentic autonomy work shipped in `0.6.0` (commit `c0457fd`), and 0.6.1/0.6.2 added KV-metadata auto-calibration + multi-architecture wrappers (Llama 3 / Mistral / Phi-3). Wheel distribution is now the **0.7.0** workstream, standalone — no co-shipping gate on other features. The original "release train + cargo" coupling was dissolved when agent modes shipped without the wheels.

**Problem Solved:** Today `pyproject.toml` pulls upstream `llama-cpp-python` and `vllm` via the `[llama]` / `[vllm]` extras. Upstream does not understand the TurboQuant flags emitted by `tqcli/core/kv_quantizer.py` (`cache_type_k=turbo3/4/2`, `kv_cache_dtype=turboquant35`, `enable_turboquant=True`, `attention_backend=TRITON_ATTN`). The CLI's graceful-fallback path silently downgrades to `kv:none` without telling the user, so the headline feature appears broken after a clean install. This blocks the public LinkedIn launch and the GitHub Sponsors page.

**Type:** Distribution / release-engineering feature added to the existing `tqCLI` product. Ships as tqCLI 0.7.0.

## 2. Target Audience
- **Public open-source end users** installing tqCLI from GitHub for the first time after the LinkedIn announcement.
- **Ampere+ NVIDIA users (RTX 30/40/50, A-series, H-series, on CUDA 12.8+)** who need the vLLM fork to get `turboquant35` + BNB_INT4 + CPU offload on Gemma 4 E2B/E4B.
- **macOS (Apple Silicon + Intel), Linux CPU, Windows CPU, older CUDA GPUs** — all covered by the llama.cpp fork, which is a superset of upstream llama-cpp-python and falls back gracefully to `kv:none` where TurboQuant kernels aren't available.
- **GitHub Sponsors prospects** — the install must feel professional and "just work" the first time.

## 3. Scope & Constraints

### In Scope
- Rename the two forks' PyPI/package names to `llama-cpp-python-turboquant` and `vllm-turboquant` so `pip` cannot accidentally resolve upstream when a user types the tq extra.
- Build + host pre-built wheels for both forks:
  - **`llama-cpp-python-turboquant`** — full matrix via `cibuildwheel` on GitHub Actions: Linux x86_64 (CPU + CUDA 12.8), Windows x86_64 (CPU + CUDA 12.8), macOS arm64 (Metal), macOS x86_64 (CPU). Published to **PyPI**. (CUDA 12.1 dropped per 2026-04-19 audit — TurboQuant kernels require 12.8+ anyway; older-CUDA users stay on CPU wheels.)
  - **`vllm-turboquant`** — single wheel: Linux x86_64 + CUDA 12.8 + Python 3.10/3.11/3.12, **one-off manual build** on the maintainer's WSL2 workstation (vLLM compile needs 16–32 GB RAM; cheapest path to launch). Published as a **GitHub Release asset** on the `ithllc/vllm-turboquant` repo.
- **Replace** the existing `[llama]` and `[vllm]` extras in `pyproject.toml` with `[llama-tq]` and `[vllm-tq]`. Update `[all]` to `[llama-tq,vllm-tq]`. No upstream-pulling extras remain.
- Pin `vllm-turboquant` to the commit after Issue #22's four-patch page-size-unification fix landed (the 2026-04-16 patches in `patches/vllm-turboquant/issue_22_page_size_fix.md`), NOT the earlier 2026-04-17 run pre-patch. Maintainer identifies the exact SHA at release time from the fork's git log. Verified good: BNB_INT4 + CPU offload + turboquant35 on Gemma 4 E2B, plus Qwen 3 4B calibration path from 0.6.1.
- New runtime **Engine Auditor** in `tqcli/core/system_info.py` that detects (a) whether the installed engine is the fork vs. upstream, (b) hardware capability (CUDA ≥ 12.8, SM ≥ 8.6), and prints a loud colored message at CLI startup when a user is on a capable GPU but running vanilla vLLM or vanilla llama-cpp-python. The Auditor MUST: (1) stay silent in `--json` headless mode and emit its findings as a structured field in the JSON metadata object on stderr; (2) flush its Rich panel to stderr BEFORE the agent orchestrator's first `chat_stream` call in `--ai-tinkering` or `--stop-trying-to-control-everything-and-just-let-go` modes, so the panel cannot interleave with streamed tool-call text or observations; (3) expose an `engine_auditor.get_status()` internal API so future agent tools (e.g., a tool-mode `tq-system-info`) can report "TurboQuant: active|fallback" to the LLM without re-detecting.
- `.github/FUNDING.yml` with the GitHub Sponsors link.
- Update `README.md`, `docs/GETTING_STARTED.md`, `docs/architecture/turboquant_kv.md`, and `docs/architecture/inference_engines.md` to reflect the new install paths.
- Update `CHANGELOG.md` with a `[0.7.0]` block (current tip is 0.6.2).
- Bump `tqcli/__init__.py` and `pyproject.toml` version from `0.6.2` to `0.7.0`.
- **macOS `[all]` fallback (added per 2026-04-19 audit):** `pyproject.toml` extras cannot be conditional on platform. Solution: keep `[all] = ["tqcli[llama-tq,vllm-tq]"]` but document prominently in `README.md`, `docs/GETTING_STARTED.md`, and the Engine Auditor panel that macOS users must install `[llama-tq]` directly (not `[all]`) because `vllm-turboquant` has no Darwin wheel. `[all]` will hard-fail on Mac — by design, with a clear error message.

### Out of Scope
- Publishing `vllm-turboquant` to PyPI (the 2 GB wheel exceeds PyPI's default 100 MB cap; not worth fighting for v1).
- Continuous CI builds of `vllm-turboquant` wheels on every commit to the fork. Manual releases only for v1.
- macOS / Windows builds of `vllm-turboquant` (vLLM requires Linux + CUDA).
- Automatic cross-CUDA-version builds for vLLM (only 12.8 for v1 — older CUDA users stay on the llama.cpp fork).
- A bespoke `tqcli install-backends` helper command (defer to v0.7.0 if user-confusion bug reports come in).
- Any change to the existing `kv_quantizer.py`, `vllm_backend.py`, or `llama_backend.py` code paths beyond adding the Engine Auditor hook.

### Constraints
- **Solo-dev budget: $0.** Must use only free GitHub Actions minutes (unlimited on public repos with standard runners), free PyPI hosting, free GitHub Releases asset hosting, and the maintainer's own WSL2 workstation for the one-off vLLM build.
- **No paid large runners, no paid cloud GPU time.** If the free 4-vCPU / 16 GB `ubuntu-latest` runner OOMs on the `llama-cpp-python-turboquant` CUDA build, we accept longer build times (ccache, incremental builds, `MAX_JOBS=2`) rather than a paid tier.
- **License (forks):** inherited from upstream. `llama-cpp-python-turboquant` is MIT (upstream `abetlen/llama-cpp-python`, Andrei Betlen); `vllm-turboquant` is Apache 2.0 (upstream `vllm-project/vllm`, preserve NOTICE). Both forks' LICENSE files must credit their respective upstreams.
- **License (umbrella `tqcli`):** Apache-2.0 (switched from MIT on 2026-04-25 — see status preamble at top of doc). Required artifacts at repo root: `LICENSE` (canonical Apache-2.0), `NOTICE` (research attribution for TurboQuant / PolarQuant / QJL + independent-implementation disclaimer + upstream-fork license declarations), `CITATION.cff` (GitHub "Cite this repository" rendering). Workstream C must preserve and update these artifacts on the 0.7.0 release.
- **No PyPI squatting.** The renamed packages make clear in their PyPI description that they are forks, not the canonical `llama-cpp-python` / `vllm`.

## 4. Key Features

1. **Renamed fork packages.** `llama-cpp-python` → `llama-cpp-python-turboquant`, `vllm` → `vllm-turboquant`. Both install into the same `llama_cpp` / `vllm` Python import namespaces, so `tqcli/core/*_backend.py` needs zero import changes.

2. **Pre-built wheel matrix for `llama-cpp-python-turboquant`.** `cibuildwheel`-driven GitHub Actions build on push-to-tag. Matrix covers CPython 3.10 / 3.11 / 3.12 across Linux (x86_64 CPU + CUDA 12.8), Windows (x86_64 CPU + CUDA 12.8), macOS (arm64 Metal + x86_64 CPU). Uploaded to PyPI via Trusted Publishing (no long-lived API tokens). CUDA 12.1 is NOT supported — TurboQuant requires 12.8+.

3. **One-off manual wheel for `vllm-turboquant`.** Maintainer runs `python -m build --wheel` on WSL2 once per release, pinning the Gemma 4 + BNB_INT4 + CPU offload + turboquant35 commit. Wheel is attached to a GitHub Release on `ithllc/vllm-turboquant`. Users install with `pip install tqcli[vllm-tq] --find-links https://github.com/ithllc/vllm-turboquant/releases/download/<tag>/`.

4. **Opinionated `pyproject.toml` extras.** Only TurboQuant-capable extras exist:
   ```toml
   [project.optional-dependencies]
   llama-tq = ["llama-cpp-python-turboquant>=0.3.0"]
   vllm-tq  = ["vllm-turboquant==<pinned-sha-version>", "bitsandbytes>=0.43.0", "accelerate>=0.30.0"]
   all      = ["tqcli[llama-tq,vllm-tq]"]
   dev      = ["pytest>=7.0", "ruff>=0.4"]
   ```

5. **Engine Auditor (loud fork-vs-upstream detection).** On CLI startup, `tqcli system info` and every `tqcli chat` invocation checks: does `llama_cpp` / `vllm` import expose a TurboQuant sentinel (e.g. `llama_cpp.TURBOQUANT_BUILD = True`, `vllm.TURBOQUANT_ENABLED = True`)? If not, and hardware is TurboQuant-capable, print a high-visibility colored block with the exact `pip install` command to fix it. If hardware is incapable (e.g. Pascal, no CUDA), stay silent — graceful fallback continues to apply. In agent modes, the auditor's findings ALSO surface via `engine_auditor.get_status()` so the LLM can explain to itself (and the user) why KV-cache growth is 4× larger than expected on a mis-configured install — this prevents the agent from hallucinating explanations for behavior it could have observed directly.

6. **BNB_INT4 + CPU offload + turboquant35 Gemma 4 path.** The pinned `vllm-turboquant` release corresponds to the commit AFTER Issue #22's four-patch page-size-unification fix landed (see `patches/vllm-turboquant/issue_22_page_size_fix.md`). Verified good: Gemma 4 E2B on 4 GB VRAM + 9.9 GB `cpu_offload_gb` + `kv_cache_dtype=turboquant35` AND Qwen 3 4B + calibrated `turboquant_kv.json`. PyPI version string encodes the pin (e.g. `0.7.0.postYYYYMMDD` where the date matches the pinned SHA's commit date).

7. **GitHub Sponsors page.** `.github/FUNDING.yml` with the `github: ithllc` line so the Sponsor button appears on the repo.

## 5. User Stories

- **As a new user on Ampere (RTX 3060, CUDA 12.8)**, I want `pip install tqcli[vllm-tq]` to pull the TurboQuant vLLM fork so that `tqcli chat --kv-quant turbo3` actually compresses my KV cache by 4.6× on the first try.

- **As a user on Apple Silicon (M2 Mac)**, I want `pip install tqcli[llama-tq]` to give me a Metal-accelerated TurboQuant llama.cpp build so that I can run Gemma 4 with KV compression on my MacBook without any CUDA hardware.

- **As a user on an older GPU (GTX 1080, CUDA 11.x)**, I want tqCLI to still install and run — just with `kv:none` — and I want the Engine Auditor to tell me *why* TurboQuant isn't active rather than staying silent.

- **As a new user reading the README**, I want exactly one install command per backend with zero "also clone this fork and build it from source" steps.

- **As the maintainer**, I want to cut a new `vllm-turboquant` release by running a single documented script on my WSL2 box, uploading the wheel to GitHub Releases, and bumping the pinned version in `pyproject.toml` — nothing more.

- **As a LinkedIn reader**, I want to click the repo, see a Sponsor button, install in one command, and have TurboQuant work on the first `tqcli chat` — so I sponsor and share the post.

## 6. Technical Requirements

### Tech Stack
- **Python 3.10 / 3.11 / 3.12** (matches current `requires-python` and the vLLM support matrix).
- **`cibuildwheel` ≥ 2.19** for the llama-cpp-python-turboquant matrix.
- **`build` ≥ 1.0** (PEP 517) for the manual vLLM wheel.
- **GitHub Actions** — standard runners only (`ubuntu-latest`, `windows-latest`, `macos-14` for arm64, `macos-13` for x86_64).
- **PyPI Trusted Publishing (OIDC)** — no long-lived tokens in repo secrets.
- **GitHub Releases** as a binary artifact host for `vllm-turboquant` wheels.

### Inputs
- Git tag on `ithllc/llama-cpp-turboquant` (e.g. `v0.3.0-tq1`) triggers the cibuildwheel workflow.
- Manual wheel upload to `ithllc/vllm-turboquant` release (e.g. `v0.7.0-tq1`).
- Hardware capability data from `tqcli/core/system_info.py::SystemInfo` (already built).

### Outputs
- PyPI package: `llama-cpp-python-turboquant` with platform-specific wheels.
- GitHub Release on `ithllc/vllm-turboquant`: one `.whl` per supported `(Python version × CUDA 12.8)` combo.
- Updated `pyproject.toml`, `CHANGELOG.md`, `README.md`, `docs/GETTING_STARTED.md`, `docs/architecture/turboquant_kv.md`, `docs/architecture/inference_engines.md`.
- New module: `tqcli/core/engine_auditor.py` (or extension of `system_info.py`).
- New file: `.github/FUNDING.yml`.

### Integration
- **No changes** to `tqcli/core/kv_quantizer.py`, `tqcli/core/vllm_backend.py`, `tqcli/core/llama_backend.py` internals. The Engine Auditor is called from `tqcli/cli.py` and `tqcli/ui/console.py` at startup.
- **Sentinel additions in the forks** — maintainer adds a one-line `TURBOQUANT_BUILD = True` module attribute to `llama_cpp/__init__.py` in the llama fork and `TURBOQUANT_ENABLED = True` in `vllm/__init__.py` of the vllm fork, so tqCLI can distinguish fork from upstream without parsing version strings.
- **CUDA compatibility gate** in `check_turboquant_compatibility` (already exists) stays authoritative. The Engine Auditor only *reports*; it does not override.

### Security
- PyPI Trusted Publishing via OIDC (no PyPI token in repo secrets).
- GitHub Release artifact signatures via GitHub's built-in attestation (Sigstore) on the llama fork's cibuildwheel workflow.
- SHA256 checksums published alongside each `vllm-turboquant` wheel in the GitHub Release body.
- No network calls at import time from either fork beyond what upstream already does.

## 7. Success Metrics

- **Zero-surprise install:** `pip install tqcli[vllm-tq]` on a fresh Ubuntu 22.04 + CUDA 12.8 + RTX 30/40 system must run `tqcli chat --kv-quant turbo3` with active TurboQuant KV compression on the first try, verified by `/stats` showing the fork's compressed KV cache size (not the upstream default).
- **First-run feature discovery:** On capable hardware with upstream engines installed, the Engine Auditor must print its colored "you're missing TurboQuant" message within the first 2 seconds of `tqcli` startup. Measured: 100% of test runs on a staging GPU emit the message exactly once.
- **macOS parity:** `pip install tqcli[llama-tq]` on an M-series Mac produces a Metal-accelerated binary that runs `tqcli chat --kv-quant turbo4` on Gemma 4 E2B with PPL impact ≤ +0.5% vs. `kv:none` on the same prompt set.
- **Install size:** `llama-cpp-python-turboquant` wheels ≤ 200 MB each (CUDA variants); CPU-only wheels ≤ 30 MB. `vllm-turboquant` wheel ≤ 2.5 GB (upstream is ~2 GB).
- **Time to launch:** All success metrics measurable within 7 days of starting implementation, enabling the LinkedIn announcement the following weekend.

## 8. Release & Documentation Requirements (v0.7.0)
To properly include this feature in the `0.7.0` release, the following files must be updated. **Note:** 0.6.0, 0.6.1, and 0.6.2 already shipped (agent modes, KV metadata calibrator, multi-arch wrappers respectively); this is a standalone wheel-distribution release.

- `pyproject.toml`: replace `[llama]` and `[vllm]` extras with `[llama-tq]` and `[vllm-tq]`; update `[all]`. Version bump from `0.6.2` to `0.7.0`.
- `tqcli/__init__.py`: bump fallback `__version__` to `0.7.0`.
- `CHANGELOG.md`: add a new `[0.7.0]` block at the top.
- `README.md`: replace the "Install" commands with the `-tq` variants; add a "What's new in 0.7.0" section. Do NOT clobber the existing 0.6.0/0.6.1/0.6.2 blocks.
- `docs/GETTING_STARTED.md`: update Step 1 install commands and the platform-specific sections (macOS, Linux, WSL2, Windows) to use `[llama-tq]` / `[vllm-tq]`. Explicitly document that macOS users install `[llama-tq]` directly, NOT `[all]`, because `vllm-turboquant` has no Darwin wheel.
- `docs/architecture/turboquant_kv.md`: add a "Distribution" section linking to the PyPI page and the GitHub Release for the vLLM fork.
- `docs/architecture/inference_engines.md`: document the Engine Auditor and the fork sentinel attributes.
- `docs/architecture/agent_orchestrator.md` (already shipped): cross-reference the Engine Auditor and note the stderr-flush ordering contract (Key Feature 5). Short paragraph, no new file.
- `docs/examples/USAGE.md`: update the `[vllm]` install example to `[vllm-tq]`.
- `.github/FUNDING.yml`: create with `github: ithllc`.
- `docs/contributing/RELEASING_WHEELS.md` (new): maintainer runbook for cutting a new `vllm-turboquant` wheel from WSL2.
- `Dockerfile` / any container image: update to use `pip install tqcli[all] --find-links <vllm-turboquant-release>` so the container gets the forks. If no Dockerfile exists in-tree, add this as a deployment-notes bullet to `docs/GETTING_STARTED.md` rather than skipping it.
- Release notes on the `v0.7.0` GitHub Release should lead with the one-command install story — the whole point of this milestone.
