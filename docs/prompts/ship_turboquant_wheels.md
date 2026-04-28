# Prompt: Ship TurboQuant Wheels (tqCLI 0.7.0)

**Target release:** tqCLI 0.7.0 — TurboQuant fork wheel distribution + LinkedIn launch
**Mode:** `/project-manager` orchestrates three parallel worktrees; `gemini -p` is used at specific checkpoints; human intervenes only at marked gates
**Budget ceiling:** $250/day (on-demand path typically lands at ~$20)
**Reference docs:**
- PRD: `docs/prd/PRD_turboquant_wheel_distribution.md`
- TP: `docs/technical_plans/TP_turboquant_wheel_distribution.md`
- Launch memory: `project_oss_launch_readiness.md`

---

## Architecture scope (locked 2026-04-22)

`TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0 12.1+PTX"` — Ampere + Ada + Hopper + DC Blackwell (sm_100) + consumer Blackwell (sm_120) + DGX Spark/GB10 (sm_121) + PTX hedge for Rubin. **Requires CUDA 13.0+ toolkit** (12.8 is insufficient for sm_121). Update PRD + TP before building.

**AMD ROCm:** out of scope for 0.7.0. **NVLink/ConnectX:** runtime-only, no wheel flags.

---

## Section 0: Human prerequisites (do BEFORE invoking the orchestrator)

Each step is solo-dev work that cannot be automated. Complete in order.

### 0.A — Create GitHub organization `tqcli`

1. Go to `github.com/account/organizations/new`, choose Free plan, name `tqcli`.
2. Set profile: bio = "TurboQuant CLI — local LLM inference with KV cache compression", URL = `tqcli.com` (secure the domain separately), email = iveyfra@gmail.com.
3. Transfer repos (Settings → Transfer ownership on each):
   - `ithllc/tqCLI` → `tqcli/tqcli` (also rename to lowercase `tqcli` during transfer)
   - `ithllc/llama-cpp-turboquant` → `tqcli/llama-cpp-turboquant`
   - `ithllc/vllm-turboquant` → `tqcli/vllm-turboquant`
4. Update local git remotes:
   ```bash
   cd /llm_models_python_code_src/tqCLI
   git remote set-url origin git@github.com:tqcli/tqcli.git
   ```
5. GitHub will redirect all old `ithllc/*` URLs permanently; existing clones continue to work.

### 0.B — Reserve the `turboquant-cli` PyPI name ✅ DONE 2026-04-25

The slug `tqcli` was unavailable (taken by an unrelated project, TranQuant), so we reserved **`turboquant-cli`** instead. Import name remains `tqcli` (dateutil pattern: `pip install python-dateutil` → `import dateutil`).

Mechanism: PyPI Trusted Publishing via Pending Publisher → first-publish from `tqcli/tqcli:.github/workflows/publish-pypi.yml` via OIDC. **No API tokens.** No `twine upload` from a local shell.

License decision during 0.B: switched the tqcli umbrella package from MIT to **Apache-2.0**. Drivers: TurboQuant lead author Amir Zandieh's QJL repo is Apache-2.0 (strongest precedent from the methodology's inventor); explicit patent grant + retaliation matter for research-derived AI; matches the `vllm-turboquant` fork's inherited Apache-2.0; same adoption ceiling as MIT. Required artifacts now at repo root:
- `LICENSE` — canonical Apache-2.0 (fetched from apache.org)
- `NOTICE` — research attribution (TurboQuant ICLR 2026, PolarQuant AISTATS 2026, QJL AAAI 2025) + independent-implementation disclaimer + upstream-fork license declarations
- `CITATION.cff` — CFF 1.2.0; GitHub renders as "Cite this repository"

For future PyPI work, use the **`tq-pypi` skill** at `.claude/skills/tq-pypi/`. For future license decisions, use the **`tq-license-review` skill** at `.claude/skills/tq-license-review/`.

Outcome:
- 0.0.0 placeholder live at https://pypi.org/project/turboquant-cli/
- Trusted Publisher auto-promoted from Pending → Active on first successful publish
- All future `turboquant-cli` versions must come from `tqcli/tqcli:.github/workflows/publish-pypi.yml` via OIDC
- `pip install turboquant-cli` resolves successfully on a clean venv

### 0.C — Trusted Publishing for `llama-cpp-python-turboquant` ✅ DONE 2026-04-26

Pending Publisher registered on PyPI: project `llama-cpp-python-turboquant`, owner `tqcli`, repo `llama-cpp-turboquant`, workflow `wheels.yml`, environment `(Any)`. Will auto-promote Pending → Active on first successful `wheels.yml` run (Workstream A).

Same pattern as 0.B. Use the **`tq-pypi` skill** at `.claude/skills/tq-pypi/` — it encodes the full flow (availability check, Pending Publisher form, OIDC workflow template, failure diagnostics). The `tqcli/tqcli:.github/workflows/publish-pypi.yml` placeholder workflow is the working reference template; rename to `wheels.yml` since this fork uses cibuildwheel (per TP Phase 2 / Workstream A).

**Pending Publisher form fields** (PyPI → Publishing → Add a pending publisher):

| Field | Value |
|---|---|
| PyPI Project Name | `llama-cpp-python-turboquant` |
| Owner | `tqcli` |
| Repository | `llama-cpp-turboquant` |
| Workflow filename | `wheels.yml` |
| Environment | (leave blank — matches what worked in 0.B) |

**Workflow requirements** (will be created in Workstream A):
- `permissions: id-token: write` at job level
- Uses `pypa/cibuildwheel@v2.19` for the matrix build
- Final `publish` job uses `pypa/gh-action-pypi-publish@release/v1` with **NO** `with.password:` argument
- Fork's `pyproject.toml` declares `name = "llama-cpp-python-turboquant"` BEFORE first publish

**Verification:** after first tag (`v0.3.0-tq1`) push triggers `wheels.yml`, run `bash .claude/skills/tq-pypi/scripts/check_name_available.sh llama-cpp-python-turboquant` — should report TAKEN by us.

**For `vllm-turboquant`:** PyPI publish is currently deferred (per PRD: distributed via GitHub Releases due to wheel-size question). When that resolves in PyPI's favor, use the same Pending Publisher pattern. NEVER use account-wide API tokens — user policy; the `tq-pypi` skill enforces this.

### 0.D — Cloud accounts (pay-as-you-go, no monthly commit) ✅ GCP DONE 2026-04-26

1. **GCP** ✅ Project `tqcli-wheel-build` provisioned 2026-04-26 against billing account `01124B-E52669-78A9D0` (`IveyTechnologyHoldings_Billing`). APIs enabled: `compute`, `cloudbilling`, `billingbudgets`, `storage`. Bucket `gs://tqcli-wheel-build/` ready in us-central1. $50 budget alert at 50/90/100% of actual spend. **Build strategy: sequential single n2-standard-8 VM** — six wheels built back-to-back (3 Python versions × 2 GPU flavors after the split, see Decision #1), ~30h wall time, ~$12 total. Stays within the default 8-vCPU regional quota; no quota increase needed.
2. **RunPod** ✅ Account validated 2026-04-26 (balance $500). CLI: `runpodctl` (`github.com/runpod/runpodctl`), supports headless `pod create`. API key stored outside the repo; treat as a secret, never commit. Confirmed-available GPUs (Community Cloud prices, 2026-04-26 GraphQL): RTX 4090 $0.34/hr, RTX 5090 $0.69/hr, B200 $5.98/hr. Total verification compute ~$7.69. Replaces Vast.ai + Lambda Labs.
3. **ASUS Ascent GX10** — owned (acquired 2026-04-25). Provides on-hand sm_121 (DGX Spark / GB10, Blackwell) verification for V6. RunPod does not carry GB10 — DGX Spark hardware is not in their catalog as of 2026-04-26.

### 0.E — Community Mac verifiers

Line up friends with:
- One M-series MacBook (arm64)
- One Intel MacBook (x86_64)
- One Mac mini (either is fine)

Each verifier will run `scripts/community_verify.sh --auto-report` after consenting to the data manifest. Script is drafted in Workstream C below.

### 0.F — Docs already set up

- PRD + TP exist at the paths above.
- `patches/vllm-turboquant/issue_22_page_size_fix.md` exists.
- Fork sentinels (`TURBOQUANT_BUILD`, `TURBOQUANT_ENABLED`) need to land in the forks during Workstream A1/B1.

---

## Section 1: Orchestration via `/project-manager`

Invoke:

```
/project-manager

Spawn three parallel Claude Code sessions in isolated Git worktrees. Each session works to the sub-prompt below. Sessions A and B produce artifacts (wheels published to PyPI / GitHub Release) that Session C pins in `pyproject.toml`. Start all three concurrently; Session C drafts in parallel and blocks only on the final merge.
```

### Workstream A — `llama-cpp-python-turboquant` wheels (free, cibuildwheel)

Hand this prompt to Worker A:

```
Branch: off `tqcli/llama-cpp-turboquant` main.

Goal: publish platform-matrix wheels to PyPI via cibuildwheel + Trusted Publishing.

Steps (read TP Phase 1 A1 + Phase 2):
1. In the fork's pyproject.toml, rename distribution to `llama-cpp-python-turboquant`. Do NOT change the `llama_cpp` import package name.
2. Add to src/llama_cpp/__init__.py:
     TURBOQUANT_BUILD = True
     TURBOQUANT_KV_TYPES = ("turbo2", "turbo3", "turbo4")
3. Update fork README.md top banner per TP A1.4.
4. Create .github/workflows/wheels.yml using pypa/cibuildwheel@v2.19.
   Matrix: ubuntu-latest + windows-latest + macos-14 (arm64) + macos-13 (x86_64), Python 3.10/3.11/3.12.
   CUDA variants: 12.8+ only. Metal on macOS-14. CPU on all.
   CIBW_ENVIRONMENT per variant sets CMAKE_ARGS correctly.
   CIBW_SKIP = "*-musllinux* *-manylinux_i686".
   Use MAX_JOBS=2 and ccache on Linux/Windows runners.
5. Add pypa/gh-action-pypi-publish@release/v1 in a final `publish` job, environment `pypi` (OIDC, no tokens).
6. Also attach wheels to the fork's GitHub Release via softprops/action-gh-release@v2.
7. Tag `v0.3.0-tq1` after PR merges — this triggers the workflow.

Verification:
- Workflow runs green end-to-end.
- pip install llama-cpp-python-turboquant on a clean venv on Linux/Mac/Windows succeeds.
- `python -c "import llama_cpp; print(llama_cpp.TURBOQUANT_BUILD)"` prints True.

Cost: $0 (GitHub Actions free for public repos).
Do NOT touch tqcli/ or the vllm fork.
```

### Workstream B — `vllm-turboquant` + `vllm-turboquant-blackwell` wheels on GCP (sequential single-VM)

Hand this prompt to Worker B:

```
Branch: off `tqcli/vllm-turboquant` main.

Goal: produce SIX vllm-turboquant wheels (3 Python versions × 2 GPU flavors) sequentially on a single GCP n2-standard-8 VM; attach to GitHub Release.

Wheel split (locked Decision #1, 2026-04-26):
- `vllm-turboquant`            — Ampere/Ada/Hopper, TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
- `vllm-turboquant-blackwell`  — Blackwell DC + consumer + DGX Spark + Rubin hedge, TORCH_CUDA_ARCH_LIST="10.0 12.0 12.1+PTX"

CUDA toolkit: 13.0+ (12.8 cannot compile sm_121).

Build strategy: ONE n2-standard-8 VM (8 vCPUs, no quota request needed), six builds sequentially. Each ~5h, total ~30h wall time, ~$12 compute cost. The build script flips pyproject.toml's name field, TORCH_CUDA_ARCH_LIST env var, and the runtime sentinels per flavor before each `python -m build` invocation.

Steps (read TP Phase 1 B1 + Phase 3):

1. In fork pyproject.toml, leave `name` as a build-time-templated field. The build script (step 7) writes either `vllm-turboquant` or `vllm-turboquant-blackwell` before each build. Preserve `vllm` import name in both flavors.

2. Add to vllm/__init__.py:
     TURBOQUANT_ENABLED = True
     TURBOQUANT_KV_DTYPES = ("turboquant25", "turboquant35")
     TURBOQUANT_BUILD_ARCH = ""        # populated at build time: "ampere-ada-hopper" or "blackwell"
     TURBOQUANT_BUILD_ARCH_LIST = ""   # populated at build time with the literal TORCH_CUDA_ARCH_LIST
   Build script writes the actual values into __init__.py before `python -m build`.

3. Add vllm/turboquant_arch_check.py with `check_arch_compatibility() -> Optional[str]`:
     - Detect runtime GPU compute capability via torch.cuda.get_device_capability().
     - Compare against TURBOQUANT_BUILD_ARCH_LIST.
     - Return clear error: "This wheel was built for Blackwell GPUs (sm_10.0/12.0/12.1+PTX). Detected your GPU as sm_8.6 (RTX 3090) — install `vllm-turboquant` instead."
     - Return None if compatible.
   Engine import path calls this on first GPU init; raises RuntimeError with the above message if mismatched. Goal: hard-fail fast with a clear message instead of silent fallback.

4. Update fork README.md per TP B1.4. Add the install compatibility table:
   | Your GPU                              | Install command                          |
   |---------------------------------------|------------------------------------------|
   | RTX 30/40-series, A100, H100, GH200   | pip install vllm-turboquant              |
   | RTX 50-series, B100/B200, GB10        | pip install vllm-turboquant-blackwell    |
   Confirm LICENSE stays Apache 2.0 + NOTICE intact.

5. Identify golden commit AFTER Issue #22 four-patch page-size fix (see patches/vllm-turboquant/issue_22_page_size_fix.md). Verify with:
     - Gemma 4 E2B + BNB_INT4 + CPU offload + turboquant35 (Section C.2 of comparison report)
     - Qwen 3 4B + calibrated turboquant_kv.json (0.6.1 path)
   Both must be green BEFORE tagging.

6. Tag `v0.7.0-tq1` on the fork.
7. Author scripts/build_wheel_gcp.sh that:
     - Provisions ONE GCP n2-standard-8 on-demand VM in us-central1 (default project: tqcli-wheel-build).
     - Installs CUDA 13.0 toolkit.
     - For each (flavor, py-version) in
         [(ampere-ada-hopper, 3.10), (ampere-ada-hopper, 3.11), (ampere-ada-hopper, 3.12),
          (blackwell, 3.10),         (blackwell, 3.11),         (blackwell, 3.12)]:
         a. Set pyproject.toml `name` = `vllm-turboquant` (ampere-ada-hopper) or `vllm-turboquant-blackwell` (blackwell).
         b. Set TORCH_CUDA_ARCH_LIST per flavor.
         c. Set MAX_JOBS=4, NVCC_THREADS=4, VLLM_TARGET_DEVICE=cuda.
         d. Write TURBOQUANT_BUILD_ARCH + TURBOQUANT_BUILD_ARCH_LIST into vllm/__init__.py.
         e. Run python -m build --wheel.
         f. Push wheel to gs://tqcli-wheel-build/0.7.0-tq1/.
     - Tears down the VM after all six builds complete (or on error, with a flag to keep-alive for debugging).
   Run as a SINGLE invocation. Wall time ~30h, cost ~$12.

8. Measure each wheel's size. If any single wheel still exceeds 2 GB despite the split: STOP. Escalate to user — fallback paths:
     (a) GitHub LFS (also has 2 GB per-file limit but separate quota)
     (b) Further granularity (split blackwell into DC sm_10.0 vs consumer/spark sm_12.0/12.1)

9. If all six are under 2 GB:
     gh release create v0.7.0-tq1 --repo tqcli/vllm-turboquant
     Upload all six .whl + SHA256SUMS. Release body includes Section C.2 numbers + the install table from step 4.

10. Document the script path + invocation in fork docs/RELEASING.md.

11. Confirm on a clean CUDA 13.0 Ubuntu VM:
     - pip install vllm-turboquant --find-links https://github.com/tqcli/vllm-turboquant/releases/expanded_assets/v0.7.0-tq1 resolves and works on RTX 4090 / A100.
     - pip install vllm-turboquant-blackwell --find-links ... resolves and works on RTX 5090 / B200.
     - Cross-flavor mismatch test: install vllm-turboquant on a Blackwell GPU → check_arch_compatibility() raises RuntimeError with the clear message.

Verification via RunPod (cost ~$7.69 total, Community Cloud per 2026-04-26 API check, see Section 3):
  - V3 (RTX 4090, ~3h Community Cloud, ~$1.02): pip install vllm-turboquant; python -c "import vllm; print(vllm.TURBOQUANT_ENABLED, vllm.TURBOQUANT_BUILD_ARCH)" prints (True, ampere-ada-hopper).
  - V4 (RTX 5090, ~1h Community Cloud, ~$0.69): pip install vllm-turboquant-blackwell; sentinel prints (True, blackwell).
  - V5 (B200, ~1h Community Cloud, ~$5.98): same as V4 with the larger card.

Do NOT touch tqcli/ or the llama fork.
```

### Workstream C — tqCLI 0.7.0 integration + community verify + docs

Hand this prompt to Worker C:

```
Branch: new `release/0.7.0` in /llm_models_python_code_src/tqCLI.

Goal: wire the forks into tqcli extras; implement Engine Auditor; update docs; create community verify script; draft release artifacts. Final merge blocks on Workstreams A + B publishing wheels.

Steps (read TP Phases 4, 5, 6):

1. Update PRD + TP with the new arch list (TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0 12.1+PTX") and CUDA 13.0 toolkit requirement. Also update all `ithllc/` URLs to `tqcli/` (repo paths: tqcli/tqcli, tqcli/llama-cpp-turboquant, tqcli/vllm-turboquant).

2. pyproject.toml: bump version 0.6.2 → 0.7.0. Replace [llama]/[vllm] extras with [llama-tq]/[vllm-tq]; update [all]. Delete old keys. (See TP C1 for exact block.)

3. tqcli/__init__.py: bump __version__ to 0.7.0.

4. Create tqcli/core/engine_auditor.py per TP C3 (EngineAuditResult dataclass, audit_llama_cpp, audit_vllm, run_audit).
   Add engine_auditor.get_status() module-level cached API (TP C5 new internal-API contract).

5. Add render_audit_warnings() in tqcli/ui/console.py per TP C4.

6. Wire into tqcli/cli.py:
     - run_audit(get_system_info()) at every CLI startup.
     - Skip when TQCLI_SUPPRESS_AUDIT=1.
     - --json: emit audit as stderr JSON metadata, not Rich panel.
     - Ordering contract (TP C5): render_audit_warnings() must console.file.flush() BEFORE AgentOrchestrator.__init__ in --ai-tinkering / --stop-trying-to-control-everything-and-just-let-go modes.

7. tests/test_engine_auditor.py per TP C6:
     - Mock llama_cpp with/without TURBOQUANT_BUILD; assert is_turboquant_fork flips.
     - Mock SystemInfo capable vs. incapable; assert should_warn flips.
     - Stderr-ordering assertion test for agent-mode flush-before-orchestrator.

8. Create scripts/community_verify.sh:
     - Prints consent manifest BEFORE any data collection.
     - Collects: macOS version, chip arch, Python version, pip install [llama-tq] result, tqcli system info output (anonymize $HOME), tqcli chat --kv-quant turbo4 --prompt "Two plus two?" --json result.
     - Does NOT collect: username, hostname, IP, free-form user input.
     - Two modes: --auto-report (uses gh CLI if installed; errors if not; does NOT ship tokens) / --manual (prints markdown block for user to paste into issue).
     - Exits non-zero if consent declined.
     - Script is idempotent; safe to re-run.

9. Create .github/ISSUE_TEMPLATE/community_verify_0_7_0.yml — pre-filled issue template the verify script points users to in --manual mode.

10. Create .github/workflows/community_verify_collect.yml — nightly scrapes the labeled issue, emits committed markdown to tests/integration_reports/community_verification/0.7.0/.

11. Create .github/FUNDING.yml:
      github: [ithllc, tqcli]
    (Both org and individual so sponsors can choose. Confirm with user if unsure.)

12. Update docs per TP Phase 6:
      - README.md: "What's new in 0.7.0" section; install commands swapped to -tq; macOS [all] caveat.
      - docs/GETTING_STARTED.md: install block + platform sections + troubleshooting row for the yellow panel.
      - docs/architecture/turboquant_kv.md: Distribution section linking to PyPI + GH Release.
      - docs/architecture/inference_engines.md: document sentinel attrs + Engine Auditor.
      - docs/architecture/agent_orchestrator.md: cross-reference Engine Auditor + stderr flush contract.
      - docs/examples/USAGE.md: update install example.
      - docs/contributing/RELEASING_WHEELS.md (new): maintainer runbook for cutting vllm-turboquant wheel from GCP.

13. CHANGELOG.md: add [0.7.0] block at top per TP C13. Do not touch 0.6.0/0.6.1/0.6.2 blocks.

14. Draft GitHub Release body at tests/release_drafts/v0.7.0.md — one-command install story leading, both wheels co-advertised.

15. Draft LinkedIn post at tests/release_drafts/linkedin_0.7.0.md — separate deliverable; user will edit.

Final merge blocker: Workstreams A + B must have published wheels with pinned versions before pyproject.toml's `vllm-turboquant==0.7.0.postYYYYMMDD` pin is valid.

Before marking complete, run TP C2a dependency harmony check:
  pip install -e ".[vllm-tq]" --find-links <vllm-turboquant-release>
  pip install -e ".[dev]"
  python -m pytest tests/test_agent_orchestrator.py -x
  python tests/test_integration_agent_modes.py
If any agent test fails with the fork installed but passed with upstream vllm, block release. Escalate to user.
```

---

## Section 2: Gemini CLI checkpoints

Invoke Gemini via `gemini -p "<prompt>"` at these specific points. Gemini runs non-interactively; capture output and apply changes.

### 2.A — Pre-build Blackwell arch cross-check (before Workstream B step 6)

```
gemini -p "I'm about to build a vLLM wheel (fork at github.com/tqcli/vllm-turboquant) with TORCH_CUDA_ARCH_LIST='8.0 8.6 8.9 9.0 10.0 12.0 12.1+PTX' using CUDA 13.0. Check the current vLLM main branch for any new compute-capability requirements or build-flag changes landed in the last 30 days that would affect this build. Also check if sm_121 (DGX Spark / GB10) has any known vLLM kernel gaps that would need a workaround. Cite specific PRs or issues. Report in under 200 words."
```

Apply any returned fixes before building.

### 2.B — Engine Auditor code review (Workstream C step 6, before commit)

```
gemini -p "Review the engine_auditor.py module below for: (1) import-error handling when llama_cpp or vllm is absent, (2) race conditions between stderr flush and the agent orchestrator's first stream chunk, (3) any path that could crash tqcli startup if the sentinel check throws unexpectedly. The module is pasted below. Flag concrete bugs only; no stylistic feedback. Respond in bullets.

<paste module content here>"
```

### 2.C — Community verify privacy review (Workstream C step 8, before commit)

```
gemini -p "Review community_verify.sh for privacy leaks. It's a shell script that end users run on their Macs; it auto-reports verification results to a GitHub issue. Enumerate EVERY piece of data it could possibly send or expose, including via pip install stderr, tqcli system info output, environment variables, shell history. Flag anything that could deanonymize a user or leak filesystem paths / hostnames. Paste script below.

<paste script content here>"
```

### 2.D — CHANGELOG + Release body polish (Workstream C step 14)

```
gemini -p "Edit this CHANGELOG 0.7.0 block and GitHub Release body for clarity, concision, and honesty. Do not add features or inflate claims. Flag any claim that is not substantiated by the PRD or TP. Strip marketing adjectives.

<paste both artifacts here>"
```

---

## Section 3: Verification matrix

Ordered by cost. Execute after Workstream B publishes the wheels.

| Cell | Platform | Hardware | Command | Cost |
|---|---|---|---|---|
| V1 | Own WSL2 | RTX A2000 (sm_8.6, Ampere) | Full `tqcli chat --kv-quant turbo3` on Gemma 4 E2B | $0 |
| V2 | Own Windows 11 Pro | RTX A2000 (same card, different OS) | llama.cpp CUDA path | $0 |
| V3 | RunPod Community Cloud 3 hr | RTX 4090 (sm_8.9, Ada) | `vllm-turboquant` + turboquant35 end-to-end | ~$1.02 |
| V4 | RunPod Community Cloud 1 hr | RTX 5090 (sm_12.0, Blackwell consumer) | `vllm-turboquant-blackwell` + turboquant35 — proves Blackwell consumer path | ~$0.69 |
| ~~V5~~ | ~~RunPod Community Cloud 1 hr~~ | ~~B200 (sm_10.0, Blackwell DC)~~ | **DEFERRED to 0.7.1 — RunPod B200 capacity unavailable on 2026-04-27 (`onDemandPrice=None spotPrice=None` for both Community and Secure clouds). V4 (RTX 5090, Blackwell consumer) proves the same wheel; B200 verification is marketing, not correctness. Tracked at [tqcli/tqcli#41](https://github.com/tqcli/tqcli/issues/41).** | n/a |
| V6 | Own ASUS Ascent GX10 | GB10 (sm_12.1, Blackwell DGX Spark) | vllm + turboquant35 — proves sm_121 path | $0 |
| V7 | Friend's M-series Mac | Apple Silicon Metal | `scripts/community_verify.sh --auto-report` | $0 |
| V8 | Friend's Intel Mac | x86_64 CPU | Same | $0 |
| V9 | Friend's Mac mini | Either chip | Same | $0 |

Assertions per cell (the `--json` output MUST contain):
- Engine Auditor silent on capable hardware with forks installed.
- TurboQuant metadata block present (`cpu_offload_gb`, `kv_cache_dtype=turboquant35`).
- Exit code 0.
- `--ai-tinkering` with closed stdin exits non-zero fast (TP V1 agent-mode smoke).
- Unrestricted headless run terminates within `max_agent_steps` without orphan vLLM workers.

**Total verification cost**: ~$1.71 (V3 + V4 only; V5 deferred to 0.7.1 — see #41).

---

## Section 4: Human gates

Execute only after Sections 1–3 complete clean.

### Gate G1 — Wheel publish confirmation
User visually confirms:
- `pypi.org/project/llama-cpp-python-turboquant/` shows all matrix wheels.
- `github.com/tqcli/vllm-turboquant/releases/tag/v0.7.0-tq1` shows three .whl files + SHA256SUMS.

### Gate G2 — Community verify intake
User confirms at least 2 Mac verifier reports landed in `tests/integration_reports/community_verification/0.7.0/` with PASS status.

### Gate G3 — Tag + release cut
Once G1 + G2 are green, user (not automation) runs:
```bash
cd /llm_models_python_code_src/tqCLI
git checkout main && git pull
git tag v0.7.0 && git push --tags
gh release create v0.7.0 --repo tqcli/tqcli --title "tqCLI 0.7.0 — TurboQuant wheel distribution" \
  --notes-file tests/release_drafts/v0.7.0.md
```

### Gate G4 — LinkedIn launch
Separate deliverable. User posts from `tests/release_drafts/linkedin_0.7.0.md` (edited to taste).

---

## Section 4.5: First-run findings (2026-04-26)

The first `/project-manager` orchestration run revealed:

1. **Workstream A fork-target mismatch.** `tqcli/llama-cpp-turboquant` is a fork of `ggml-org/llama.cpp` (C++ engine), NOT `abetlen/llama-cpp-python` (Python bindings). The PRD/TP and 0.C PyPI registration assumed the latter. Resolution: create new repo `tqcli/llama-cpp-python-turboquant` from `abetlen/llama-cpp-python`, re-register PyPI Pending Publisher with the new repo name. 6 staged artifacts at `patches/llama-cpp-turboquant/` drop in cleanly to the new repo.

2. **Workstream B prep complete, build paused.** All required artifacts authored. Cross-repo application + verification + tag deferred to maintainer. GCP build held until Workstream A resolves (Option 3 — cohesive launch).

3. **Workstream C complete.** Real code in branch `task-3-release-tqcli-1777190346`, blocked on B's wheel pin.

4. **`/project-manager` orchestrator caveat.** Workers in worktree-of-repo-X cannot push to repo Y. They correctly stage artifacts and exit. Cross-repo prep needs the maintainer (or `tq-cross-repo-prep` skill driving from the umbrella repo).

5. **New release-engineering skills created** to encode lessons: `tq-pre-release-verify`, `tq-cross-repo-prep`, `tq-wheel-orchestrator`, `tq-release-conductor`. The first three are immediately reusable; the conductor wraps the others.

## Section 4.6: Build-execution findings (2026-04-26 evening → 2026-04-27 early UTC)

After Sections 0.A–0.D + cohesive prep + tagging both forks, the actual build phase surfaced three real bugs that must be backported before the next release cycle:

### Bug 1 — `setuptools_scm` rejects non-PEP-440 git tags

The fork's `setup.py` calls `setuptools_scm.get_version()` directly. With git tag `v0.7.0-tq1`, `packaging.Version()` raises `InvalidVersion: 'v0.7.0-tq1'` because `-tq1` is not a valid PEP 440 segment. **Editing `pyproject.toml` to set a static version does NOT fix this** — the call path is via `setup.py`, not via the `dynamic = ["version"]` mechanism.

**Fix:** export `SETUPTOOLS_SCM_PRETEND_VERSION=0.7.0.post20260426` (or matching PEP 440-valid version) before `python -m build`. This is the canonical setuptools_scm escape hatch and is now in `_build_one_wheel.sh` on the VM.

**Backport target:** add the env var to the fork's `scripts/_build_one_wheel.sh` so future tagged releases of the form `vX.Y.Z-tqN` build without manual intervention.

### Bug 2 — vLLM `setup.py` auto-throttles to `-j=1` under RAM pressure

n2-standard-8 has 32 GB RAM. vLLM's `compute_num_jobs()` auto-detects available memory and divides by ~7 GB/job. Even with `MAX_JOBS=4` set in env, the actual cmake/ninja invocation came out as `cmake --build . -j=1 ninja -j 1` — single-threaded. Build #1 was at ~25 files/hour, projecting **80–120h total**.

**Fix (applied 2026-04-27 ~01:58 UTC):**
- Added 16 GB swap (`/swapfile`) to give the auto-detector more breathing room
- Force `MAX_JOBS=8`, `NVCC_THREADS=2`, `CMAKE_BUILD_PARALLEL_LEVEL=8`

**Backport target:** the fork's `scripts/_build_one_wheel.sh` should set MAX_JOBS based on `nproc` × 1.0 (not vLLM's RAM heuristic) and add explicit swap-creation as a prerequisite step. **Also document n2-standard-16 (64 GB) as the safer machine class** for future builds — n2-standard-8 (32 GB) is right at the edge for 4-way parallel CUDA compile.

### Bug 3 — `_build_one_wheel.sh` build-deps list incomplete

The helper used `pip install --upgrade pip build wheel setuptools "torch>=2.4" ninja` and ran `python -m build --no-isolation`. With `--no-isolation`, the venv MUST contain ALL of the fork's `[build-system].requires`. Missing: `setuptools-scm`, `packaging`, `cmake`, `jinja2`, `numpy`. Also `setuptools<81` (fork pins a max), `torch==2.10.0` (fork pins exact).

**Fix (applied 2026-04-27 on VM):** install the full declared list verbatim:
```
pip install build wheel "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" \
            "packaging>=24.2" "torch==2.10.0" cmake ninja jinja2 numpy
```

**Backport target:** same — fork's `scripts/_build_one_wheel.sh` should derive the build-deps list from `pyproject.toml`'s `[build-system].requires` rather than hand-list a partial set.

### Build-execution path moved to VM-self-driving (2026-04-27 ~01:58 UTC)

The build loop now runs **entirely on the GCP VM in detached tmux**, not orchestrated from local WSL2. This was forced by the user's WSL2 reboot schedule (Mon 6 AM EDT + 4:45 PM EDT terminations) which would otherwise kill the local SSH-driven loop mid-build.

**Monitoring path** (no local WSL2 dependency):
- `gsutil ls gs://tqcli-wheel-build/0.7.0-tq1/_status/` — progress sentinels
- `gcloud compute ssh vllm-tq-builder --zone us-central1-a --project tqcli-wheel-build --command 'tmux capture-pane -t build -p | tail -30'` — live tmux output
- VM **self-tears-down** on success (writes `done.<ts>` sentinel + runs `sudo shutdown -h now`)
- VM **stays up** on any per-wheel failure for debugging (writes `aborted.<ts>` + per-wheel `*.fail` sentinels)

### Workstream A (cibuildwheel) status

- Tag `v0.3.0-tq1` pushed; GitHub Actions matrix in progress.
- CPU + sdist + Metal cells succeeded.
- **All CUDA cells (Linux + Windows × 3 Python) failed at the `Install CUDA 12.8 toolkit` step** using `Jimver/cuda-toolkit@v0.2.16`. Likely action-version compatibility issue with CUDA 12.8.
- **Recommended fast-follow:** in 0.3.1-tq2, bump action to `Jimver/cuda-toolkit@v0.2.19+` and re-tag. CPU/Metal wheels publish from current run; CUDA users fall back to source build until 0.3.1.

### Workstream A v0.3.1-tq2 fast-follow journey (2026-04-27 evening UTC)

The "fast-follow" turned out to be five iterations deep — each fixed a real bug, but the previous bug masked the next one:

| # | Tag commit | Failure | Root cause | Fix |
|---|---|---|---|---|
| 1 | `d736355` | `Error: Version not available: 12.8.0` (Jimver) | `Jimver/cuda-toolkit@v0.2.19` (Nov 2024) predates CUDA 12.8 release (Jan 2025) | Bump to `v0.2.30` (Dec 2025) |
| 2 | `ac62013` | `apt: cuda-cusolver-12-8 not found` | NVIDIA renamed library packages from `cuda-<lib>-12-8` → `lib<lib>-12-8` in CUDA 12+ | Split `sub-packages` (toolchain) from `non-cuda-sub-packages` (libs) per Jimver issue #371; drop unused cusparse/cusolver |
| 3 | `472915b` | `cibuildwheel: Malformed environment option 'CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=80;86;89;90 MAX_JOBS=2'` | YAML `>-` folded scalar emits the value unquoted; cibuildwheel's shell-style parser splits on whitespace mid-value | Switch to `\|` literal block + double-quote `CMAKE_ARGS` |
| 4 | `ceb702d` | `CMake Error ... CUDA Toolkit not found ... Could not find nvcc executable` | **The real blocker.** cibuildwheel for Linux runs the build inside a sandboxed manylinux Docker container; Jimver-on-host CUDA was invisible to nvcc inside the sandbox. (Windows cibuildwheel runs on the host, so Windows isn't affected.) | **Path A:** drop Linux Jimver step entirely; switch `CIBW_MANYLINUX_X86_64_IMAGE` to `ghcr.io/scikit-build/manylinux_2_28_x86_64_cuda:12.8` (scikit-build org's purpose-built image with nvcc + libcublas + libcurand pre-installed at `/usr/local/cuda`) |
| 5 | `0a4d105` | (running) | Path A applied | — |

**Path A (shipped in 0.7.0) vs Path B (proper fix for 0.7.1):**

Path A depends on the scikit-build org's CUDA-enabled manylinux image staying current. If that image is deleted or lags CUDA point releases, our build breaks with no internal recovery path. **Path B** is the more resilient design: use the canonical PyPA `quay.io/pypa/manylinux_2_28_x86_64` image and install CUDA *inside* the container at build time via `CIBW_BEFORE_ALL_LINUX` (DNF — manylinux_2_28 is AlmaLinux 8, not Ubuntu). Trade-off is +5 min per cell × 3 cells (~15 min build time added).

Path B is tracked as a 0.7.1 follow-up: **[tqcli/llama-cpp-python-turboquant#3](https://github.com/tqcli/llama-cpp-python-turboquant/issues/3)**.

**Lessons captured for next launch:**
- Don't trust the Jimver host install for Linux cibuildwheel — host installs do not survive into the manylinux sandbox.
- Verify Jimver action version supports the target CUDA version (action's version map is baked-in per release; v0.2.19 was Nov 2024, before CUDA 12.8 shipped in Jan 2025).
- NVIDIA's CUDA 12+ apt namespace split (`cuda-X-Y-Z` for tools, `libX-Y-Z` for libs) breaks any sub-package list copy-pasted from CUDA 11.x examples.
- YAML `>-` folded scalars + cibuildwheel's CIBW_ENVIRONMENT shell-parser require explicit quoting around values containing spaces or semicolons.

### Workstream A v0.3.1-tq2 → v0.3.1-tq3 (iter #11 → iter #12, 2026-04-28 UTC)

After the Path A image switch landed, six more iterations chased CUDA-cell failures (auditwheel platform tag, std::filesystem on Intel Mac, macOS x86_64 dropped to 0.7.1 follow-up, mtmd-cli linker missing cudart, etc.). **Iter #11** attempted to fix the remaining CUDA failures via global linker flags but broke CMake's compiler-detection probe; **iter #12** is the first iter built around the no-scope-cut principle (`feedback_no_scope_cut.md`):

**Failure modes resolved in iter #12 (`v0.3.1-tq3`):**

| Symptom | Root cause | Iter #12 fix |
|---|---|---|
| Linux CUDA: `ld: cannot find -lcudart` at `testCCompiler.c` | Global `CMAKE_EXE_LINKER_FLAGS=-Wl,--no-as-needed,-lcudart` applied to CMake's TryCompile probe; `LIBRARY_PATH` doesn't propagate into the probe's stripped-env link | Add `-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY` to skip the probe link; embed `-L/usr/local/cuda/lib64` directly in `CMAKE_EXE_LINKER_FLAGS` for the real project link |
| Windows CUDA: `FileNotFoundError: Could not find module 'llama.dll'` in `CIBW_TEST_COMMAND` | Test venv has no CUDA toolkit; `cudart64_12.dll` not on PATH; sentinels live in `__init__.py` after `from .llama_cpp import *` so `import llama_cpp` always loads the C ext | `delvewheel repair --add-path "C:/Program Files/.../CUDA/v12.8/bin"` bundles CUDA DLLs into `llama_cpp.libs/`; new 16-line `os.add_dll_directory` stub at top of `__init__.py` registers the bundled directory before the C ext loads |

**The PyTorch +cuXXX pattern for Linux** (introduced in iter #12):

Don't bundle CUDA libs into the Linux wheel — exclude them from auditwheel and let the `nvidia-cuda-runtime-cu12` PyPI wheel provide them at install time. Then patchelf the C extension's RPATH so it finds pip-installed nvidia/* at runtime.

```yaml
# wheels.yml (Linux CUDA cell)
CIBW_REPAIR_WHEEL_COMMAND_LINUX: bash {project}/scripts/repair_linux_wheel.sh "{wheel}" "{dest_dir}" "${{ matrix.variant }}"
CIBW_BEFORE_TEST_LINUX: |
  if [ "${{ matrix.variant }}" = "cuda" ]; then
    pip install nvidia-cuda-runtime-cu12==12.8.57 nvidia-cublas-cu12==12.8.3.14
  fi
```

```bash
# scripts/repair_linux_wheel.sh (CUDA branch)
auditwheel repair --plat manylinux2014_x86_64 \
    --exclude libcudart.so.12 \
    --exclude libcublas.so.12 \
    --exclude libcublasLt.so.12 \
    -w "$DEST_DIR" "$WHEEL"
# Then patchelf RPATH on every .so so it finds pip-installed nvidia/*/lib/
patchelf --add-rpath '$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN/../../nvidia/cublas/lib' "$so"
```

**End-user install paths** baked into iter #12:

```bash
pip install llama-cpp-python-turboquant[cuda12]   # Linux: pulls nvidia-* runtime libs from PyPI
pip install llama-cpp-python-turboquant            # Windows: CUDA DLLs already bundled by delvewheel
```

The `[cuda12]` extra in `pyproject.toml` pins exactly to the 12.8.0 series:
```toml
[project.optional-dependencies]
cuda12 = [
    "nvidia-cuda-runtime-cu12==12.8.57",
    "nvidia-cublas-cu12==12.8.3.14",
]
```

**Lessons captured for next launch (add to `tq-wheel-build-audit` skill):**
- Global `CMAKE_EXE_LINKER_FLAGS=-l<lib>` applies to CMake's TryCompile probe; the probe runs in a sandboxed dir where `LIBRARY_PATH` may not propagate, so ld can't find `-l<lib>` even when the host env has it set. Always pair with `-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY` (skips probe link) and embed `-L<dir>` directly in the linker flag (don't rely on env propagation).
- `CIBW_TEST_COMMAND` that imports the package always runs `__init__.py`. If `__init__.py` loads a C extension with non-bundled runtime deps (CUDA DLLs, etc.), tests on a vanilla CIBW runner fail. Either bundle the deps via `delvewheel`/`auditwheel`, or install them in `CIBW_BEFORE_TEST_*`. **Do not** scope-cut by switching to text-read of source files — see `feedback_no_scope_cut.md`.
- delvewheel for Windows CUDA needs `--add-path "<cuda toolkit bin>"` — Jimver's installed location isn't on the default DLL search path that delvewheel scans.
- Pin `nvidia-cuda-runtime-cu12` to exact micro-version (`==12.8.57`) to prevent pip from drifting to 12.9.x (which mismatches a CUDA 12.8 build toolkit).
- Use atomic `gh api git/trees + git/commits + git/refs` for cross-machine commit application — survives WSL2 reboot mid-flight, no clone required.

**Bracket discipline** (added to runbook): cap retry iterations at 2 / ~6h GA wall time before re-evaluating strategy. The 11-iteration history of v0.3.1-tq2 was driven by surface-symptom fixes that masked the next bug; the bracket forces a strategy audit instead of a 12th surface fix.

## Section 5: Rollback

If 0.7.0 breaks installs in the wild:
1. Yank `tqcli==0.7.0` via PyPI UI "yank" button.
2. Tag `0.7.1` that restores `[llama]` / `[vllm]` extras as aliases for `-tq` variants.
3. Do NOT delete the `vllm-turboquant` GitHub Release — mid-download users need stability.

---

## Total cost estimate (on-demand, no preempt risk)

| Item | Cost |
|---|---|
| GCP `n2-standard-8` sequential × ~30 hr (6 wheels) | ~$11.64 |
| RunPod RTX 4090 (Community), 3 hr | ~$1.02 |
| RunPod RTX 5090 (Community), 1 hr | ~$0.69 |
| RunPod B200 (Community), 1 hr | ~$5.98 |
| Own ASUS Ascent GX10, sm_121 verify | $0 |
| GCP egress 6 GB | ~$0.60 |
| **Total D-day** | **~$19.93** |

Well under the $250/day ceiling. Wall time stretches to ~30 hours (sequential build, no GCP quota request needed); a parallel 3-VM build with quota would shrink wall time to ~6h at roughly the same $ cost. Repeatable on every rebuild (realistic cadence: 4–8/year initially, settling to 2–4/year; annual compute ~$40–150).

---

## Invocation

When ready, paste into a fresh Claude Code session:

```
Read /llm_models_python_code_src/tqCLI/docs/prompts/ship_turboquant_wheels.md and execute it. I have completed all Section 0 prerequisites. Start at Section 1.
```
