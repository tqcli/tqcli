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

### 0.C — Trusted Publishing for `llama-cpp-python-turboquant`

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

### 0.D — Cloud accounts (pay-as-you-go, no monthly commit)

1. **GCP** — already have access. Enable Compute Engine API in a project `tqcli-wheel-build`. Set budget alert at $50.
2. **Vast.ai** — create account at `vast.ai`, add $30 credit. For RTX 4090 + RTX 5090 verification.
3. **Lambda Labs** — create account at `lambda.ai`, add payment. For B200 verification.
4. **ASUS Ascent GX10** — already owned (acquired 2026-04-25). Provides on-hand sm_121 (DGX Spark / GB10, Blackwell) verification. No cloud account needed for this cell.

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

### Workstream B — `vllm-turboquant` wheel on rented GCP

Hand this prompt to Worker B:

```
Branch: off `tqcli/vllm-turboquant` main.

Goal: produce one vllm-turboquant wheel per Python 3.10/3.11/3.12 on GCP on-demand; attach to GitHub Release.

Arch list: TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0 12.1+PTX"
CUDA toolkit: 13.0+ (12.8 cannot compile sm_121).

Steps (read TP Phase 1 B1 + Phase 3):

1. In fork pyproject.toml, rename distribution to `vllm-turboquant`. Preserve `vllm` import name.
2. Add to vllm/__init__.py:
     TURBOQUANT_ENABLED = True
     TURBOQUANT_KV_DTYPES = ("turboquant25", "turboquant35")
3. Update fork README.md per TP B1.4. Confirm LICENSE stays Apache 2.0 + NOTICE intact.
4. Identify golden commit AFTER Issue #22 four-patch page-size fix (see patches/vllm-turboquant/issue_22_page_size_fix.md). Verify with:
     - Gemma 4 E2B + BNB_INT4 + CPU offload + turboquant35 (Section C.2 of comparison report)
     - Qwen 3 4B + calibrated turboquant_kv.json (0.6.1 path)
   Both must be green.
5. Tag `v0.7.0-tq1` on the fork.
6. Author scripts/build_wheel_gcp.sh that:
     - Provisions a GCP n2-standard-16 on-demand VM in us-central1
     - Installs CUDA 13.0 toolkit
     - Sets the arch list above, MAX_JOBS=4, NVCC_THREADS=4, VLLM_TARGET_DEVICE=cuda
     - Runs python -m build --wheel
     - Pushes the wheel to a GCS bucket `gs://tqcli-wheel-build/0.7.0-tq1/`
     - Tears down the VM
   Run it three times in parallel for Python 3.10, 3.11, 3.12 (spawn 3 VMs; ~6 hr each on-demand; cost ~$4.66/VM = ~$14 total).
7. Measure wheel sizes. If any exceeds 2 GB: STOP. Escalate to user — two paths:
     (a) LFS-host the wheel (GitHub LFS has 2 GB per-file limit; may also fail)
     (b) Split into vllm-turboquant + vllm-turboquant-blackwell extras (Ampere/Ada/Hopper vs sm_100/120/121)
   Do NOT make this call unilaterally.
8. If sizes are under 2 GB: gh release create v0.7.0-tq1 --repo tqcli/vllm-turboquant with the three .whl files + SHA256SUMS. Release body pastes Section C.2 numbers verbatim.
9. Document the script path + invocation in fork docs/RELEASING.md.
10. Confirm: pip install vllm-turboquant --find-links https://github.com/tqcli/vllm-turboquant/releases/expanded_assets/v0.7.0-tq1 resolves on a clean CUDA 13.0 Ubuntu VM.

Verification via Vast.ai RTX 4090 (cost ~$0.87 for 3 hr):
- pip install succeeds; python -c "import vllm; print(vllm.TURBOQUANT_ENABLED)" prints True.
- tqcli chat --model gemma-4-e2b-it-vllm --engine vllm --kv-quant turbo3 --prompt "Paris?" --json emits Section C.2 metadata.

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
| V3 | Vast.ai on-demand 3 hr | RTX 4090 (sm_8.9, Ada) | vllm + turboquant35 end-to-end | ~$0.87 |
| V4 | Vast.ai on-demand 1 hr | RTX 5090 (sm_12.0, Blackwell consumer) | vllm + turboquant35 — proves Blackwell consumer path | ~$0.51 |
| V5 | Lambda Labs 1 hr | B200 (sm_10.0, Blackwell DC) | vllm + turboquant35 — LinkedIn-worthy | ~$3.49 |
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

**Total verification cost**: ~$5.35.

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

## Section 5: Rollback

If 0.7.0 breaks installs in the wild:
1. Yank `tqcli==0.7.0` via PyPI UI "yank" button.
2. Tag `0.7.1` that restores `[llama]` / `[vllm]` extras as aliases for `-tq` variants.
3. Do NOT delete the `vllm-turboquant` GitHub Release — mid-download users need stability.

---

## Total cost estimate (on-demand, no preempt risk)

| Item | Cost |
|---|---|
| GCP `n2-standard-16` on-demand × 3 parallel VMs × 6 hr | ~$13.97 |
| Vast.ai RTX 4090, 3 hr | ~$0.87 |
| Vast.ai RTX 5090, 1 hr | ~$0.51 |
| Lambda Labs B200, 1 hr | ~$3.49 |
| Own ASUS Ascent GX10, sm_121 verify | $0 |
| GCP egress 6 GB | ~$0.60 |
| **Total D-day** | **~$19.44** |

Well under the $250/day ceiling. Repeatable at this cost on every rebuild (realistic cadence: 4–8/year initially, settling to 2–4/year; annual compute ~$40–160).

---

## Invocation

When ready, paste into a fresh Claude Code session:

```
Read /llm_models_python_code_src/tqCLI/docs/prompts/ship_turboquant_wheels.md and execute it. I have completed all Section 0 prerequisites. Start at Section 1.
```
