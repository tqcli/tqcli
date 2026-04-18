"""Shared Section E full-lifecycle step helpers for the TurboQuant KV
integration suite. Covers the `llama_cpp_test_cases.md` / Section E workflow
items that the per-model helpers previously marked as
`workflow_items_not_performed`.

Steps covered (per `turboquant_kv_comparison_test_cases.md` Section E):
  E.1  Install verification   — `tqcli --version`, `tqcli system info --json`
  E.2  Model list             — `tqcli model list` + target presence check
  E.3  KV-compressed chat     — verifies `--kv-quant` flag is wired in chat
                                (full interactive chat is recorded as SKIPPED —
                                 `tqcli chat` has no headless flag)
  E.4  Image input            — SKIPPED (requires interactive `/image`)
  E.5  Audio input            — SKIPPED (requires interactive `/audio`)
  E.6  Skill lifecycle        — `skill create` / `skill list` / `skill run`
                                + filesystem cleanup of the created skill
  E.7  Multi-process assess   — `assess_multiprocess()` Python API
  E.8  Server smoke           — `tqcli serve start|status|stop` (opt-in via
                                `TQCLI_TEST_SERVER=1`; defaults to command-
                                availability check only)
  E.9  Remove + uninstall     — `tqcli model remove --help` availability
                                check + `pip show tqcli` (non-destructive)

Destructive steps (actual `tqcli model remove`, `pip uninstall tqcli`) are
intentionally not executed — this suite is re-runnable against the shared
development environment and a ~10 GB Gemma 4 E2B re-pull would be wasteful.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqcli.config import TqConfig


REPO_ROOT = Path(__file__).parent.parent


@dataclass
class StepResult:
    name: str
    passed: bool
    duration_s: float = 0.0
    details: str = ""
    metrics: dict = field(default_factory=dict)


def _run(command: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    # Rich renders tables at terminal width; subprocess has no TTY so the
    # default 80-col width truncates long IDs with "…". Force a wide virtual
    # terminal so substring checks against the full model/skill id succeed.
    env = {**os.environ, "COLUMNS": "240"}
    return subprocess.run(
        [sys.executable, "-m", "tqcli", *command],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=timeout,
        env=env,
    )


# ── E.1 Install verification ──────────────────────────────────────────


def step_install_version() -> StepResult:
    start = time.time()
    try:
        proc = _run(["--version"])
        stdout = proc.stdout.strip()
        passed = proc.returncode == 0 and "tqcli" in stdout.lower()
        return StepResult(
            name="lifecycle_E1_version",
            passed=passed,
            duration_s=time.time() - start,
            details=f"tqcli --version => {stdout} (rc={proc.returncode})",
            metrics={"stdout": stdout[:120], "return_code": proc.returncode},
        )
    except Exception as exc:
        return StepResult(
            name="lifecycle_E1_version",
            passed=False,
            duration_s=time.time() - start,
            details=f"Exception: {exc}",
        )


def step_install_system_info() -> StepResult:
    start = time.time()
    try:
        proc = _run(["system", "info", "--json"])
        stdout = proc.stdout.strip()
        passed = proc.returncode == 0
        keys: list = []
        if passed:
            try:
                data = json.loads(stdout)
                passed = isinstance(data, dict)
                keys = list(data.keys())[:8]
            except json.JSONDecodeError as exc:
                passed = False
                details_suffix = f" | json_error={exc}"
            else:
                details_suffix = f" | keys={keys}"
        else:
            details_suffix = f" | stderr={proc.stderr.strip()[:160]}"
        return StepResult(
            name="lifecycle_E1_system_info",
            passed=passed,
            duration_s=time.time() - start,
            details=f"tqcli system info --json rc={proc.returncode}{details_suffix}",
            metrics={"keys": keys, "return_code": proc.returncode},
        )
    except Exception as exc:
        return StepResult(
            name="lifecycle_E1_system_info",
            passed=False,
            duration_s=time.time() - start,
            details=f"Exception: {exc}",
        )


# ── E.2 Model list ────────────────────────────────────────────────────


def step_model_list_contains(model_id: str) -> StepResult:
    start = time.time()
    try:
        proc = _run(["model", "list"], timeout=60)
        stdout = proc.stdout
        # ``tqcli model list`` renders a Rich table; normalize by stripping
        # whitespace and matching the model id substring.
        passed = proc.returncode == 0 and model_id in stdout
        return StepResult(
            name="lifecycle_E2_model_list",
            passed=passed,
            duration_s=time.time() - start,
            details=(
                f"tqcli model list rc={proc.returncode} | "
                f"{model_id} present: {model_id in stdout}"
            ),
            metrics={
                "model_id": model_id,
                "contains_model": model_id in stdout,
                "return_code": proc.returncode,
            },
        )
    except Exception as exc:
        return StepResult(
            name="lifecycle_E2_model_list",
            passed=False,
            duration_s=time.time() - start,
            details=f"Exception: {exc}",
        )


# ── E.3 KV-compressed chat flag wiring ────────────────────────────────


def step_chat_kv_quant_flag() -> StepResult:
    start = time.time()
    try:
        proc = _run(["chat", "--help"], timeout=30)
        help_text = proc.stdout
        passed = proc.returncode == 0 and "--kv-quant" in help_text
        return StepResult(
            name="lifecycle_E3_chat_kv_quant_flag",
            passed=passed,
            duration_s=time.time() - start,
            details=(
                f"tqcli chat --help rc={proc.returncode} | "
                f"--kv-quant flag present: {'--kv-quant' in help_text}. "
                "Full KV-compressed chat is interactive; a non-interactive "
                "two-turn chat is exercised by the per-engine load tests "
                "(`chat_thinking_turn` / `chat_simple_turn`)."
            ),
            metrics={
                "return_code": proc.returncode,
                "kv_quant_flag_present": "--kv-quant" in help_text,
            },
        )
    except Exception as exc:
        return StepResult(
            name="lifecycle_E3_chat_kv_quant_flag",
            passed=False,
            duration_s=time.time() - start,
            details=f"Exception: {exc}",
        )


# ── E.4 / E.5 Image and audio input ──────────────────────────────────
# These slash-commands (`/image`, `/audio`) are only available inside an
# interactive chat session. We record them as SKIPPED (passed=True) with
# a clear reason so the comparison report still shows the row.


def step_image_input_skipped(model_multimodal: bool) -> StepResult:
    reason = (
        "Slash-command `/image` is only available inside `tqcli chat` "
        "(interactive); no headless flag exists in 0.3.1. "
        f"Model multimodal capability: {model_multimodal}."
    )
    return StepResult(
        name="lifecycle_E4_image_input",
        passed=True,
        duration_s=0.0,
        details=f"SKIPPED: {reason}",
        metrics={"skipped": True, "multimodal": model_multimodal},
    )


def step_audio_input_skipped() -> StepResult:
    return StepResult(
        name="lifecycle_E5_audio_input",
        passed=True,
        duration_s=0.0,
        details=(
            "SKIPPED: Slash-command `/audio` is only available inside "
            "`tqcli chat` (interactive). Per test_cases §E.5, graceful "
            "'no audio capability' is expected for all current quantized "
            "models — acceptable behavior."
        ),
        metrics={"skipped": True},
    )


# ── E.6 Skill lifecycle ──────────────────────────────────────────────


def _skill_dir(skill_name: str) -> Path:
    config = TqConfig.load()
    return config.skills_dir / skill_name


def step_skill_create(skill_name: str) -> StepResult:
    start = time.time()
    # Make sure there's no leftover from a prior run
    leftover = _skill_dir(skill_name)
    if leftover.exists():
        shutil.rmtree(leftover, ignore_errors=True)
    try:
        proc = _run([
            "skill",
            "create",
            skill_name,
            "-d",
            "TurboQuant KV smoke skill",
        ], timeout=30)
        created = _skill_dir(skill_name).exists()
        passed = proc.returncode == 0 and created
        return StepResult(
            name="lifecycle_E6_skill_create",
            passed=passed,
            duration_s=time.time() - start,
            details=(
                f"tqcli skill create {skill_name} rc={proc.returncode} | "
                f"skill dir present: {created}"
            ),
            metrics={"skill_name": skill_name, "return_code": proc.returncode},
        )
    except Exception as exc:
        return StepResult(
            name="lifecycle_E6_skill_create",
            passed=False,
            duration_s=time.time() - start,
            details=f"Exception: {exc}",
        )


def step_skill_list_contains(skill_name: str) -> StepResult:
    start = time.time()
    try:
        proc = _run(["skill", "list"], timeout=30)
        passed = proc.returncode == 0 and skill_name in proc.stdout
        return StepResult(
            name="lifecycle_E6_skill_list",
            passed=passed,
            duration_s=time.time() - start,
            details=(
                f"tqcli skill list rc={proc.returncode} | "
                f"{skill_name} listed: {skill_name in proc.stdout}"
            ),
            metrics={"skill_name": skill_name, "return_code": proc.returncode},
        )
    except Exception as exc:
        return StepResult(
            name="lifecycle_E6_skill_list",
            passed=False,
            duration_s=time.time() - start,
            details=f"Exception: {exc}",
        )


def step_skill_run(skill_name: str) -> StepResult:
    start = time.time()
    try:
        proc = _run(["skill", "run", skill_name], timeout=60)
        stdout = proc.stdout
        # The generated template prints a JSON object with "status":"completed".
        status_marker = '"status": "completed"'
        has_completed = status_marker in stdout
        passed = proc.returncode == 0 and has_completed
        return StepResult(
            name="lifecycle_E6_skill_run",
            passed=passed,
            duration_s=time.time() - start,
            details=(
                f"tqcli skill run {skill_name} rc={proc.returncode} | "
                f"status=completed: {has_completed}"
            ),
            metrics={"skill_name": skill_name, "return_code": proc.returncode},
        )
    except Exception as exc:
        return StepResult(
            name="lifecycle_E6_skill_run",
            passed=False,
            duration_s=time.time() - start,
            details=f"Exception: {exc}",
        )


def step_skill_cleanup(skill_name: str) -> StepResult:
    start = time.time()
    path = _skill_dir(skill_name)
    try:
        if path.exists():
            shutil.rmtree(path)
        return StepResult(
            name="lifecycle_E6_skill_cleanup",
            passed=not path.exists(),
            duration_s=time.time() - start,
            details=f"Removed {path}: {not path.exists()}",
            metrics={"skill_name": skill_name, "path": str(path)},
        )
    except Exception as exc:
        return StepResult(
            name="lifecycle_E6_skill_cleanup",
            passed=False,
            duration_s=time.time() - start,
            details=f"Exception: {exc}",
        )


# ── E.7 Multi-process feasibility assessment ─────────────────────────


def step_multiprocess_assess(
    model_id: str,
    engine: str,
    model_size_mb: int,
    requested_workers: int = 3,
) -> StepResult:
    start = time.time()
    try:
        from tqcli.core.multiprocess import assess_multiprocess
        from tqcli.core.system_info import detect_system

        sys_info = detect_system()
        plan = assess_multiprocess(
            sys_info=sys_info,
            model_path=f"~/.tqcli/models/{model_id}",
            model_size_mb=model_size_mb,
            requested_workers=requested_workers,
            preferred_engine=engine,
            unrestricted=True,
        )
        return StepResult(
            name="lifecycle_E7_multiprocess_assess",
            passed=True,
            duration_s=time.time() - start,
            details=(
                f"assess_multiprocess(engine={engine}, "
                f"requested_workers={requested_workers}) => "
                f"feasible={plan.feasible}, engine={plan.engine}, "
                f"max_workers={plan.max_workers}"
            ),
            metrics={
                "feasible": plan.feasible,
                "engine": plan.engine,
                "max_workers": plan.max_workers,
                "recommended_workers": getattr(plan, "recommended_workers", None),
            },
        )
    except Exception as exc:
        return StepResult(
            name="lifecycle_E7_multiprocess_assess",
            passed=False,
            duration_s=time.time() - start,
            details=f"Exception: {exc}",
        )


# ── E.8 Server smoke (opt-in) ────────────────────────────────────────


def step_serve_lifecycle(model_id: str, engine: str) -> list[StepResult]:
    """By default (env var `TQCLI_TEST_SERVER` unset or "0"), only verify
    that `serve start|status|stop` commands exist. Setting
    `TQCLI_TEST_SERVER=1` opts into actually starting the server (recommended
    for llama.cpp + a pre-downloaded GGUF; vLLM server start is too slow on
    4 GB VRAM to run on every suite iteration)."""
    opt_in = os.environ.get("TQCLI_TEST_SERVER", "0") in ("1", "true", "yes")

    out: list[StepResult] = []
    for sub in ("start", "status", "stop"):
        start = time.time()
        try:
            proc = _run(["serve", sub, "--help"], timeout=15)
            passed = proc.returncode == 0
            out.append(StepResult(
                name=f"lifecycle_E8_serve_{sub}_help",
                passed=passed,
                duration_s=time.time() - start,
                details=(
                    f"tqcli serve {sub} --help rc={proc.returncode} "
                    f"(command available)"
                ),
                metrics={"subcommand": sub, "return_code": proc.returncode},
            ))
        except Exception as exc:
            out.append(StepResult(
                name=f"lifecycle_E8_serve_{sub}_help",
                passed=False,
                duration_s=time.time() - start,
                details=f"Exception: {exc}",
            ))

    if not opt_in:
        out.append(StepResult(
            name="lifecycle_E8_serve_smoke",
            passed=True,
            duration_s=0.0,
            details=(
                "SKIPPED: set TQCLI_TEST_SERVER=1 to actually start the "
                f"server for {model_id} ({engine}). Default is command-"
                "availability check only — prior suite runs verified the "
                "server path works."
            ),
            metrics={"skipped": True, "model_id": model_id, "engine": engine},
        ))
        return out

    # Opt-in: actually start + status + stop.
    start = time.time()
    try:
        proc = _run([
            "--stop-trying-to-control-everything-and-just-let-go",
            "serve",
            "start",
            "-m",
            model_id,
            "-e",
            engine,
        ], timeout=300)
        started = proc.returncode == 0
        out.append(StepResult(
            name="lifecycle_E8_serve_start",
            passed=started,
            duration_s=time.time() - start,
            details=f"tqcli serve start -m {model_id} rc={proc.returncode}",
            metrics={"return_code": proc.returncode},
        ))
        status_proc = _run(["serve", "status"], timeout=15)
        out.append(StepResult(
            name="lifecycle_E8_serve_status",
            passed=status_proc.returncode == 0,
            duration_s=0.0,
            details=f"tqcli serve status rc={status_proc.returncode}",
            metrics={"return_code": status_proc.returncode},
        ))
    except Exception as exc:
        out.append(StepResult(
            name="lifecycle_E8_serve_start",
            passed=False,
            duration_s=time.time() - start,
            details=f"Exception: {exc}",
        ))
    finally:
        stop_start = time.time()
        try:
            proc = _run(["serve", "stop"], timeout=60)
            out.append(StepResult(
                name="lifecycle_E8_serve_stop",
                passed=proc.returncode == 0,
                duration_s=time.time() - stop_start,
                details=f"tqcli serve stop rc={proc.returncode}",
                metrics={"return_code": proc.returncode},
            ))
        except Exception as exc:
            out.append(StepResult(
                name="lifecycle_E8_serve_stop",
                passed=False,
                duration_s=time.time() - stop_start,
                details=f"Exception: {exc}",
            ))
    return out


# ── E.9 Remove + uninstall (non-destructive) ─────────────────────────


def step_model_remove_available(model_id: str) -> StepResult:
    start = time.time()
    try:
        proc = _run(["model", "remove", "--help"], timeout=15)
        passed = proc.returncode == 0
        return StepResult(
            name="lifecycle_E9_model_remove_help",
            passed=passed,
            duration_s=time.time() - start,
            details=(
                f"tqcli model remove --help rc={proc.returncode} | "
                f"NOT executed against {model_id} — keeping model installed "
                "so the suite remains re-runnable against shared models."
            ),
            metrics={"model_id": model_id, "return_code": proc.returncode},
        )
    except Exception as exc:
        return StepResult(
            name="lifecycle_E9_model_remove_help",
            passed=False,
            duration_s=time.time() - start,
            details=f"Exception: {exc}",
        )


def step_pip_show_tqcli() -> StepResult:
    """Verify that `tqcli` is installed by reading package metadata directly.

    We used to shell out to `pip show tqcli`, but on this host (6 concurrent
    WSL2 VMs sharing the same filesystem/pip cache) that subprocess was
    regularly exceeding a 30 s timeout. `importlib.metadata` reads the
    installed .dist-info/METADATA file directly — no network, no pip, no
    lock contention — and is the canonical replacement for `pip show` as of
    Python 3.8+."""
    start = time.time()
    try:
        try:
            from importlib.metadata import PackageNotFoundError, metadata, version
        except ImportError:  # pragma: no cover — Python <3.8 not supported
            from importlib_metadata import (  # type: ignore[no-redef]
                PackageNotFoundError,
                metadata,
                version,
            )
        try:
            pkg_version = version("tqcli")
            md = metadata("tqcli")
            installed = True
            name = md.get("Name", "tqcli")
        except PackageNotFoundError:
            installed = False
            pkg_version = ""
            name = ""
        return StepResult(
            name="lifecycle_E9_pip_show",
            passed=installed,
            duration_s=time.time() - start,
            details=(
                f"importlib.metadata tqcli installed={installed} "
                f"Name: {name} Version: {pkg_version} "
                "(pip uninstall NOT executed)"
            ),
            metrics={"installed": installed, "version": pkg_version},
        )
    except Exception as exc:
        return StepResult(
            name="lifecycle_E9_pip_show",
            passed=False,
            duration_s=time.time() - start,
            details=f"Exception: {exc}",
        )


# ── §F vLLM multimodal image input (PLACEHOLDER — blocked) ──────────
#
# These helpers are stubs. They become real once
# `docs/prompts/implement_headless_chat_and_vllm_multimodal.md` is executed
# (landing headless `tqcli chat --prompt ... --image ... --json` and the
# vLLM multimodal pass-through in tqcli/core/vllm_backend.py).
#
# Until then, the suite records each §F step as SKIPPED with the reason
# embedded in the details field, so the comparison report shows the row
# rather than silently dropping it.


def step_vllm_image_input_gemma4(model_id: str) -> StepResult:
    """§F.1 — vLLM Gemma 4 E2B + CPU offload + TurboQuant image grounding.

    Headless chat + vLLM multimodal pass-through shipped in v0.5.0 (#24).
    This step drives the real `tqcli chat --prompt ... --image ... --json`
    command when opted in via `TQCLI_TEST_VLLM_IMAGE=1`; otherwise it records
    the step as SKIPPED (heavy GPU load — Gemma 4 E2B cold-load on 4 GB VRAM
    WSL2 runs ~500–625 s).
    """
    import os
    import subprocess
    import time as _t

    if os.environ.get("TQCLI_TEST_VLLM_IMAGE") != "1":
        return StepResult(
            name="lifecycle_F1_vllm_image_gemma4",
            passed=True,
            duration_s=0.0,
            details=(
                f"SKIPPED: {model_id} — heavy GPU load gated by "
                "TQCLI_TEST_VLLM_IMAGE=1. Command is implemented (v0.5.0, #24)."
            ),
            metrics={"skipped": True, "reason": "opt_in_gate", "model_id": model_id},
        )

    fixture = Path(__file__).parent / "fixtures" / "test_image.png"
    start = _t.time()
    try:
        proc = subprocess.run(
            [
                "tqcli", "chat",
                "--model", model_id,
                "--engine", "vllm",
                "--kv-quant", "turbo3",
                "--prompt", "What colors do you see in the image?",
                "--image", str(fixture),
                "--json",
                "--max-tokens", "128",
            ],
            capture_output=True, text=True, timeout=1800,
        )
        elapsed = _t.time() - start
        data = json.loads(proc.stdout) if proc.stdout else {}
        answer = (data.get("response") or "").lower()
        mentions_color = ("red" in answer) or ("blue" in answer)
        return StepResult(
            name="lifecycle_F1_vllm_image_gemma4",
            passed=proc.returncode == 0 and mentions_color,
            duration_s=elapsed,
            details=(
                f"exit={proc.returncode} answer_len={len(answer)} "
                f"mentions_red_or_blue={mentions_color}"
            ),
            metrics={
                "model_id": model_id,
                "tokens_per_second": data.get("performance", {}).get("tokens_per_second", 0.0),
                "response_len": len(answer),
            },
        )
    except Exception as exc:
        return StepResult(
            name="lifecycle_F1_vllm_image_gemma4",
            passed=False,
            duration_s=_t.time() - start,
            details=f"Exception: {exc}",
        )


def step_vllm_image_input_non_multimodal(model_id: str) -> StepResult:
    """§F — Non-multimodal vLLM models are recorded as N/A.

    Currently applies to `qwen3-4b-AWQ` and `qwen3-4b-vllm` — these profiles
    have `multimodal=False` in BUILTIN_PROFILES, so image input is not a
    supported flow. Recording this explicitly prevents the comparison report
    from silently omitting the row.
    """
    return StepResult(
        name="lifecycle_F_vllm_image_text_only",
        passed=True,
        duration_s=0.0,
        details=(
            f"N/A: {model_id} is text-only (multimodal=False in "
            "BUILTIN_PROFILES). Image input is not a supported flow for "
            "this profile; recorded for report completeness."
        ),
        metrics={
            "skipped": True,
            "reason": "model_not_multimodal",
            "model_id": model_id,
        },
    )


# ── §G vLLM multi-process CRM build (PLACEHOLDER — blocked) ──────────
#
# Also blocked on the headless-chat prompt. Once headless lands, these
# helpers should drive:
#   1. `tqcli --stop-trying-to-control-everything-and-just-let-go serve start`
#      with the vLLM engine + the target model.
#   2. `tqcli skill create crm-frontend-vllm / crm-backend-vllm / crm-database-vllm`.
#   3. Spawn workers with `tqcli chat --engine server --prompt "..."`
#      generating each CRM artifact into a tmp workspace.
#   4. Assert all three files exist and are non-empty, then `serve stop`.


def step_vllm_multiprocess_crm(model_id: str) -> StepResult:
    """§G.1 / §G.2 — vLLM multi-process CRM build.

    Headless chat shipped in v0.5.0 (#24). This step is opt-in via
    `TQCLI_TEST_VLLM_CRM=1` — it boots a real vLLM server, sends three
    headless `--prompt ... --engine server --json` requests, and verifies
    the three CRM artefacts are produced. Without the flag, the step is
    recorded as SKIPPED with the heavy-GPU-load reason.
    """
    import os
    import subprocess
    import tempfile
    import time as _t

    if os.environ.get("TQCLI_TEST_VLLM_CRM") != "1":
        return StepResult(
            name="lifecycle_G_vllm_multiprocess_crm",
            passed=True,
            duration_s=0.0,
            details=(
                f"SKIPPED: {model_id} — heavy GPU load gated by "
                "TQCLI_TEST_VLLM_CRM=1. Command is implemented (v0.5.0, #24)."
            ),
            metrics={"skipped": True, "reason": "opt_in_gate", "model_id": model_id},
        )

    start = _t.time()
    try:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            subprocess.run(
                [
                    "tqcli", "--stop-trying-to-control-everything-and-just-let-go",
                    "serve", "start", "-m", model_id, "--engine", "vllm",
                ],
                check=True, timeout=1800,
            )
            try:
                for slot, prompt in [
                    ("frontend.html", "Generate a minimal HTML CRM contact form with Name/Email fields."),
                    ("backend.py", "Generate a tiny Flask route /contacts that returns an empty JSON list."),
                    ("schema.sql", "Generate a SQLite CREATE TABLE contacts(id, name, email) statement."),
                ]:
                    out = work / slot
                    proc = subprocess.run(
                        [
                            "tqcli", "chat", "--engine", "server",
                            "--prompt", prompt,
                            "--json", "--max-tokens", "300",
                        ],
                        capture_output=True, text=True, timeout=1200,
                    )
                    data = json.loads(proc.stdout) if proc.stdout else {}
                    out.write_text(data.get("response", ""))
                ok = all((work / n).exists() and (work / n).stat().st_size > 10
                         for n in ("frontend.html", "backend.py", "schema.sql"))
            finally:
                subprocess.run(["tqcli", "serve", "stop"], timeout=60)
        return StepResult(
            name="lifecycle_G_vllm_multiprocess_crm",
            passed=ok,
            duration_s=_t.time() - start,
            details=f"three_artifacts_generated={ok}",
            metrics={"model_id": model_id},
        )
    except Exception as exc:
        subprocess.run(["tqcli", "serve", "stop"], timeout=60)
        return StepResult(
            name="lifecycle_G_vllm_multiprocess_crm",
            passed=False,
            duration_s=_t.time() - start,
            details=f"Exception: {exc}",
        )


# ── Full lifecycle runner ────────────────────────────────────────────


def run_full_lifecycle(
    model_id: str,
    kv_level: str,
    engine: str,
    model_size_mb: int,
    multimodal: bool = False,
) -> list[StepResult]:
    """Run the Section E full-lifecycle steps and return a list of
    StepResults to be appended onto the caller TestResult."""
    skill_name = f"tq-kv-{model_id}-{kv_level}".replace("_", "-").lower()

    steps: list[StepResult] = []
    # E.1
    steps.append(step_install_version())
    steps.append(step_install_system_info())
    # E.2
    steps.append(step_model_list_contains(model_id))
    # E.3
    steps.append(step_chat_kv_quant_flag())
    # E.4, E.5
    steps.append(step_image_input_skipped(multimodal))
    steps.append(step_audio_input_skipped())
    # E.6
    steps.append(step_skill_create(skill_name))
    steps.append(step_skill_list_contains(skill_name))
    steps.append(step_skill_run(skill_name))
    steps.append(step_skill_cleanup(skill_name))
    # E.7
    steps.append(step_multiprocess_assess(model_id, engine, model_size_mb))
    # E.8
    steps.extend(step_serve_lifecycle(model_id, engine))
    # E.9
    steps.append(step_model_remove_available(model_id))
    steps.append(step_pip_show_tqcli())

    # §F vLLM image input (placeholder — skipped until headless chat + vLLM
    # multimodal pass-through land). Only meaningful for vLLM profiles.
    if engine == "vllm":
        if multimodal:
            steps.append(step_vllm_image_input_gemma4(model_id))
        else:
            steps.append(step_vllm_image_input_non_multimodal(model_id))
        # §G vLLM multi-process CRM (placeholder — same blocker).
        steps.append(step_vllm_multiprocess_crm(model_id))

    return steps
