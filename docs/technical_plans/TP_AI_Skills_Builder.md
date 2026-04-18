# Technical Implementation Plan: AI Skills Builder Command

## Overview
This document outlines the technical phases required to implement the "AI Skills Builder" command within `tqCLI`. This feature allows users to directly convert their PRD and Technical Implementation Plans into functional `tqCLI` skills positioned in the `~/.tqcli/skills/` directory, adhering strictly to an opt-in, non-automated workflow for legal and security compliance.

## Architecture
- **CLI Layer:** A new command group/action in `tqcli/cli.py` (`tqcli skill generate`).
- **Processing Layer:** A new module `tqcli/core/skill_generator.py` responsible for reading the markdown files and orchestrating the LLM prompt.
- **LLM Integration:** Reuse `tqCLI`'s existing LLM abstraction (e.g., `vllm`, `llama.cpp`, or external APIs depending on the user's `config.yaml`).
- **Output Layer:** Disk writing operations targeting `~/.tqcli/skills/`.

## Phase 1: CLI Interface & File Parsing
### Objectives
Establish the user-facing command and ensure it securely reads local markdown files.
### Implementation Steps
1. Add `@click.command("generate")` under the `skill` group in `tqcli/cli.py`.
2. Define arguments: `--prd` (path to PRD) and `--plan` (path to tech plan). Add `--name` for the target skill directory.
3. Implement file validation (ensure files exist, are readable, and size is within token context limits).
### Files
- `tqcli/cli.py` (Modified)
- `tqcli/core/skill_generator.py` (Created)
### Dependencies
- None (Core Python `pathlib` and `click`).

## Phase 2: LLM Orchestration & Prompt Engineering
### Objectives
Translate the ingested markdown text into valid Python code and `SKILL.md` configurations.
### Implementation Steps
1. Create a structured system prompt in `tqcli/core/skill_generator.py` instructing the LLM to output file contents utilizing specific Markdown code blocks (e.g., `@@@filename.py@@@`).
2. Pass the aggregated PRD and Tech Plan text to the active `tqCLI` model engine.
3. Implement a parser to extract the generated files from the LLM's raw text response.
### Files
- `tqcli/core/skill_generator.py` (Modified)
- `tqcli/prompts/skill_generation_prompt.jinja` (Created)
### Dependencies
- Phase 1 (CLI wiring).
- Existing `tqCLI` LLM backend modules.

## Phase 3: Disk I/O & User Review Workflow
### Objectives
Safely write the parsed code to the user's skill directory with a mandatory interactive review step.
### Implementation Steps
1. In `skill_generator.py`, create a temporary generation cache.
2. Prompt the user in the CLI: "Generated <name>. Would you like to review the code before saving? [Y/n]"
3. If approved, copy the files into `Path.home() / ".tqcli" / "skills" / <name> /`.
4. Set execution permissions (`chmod +x`) on the generated Python scripts.
### Files
- `tqcli/core/skill_generator.py` (Modified)
- `tqcli/utils/file_system.py` (Modified/Created for safe writes)
### Dependencies
- Phase 2 (LLM parser).

## Phase 4: Testing & Documentation
### Objectives
Ensure reliability and document the feature for end-users.
### Implementation Steps
1. Write unit tests for the LLM output parser to ensure it splits multi-file outputs correctly.
2. Write integration tests mimicking a dummy PRD and asserting the files appear in a mocked `~/.tqcli/skills/` directory.
3. Update `docs/GETTING_STARTED.md` and `docs/ARCHITECTURE.md` to feature the new `tqcli skill generate` command.
4. Finalize the `0.5.0` release bundle: Add feature details to `README.md`, update the `CHANGELOG.md`, and place a working code placeholder in `docs/examples/USAGE.md`.
### Files
- `tests/test_skill_generator.py` (Created)
- `docs/GETTING_STARTED.md` (Modified)
- `README.md` (Modified for v0.5.0)
- `CHANGELOG.md` (Modified for v0.5.0)
- `docs/examples/USAGE.md` (Modified for v0.5.0)
### Dependencies
- Phase 3 complete.

## Risk Assessment
- **Context Limit Exceeded:** Large PRDs/Plans might exceed local model context windows. *Mitigation:* Implement token counting and warn users if their files are too large, suggesting truncated versions.
- **Malicious LLM Code Execution:** Generated code could be syntactically flawed or dangerous. *Mitigation:* The explicit user review step (Phase 3) completely isolates generation from execution, shifting the final approval squarely to the human.

## Success Criteria
- The `tqcli skill generate` command correctly parses two target markdown files and outputs a functional `SKILL.md` and `script.py` in the user's local `~/.tqcli/skills/` directory.
- The command explicitly halts and asks the user for permission before writing the final active code.
- Successfully passes the test suite with a 100% integration check rate on mocked PRDs.