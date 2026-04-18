# Product Requirements Document: AI Skills Builder Command

## 1. Introduction
**Product/Feature Name:** AI Skills Builder Command (`tqcli skill generate`)
**Elevator Pitch:** An on-demand, fully autonomous AI pipeline within `tqCLI` that dynamically reads a user's PRD and Technical Implementation Plan to auto-generate the exact, ready-to-use Python code and configuration (`SKILL.md`) for a custom tqCLI skill.
**Problem Solved:** Currently, users must manually write Python code and configuration files to build new skills for `tqCLI`. This feature bridges the gap between natural language planning (PRDs/Tech Plans) and functional CLI extensions by generating the codebase automatically upon user request.
**Type:** Addition to the existing `tqCLI` product.

## 2. Target Audience
- **Developers & DevOps Engineers:** Users looking to rapidly extend `tqCLI` with custom workflows without manually writing boilerplate skill code.
- **Project Managers / Tech Leads:** Users who draft PRDs and Tech Plans and want to instantly convert those specs into functional CLI tooling.

## 3. Scope & Constraints
- **In Scope:** 
  - A new CLI command (e.g., `tqcli skill generate <prd_path> <tech_plan_path>`).
  - Parsing existing Markdown PRDs and Technical Implementation Plans.
  - Utilizing an underlying LLM (via `tqCLI`'s existing LLM engine integration) to generate the Python script and `SKILL.md`.
  - Automatically placing the generated artifacts into the user's `~/.tqcli/skills/` directory.
- **Out of Scope:** 
  - Automatically generating skills in the background without explicit user invocation (security/compliance requirement).
  - Automatically importing skills from other CLIs natively without a dedicated import command.
- **Legal/Security Constraints:** Must be strictly opt-in ("on-demand"). Code generation runs locally or via the explicitly configured LLM provider to avoid unauthorized data exfiltration.

## 4. Key Features
1. **Explicit Invocation Command:** A specific terminal command (`tqcli skill generate`) that requires the user to pass paths to their PRD and Technical Plan.
2. **Context-Aware Code Generation:** The feature reads the provided markdown documents, extracts the required architecture/API contracts, and uses the LLM to write the exact Python implementation.
3. **Structured Template Output:** Ensures the generated skill adheres to the `tqCLI` standard (creating a directory with `SKILL.md` and the executable Python scripts).
4. **Safety & Review Prompt:** Prompts the user to review the generated code before it becomes actively executable by the CLI, adhering to strict security protocols.

## 5. User Stories
- **As a developer**, I want to run `tqcli skill generate docs/prd.md docs/plan.md` so that I don't have to manually write the boilerplate code for my new CLI tool.
- **As a security-conscious engineer**, I want the skill generation to only happen when I explicitly type the command, so that my proprietary PRDs aren't automatically scanned or sent to external APIs without my consent.
- **As a technical lead**, I want the generated skill to automatically appear in my `~/.tqcli/skills/` folder so that my team can immediately test it utilizing `tqcli skill run`.

## 6. Technical Requirements
- **Tech Stack:** Python 3.11+, `click` (for CLI command routing), existing `tqCLI` LLM engine wrappers.
- **Inputs:** Local file paths (Markdown).
- **Outputs:** Python (`.py`) files and Markdown (`SKILL.md`) saved to `~/.tqcli/skills/<skill_name>/`.
- **LLM Integration:** Must use the currently active LLM backend configured in `~/.tqcli/config.yaml` with a robust system prompt designed for strict Python code formatting.

## 7. Success Metrics
- **Generation Success Rate:** >90% of generated skills execute without syntax errors on the first run.
- **Time to Value:** Reduces the time to create a new custom skill from ~30 minutes to <2 minutes.
- **Adoption:** Feature is utilized by at least 30% of active `tqCLI` developers.

## 8. Release & Documentation Requirements (v0.5.0)
To properly include this feature in the upcoming `0.5.0` release, the following files must be updated:
- `README.md`: Add a bullet point under the "What's new in 0.5.0" section.
- `CHANGELOG.md`: Include the new `tqcli skill generate` command in the 0.5.0 release notes.
- `docs/examples/USAGE.md`: Provide a verified integration bash example showcasing the command in action flow.