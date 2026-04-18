# Product Requirements Document: AI Tinkering and Autonomous Modes

## 1. Introduction
**Product/Feature Name:** Agentic Orchestrator for `tqCLI` (Modes: `manual` [default], `ai_tinkering`, and `unrestricted`)
**Elevator Pitch:** Transform `tqCLI` from a passive LLM inference wrapper into a scalable AI Software Engineer by introducing "Shared Determinism" (`--ai-tinkering` mode) and full unbridled ReAct autonomy (`--stop-trying-to-control-everything-and-just-let-go`). 
**Problem Solved:** Users currently have to manually type or copy/paste every CLI command, file edit, or sequence of actions suggested by the LLM. By injecting a structured agentic loop with safety guards, `tqCLI` can proactively complete complex workflows locally while respecting corporate security thresholds via human-in-the-loop confirmations.
**Type:** Core orchestration feature enhancement.

## 2. Target Audience
- **Everyday Developers & QA:** Users who want AI suggestions to be ready-to-execute with a simple `Y/n` prompt (via `ai_tinkering`), eliminating copy-paste fatigue.
- **Power Users & Automators:** Hackathon participants and researchers who want full "auto-GPT" style autonomous file reading, writing, and deploying without interruptions (via the "yolo" unrestricted flag).
- **Enterprise Security Teams:** Organizations that demand "manual" mode for CI/CD pipelines to prevent uncontrolled execution.

## 3. Scope & Constraints
- **In Scope:** 
  - Extending the core chat loop (`tqcli/ui/interactive.py`) to support `<tool_call>` and `<staged_tool_call>` tagging.
  - Dynamically marshaling available skills (e.g., `tq-system-info`, `tq-model-manager`, and newly generated skills) into the LLM's system prompt schema.
  - Implementing an interceptor for 'Shared Determinism' (`ai_tinkering`) that halts output, presents the staged tool, and captures user consensus `[Y/n/Edit]`.
  - Creating foundational agent tools: `tq-file-system` (read/write), `tq-terminal` (execute bash), and `tq-interactive-prompt` (to capture sensitive user auth mid-loop).
- **Out of Scope:** 
  - Overriding explicitly blocked OS-level directories.
  - Autonomously downloading new unverified LLM models off the internet without user permission in `ai_tinkering` mode.
- **Constraints:** The `manual` mode MUST remain the absolute baseline default, ensuring CI/CD systems do not break due to unexpected confirmation prompts.

## 4. Key Features
1. **Three-Tier Autonomy System:**
   - **Tier 1 (Manual):** Standard secure chatbot (current state).
   - **Tier 2 (AI Tinkering):** Model stages executable commands/skills. CLI intercepts and asks `Proceed? [Y/n/Edit]`.
   - **Tier 3 (Unrestricted/Yolo):** Model autonomously executes skills in a continuous Reason-Act-Observe loop until the user's root goal is accomplished.
2. **Skill Schema Injection:** LLM automatically receives JSON schema definitions of all matching `.tqcli/skills/` tools during initialization.
3. **Human-in-the-Loop Auth Tool:** The `tq-interactive-prompt` skill allows the unrestricted agent to explicitly pause its loop and securely ask the user for passwords/tokens (e.g., "Please log into GCP to proceed").

## 5. User Stories
- **As a Developer**, I want to type "Deploy this repo" in `--ai-tinkering` mode so the AI can stage the build and deploy commands, allowing me to carefully review and approve them step-by-step.
- **As a Power User**, I want to run `tqcli chat --stop-trying-to-control-everything-and-just-let-go` so the LLM can autonomously read my logs, synthesize a fix, write the Python file patches, and test it until the tests pass without stopping to ask me for permission.
- **As an Enterprise SecOps Lead**, I want the default mode of `tqCLI` to remain completely manual so background CI scripts cannot be hijacked by malicious prompt injections into staging autonomous shell commands.

## 6. Technical Requirements
- **Skill Sets Needed:**
  - *Existing:* `tq-system-info` (read system state), `tq-model-manager` (swap models if it realizes it needs a coding model).
  - *New Core Skills Required:* `tq-file-read`, `tq-file-write`, `tq-terminal-exec`, `tq-interactive-prompt`.
- **Parsing Logic:** Robust JSON/XML parsing over streamed tool blocks to intercept execution signals before they render visually.
- **LLM Compatibility:** Requires utilizing the existing thinking/reasoning blocks of Qwen 3 and Gemma 4 via `vLLM`/`llama.cpp` to validate reasoning *before* actions occur.

## 7. Success Metrics
- **ReAct Loop Stability:** In unrestricted mode, the agent successfully loops Tool Call -> Observation -> Next Tool Call at least 5 times without crashing over badly formatted JSON.
- **Interception Accuracy:** 100% of actionable staged tool calls in `ai_tinkering` mode freeze and enforce user confirmation.
- **CLI Compatibility:** 0% regression on existing `manual` headless automated integration tests.

## 8. Release & GitHub Lifecycle
To ensure transparent tracking of this feature, the implementation must follow these lifecycle steps:
- **Issue Tracking:** Generate an issue on GitHub detailing the three-tier agentic mode (manual, AI Tinkering, unrestricted).
- **Changelog Scope:** Determine if this feature targets version `0.5.1` or if it's large enough to bump to `0.6.0`, and update `CHANGELOG.md` accordingly.
- **Documentation Updates:** Detail the new flags (`--ai-tinkering` and `--stop-trying-to-control-everything-and-just-let-go`) inside the root `README.md`, expand the `docs/ARCHITECTURE.md` to map the agent orchestration loop, and add end-to-end examples in `docs/examples/USAGE.md`.
- **Issue Resolution:** Upon successful suite execution, reply to the GitHub ticket with the final pull request or commit summary and explicitly close the issue.
- **Deployment:** Commit the codebase updates along with these documentation changes and push them directly to the upstream repository.