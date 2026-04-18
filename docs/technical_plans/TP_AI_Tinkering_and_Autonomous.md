# Technical Implementation Plan: AI Tinkering and Autonomous Modes

## Overview
This technical plan details the implementation of a structured Agentic Orchestrator and ReAct loop within `tqCLI`. It establishes the architecture for mapping local Python skills into LLM-compatible tool schemas, handling the tri-state autonomy modes (`manual`, `ai_tinkering`, `unrestricted`), and implementing the middleware required for safe human-in-the-loop task staging.

## Architecture
- **Inference Engine Update:** Extend `engine.chat()` within implementations (`llama_backend.py` and `vllm_backend.py`) to consume an optional `tools: list[dict]` parameter.
- **Middleware Layer:** A new module `tqcli/core/agent_orchestrator.py` that intercepts streamed output, searches for `<tool_call>` (autonomy) or `<staged_tool_call>` (tinkering) blocks, and coordinates execution.
- **Core Agent Skills:** Addition of vital OS-level tools (File I/O, Terminal shell, Human Prompt) to the `tqcli/skills/builtin/` directory.

---

## Phase 1: Skill Schema Translation & Tool Injection
### Objectives
Make `tqCLI`'s internal skills mathematically visible to the LLM via tool schemas.
### Implementation Steps
1. Update `tqcli/skills/loader.py:SkillLoader` to dynamically parse Python docstrings and `SKILL.md` arguments into OpenAI-compatible JSON Schema function definitions.
2. Filter skills by "Safe" (Read-Only) vs "Actionable" (Write/Execute) to establish trust boundaries.
3. Automatically inject these schemas into the system prompt context initialization inside `tqcli/ui/interactive.py`.
### New Core Skills Needed
1. `tq-file-read` (Safe)
2. `tq-file-write` (Actionable)
3. `tq-terminal-exec` (Actionable)
4. `tq-interactive-prompt` (Safe - halts execution to capture secure inputs like passwords)

---

## Phase 2: Interceptor Middleware (AI Tinkering Mode)
### Objectives
Implement the "Shared Determinism" framework to securely authorize or reject actions.
### Implementation Steps
1. Add an `--ai-tinkering` flag to `tqcli.py` arguments. Pass state to `interactive.py`.
2. Enhance `chat_turn` to scan for `<staged_tool_call>`. When encountered, halt the LLM stream.
3. Render a `rich.prompt.Confirm` block showing the Tool Name and JSON arguments.
   **Code Example:**
   ```python
   # Inside agent_orchestrator.py
   def handle_staged_tool(tool_name: str, args: dict, mode: str) -> str:
       if mode == "ai_tinkering" and not is_safe_tool(tool_name):
           console.print(f"[yellow]Agent wants to run {tool_name} with {args}[/yellow]")
           choice = Prompt.ask("Proceed?", choices=["y", "n", "edit"], default="y")
           if choice == "n":
               return "Observation: User denied execution. Request alternatives."
           elif choice == "edit":
               # logic to open interactive temp file for arg manipulation
               pass
       
       # Execute the skill via the existing skill loader subprocess logic
       stdout = execute_skill(tool_name, args)
       return f"Observation: {stdout}"
   ```

---

## Phase 3: ReAct Loop (Unrestricted / Yolo Mode)
### Objectives
Enable multi-turn autonomous chaining by feeding tool outputs directly back into the context window as invisible turns.
### Implementation Steps
1. Map the existing `--stop-trying-to-control-everything-and-just-let-go` flag to `mode="unrestricted"`.
2. Wrap `engine.chat()` inside an orchestration `while` loop (up to a `max_steps` limit, e.g., 10).
3. If a `<tool_call>` is emitted, intercept it, force execution instantly (skipping the prompt), append the output to `messages` as a `tool` role, and re-trigger `engine.chat()`.
4. Break the loop only when the LLM outputs standard reply text concluding the task.

---

## Phase 4: Testing & Rubrics
### Objectives
Verify strict execution limits and ReAct integrity across modes.
### Code Examples & Testing Rubric
* **Unit Test 1 (Tinkering Enforcement):** 
  - *Setup:* Mock `engine` emits `<staged_tool_call>` for `tq-terminal-exec`. Mode is `ai_tinkering`.
  - *Action:* Provide `n` to the CLI prompt pipe.
  - *Assertion:* Ensure `subprocess.run` inside the skill loader was **never** called, and LLM receives the denial observation.
* **Unit Test 2 (Yolo Unrestricted Loop):**
  - *Setup:* Mock `engine` emits `<tool_call>` for `tq-system-info`. Mode is `unrestricted`.
  - *Action:* Allow normal flow.
  - *Assertion:* Ensure `subprocess.run` fires immediately, the `Observation` is appended to the history, and `engine.chat` runs for turn #2 automatically without human IO.
* **Unit Test 3 (Manual Default Silence):**
  - *Setup:* App loads in default manual mode.
  - *Assertion:* Ensure the `tools` array injected into the backend is explicitly empty. The LLM operates strictly as a text assistant to avoid hallucinating tool tags into standard markdown generation.

## Phase 5: GitHub Tracking, Documentation & Deployment
### Objectives
Implement continuous integration tracking, update all associated developer and user documentation, and properly deploy the code.
### Implementation Steps
1. **Repository Tracking:** Generate an official GitHub issue detailing the "AI Tinkering & Autonomy Orchestrator" feature build-out.
2. **Version Alignment:** Determine the proper semver bump in `CHANGELOG.md` (e.g., `v0.6.0` if this follows `v0.5.0` releases).
3. **Core Documentation Updates:**
   - Modify the root `README.md` to introduce the `--ai-tinkering` setup and explicitly define the expanded power of `--stop-trying-to-control-everything-and-just-let-go`.
   - Update `docs/ARCHITECTURE.md` to incorporate the new `agent_orchestrator.py` middleware flow sitting between the CLI and the `vllm`/`llama` backends.
   - Insert full workflow demonstrations spanning the three levels of autonomy inside `docs/examples/USAGE.md`.
4. **Issue Finalization & Merge:** After unit tests are passing and the pull request is confirmed, post a detailed summary of the solution back to the GitHub issue and close it.
5. **Code Push:** Commit all source code changes, tests, and documentation modifications and push directly to the remote repository.

## Risk Assessment
- **Context Window Exhaustion:** Repeated tool errors (e.g., tracebacks) feeding back into the observation loop can quickly fill the 4k-8k input threshold of local hardware. *Mitigation:* Implement a `Observation` truncator that clips stdout to the last 1000 characters if it spans a massive stack trace.
- **Silent Damage in Yolo Mode:** A bad regex passed to `tq-terminal-exec` by the model could `rm -rf` user directories. *Mitigation:* Limit ReAct `max_steps`, and explicitly enforce that the user passing the yolo command accepts complete responsibility for disk state.