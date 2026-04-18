**Answer to the Developer:**

Based on a deep review of the `tqCLI` codebase (specifically `tqcli/cli.py` and `tqcli/skills/loader.py`), here is the exact architectural breakdown of how skills operate and the level of autonomy the models currently possess:

### 1. Zero Model Autonomy (By Design)
The hosted thinking models (like Qwen 3 or Gemma 4) **do not** autonomously pick or execute skills on their own. 

Unlike agentic loops (such as Claude Code's REPL where the model emits a JSON `<tool_call>` that the CLI automatically executes in the background), `tqCLI` treats models strictly as passive inference engines. The LLM does not get injected with a list of available `~/.tqcli/skills/` and it does not make the determination of when to run one. 

### 2. User-Driven Invocation (Confirmed in Codebase)
You are completely correct that the end-user dictates what skills to use. 
In `tqcli/cli.py` (around line 840), the CLI exposes `@skill_group.command("run")`. When a user types:
`tqcli skill run prd-generator`

The CLI uses the `SkillLoader` to find the Python script for that skill and executes it using `subprocess.run()`. 

**The architectural inversion:** The *model* isn't running the *skill*; the *skill* (a Python script) is running the *model*. The Python script contains the structured interview logic (like asking about target audiences) and simply makes standard chat completion calls to the LLM backend to process the text.

### 3. Why this is the correct approach currently
This lack of agentic autonomy perfectly aligns with the strict security and compliance bounds established in the PRD (like the refusal to automatically background-scan competitor `.claude/` directories). 
By forcing human-in-the-loop invocation, `tqCLI` guarantees that no LLM will unexpectedly start spinning up multi-phase technical planners, writing files to disk, or executing arbitrary code without explicit user consent. 

**Summary to relay:**
*"The hosted models have zero autonomy to pick or run skills themselves. The current architecture strictly dictates that the end-user must explicitly execute a skill via `tqcli skill run <name>`. The skill executes as a wrapper script that drives the model, rather than the model driving the skill. This guarantees our security and compliance requirements by keeping a human explicitly in the loop for tool execution."*