---
**Subject:** Re: Clarification on `tqcli skill generate` output directory (`~/.tqcli/skills/` vs `.claude/skills/`)
---

Excellent question and a great catch on the current loader path. 

The PRD is correct as written: the generated skills MUST write to the user-level hidden directory `~/.tqcli/skills/<name>/`. **That is the intentional design.**

Here is the reasoning based on the project's compliance requirements and current CLI architecture:

### 1. Separation of Developer vs. End-User State
The `.claude/skills/` folder currently being read by `tqcli/skills/loader.py` is a local project-level directory used by us (the developers) while working on the codebase. 
However, when an end-user strictly installs the tool via `pip install`, they will invoke `tqcli` context-free across their entire system. They will not have a `.claude/skills/` directory in every folder they execute the CLI from, nor do we want to force them to create one.

By generating new skills into `~/.tqcli/skills/`, we guarantee a unified, global sandbox for the user's custom extensions, identical to how `~/.tqcli/models/` and `~/.tqcli/config.yaml` operate today.

### 2. Legal, Security, and Compliance Bounds
As discussed in the compliance reviews, we must not automatically cross-pollinate tools from competitor/external CLIs (like `.claude` or `.gemini`). Reading from or writing to a `.claude/` directory creates immediate data governance and sandbox bleeding risks if the user actually uses Claude Code.

By explicitly forcing `tqCLI` tools to live in `~/.tqcli/skills/`, we maintain strict sandboxing, proving that our CLI only executes and manages code within its own opt-in jurisdiction.

### What this means for the implementation ticket:
When you write the GitHub issue, please explicitly state:
1. The new feature will write to `~/.tqcli/skills/<name>/` as defined in the PRD.
2. The existing tool loader (`tqcli/skills/loader.py`) **must be updated** in this PR to read from the new global `~/.tqcli/skills/` path (in addition to or instead of its current relative `.claude/skills/` path).

Let me know if you need anything else before opening the ticket!