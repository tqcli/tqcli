"""AI Skills Builder — generate tqCLI skills from a PRD + Technical Plan.

Reads two markdown files, sends them through the active inference engine with a
structured prompt, parses the `<file path="...">...</file>` blocks out of the
response, validates the generated Python with `ast.parse()`, and (after user
review) writes the skill into `~/.tqcli/skills/<name>/`.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from tqcli.core.engine import ChatMessage, InferenceEngine

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "skill_generation_prompt.md"

_SLUG_RE = re.compile(r"[^a-zA-Z0-9_-]+")
# Primary: <file path="...">...</file>   (quoted or single-quoted)
_FILE_BLOCK_RE = re.compile(
    r"""<file\s+path=['"](?P<path>[^'"]+)['"]\s*>(?P<body>.*?)(?:</file>|\Z)""",
    re.DOTALL,
)
# Fallback: FILE: path   followed by a fenced code block on the next line(s).
_FENCED_FILE_RE = re.compile(
    r"""(?:^|\n)\s*(?:FILE|File|file):\s*(?P<path>[^\s`]+)\s*\n+```[a-zA-Z0-9+._-]*\n(?P<body>.*?)(?:\n```|\Z)""",
    re.DOTALL,
)
_THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.DOTALL)


@dataclass
class GeneratedFile:
    relative_path: str
    content: str
    is_python: bool
    ast_ok: bool
    error: str = ""


@dataclass
class GenerationResult:
    skill_name: str
    target_dir: Path
    files: list[GeneratedFile] = field(default_factory=list)
    thought: str = ""
    raw_response: str = ""

    @property
    def valid(self) -> bool:
        if not self.files:
            return False
        if not any(f.relative_path == "SKILL.md" for f in self.files):
            return False
        return all(f.ast_ok for f in self.files if f.is_python)

    @property
    def errors(self) -> list[str]:
        errs = []
        if not self.files:
            errs.append("No <file> blocks parsed from model output.")
        if not any(f.relative_path == "SKILL.md" for f in self.files):
            errs.append("SKILL.md missing from generated files.")
        for f in self.files:
            if f.is_python and not f.ast_ok:
                errs.append(f"{f.relative_path}: {f.error}")
        return errs


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.strip().lower())
    slug = slug.strip("-")
    return slug or "skill"


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text()


def build_prompt(prd_text: str, plan_text: str, skill_name: str) -> str:
    template = load_prompt_template()
    slug = slugify(skill_name)
    return (
        template.replace("{{ skill_name }}", skill_name)
        .replace("{{ skill_slug }}", slug)
        .replace("{{ prd }}", prd_text)
        .replace("{{ plan }}", plan_text)
    )


def _safe_relative_path(raw: str) -> str | None:
    """Reject absolute paths, parent-traversal, or backslash windows paths."""
    if not raw:
        return None
    if raw.startswith("/") or raw.startswith("\\"):
        return None
    if "\\" in raw:
        return None
    parts = raw.split("/")
    if any(p in ("", "..", ".") for p in parts):
        return None
    if not parts[-1]:
        return None
    return "/".join(parts)


def _make_generated_file(raw_path: str, body: str) -> GeneratedFile | None:
    safe = _safe_relative_path(raw_path.strip())
    if safe is None:
        return None
    body = body.strip("\n")
    is_py = safe.endswith(".py")
    ast_ok = True
    err = ""
    if is_py:
        try:
            ast.parse(body)
        except SyntaxError as e:
            ast_ok = False
            err = f"{e.__class__.__name__}: {e.msg} (line {e.lineno})"
    return GeneratedFile(
        relative_path=safe,
        content=body,
        is_python=is_py,
        ast_ok=ast_ok,
        error=err,
    )


def parse_model_output(text: str) -> tuple[str, list[GeneratedFile]]:
    """Extract <thought> and file blocks from the raw LLM text.

    Primary format: <file path="...">...</file>.
    Fallback format: `FILE: path` header followed by a fenced code block.
    """
    thought = ""
    m = _THOUGHT_RE.search(text)
    if m:
        thought = m.group(0)

    files: list[GeneratedFile] = []
    seen: set[str] = set()

    for m in _FILE_BLOCK_RE.finditer(text):
        gf = _make_generated_file(m.group("path"), m.group("body"))
        if gf and gf.relative_path not in seen:
            seen.add(gf.relative_path)
            files.append(gf)

    if not files:
        for m in _FENCED_FILE_RE.finditer(text):
            gf = _make_generated_file(m.group("path"), m.group("body"))
            if gf and gf.relative_path not in seen:
                seen.add(gf.relative_path)
                files.append(gf)

    return thought, files


def generate_skill(
    engine: InferenceEngine,
    prd_path: Path,
    plan_path: Path,
    skill_name: str,
    max_tokens: int = 2500,
    temperature: float = 0.2,
) -> GenerationResult:
    """Run the LLM and parse its output. Does NOT write to disk."""
    prd_text = prd_path.read_text()
    plan_text = plan_path.read_text()
    prompt = build_prompt(prd_text, plan_text, skill_name)
    # Qwen3 auto-enters <think> mode and burns the token budget on reasoning.
    # `/no_think` is Qwen3's per-turn opt-out; harmless on Gemma 4 / other families.
    prompt = prompt + "\n\n/no_think"

    messages = [
        ChatMessage(
            role="system",
            content=(
                "You generate minimal, working tqCLI skills. Follow the output "
                "format exactly. Do NOT emit any <think> or reasoning blocks. "
                "Emit only the <thought> and <file> tags defined in the user "
                "message."
            ),
        ),
        ChatMessage(role="user", content=prompt),
    ]
    result = engine.chat(messages, max_tokens=max_tokens, temperature=temperature)
    thought, files = parse_model_output(result.text)

    target_dir = Path.home() / ".tqcli" / "skills" / slugify(skill_name)
    return GenerationResult(
        skill_name=skill_name,
        target_dir=target_dir,
        files=files,
        thought=thought,
        raw_response=result.text,
    )


def write_skill(result: GenerationResult, overwrite: bool = False) -> Path:
    """Write the generated files to ~/.tqcli/skills/<name>/. Caller must have reviewed."""
    if result.target_dir.exists() and any(result.target_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{result.target_dir} already exists and is non-empty — pass overwrite=True")

    result.target_dir.mkdir(parents=True, exist_ok=True)
    for gf in result.files:
        out = result.target_dir / gf.relative_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(gf.content)
        if gf.is_python:
            out.chmod(0o755)
    # Persist the raw model response alongside for inspection/debugging.
    (result.target_dir / ".raw_response.md").write_text(result.raw_response)
    return result.target_dir
