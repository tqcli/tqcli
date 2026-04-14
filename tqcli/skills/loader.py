"""Skill discovery and loading — mirrors Claude Code's skill architecture."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillMetadata:
    name: str
    description: str
    path: Path
    has_scripts: bool = False
    has_templates: bool = False
    has_memory: bool = False
    scripts: list[Path] = field(default_factory=list)


def parse_skill_frontmatter(content: str) -> dict[str, str]:
    """Parse YAML-like frontmatter from SKILL.md."""
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {}
    frontmatter = {}
    for line in match.group(1).split("\n"):
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip()
    return frontmatter


class SkillLoader:
    """Discovers and loads skills from a skills directory."""

    def __init__(self, skills_dirs: list[Path]):
        self.skills_dirs = skills_dirs
        self._skills: dict[str, SkillMetadata] = {}

    def discover(self) -> dict[str, SkillMetadata]:
        self._skills.clear()
        for skills_dir in self.skills_dirs:
            if not skills_dir.exists():
                continue
            for skill_dir in sorted(skills_dir.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue
                content = skill_md.read_text()
                meta = parse_skill_frontmatter(content)
                name = meta.get("name", skill_dir.name)

                scripts = []
                scripts_dir = skill_dir / "scripts"
                if scripts_dir.exists():
                    scripts = sorted(scripts_dir.glob("*.py"))

                skill = SkillMetadata(
                    name=name,
                    description=meta.get("description", ""),
                    path=skill_dir,
                    has_scripts=len(scripts) > 0,
                    has_templates=(skill_dir / "templates").exists(),
                    has_memory=(skill_dir / "memory").exists(),
                    scripts=scripts,
                )
                self._skills[name] = skill

        return self._skills

    def get_skill(self, name: str) -> SkillMetadata | None:
        if not self._skills:
            self.discover()
        return self._skills.get(name)

    def list_skills(self) -> list[SkillMetadata]:
        if not self._skills:
            self.discover()
        return list(self._skills.values())

    def get_tq_skills(self) -> list[SkillMetadata]:
        """Return only tqCLI-specific skills (tq-* prefix)."""
        return [s for s in self.list_skills() if s.name.startswith("tq-")]
