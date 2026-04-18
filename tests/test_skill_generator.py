"""Unit tests for tqcli.core.skill_generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from tqcli.core.skill_generator import (
    GenerationResult,
    _safe_relative_path,
    build_prompt,
    parse_model_output,
    slugify,
    write_skill,
)


def test_slugify_basic():
    assert slugify("My New Skill") == "my-new-skill"
    assert slugify("a / b") == "a-b"
    assert slugify("") == "skill"
    assert slugify("  --weird__name  ") == "weird__name"


def test_safe_relative_path_rejects_traversal():
    assert _safe_relative_path("../evil.py") is None
    assert _safe_relative_path("/etc/passwd") is None
    assert _safe_relative_path("scripts/../evil.py") is None
    assert _safe_relative_path("sub\\x.py") is None
    assert _safe_relative_path("") is None
    assert _safe_relative_path("scripts/") is None


def test_safe_relative_path_accepts_normal():
    assert _safe_relative_path("SKILL.md") == "SKILL.md"
    assert _safe_relative_path("scripts/run.py") == "scripts/run.py"


def test_parse_model_output_two_files():
    text = (
        "<thought>Short reasoning here.</thought>\n"
        "<file path=\"SKILL.md\">---\nname: demo\ndescription: test\n---\n# demo\n</file>\n"
        "<file path=\"scripts/run.py\">print('hello')\n</file>\n"
    )
    thought, files = parse_model_output(text)
    assert "<thought>" in thought
    assert len(files) == 2
    paths = [f.relative_path for f in files]
    assert paths == ["SKILL.md", "scripts/run.py"]
    py = [f for f in files if f.is_python][0]
    assert py.ast_ok
    assert py.content.startswith("print")


def test_parse_model_output_rejects_traversal():
    text = (
        "<file path=\"../../evil.py\">bad()</file>\n"
        "<file path=\"SKILL.md\">ok</file>\n"
    )
    _, files = parse_model_output(text)
    assert len(files) == 1
    assert files[0].relative_path == "SKILL.md"


def test_parse_model_output_flags_broken_python():
    text = (
        "<file path=\"SKILL.md\">ok</file>\n"
        "<file path=\"scripts/bad.py\">def broken(:\n</file>\n"
    )
    _, files = parse_model_output(text)
    py = [f for f in files if f.is_python][0]
    assert not py.ast_ok
    assert "SyntaxError" in py.error


def test_parse_model_output_handles_unclosed_file_tag():
    text = (
        "<file path=\"SKILL.md\">---\nname: x\n---\nbody\n"
    )
    _, files = parse_model_output(text)
    assert len(files) == 1
    assert files[0].relative_path == "SKILL.md"
    assert "body" in files[0].content


def test_generation_result_valid_and_errors():
    good = GenerationResult(
        skill_name="demo",
        target_dir=Path("/tmp/does-not-matter"),
        files=[
            parse_model_output('<file path="SKILL.md">ok</file>')[1][0],
            parse_model_output('<file path="scripts/x.py">x = 1</file>')[1][0],
        ],
    )
    assert good.valid
    assert good.errors == []

    bad = GenerationResult(skill_name="demo", target_dir=Path("/tmp"))
    assert not bad.valid
    assert "No <file> blocks" in bad.errors[0]


def test_write_skill_writes_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # Force target_dir to land under tmp_path even though Path.home() was already
    # evaluated inside generate_skill — we construct the result manually.
    _, files = parse_model_output(
        '<file path="SKILL.md">---\nname: demo\ndescription: t\n---\n</file>\n'
        '<file path="scripts/run.py">print(\'hi\')\n</file>\n'
    )
    result = GenerationResult(
        skill_name="demo",
        target_dir=tmp_path / ".tqcli" / "skills" / "demo",
        files=files,
    )
    path = write_skill(result)
    assert (path / "SKILL.md").exists()
    assert (path / "scripts" / "run.py").exists()
    assert (path / "scripts" / "run.py").read_text().startswith("print")


def test_build_prompt_interpolates(tmp_path):
    prd = tmp_path / "prd.md"
    plan = tmp_path / "plan.md"
    prd.write_text("PRD-BODY")
    plan.write_text("PLAN-BODY")
    out = build_prompt(prd.read_text(), plan.read_text(), "My Skill")
    assert "PRD-BODY" in out
    assert "PLAN-BODY" in out
    assert "My Skill" in out
    assert "my-skill" in out  # slug
