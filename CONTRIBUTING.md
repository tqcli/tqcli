# Contributing to tqCLI

Thank you for your interest in contributing to tqCLI. This document explains how to contribute effectively and what to expect during the review process.

## Before You Start

1. **Check existing issues** — someone may already be working on what you have in mind.
2. **Open an issue first** for non-trivial changes. This avoids wasted effort if the approach doesn't align with the project direction.
3. **Read the [Architecture Guide](docs/ARCHITECTURE.md)** to understand how the codebase is structured.

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- A Unix-like environment (Linux, macOS, or WSL2 on Windows)

### Clone and Install

```bash
git clone https://github.com/ithllc/tqCLI.git
cd tqCLI

# Create a virtual environment (required)
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
tqcli --version
python -m pytest tests/ -v
```

### Optional: Install inference backends for testing

```bash
# llama.cpp backend (cross-platform)
pip install llama-cpp-python

# vLLM backend (Linux + NVIDIA GPU only)
pip install vllm
```

## Making Changes

### Branch Naming

Create a branch from `main` with a descriptive name:

```
feature/add-mixtral-support
fix/router-math-classification
docs/improve-getting-started
```

### Code Standards

- **Python 3.10+** — use type hints, `from __future__ import annotations`
- **Formatting** — run `ruff check` and `ruff format` before committing
- **No unnecessary dependencies** — every new dependency must be justified in the PR description
- **Tests required** — new features need tests, bug fixes need a regression test
- **No model weights in the repo** — only configurations, references, and download logic

### Running Tests

```bash
# Full test suite
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ -v --cov=tqcli

# Linting
ruff check tqcli/ tests/
ruff format --check tqcli/ tests/
```

### What Belongs Where

| Change Type | Location |
|-------------|----------|
| New inference backend | `tqcli/core/` — implement `InferenceEngine` ABC |
| New model family | `tqcli/core/model_registry.py` — add `ModelProfile` entries |
| New CLI command | `tqcli/cli.py` — add Click command group/command |
| New skill | `.claude/skills/tq-<name>/` — follow existing SKILL.md pattern |
| Bug fix | Wherever the bug is, plus a test in `tests/` |
| Documentation | `docs/` or update `README.md` |

## Pull Request Process

1. **Fork the repo** and create your branch from `main`.
2. **Make your changes** with tests.
3. **Run the full test suite** — all tests must pass.
4. **Run the linter** — no ruff errors.
5. **Open a PR** using the PR template.
6. **Wait for review** — a maintainer will review your PR. Expect feedback; it's part of the process, not a rejection.

### PR Requirements

- [ ] All tests pass (`python -m pytest tests/ -v`)
- [ ] Linter passes (`ruff check tqcli/ tests/`)
- [ ] New code has tests
- [ ] No new dependencies without justification
- [ ] PR description explains *what* and *why*
- [ ] No secrets, credentials, or API keys
- [ ] No model weight files (`.gguf`, `.bin`, `.safetensors`)

### Review Timeline

- Small fixes (typos, docs): reviewed within a few days
- Features and non-trivial changes: reviewed within a week
- Large architectural changes: discuss in an issue first

## What We Accept

- Bug fixes with regression tests
- New model family support (with documented strength scores and sources)
- New inference backend integrations
- Performance improvements with benchmarks
- Documentation improvements
- Platform-specific fixes (especially Windows/macOS edge cases)
- Security improvements

## What We Don't Accept

- Changes that add model weight files to the repo
- PRs without tests for new functionality
- Dependency additions without clear justification
- Changes that break cross-platform compatibility without discussion
- Code that phones home, collects telemetry, or makes network requests without explicit user action
- Obfuscated code

## Reporting Issues

- **Bugs**: Use the "Bug Report" issue template. Include your `tqcli system info --json` output.
- **Feature requests**: Use the "Feature Request" issue template.
- **Security vulnerabilities**: Do NOT open a public issue. See [SECURITY.md](SECURITY.md).

## Code of Conduct

This project follows a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you agree to uphold it.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
