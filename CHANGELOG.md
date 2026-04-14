# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-13

### Added

- Initial release of tqCLI (TurboQuant CLI)
- Cross-platform CLI with Click commands: `chat`, `system info`, `model list/pull/remove`, `benchmark`, `security audit`, `skills`, `handoff`, `config show/init`
- **Inference backends**: llama.cpp (via llama-cpp-python) and vLLM
- **Model registry**: 6 pre-configured profiles across 3 model families
  - Google Gemma 4 (12B, 27B)
  - Qwen2.5-Coder (7B, 32B)
  - Qwen2.5-Instruct (7B, 32B)
- **Smart router**: keyword-based prompt classification with domain-specific model ranking
- **Performance monitor**: real-time tokens/second tracking with configurable thresholds
- **Handoff system**: generates context files for Claude Code, Gemini CLI, Aider
- **Security layer**: venv isolation, environment detection (WSL2/container/bare-metal), resource guards, audit logging
- **Skills system**: 5 tqCLI skills (system-info, model-manager, benchmark, security-audit, handoff-generator)
- **Rich terminal UI**: tables, panels, streaming output, colored performance stats
- Hardware auto-detection: CPU, RAM, GPU/VRAM, Apple Silicon Metal, WSL2
- YAML configuration at `~/.tqcli/config.yaml`
- 11 passing tests covering all core modules
