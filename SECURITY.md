# Security Policy

## Reporting a Vulnerability

**Do NOT open a public GitHub issue for security vulnerabilities.**

If you discover a security vulnerability in tqCLI, please report it responsibly:

1. **Email**: Send a detailed report to the project maintainer via GitHub's private security advisory feature (Settings > Security > Advisories > New draft advisory).
2. **Include**:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if you have one)

You will receive an acknowledgment within 48 hours, and we will work with you to understand and address the issue before any public disclosure.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Security Model

tqCLI runs quantized LLMs locally on the user's machine. The security boundaries are:

### What tqCLI Does

- Loads GGUF model files from disk into memory for inference
- Downloads model files from HuggingFace Hub over HTTPS
- Writes configuration, audit logs, and handoff files to `~/.tqcli/`
- Optionally creates a Python virtual environment at `~/.tqcli/venv`
- Detects hardware (CPU, RAM, GPU) by reading system files and running `nvidia-smi`

### What tqCLI Does NOT Do

- Send prompts, responses, or usage data to external servers
- Collect telemetry or analytics
- Execute arbitrary code from downloaded model files (GGUF files are weight tensors, not executable code)
- Require network access after model download (fully offline capable)
- Run with elevated/root privileges (should not be run as root)

### Threat Model

| Threat | Mitigation |
|--------|------------|
| Malicious model files | GGUF format is data-only (tensor weights); tqCLI does not execute code from model files. Always download from trusted HuggingFace repos. |
| Dependency supply chain | Dependencies are pinned. Review `pyproject.toml` before installing. CI runs dependency audit. |
| Prompt injection via model output | tqCLI does not execute model output as code. Handoff files are markdown text, not scripts. |
| Resource exhaustion | Resource guards enforce RAM and GPU memory limits. Configurable thresholds. |
| Audit log tampering | Audit log is append-only by design. File permissions should restrict write access. |
| Malicious contributions | All PRs require review, CI must pass, CODEOWNERS enforces maintainer approval on critical paths. |

### Security Best Practices for Users

1. **Run in an isolated environment** — WSL2, Docker container, or a dedicated virtual environment. Do not install tqCLI system-wide as root.
2. **Only download models from trusted sources** — the built-in model registry points to official HuggingFace repos. Verify repo ownership before downloading community models.
3. **Review the audit log** — check `~/.tqcli/audit.log` periodically for unexpected events.
4. **Keep dependencies updated** — run `pip install --upgrade tqcli` to get security patches.
5. **Do not expose the inference server to the network** — if using llama.cpp server mode, bind to `localhost` only.

## Security-Related Configuration

```yaml
# ~/.tqcli/config.yaml
security:
  use_venv: true              # Isolate tqCLI packages in a venv
  sandbox_enabled: true       # Enable sandboxing
  audit_log: true             # Log all security-relevant events
  audit_log_path: ~/.tqcli/audit.log
  max_memory_percent: 80.0    # Prevent OOM by limiting memory usage
  max_gpu_memory_percent: 90.0
```

## Responsible Disclosure

We follow a coordinated disclosure process:

1. Reporter submits vulnerability privately
2. We confirm and assess severity within 48 hours
3. We develop a fix
4. We release the fix and credit the reporter (unless they prefer anonymity)
5. Public disclosure after the fix is available
