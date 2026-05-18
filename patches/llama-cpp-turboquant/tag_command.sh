#!/usr/bin/env bash
# Tag command for triggering the wheel publish, plus post-publish verification.
#
# Run AFTER:
#  1. `tqcli/llama-cpp-python-turboquant` repo exists with Artifacts 1-4 applied.
#  2. PyPI Pending Publisher is registered against
#     project=llama-cpp-python-turboquant, owner=tqcli, repo=llama-cpp-python-turboquant,
#     workflow=wheels.yml, environment=(blank).

set -euo pipefail

# 1. Tag and push.
gh repo clone tqcli/llama-cpp-python-turboquant
cd llama-cpp-python-turboquant
git tag v0.3.0-tq1
git push origin v0.3.0-tq1

# 2. Watch the workflow run.
gh run watch --exit-status --repo tqcli/llama-cpp-python-turboquant

# 3. Verify on PyPI.
pip index versions llama-cpp-python-turboquant
# Expected: 0.3.0

# 4. Smoke-install on a clean venv.
python -m venv /tmp/v-llcpt
# shellcheck disable=SC1091
source /tmp/v-llcpt/bin/activate
pip install llama-cpp-python-turboquant
python -c "import llama_cpp; print(llama_cpp.TURBOQUANT_BUILD, llama_cpp.TURBOQUANT_KV_TYPES)"
# Expected: True ('turbo2', 'turbo3', 'turbo4')

# 5. (optional) Confirm the PyPI Pending Publisher auto-promoted to Active.
bash .claude/skills/tq-pypi/scripts/check_name_available.sh llama-cpp-python-turboquant
# Expected: TAKEN by us
