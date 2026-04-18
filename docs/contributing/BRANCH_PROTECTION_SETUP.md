# Branch Protection Setup Guide

This guide documents the branch protection rules to configure on the GitHub repository. These settings cannot be set via files — they must be configured in the GitHub UI or via the API.

## Configure via GitHub UI

Go to: **Settings > Branches > Add branch protection rule**

### Rule: Protect `main`

**Branch name pattern:** `main`

**Enable these settings:**

- [x] **Require a pull request before merging**
  - [x] Require approvals: **1**
  - [x] Dismiss stale pull request approvals when new commits are pushed
  - [x] Require review from Code Owners (uses `.github/CODEOWNERS`)

- [x] **Require status checks to pass before merging**
  - [x] Require branches to be up to date before merging
  - Required status checks (add after first CI run):
    - `test (ubuntu-latest, 3.12)`
    - `test (macos-latest, 3.12)`
    - `test (windows-latest, 3.12)`
    - `security-audit`

- [x] **Require signed commits** (optional but recommended)

- [x] **Do not allow bypassing the above settings**
  - This prevents even admins from pushing directly to main

- [ ] Require linear history (optional — keeps git log clean)

- [x] **Restrict who can push to matching branches**
  - Only allow: repository admins

### Additional Settings (Settings > General)

- [x] **Automatically delete head branches** — clean up merged PR branches
- Under "Pull Requests":
  - [x] Allow squash merging (recommended for clean history)
  - [x] Allow merge commits
  - [ ] Allow rebase merging (disable to keep consistent merge style)

## Configure via GitHub CLI

If you prefer the command line:

```bash
# Requires gh CLI authenticated with admin access
gh api repos/ithllc/tqCLI/branches/main/protection \
  --method PUT \
  --field required_status_checks='{"strict":true,"contexts":["test (ubuntu-latest, 3.12)","security-audit"]}' \
  --field enforce_admins=true \
  --field required_pull_request_reviews='{"required_approving_review_count":1,"dismiss_stale_reviews":true,"require_code_owner_reviews":true}' \
  --field restrictions=null
```

## When to Apply

Apply these rules **after the first successful CI run** — GitHub needs to see the status check names before you can require them. The recommended sequence:

1. Push the CI workflow (`.github/workflows/ci.yml`)
2. Open a test PR to trigger CI
3. Verify all checks pass
4. Configure branch protection with the check names
5. Merge the test PR
6. All future changes must go through PR review + CI
