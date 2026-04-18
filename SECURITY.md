# Security Policy

This document outlines security practices for the NOVA-AI-Voice-Assistant project.

---

## Dependency Vulnerability Scanning

### Running pip-audit locally

Before committing any changes that touch `requirements.txt`, run `pip-audit` to
check all pinned dependencies against the OSV and PyPI Advisory databases:

```bash
# Install pip-audit (one-time setup)
pip install pip-audit

# Scan the current environment against requirements.txt
pip-audit -r requirements.txt
```

A clean run will exit with code `0` and print no vulnerabilities. If issues are
found, update the affected package to a patched version in `requirements.txt`
and re-run the scan before opening a pull request.

### Automated scanning with GitHub Dependabot

Enable Dependabot in the repository to receive automated pull requests whenever
a pinned dependency has a known CVE:

1. Go to **Settings → Security → Code security and analysis**.
2. Enable **Dependabot alerts** and **Dependabot security updates**.
3. Optionally add a `.github/dependabot.yml` to configure the update schedule:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
```

Dependabot will open PRs against `requirements.txt` automatically when patches
are available, keeping the pinned versions current without manual tracking.

---

## API Key and Secret Management

**Never commit secrets to the repository.** All API keys, tokens, database
credentials, and other sensitive values must be managed as environment
variables.

### On Railway

Set secrets through the Railway dashboard under **Variables** for each service.
They are injected at runtime and are never stored in the codebase or build
artifacts. See the [Railway docs](https://docs.railway.app/guides/variables)
for details.

### Locally

Copy `.env.example` (if present) to `.env` and fill in your values:

```bash
cp .env.example .env
```

The `.env` file is listed in `.gitignore` and will never be committed. Do not
remove that entry. Never add a `.env` file to a commit, even temporarily.

### What to do if a secret is accidentally committed

1. **Rotate the secret immediately** — assume it is compromised.
2. Remove it from the repository history using `git filter-repo` or BFG
   Repo-Cleaner.
3. Force-push the cleaned history and notify all collaborators to re-clone.
4. Audit access logs for the exposed credential.

---

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please open a
**private** GitHub Security Advisory rather than a public issue. Navigate to
**Security → Advisories → New draft security advisory** in the repository and
provide a description, reproduction steps, and impact assessment. We aim to
respond within 72 hours.
