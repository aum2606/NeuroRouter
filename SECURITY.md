# Security policy

## Reporting a vulnerability

Please do not publish exploitable details in a public issue. Report vulnerabilities through
GitHub's private vulnerability reporting feature for this repository. Include the affected
component, reproduction steps, impact, and any suggested mitigation.

## Supported version

Security fixes target the current `main` branch while the project is pre-1.0.

## Trust boundaries

- `.env` and hosting-platform secrets must never be committed or pasted into request text.
- Request text, model outputs, evidence, and raw provider responses are untrusted data.
- SQLite traces can contain request and response content; protect and expire them accordingly.
- Uploaded PDF, TXT, and Markdown files are size- and type-checked, but should still be treated as
  untrusted content.
- The optional Python runner is disabled by default. Its AST restrictions, timeout, isolated mode,
  and output limits reduce accidental harm but do **not** make it a security sandbox.

Do not enable code execution for an internet-facing or multi-tenant deployment. Use an operating
system or container isolation boundary with no credentials, no network, a read-only filesystem,
strict CPU/memory limits, and disposable storage before accepting untrusted code.

## Data handling

NeuroRouter avoids placing API secrets in router state and trace payloads. It does not automatically
redact secrets that a user manually enters into a prompt or uploaded document. Clear `data/` before
sharing a working directory or screen recording, and configure retention appropriate to the data
being processed.
