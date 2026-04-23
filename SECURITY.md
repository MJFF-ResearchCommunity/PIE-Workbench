# Security Policy

## Reporting a Vulnerability

PIE Workbench is a research tool used by clinicians and scientists working
with sensitive health data. If you believe you have found a security
vulnerability, please report it privately rather than opening a public issue.

**How to report:**

- Email the maintainers at `Cameron@AllianceAI.co`, or
- Use GitHub's private vulnerability reporting feature:
  https://github.com/MJFF-ResearchCommunity/PIE-Workbench/security/advisories/new

Please include:

- A clear description of the issue and its potential impact
- Steps to reproduce (minimal example if possible)
- The affected version / commit SHA
- Any suggested mitigation, if you have one

We will acknowledge the report within 5 business days, investigate, and keep
you updated on a fix timeline. We ask that you do not publicly disclose the
issue until a fix has been released.

## Scope

This policy covers the PIE Workbench application (this repository) and its
direct dependencies maintained under the
[MJFF-ResearchCommunity](https://github.com/MJFF-ResearchCommunity) GitHub
organization (`PIE`, `PIE-clean`).

## Data handling note

PIE Workbench runs locally. It does not transmit data to any external service.
The backend binds to `127.0.0.1` only. If you discover a configuration or code
path that changes this, please report it as a vulnerability.
