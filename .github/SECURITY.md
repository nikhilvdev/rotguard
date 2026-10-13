# Security Policy

## Supported Versions

`rotguard` is pre-release. Security fixes target the latest released version only. Please upgrade
before reporting an issue to confirm it is still present.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Use GitHub's private vulnerability reporting instead:
[Report a vulnerability](https://github.com/nikhilvdev/rotguard/security/advisories/new)
(the "Security" tab of this repository, then "Report a vulnerability"). This opens a private
advisory visible only to maintainers until a fix is ready.

Please include:

- A description of the vulnerability and its potential impact
- Steps to reproduce, or a minimal proof-of-concept
- The affected version(s)

Note that rotguard stores message text in a local SQLite file (ledger entries, provenance, and the
text of context items). Treat that file as sensitive as the conversations it came from.
