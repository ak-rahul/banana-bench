# Security Policy

## Supported Versions

`banana-bench` follows [Semantic Versioning](https://semver.org/). Only the
most recently published minor version on PyPI receives security fixes; there
is no long-term-support branch at this project's current size.

| Version | Supported          |
| ------- | ------------------ |
| Latest  | :white_check_mark: |
| Older   | :x:                |

## Reporting a Vulnerability

Please **do not** open a public GitHub issue for a suspected security
vulnerability.

Instead, report it privately using
[GitHub's private vulnerability reporting](https://github.com/ak-rahul/banana-bench/security/advisories/new)
for this repository, or by emailing
[ak-rahul@users.noreply.github.com](mailto:ak-rahul@users.noreply.github.com).
Include:

- A description of the vulnerability and its potential impact.
- Steps to reproduce it (a minimal code sample is ideal).
- The affected version(s), if known.

This is a single-maintainer, volunteer-run project, so please allow a
reasonable amount of time for a response — an acknowledgment within a few
days and a fix or mitigation plan as soon as practical after that. Credit
will be given in the release notes for any confirmed report, unless you
prefer to remain anonymous.

## Scope

`banana-bench` is a library of pure mathematical functions and benchmarking
utilities; it does not execute untrusted input, make network calls, or
handle secrets/credentials as part of its core functionality. The most
plausible categories of report are:

- A dependency (NumPy, or an optional extra like matplotlib/joblib/plotly)
  pulling in a known vulnerability via `pyproject.toml`'s pinned/allowed
  version ranges.
- Unsafe deserialization or arbitrary code execution reachable through a
  public API (e.g. a file-loading utility) that doesn't already document
  itself as trusting its input.

Reports about the CI/release pipeline (`publish.yml`'s PyPI trusted
publishing via OIDC, in particular) are also welcome and taken seriously.
