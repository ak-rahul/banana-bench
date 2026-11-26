# Development task runner for banana-bench. Run `just` to list recipes.
#
# `just` is installed with the dev extra (`pip install -e ".[dev]"`, via the
# `rust-just` package) and works in PowerShell, cmd, Git Bash, macOS and Linux.
# The cross-platform logic lives in scripts/dev.py; these recipes only wrap it.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# Prefer the project's .venv interpreter; fall back to whatever `python` is on PATH.
python := if path_exists(".venv/Scripts/python.exe") == "true" { ".venv/Scripts/python.exe" } else if path_exists(".venv/bin/python") == "true" { ".venv/bin/python" } else { "python" }

# This `just` binary, callable even when it isn't on PATH (PowerShell needs `&` to run a quoted path).
just := if os_family() == "windows" { "& " + quote(just_executable()) } else { quote(just_executable()) }

[private]
default:
    @{{just}} --justfile {{quote(justfile())}} --list

# Install the package in editable mode with all dev dependencies
install:
    {{python}} -m pip install -e ".[dev]"

# Remove caches and build artifacts (pass --dry-run to preview)
clean *args:
    {{python}} scripts/dev.py clean {{args}}

# Clean, build the sdist and wheel, then run twine check
build:
    {{python}} scripts/dev.py build

# Run every CI gate: ruff, ruff format --check, mypy, pytest
check:
    {{python}} scripts/dev.py check

# Run pytest; extra arguments are passed through (e.g. `just test tests/test_cli.py -k list`)
test *args:
    {{python}} -m pytest {{args}}

# Auto-format and apply safe lint fixes
fmt:
    {{python}} -m ruff format bananabench tests scripts
    {{python}} -m ruff check --fix bananabench tests scripts

# Build the Sphinx HTML docs into docs/_build/html (needs the docs extra)
docs:
    {{python}} -m sphinx -b html docs docs/_build/html
