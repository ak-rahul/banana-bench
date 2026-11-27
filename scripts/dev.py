"""
Development helper for banana-bench.

Cross-platform (Windows, macOS, Linux) and standard-library only, so it runs
the same everywhere without needing ``make`` or a shell.

Usage::

    python scripts/dev.py clean            # remove caches and build artifacts
    python scripts/dev.py clean --dry-run  # list what would be removed
    python scripts/dev.py build            # clean, build sdist + wheel, twine check
    python scripts/dev.py check            # lint, format check, type check, tests (as CI)
    python scripts/dev.py release-notes    # current version's CHANGELOG.md section (+ release gate)
    python scripts/dev.py version          # pyproject.toml's version
    python scripts/dev.py ci-changes ...   # CI: which jobs a push/PR needs (ci.yml)
    python scripts/dev.py release-check    # CI: is pyproject.toml's version a new release? (publish.yml)

The two ``ci``/``release`` commands hold the logic the workflows used to carry as
inline shell and Python, so it's linted, type-checked and unit-tested
(tests/test_dev_script.py) like the rest of the code. Both print their result and,
when ``GITHUB_OUTPUT`` is set, also append it there as ``key=value`` lines.

``clean`` never touches the ``.git`` directory or any virtual environment
(any directory containing ``pyvenv.cfg``, plus the usual ``.venv``/``venv``
names), so it is safe to run inside an activated environment.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Directories removed wherever they appear in the tree.
CACHE_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}

# Paths removed only at the repository root (relative to ROOT).
ROOT_ARTIFACTS = [
    "build",
    "dist",
    ".eggs",
    "htmlcov",
    ".coverage",
    "coverage.xml",
    "docs/_build",
]

# Globs (relative to ROOT) for artifacts whose exact name varies.
ROOT_GLOBS = ["*.egg-info", ".coverage.*"]

# Never descend into these.
SKIP_DIR_NAMES = {".git", ".venv", "venv", "env", "ENV", ".tox", ".nox", "node_modules"}


def _is_venv(path: Path) -> bool:
    return (path / "pyvenv.cfg").is_file()


def _find_targets() -> list[Path]:
    targets: list[Path] = []

    for rel in ROOT_ARTIFACTS:
        path = ROOT / rel
        if path.exists():
            targets.append(path)
    for pattern in ROOT_GLOBS:
        targets.extend(Path(p) for p in glob.glob(str(ROOT / pattern)))

    def walk(directory: Path) -> None:
        for child in directory.iterdir():
            if child.is_symlink():
                continue
            if child.is_dir():
                if child.name in SKIP_DIR_NAMES or _is_venv(child):
                    continue
                if child.name in CACHE_DIR_NAMES:
                    targets.append(child)
                    continue
                walk(child)
            elif child.suffix in {".pyc", ".pyo"}:
                targets.append(child)

    walk(ROOT)
    # De-duplicate while keeping order (a root artifact could also match a glob).
    return list(dict.fromkeys(targets))


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def clean(dry_run: bool = False) -> int:
    targets = _find_targets()
    if not targets:
        print("Nothing to clean.")
        return 0
    verb = "Would remove" if dry_run else "Removing"
    for path in targets:
        print(f"{verb}: {path.relative_to(ROOT)}")
        if not dry_run:
            _remove(path)
    print(f"{len(targets)} path(s) {'found' if dry_run else 'removed'}.")
    return 0


def _run(*cmd: str) -> int:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def build() -> int:
    for rel in ("build", "dist"):
        if (ROOT / rel).exists():
            shutil.rmtree(ROOT / rel)
    for path in glob.glob(str(ROOT / "*.egg-info")):
        shutil.rmtree(path)

    rc = _run(sys.executable, "-m", "build")
    if rc != 0:
        return rc
    dists = sorted(glob.glob(str(ROOT / "dist" / "*")))
    return _run(sys.executable, "-m", "twine", "check", "--strict", *dists)


def check() -> int:
    """Run the same gates as CI (ci.yml's quality and test jobs), stopping at the first failure."""
    steps = [
        (sys.executable, "-m", "ruff", "check", "bananabench", "tests", "scripts"),
        (sys.executable, "-m", "ruff", "format", "--check", "bananabench", "tests", "scripts"),
        (sys.executable, "-m", "mypy", "bananabench"),
        (sys.executable, "-m", "pytest", "tests"),
    ]
    for step in steps:
        rc = _run(*step)
        if rc != 0:
            return rc
    print("\nAll checks passed.")
    return 0


def _project_field(name: str) -> str:
    """Read a ``name = "value"`` line from pyproject.toml (regex, so it works on 3.10 too)."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(rf'^{re.escape(name)}\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not match:
        raise SystemExit(f"error: no `{name}` field found in pyproject.toml")
    return match.group(1)


def _changelog_section(version: str) -> tuple[str, str]:
    """Return ``(date, body)`` of CHANGELOG.md's ``## [version] - YYYY-MM-DD`` section."""
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    header = re.search(
        rf"^## \[{re.escape(version)}\][^\n]*$", text, re.MULTILINE
    )  # "### [x.y.z]" entries in "Previous history" don't match: they start with three #'s.
    if not header:
        raise SystemExit(f"error: CHANGELOG.md has no '## [{version}]' section")
    date = re.search(r"- (\d{4}-\d{2}-\d{2})\s*$", header.group(0))
    if not date:
        raise SystemExit(f"error: CHANGELOG.md's [{version}] header has no release date")
    rest = text[header.end() :]
    next_header = re.search(r"^## ", rest, re.MULTILINE)
    body = (rest[: next_header.start()] if next_header else rest).strip()
    if not body:
        raise SystemExit(f"error: CHANGELOG.md's [{version}] section is empty")
    return date.group(1), body


def _check_citation(version: str, date: str) -> None:
    """Fail unless CITATION.cff's version and date-released match the release."""
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    for field, expected in (("version", version), ("date-released", date)):
        match = re.search(rf"^{field}:\s*['\"]?([^'\"\s]+)", text, re.MULTILINE)
        found = match.group(1) if match else None
        if found != expected:
            raise SystemExit(
                f"error: CITATION.cff {field} is {found!r}, expected {expected!r} "
                f"(from pyproject.toml / CHANGELOG.md)"
            )


def _unwrap(markdown: str) -> str:
    """Join hard-wrapped continuation lines into their paragraph or list item.

    CHANGELOG.md is wrapped at ~80 columns for editing, but GitHub renders single
    newlines in a Release body as line breaks, which would leave the notes ragged.
    Headings, list-item starts, blank lines, tables and fenced code are kept as-is.
    """
    out: list[str] = []
    in_fence = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        starts_block = (
            in_fence
            or not stripped
            or stripped.startswith(("#", "|", ">"))
            or re.match(r"([-*+]|\d+\.)\s", stripped) is not None
        )
        prev = out[-1].strip() if out else ""
        if starts_block or not prev or prev.startswith(("#", "|", "```")):
            out.append(line)
        else:
            # A line broken right after "/" (e.g. "`a`/\n`b`") rejoins without a space.
            joiner = "" if out[-1].rstrip().endswith("/") else " "
            out[-1] = f"{out[-1].rstrip()}{joiner}{stripped}"
    return "\n".join(out)


def release_notes(output: str | None = None) -> int:
    """Print (or write) GitHub Release notes for the current version, taken from CHANGELOG.md.

    Doubles as the release gate: fails unless CHANGELOG.md has a dated, non-empty section for
    pyproject.toml's version and CITATION.cff agrees on both the version and the date.
    """
    version = _project_field("version")
    date, body = _changelog_section(version)
    _check_citation(version, date)

    repo = _project_field("Repository").rstrip("/")
    notes = _unwrap(body) + f"\n\n**Full changelog:** {repo}/blob/v{version}/CHANGELOG.md\n"
    if output:
        Path(output).write_text(notes, encoding="utf-8")
        print(f"Wrote release notes for v{version} to {output}")
    else:
        # The changelog may contain non-ASCII text; don't crash on legacy console codepages.
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stdout.write(notes)
    return 0


def _write_outputs(outputs: dict[str, str]) -> None:
    """Print ``key=value`` lines, and append them to ``$GITHUB_OUTPUT`` when it's set."""
    lines = "".join(f"{key}={value}\n" for key, value in outputs.items())
    sys.stdout.write(lines)
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(lines)


# Paths that can't affect the package, its tests, or the lint/type checks. fnmatch's
# "*" also matches "/", so "*.md" covers Markdown anywhere. scripts/ isn't here: it's
# linted, and tests/test_dev_script.py tests it.
DOCS_ONLY_PATTERNS = (
    "docs/*",
    "*.md",
    "examples/*",
    "justfile",
    "CITATION.cff",
    ".readthedocs.yaml",
    ".github/dependabot.yml",
    ".github/workflows/publish.yml",
)

_NULL_SHA = "0" * 40


def ci_diff_base(event: str, pr_base: str = "", before: str = "") -> str | None:
    """The commit to diff against, or None when CI should just run everything.

    Pull requests diff against their base; pushes against the previous tip. A manual
    run, or the first push of a new branch (``before`` is all zeros), has no base.
    """
    if event == "pull_request":
        return pr_base or None
    if event == "push" and before and before != _NULL_SHA:
        return before
    return None


def ci_needed_jobs(files: list[str]) -> dict[str, bool]:
    """Which CI jobs a change set needs: anything outside DOCS_ONLY_PATTERNS needs both."""
    needed = any(
        not any(fnmatch.fnmatch(path, pattern) for pattern in DOCS_ONLY_PATTERNS) for path in files
    )
    return {"tests": needed, "quality": needed}


def ci_changes(event: str, pr_base: str, before: str, head: str) -> int:
    base = ci_diff_base(event, pr_base, before)
    files: list[str] | None = None
    if base is None:
        print("Manual run or new branch: running everything.")
    else:
        diff = subprocess.run(
            ["git", "diff", "--name-only", base, head],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if diff.returncode != 0:
            print(f"Could not diff against {base} (e.g. after a force-push): running everything.")
        else:
            files = [line for line in diff.stdout.splitlines() if line]
            print("Changed files:", *files, sep="\n  ")
    jobs = {"tests": True, "quality": True} if files is None else ci_needed_jobs(files)
    _write_outputs({key: str(value).lower() for key, value in jobs.items()})
    return 0


def release_decision(version: str, tags: list[str]) -> tuple[bool, str | None]:
    """Decide whether ``version`` is a new release, given the repository's tags.

    Returns ``(release, latest)``: ``release`` is False when tag ``v<version>`` already
    exists. Raises SystemExit when the version isn't higher than the latest released
    one. Versions compare by PEP 440 (``packaging``), so 0.4.0 ranks above 0.4.0rc1,
    which plain ``sort -V`` gets wrong. Tags that aren't PEP 440 versions are ignored.
    """
    from packaging.version import InvalidVersion, Version

    if f"v{version}" in tags:
        return False, version
    released = []
    for tag in tags:
        if not tag.startswith("v"):
            continue
        try:
            released.append(Version(tag[1:]))
        except InvalidVersion:
            continue
    latest = max(released) if released else None
    if latest is not None and Version(version) <= latest:
        raise SystemExit(f"error: version {version} is not higher than the latest release {latest}")
    return True, str(latest) if latest is not None else None


def release_check() -> int:
    version = _project_field("version")
    tags = subprocess.run(
        ["git", "tag", "-l", "v*"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    release, latest = release_decision(version, tags)
    if release:
        print(f"New version v{version} (previous: v{latest or 'none'}): releasing.")
    else:
        print(f"v{version} already exists: nothing to release.")
    _write_outputs({"version": version, "release": str(release).lower()})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/dev.py", description="banana-bench development helper."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_clean = sub.add_parser("clean", help="remove caches and build artifacts")
    p_clean.add_argument("-n", "--dry-run", action="store_true", help="list what would be removed")
    sub.add_parser("build", help="clean, build sdist and wheel, then run twine check")
    sub.add_parser("check", help="run ruff, ruff format --check, mypy and pytest (as CI)")
    p_notes = sub.add_parser(
        "release-notes", help="print the current version's CHANGELOG.md section as release notes"
    )
    p_notes.add_argument("-o", "--output", help="write to this file instead of stdout")
    sub.add_parser("version", help="print pyproject.toml's version")
    p_changes = sub.add_parser("ci-changes", help="CI: decide which jobs a push or PR needs")
    p_changes.add_argument("--event", required=True, help="github.event_name")
    p_changes.add_argument("--pr-base", default="", help="pull request base SHA")
    p_changes.add_argument("--before", default="", help="push event's previous SHA")
    p_changes.add_argument("--head", required=True, help="the commit under test")
    sub.add_parser(
        "release-check",
        help="CI: is pyproject.toml's version a new release? (fails if not higher than the last)",
    )

    args = parser.parse_args(argv)
    if args.command == "ci-changes":
        return ci_changes(args.event, args.pr_base, args.before, args.head)
    if args.command == "release-check":
        return release_check()
    if args.command == "clean":
        return clean(dry_run=args.dry_run)
    if args.command == "build":
        return build()
    if args.command == "release-notes":
        return release_notes(output=args.output)
    if args.command == "version":
        print(_project_field("version"))
        return 0
    return check()


if __name__ == "__main__":
    sys.exit(main())
