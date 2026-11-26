"""
Development helper for banana-bench.

Cross-platform (Windows, macOS, Linux) and standard-library only, so it runs
the same everywhere without needing ``make`` or a shell.

Usage::

    python scripts/dev.py clean            # remove caches and build artifacts
    python scripts/dev.py clean --dry-run  # list what would be removed
    python scripts/dev.py build            # clean, build sdist + wheel, twine check
    python scripts/dev.py check            # lint, format check, type check, tests (as CI)

``clean`` never touches the ``.git`` directory or any virtual environment
(any directory containing ``pyvenv.cfg``, plus the usual ``.venv``/``venv``
names), so it is safe to run inside an activated environment.
"""

from __future__ import annotations

import argparse
import glob
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
    """Run the same gates as CI's quality.yml and test.yml, stopping at the first failure."""
    steps = [
        (sys.executable, "-m", "ruff", "check", "bananabench", "tests"),
        (sys.executable, "-m", "ruff", "format", "--check", "bananabench", "tests"),
        (sys.executable, "-m", "mypy", "bananabench"),
        (sys.executable, "-m", "pytest", "tests"),
    ]
    for step in steps:
        rc = _run(*step)
        if rc != 0:
            return rc
    print("\nAll checks passed.")
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

    args = parser.parse_args(argv)
    if args.command == "clean":
        return clean(dry_run=args.dry_run)
    if args.command == "build":
        return build()
    return check()


if __name__ == "__main__":
    sys.exit(main())
