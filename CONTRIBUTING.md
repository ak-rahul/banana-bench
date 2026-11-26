# **Contributing to banana-bench**

Thank you for your interest in contributing! We welcome contributions of all kinds.

## Quick Start

1. Fork and clone the repository
```
git clone https://github.com/ak-rahul/banana-bench.git
cd banana-bench
```

2. Install in development mode
```
pip install -e ".[dev]"
```

3. Create a branch
```
git checkout -b feature/your-feature-name
```

---

## Adding a New Function

Add your function to `bananabench/functions.py`:
```
def your_function(x: np.ndarray) -> float:
"""
Your Function Name.
Domain: |x_i| ≤ 10.
Dimension: n.
Global minimum: f(x*) = value at x = location.

References:
 Author, Title, Year.
"""
x = np.asarray(x, dtype=float)
# Your implementation
return result
```

Add it to `__init__.py` exports and write tests in `tests/test_functions.py`.

## Testing

We use `pytest` for testing. To run the tests:

1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Run all tests:
   ```bash
   pytest
   ```

3. Run with coverage report:
   ```bash
   pytest --cov=bananabench
   ```

## Development Scripts

Common development chores are wrapped in a [`justfile`](justfile). The
[`just`](https://github.com/casey/just) command runner is installed with the
dev extra (`pip install -e ".[dev]"`), and works in PowerShell, cmd, Git Bash,
macOS, and Linux. Run `just` on its own to list every recipe.

```bash
just clean            # remove caches (__pycache__, .pytest_cache, .mypy_cache, .ruff_cache)
                      # and build artifacts (build/, dist/, *.egg-info, coverage files, docs/_build)
just clean --dry-run  # list what would be removed without deleting anything
just fmt              # auto-format and apply safe lint fixes
just test             # run pytest; extra arguments pass through, e.g. `just test -k ackley`
just check            # ruff, ruff format --check, mypy, and pytest, as CI runs them
just build            # clean, build the sdist and wheel, then run twine check
just docs             # build the Sphinx docs into docs/_build/html (needs the docs extra)
just release-notes    # check CHANGELOG/CITATION are release-ready, preview the release notes
```

The recipes use the project's `.venv` interpreter when one exists, so they
work without activating it. The logic itself lives in `scripts/dev.py`
(standard library only), so `python scripts/dev.py clean|build|check` works
too if you don't have `just`. `clean` never touches `.git` or any virtual
environment.

## Releasing (maintainers)

Releases are fully automatic. Merging a version bump into `main` publishes
it. Don't create tags or GitHub Releases by hand.

1. In your PR, bump `version` in `pyproject.toml` and `CITATION.cff`, move the
   `[Unreleased]` entries under a new `## [X.Y.Z] - YYYY-MM-DD` heading in
   `CHANGELOG.md`, and set the same date as `date-released` in
   `CITATION.cff`.
2. Run `just release-notes` to check all of the above and preview the notes.
   Run `just build` to check the package.
3. Merge into `main`.

Any push to `main` that changes `pyproject.toml` runs
`.github/workflows/publish.yml`. If the version has no `vX.Y.Z` tag yet, the
workflow:

1. Checks that the new version is higher than every existing release, that
   `CHANGELOG.md` has a dated, non-empty section for it, and that
   `CITATION.cff` matches its version and date. If any check fails, the
   workflow stops before anything is uploaded.
2. Runs the test suite.
3. Builds the package and publishes it to PyPI.
4. Creates the `vX.Y.Z` tag and the GitHub Release. The release notes are the
   changelog section, and the sdist and wheel are attached. A version such
   as `0.4.0rc1` is marked as a pre-release.

If the version didn't change, every job after the first is skipped. If a run
fails partway, fix the cause and re-run it from the Actions tab. "Run
workflow" also works, since PyPI uploads that already succeeded are skipped.

## Code Style

- Run `ruff format bananabench tests` and `ruff check bananabench tests` before committing
- Use type hints
- Follow NumPy docstring style

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Push your branch and create a PR
4. Respond to review comments

## Questions?

Open an issue or contact the maintainers.

## License

By contributing, you agree your contributions will be licensed under the MIT License.

---

**Thank you for making banana-bench better!** 🚀
