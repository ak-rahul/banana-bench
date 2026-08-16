# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`banana-bench` is the renamed continuation of `optimization-benchmarks`. Versioning
restarts at **0.1.0** for the `banana-bench` name itself — see "Previous history"
below for the `optimization-benchmarks` releases that came before it; the
functionality from that history carries forward unchanged.

## [Unreleased]

## [0.1.0] - 2026-08-16

First release under the `banana-bench` name (PyPI: `banana-bench`, import name:
`bananabench`, CLI command: `banana-bench`). Named after Rosenbrock's function —
part of this suite (`rosenbrock`) — famously nicknamed the "banana function" for
its curved valley shape.

Functionally this is everything `optimization-benchmarks` had accumulated through
its `0.4.0` release, plus the fixes and rewrite below.

### Added
- 55+ benchmark functions (`ackley`, `sphere`, `rosenbrock`, ...), each with
  `function`, `properties`, `bounds`, `default_dim`, `known_minimum`, and
  `optimal_point` registered in `BENCHMARK_SUITE`.
- CLI (`banana-bench --list` / `--info` / `--metadata` / `--function`), with
  single-point and CSV batch evaluation (`--input`/`--output`, `--jobs` for
  parallel batches).
- `BenchmarkRunner` / `quick_benchmark` for systematically testing an optimizer
  across the suite, with serial or parallel (`n_jobs`, via joblib) execution and
  reproducible per-run seeding.
- Function wrappers (`NoisyFunction`, `ShiftedFunction`, `RotatedFunction`,
  `BenchmarkWrapper`) for robustness testing.
- `cec` module (`CECFunction`, `CECCompositionFunction`) for CEC 2017/2020-style
  shifted/rotated/composite functions.
- `g_suite` module with the constrained G-suite (`g04`, `g06`, `g08`, `g11`).
- `gradients` module (`approximate_gradient`, `GradientWrapper`) for
  finite-difference gradients/Jacobians, for interop with gradient-based
  optimizers such as SciPy.
- Optional `visualization` module (2D/3D plots, convergence, trajectories,
  heatmaps, animated trajectories, multi-format export) and `plotly_viz` module
  for interactive surface/contour plots.
- Sphinx/Read the Docs documentation build.

### Fixed
- **`import bananabench` crashed entirely without matplotlib installed**,
  contradicting the documented "core functionality depends only on NumPy"
  guarantee. `visualization.py`'s functions are annotated `-> plt.Figure`;
  without `from __future__ import annotations`, Python evaluates that
  annotation at import time, which raised `NameError` (not the `ImportError`
  the surrounding try/except was written to catch) whenever matplotlib was
  absent. Fixed by deferring annotation evaluation, and by having
  `bananabench.__visualization_available__` read `visualization.MATPLOTLIB_AVAILABLE`
  directly instead of inferring availability from whether the import raised
  (which, once the crash was fixed, always succeeded, permanently reporting
  `True`). Verified in clean venvs with and without matplotlib installed.
- **Test suite rewritten**: tests are now organized one file per `bananabench`
  module (`test_functions.py`, `test_metadata.py`, `test_wrappers.py`,
  `test_cec.py`, `test_g_suite.py`, `test_gradients.py`, `test_plotly_viz.py`,
  ...) instead of being split across `test_new_modules.py` /
  `test_v030_features.py` / `test_v040_features.py` by the version a feature was
  added in. Global-minimum tests are now driven directly from `BENCHMARK_SUITE`
  so every registered function is checked automatically, rather than a
  hand-picked subset.
- **`branin` metadata**: third optimal point corrected from `(9.425, 2.425)` to
  `(9.42478, 2.475)` — the old value didn't actually reach the claimed minimum.
- **`chichinadze` metadata**: `known_minimum` corrected from `-43.3159` to
  `-43.868648` to match the function's actual value at its optimal point.
- **`step` metadata**: `known_minimum`/`optimal_point` corrected from `0.0`/`0.5`
  to `2.5`/`0.0` (for the default 10-dimensional case) — the shipped formula,
  `sum((floor(x) + 0.5)**2)`, can never reach 0 since `floor` is integer-valued.
- **`holzman1`**: identified as returning ~13.74, not the claimed 0, at its
  documented optimum, and producing NaN/very large values elsewhere inside its
  own documented domain — flagged with an `xfail(strict=True)` test rather than
  silently left unverified; likely an implementation bug, not yet fixed.

### Changed
- Package metadata (`pyproject.toml` description/classifiers, `CITATION.cff`)
  no longer describes the package as archived/renamed-away — that language was
  left over from the `optimization-benchmarks` stub and had been carried into
  `banana-bench`'s own metadata by mistake.
- Removed hardcoded version numbers from module docstrings
  (`benchmarking.py`, `utils.py`, `cli.py`) that would otherwise go stale on
  every future release.
- **Import name is now `bananabench`** (was `banana_bench`) — mirrors the
  scikit-learn/`sklearn`, beautifulsoup4/`bs4` pattern. PyPI distribution name
  (`banana-bench`) and CLI command (`banana-bench`) are unchanged.
- **`__version__` is now read from installed package metadata**
  (`importlib.metadata.version("banana-bench")`) instead of being hardcoded in
  `__init__.py`, so there's a single source of truth (`pyproject.toml`) instead
  of two that can drift.
- **Tooling: `black` + `isort` + `flake8` replaced with `ruff`** (`ruff format`
  + `ruff check`) — one fast tool instead of three, matching current practice
  in numpy/pandas/fastapi. CI (`quality.yml`), pre-commit config, and
  `CONTRIBUTING.md` updated accordingly.
- License declared as an SPDX expression (`license = "MIT"`) instead of
  `{text = "MIT"}`; dropped the now-redundant `License :: OSI Approved :: MIT
  License` classifier.
- Added `.github/dependabot.yml` (pip + GitHub Actions) and `workflow_dispatch`
  triggers on `test.yml`/`quality.yml` for manual re-runs.

### Removed
- The no-op `setup.py` shim — unnecessary with `setuptools>=61` + `pyproject.toml`.
- `scripts/` (`add_properties.py`, `create_notebooks.py`, `verify_fixes.py`):
  one-off historical migration/generation/audit scripts, already applied or
  superseded by the real test suite. `add_properties.py` would have corrupted
  `metadata.py` if re-run (duplicate `"properties"` keys); `create_notebooks.py`
  had drifted out of sync with the hand-maintained notebooks it originally
  generated (it referenced a function, `plot_heatmap`, that doesn't exist —
  the real name is `plot_search_heatmap`).

### Fixed (packaging & docs)
- **`MANIFEST.in` was silently broken**: it still referenced the pre-rename
  `banana_bench/` directory, the deleted `setup.py`, and the deleted `scripts/`
  — a source distribution built from it would have shipped without the actual
  package code. Corrected to the real `bananabench/` layout.
- Filled in the unfinished `[your-email@example.com]` placeholder in
  `CODE_OF_CONDUCT.md`.
- `docs/BENCHMARK_FUNCTIONS.md` had drifted from `BENCHMARK_SUITE` (stale
  `branin`/`chichinadze`/`step` values from before this release's metadata
  fixes, `langerman`'s pre-fix `default_dim=3`, and 5 functions — `watson`,
  `xor`, `zimmerman`, `lennard_jones`, `freudenstein_roth` — missing entirely).
  Replaced hand-maintenance with `docs/generate_benchmark_functions.py`, a
  reusable generator that derives the whole file from `BENCHMARK_SUITE` and
  each function's docstring.
- `docs/USER_GUIDE.md`, `API_REFERENCE.md`, `VISUALIZATION.md` had broken code
  examples: `info['bounds'] * info['default_dim']` (wrong for per-dimension
  bounds like `branin`'s — use `get_bounds()`), a duplicated section heading,
  a `BenchmarkRunner` signature missing `n_jobs`, a `Type Hints` snippet
  missing its `List` import, `plot_convergence(..., true_minimum=...)` (the
  real parameter is `known_minimum`), and an undefined `visited_points`
  variable. All examples in these docs are now verified against the actual
  function signatures.
- Sphinx docs now build with zero warnings (`python -m sphinx -b html docs
  docs/_build/html -W`): added `myst_heading_anchors` so the `[name](#name)`
  links throughout `BENCHMARK_FUNCTIONS.md` resolve, and created the
  `docs/_static/` directory `conf.py` already referenced but that didn't exist.
- `docs/index.md` had a leftover `# Optimization Benchmarks Documentation`
  title from before the rename; now a proper landing page (install +
  quickstart + toctree).

### Examples rewritten
- Replaced the old `examples/` (4 scripts + 2 notebooks, several with dead
  code, an `as ob` import alias left over from the pre-rename package name,
  and PNG files dumped into the current directory as a side effect) with 6
  focused, numbered scripts, each independently runnable and verified to
  actually produce correct output: `01_quickstart`, `02_custom_optimizer_benchmarking`,
  `03_robustness_wrappers`, `04_gradient_based_optimization` (SciPy L-BFGS-B via
  `GradientWrapper`), `05_constrained_optimization` (SciPy SLSQP on the
  `g_suite` functions), `06_visualization_gallery`. The wrappers, gradients,
  and g_suite modules previously had no example coverage at all.
- Added `bananabench/__main__.py` so `python -m bananabench` works as an
  alternative to the `banana-bench` entry point (matches `python -m black` /
  `python -m pytest` / `python -m mypy`).

## Previous history (as `optimization-benchmarks`)

The entries below predate the rename and use `optimization-benchmarks`'
own version numbers, which are unrelated to `banana-bench`'s versioning above.

### [0.4.0] - 2026-01-29

#### Renamed
- **Project renamed from `optimization-benchmarks` to `banana-bench`.** Since PyPI
  does not support renaming a project in place, `banana-bench` was published as a
  new PyPI project.

#### Added
- **New Function Wrappers**: Added `BenchmarkWrapper`, `NoisyFunction`, `ShiftedFunction`, and `RotatedFunction` for advanced testing scenarios.
- **Parallel Benchmarking**: Added `n_jobs` parameter to `BenchmarkRunner` and CLI (`--jobs`) for parallel execution.
- **Reproducible Parallel Runs**: Each run in `run_suite` (serial or parallel) now gets a distinct, seed-derived `run_seed`, so parallel benchmarking results are reproducible across processes.
- **Visualization**: New `animate_trajectory_2d` function for optimizing visualization.
- **Interactive Visualization**: New `plotly_viz` module (`plot_surface_interactive`, `plot_contour_interactive`) for interactive 3D/contour plots.
- **CEC-Style Framework**: New `cec` module (`CECFunction`, `CECCompositionFunction`) for building shifted/rotated/composite benchmark functions in the style of the CEC 2017/2020 suites.
- **Constrained G-Suite**: New `g_suite` module with classic constrained benchmark problems `g04`, `g06`, `g08`, `g11` from the CEC 2006 constrained optimization session.
- **Gradient Estimation**: New `gradients` module (`approximate_gradient`, `GradientWrapper`) providing finite-difference gradients/Jacobians for black-box functions, for use with gradient-based optimizers such as SciPy.
- **Function Metadata**: Added a `properties` field to every entry in `BENCHMARK_SUITE`, tagging continuity, differentiability, separability, scalability, and modality/convexity where established in the literature.
- **Typing & strictness**: Added type hints to `metadata.py`, explicit `__all__` exports in `functions.py`.
- **CI/CD**: Added PyPI publishing workflow, coverage reporting, and badges to README.
- **Progress Tracking**: Enhanced CLI with parallel batch evaluation support.

#### Changed
- **Documentation Overhaul**: Major updates to `USER_GUIDE.md`; expanded `API_REFERENCE.md`; created `BENCHMARK_FUNCTIONS.md`; updated `CONTRIBUTING.md`; added Sphinx/Read the Docs documentation build.
- **Dependencies**: Removed Python 3.8 support (min version 3.9). Unified dev dependencies. Added optional `interactive` extra for Plotly.
- **CI/CD Hardening**: Updated GitHub Actions to non-deprecated versions, added explicit least-privilege `permissions`, pip caching, and concurrency cancellation; the publish workflow now runs the full test suite and validates package metadata (`twine check`) before uploading to PyPI.

#### Fixed
- **Reproducibility**: Fixed bug in `NoisyFunction` RNG handling.
- **`langerman` dimensionality**: Fixed `default_dim` (was 3, now 10) to match its fixed 10-dimensional coefficient table.
- **Missing exports**: `watson`, `xor`, `zimmerman`, `lennard_jones`, `fraudenstein_roth`, and `freudenstein_roth` are now included in `bananabench.__all__`.
- **Code Quality**: Cleaned up AI-generated artifacts, resolved all Mypy errors, and applied Black/Isort formatting globally.

### [0.3.0] - 2025-12-11

#### Added
- **Progress Bars**: tqdm integration for visual progress tracking during benchmarking.
- **Heatmap Visualization**: New `plot_search_heatmap()` function.
- **Multi-Format Export**: `save_plot()` utility supporting PNG, SVG, PDF, EPS formats.
- **Enhanced Color Schemes**: `COLORMAPS` constant with 9 colormap choices.
- **Batch Plotting**: New `batch_plot_functions()` for generating multiple plots at once.

#### Changed
- Updated `tqdm>=4.65.0` as core dependency.
- Improved plot aesthetics with better default settings.
- Added proper academic citations to README.

#### Fixed
- Minor bug fixes in visualization module.
- Improved error handling in export functions.

### [0.2.0] - 2025-12-06

#### Added
- New `utils` module with helper functions.
- New `visualization` module (requires matplotlib).
- New `benchmarking` module with systematic testing tools.
- Examples directory with complete usage demonstrations.
- Optional dependency group `[viz]` for visualization features.
- Comprehensive test suite for new modules.

#### Changed
- Updated `__init__.py` to export new utilities.
- Improved package metadata in `pyproject.toml`.

#### Dependencies
- Added optional `matplotlib>=3.3.0` for visualization features.
- Core functionality remains dependency-free (only numpy required).

### [0.1.1] - 2025-10-17

#### Added
- Added `BENCHMARK_SUITE` metadata dictionary.
- New helper functions: `get_function_info()`, `get_all_functions()`, `get_bounds()`, `get_function_list()`.
- Metadata includes bounds, dimensions, known minima, and optimal points for all 55 functions.

### [0.1.0] - 2025-10-16

#### Added
- Initial release with 55 benchmark functions.
- Command-line interface (`banana-bench`).
- Comprehensive test suite.
- Full academic citations.
