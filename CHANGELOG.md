# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`banana-bench` is the renamed continuation of `optimization-benchmarks`. Versioning
restarts at **0.1.0** for the `banana-bench` name itself — see "Previous history"
below for the `optimization-benchmarks` releases that came before it; the
functionality from that history carries forward unchanged.

## [Unreleased]

## [0.4.0] - 2026-12-26

### Added
- New `bananabench.tracking` module. `TrackedFunction(func, max_evals=None,
  record_trace=True)` wraps any objective and counts evaluations. It records the
  best value and point seen so far, and an anytime best-so-far trace: the first
  evaluation plus every strict improvement, which is what data profiles and
  runtime ECDFs need. With `max_evals` it raises `BudgetExhausted` instead of
  performing evaluation `max_evals + 1`. It is purely observational (returns
  exactly `func(x)`) and composes with `wrappers`/`cec`.
- `BenchmarkRunner(max_evals=..., record_trace=False, verify=True)`. Algorithms
  now receive a `TrackedFunction`, so every run reports `n_evals` and
  `max_evals`.
  - `max_evals` is a fixed count or a function of the dimension, e.g.
    `lambda d: 10_000 * d`. When an algorithm hits the budget, the runner
    catches `BudgetExhausted` and scores the best point evaluated so far
    (`status="budget_exhausted"`, still a success), so algorithms don't need
    to handle budgets themselves.
  - `record_trace=True` stores each run's `trace` and `first_cost`. JSON
    output keeps them; CSV leaves them out.
  - `quick_benchmark` accepts `max_evals` and `verify`. The `run_suite`
    verbose table, `print_summary` and `compare_algorithms` show evaluation
    counts.
- New `bananabench.analysis` module, which compares solvers the way the
  optimization literature does, across a set of problems rather than by one
  aggregate number. Every definition follows its source paper.
  - `performance_profile` (Dolan & Moré 2002) and `data_profile` (Moré & Wild
    2009, convergence test `f(x0) - f(x) >= (1 - tau)(f(x0) - f_L)`, budgets in
    simplex gradients).
  - `expected_running_time` (ERT, COCO's aRT; Auger & Hansen 2005) and
    `runtime_ecdf` (COCO). `COCO_TARGETS` holds COCO's 51 default targets from
    1e2 to 1e-8.
  - `evaluations_to_target`.
  - Adapters `histories_from_runners`, `ert_from_runners` and
    `performance_profile_from_runners` build these inputs from runners that used
    `record_trace=True`.
  - The pure functions take plain mappings, so they also work on data from
    outside `bananabench`.
- `analysis.statistical_comparison(runners, metric="error", alpha=0.05)`
  implements Demšar's (2006) procedure for comparing algorithms over multiple
  problems:
  - average ranks;
  - the Friedman test with the Iman–Davenport correction;
  - the Nemenyi critical difference;
  - pairwise Wilcoxon signed-rank tests with Holm's (1979) correction.
- `analysis.center_bias_test(algorithm, ...)`: Kudela's (2022) test for
  center bias. Most classic functions have their optimum at the center of the
  box, so an algorithm that pulls its search toward the center looks far better
  than it is. Kudela's follow-up study found 47 of 90 published methods doing
  this. The test runs the algorithm on each function and on a copy shifted by
  10% of the range, and flags the algorithm when the geometric mean of the
  shifted-to-unshifted error ratios exceeds 10. It follows Kudela's protocol and
  defaults to the 10 functions of his test set that the suite has
  (`analysis.CENTER_BIAS_FUNCTIONS`). Runs are scored on what they actually
  evaluated. See `examples/14_center_bias.py`.

  It returns structured results plus a text `report`. It requires SciPy,
  available through the new `stats` extra (`pip install 'banana-bench[stats]'`).
- `visualization.plot_performance_profile`, `plot_data_profile` and
  `plot_runtime_ecdf`: step plots in a colorblind-safe palette (Okabe & Ito).
- DTLZ1–DTLZ7 (Deb, Thiele, Laumanns & Zitzler 2005), the standard scalable
  many-objective suite, in `multiobjective`.
  - Each takes `n_objectives` (M, default 3) and uses the paper's
    `n = M + k - 1` defaults (`k` = 5, 10, 10, 10, 10, 10, 20).
  - Every formula was cross-checked against two independent implementations:
    pymoo (agreement to 2.4e-15 relative error at random points for M = 2–4)
    and Platypus.
  - New registries `DTLZ_SUITE` and `MO_SUITE` (ZDT + DTLZ), with helpers
    `get_mo_function`, `get_mo_default_dim` and `get_mo_n_objectives`.
  - `get_pareto_front(name, n_points, n_objectives)` returns the true front
    for each.
- `multiobjective.das_dennis(n_objectives, n_partitions)`: uniformly spaced
  reference points on the simplex (Das & Dennis 1998).
- `MOBenchmarkRunner(n_objectives=..., max_evals=...)` runs DTLZ problems and
  counts evaluations. A run that hits its budget is scored on the
  non-dominated set of everything it evaluated. Hypervolume is now reported
  for every problem, not only bi-objective ones.
- `MOBenchmarkRunner.save_results(filepath, format)` (CSV or JSON) and
  `MOBenchmarkRunner.get_summary_stats()`, matching `BenchmarkRunner`.
  Multi-objective results previously couldn't be saved without hand-written
  code. The summary has no mean hypervolume, because each problem uses its own
  reference point and the values aren't comparable across problems.
- CLI: `--suite multiobjective` covers DTLZ, and a new `--objectives M` flag
  sets the number of objectives, e.g.
  `banana-bench --function dtlz2 --suite multiobjective --objectives 3 --values ...`.
- Examples:
  - `examples/11_budgeted_benchmarking.py`: equal evaluation budgets, with
    verification catching a misreported result.
  - `examples/12_performance_profiles.py`: ERT, performance and data profiles,
    a runtime ECDF and a statistical comparison.
  - `examples/13_many_objective_dtlz.py`: DTLZ with three and five objectives,
    exact and Monte Carlo hypervolume.
  - `examples/14_center_bias.py`: two hill climbers that differ in one line,
    one of them pulled toward the center, and the test telling them apart.

  The user guide gains sections on evaluation budgets, profiles, statistical
  comparison, center bias and many-objective problems, and
  `docs/BENCHMARK_FUNCTIONS.md` a DTLZ section.
- Documentation site:
  - an API page generated from the docstrings, so it can't drift from the code;
  - an Examples page listing all 14 scripts;
  - the changelog.

  The README and docs quick starts use a real random search under an
  evaluation budget. The old example returned the origin without evaluating
  anything, and scored perfectly only because sphere and ackley have their
  optimum there.

### Changed
- `BenchmarkRunner` now verifies results by default. It re-evaluates the
  benchmark function at the reported `best_x` (not counted against the budget)
  and scores that value. The algorithm's own figure is kept as
  `reported_cost`, and `cost_mismatch=True` flags a report that disagrees.
  Previously a buggy or dishonest algorithm could report any `best_cost`, and
  its `error` was computed from it unchecked. Numbers change only for
  algorithms whose reported cost was wrong; `verify=False` restores the old
  behavior.
- `MOBenchmarkRunner` verifies too (`verify=True` by default). It re-evaluates
  the problem at every returned point in `X` (not counted against the budget)
  and scores those objectives instead of the reported `F`. `n_mismatched`
  counts the points whose report disagreed.
- `BenchmarkRunner` validates the reported result. A `best_x` without exactly
  `dim` entries or with NaN/inf, or a `best_cost` that isn't a number, is now
  an `invalid_result`. A wrong-length `best_x` was previously scored as a
  success on scalable functions, and a non-numeric `best_cost` raised out of
  `run_suite` and lost every other function's result. A `best_x` outside the
  search bounds is still scored but flagged `out_of_bounds=True`, counted in
  `n_out_of_bounds`, and warned about in `print_summary`.
- `multiobjective.hypervolume` supports any number of objectives instead of
  raising `NotImplementedError` above two. It is exact by default: a sweep for
  2 objectives, slicing for 3, and the WFG algorithm (While, Bradstreet & Barone
  2012) for more. `method="monte_carlo"` gives a seeded estimate for large
  fronts, and `method="auto"` (default) switches to it beyond 100 points in 4
  or more objectives. Results were cross-checked against pymoo's independent
  implementation for 2–5 objectives. Two-objective results are unchanged.
- `get_mo_function_list()` and `MOBenchmarkRunner.run_suite()` (with no
  `functions`) now include the DTLZ problems as well as ZDT.
- `import bananabench` is faster (about 0.5 s instead of 1 s in testing).
  `bananabench.visualization` now imports matplotlib the first time a plot is
  drawn, not when the package is imported. `MATPLOTLIB_AVAILABLE` still reports
  whether matplotlib is installed, and the missing-matplotlib warning and
  `ImportError` are unchanged. The CLI starts faster for the same reason.
- CI: `test.yml` and `quality.yml` are merged into one `ci.yml` workflow,
  ending in a single `✅ CI passed` job for branch protection to require. A
  first `changes` job skips the test matrix and lint for docs-only changes
  through job-level conditions instead of `paths-ignore`. A workflow skipped by
  a path filter reports no status, which left required checks waiting forever;
  a skipped job reports success.
- CI runs on pushes to `main`, on pull requests, and manually, instead of on
  every push to every branch. A pull request branch previously ran everything
  twice, once for the push and once for the pull request.
- CI logic moved out of the workflow files. The changed-files classification
  and the release version check were inline shell and Python in the YAML, and
  nothing linted or tested them. They're now `scripts/dev.py ci-changes` and
  `scripts/dev.py release-check`, with unit tests in
  `tests/test_dev_script.py`, so `scripts/` changes now run the tests too. A
  local composite action (`.github/actions/setup-python-env`) replaces the
  Python setup and install steps the five jobs each repeated.
- The release workflow only releases from `main`. Its manual "Run workflow"
  trigger accepted any branch, so running it on a feature branch with an
  untagged version would have published that branch to PyPI.

### Removed
- Python 3.10 support. `requires-python` is now `>=3.11`. Python 3.10 reached
  end-of-life in October 2026, and dropping it follows the SPEC 0 policy NumPy
  and SciPy use. Python 3.15 is added to the classifiers and the CI matrix,
  which now tests 3.11–3.15 on Linux, Windows and macOS.

### Fixed
- `holzman1` returned wrong values, and NaN for points inside its own
  documented domain, so its documented minimum `f(50, 25, 1.5) = 0` was never
  reached. It's the Gulf Research and Development function (Moré, Garbow &
  Hillstrom 1981, problem 11), and the earlier transcription had four errors:
  unsquared residuals, a missing minus sign and absolute value in the
  exponent, and a `0.1*i` target instead of `i/100`. It now uses the source
  formula with 99 residuals, cross-checked against SciPy's global-optimization
  benchmark `Gulf` and Burkardt's TEST_OPT. Values differ from 0.3.0 and
  earlier. Those values were wrong, so any results recorded against them
  should be regenerated. It also raises `ValueError` for input that isn't
  3-dimensional, like every other fixed-dimension function.
- `BenchmarkRunner.run_suite`'s verbose table, `print_summary`, and
  `MOBenchmarkRunner`'s verbose table still printed tiny values with plain
  `:12.6f`, so `gear`'s ~2.7e-12 known minimum showed as `0.000000`. They now
  share a single formatter with every CLI `--list` table and
  `metadata.get_function_list()`, which previously each kept their own copy.
- `step2` and `plateau` documented a global minimum of 30 at `x = 0`, but
  `floor` goes negative across their domain: both reach 0 on the corner
  plateau `x_i in [-5.12, -5)`, where every `floor(x_i) = -6`. This is De
  Jong's F3 step function, and its reference implementation (Baeck's
  `f_3.c`) gives the same minimum. The function values are unchanged.
  `known_minimum` is now 0 and `optimal_point` is `-5.06` in every coordinate,
  so any `error` recorded against the old minimum of 30 was off by 30. A new
  test samples every function's domain, including the box corners, and fails
  if any point undercuts its documented minimum. It's what found these two.
- `save_results(format="json")` wrote `NaN` and `Infinity` for non-finite
  values, which aren't valid JSON, so strict parsers (JavaScript, `jq`, most
  other languages) rejected the whole file. They're now written as `null`.
- `MOBenchmarkRunner.run_single` raised `ValueError` when an algorithm
  returned `F` with the wrong number of objectives, taking down the whole
  `run_suite`. It's now an `invalid_result` for that function, as are
  non-finite objective values, which were previously scored with `igd=nan`.

## [0.3.0] - 2026-11-26

### Added
- `examples/09_cec_style_composition.py`: demonstrates `cec.CECFunction`/
  `CECCompositionFunction` wrapping and composing a user-defined function (not
  part of `bananabench`), since the framework was already general enough to do
  this but it wasn't shown anywhere. `docs/API_REFERENCE.md`'s `cec` section
  reworded from "wraps any benchmark function" to make clear `base_func` isn't
  restricted to the bundled suite.
- New `benchmarking.compare_algorithms(runners)`: builds a side-by-side text comparison
  table across multiple already-benchmarked `BenchmarkRunner` instances, ranked by mean
  error ascending (an algorithm with zero successful runs sorts last rather than
  raising). Replaces the loop-and-save-separate-CSVs pattern previously shown in
  `USER_GUIDE.md`'s "Comparing Multiple Algorithms" section with one readable table.
- New `metadata.query_functions(separable=None, scalable=None, differentiable=None,
  continuous=None, unimodal=None, convex=None)`: finds function names by combining
  `BENCHMARK_SUITE`'s `properties` tags declaratively (e.g.
  `query_functions(separable=False, unimodal=False, scalable=True)` for "every
  non-separable multimodal scalable function") instead of hand-filtering the
  registry. Each parameter is `True`/`False`/`None`; a function carrying neither tag
  on a requested axis (modality/convexity aren't always asserted) matches neither
  side. `scalable=False` means the tag's absence, since there's no explicit
  "non-scalable" tag in the schema. Scoped to the scalar suite only.
- New `engineering` module: the classic engineering design suite --
  `spring_design` (Belegundu 1982), `pressure_vessel` (Kannan & Kramer 1994),
  `welded_beam` (Ragsdell & Phillips 1976), and `gear_train` (Sandgren 1990).
  Follows `g_suite`'s `(objective, inequality_violations,
  equality_violations)` convention; unlike the CEC2006 G-suite (one shared
  source report), each of these four problems comes from its own original
  paper, so every formula was cross-checked against at least two independent
  transcriptions before landing. `pressure_vessel`'s documented optimum is a
  proven global optimum (Yang, Huyck, Karamanoglu & Khan 2013's Lagrange
  multiplier analysis), not merely a literature best-known value.
  `gear_train` is a mixed-integer problem in its original formulation (four
  gear tooth counts) with no constraint beyond its box bounds; this module
  evaluates it at whatever real-valued `x` is passed in, consistent with
  every other function in the package, rather than rounding internally.
  New `engineering.ENGINEERING_SUITE` registry (`function`, `dim`,
  `n_inequality`, `n_equality`, `known_minimum`), same shape as
  `g_suite.G_SUITE` and for the same reason (bounds vary too much
  per-problem to fit one schema). `get_engineering_function_list()` is the
  accessor.
- CLI: `--suite` gains an `engineering` choice (`spring_design`,
  `pressure_vessel`, `welded_beam`, `gear_train`), sharing `constrained`'s
  result shape since both follow the same convention.
- `examples/08_engineering_design.py`: `spring_design` and `welded_beam`
  wired into `scipy.optimize.minimize`'s SLSQP solver, in the style of
  `05_constrained_optimization.py`.
- `docs/generate_benchmark_functions.py` now also generates an "Engineering
  Design Suite" section in `docs/BENCHMARK_FUNCTIONS.md` from
  `ENGINEERING_SUITE`, alongside the existing `BENCHMARK_SUITE`/`G_SUITE`/
  `ZDT_SUITE`-derived sections.
- CLI: a banner (bold-yellow `ansi_shadow`-font block letters) now prints on
  bare invocation and `--help` only -- never on `--list`/`--info`/
  `--metadata`/`--function`'s machine-consumable output paths, since those
  are meant to be piped/parsed. List headers/footers and status text now use
  ANSI color, gated behind both `sys.stdout.isatty()` and the `NO_COLOR`
  environment variable convention (no-color.org), matching how `ruff`/`uv`
  stay well-behaved in CI logs and redirected output. Zero new dependencies:
  the banner string was generated once via `pyfiglet` as an authoring tool
  and hardcoded, not shipped as a runtime dependency. Also added a
  `--version` flag.
- New `bananabench.constraints` module: `feasibility_compare(a, b, tol=1e-6)`
  ranks two `(objective, inequality_violations, equality_violations)`
  results -- the return shape of every `g_suite`/`engineering` function --
  using Deb (2000)'s standard feasibility rule (feasible beats infeasible,
  then lower objective among feasible pairs, then smaller total violation
  among infeasible pairs). `BenchmarkRunner` assumes a scalar cost and
  doesn't fit this tuple shape, so this closes a real gap: comparing
  constrained-suite results previously meant hand-rolling the rule.
  `is_feasible(...)` and `constraint_violation(...)` are the two building
  blocks it's implemented on top of, also exported for standalone use.
- `examples/10_optuna_integration.py`: wraps a `BENCHMARK_SUITE` function as
  an Optuna `objective(trial) -> float` using `trial.suggest_float` per
  dimension, built from the function's own `bounds`/metadata rather than a
  hardcoded search space. Requires Optuna (`pip install optuna`), now
  available as part of the `dev`/`all` extras; not a core dependency.
- New `SECURITY.md`: private vulnerability-reporting instructions (GitHub
  private advisories or direct email), standard practice for an actively
  maintained open-source project.

### Changed
- `MOBenchmarkRunner.run_single` now isolates an algorithm call's exception
  into a failure result dict (`status`, `success`, `error_message`) instead
  of letting it propagate and abort the rest of `run_suite`'s batch, matching
  `BenchmarkRunner.run_single`'s existing behavior. A successful result now
  also carries `status="success"`/`success=True`, so callers can distinguish
  the two without a `try`/`except` of their own. `run_suite`'s verbose
  printer was updated to render a `FAILED` row for these instead of raising
  `KeyError` on the now-absent `igd` key.
- `README.md`'s feature list now covers the constrained (G-suite,
  engineering), multi-objective (ZDT), CEC-style composition, and CLI
  features, and lists every optional install extra rather than only `viz`.
  Its documentation links are now absolute GitHub URLs, so they also work
  on the PyPI project page, where relative `docs/...` links returned 404.

### Removed
- Python 3.9 support. `requires-python` is now `>=3.10`. Python 3.9 reached
  end-of-life in October 2025, and GitHub's current Ubuntu runner image no
  longer ships a 3.9 build. CI now tests Python 3.10 through 3.14, and 3.13
  and 3.14 are added to the package classifiers.

### Fixed
- The Sphinx docs (`docs/conf.py`) hardcoded `release = '0.1.0'`, so the
  published documentation showed the wrong version. It now reads the version
  from the installed package metadata, keeping `pyproject.toml` as the single
  source of truth.
- `BenchmarkRunner.run_suite(verbose=True)` (including the README quick
  start) and `visualization.batch_plot_functions` crashed
  with `UnicodeEncodeError` whenever stdout used a legacy codepage, e.g.
  Windows with output redirected to a file or pipe, because their progress
  lines printed a non-ASCII check mark. They now print ASCII only (`*` marks
  a run whose mean error is below 1.0).
- `utils.normalize_bounds` had a silent-data-loss fallback: for a bounds list
  that didn't match any documented format (e.g. an explicit per-dimension
  bounds list whose length doesn't match the requested `dim`), it replicated
  just the *first* `(min, max)` pair across every dimension instead of
  raising, discarding the rest of the input without telling the caller. This
  was reachable in practice: `plot_function_2d('holzman1')` (any 3D+
  function plotted without an explicit `bounds=` override) silently swapped
  holzman1's documented `[(0.1, 100), (0, 25.6), (0, 5)]` bounds for
  `[(0.1, 100), (0.1, 100)]`. The fallback is removed; this shape now raises
  `ValueError` like every other unrecognized bounds format already did.
- CLI `--list` (all four suites) and `metadata.get_function_list()` displayed
  a known minimum as `0.0000`/`0.000000` whenever fixed-point formatting
  rounded it away entirely -- the scalar suite's `gear` and the engineering
  suite's `gear_train` both have a known minimum around 2.7e-12, which read
  as an exact zero. All four `--list` tables and `get_function_list()` now
  share the same fallback: switch to scientific notation whenever
  fixed-point rounding would display a nonzero value as all-zero.
- CLI `SUITE_CHOICES` was ordered `(scalar, constrained, multiobjective,
  engineering)` in code, so `--help` displayed that order, while the
  documentation (`docs/API_REFERENCE.md`) used
  `{scalar,constrained,engineering,multiobjective}`.
  Reordered the code to match the documented convention.
- `visualization.py` had a duplicate `_check_matplotlib()` definition (the
  second silently shadowed the first, byte-identical, so purely dead code);
  removed, and the now-unnecessary `F811` per-file lint ignore for this
  module dropped from `pyproject.toml`.
- `plot_search_heatmap` emitted a spurious `UserWarning: No artists with
  labels found...` for any function whose `optimal_point` is given in the
  "replicate this value across dimensions" shorthand (e.g. sphere's
  `optimal_point=[0.0]`) rather than an explicit `(x, y)` pair, because its
  own inline optimal-point handling didn't recognize that shape (unlike
  `plot_function_2d`/`plot_function_3d`, which already delegate to
  `_parse_optimal_point` for exactly this reason) yet still called
  `ax.legend()` unconditionally. Now reuses `_parse_optimal_point` and only
  calls `ax.legend()` when there's an actual labeled artist to show.
- `BenchmarkRunner.save_results(format="csv")` crashed with
  `ValueError: dict contains fields not in fieldnames` whenever a run mixed
  successes and failures, since `csv.DictWriter`'s fieldnames were taken
  from only the first result's keys, and a failed run (bounds error,
  exception, or invalid result) returns a different, smaller key set than a
  successful one. Fieldnames are now the union of keys across every result,
  in first-seen order, so mixed-outcome runs save correctly with blank
  cells for keys a given row doesn't have.
- `cec.CECCompositionFunction` crashed with `TypeError: unsupported operand
  type(s) for -: 'float' and 'NoneType'` when a component `CECFunction` was
  built without an explicit `shift_vector` (the common "no shift" case,
  since `shift_vector` defaults to `None` rather than being omitted).
  `getattr(func, "shift_vector", ...)`'s default only fires when an
  attribute is *absent*, not when it's present but `None`, so the fallback
  to "optimum at the origin" was never actually reached. Every existing test
  happened to always pass an explicit `shift_vector`, so this went uncaught
  until exercised directly.

## [0.2.0] - 2026-08-16

### Added
- `g_suite` completed: constrained G-suite now covers all 24 problems (`g01`-`g24`, CEC2006),
  up from the 4 (`g04`, `g06`, `g08`, `g11`) shipped in 0.1.0. Every formula, coefficient, and
  known-optimum value was transcribed directly from the CEC2006 technical report and
  cross-checked against pagmo's independent `cec2006` implementation where the two overlap.
  `g20`'s documented "best known" point is itself acknowledged as slightly infeasible in the
  source report — its `known_minimum` is indicative, not a strict target the way the others
  are. `g23`'s published optimal vector is corrupted by a text-extraction artifact in the
  source PDF (one coordinate garbled such that its 4th equality constraint doesn't check out);
  the point used in tests was reconstructed from the other three equality constraints and
  verified to land within 0.002 of the documented optimum.
- New `multiobjective` module: the ZDT multi-objective suite (`zdt1`-`zdt4`, `zdt6`; `zdt5` is
  binary-encoded and deliberately excluded, since it doesn't fit the `ndarray -> ndarray`
  convention every other function in the package follows). Each function returns a 2-element
  `[f1, f2]` objective array rather than a single float. Kept in its own `ZDT_SUITE` registry
  (`get_mo_function_list`, `get_mo_bounds`) rather than `metadata.BENCHMARK_SUITE`, which
  assumes a scalar `known_minimum` that doesn't apply to a Pareto front. `get_pareto_front()`
  derives each true front analytically from the function's own formula (sweeping `x1` along
  the manifold that minimizes its `g` term, then keeping the non-dominated subset) rather than
  transcribing it from the source paper — this also correctly reconstructs ZDT3's disconnected
  front and ZDT6's non-full-domain front. `igd()` and `non_dominated_front()` are general;
  `hypervolume()` currently supports 2-objective fronts only (every ZDT problem is bi-objective;
  general n-objective hypervolume is deferred to when DTLZ/WFG land).
- New `MOBenchmarkRunner` in `benchmarking.py`: the multi-objective counterpart to
  `BenchmarkRunner`, scoring a user-supplied `algorithm(func, bounds, n_objectives, **kwargs)
  -> (X, F)` against the ZDT suite's true Pareto fronts via IGD (always) and hypervolume
  (2-objective problems only). Kept as a separate class rather than an extension of
  `BenchmarkRunner`, since the two don't share a "best cost" concept.
- `examples/07_multiobjective_optimization.py`: direct ZDT evaluation plus a minimal
  evolutionary optimizer run through `MOBenchmarkRunner`, in the style of the existing
  numbered examples.
- New `g_suite.G_SUITE` registry (`function`, `dim`, `n_inequality`, `n_equality`,
  `known_minimum`), `get_g_function_list()`: the constrained suite's counterpart to
  `multiobjective.ZDT_SUITE`, for tools that need to enumerate it without hand-parsing
  docstrings. Constraint counts were cross-checked against a live call to every one of the 24
  functions, not just transcribed.
- CLI: new `--suite {scalar,constrained,multiobjective}` flag (default `scalar`, so existing
  invocations are unaffected) on `--list`/`--info`/`--metadata`/`--function`, resolving function
  names against `BENCHMARK_SUITE`, `g_suite.G_SUITE`, or `multiobjective.ZDT_SUITE` respectively.
  `g01`-`g24` and `zdt1`-`zdt4`/`zdt6` are now listable, introspectable, and evaluable (single
  point or CSV batch) from the command line, previously only reachable via Python. Evaluation
  output is shaped per suite (a plain number; an
  `{objective, inequality_violations, equality_violations}` object; an `{objectives: [f1, f2]}`
  object).
- `docs/generate_benchmark_functions.py` now also generates "Constrained Suite (g_suite)" and
  "Multi-Objective Suite (ZDT)" sections in `docs/BENCHMARK_FUNCTIONS.md` from `G_SUITE`/
  `ZDT_SUITE`, alongside the existing `BENCHMARK_SUITE`-derived table.
- 15 new scalar functions, the first slice of the MVF gap-fill (`ROADMAP.md`): `hansen`,
  `hartman3`, `hartman6`, `neumaier_perm`, `neumaier_perm0`, `neumaier_powersum`,
  `neumaier_trid`, `paviani`, `plateau`, `powell`, `shekel2`, `shekel4_5`, `shekel4_7`,
  `shekel4_10`, `shubert` — bringing `BENCHMARK_SUITE` from 59 to 74 functions. Every formula
  was transcribed from `docs-bench/mvf.pdf`'s C source appendix (more reliable than its prose
  section, which has the same OCR/text-extraction corruption already seen in g23's optimal
  point) and independently verified via numerical optimization against the documented
  known-minimum/optimal-point before being committed. Several source bugs were caught and
  corrected in the process rather than reproduced:
  - `hartman6`: the source's coefficient table has `p[2][1] = 0.1415`; both the source's own
    prose description of the same function and independent literature (sfu.ca/~ssurjano) agree
    on `0.1451`, which is also the value that actually reaches the documented -3.322368 optimum.
  - `powell`: the source's stated optimal point, `(3,-1,0,1,...)`, does not reach its own
    claimed minimum of 0 (evaluates to 215 instead) — the true global minimum is 0 at the zero
    vector, matching the standard external reference for this function.
  - The "Neumaier Perm"/"Perm0" C source sums per-term squares instead of squaring each
    per-`k` inner sum, and Perm0's C code also indexes one element out of bounds (`x[i+1]` in
    an `i`-length loop); both were implemented from the (correct, and independently
    well-established) squared-outer-sum formula instead.
  - "Neumaier Trid"'s C source never initializes its accumulator (`s2`), a latent
    undefined-behavior bug; fixed by initializing it to 0.
  - "Shekel2" (De Jong's function 5): the source computes one coordinate's penalty term to the
    9th power and the other to the 6th (should be symmetric, both to the 6th), and offsets its
    denominator by `j` instead of `j+1` (creating a division-by-zero at the very first grid
    point, `x=(-32,-32)`). Implemented as the standard, symmetric, singularity-free formula.
  - `shubert2`/`shubert3` and the 17-D `Cola`/10-D `Shekel10`/`OddSquare`/`Rana` functions from
    the same gap-fill list were **not** included in this slice: `shubert2`/`shubert3` turned out
    to be a near-duplicate or under-specified variant of `shubert` once decoded, and the other
    four have no documented known-minimum anywhere in the source (`Cola` also has an
    index-mapping scheme too ambiguous to resolve with confidence) — left for a follow-up
    rather than guessing at unverifiable values.

### Fixed
- `docs/API_REFERENCE.md`'s `g_suite` section had gone stale after the G-suite completion above
  landed: it still listed only the original 4 functions (`g04`, `g06`, `g08`, `g11`) instead of
  all 24.

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
