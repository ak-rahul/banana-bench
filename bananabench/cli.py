"""
Command-line interface for the banana-bench package.

Provides utilities to evaluate benchmark functions from the command line,
supporting single evaluations, batch processing from CSV files (with parallel
support via --jobs), function introspection, and metadata queries.

--suite selects which registry a function name is resolved against:
'scalar' (default, functions.py/BENCHMARK_SUITE -- unchanged from prior
versions, so existing invocations keep working exactly as before),
'constrained' (g_suite.G_SUITE, g01-g24), 'multiobjective'
(multiobjective.MO_SUITE: zdt1-zdt4, zdt6 and dtlz1-dtlz7; --objectives sets M for
DTLZ), or 'engineering'
(engineering.ENGINEERING_SUITE, the classic design problems). 'engineering'
shares 'constrained''s result shape (objective + inequality/equality
violation arrays), since both follow the same convention; 'multiobjective'
returns an M-element objective array instead, so evaluation results are
formatted per-suite via `_format_result`.

References:
-----------
[1] Adorio, E. P. (2005). MVF - Multivariate Test Functions Library in C.
"""

import argparse
import csv
import inspect
import json
import os
import sys

from bananabench import (
    BENCHMARK_SUITE,
    __version__,
    engineering,
    functions,
    g_suite,
    get_all_functions,
    get_function_info,
)
from bananabench import multiobjective as mo
from bananabench.utils import _format_value

SUITE_CHOICES = ("scalar", "constrained", "engineering", "multiobjective")

# Function docstrings contain Unicode math symbols (e.g. ≤, π). On Windows,
# stdout/stderr default to the legacy console codepage (e.g. cp1252), which cannot
# encode them and raises UnicodeEncodeError. Force UTF-8 so `--info`/`--list` work
# consistently across platforms.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


# --- Terminal banner & color -------------------------------------------------
# Generated once with `pyfiglet.Figlet(font="ansi_shadow", width=200).renderText(
# "BANANA-BENCH")` rather than hand-drawn, so the glyphs are guaranteed correctly
# aligned -- the explicit `width=200` matters: pyfiglet's default ~80-column wrap
# width chops this render into disconnected stacked letter-groups. "ansi_shadow"
# has no lowercase glyphs at all (verified: it renders 'B' and 'b' identically),
# so the banner is uppercase; the hyphen renders as a small mid-height block
# glyph spanning only the two middle rows, which is this font's actual hyphen
# design rather than a rendering defect.
_BANNER = r"""
██████╗  █████╗ ███╗   ██╗ █████╗ ███╗   ██╗ █████╗       ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗
██╔══██╗██╔══██╗████╗  ██║██╔══██╗████╗  ██║██╔══██╗      ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║
██████╔╝███████║██╔██╗ ██║███████║██╔██╗ ██║███████║█████╗██████╔╝█████╗  ██╔██╗ ██║██║     ███████║
██╔══██╗██╔══██║██║╚██╗██║██╔══██║██║╚██╗██║██╔══██║╚════╝██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║
██████╔╝██║  ██║██║ ╚████║██║  ██║██║ ╚████║██║  ██║      ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝      ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝
""".strip("\n")


def _supports_color() -> bool:
    """Only colorize when writing to a real terminal, and never when NO_COLOR is
    set (any non-empty value -- see https://no-color.org/), matching the same two
    checks ruff/uv use to stay well-behaved in CI logs and redirected output."""
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _style(text: str, code: str) -> str:
    """Wrap `text` in an ANSI SGR code, or return it unchanged if colors are off."""
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _print_banner() -> None:
    print(_style(_BANNER, "1;33"))  # bold yellow -- the banana in banana-bench
    print(_style(f"  optimization benchmark suite  ·  v{__version__}", "2"))
    print()


def get_available_functions():
    """
    Retrieve a sorted list of all available function names.
    Now uses metadata module for consistency.
    """
    return get_all_functions()


def _get_suite_function(suite, name, n_objectives=None):
    """Resolve a function name against the chosen suite's registry; None if not found.
    For DTLZ problems, ``n_objectives`` (default 3) is bound into the returned function."""
    if suite == "constrained":
        entry = g_suite.G_SUITE.get(name)
        return entry["function"] if entry else None
    if suite == "engineering":
        entry = engineering.ENGINEERING_SUITE.get(name)
        return entry["function"] if entry else None
    if suite == "multiobjective":
        if name not in mo.MO_SUITE:
            return None
        try:
            return mo.get_mo_function(name, n_objectives)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    return getattr(functions, name, None) if hasattr(functions, name) else None


def _format_known_min(value: float, width: int = 12, precision: int = 4) -> str:
    """Right-align a known_minimum for the CLI's ``--list`` tables (see
    ``utils._format_value``, which also handles values like gear's ~2.7e-12)."""
    return _format_value(value, width=width, precision=precision)


def _print_list_header(title: str, width: int = 80) -> None:
    """Shared banner-bar/title styling for every `--list` table (colorized when
    supported, matching `_supports_color()`'s NO_COLOR/isatty rules)."""
    border = _style("=" * width, "36")
    print(border)
    print(_style(title, "1;36"))
    print(border)


def _print_list_footer(width: int = 80) -> None:
    print(_style("=" * width, "36"))


def _format_result(suite, raw):
    """Shape a raw function return value for JSON output, per suite convention."""
    if suite in ("constrained", "engineering"):
        f, g, h = raw
        return {
            "objective": float(f),
            "inequality_violations": [float(v) for v in g],
            "equality_violations": [float(v) for v in h],
        }
    if suite == "multiobjective":
        return {"objectives": [float(v) for v in raw]}
    return raw


def print_function_list(suite="scalar"):
    """
    Print the list of available functions with metadata information, for the
    chosen suite ('scalar' by default).
    """
    if suite == "constrained":
        _print_constrained_list()
    elif suite == "engineering":
        _print_engineering_list()
    elif suite == "multiobjective":
        _print_multiobjective_list()
    else:
        _print_scalar_list()


def _print_scalar_list():
    _print_list_header("Available Benchmark Functions")
    print(f"{'Function':<25} | {'Dim':>3} | {'Known Min':>12} | {'Bounds'}")
    print("-" * 80)

    for name in get_all_functions():
        if name in BENCHMARK_SUITE:
            meta = BENCHMARK_SUITE[name]
            bounds_str = str(meta["bounds"][0]) if len(meta["bounds"]) == 1 else "varied"
            print(
                f"{name:<25} | {meta['default_dim']:3d} | "
                f"{_format_known_min(meta['known_minimum'])} | {bounds_str}"
            )
        else:
            # Fallback for functions not in metadata
            print(f"{name:<25} | {'?':>3} | {'?':>12} | ?")

    _print_list_footer()
    print(f"Total: {len(get_all_functions())} functions")
    print(
        "Also available: --suite constrained "
        f"({len(g_suite.G_SUITE)} functions), --suite engineering "
        f"({len(engineering.ENGINEERING_SUITE)} functions), --suite multiobjective "
        f"({len(mo.MO_SUITE)} functions)"
    )


def _print_constrained_list():
    _print_list_header("Available Constrained Functions (g_suite, CEC2006)")
    print(f"{'Function':<10} | {'Dim':>3} | {'Ineq':>4} | {'Eq':>4} | {'Known Min':>14}")
    print("-" * 80)

    for name in g_suite.get_g_function_list():
        entry = g_suite.G_SUITE[name]
        print(
            f"{name:<10} | {entry['dim']:3d} | {entry['n_inequality']:4d} | "
            f"{entry['n_equality']:4d} | {_format_known_min(entry['known_minimum'], width=14, precision=6)}"
        )

    _print_list_footer()
    print(f"Total: {len(g_suite.G_SUITE)} functions")


def _print_engineering_list():
    _print_list_header("Available Engineering Design Functions")
    print(f"{'Function':<16} | {'Dim':>3} | {'Ineq':>4} | {'Eq':>4} | {'Known Min':>14}")
    print("-" * 80)

    for name in engineering.get_engineering_function_list():
        entry = engineering.ENGINEERING_SUITE[name]
        print(
            f"{name:<16} | {entry['dim']:3d} | {entry['n_inequality']:4d} | "
            f"{entry['n_equality']:4d} | {_format_known_min(entry['known_minimum'], width=14, precision=6)}"
        )

    _print_list_footer()
    print(f"Total: {len(engineering.ENGINEERING_SUITE)} functions")


def _print_multiobjective_list():
    _print_list_header("Available Multi-Objective Functions (ZDT and DTLZ suites)")
    print(f"{'Function':<10} | {'Dim':>3} | {'Objectives':>10} | Notes")
    print("-" * 80)

    for name in mo.get_mo_function_list():
        entry = mo.MO_SUITE[name]
        note = (
            f"any M >= 2 via --objectives; dim = M + {entry['k'] - 1}"
            if name in mo.DTLZ_SUITE
            else "bi-objective"
        )
        print(f"{name:<10} | {entry['default_dim']:3d} | {entry['n_objectives']:10d} | {note}")

    _print_list_footer()
    print(f"Total: {len(mo.MO_SUITE)} functions")


def print_function_info(func_name, suite="scalar"):
    """
    Print comprehensive information for the specified function, for the
    chosen suite ('scalar' by default).
    """
    if suite == "constrained":
        _print_constrained_info(func_name)
    elif suite == "engineering":
        _print_engineering_info(func_name)
    elif suite == "multiobjective":
        _print_multiobjective_info(func_name)
    else:
        _print_scalar_info(func_name)


def _print_scalar_info(func_name):
    if not hasattr(functions, func_name):
        print(f"Error: Function '{func_name}' not found.", file=sys.stderr)
        sys.exit(1)

    func = getattr(functions, func_name)
    doc = inspect.getdoc(func)

    print("=" * 80)
    print(f"Function: {func_name}")
    print("=" * 80)

    # Show documentation
    if doc:
        print(f"\nDocumentation:\n{doc}\n")
    else:
        print("\nNo documentation available.\n")

    # Show metadata if available
    if func_name in BENCHMARK_SUITE:
        meta = BENCHMARK_SUITE[func_name]
        print("Metadata:")
        print(f"  Dimension:      {meta['default_dim']}")
        print(f"  Bounds:         {meta['bounds']}")
        print(f"  Known minimum:  {meta['known_minimum']}")
        if meta["optimal_point"] is not None:
            print(f"  Optimal point:  {meta['optimal_point']}")
    else:
        print("Metadata: Not available")

    print("=" * 80)


def _print_constrained_info(func_name):
    if func_name not in g_suite.G_SUITE:
        print(
            f"Error: Function '{func_name}' not found in the constrained (g_suite) suite.",
            file=sys.stderr,
        )
        sys.exit(1)

    entry = g_suite.G_SUITE[func_name]
    doc = inspect.getdoc(entry["function"])

    print("=" * 80)
    print(f"Function: {func_name}  (constrained / g_suite)")
    print("=" * 80)
    if doc:
        print(f"\nDocumentation:\n{doc}\n")
    else:
        print("\nNo documentation available.\n")

    print("Metadata:")
    print(f"  Dimension:              {entry['dim']}")
    print(f"  Inequality constraints: {entry['n_inequality']}")
    print(f"  Equality constraints:   {entry['n_equality']}")
    print(f"  Known minimum:          {entry['known_minimum']}")
    print("=" * 80)


def _print_engineering_info(func_name):
    if func_name not in engineering.ENGINEERING_SUITE:
        print(
            f"Error: Function '{func_name}' not found in the engineering suite.",
            file=sys.stderr,
        )
        sys.exit(1)

    entry = engineering.ENGINEERING_SUITE[func_name]
    doc = inspect.getdoc(entry["function"])

    print("=" * 80)
    print(f"Function: {func_name}  (engineering)")
    print("=" * 80)
    if doc:
        print(f"\nDocumentation:\n{doc}\n")
    else:
        print("\nNo documentation available.\n")

    print("Metadata:")
    print(f"  Dimension:              {entry['dim']}")
    print(f"  Inequality constraints: {entry['n_inequality']}")
    print(f"  Equality constraints:   {entry['n_equality']}")
    print(f"  Known minimum:          {entry['known_minimum']}")
    print("=" * 80)


def _print_multiobjective_info(func_name):
    if func_name not in mo.MO_SUITE:
        print(
            f"Error: Function '{func_name}' not found in the multiobjective suite.",
            file=sys.stderr,
        )
        sys.exit(1)

    entry = mo.MO_SUITE[func_name]
    family = "DTLZ" if func_name in mo.DTLZ_SUITE else "ZDT"
    doc = inspect.getdoc(entry["function"])

    print("=" * 80)
    print(f"Function: {func_name}  (multiobjective / {family})")
    print("=" * 80)
    if doc:
        print(f"\nDocumentation:\n{doc}\n")
    else:
        print("\nNo documentation available.\n")

    print("Metadata:")
    print(f"  Default dimension:  {entry['default_dim']}")
    objectives = entry["n_objectives"]
    if func_name in mo.DTLZ_SUITE:
        print(f"  Objectives:         {objectives} by default (any M >= 2, --objectives)")
        print(f"  Distance vars (k):  {entry['k']}  (dimension = M + k - 1)")
    else:
        print(f"  Objectives:         {objectives}")
    bounds = "[0, 1]^n" if func_name in mo.DTLZ_SUITE else mo.get_mo_bounds(func_name)
    print(f"  Bounds:             {bounds}")
    print(f"  Properties:         {', '.join(entry['properties'])}")
    print("=" * 80)


def print_metadata(func_name, suite="scalar"):
    """
    Print only metadata for the specified function, for the chosen suite
    ('scalar' by default), as JSON.
    """
    if suite == "constrained":
        if func_name not in g_suite.G_SUITE:
            print(
                f"Error: No metadata available for function '{func_name}' in the constrained suite.",
                file=sys.stderr,
            )
            sys.exit(1)
        entry = g_suite.G_SUITE[func_name]
        metadata_dict = {
            "function": func_name,
            "suite": "constrained",
            "dimension": entry["dim"],
            "n_inequality": entry["n_inequality"],
            "n_equality": entry["n_equality"],
            "known_minimum": entry["known_minimum"],
        }
    elif suite == "engineering":
        if func_name not in engineering.ENGINEERING_SUITE:
            print(
                f"Error: No metadata available for function '{func_name}' in the "
                "engineering suite.",
                file=sys.stderr,
            )
            sys.exit(1)
        entry = engineering.ENGINEERING_SUITE[func_name]
        metadata_dict = {
            "function": func_name,
            "suite": "engineering",
            "dimension": entry["dim"],
            "n_inequality": entry["n_inequality"],
            "n_equality": entry["n_equality"],
            "known_minimum": entry["known_minimum"],
        }
    elif suite == "multiobjective":
        if func_name not in mo.MO_SUITE:
            print(
                f"Error: No metadata available for function '{func_name}' in the "
                "multiobjective suite.",
                file=sys.stderr,
            )
            sys.exit(1)
        entry = mo.MO_SUITE[func_name]
        metadata_dict = {
            "function": func_name,
            "suite": "multiobjective",
            "default_dimension": entry["default_dim"],
            "n_objectives": entry["n_objectives"],
            "scalable_objectives": func_name in mo.DTLZ_SUITE,
            "bounds": mo.get_mo_bounds(func_name),
            "properties": entry["properties"],
        }
        if func_name in mo.DTLZ_SUITE:
            metadata_dict["k"] = entry["k"]
    else:
        if func_name not in BENCHMARK_SUITE:
            print(f"Error: No metadata available for function '{func_name}'.", file=sys.stderr)
            sys.exit(1)
        meta = BENCHMARK_SUITE[func_name]
        metadata_dict = {
            "function": func_name,
            "default_dimension": meta["default_dim"],
            "bounds": meta["bounds"],
            "known_minimum": meta["known_minimum"],
            "optimal_point": meta["optimal_point"],
        }

    print(json.dumps(metadata_dict, indent=2))


def evaluate_function(func_name, values, suite="scalar", n_objectives=None):
    """
    Evaluate the given function with the provided input values.
    Returns a dictionary containing the input and the result.
    """
    func = _get_suite_function(suite, func_name, n_objectives)
    if func is None:
        print(f"Error: Function '{func_name}' not found in suite '{suite}'.", file=sys.stderr)
        sys.exit(1)

    try:
        # Convert input values to float
        x = [float(v) for v in values]
    except ValueError as e:
        print(f"Error: Unable to convert input values to float: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        raw = func(x)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    result = _format_result(suite, raw)
    return {"input": x, "result": result}


def evaluate_function_batch(func_name, input_file, n_jobs=1, suite="scalar", n_objectives=None):
    """
    Evaluate the given function on a batch of input vectors from a CSV file.
    Returns a list of dictionaries with inputs and results.
    Support parallel execution via n_jobs.
    """
    results = []

    func = _get_suite_function(suite, func_name, n_objectives)
    if func is None:
        print(f"Error: Function '{func_name}' not found in suite '{suite}'.", file=sys.stderr)
        sys.exit(1)

    def _run(x):
        return _format_result(suite, func(x))

    try:
        inputs = []
        with open(input_file, "r", newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row_num, row in enumerate(reader, start=1):
                if not row:
                    continue  # Skip empty lines

                try:
                    x = [float(v) for v in row]
                    inputs.append(x)
                except ValueError as e:
                    print(f"Error: Invalid number in CSV at line {row_num}: {e}", file=sys.stderr)
                    sys.exit(1)

        if not inputs:
            return []

        # Parallel execution
        if n_jobs > 1:
            try:
                from joblib import Parallel, delayed

                outputs = Parallel(n_jobs=n_jobs)(delayed(_run)(x) for x in inputs)
                results = [{"input": x, "result": y} for x, y in zip(inputs, outputs)]
            except ImportError:
                print(
                    "Warning: joblib not installed. Falling back to serial execution.",
                    file=sys.stderr,
                )
                outputs = [_run(x) for x in inputs]
                results = [{"input": x, "result": y} for x, y in zip(inputs, outputs)]
        else:
            # Serial execution
            outputs = [_run(x) for x in inputs]
            results = [{"input": x, "result": y} for x, y in zip(inputs, outputs)]

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file '{input_file}': {e}", file=sys.stderr)
        sys.exit(1)

    return results


def main():
    """
    Entry point for the bananabench CLI.
    """
    # Banner only on bare invocation and --help -- never on --list/--info/--metadata/
    # --function, whose output is machine-consumable (piped to `jq`, saved via
    # --output, or parsed by a script) and would be corrupted by extra stdout text.
    if len(sys.argv) == 1 or any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        _print_banner()

    parser = argparse.ArgumentParser(
        description="Command-line interface for evaluating optimization benchmark functions.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  banana-bench --list\n"
            "  banana-bench --info ackley\n"
            "  banana-bench --metadata ackley\n"
            "  banana-bench --function ackley --values 0 0 0\n"
            "  banana-bench --function rastrigin --input points.csv --output results.json\n"
            "  banana-bench --list --suite constrained\n"
            "  banana-bench --function g01 --suite constrained --values 1 1 1 1 1 1 1 1 1 3 3 3 1\n"
            "  banana-bench --list --suite engineering\n"
            "  banana-bench --function spring_design --suite engineering --values 0.05 0.25 2.0\n"
            "  banana-bench --list --suite multiobjective\n"
            "  banana-bench --function zdt1 --suite multiobjective --values 0.5 0 0\n"
            "  banana-bench --function dtlz2 --suite multiobjective --objectives 3 "
            "--values 0.5 0.5 0.5 0.5\n"
        ),
    )

    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--list", action="store_true", help="List all available functions with metadata"
    )
    parser.add_argument(
        "--suite",
        choices=SUITE_CHOICES,
        default="scalar",
        help=(
            "Which function registry to use (default: scalar). 'constrained' resolves against "
            "g_suite (g01-g24); 'engineering' resolves against the classic design suite "
            "(spring_design, pressure_vessel, welded_beam, gear_train); 'multiobjective' "
            "resolves against the ZDT (zdt1-zdt4, zdt6) and DTLZ (dtlz1-dtlz7) suites."
        ),
    )
    parser.add_argument(
        "--objectives",
        metavar="M",
        type=int,
        help=(
            "Number of objectives for DTLZ problems (default 3; any M >= 2). "
            "Only valid with --suite multiobjective."
        ),
    )
    parser.add_argument(
        "--info",
        metavar="FUNCTION",
        help="Show documentation and metadata for the specified function",
    )
    parser.add_argument(
        "--metadata",
        metavar="FUNCTION",
        help="Show only metadata for the specified function (JSON format)",
    )
    parser.add_argument("--function", metavar="FUNCTION", help="Name of the function to evaluate")
    parser.add_argument(
        "--values",
        metavar="N",
        nargs="+",
        help="Input values for single evaluation (space-separated)",
    )
    parser.add_argument(
        "--input", metavar="FILE", help="CSV file with input vectors for batch evaluation"
    )
    parser.add_argument(
        "--output", metavar="FILE", help="Output file to write results in JSON format"
    )
    parser.add_argument(
        "--jobs",
        metavar="N",
        type=int,
        default=1,
        help="Number of parallel jobs for batch processing",
    )

    args = parser.parse_args()

    if args.objectives is not None and args.suite != "multiobjective":
        print("Error: --objectives requires --suite multiobjective.", file=sys.stderr)
        sys.exit(1)

    # Handle --list
    if args.list:
        if args.info or args.metadata or args.function or args.values or args.input or args.output:
            print("Error: --list cannot be combined with other options.", file=sys.stderr)
            sys.exit(1)
        print_function_list(suite=args.suite)
        sys.exit(0)

    # Handle --info
    if args.info is not None:
        if args.metadata or args.function or args.values or args.input or args.output:
            print("Error: --info cannot be combined with other options.", file=sys.stderr)
            sys.exit(1)
        print_function_info(args.info, suite=args.suite)
        sys.exit(0)

    # Handle --metadata
    if args.metadata is not None:
        if args.info or args.function or args.values or args.input or args.output:
            print("Error: --metadata cannot be combined with other options.", file=sys.stderr)
            sys.exit(1)
        print_metadata(args.metadata, suite=args.suite)
        sys.exit(0)

    # From here, require --function
    if not args.function:
        if len(sys.argv) == 1:
            # Bare invocation: the banner is already on screen (printed above) --
            # follow it with a getting-started nudge instead of a bare error.
            print("Try one of these to get started:\n")
            print("  banana-bench --list")
            print("  banana-bench --info ackley")
            print("  banana-bench --function ackley --values 0 0 0")
            print("  banana-bench --help\n")
        else:
            print("Error: --function is required for evaluation.", file=sys.stderr)
            parser.print_usage(sys.stderr)
        sys.exit(1)

    func_name = args.function

    # Determine evaluation mode
    if args.values and args.input:
        print("Error: --values and --input cannot be used together.", file=sys.stderr)
        sys.exit(1)

    if not args.values and not args.input:
        print(
            "Error: Either --values or --input must be provided for function evaluation.",
            file=sys.stderr,
        )
        parser.print_usage(sys.stderr)
        sys.exit(1)

    output_data = {"function": func_name}

    # Single evaluation
    if args.values:
        result_entry = evaluate_function(
            func_name, args.values, suite=args.suite, n_objectives=args.objectives
        )
        output_data["result"] = result_entry["result"]
        output_data["input"] = result_entry["input"]

    # Batch evaluation
    elif args.input:
        n_jobs = args.jobs if args.jobs else 1
        results = evaluate_function_batch(
            func_name, args.input, n_jobs=n_jobs, suite=args.suite, n_objectives=args.objectives
        )
        output_data["results"] = results

    # Output results
    output_json = json.dumps(output_data, indent=2)

    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(output_json)
        except Exception as e:
            print(f"Error writing to output file '{args.output}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
