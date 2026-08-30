"""
Command-line interface for the banana-bench package.

Provides utilities to evaluate benchmark functions from the command line,
supporting single evaluations, batch processing from CSV files (with parallel
support via --jobs), function introspection, and metadata queries.

--suite selects which registry a function name is resolved against:
'scalar' (default, functions.py/BENCHMARK_SUITE -- unchanged from prior
versions, so existing invocations keep working exactly as before),
'constrained' (g_suite.G_SUITE, g01-g24), or 'multiobjective'
(multiobjective.ZDT_SUITE, zdt1-zdt6). The three have different evaluation
result shapes (a scalar float; an objective + inequality/equality violation
arrays; a 2-element objective array), so evaluation results are formatted
per-suite via `_format_result`.

References:
-----------
[1] Adorio, E. P. (2005). MVF - Multivariate Test Functions Library in C.
"""

import argparse
import csv
import inspect
import json
import sys

from bananabench import BENCHMARK_SUITE, functions, g_suite, get_all_functions, get_function_info
from bananabench import multiobjective as mo

SUITE_CHOICES = ("scalar", "constrained", "multiobjective")

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


def get_available_functions():
    """
    Retrieve a sorted list of all available function names.
    Now uses metadata module for consistency.
    """
    return get_all_functions()


def _get_suite_function(suite, name):
    """Resolve a function name against the chosen suite's registry; None if not found."""
    if suite == "constrained":
        entry = g_suite.G_SUITE.get(name)
        return entry["function"] if entry else None
    if suite == "multiobjective":
        entry = mo.ZDT_SUITE.get(name)
        return entry["function"] if entry else None
    return getattr(functions, name, None) if hasattr(functions, name) else None


def _format_result(suite, raw):
    """Shape a raw function return value for JSON output, per suite convention."""
    if suite == "constrained":
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
    elif suite == "multiobjective":
        _print_multiobjective_list()
    else:
        _print_scalar_list()


def _print_scalar_list():
    print("=" * 80)
    print("Available Benchmark Functions")
    print("=" * 80)
    print(f"{'Function':<25} | {'Dim':>3} | {'Known Min':>12} | {'Bounds'}")
    print("-" * 80)

    for name in get_all_functions():
        if name in BENCHMARK_SUITE:
            meta = BENCHMARK_SUITE[name]
            bounds_str = str(meta["bounds"][0]) if len(meta["bounds"]) == 1 else "varied"
            print(
                f"{name:<25} | {meta['default_dim']:3d} | {meta['known_minimum']:12.4f} | {bounds_str}"
            )
        else:
            # Fallback for functions not in metadata
            print(f"{name:<25} | {'?':>3} | {'?':>12} | ?")

    print("=" * 80)
    print(f"Total: {len(get_all_functions())} functions")
    print(
        "Also available: --suite constrained "
        f"({len(g_suite.G_SUITE)} functions), --suite multiobjective ({len(mo.ZDT_SUITE)} functions)"
    )


def _print_constrained_list():
    print("=" * 80)
    print("Available Constrained Functions (g_suite, CEC2006)")
    print("=" * 80)
    print(f"{'Function':<10} | {'Dim':>3} | {'Ineq':>4} | {'Eq':>4} | {'Known Min':>14}")
    print("-" * 80)

    for name in g_suite.get_g_function_list():
        entry = g_suite.G_SUITE[name]
        print(
            f"{name:<10} | {entry['dim']:3d} | {entry['n_inequality']:4d} | "
            f"{entry['n_equality']:4d} | {entry['known_minimum']:14.6f}"
        )

    print("=" * 80)
    print(f"Total: {len(g_suite.G_SUITE)} functions")


def _print_multiobjective_list():
    print("=" * 80)
    print("Available Multi-Objective Functions (ZDT suite)")
    print("=" * 80)
    print(f"{'Function':<10} | {'Dim':>3} | {'Objectives':>10}")
    print("-" * 80)

    for name in mo.get_mo_function_list():
        entry = mo.ZDT_SUITE[name]
        print(f"{name:<10} | {entry['default_dim']:3d} | {entry['n_objectives']:10d}")

    print("=" * 80)
    print(f"Total: {len(mo.ZDT_SUITE)} functions")


def print_function_info(func_name, suite="scalar"):
    """
    Print comprehensive information for the specified function, for the
    chosen suite ('scalar' by default).
    """
    if suite == "constrained":
        _print_constrained_info(func_name)
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


def _print_multiobjective_info(func_name):
    if func_name not in mo.ZDT_SUITE:
        print(
            f"Error: Function '{func_name}' not found in the multiobjective (ZDT) suite.",
            file=sys.stderr,
        )
        sys.exit(1)

    entry = mo.ZDT_SUITE[func_name]
    doc = inspect.getdoc(entry["function"])

    print("=" * 80)
    print(f"Function: {func_name}  (multiobjective / ZDT)")
    print("=" * 80)
    if doc:
        print(f"\nDocumentation:\n{doc}\n")
    else:
        print("\nNo documentation available.\n")

    print("Metadata:")
    print(f"  Default dimension:  {entry['default_dim']}")
    print(f"  Objectives:         {entry['n_objectives']}")
    print(f"  Bounds:             {mo.get_mo_bounds(func_name)}")
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
    elif suite == "multiobjective":
        if func_name not in mo.ZDT_SUITE:
            print(
                f"Error: No metadata available for function '{func_name}' in the "
                "multiobjective suite.",
                file=sys.stderr,
            )
            sys.exit(1)
        entry = mo.ZDT_SUITE[func_name]
        metadata_dict = {
            "function": func_name,
            "suite": "multiobjective",
            "default_dimension": entry["default_dim"],
            "n_objectives": entry["n_objectives"],
            "bounds": mo.get_mo_bounds(func_name),
            "properties": entry["properties"],
        }
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


def evaluate_function(func_name, values, suite="scalar"):
    """
    Evaluate the given function with the provided input values.
    Returns a dictionary containing the input and the result.
    """
    func = _get_suite_function(suite, func_name)
    if func is None:
        print(f"Error: Function '{func_name}' not found in suite '{suite}'.", file=sys.stderr)
        sys.exit(1)

    try:
        # Convert input values to float
        x = [float(v) for v in values]
    except ValueError as e:
        print(f"Error: Unable to convert input values to float: {e}", file=sys.stderr)
        sys.exit(1)

    result = _format_result(suite, func(x))
    return {"input": x, "result": result}


def evaluate_function_batch(func_name, input_file, n_jobs=1, suite="scalar"):
    """
    Evaluate the given function on a batch of input vectors from a CSV file.
    Returns a list of dictionaries with inputs and results.
    Support parallel execution via n_jobs.
    """
    results = []

    func = _get_suite_function(suite, func_name)
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
            "  banana-bench --list --suite multiobjective\n"
            "  banana-bench --function zdt1 --suite multiobjective --values 0.5 0 0\n"
        ),
    )

    parser.add_argument(
        "--list", action="store_true", help="List all available functions with metadata"
    )
    parser.add_argument(
        "--suite",
        choices=SUITE_CHOICES,
        default="scalar",
        help=(
            "Which function registry to use (default: scalar). 'constrained' resolves against "
            "g_suite (g01-g24); 'multiobjective' resolves against the ZDT suite (zdt1-zdt4, zdt6)."
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
        result_entry = evaluate_function(func_name, args.values, suite=args.suite)
        output_data["result"] = result_entry["result"]
        output_data["input"] = result_entry["input"]

    # Batch evaluation
    elif args.input:
        n_jobs = args.jobs if args.jobs else 1
        results = evaluate_function_batch(func_name, args.input, n_jobs=n_jobs, suite=args.suite)
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
