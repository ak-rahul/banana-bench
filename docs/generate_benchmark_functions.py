"""
Regenerate docs/BENCHMARK_FUNCTIONS.md from BENCHMARK_SUITE, g_suite.G_SUITE,
multiobjective.ZDT_SUITE, and functions.py.

This is a reusable, idempotent generator (safe to re-run any time), not a one-off
migration script: BENCHMARK_FUNCTIONS.md is a *derived* view of these registries,
and hand-editing it lets it drift from the actual metadata (this has happened
before — see CHANGELOG.md). Run it whenever a function or its metadata changes:

    python docs/generate_benchmark_functions.py

The constrained (g_suite) and multi-objective (multiobjective) suites are kept
as separate sections rather than merged into the main table, since they don't
fit its columns: g_suite functions return (objective, inequality_violations,
equality_violations), and multiobjective functions return a 2-element
objective array -- neither has a single "known minimum" the way BENCHMARK_SUITE
functions do.
"""

import inspect
from pathlib import Path

from bananabench import g_suite
from bananabench import multiobjective as mo
from bananabench.functions import fraudenstein_roth, freudenstein_roth
from bananabench.metadata import BENCHMARK_SUITE, get_all_functions


def _clean_doc(func) -> str:
    # inspect.cleandoc dedents continuation lines the way PEP 257 docstrings are
    # written (indented to match the def), so they render as prose, not a
    # Markdown code block (which 4-space-indented lines would otherwise become).
    return inspect.cleandoc(func.__doc__ or "")


def _format_number(value: float) -> str:
    if value == int(value) and abs(value) < 1e6:
        return str(int(value))
    text = f"{value:.6g}"
    return text


def _format_domain(bounds: list) -> str:
    # Bounds are stored either as a single (low, high) to replicate across
    # dimensions, or as one explicit (low, high) per dimension — those explicit
    # per-dimension bounds are still a uniform "^n" domain if they all agree.
    if len(set(bounds)) == 1:
        low, high = bounds[0]
        return f"[{_format_number(low)}, {_format_number(high)}]^n"
    return "Varies"


def _format_optimal_point(optimal_point) -> str:
    if isinstance(optimal_point[0], (tuple, list)):
        points = ", ".join(f"({', '.join(_format_number(v) for v in p)})" for p in optimal_point)
        return f"x = [{points}]"
    return f"x = [{', '.join(_format_number(v) for v in optimal_point)}]"


def _function_section(name: str, info: dict) -> str:
    func = info["function"]
    lines = [f"### {name}", ""]
    lines.append(f"**Default Dimension:** {info['default_dim']}")
    lines.append(f"**Known Minimum:** {_format_number(info['known_minimum'])}")
    if info["optimal_point"] is not None:
        lines.append(f"**Optimal Point:** `{_format_optimal_point(info['optimal_point'])}`")
    lines.append("")
    lines.append("#### Description")
    lines.append(_clean_doc(func))
    lines.append("")
    lines.append("#### Code Example")
    lines.append("```python")
    lines.append(f"from bananabench import {name}")
    lines.append("import numpy as np")
    lines.append("")
    lines.append(f"# Run {name}")
    lines.append(f"x = np.zeros({info['default_dim']})")
    lines.append(f"result = {name}(x)")
    lines.append("print(f'result: {result}')")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _g_suite_section(name: str, entry: dict) -> str:
    lines = [f"### {name}", ""]
    lines.append(f"**Dimension:** {entry['dim']}")
    lines.append(
        f"**Constraints:** {entry['n_inequality']} inequality, {entry['n_equality']} equality"
    )
    lines.append(f"**Known Minimum:** {_format_number(entry['known_minimum'])}")
    lines.append("")
    lines.append("#### Description")
    lines.append(_clean_doc(entry["function"]))
    lines.append("")
    lines.append("#### Code Example")
    lines.append("```python")
    lines.append("from bananabench import g_suite")
    lines.append("import numpy as np")
    lines.append("")
    lines.append(f"x = np.ones({entry['dim']})")
    lines.append(f"objective, inequality_violations, equality_violations = g_suite.{name}(x)")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _mo_section(name: str, entry: dict) -> str:
    dim = entry["default_dim"]
    lines = [f"### {name}", ""]
    lines.append(f"**Default Dimension:** {dim}")
    lines.append(f"**Objectives:** {entry['n_objectives']}")
    lines.append(f"**Properties:** {', '.join(entry['properties'])}")
    lines.append("")
    lines.append("#### Description")
    lines.append(_clean_doc(entry["function"]))
    lines.append("")
    lines.append("#### Code Example")
    lines.append("```python")
    lines.append("from bananabench import multiobjective as mo")
    lines.append("import numpy as np")
    lines.append("")
    lines.append(f"x = np.zeros({dim})")
    lines.append(f"f1, f2 = mo.{name}(x)")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def generate() -> str:
    names = get_all_functions()
    lines = [
        "# Benchmark Functions Reference",
        "",
        "This document provides a detailed reference for all benchmark functions available in the "
        "package.",
        "",
        "> Generated by `docs/generate_benchmark_functions.py` from `BENCHMARK_SUITE` — do not hand-edit "
        "the tables/sections below; re-run the script instead.",
        "",
        "## Summary",
        "",
        f"Total functions: {len(names)}",
        "",
        "| Function | Dimensions | Domain | Global Minimum |",
        "|----------|------------|--------|----------------|",
    ]
    for name in names:
        info = BENCHMARK_SUITE[name]
        domain = _format_domain(info["bounds"])
        lines.append(
            f"| [{name}](#{name}) | {info['default_dim']}D | {domain} | "
            f"{_format_number(info['known_minimum'])} |"
        )

    lines.append("")
    lines.append("## Detailed Descriptions")
    lines.append("")
    for name in names:
        lines.append(_function_section(name, BENCHMARK_SUITE[name]))

    lines.append("## Constrained Suite (g_suite)")
    lines.append("")
    lines.append(
        "The classic CEC2006 constrained G-suite (`g01`-`g24`). Not part of `BENCHMARK_SUITE` -- "
        "each function returns `(objective, inequality_violations, equality_violations)` rather "
        "than a single float, so this section is generated from its own `g_suite.G_SUITE` "
        "registry instead. See `bananabench.g_suite` for details."
    )
    lines.append("")
    lines.append(f"Total functions: {len(g_suite.G_SUITE)}")
    lines.append("")
    lines.append("| Function | Dimensions | Inequality | Equality | Known Minimum |")
    lines.append("|----------|------------|------------|----------|----------------|")
    for name in g_suite.get_g_function_list():
        entry = g_suite.G_SUITE[name]
        lines.append(
            f"| [{name}](#{name}) | {entry['dim']}D | {entry['n_inequality']} | "
            f"{entry['n_equality']} | {_format_number(entry['known_minimum'])} |"
        )
    lines.append("")
    for name in g_suite.get_g_function_list():
        lines.append(_g_suite_section(name, g_suite.G_SUITE[name]))

    lines.append("## Multi-Objective Suite (ZDT)")
    lines.append("")
    lines.append(
        "The ZDT multi-objective suite (`zdt1`-`zdt4`, `zdt6`; `zdt5` is binary-encoded and "
        "excluded). Each function returns a 2-element `[f1, f2]` array rather than a single "
        "float, so this section is generated from its own `multiobjective.ZDT_SUITE` registry "
        "instead. See `bananabench.multiobjective` for details."
    )
    lines.append("")
    lines.append(f"Total functions: {len(mo.ZDT_SUITE)}")
    lines.append("")
    lines.append("| Function | Default Dim | Objectives | Properties |")
    lines.append("|----------|--------------|------------|------------|")
    for name in mo.get_mo_function_list():
        entry = mo.ZDT_SUITE[name]
        lines.append(
            f"| [{name}](#{name}) | {entry['default_dim']}D | {entry['n_objectives']} | "
            f"{', '.join(entry['properties'])} |"
        )
    lines.append("")
    for name in mo.get_mo_function_list():
        lines.append(_mo_section(name, mo.ZDT_SUITE[name]))

    lines.append("## Spelling Aliases")
    lines.append("")
    lines.append(
        "Not part of `BENCHMARK_SUITE` (no default dimension/bounds/known-minimum metadata "
        "registered), but importable directly:"
    )
    lines.append("")
    lines.append(
        "- `freudenstein_roth` — corrected spelling of `fraudenstein_roth` (both importable "
        "for backward compatibility; `freudenstein_roth is fraudenstein_roth`)."
    )
    doc_lines = _clean_doc(fraudenstein_roth).splitlines()
    lines.extend(f"  {line}" for line in doc_lines)
    lines.append("")
    assert freudenstein_roth is fraudenstein_roth

    return "\n".join(lines).rstrip() + "\n"


if __name__ == "__main__":
    output_path = Path(__file__).parent / "BENCHMARK_FUNCTIONS.md"
    output_path.write_text(generate(), encoding="utf-8")
    print(f"Wrote {output_path} ({len(get_all_functions())} functions)")
