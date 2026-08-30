"""
Tests for the ``banana-bench`` command-line interface.

Runs the CLI as a subprocess (rather than calling functions directly) so that
these tests actually exercise stdout/stderr encoding, which is where the
CLI has broken before on Windows consoles using a non-UTF-8 codepage.
"""

import csv
import json
import subprocess
import sys

import pytest

from bananabench import g_suite, get_all_functions
from bananabench import multiobjective as mo


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "bananabench.cli", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_python_dash_m_bananabench_is_equivalent_to_the_cli_module():
    """python -m bananabench should work the same as python -m bananabench.cli."""
    result = subprocess.run(
        [sys.executable, "-m", "bananabench", "--list"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0
    assert "sphere" in result.stdout


class TestList:
    def test_list_runs_successfully(self):
        result = run_cli("--list")
        assert result.returncode == 0
        assert "sphere" in result.stdout

    def test_list_includes_every_registered_function(self):
        result = run_cli("--list")
        for name in get_all_functions():
            assert name in result.stdout


class TestInfo:
    @pytest.mark.parametrize("name", get_all_functions())
    def test_info_handles_unicode_docstrings(self, name):
        # Regression test: function docstrings contain Unicode math symbols
        # (e.g. ≤, π) that previously raised UnicodeEncodeError on
        # Windows consoles using a legacy (non-UTF-8) codepage.
        result = run_cli("--info", name)
        assert result.returncode == 0, result.stderr
        assert "UnicodeEncodeError" not in result.stderr

    def test_unknown_function_fails_gracefully(self):
        result = run_cli("--info", "not_a_real_function")
        assert result.returncode != 0


class TestMetadata:
    def test_outputs_valid_json(self):
        result = run_cli("--metadata", "sphere")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["default_dimension"] == 10

    def test_unknown_function_fails_gracefully(self):
        result = run_cli("--metadata", "not_a_real_function")
        assert result.returncode != 0


class TestFunctionEvaluation:
    def test_evaluates_single_point(self):
        result = run_cli("--function", "sphere", "--values", "1", "2", "3")
        assert result.returncode == 0
        assert "14" in result.stdout  # 1^2 + 2^2 + 3^2

    def test_requires_values_or_input(self):
        result = run_cli("--function", "sphere")
        assert result.returncode != 0

    def test_values_and_input_together_is_an_error(self, tmp_path):
        csv_path = tmp_path / "points.csv"
        csv_path.write_text("0,0,0\n")
        result = run_cli("--function", "sphere", "--values", "1", "2", "--input", str(csv_path))
        assert result.returncode != 0


class TestBatchEvaluation:
    def test_batch_from_csv_writes_json_output(self, tmp_path):
        input_path = tmp_path / "points.csv"
        output_path = tmp_path / "results.json"
        with open(input_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([0, 0, 0])
            writer.writerow([1, 1, 1])

        result = run_cli(
            "--function", "sphere", "--input", str(input_path), "--output", str(output_path)
        )
        assert result.returncode == 0, result.stderr
        assert output_path.exists()

        data = json.loads(output_path.read_text())
        assert data["function"] == "sphere"
        assert len(data["results"]) == 2
        assert data["results"][0]["result"] == pytest.approx(0.0)
        assert data["results"][1]["result"] == pytest.approx(3.0)

    def test_batch_rejects_non_numeric_row(self, tmp_path):
        input_path = tmp_path / "bad.csv"
        input_path.write_text("0,not_a_number,0\n")
        result = run_cli("--function", "sphere", "--input", str(input_path))
        assert result.returncode != 0


class TestConstrainedSuite:
    def test_list_shows_every_g_function(self):
        result = run_cli("--list", "--suite", "constrained")
        assert result.returncode == 0
        for name in g_suite.get_g_function_list():
            assert name in result.stdout

    def test_scalar_list_mentions_other_suites(self):
        result = run_cli("--list")
        assert "constrained" in result.stdout
        assert "multiobjective" in result.stdout

    def test_info_shows_docstring_and_constraint_counts(self):
        result = run_cli("--info", "g01", "--suite", "constrained")
        assert result.returncode == 0, result.stderr
        assert "Inequality constraints: 9" in result.stdout
        assert "Equality constraints:   0" in result.stdout

    def test_info_unknown_function_fails_gracefully(self):
        result = run_cli("--info", "sphere", "--suite", "constrained")
        assert result.returncode != 0

    def test_metadata_outputs_valid_json(self):
        result = run_cli("--metadata", "g01", "--suite", "constrained")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["suite"] == "constrained"
        assert data["dimension"] == 13
        assert data["n_inequality"] == 9

    def test_evaluates_at_known_optimum(self):
        result = run_cli(
            "--function",
            "g01",
            "--suite",
            "constrained",
            "--values",
            *"1 1 1 1 1 1 1 1 1 3 3 3 1".split(),
        )
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert data["result"]["objective"] == pytest.approx(-15.0)
        assert all(v <= 1e-6 for v in data["result"]["inequality_violations"])
        assert data["result"]["equality_violations"] == []

    def test_batch_from_csv(self, tmp_path):
        input_path = tmp_path / "points.csv"
        with open(input_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([13.0, 0.0])

        result = run_cli("--function", "g06", "--suite", "constrained", "--input", str(input_path))
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert "objective" in data["results"][0]["result"]

    def test_unknown_function_fails_gracefully(self):
        result = run_cli("--function", "g99", "--suite", "constrained", "--values", "1")
        assert result.returncode != 0


class TestMultiobjectiveSuite:
    def test_list_shows_every_zdt_function(self):
        result = run_cli("--list", "--suite", "multiobjective")
        assert result.returncode == 0
        for name in mo.get_mo_function_list():
            assert name in result.stdout
        assert "zdt5" not in result.stdout

    def test_info_shows_docstring_and_metadata(self):
        result = run_cli("--info", "zdt1", "--suite", "multiobjective")
        assert result.returncode == 0, result.stderr
        assert "Objectives:         2" in result.stdout

    def test_metadata_outputs_valid_json(self):
        result = run_cli("--metadata", "zdt1", "--suite", "multiobjective")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["suite"] == "multiobjective"
        assert data["n_objectives"] == 2
        assert len(data["bounds"]) == 30

    def test_evaluate_matches_analytic_front(self):
        # x1=1, x[1:]=0 -> g=1, f2 = 1 - sqrt(f1/g) = 0 for zdt1.
        values = ["1"] + ["0"] * 29
        result = run_cli("--function", "zdt1", "--suite", "multiobjective", "--values", *values)
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        f1, f2 = data["result"]["objectives"]
        assert f1 == pytest.approx(1.0)
        assert f2 == pytest.approx(0.0, abs=1e-9)

    def test_unknown_function_fails_gracefully(self):
        result = run_cli("--function", "zdt5", "--suite", "multiobjective", "--values", "1")
        assert result.returncode != 0
