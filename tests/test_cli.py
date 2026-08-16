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

from bananabench import get_all_functions


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
