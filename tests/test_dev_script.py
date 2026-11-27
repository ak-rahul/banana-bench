"""
Tests for scripts/dev.py's CI logic: which jobs a change set needs (ci.yml) and
whether a version is a new release (publish.yml). A bug in either one wastes CI
or, worse, skips tests or publishes the wrong version, and a PyPI upload can't be
undone. scripts/ isn't a package, so the module is loaded from its path.
"""

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "dev", Path(__file__).resolve().parent.parent / "scripts" / "dev.py"
)
dev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dev)


class TestCiDiffBase:
    def test_pull_request_diffs_against_its_base(self):
        assert dev.ci_diff_base("pull_request", pr_base="abc123") == "abc123"

    def test_push_diffs_against_previous_tip(self):
        assert dev.ci_diff_base("push", before="def456") == "def456"

    def test_new_branch_push_has_no_base(self):
        assert dev.ci_diff_base("push", before="0" * 40) is None

    @pytest.mark.parametrize("event", ["workflow_dispatch", "schedule"])
    def test_other_events_run_everything(self, event):
        assert dev.ci_diff_base(event, pr_base="abc", before="def") is None

    def test_missing_shas_run_everything(self):
        assert dev.ci_diff_base("pull_request") is None
        assert dev.ci_diff_base("push") is None


class TestCiNeededJobs:
    @pytest.mark.parametrize(
        "files",
        [
            ["README.md"],
            ["docs/USER_GUIDE.md", "docs/conf.py", "CHANGELOG.md"],
            ["examples/14_center_bias.py"],
            ["docs/api/index.md"],
            ["CITATION.cff", ".readthedocs.yaml", "justfile"],
            [".github/dependabot.yml", ".github/workflows/publish.yml"],
            [],
        ],
    )
    def test_docs_only_changes_skip_both_jobs(self, files):
        assert dev.ci_needed_jobs(files) == {"tests": False, "quality": False}

    @pytest.mark.parametrize(
        "files",
        [
            ["bananabench/analysis.py"],
            ["tests/test_analysis.py"],
            ["pyproject.toml"],
            ["scripts/dev.py"],  # tested by this file, so it needs the tests
            [".github/workflows/ci.yml"],
            [".github/actions/setup-python-env/action.yml"],
            ["README.md", "bananabench/utils.py"],  # one code file is enough
        ],
    )
    def test_code_changes_need_both_jobs(self, files):
        assert dev.ci_needed_jobs(files) == {"tests": True, "quality": True}


class TestReleaseDecision:
    @pytest.fixture(autouse=True)
    def _needs_packaging(self):
        pytest.importorskip("packaging")

    def test_existing_tag_is_not_released_again(self):
        assert dev.release_decision("0.4.0", ["v0.3.0", "v0.4.0"]) == (False, "0.4.0")

    def test_higher_version_is_released(self):
        assert dev.release_decision("0.4.0", ["v0.2.0", "v0.3.0"]) == (True, "0.3.0")

    def test_first_release(self):
        assert dev.release_decision("0.1.0", []) == (True, None)

    def test_lower_version_fails(self):
        with pytest.raises(SystemExit, match="not higher"):
            dev.release_decision("0.3.1", ["v0.4.0"])

    def test_final_ranks_above_its_release_candidate(self):
        # Plain `sort -V` ranks 0.4.0rc1 above 0.4.0; PEP 440 doesn't.
        assert dev.release_decision("0.4.0", ["v0.4.0rc1"]) == (True, "0.4.0rc1")
        with pytest.raises(SystemExit):
            dev.release_decision("0.4.0rc1", ["v0.4.0"])

    def test_non_version_tags_are_ignored(self):
        assert dev.release_decision("0.4.0", ["v0.3.0", "vnext", "latest"]) == (True, "0.3.0")


def test_outputs_go_to_stdout_and_github_output(tmp_path, monkeypatch, capsys):
    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    dev._write_outputs({"tests": "true", "quality": "false"})
    assert out.read_text() == "tests=true\nquality=false\n"
    assert capsys.readouterr().out == "tests=true\nquality=false\n"


def test_ci_changes_runs_everything_without_a_base(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    assert dev.main(["ci-changes", "--event", "workflow_dispatch", "--head", "HEAD"]) == 0
    assert "tests=true\nquality=true\n" in capsys.readouterr().out


def test_ci_changes_diffs_real_commits(tmp_path, monkeypatch, capsys):
    # HEAD against itself: no changed files, so nothing is needed.
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    dev.main(["ci-changes", "--event", "push", "--before", "HEAD", "--head", "HEAD"])
    assert "tests=false\nquality=false\n" in capsys.readouterr().out


def test_ci_changes_runs_everything_when_diff_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    dev.main(["ci-changes", "--event", "push", "--before", "f" * 40, "--head", "HEAD"])
    out = capsys.readouterr().out
    assert "Could not diff" in out
    assert "tests=true\nquality=true\n" in out
