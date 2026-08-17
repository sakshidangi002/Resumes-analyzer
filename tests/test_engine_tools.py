"""Tool-layer tests: the sandbox, the test runner, git safety, AST checks.

These are the security boundary of the engine. If path validation or the command
allowlist is wrong, every node above them is unsafe regardless of how careful it is.
"""
import pytest

from graph_engine.config import EngineConfig
from graph_engine.tools import ast_checks, git_tools
from graph_engine.tools.repository import PathNotAllowed, Repository
from graph_engine.tools.test_runner import TestRunner, UnsafeTestTarget


@pytest.fixture
def repo(tmp_path):
    """A miniature repository with an in-scope package and an out-of-scope one."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "secret.py").write_text("token = 'abc'\n", encoding="utf-8")
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "vendored.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "pkg" / "alembic").mkdir()
    (tmp_path / "pkg" / "alembic" / "001_x.py").write_text("z = 3\n", encoding="utf-8")

    config = EngineConfig(goal="g", scope="pkg", repo_root=tmp_path, apply_fixes=True)
    return Repository(config=config, run_id="testrun")


# --------------------------------------------------------------------------
# path sandbox
# --------------------------------------------------------------------------
def test_lists_only_python_files_in_scope(repo):
    files = repo.list_python_files()
    assert "pkg/mod.py" in files
    assert "other/secret.py" not in files


def test_excluded_directories_are_never_listed(repo):
    assert not any(".venv" in f for f in repo.list_python_files("."))


@pytest.mark.parametrize("bad", ["../outside.py", "../../etc/passwd", "pkg/../../escape.py"])
def test_path_traversal_is_rejected(repo, bad):
    with pytest.raises(PathNotAllowed):
        repo.read(bad)


def test_reading_an_excluded_directory_is_rejected(repo):
    with pytest.raises(PathNotAllowed):
        repo.read(".venv/lib/vendored.py")


def test_write_outside_scope_is_rejected(repo):
    with pytest.raises(PathNotAllowed, match="outside the run scope"):
        repo.write("other/secret.py", "hacked = True\n", reason="test")


def test_write_to_a_protected_path_is_rejected(repo):
    """Migrations must never be edited autonomously, even inside the scope."""
    with pytest.raises(PathNotAllowed, match="protected"):
        repo.write("pkg/alembic/001_x.py", "z = 4\n", reason="test")


def test_write_to_a_non_python_file_is_rejected(repo, tmp_path):
    (tmp_path / "pkg" / "notes.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(PathNotAllowed, match="not a Python source file"):
        repo.write("pkg/notes.txt", "bye", reason="test")


# --------------------------------------------------------------------------
# multiple scopes (whole-application runs)
# --------------------------------------------------------------------------
@pytest.fixture
def multi_repo(tmp_path):
    """Two in-scope roots plus one that is not declared."""
    for name in ("pkg", "other", "untouched"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "mod.py").write_text("x = 1\n", encoding="utf-8")

    config = EngineConfig(
        goal="g", scopes=("pkg", "other"), repo_root=tmp_path, apply_fixes=True
    )
    return Repository(config=config, run_id="multi")


def test_multiple_scopes_are_all_listed(multi_repo):
    files = multi_repo.list_python_files()
    assert "pkg/mod.py" in files
    assert "other/mod.py" in files
    assert "untouched/mod.py" not in files


def test_writes_are_allowed_in_any_declared_scope(multi_repo):
    multi_repo.write("pkg/mod.py", "x = 2\n", reason="test")
    multi_repo.write("other/mod.py", "x = 3\n", reason="test")
    assert sorted(multi_repo.changed_files()) == ["other/mod.py", "pkg/mod.py"]


def test_writes_outside_every_scope_are_still_rejected(multi_repo):
    with pytest.raises(PathNotAllowed, match="outside the run scope"):
        multi_repo.write("untouched/mod.py", "x = 9\n", reason="test")


def test_overlapping_scopes_do_not_duplicate_files(tmp_path):
    """Otherwise every finding in the overlap would be reported twice."""
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "sub" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    config = EngineConfig(goal="g", scopes=("pkg", "pkg/sub"), repo_root=tmp_path)
    files = Repository(config=config, run_id="o").list_python_files()
    assert files == ["pkg/sub/mod.py"]


def test_a_config_needs_at_least_one_scope():
    with pytest.raises(ValueError, match="scope"):
        EngineConfig(goal="g")


# --------------------------------------------------------------------------
# backup / restore
# --------------------------------------------------------------------------
def test_write_backs_up_the_original_and_restore_is_byte_exact(repo):
    original = repo.read("pkg/mod.py")
    repo.write("pkg/mod.py", "x = 999\n", reason="test")
    assert repo.read("pkg/mod.py") == "x = 999\n"
    assert repo.changed_files() == ["pkg/mod.py"]

    repo.restore_all()
    assert repo.read("pkg/mod.py") == original


def test_restore_returns_the_first_original_after_several_writes(repo):
    original = repo.read("pkg/mod.py")
    repo.write("pkg/mod.py", "x = 2\n", reason="first")
    repo.write("pkg/mod.py", "x = 3\n", reason="second")
    repo.restore_all()
    # The backup is taken once, on the first write — not overwritten by the second.
    assert repo.read("pkg/mod.py") == original


# --------------------------------------------------------------------------
# test runner
# --------------------------------------------------------------------------
@pytest.mark.parametrize("target", ["-x", "--exec=rm", "-p", ""])
def test_flag_like_test_targets_are_rejected(tmp_path, target):
    runner = TestRunner(repo_root=tmp_path)
    with pytest.raises(UnsafeTestTarget):
        runner._validate(target)


def test_test_target_outside_the_repo_is_rejected(tmp_path):
    runner = TestRunner(repo_root=tmp_path)
    with pytest.raises(UnsafeTestTarget):
        runner._validate("../../elsewhere/test_x.py")


def test_pytest_output_is_parsed_into_structured_results():
    stdout = (
        "FAILED tests/test_a.py::test_one - AssertionError: expected True\n"
        "FAILED tests/test_b.py::test_two - ValueError: bad\n"
        "2 failed, 429 passed, 3 skipped in 24.27s\n"
    )
    result = TestRunner._parse(stdout, "", 1, 24.27, "targeted", ["py", "-m", "pytest", "tests"], False)

    assert result["ok"] is False
    assert result["passed"] == 429
    assert result["failed"] == 2
    assert result["skipped"] == 3
    assert [f["nodeid"] for f in result["failures"]] == [
        "tests/test_a.py::test_one", "tests/test_b.py::test_two",
    ]
    assert "AssertionError" in result["failures"][0]["message"]


def test_a_clean_run_is_ok():
    result = TestRunner._parse("429 passed in 24s\n", "", 0, 24.0, "regression", ["pytest"], False)
    assert result["ok"] is True and result["failed"] == 0


def test_a_timeout_is_neither_pass_nor_fail():
    result = TestRunner._parse("", "", -1, 300.0, "targeted", ["pytest"], True)
    assert result["timed_out"] is True
    assert result["ok"] is False


# --------------------------------------------------------------------------
# git safety
# --------------------------------------------------------------------------
@pytest.mark.parametrize("subcommand", ["checkout", "reset", "stash", "commit", "push", "clean"])
def test_mutating_git_subcommands_are_blocked(tmp_path, subcommand):
    """The engine must never be able to destroy uncommitted work."""
    with pytest.raises(git_tools.GitCommandNotAllowed):
        git_tools._git(tmp_path, subcommand, "--hard")


def test_untracked_files_are_not_treated_as_dirty(tmp_path):
    dirty = {"a.py": "??", "b.py": "M"}
    assert git_tools.is_file_dirty(tmp_path, "a.py", dirty) is False
    assert git_tools.is_file_dirty(tmp_path, "b.py", dirty) is True


# --------------------------------------------------------------------------
# AST checks
# --------------------------------------------------------------------------
def test_plain_none_equality_is_reported_without_query_context():
    findings = ast_checks.analyze_file("m.py", "if x == None:\n    pass\n")
    hits = [f for f in findings if f["code"] == ast_checks.CMP_NONE]
    assert len(hits) == 1
    assert hits[0]["in_query_context"] is False


def test_none_equality_inside_a_sqlalchemy_filter_is_flagged_as_query_context():
    """The fact that lets triage dismiss it instead of breaking the query."""
    source = "rows = db.query(R).filter(R.deleted_at == None).all()\n"
    hits = [f for f in ast_checks.analyze_file("m.py", source)
            if f["code"] == ast_checks.CMP_NONE]
    assert len(hits) == 1
    assert hits[0]["in_query_context"] is True


def test_swallowed_exception_is_reported():
    source = "try:\n    go()\nexcept Exception:\n    pass\n"
    codes = [f["code"] for f in ast_checks.analyze_file("m.py", source)]
    assert ast_checks.EXCEPT_PASS in codes


def test_a_syntax_error_is_a_finding_not_a_crash():
    findings = ast_checks.analyze_file("m.py", "def broken(:\n")
    assert findings[0]["code"] == "E999"
    assert findings[0]["severity"] == "high"


def test_a_file_with_a_utf8_bom_is_not_reported_as_a_syntax_error(tmp_path):
    """Several files in this repo carry a BOM. Python imports them fine, so
    reading them with plain utf-8 would invent an E999 on healthy code."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "bom.py").write_bytes(b"\xef\xbb\xbfx = 1\n")
    config = EngineConfig(goal="g", scope="pkg", repo_root=tmp_path)
    repo = Repository(config=config, run_id="bom")

    source = repo.read("pkg/bom.py")
    assert not source.startswith("﻿")
    assert ast_checks.analyze_file("pkg/bom.py", source) == []
