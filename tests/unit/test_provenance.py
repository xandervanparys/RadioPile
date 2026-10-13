"""
Tests for build provenance capture (pipeline/io.py :: _detect_git_commit).

A config snapshot that names a commit while the working tree holds uncommitted
changes does not reproduce the artifact. These tests pin the dirty-tree suffix
that makes that condition visible.
"""

import subprocess
from unittest import mock

import pytest

from pipeline import io as pipeline_io


def _completed(stdout: str = "", returncode: int = 0):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr="")


class TestDetectGitCommit:

    def test_clean_tree_returns_bare_sha(self):
        """Empty `git status --porcelain` means no suffix."""
        with mock.patch.object(pipeline_io, "_git") as g:
            g.side_effect = [_completed("abc123def456\n"), _completed("")]
            assert pipeline_io._detect_git_commit() == "abc123def456"

    def test_dirty_tree_is_flagged(self):
        """Modified tracked files must produce a -dirty suffix."""
        with mock.patch.object(pipeline_io, "_git") as g:
            g.side_effect = [_completed("abc123def456\n"),
                             _completed(" M datasets/aggregation/config.py\n")]
            assert pipeline_io._detect_git_commit() == "abc123def456-dirty"

    def test_untracked_files_do_not_count_as_dirty(self):
        """Ignored and untracked paths (data/, release/) must not flag a build."""
        with mock.patch.object(pipeline_io, "_git") as g:
            g.side_effect = [_completed("abc123def456\n"), _completed("")]
            pipeline_io._detect_git_commit()
        # The status call must suppress untracked files, or every build with a
        # populated data/ directory would read as dirty.
        assert "--untracked-files=no" in g.call_args_list[1].args

    def test_status_failure_does_not_invent_dirtiness(self):
        """If status cannot run, fall back to the bare SHA rather than guessing."""
        with mock.patch.object(pipeline_io, "_git") as g:
            g.side_effect = [_completed("abc123def456\n"), _completed("", returncode=128)]
            assert pipeline_io._detect_git_commit() == "abc123def456"

    def test_no_git_returns_none(self):
        with mock.patch.object(pipeline_io, "_git", return_value=None):
            assert pipeline_io._detect_git_commit() is None

    def test_rev_parse_failure_returns_none(self):
        with mock.patch.object(pipeline_io, "_git") as g:
            g.side_effect = [_completed("", returncode=128)]
            assert pipeline_io._detect_git_commit() is None

    def test_empty_sha_returns_none(self):
        with mock.patch.object(pipeline_io, "_git") as g:
            g.side_effect = [_completed("\n")]
            assert pipeline_io._detect_git_commit() is None


class TestGitHelper:

    def test_missing_git_binary_returns_none(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            assert pipeline_io._git("rev-parse", "HEAD") is None

    def test_timeout_returns_none(self):
        with mock.patch("subprocess.run",
                        side_effect=subprocess.TimeoutExpired(cmd="git", timeout=5)):
            assert pipeline_io._git("status") is None
