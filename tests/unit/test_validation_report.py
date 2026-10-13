"""
Tests for the file-fingerprint and provenance parts of scripts/validate_datasets.py.

A validation report is evidence about specific bytes. These tests cover the
pieces that tie a report to those bytes: the digests, the before/after
comparison, and how the result is surfaced in the report.
"""

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import validate_datasets as vd  # noqa: E402
from datasets.registry import DatasetEntry  # noqa: E402


def _entry(path: Path) -> DatasetEntry:
    return DatasetEntry(name="synthetic", processed_h5_path=path)


def _results(path: Path):
    record = vd._check("file exists", True)
    return {"synthetic": (True, [record], _entry(path))}


class TestFingerprintFile:

    def test_digests_match_hashlib(self, tmp_path):
        payload = b"radiopile" * 1000
        target = tmp_path / "a.h5"
        target.write_bytes(payload)

        fp = vd.fingerprint_file(target)

        assert fp["size_bytes"] == len(payload)
        assert fp["sha256"] == hashlib.sha256(payload).hexdigest()
        assert fp["md5"] == hashlib.md5(payload).hexdigest()

    def test_file_larger_than_one_chunk(self, tmp_path, monkeypatch):
        monkeypatch.setattr(vd, "HASH_CHUNK", 7)
        payload = bytes(range(256)) * 3
        target = tmp_path / "a.h5"
        target.write_bytes(payload)

        assert vd.fingerprint_file(target)["sha256"] == hashlib.sha256(payload).hexdigest()

    def test_missing_file_is_none(self, tmp_path):
        assert vd.fingerprint_file(tmp_path / "absent.h5") is None

    def test_fingerprint_files_keeps_keys(self, tmp_path):
        a, b = tmp_path / "a.h5", tmp_path / "b.h5"
        a.write_bytes(b"a")
        b.write_bytes(b"b")

        out = vd.fingerprint_files({"first": a, "second": b, "gone": tmp_path / "c.h5"})

        assert out["first"]["md5"] == hashlib.md5(b"a").hexdigest()
        assert out["second"]["md5"] == hashlib.md5(b"b").hexdigest()
        assert out["gone"] is None


class TestFileRecord:

    def test_identical_passes_are_unchanged(self, tmp_path):
        target = tmp_path / "a.h5"
        target.write_bytes(b"abc")
        fp = vd.fingerprint_file(target)

        record = vd._file_record(target, fp, dict(fp))

        assert record["unchanged_during_validation"] is True
        assert record["sha256"] == fp["sha256"]
        assert record["md5"] == fp["md5"]

    def test_content_change_is_detected(self, tmp_path):
        target = tmp_path / "a.h5"
        target.write_bytes(b"abc")
        before = vd.fingerprint_file(target)
        target.write_bytes(b"abd")

        record = vd._file_record(target, before, vd.fingerprint_file(target))

        assert record["unchanged_during_validation"] is False

    def test_touch_without_content_change_is_detected(self, tmp_path):
        # Same bytes, later mtime: the file was rewritten, so the checks may
        # have seen an intermediate state.
        target = tmp_path / "a.h5"
        target.write_bytes(b"abc")
        before = vd.fingerprint_file(target)
        after = dict(before, mtime_ns=before["mtime_ns"] + 1)

        assert vd._file_record(target, before, after)["unchanged_during_validation"] is False

    def test_missing_pass_is_undetermined(self, tmp_path):
        target = tmp_path / "a.h5"
        record = vd._file_record(target, None, None)

        assert record["unchanged_during_validation"] is None
        assert "sha256" not in record


class TestReport:

    def test_without_fingerprints_report_says_so(self, tmp_path):
        report = vd._build_report(_results(tmp_path / "a.h5"))

        assert report["files_fingerprinted"] is False
        assert report["files_unchanged_during_validation"] is None
        assert report["datasets"]["synthetic"]["file"] is None

    def test_unchanged_files_reported(self, tmp_path):
        target = tmp_path / "a.h5"
        target.write_bytes(b"abc")
        fp = {"synthetic": vd.fingerprint_file(target)}

        report = vd._build_report(_results(target), fingerprints=(fp, dict(fp)))

        assert report["files_fingerprinted"] is True
        assert report["files_unchanged_during_validation"] is True
        assert report["datasets"]["synthetic"]["file"]["sha256"] == fp["synthetic"]["sha256"]

    def test_changed_file_does_not_alter_check_outcome(self, tmp_path):
        target = tmp_path / "a.h5"
        target.write_bytes(b"abc")
        before = {"synthetic": vd.fingerprint_file(target)}
        target.write_bytes(b"abcd")
        after = {"synthetic": vd.fingerprint_file(target)}

        report = vd._build_report(_results(target), fingerprints=(before, after))

        assert report["files_unchanged_during_validation"] is False
        assert report["overall_pass"] is True

    def test_missing_file_is_not_unchanged(self, tmp_path):
        missing = {"synthetic": None}

        report = vd._build_report(_results(tmp_path / "a.h5"), fingerprints=(missing, missing))

        assert report["files_unchanged_during_validation"] is False


class TestProvenance:

    def test_records_validator_and_config_sources(self):
        prov = vd.code_provenance(["sub_ghz", "aggregation"])

        assert prov["validator_version"] == vd.VALIDATOR_VERSION
        for rel in ("scripts/validate_datasets.py", "datasets/registry.py",
                    "pipeline/schemas.py", "datasets/sub_ghz/config.py",
                    "datasets/aggregation/config.py"):
            assert prov["source_sha256"][rel] is not None, rel

    def test_dirty_flag_agrees_with_listed_changes(self):
        prov = vd.code_provenance([])
        if prov["git_dirty"] is None:
            pytest.skip("not running inside a git checkout")

        assert prov["git_dirty"] == bool(prov["uncommitted_changes"])
