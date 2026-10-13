"""
Tests for the raw-input checksum manifest, docs/RAW_INPUT_CHECKSUMS.txt.

The manifest ships in both Zenodo records. It is written by
`verify_raw_inputs.py --write`, which needs every raw input present, so a
regeneration is rare and its output is committed. These tests keep the
committed file and its generator from drifting apart without hashing anything.
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import verify_raw_inputs as vri  # noqa: E402

MANIFEST = REPO / "docs" / "RAW_INPUT_CHECKSUMS.txt"
TOTAL_LINE = re.compile(r"^# Total: (\d+) files, (\d+\.\d\d) GB$")
ENTRY_LINE = re.compile(r"^[0-9a-f]{64}  data/[a-z0-9_]+/raw/\S.*$")


def _manifest_lines() -> list[str]:
    return MANIFEST.read_text().split("\n")


def _recorded_total() -> tuple[int, float]:
    for line in _manifest_lines():
        match = TOTAL_LINE.match(line)
        if match:
            return int(match.group(1)), float(match.group(2))
    raise AssertionError("manifest has no '# Total:' line")


class TestManifestHeader:

    def test_generator_reproduces_the_committed_header(self):
        n_files, gigabytes = _recorded_total()
        header = vri.manifest_header(n_files, int(gigabytes * 1e9))

        assert _manifest_lines()[: len(header)] == header

    def test_header_is_comments_closed_by_a_blank_line(self):
        header = vri.manifest_header(1, 1)

        assert header[-1] == ""
        assert all(line.startswith("#") for line in header[:-1])


class TestManifestEntries:

    def test_every_entry_is_a_sha256_and_a_raw_path(self):
        entries = [line for line in _manifest_lines() if line and not line.startswith("#")]

        assert entries
        for line in entries:
            assert ENTRY_LINE.match(line), line

    def test_entry_count_matches_the_recorded_total(self):
        entries = [line for line in _manifest_lines() if line and not line.startswith("#")]

        assert len(entries) == _recorded_total()[0]

    def test_no_path_is_listed_twice(self):
        paths = [line.split("  ", 1)[1] for line in _manifest_lines()
                 if line and not line.startswith("#")]

        assert len(paths) == len(set(paths))
