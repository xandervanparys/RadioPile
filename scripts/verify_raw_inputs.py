#!/usr/bin/env python3
"""Verify the raw source files against the checksums recorded for v1.0.

Purpose
-------
RadioPile is built from seven third-party datasets that are not redistributed
with it. A rebuilder has to fetch them from their original homes, and two of
those homes are not versioned: the ITS Band captures live in a personal GitLab
repository, and the NIST record contains five files of which the pipeline uses
three. Downloading "the same dataset" is therefore not the same as having the
same bytes.

This records the SHA-256 of every raw file the v1.0 corpus was built from, so
that question can be settled instead of assumed. Without it, a failed
reproduction cannot be told apart from a changed input.

Usage
-----
    python scripts/verify_raw_inputs.py                  # verify everything
    python scripts/verify_raw_inputs.py --dataset sub_ghz
    python scripts/verify_raw_inputs.py --write          # regenerate the manifest

Raw data elsewhere — 156 GB rarely lives inside a clone:

    python scripts/verify_raw_inputs.py --data-root /mnt/scratch/radiopile-raw

Manifest without a clone. Both Zenodo records ship a copy, so it can be used
next to the data it describes:

    python scripts/verify_raw_inputs.py --manifest ./RAW_INPUT_CHECKSUMS.txt \
        --data-root .

The manifest is sha256sum-compatible, so it also works without this script:

    sha256sum -c docs/RAW_INPUT_CHECKSUMS.txt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

DEFAULT_MANIFEST = Path("docs/RAW_INPUT_CHECKSUMS.txt")
DEFAULT_DATA_ROOT = Path("data")
CHUNK = 8 << 20

# Manifest paths are always recorded repo-relative ("data/<name>/raw/..."), so
# the file stays portable and means the same thing wherever it travels — it
# ships inside both Zenodo records, where there is no repository around it.
# --data-root substitutes for the leading "data" when resolving them, which is
# what lets someone verify a 156 GB download sitting on a scratch mount.
RECORDED_PREFIX = "data"


def resolve(rel: str, data_root: Path) -> Path:
    """Map a recorded repo-relative path onto the local data root."""
    parts = Path(rel).parts
    if parts and parts[0] == RECORDED_PREFIX:
        return data_root.joinpath(*parts[1:])
    return Path(rel)


def record_path(p: Path, data_root: Path) -> str:
    """Inverse of resolve: the repo-relative form to write into the manifest."""
    rel = p.relative_to(data_root)
    return (Path(RECORDED_PREFIX) / rel).as_posix()

# Image files found under raw/ are plots produced locally, not upstream inputs.
# Hashing them would hand a rebuilder failures for files they were never meant
# to have. Dotfiles (.gitkeep, .gitignore) are skipped for the same reason.
NOT_INPUTS = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}

# Directories under raw/ holding files that are not v1.0 inputs. The guiding
# rule is to record what a rebuilder will actually have: the two unused NIST
# .h5 files stay in, because the NIST record ships all five and recording them
# documents which were used; its_band/previous_raw_files/ is left out, because
# a rebuilder following DATASETS.md fetches only the six 25 Msps files and
# would otherwise face 23 spurious MISSING entries.
SKIP_DIRS = {"previous_raw_files"}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            h.update(b)
    return h.hexdigest()


def _upstream_digest(p: Path) -> str | None:
    """Read the publisher's own SHA-256 for a file, if it shipped one.

    NIST distributes a <file>.sha256 beside each of its data files, containing
    a bare 64-character digest (no filename, and not always newline
    terminated). No other source in the corpus publishes checksums, which is
    why this manifest is the only verification path for the other six.
    """
    side = p.with_name(p.name + ".sha256")
    if not side.exists():
        return None
    text = side.read_text(errors="ignore").strip()
    token = text.split()[0] if text else ""
    return token.lower() if len(token) == 64 and all(
        c in "0123456789abcdef" for c in token.lower()) else None


def check_upstream(dataset: str | None, manifest: Path, data_root: Path) -> int:
    """Compare manifest digests against the publishers' own, where published.

    Reads the manifest rather than re-hashing: the local digest is already recorded,
    so this is a string comparison. Agreement means two independent
    attestations that the input is intact — this one and the publisher's — which is
    a stronger claim than either alone.
    """
    if not manifest.exists():
        return _fail(f"no manifest at {manifest}; run --write first")
    recorded = {}
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            digest, _, rel = line.partition("  ")
            recorded[rel] = digest

    agree = disagree = 0
    missing_locally = []
    for rel, ours in sorted(recorded.items()):
        if dataset and not rel.startswith(f"data/{dataset}/"):
            continue
        p = resolve(rel, data_root)
        if p.name.endswith(".sha256"):
            continue
        up = _upstream_digest(p)
        if up is None:
            continue
        if not p.exists():
            missing_locally.append(rel)
            continue
        if up == ours.lower():
            print(f"  ✓ {rel}")
            agree += 1
        else:
            print(f"  ✗ {rel}\n      ours     {ours}\n      upstream {up}")
            disagree += 1

    if not agree and not disagree:
        print("  no upstream checksums found for the selected files.")
        return 0
    print(f"\n{agree} file(s) agree with the publisher, {disagree} disagree.")
    for rel in missing_locally:
        print(f"  (not checked, absent locally: {rel})")
    if disagree:
        print("A disagreement means our manifest and the publisher describe "
              "different bytes. Resolve before relying on either.")
    return 1 if disagree else 0


def raw_files(dataset: str | None, data_root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for d in sorted(data_root.iterdir()):
        if not (d / "raw").is_dir() or d.name == "_template":
            continue
        if dataset and d.name != dataset:
            continue
        files = sorted(p for p in (d / "raw").rglob("*")
                       if p.is_file()
                       and p.suffix.lower() not in NOT_INPUTS
                       and not p.name.startswith(".")
                       and not SKIP_DIRS.intersection(p.parts))
        if files:
            out[d.name] = files
    return out


def write(dataset: str | None, manifest: Path, data_root: Path) -> int:
    groups = raw_files(dataset, data_root)
    if not groups:
        return _fail(f"no raw files found under {data_root}/*/raw/")
    if dataset:
        return _fail("--write regenerates the whole manifest; omit --dataset")

    lines = []
    total_files = total_bytes = 0
    for name, files in groups.items():
        size = sum(p.stat().st_size for p in files)
        lines.append(f"# {name}: {len(files)} files, {size / 1e9:.2f} GB")
        for p in files:
            print(f"  {p}", flush=True)
            lines.append(f"{sha256_of(p)}  {record_path(p, data_root)}")
        lines.append("")
        total_files += len(files)
        total_bytes += size
    lines = manifest_header(total_files, total_bytes) + lines
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines))
    print(f"\nwrote {manifest}: {total_files} files, {total_bytes / 1e9:.2f} GB")
    return 0


def manifest_header(total_files: int, total_bytes: int) -> list[str]:
    """Comment block at the top of the manifest, ending in a blank line."""
    return [
        "# RadioPile v1.0 — SHA-256 of every raw input file.",
        "#",
        "# These are the exact bytes the v1.0 corpus was built from. The source",
        "# datasets are third-party and not redistributed here; see",
        "# docs/DATASETS.md for where each comes from. Two sources are not",
        "# versioned upstream, so matching these hashes is the only way to confirm",
        "# a download is the same input rather than merely the same dataset.",
        "#",
        "# Scope: what a rebuilder will actually have. Image files under raw/ are",
        "# locally produced plots, not inputs, and are excluded; so is",
        "# its_band/previous_raw_files/, the superseded 10 Msps exploration set.",
        "#",
        "# Two NIST files (2_4ghz_outdoor.h5, 2_4ghz_bluetooth.h5) are recorded here",
        "# but not built into v1.0. See DATASETS.md, nist_wifi_bluetooth section.",
        "#",
        "# NIST publishes its own .sha256 for 11 of its 13 upstream files; these agree",
        "# with all 11. Compare them with --upstream (no hashing).",
        "#",
        f"# Total: {total_files} files, {total_bytes / 1e9:.2f} GB",
        "#",
        "# Verify:  python scripts/verify_raw_inputs.py",
        "#     or:  sha256sum -c docs/RAW_INPUT_CHECKSUMS.txt",
        "#",
        "# Regenerate (only with the original inputs present):",
        "#          python scripts/verify_raw_inputs.py --write",
        "",
    ]


def verify(dataset: str | None, manifest: Path, data_root: Path) -> int:
    if not manifest.exists():
        return _fail(f"no manifest at {manifest}")
    recorded: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        recorded[rel] = digest

    groups = raw_files(dataset, data_root)
    present = {record_path(p, data_root) for files in groups.values() for p in files}
    expected = {r for r in recorded
                if not dataset or r.startswith(f"data/{dataset}/")}

    ok = bad = 0
    for rel in sorted(expected):
        p = resolve(rel, data_root)
        if not p.exists():
            print(f"  MISSING   {rel}  (looked in {p})")
            bad += 1
            continue
        got = sha256_of(p)
        if got == recorded[rel]:
            ok += 1
        else:
            print(f"  MISMATCH  {rel}\n            recorded {recorded[rel]}\n            found    {got}")
            bad += 1

    extra = sorted(present - expected)
    for rel in extra:
        print(f"  UNRECORDED {rel}  (present locally, not in the manifest)")

    print(f"\n{ok} file(s) match, {bad} problem(s), {len(extra)} unrecorded.")
    print("\nAgainst the publishers' own checksums:")
    bad += check_upstream(dataset, manifest, data_root)
    if bad:
        print("A mismatch means your input differs from what v1.0 was built from.")
        print("Rebuilding from it will not reproduce the published corpus.")
    return 1 if bad else 0


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="verify only this dataset's raw files")
    ap.add_argument("--write", action="store_true",
                    help="regenerate the manifest from the local raw files")
    ap.add_argument("--upstream", action="store_true",
                    help="only compare the manifest against publishers' own "
                         "checksums (no hashing)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST,
                    help=f"checksum file to use (default {DEFAULT_MANIFEST}). "
                         f"Both Zenodo records ship a copy, so this can point "
                         f"at it directly without cloning the repository.")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    dest="data_root",
                    help=f"directory holding the <dataset>/raw/ trees "
                         f"(default {DEFAULT_DATA_ROOT}). Recorded paths stay "
                         f"repo-relative; this replaces their leading "
                         f"'{RECORDED_PREFIX}' when resolving them.")
    a = ap.parse_args()
    if not a.data_root.is_dir():
        return _fail(f"--data-root {a.data_root} is not a directory")
    if a.write:
        return write(a.dataset, a.manifest, a.data_root)
    if a.upstream:
        return check_upstream(a.dataset, a.manifest, a.data_root)
    return verify(a.dataset, a.manifest, a.data_root)


if __name__ == "__main__":
    raise SystemExit(main())
