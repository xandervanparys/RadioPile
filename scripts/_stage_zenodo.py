#!/usr/bin/env python3
"""Stage a built release bundle into two flat Zenodo upload directories.

RadioPile v1.0 is ~238 GB, over Zenodo's per-record ceiling, so it publishes as
two linked records split by artifact type:

    record_a  the aggregated corpus and its sidecars
    record_b  the seven per-dataset artifacts

Zenodo records have no folder structure, so both are flattened. Files are
HARDLINKED from the bundle, costing no additional disk, and nothing under the
bundle is modified.

MD5SUMS.txt is written per record for comparison against what Zenodo reports
after upload. It is deliberately NOT uploaded: Zenodo computes and displays its
own MD5 for every file, and shipping a second copy of the same fact invites the
two disagreeing.

Usage:
    python scripts/_stage_zenodo.py                      # from release/v1.0
    python scripts/_stage_zenodo.py --bundle release/v1.0 --out release/zenodo_upload
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

TEMPLATES = Path("release_templates/zenodo")

RECORD_A_FILES = [
    "aggregated/radiopile_v1.0.h5",
    "aggregated/radiopile_v1.0.h5.config.json",
    "aggregated/radiopile_v1.0.croissant.json",
    "aggregated/aggregation_config.py",
]

# Staged into BOTH records, hardlinked from the same bundle file so the two
# copies cannot drift. validate_datasets.py reports on all eight artifacts in
# one document — the seven per-dataset files and the aggregate — so a reader who
# takes only the per-dataset record would otherwise have no evidence that what
# they downloaded was validated.
SHARED_FILES = [
    "validation_report.json",
]

# Staged into both records from the repository rather than the bundle: the raw
# input checksums are what lets a downloader confirm their own copies of the
# third-party sources match the ones v1.0 was built from. Two of those sources
# are unversioned upstream, so for them this is the only verification path.
FROM_REPO = [
    Path("docs/RAW_INPUT_CHECKSUMS.txt"),
]


def md5_of(p: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.md5()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stage(bundle: Path, out: Path) -> int:
    if not bundle.is_dir():
        sys.exit(f"ERROR: no bundle at {bundle}. Run `make release` first.")
    for t in ("RECORD_A_README.md", "RECORD_A_LICENSES.md",
              "RECORD_B_README.md", "RECORD_B_LICENSES.md"):
        if not (TEMPLATES / t).exists():
            sys.exit(f"ERROR: missing template {TEMPLATES / t}")

    a, b = out / "record_a", out / "record_b"
    for d in (a, b):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # --- Record A -------------------------------------------------------
    for rel in RECORD_A_FILES:
        src = bundle / rel
        if not src.exists():
            sys.exit(f"ERROR: bundle is missing {rel}")
        (a / src.name).hardlink_to(src)

    # --- Shared: one bundle file, hardlinked into both records -----------
    for rel in SHARED_FILES:
        src = bundle / rel
        if not src.exists():
            sys.exit(f"ERROR: bundle is missing {rel}, which both records require.\n"
                     f"       It is written by `make release` (the validate step).\n"
                     f"       Re-run `make release` rather than staging without it —\n"
                     f"       a record that ships no validation evidence is worse than\n"
                     f"       one that fails to stage.")
        for d in (a, b):
            (d / src.name).hardlink_to(src)

    # --- Record B: every per-dataset file, flattened --------------------
    per = bundle / "per_dataset"
    if not per.is_dir():
        sys.exit(f"ERROR: no per_dataset/ in {bundle}")
    seen: dict[str, Path] = {}
    for src in sorted(per.rglob("*")):
        if not src.is_file():
            continue
        if src.name in seen:
            sys.exit(f"ERROR: filename collision when flattening: {src.name}\n"
                     f"       {seen[src.name]}\n       {src}")
        seen[src.name] = src
        (b / src.name).hardlink_to(src)

    # --- Repository files shipped with both records ----------------------
    for src in FROM_REPO:
        if not src.exists():
            sys.exit(f"ERROR: {src} is missing; both records ship it.\n"
                     f"       Generate it with scripts/verify_raw_inputs.py --write")
        for d in (a, b):
            shutil.copy2(src, d / src.name)

    # --- Record-specific docs (written, not hardlinked) -----------------
    shutil.copy2(TEMPLATES / "RECORD_A_README.md", a / "README.md")
    shutil.copy2(TEMPLATES / "RECORD_A_LICENSES.md", a / "LICENSES.md")
    shutil.copy2(TEMPLATES / "RECORD_B_README.md", b / "README.md")
    shutil.copy2(TEMPLATES / "RECORD_B_LICENSES.md", b / "LICENSES.md")

    # --- Checksums ------------------------------------------------------
    total = {}
    staged_md5: dict[str, str] = {}
    for name, d in (("record_a", a), ("record_b", b)):
        files = sorted(p for p in d.iterdir() if p.is_file())
        digests = {p.name: md5_of(p) for p in files}
        staged_md5.update({k: v for k, v in digests.items() if k.endswith(".h5")})
        lines = [f"{digests[p.name]}  {p.name}" for p in files]
        (d / "MD5SUMS.txt").write_text("\n".join(lines) + "\n")
        size = sum(p.stat().st_size for p in files)
        total[name] = (len(files), size)
        print(f"{name}: {len(files)} files to upload, {size / 1e9:.2f} GB")
        print(f"          MD5SUMS.txt written ({len(lines)} entries, not uploaded)")

    # The validator opens the build outputs under data/, and the bundle holds
    # copies of them. A report made with --fingerprint records the MD5 of what it
    # checked, so the two can be compared: a staged file that differs is one the
    # report does not describe.
    report_path = bundle / "validation_report.json"
    if report_path.exists():
        import json as _json
        datasets = _json.loads(report_path.read_text()).get("datasets", {})
        validated = {
            ("radiopile_v1.0.h5" if key == "aggregation" else f"{key}.h5"):
                ((entry.get("file") or {}).get("md5"))
            for key, entry in datasets.items()
        }
        if not any(validated.values()):
            print("\nWARNING: validation_report.json records no file hashes, so the staged")
            print("         files cannot be tied to it. Rebuild the bundle with a validator")
            print("         run that passes --fingerprint (make release does).\n")
        else:
            differing = sorted(
                n for n, digest in staged_md5.items() if validated.get(n) != digest
            )
            if differing:
                sys.exit("ERROR: staged file(s) differ from the ones the validation "
                         "report describes: " + ", ".join(differing) + "\n"
                         "       Re-run `make release` so the bundle and its report "
                         "come from one build.")
            print(f"validated: all {len(staged_md5)} staged HDF5 files match the MD5s in "
                  f"validation_report.json")

    # A bundled file modified after the report was written is no longer the file
    # the report describes. The MD5 comparison above catches that when the report
    # is fingerprinted; this check also covers a report made without --fingerprint.
    report = bundle / "validation_report.json"
    if report.exists():
        import json as _json
        stamp = _json.loads(report.read_text()).get("timestamp", "")
        newer = [q for q in bundle.rglob("*.h5")
                 if q.stat().st_mtime > report.stat().st_mtime]
        if newer:
            print(f"\nWARNING: validation_report.json ({stamp[:19]}) predates "
                  f"{len(newer)} artifact(s) it describes:")
            for q in newer[:4]:
                print(f"  {q.relative_to(bundle)}")
            print("Re-run `make release FORCE=1`, then this script, so the bundle and "
                  "its report come from one build.\n")

    for rel in SHARED_FILES:
        name = Path(rel).name
        ia, ib = (a / name).stat().st_ino, (b / name).stat().st_ino
        state = "same inode" if ia == ib else f"DIFFERENT inodes {ia} vs {ib}"
        print(f"shared:   {name} in both records ({state})")
        if ia != ib:
            sys.exit(f"ERROR: {name} was not hardlinked into both records; "
                     f"the copies could drift.")

    # An HDF5 carrying "[RECORD_A_DOI]" in its attributes would ship a
    # placeholder as though it were a value, and unlike a README nobody would
    # see it before upload. Check what is actually in the files.
    import re as _re
    import h5py as _h5py
    baked = []
    for q in sorted(bundle.rglob("*.h5")):
        with _h5py.File(q, "r") as fh:
            for k, v in fh.attrs.items():
                # Match only the placeholder shape, [UPPER_SNAKE], not any
                # bracketed text — snr_range_db legitimately reads "[-20, 20] dB".
                if isinstance(v, str) and _re.search(r"\[[A-Z][A-Z0-9_]*\]", v):
                    baked.append(f"{q.relative_to(bundle)}  {k}={v[:46]}")
    if baked:
        print("\nWARNING: unresolved placeholders inside HDF5 attributes:")
        for line in baked[:6]:
            print(f"  {line}")
        if len(baked) > 6:
            print(f"  ... and {len(baked) - 6} more")
        print("Resolve with scripts/_set_release_identifiers.py before uploading;")
        print("these are inside the data files, where a reader cannot see them coming.\n")

    placeholders = []
    for d in (a, b):
        for doc in ("README.md", "LICENSES.md"):
            text = (d / doc).read_text()
            if any(k in text for k in ("[RECORD_A_DOI]", "[RECORD_B_DOI]", "[CODE_DOI]")):
                placeholders.append(str(d / doc))
    if placeholders:
        print("\nNOTE: DOI placeholders remain in:")
        for p in placeholders:
            print(f"  {p}")
        print("Run scripts/_set_release_identifiers.py once the DOIs are reserved, then re-stage.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, default=Path("release/v1.0"))
    ap.add_argument("--out", type=Path, default=Path("release/zenodo_upload"))
    return stage(ap.parse_args().bundle, ap.parse_args().out)


if __name__ == "__main__":
    raise SystemExit(main())
