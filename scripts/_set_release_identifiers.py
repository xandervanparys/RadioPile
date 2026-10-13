#!/usr/bin/env python3
"""Record reserved Zenodo DOIs in the source tree, and optionally in built files.

The DOIs live in datasets/aggregation/config.py, so a build writes them into the
aggregate's HDF5 root attributes. The root attributes are the only thing a
holder of the bare file can still read once it has been separated from its
README and Croissant descriptor.

Usage
-----
    # 1. Reserve the DOIs on the Zenodo drafts (do not publish).
    # 2. Record them in the source tree:
    python scripts/_set_release_identifiers.py \
        --record-a 10.5281/zenodo.1234567 \
        --record-b 10.5281/zenodo.1234568 \
        --code     10.5281/zenodo.1234569

    # 3. Build (python run.py aggregation), then make release FORCE=1.

--patch-h5 covers one exceptional case: a DOI changed after the build. It writes
the identifiers and license attributes into an existing file under data/ in
place, without rewriting any dataset. It writes only those attributes, with the
values a rebuild from the same commit would write. Patch the files under data/ and then run
`make release`; a file patched inside the bundle no longer matches the
validation report and staging refuses it.

    python scripts/_set_release_identifiers.py --patch-h5 data/aggregation/processed/aggregated.h5

Pass --dry-run to see the replacements without writing anything.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CONFIG = Path("datasets/aggregation/config.py")
IDENTIFIERS = Path("datasets/identifiers.py")

# Files carrying [RECORD_A_DOI] / [RECORD_B_DOI] / [CODE_DOI] placeholders.
PLACEHOLDER_FILES = [
    IDENTIFIERS,
    Path("README.md"),
    Path("CITATION.cff"),
    Path("release_templates/RELEASE_README.md"),
    Path("release_templates/LICENSES.md"),
    Path("release_templates/zenodo/RECORD_A_README.md"),
    Path("release_templates/zenodo/RECORD_A_LICENSES.md"),
    Path("release_templates/zenodo/RECORD_B_README.md"),
    Path("release_templates/zenodo/RECORD_B_LICENSES.md"),
]

DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


def _check(doi: str, label: str) -> str:
    doi = doi.strip()
    if doi.lower().startswith(("http://", "https://")):
        doi = doi.split("doi.org/", 1)[-1]
    if not DOI_RE.match(doi):
        sys.exit(f"ERROR: {label} does not look like a DOI: {doi!r}\n"
                 f"       Expected a bare DOI such as 10.5281/zenodo.1234567")
    return doi


def set_placeholders(a: str, b: str, code: str, dry: bool) -> int:
    mapping = {"[RECORD_A_DOI]": a, "[RECORD_B_DOI]": b, "[CODE_DOI]": code}
    touched = 0
    for path in PLACEHOLDER_FILES:
        if not path.exists():
            print(f"  skip {path} (not found)")
            continue
        text = path.read_text()
        n = sum(text.count(k) for k in mapping)
        if not n:
            continue
        for k, v in mapping.items():
            text = text.replace(k, v)
        if dry:
            print(f"  would update {path}  ({n} replacement(s))")
        else:
            path.write_text(text)
            print(f"  updated {path}  ({n} replacement(s))")
        touched += 1
    return touched


def _load(path: Path):
    """Import a dataset config as a package module.

    Loading it by file path would bypass the package system and its own
    `from pipeline.schemas import ...` would fail, so go through the normal
    import machinery with the repository root on sys.path. Must be run from the
    repository root, which the release sequence does anyway.
    """
    import importlib

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    dotted = ".".join(path.with_suffix("").parts)
    return importlib.import_module(dotted)


def patch_h5(h5: Path, dry: bool) -> int:
    """Write identifiers into one HDF5's root attributes, in place.

    Every file carries the DOI of the record it is published in, pointers to
    its sibling records, and its own license, which is
    the attribute that most needs to travel: a holder of the bare HDF5 has no
    sidecar to read, and must not have to guess what they may do with it.

    The per-dataset licenses differ (NIST is public domain, its_band is
    CC-BY-4.0, the rest are CC-BY-NC-SA-4.0), so each is read from that
    dataset's own config rather than assumed.
    """
    import h5py

    if not h5.exists():
        sys.exit(f"ERROR: no such file: {h5}")

    agg = _load(CONFIG)
    is_aggregate = "radiopile" in h5.name or h5.name == "aggregated.h5"

    if is_aggregate:
        cfg = agg
    else:
        # Per-dataset file: identify it by stem and read its own config.
        name = h5.name[:-3]
        ds_cfg = Path("datasets") / name / "config.py"
        if not ds_cfg.exists():
            sys.exit(f"ERROR: no config for {name} at {ds_cfg}. Per-dataset files "
                     f"must be named <registry key>.h5.")
        cfg = _load(ds_cfg)
    want = {key: cfg.GLOBAL_METADATA[key]
            for key in ("doi", "related_identifiers", "license", "license_note")}

    # Never write an unresolved placeholder into a file that may be published.
    # A baked-in "[RECORD_A_DOI]" is worse than an absent attribute: it looks
    # like a value. Licenses are known now; DOI-derived attributes wait.
    deferred = {k: v for k, v in want.items()
                if re.search(r"\[[A-Z][A-Z0-9_]*\]", str(v))}
    want = {k: v for k, v in want.items() if k not in deferred}

    with h5py.File(h5, "r") as f:
        before = {k: f.attrs.get(k) for k in want}
    print(f"  {h5}")
    for k, v in want.items():
        cur = before[k]
        mark = "=" if str(cur) == str(v) else "->"
        print(f"    {k:20s} {str(cur)[:38]!r} {mark} {str(v)[:48]!r}")
    for k in deferred:
        print(f"    {k:20s} DEFERRED — still contains a placeholder; "
              f"re-run after the DOIs are reserved")
    if not want:
        print("  nothing to write yet.")
        return 0
    if dry:
        print("  DRY RUN — nothing written.")
        return 0

    size_before = h5.stat().st_size
    with h5py.File(h5, "r+") as f:
        for k, v in want.items():
            f.attrs[k] = v
    size_after = h5.stat().st_size
    with h5py.File(h5, "r") as f:
        for k, v in want.items():
            assert f.attrs[k] == v, f"{k} did not persist"
    print(f"  written in place. size {size_before:,} -> {size_after:,} "
          f"(+{size_after - size_before} B of header space; datasets untouched)")
    print("  NOTE: the file's MD5 has changed. Run `make release FORCE=1` and then")
    print("        scripts/_stage_zenodo.py so the bundle, its validation report and")
    print("        both records describe this file.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record-a", help="DOI reserved for the aggregated record")
    ap.add_argument("--record-b", help="DOI reserved for the per-dataset record")
    ap.add_argument("--code", help="DOI for the tagged source release")
    ap.add_argument("--patch-h5", type=Path,
                    help="write the recorded DOIs and license attributes into this "
                         "HDF5's root attributes (a file under data/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would change without writing anything")
    args = ap.parse_args()

    if args.patch_h5:
        return patch_h5(args.patch_h5, args.dry_run)

    if not all([args.record_a, args.record_b, args.code]):
        ap.error("--record-a, --record-b and --code are all required "
                 "(or use --patch-h5)")

    a = _check(args.record_a, "--record-a")
    b = _check(args.record_b, "--record-b")
    c = _check(args.code, "--code")
    print(f"record A : {a}\nrecord B : {b}\ncode     : {c}\n")
    n = set_placeholders(a, b, c, args.dry_run)
    print(f"\n{n} file(s) {'would be ' if args.dry_run else ''}updated.")
    if not args.dry_run:
        print("Next: python run.py aggregation, then make release FORCE=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
