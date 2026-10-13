#!/usr/bin/env python3
"""Upload a staging directory to an EXISTING Zenodo draft deposition.

Deliberately cannot publish. There is no call to the /actions/publish endpoint
anywhere in this file, so a mistake here cannot mint a DOI or make anything
public. Publishing stays a manual step in the Zenodo web UI.

The token is read from the environment and never written to disk, never logged,
and never placed in a URL query string (it is sent as a Bearer header, so it
does not leak into server access logs either).

Usage:
    export ZENODO_TOKEN=...                 # not stored anywhere by this script
    python scripts/_zenodo_upload.py --deposition 1234567 \\
        --dir release/zenodo_upload/record_a [--dry-run]

Resuming is safe: files already present in the draft with a matching MD5 are
skipped, so an interrupted 116 GB upload can be restarted without starting over.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests is required: pip install requests")

ZENODO_API = "https://zenodo.org/api"
SANDBOX_API = "https://sandbox.zenodo.org/api"
CHUNK = 8 * 1024 * 1024


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deposition", required=True,
                    help="numeric id of an existing Zenodo DRAFT deposition")
    ap.add_argument("--dir", required=True, type=Path,
                    help="staging directory whose files are uploaded (flat)")
    ap.add_argument("--sandbox", action="store_true",
                    help="use sandbox.zenodo.org instead of zenodo.org")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be uploaded, contact the API only to read")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        return _fail("ZENODO_TOKEN is not set. Export it in your shell; this script "
                     "never reads a token from a file or an argument.")

    staging: Path = args.dir
    if not staging.is_dir():
        return _fail(f"not a directory: {staging}")

    # MD5SUMS.txt is a local cross-check against what Zenodo reports; it is not
    # part of the record. Zenodo publishes its own MD5 for every file, and a
    # second copy of the same fact inside the record can only ever disagree.
    EXCLUDE = {"MD5SUMS.txt"}
    files = sorted(p for p in staging.iterdir()
                   if p.is_file() and p.name not in EXCLUDE)
    for name in sorted(EXCLUDE):
        if (staging / name).exists():
            print(f"  (excluded from upload: {name})")
    if not files:
        return _fail(f"no files in {staging}")

    api = SANDBOX_API if args.sandbox else ZENODO_API
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    # --- Read the deposition and refuse anything that is not an unpublished draft.
    r = session.get(f"{api}/deposit/depositions/{args.deposition}", timeout=60)
    if r.status_code == 404:
        return _fail(f"deposition {args.deposition} not found (wrong id, or wrong "
                     f"instance — try --sandbox?)")
    if r.status_code in (401, 403):
        return _fail(f"token rejected ({r.status_code}). Check ZENODO_TOKEN has the "
                     f"deposit:write scope and matches this instance.")
    r.raise_for_status()
    dep = r.json()

    if dep.get("submitted") or dep.get("state") == "done":
        return _fail(f"deposition {args.deposition} is already published. This script "
                     f"only writes to unpublished drafts; refusing.")

    bucket = dep.get("links", {}).get("bucket")
    if not bucket:
        return _fail("deposition has no bucket link — it may not be a draft.")

    title = dep.get("title") or dep.get("metadata", {}).get("title") or "(untitled)"
    total = sum(p.stat().st_size for p in files)
    print(f"deposition : {args.deposition}  [{'SANDBOX' if args.sandbox else 'zenodo.org'}]")
    print(f"title      : {title}")
    print(f"state      : {dep.get('state')}  submitted={dep.get('submitted')}")
    print(f"staging    : {staging}  ({len(files)} files, {human(total)})\n")

    existing = {f["filename"]: f.get("checksum", "").replace("md5:", "")
                for f in dep.get("files", [])}

    planned, skipped = [], []
    for p in files:
        digest = md5_of(p)
        if existing.get(p.name) == digest:
            skipped.append(p)
        else:
            planned.append((p, digest))

    for p in skipped:
        print(f"  skip   {p.name}  (already present, md5 matches)")
    for p, digest in planned:
        print(f"  upload {p.name}  {human(p.stat().st_size)}  md5={digest}")

    if args.dry_run:
        print(f"\nDRY RUN — nothing uploaded. "
              f"{len(planned)} file(s), {human(sum(p.stat().st_size for p, _ in planned))} would be sent.")
        return 0

    if not planned:
        print("\nNothing to do; draft already matches the staging directory.")
        return 0

    print()
    for p, digest in planned:
        print(f"uploading {p.name} ({human(p.stat().st_size)}) ...", flush=True)
        with p.open("rb") as fh:
            resp = session.put(f"{bucket}/{p.name}", data=fh, timeout=None)
        if not resp.ok:
            return _fail(f"upload of {p.name} failed: {resp.status_code} {resp.text[:300]}")
        got = resp.json().get("checksum", "").replace("md5:", "")
        if got != digest:
            return _fail(f"checksum mismatch for {p.name}: local {digest}, Zenodo {got}")
        print(f"  ok, md5 confirmed by Zenodo: {got}", flush=True)

    print(f"\nDone. {len(planned)} file(s) uploaded to draft {args.deposition}.")
    print("NOT published — publish manually in the Zenodo web UI when ready.")
    return 0


def _fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
