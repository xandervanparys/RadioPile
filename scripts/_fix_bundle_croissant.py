"""Rewrite Croissant distribution references inside a release bundle.

The Croissant generators run at build time, so each descriptor records the
path the HDF5 had *in the working tree* (``aggregated.h5``,
``data/<name>/processed/<name>.h5``). Those references dangle once the file is
copied into a bundle and renamed, so a consumer resolving ``contentUrl``
relative to the descriptor finds nothing.

This rewrites ``distribution[].name`` and ``distribution[].contentUrl`` in each
bundled ``*.croissant.json`` to the bare filename of the sibling ``.h5`` that
sits next to it, so every descriptor resolves relative to its own location.

Only the bundled copies are touched; in-tree descriptors stay as generated. The
file's ``@id``, which the descriptor's fields refer to, and its ``sha256`` are
left as they are: the bundled file is a byte-for-byte copy.

Usage:
    python scripts/_fix_bundle_croissant.py <release_dir>
"""

import argparse
import json
import sys
from pathlib import Path


def fix_descriptor(croissant_path: Path) -> tuple[bool, str]:
    """Point one descriptor's distribution entries at its sibling HDF5.

    Returns (changed, message).
    """
    siblings = sorted(croissant_path.parent.glob("*.h5"))
    if len(siblings) != 1:
        return False, (
            f"expected exactly 1 sibling .h5, found {len(siblings)}"
            f" — leaving {croissant_path.name} untouched"
        )

    target = siblings[0].name

    with open(croissant_path) as f:
        doc = json.load(f)

    dist = doc.get("distribution")
    if not dist:
        return False, f"no distribution block in {croissant_path.name}"

    changed = False
    for entry in dist:
        if entry.get("name") != target:
            entry["name"] = target
            changed = True
        if entry.get("contentUrl") != target:
            entry["contentUrl"] = target
            changed = True

    if changed:
        with open(croissant_path, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")

    return changed, f"-> {target}"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("release_dir", type=Path, help="release bundle directory")
    release_dir = parser.parse_args().release_dir
    if not release_dir.is_dir():
        print(f"ERROR: {release_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    descriptors = sorted(release_dir.rglob("*.croissant.json"))
    if not descriptors:
        print(f"ERROR: no *.croissant.json found under {release_dir}", file=sys.stderr)
        sys.exit(1)

    failures = 0
    for path in descriptors:
        changed, msg = fix_descriptor(path)
        rel = path.relative_to(release_dir)
        if msg.startswith("->"):
            print(f"    {rel} {msg}" + ("" if changed else "  (already correct)"))
        else:
            print(f"    WARN: {rel}: {msg}", file=sys.stderr)
            failures += 1

    print(f"    rewrote {len(descriptors) - failures}/{len(descriptors)} descriptors")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
