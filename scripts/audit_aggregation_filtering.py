"""
Audit aggregation label losses and window-size adaptation.

Simulates the aggregation pipeline (datasets/aggregation/build_dataset.py)
WITHOUT building any output. Reports, per enabled dataset:

  1. Whether the processed HDF5 exists (datasets not yet built are flagged)
  2. Primary label distribution under the current PRIMARY_LABEL_TYPE
  3. Count and percentage of windows whose primary label resolves to "unknown"
     (these are written to the output HDF5 as-is — nothing is hard-dropped —
     but they contribute nothing to the primary classification task)
  4. Window-size adaptation: pad count (source < target) or crop count
     (source > target) and the number of samples gained/lost per window

Disabled datasets (enabled=False) are listed separately with their HDF5 window
count if the file exists.

Usage:
    python scripts/audit_aggregation_filtering.py
    python scripts/audit_aggregation_filtering.py --label-type technology
    python scripts/audit_aggregation_filtering.py --sample 5000

Options:
    --label-type  Primary label type to simulate.  Defaults to the value in
                  datasets/aggregation/config.py (currently "technology").
                  "modulation" shows the same audit for the modulation label.

    --sample      Number of windows to sample per (dataset, label) group when
                  reading modulation_type for large datasets (> 1M windows).
                  Larger values are more accurate but slower.  Default: 2000.
                  Pass 0 to always read the full array (slow for NIST ~20 M).

Note on "unknown":
    The aggregation pipeline includes every window regardless of whether its
    primary label is "unknown".  The "unknown" count reported here is therefore
    NOT a hard filter — it measures how many windows are effectively unlabeled
    for the chosen primary task (e.g. modulation recognition vs. technology
    recognition).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.aggregation.build_dataset import (
    _load_dataset_config,
    extract_classification_metadata,
    get_primary_label,
)
from datasets.aggregation.config import (
    PRIMARY_LABEL_TYPE,
    SELECTED_DATASETS,
    TARGET_WINDOW_SIZE,
)
from datasets.registry import get_dataset

# ── Helpers ───────────────────────────────────────────────────────────────────

PASS = "✓"
WARN = "⚠"
INFO = "·"


def _decode(v: object) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return str(v)


def _decode_array(arr: np.ndarray) -> list[str]:
    return [_decode(x) for x in arr.tolist()]


def _read_vlen_sample(ds: h5py.Dataset, indices: np.ndarray) -> list[str]:
    """
    Read variable-length string values at the given *sorted* indices.

    h5py fancy-indexing works on vlen datasets when indices are sorted.
    Returns a list of decoded strings.
    """
    raw = ds[indices]
    return [_decode(v) for v in raw]


def _label_modtype_distribution(
    h5_path: Path,
    max_sample_per_label: int,
) -> tuple[Counter, int, int, bool]:
    """
    Return (counter, n_windows, source_window_size, used_sampling).

    counter keys are (label_name, modulation_type_or_None).
    used_sampling is True when any label group was sub-sampled.
    """
    counter: Counter = Counter()
    used_sampling = False

    with h5py.File(h5_path, "r") as f:
        n_windows: int = int(f["data"].shape[0])
        source_window_size: int = int(f["data"].shape[2])

        label_ids: np.ndarray = f["labels"][:]
        raw_names = f["label_names"][:] if "label_names" in f else np.array([])
        label_names = _decode_array(raw_names)

        has_modtype = "meta" in f and "modulation_type" in f["meta"]
        mod_ds: Optional[h5py.Dataset] = f["meta/modulation_type"] if has_modtype else None

        unique_ids = np.unique(label_ids)

        for lid in unique_ids:
            label = label_names[int(lid)] if int(lid) < len(label_names) else str(lid)
            indices = np.where(label_ids == lid)[0]
            n_label = len(indices)

            if not has_modtype or mod_ds is None:
                counter[(label, None)] += n_label
                continue

            if max_sample_per_label > 0 and n_label > max_sample_per_label:
                # Uniform sample across the label group
                sample_pos = np.linspace(0, n_label - 1, max_sample_per_label, dtype=int)
                sample_idx = np.sort(indices[sample_pos])
                used_sampling = True
            else:
                sample_idx = np.sort(indices)

            mod_vals = _read_vlen_sample(mod_ds, sample_idx)
            sample_counts = Counter(mod_vals)

            if max_sample_per_label > 0 and n_label > max_sample_per_label:
                # Scale sample counts back to full label count
                scale = n_label / len(sample_idx)
                for mod_val, cnt in sample_counts.items():
                    counter[(label, mod_val)] += int(round(cnt * scale))
                # Correct any rounding drift so totals stay exact
                label_total = sum(counter[(label, mv)] for mv in sample_counts)
                drift = n_label - label_total
                if drift != 0:
                    most_common_mod = sample_counts.most_common(1)[0][0]
                    counter[(label, most_common_mod)] += drift
            else:
                for mod_val, cnt in sample_counts.items():
                    counter[(label, mod_val)] += cnt

    return counter, n_windows, source_window_size, used_sampling


# ── Per-dataset audit ─────────────────────────────────────────────────────────

def audit_dataset(
    dataset_name: str,
    h5_path: Path,
    primary_label_type: str,
    max_sample_per_label: int,
) -> dict:
    """
    Open the processed HDF5 and simulate the aggregation label routing.

    Returns a result dict with keys:
        n_windows, source_window_size, pad_count, crop_count,
        primary_label_counts, unknown_count, used_sampling, label_represents
    """
    registry_entry = get_dataset(dataset_name)
    config_module = _load_dataset_config(registry_entry)

    counter, n_windows, source_window_size, used_sampling = _label_modtype_distribution(
        h5_path, max_sample_per_label
    )

    # Window-size adaptation counts
    if source_window_size < TARGET_WINDOW_SIZE:
        pad_count, crop_count = n_windows, 0
    elif source_window_size > TARGET_WINDOW_SIZE:
        pad_count, crop_count = 0, n_windows
    else:
        pad_count, crop_count = 0, 0

    # Simulate extract_classification_metadata + get_primary_label for each group
    primary_label_counts: Counter = Counter()

    for (label_name, mod_type), count in counter.items():
        fake_meta: dict = {}
        if mod_type is not None:
            fake_meta["modulation_type"] = mod_type

        technology, modulation, waveform, _radar_signal = extract_classification_metadata(
            label_name, fake_meta, registry_entry, config_module
        )
        primary = get_primary_label(technology, modulation, waveform, primary_label_type)
        primary_label_counts[primary] += count

    unknown_count = primary_label_counts.get("unknown", 0)

    return {
        "n_windows": n_windows,
        "source_window_size": source_window_size,
        "pad_count": pad_count,
        "crop_count": crop_count,
        "primary_label_counts": primary_label_counts,
        "unknown_count": unknown_count,
        "used_sampling": used_sampling,
        "label_represents": str(registry_entry.label_represents),
    }


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_pct(n: int, total: int) -> str:
    if total == 0:
        return "  n/a "
    return f"{100 * n / total:5.1f}%"


def _fmt_window_size(source: int, target: int, n_windows: int) -> str:
    if source == target:
        return f"{source}  (exact match, no adaptation)"
    if source < target:
        delta = target - source
        return (
            f"{source}  →  {target}  "
            f"(+{delta} samples zero-padded per window, "
            f"all {n_windows:,} windows padded)"
        )
    delta = source - target
    return (
        f"{source}  →  {target}  "
        f"(-{delta} samples cropped per window, "
        f"all {n_windows:,} windows cropped)"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--label-type",
        default=PRIMARY_LABEL_TYPE,
        choices=["technology", "modulation", "waveform"],
        help=(
            f"Primary label type to simulate "
            f"(default: '{PRIMARY_LABEL_TYPE}' from aggregation config)"
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=2000,
        metavar="N",
        help=(
            "Max windows sampled per (dataset, label) group when the group "
            "exceeds this size.  0 = always read in full (slow for NIST).  "
            "Default: 2000."
        ),
    )
    parser.add_argument(
        "--save-report", metavar="PATH", default=None,
        help="If set, also write a structured JSON audit report to PATH.",
    )
    args = parser.parse_args()
    primary_label_type: str = args.label_type
    max_sample: int = args.sample

    print("=" * 70)
    print("  Aggregation Filtering Audit")
    print("=" * 70)
    print(f"  Primary label type : {primary_label_type}")
    print(f"  Target window size : {TARGET_WINDOW_SIZE}")
    if max_sample > 0:
        print(f"  Sampling           : up to {max_sample:,} windows per (dataset, label)")
    else:
        print("  Sampling           : disabled (full arrays read)")
    print()

    disabled_rows: list[tuple[str, str]] = []    # (name, note)
    missing_rows:  list[str] = []
    audit_results: list[tuple[str, dict]] = []

    for dataset_name, config in SELECTED_DATASETS.items():
        if not config.get("enabled", True):
            # Peek at HDF5 if it exists for the window count
            try:
                entry = get_dataset(dataset_name)
                if entry.processed_h5_path.exists():
                    with h5py.File(entry.processed_h5_path, "r") as f:
                        n = int(f["data"].shape[0]) if "data" in f else 0
                    disabled_rows.append((dataset_name, f"{n:,} windows (HDF5 exists)"))
                else:
                    disabled_rows.append((dataset_name, "HDF5 not built"))
            except KeyError:
                disabled_rows.append((dataset_name, "not in registry"))
            continue

        try:
            registry_entry = get_dataset(dataset_name)
        except KeyError:
            missing_rows.append(f"{dataset_name}  (not in registry)")
            continue

        h5_path = registry_entry.processed_h5_path
        if not h5_path.exists():
            missing_rows.append(f"{dataset_name}  ({h5_path})")
            continue

        print(f"  Auditing {dataset_name} …", flush=True)
        result = audit_dataset(dataset_name, h5_path, primary_label_type, max_sample)
        audit_results.append((dataset_name, result))

    # ── Per-dataset report ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Per-dataset results")
    print("=" * 70)

    grand_total   = 0
    grand_unknown = 0
    grand_pad     = 0
    grand_crop    = 0

    for dataset_name, r in audit_results:
        n         = r["n_windows"]
        unk       = r["unknown_count"]
        src_w     = r["source_window_size"]
        pad       = r["pad_count"]
        crop      = r["crop_count"]
        sampling  = r["used_sampling"]
        label_rep = r["label_represents"]

        grand_total   += n
        grand_unknown += unk
        grand_pad     += pad
        grand_crop    += crop

        unk_pct = _fmt_pct(unk, n)
        marker  = PASS if unk == 0 else WARN

        print()
        print(f"  [{marker}] {dataset_name}")
        print(f"       label_represents  : {label_rep}")
        print(f"       source window     : {_fmt_window_size(src_w, TARGET_WINDOW_SIZE, n)}")

        if src_w != TARGET_WINDOW_SIZE:
            src_iq  = src_w
            tgt_iq  = TARGET_WINDOW_SIZE
            delta   = tgt_iq - src_iq   # positive = pad, negative = crop
            samples_note = (
                f"+{delta:,} IQ samples added (zero-pad)" if delta > 0
                else f"{delta:,} IQ samples removed (crop)"
            )
            print(f"       IQ impact per win : {samples_note}")

        print(f"       total windows     : {n:,}")
        print(f"       unknown label     : {unk:,}  ({unk_pct})", end="")

        if unk == n:
            print("  ← entire dataset unlabeled for this task")
        elif unk == 0:
            print("  ← fully labeled")
        else:
            print()

        if sampling:
            print(f"       {WARN} modulation_type distribution estimated from sample")

        # Primary label breakdown (sorted by count, unknown last)
        counts = r["primary_label_counts"]
        if counts:
            print(f"       primary label breakdown:")
            unknown_entry = counts.pop("unknown", None)
            for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                print(f"         {label:<35} {cnt:>10,}  ({_fmt_pct(cnt, n)})")
            if unknown_entry is not None:
                print(f"         {'unknown':<35} {unknown_entry:>10,}  ({_fmt_pct(unknown_entry, n)})  ← no {primary_label_type} info")
                counts["unknown"] = unknown_entry   # restore

    # ── Missing / disabled ────────────────────────────────────────────────────
    if missing_rows:
        print()
        print(f"  {WARN} Datasets not yet built (HDF5 missing):")
        for row in missing_rows:
            print(f"       {row}")

    if disabled_rows:
        print()
        print(f"  {INFO} Disabled datasets (enabled=False — excluded from aggregation):")
        for name, note in disabled_rows:
            print(f"       {name}  ({note})")

    # ── Aggregate summary ─────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Aggregate summary  (enabled datasets with existing HDF5)")
    print("=" * 70)

    if grand_total == 0:
        print("  No datasets available.")
        return

    grand_meaningful = grand_total - grand_unknown

    print(f"  Total windows              : {grand_total:>14,}")
    print(f"  Primary label = 'unknown'  : {grand_unknown:>14,}  ({_fmt_pct(grand_unknown, grand_total)})")
    print(f"  Primary label meaningful   : {grand_meaningful:>14,}  ({_fmt_pct(grand_meaningful, grand_total)})")
    print()
    print(f"  Zero-padded windows        : {grand_pad:>14,}  ({_fmt_pct(grand_pad, grand_total)})")
    print(f"  Cropped windows            : {grand_crop:>14,}  ({_fmt_pct(grand_crop, grand_total)})")
    print()
    print(
        f"  Note: ALL {grand_total:,} windows are written to the aggregated HDF5.\n"
        f"        'unknown' is not dropped — it becomes a label class."
    )
    other = "modulation" if primary_label_type == "technology" else "technology"
    print()
    print(f"  Run with --label-type {other} for the same audit of the {other} label.")
    print("=" * 70)

    if args.save_report:
        report = {
            "audit": "aggregation_filtering",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "primary_label_type": primary_label_type,
            "target_window_size": TARGET_WINDOW_SIZE,
            "sample_per_label": max_sample,
            "summary": {
                "grand_total": grand_total,
                "grand_unknown": grand_unknown,
                "grand_meaningful": grand_total - grand_unknown,
                "grand_pad": grand_pad,
                "grand_crop": grand_crop,
            },
            "datasets": {
                name: {
                    "n_windows":            r["n_windows"],
                    "source_window_size":   r["source_window_size"],
                    "pad_count":            r["pad_count"],
                    "crop_count":           r["crop_count"],
                    "unknown_count":        r["unknown_count"],
                    "used_sampling":        r["used_sampling"],
                    "label_represents":     r["label_represents"],
                    "primary_label_counts": dict(r["primary_label_counts"]),
                }
                for name, r in audit_results
            },
            "missing": missing_rows,
            "disabled": [{"name": n, "note": note} for n, note in disabled_rows],
        }
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\n  Audit report written to {report_path}")


if __name__ == "__main__":
    main()
