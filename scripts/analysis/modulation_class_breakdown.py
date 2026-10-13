"""
Per-modulation-class source breakdown for the aggregated RadioPile dataset.

Analysis script behind the paper's per-class source table (not part of the
validation pipeline). Mirrors `technology_class_breakdown.py` but over
`meta/modulation_type` instead of `meta/technology_type`, with a different
display rule for simulated sources (no physical sample rate → report the
native source window size in samples).

Filtering:
  - Drops windows where modulation_type ∈ {"unknown", "noise"}.

For each remaining class, lists contributing sources (split by sample rate
when a single source contributes at multiple rates) with count, % of class,
sample rate and window duration. Simulated sources (RML2016 / RML2018) show
"simulated" / "<N> samples (simulated)" using the native source window size
read from `meta/original_window_size`.

Usage:
    python scripts/analysis/modulation_class_breakdown.py
    python scripts/analysis/modulation_class_breakdown.py --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


AGG_H5 = Path("data/aggregation/processed/aggregated.h5")
EXCLUDE_VALUES = {"unknown", "noise"}

def _decode(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "O":
        return np.array([s.decode() if isinstance(s, bytes) else s for s in arr])
    return arr


def format_sample_rate(sr: float | None) -> str:
    if sr is None or (isinstance(sr, float) and np.isnan(sr)):
        return "simulated"
    if sr >= 1e6:
        return f"{sr/1e6:g} MHz"
    if sr >= 1e3:
        return f"{sr/1e3:g} kHz"
    return f"{sr:g} Hz"


def format_duration(window_size: int, sr: float | None,
                    native_window: int | None) -> str:
    if sr is None or (isinstance(sr, float) and (np.isnan(sr) or sr <= 0)):
        if native_window is not None:
            return f"{native_window} samples (simulated)"
        return "simulated"
    secs = window_size / sr
    if secs >= 1e-3:
        return f"{secs*1e3:.4g} ms"
    return f"{secs*1e6:.4g} µs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, default=AGG_H5)
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional CSV output path")
    args = ap.parse_args()

    if not args.h5.exists():
        print(f"ERROR: {args.h5} not found")
        return 1

    with h5py.File(args.h5, "r") as f:
        window_size = int(f.attrs["target_window_size"])
        mod    = _decode(f["meta/modulation_type"][:])
        source = _decode(f["meta/source_dataset"][:])
        sr_arr = f["meta/sample_rate_hz"][:].astype(np.float64)
        ows    = f["meta/original_window_size"][:].astype(np.int64)

    keep = ~np.isin(mod, list(EXCLUDE_VALUES))
    n_total_pre  = len(mod)
    n_total_post = int(keep.sum())

    mod    = mod[keep]
    source = source[keep]
    sr_arr = sr_arr[keep]
    ows    = ows[keep]

    SR_SENT = -1.0
    OWS_SENT = -1
    sr_bucket  = np.where(np.isfinite(sr_arr), sr_arr, SR_SENT)
    ows_bucket = np.where(ows > 0, ows, OWS_SENT)

    cells: dict[tuple[str, str, float, int], int] = defaultdict(int)
    keys = np.array([
        f"{m}|{s}|{r}|{w}"
        for m, s, r, w in zip(mod.tolist(), source.tolist(),
                              sr_bucket.tolist(), ows_bucket.tolist())
    ])
    uniq, counts = np.unique(keys, return_counts=True)
    for key, c in zip(uniq, counts):
        m, s, r_str, w_str = key.split("|", 3)
        cells[(m, s, float(r_str), int(w_str))] = int(c)

    classes = sorted({m for (m, _, _, _) in cells.keys()})
    class_totals = {
        cls: sum(n for (m, _, _, _), n in cells.items() if m == cls)
        for cls in classes
    }
    classes_sorted = sorted(classes, key=lambda c: -class_totals[c])

    print(f"Aggregated dataset: {args.h5}")
    print(f"Total windows: {n_total_pre:,}   Target window size: {window_size} samples")
    print(f"After filtering (drop modulation_type ∈ {sorted(EXCLUDE_VALUES)}): "
          f"{n_total_post:,} windows  |  {len(classes_sorted)} modulation classes")
    print()

    rows_out: list[dict] = []
    for cls in classes_sorted:
        rows = [
            (src, r, w, n)
            for (m, src, r, w), n in cells.items()
            if m == cls and n > 0
        ]
        rows.sort(key=lambda x: -x[3])
        class_total = class_totals[cls]

        print(f"── modulation_type = {cls!r}   (class total: {class_total:,} windows)")
        header = (f"  {'source':<42} {'count':>12} {'% of class':>10}   "
                  f"{'sample rate':>15}   {'window duration':>26}")
        print(header)
        print("  " + "-" * (len(header) - 2))
        for src, r, w, count in rows:
            sr = None if r == SR_SENT else r
            native = None if w == OWS_SENT else w
            pct = 100.0 * count / class_total
            print(f"  {src:<42} {count:>12,} {pct:>9.2f}%   "
                  f"{format_sample_rate(sr):>15}   "
                  f"{format_duration(window_size, sr, native):>26}")
            rows_out.append({
                "modulation_type": cls,
                "source_dataset": src,
                "windows": count,
                "pct_of_class": round(pct, 4),
                "sample_rate_hz": "" if sr is None else sr,
                "sample_rate_pretty": format_sample_rate(sr),
                "window_duration": format_duration(window_size, sr, native),
                "native_source_window": "" if native is None else native,
                "class_total": class_total,
            })
        print()

    n_classes = len(classes_sorted)
    n_windows = sum(class_totals.values())
    print("── Summary ─────────────────────────────────────────────")
    print(f"  Modulation classes (post-filter): {n_classes}")
    print(f"  Total windows (post-filter):      {n_windows:,}")

    notes: list[str] = []
    contributing_sources = sorted({src for (_, src, _, _) in cells.keys()})
    notes.append(f"Contributing sources: {contributing_sources}")
    nist_total = sum(n for (_, src, _, _), n in cells.items() if src == "nist_wifi_bluetooth")
    if n_windows and nist_total / n_windows > 0.7:
        notes.append(
            f"nist_wifi_bluetooth dominates: {nist_total:,} / {n_windows:,} "
            f"({100*nist_total/n_windows:.1f}%) — modulation totals are heavily "
            "skewed by NIST's 64QAM/256QAM contribution."
        )
    for note in notes:
        print(f"  Note: {note}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"\nWrote CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
