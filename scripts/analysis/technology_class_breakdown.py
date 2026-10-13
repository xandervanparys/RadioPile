"""
Per-technology-class source breakdown for the aggregated RadioPile dataset.

For each technology_type class, reports which source datasets contribute
how many windows (count + percentage of class), and the sampling rate /
window duration each contribution represents.

Sample rate is read per-window from /meta/sample_rate_hz. It is NaN for the
simulated sources, which have no physical rate. Window duration =
target_window_size / sample_rate_hz.

Usage:
    python scripts/analysis/technology_class_breakdown.py
    python scripts/analysis/technology_class_breakdown.py --csv out.csv
    python scripts/analysis/technology_class_breakdown.py --include-unknown
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


AGG_H5 = Path("data/aggregation/processed/aggregated.h5")

def _decode(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "O":
        return np.array([s.decode() if isinstance(s, bytes) else s for s in arr])
    return arr


def format_sample_rate(sr: float | None) -> str:
    if sr is None:
        return "n/a (simulated)"
    if isinstance(sr, float) and np.isnan(sr):
        return "mixed"
    if sr >= 1e6:
        return f"{sr/1e6:g} MHz"
    if sr >= 1e3:
        return f"{sr/1e3:g} kHz"
    return f"{sr:g} Hz"


def format_duration(window_size: int, sr: float | None) -> str:
    if sr is None or (isinstance(sr, float) and (np.isnan(sr) or sr <= 0)):
        return "n/a"
    secs = window_size / sr
    if secs >= 1e-3:
        return f"{secs*1e3:.4g} ms"
    return f"{secs*1e6:.4g} µs"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", type=Path, default=AGG_H5,
                    help=f"Aggregated HDF5 path (default: {AGG_H5})")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional CSV output path")
    ap.add_argument("--include-unknown", action="store_true",
                    help="Also report the 'unknown' technology_type bucket")
    args = ap.parse_args()

    if not args.h5.exists():
        print(f"ERROR: {args.h5} not found")
        return 1

    with h5py.File(args.h5, "r") as f:
        window_size = int(f.attrs["target_window_size"])
        tech     = _decode(f["meta/technology_type"][:])
        source   = _decode(f["meta/source_dataset"][:])
        sr_arr   = f["meta/sample_rate_hz"][:].astype(np.float64)

    # Per-window sample rate (Hz). NaN means no physical rate (simulated).
    sr_per_win = sr_arr

    classes = sorted(set(tech.tolist()))
    classes_to_show = [c for c in classes if args.include_unknown or c != "unknown"]

    # Aggregate (class, source, sample_rate) cells. NaN sr collapses to a single
    # "n/a" bucket so simulated datasets don't fragment.
    cells: dict[tuple[str, str, float], int] = defaultdict(int)
    sr_key = np.where(np.isfinite(sr_per_win), sr_per_win, np.nan)
    # Bucket NaN to a sentinel so np.unique groups them together.
    SENT = -1.0
    sr_bucket = np.where(np.isfinite(sr_key), sr_key, SENT)

    triples = list(zip(tech.tolist(), source.tolist(), sr_bucket.tolist()))
    # Use Counter via numpy: build composite key strings.
    keys = np.array([f"{t}|{s}|{r}" for t, s, r in triples])
    uniq, counts = np.unique(keys, return_counts=True)
    for key, c in zip(uniq, counts):
        t, s, r_str = key.split("|", 2)
        r = float(r_str)
        cells[(t, s, r)] = int(c)

    # Console report
    print(f"Aggregated dataset: {args.h5}")
    print(f"Total windows: {len(tech):,}   Target window size: {window_size} samples")
    print()

    rows_out: list[dict] = []
    for cls in classes_to_show:
        rows = [(src, r, n) for (t, src, r), n in cells.items() if t == cls and n > 0]
        if not rows:
            continue
        rows.sort(key=lambda x: -x[2])
        class_total = sum(n for _, _, n in rows)

        print(f"── technology_type = {cls!r}   (class total: {class_total:,} windows)")
        header = f"  {'source':<42} {'count':>12} {'% of class':>10}   "\
                 f"{'sample rate':>15}   {'window duration':>16}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for src, r, count in rows:
            sr = None if r == SENT else r
            pct = 100.0 * count / class_total
            print(f"  {src:<42} {count:>12,} {pct:>9.2f}%   "
                  f"{format_sample_rate(sr):>15}   {format_duration(window_size, sr):>16}")
            rows_out.append({
                "technology_type": cls,
                "source_dataset": src,
                "windows": count,
                "pct_of_class": round(pct, 4),
                "sample_rate_hz": "" if sr is None else sr,
                "sample_rate_pretty": format_sample_rate(sr),
                "window_duration": format_duration(window_size, sr),
                "class_total": class_total,
            })
        print()

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
        print(f"Wrote CSV: {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
