"""
Audit noise filter losses across the full pipeline.

Reports, per dataset, how many raw IQ windows would be dropped by the
configured noise power filter when the pipeline is run from scratch.

Key facts:
  - Noise filtering happens on RAW windows BEFORE normalization.
    After unit-power normalization every window has power ≈ 0 dB, so the
    processed HDF5 files cannot be used as input — this script must read
    the raw data directories.
  - Only two datasets have noise filtering enabled:
      sub_ghz               "auto" — 1st-percentile noise floor + 6 dB margin,
                            detected once per .mat file.  Noise-class files are
                            explicitly skipped (all windows kept).
      nist_wifi_bluetooth   "auto" — same method, threshold detected once from the
                            first capture of each source file, applied to all captures.
  - Five datasets have NOISE_POWER_THRESHOLD_DB = None (filtering disabled):
      rml2016_10a, rml2018_01a, radchar, its_band,
      lte_wifi_dvbt
    These contribute 0 filtered windows regardless of signal power.

Approach:
  sub_ghz   — full exact pass over every .mat file (files are small)
  nist      — threshold from first capture; filter rate estimated from
              evenly spaced captures per file (default 10 of 180),
              then extrapolated.  Use --nist-captures 0 for an exact pass
              (~21 GB of reads per file — slow).

Usage:
    python scripts/audit_noise_filtering.py
    python scripts/audit_noise_filtering.py --nist-captures 20
    python scripts/audit_noise_filtering.py --nist-captures 0   # exact, slow
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preprocessing import detect_noise_floor_percentile
from pipeline.windowing import window_iq_signal

# ── Dataset config imports ────────────────────────────────────────────────────

from datasets.sub_ghz.config import (
    RAW_DATA_ROOT as SUBGHZ_RAW_ROOT,
    LABEL_REMAP as SUBGHZ_LABEL_REMAP,
    WINDOW_SIZE as SUBGHZ_WINDOW_SIZE,
    HOP_SIZE as SUBGHZ_HOP_SIZE,
    PAD_LAST as SUBGHZ_PAD_LAST,
    NOISE_POWER_THRESHOLD_DB as SUBGHZ_NOISE_CFG,
    NOISE_FLOOR_DETECTION_PARAMS as SUBGHZ_NOISE_PARAMS,
    NOISE_FLOOR_MAX_SAMPLES as SUBGHZ_MAX_SAMPLES,
)
from datasets.sub_ghz.build_dataset import infer_label_from_path as subghz_infer_label

from datasets.nist_wifi_bluetooth.config import (
    RAW_DATA_ROOT as NIST_RAW_ROOT,
    FILE_CONFIG as NIST_FILE_CONFIG,
    WINDOW_SIZE as NIST_WINDOW_SIZE,
    NOISE_FLOOR_DETECTION_PARAMS as NIST_NOISE_PARAMS,
    NOISE_FLOOR_MAX_SAMPLES as NIST_MAX_SAMPLES,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

WARN = "⚠"
INFO = "·"

# Datasets where filtering is explicitly disabled.
# Values: (reason, processed_hdf5_path)
DISABLED: dict[str, tuple[str, Path]] = {
    "rml2016_10a": (
        "SNR levels are part of dataset design",
        Path("data/rml2016_10a/processed/rml2016_10a.h5"),
    ),
    "rml2018_01a": (
        "SNR levels are part of dataset design",
        Path("data/rml2018_01a/processed/rml2018_01a.h5"),
    ),
    "radchar": (
        "SNR levels are part of dataset design",
        Path("data/radchar/processed/radchar.h5"),
    ),
    "its_band": (
        "signals are continuous — no inter-frame gaps to filter",
        Path("data/its_band/processed/its_band.h5"),
    ),
    "lte_wifi_dvbt": (
        "signals are continuous",
        Path("data/lte_wifi_dvbt/processed/lte_wifi_dvbt.h5"),
    ),
}


def _h5_window_count(path: Path) -> int | None:
    """Return /data.shape[0] from a processed HDF5, or None if unavailable."""
    try:
        with h5py.File(path, "r") as f:
            return int(f["data"].shape[0]) if "data" in f else None
    except Exception:
        return None


def _powers_db_for_iq(iq: np.ndarray, window_size: int, hop_size: int, pad_last: bool) -> np.ndarray:
    """
    Window a [N, 2] float32 IQ array and return per-window power in dB.

    Uses vectorized numpy — no Python loop over windows.
    """
    windows, _ = window_iq_signal(iq, window_size, hop_size, pad_last)
    if windows.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    # windows: (n, 2, W)
    power_lin = np.mean(windows[:, 0, :] ** 2 + windows[:, 1, :] ** 2, axis=1)
    return (10.0 * np.log10(power_lin + 1e-12)).astype(np.float32)


def _auto_threshold(powers_db: np.ndarray, noise_params: dict, max_samples: int) -> float:
    """
    Derive noise threshold from a power array using the percentile+margin method.

    Mirrors determine_noise_threshold("auto") in pipeline/preprocessing.py.
    """
    params = dict(noise_params)
    margin_db = float(params.get("margin_db", 0.0))
    percentile = float(params.get("percentile", 1.0))
    sample = powers_db[:max_samples]
    floor_db = detect_noise_floor_percentile(sample, percentile=percentile)
    return floor_db + margin_db


# ── Sub-GHz ───────────────────────────────────────────────────────────────────

def audit_subghz() -> dict:
    """
    Exact audit of sub_ghz noise filtering over all raw .mat files.

    Noise-class files are not filtered (matches build_dataset.py behavior).
    Returns summary dict; prints per-file detail.
    """
    if not SUBGHZ_RAW_ROOT.exists():
        print(f"  {WARN} Raw directory not found: {SUBGHZ_RAW_ROOT}")
        return {"error": str(SUBGHZ_RAW_ROOT)}

    mat_files = sorted(SUBGHZ_RAW_ROOT.rglob("*.mat"))
    if not mat_files:
        print(f"  {WARN} No .mat files under {SUBGHZ_RAW_ROOT}")
        return {"error": "no mat files"}

    total = 0
    filtered = 0
    per_file: list[dict] = []

    col_w = max(len(p.name) for p in mat_files) + 2

    for path in mat_files:
        try:
            raw_label = subghz_infer_label(path)
        except ValueError as e:
            print(f"  {WARN} {path.name}: could not infer label ({e}), skipping")
            continue

        is_noise = raw_label == "noise"

        try:
            mat = scipy.io.loadmat(path)
        except Exception as e:
            print(f"  {WARN} {path.name}: load error ({e}), skipping")
            continue

        if "IQ_samples" not in mat:
            print(f"  {WARN} {path.name}: missing IQ_samples, skipping")
            continue

        iq_raw = mat["IQ_samples"].squeeze()
        if not np.iscomplexobj(iq_raw):
            print(f"  {WARN} {path.name}: IQ not complex, skipping")
            continue

        iq = np.stack([iq_raw.real, iq_raw.imag], axis=1).astype(np.float32)
        powers = _powers_db_for_iq(iq, SUBGHZ_WINDOW_SIZE, SUBGHZ_HOP_SIZE, SUBGHZ_PAD_LAST)
        n = len(powers)

        if is_noise or SUBGHZ_NOISE_CFG is None:
            file_filtered = 0
            threshold_db = None
        else:
            threshold_db = _auto_threshold(powers, SUBGHZ_NOISE_PARAMS, SUBGHZ_MAX_SAMPLES)
            file_filtered = int(np.sum(powers < threshold_db))

        total += n
        filtered += file_filtered

        label = SUBGHZ_LABEL_REMAP.get(raw_label, raw_label)
        pct = f"{100 * file_filtered / n:.1f}%" if n > 0 else "n/a"
        thr = f"{threshold_db:+.1f} dB" if threshold_db is not None else "disabled"
        noise_tag = "  [noise class — filter skipped]" if is_noise else ""
        print(
            f"  {path.name:<{col_w}} label={label:<16} "
            f"thr={thr:<12}  filtered={file_filtered:>6,}/{n:<6,} ({pct}){noise_tag}"
        )

        per_file.append({
            "path": path,
            "label": label,
            "is_noise": is_noise,
            "n": n,
            "filtered": file_filtered,
            "threshold_db": threshold_db,
        })

    return {
        "total": total,
        "filtered": filtered,
        "pass": total - filtered,
        "per_file": per_file,
        "sampled": False,
    }


# ── NIST WiFi ────────────────────────────────────────────────────────────────

_NIST_INT16_SCALE = 32768.0


def _load_nist_capture_iq(f: h5py.File, key: str) -> np.ndarray:
    """Load one NIST capture → float32 IQ [N_pairs, 2]."""
    raw = np.array(f[key], dtype=np.float32) / _NIST_INT16_SCALE
    return raw.reshape(-1, 2)


def audit_nist(n_sample: int) -> dict:
    """
    Audit NIST noise filtering.

    Threshold is detected from the first capture of each source file (same as
    the build script).  Filter rate is then estimated from `n_sample` evenly
    spaced captures (pass n_sample=0 for an exact pass over all captures).

    Returns summary dict; prints per-file detail.
    """
    if not NIST_RAW_ROOT.exists():
        print(f"  {WARN} Raw directory not found: {NIST_RAW_ROOT}")
        return {"error": str(NIST_RAW_ROOT)}

    grand_total = 0
    grand_filtered = 0
    per_file: list[dict] = []
    any_sampled = False

    for stem, (freq_hz, env, label) in NIST_FILE_CONFIG.items():
        h5_path = NIST_RAW_ROOT / f"{stem}.h5"
        if not h5_path.exists():
            print(f"  {WARN} Missing: {h5_path.name}")
            per_file.append({"stem": stem, "error": "not found"})
            continue

        with h5py.File(h5_path, "r") as f:
            capture_keys = list(f.keys())
            n_caps = len(capture_keys)

            # Threshold: detected from first capture only (matches build script)
            iq_first = _load_nist_capture_iq(f, capture_keys[0])

        n_win_per_cap = len(iq_first) // NIST_WINDOW_SIZE  # truncating division
        file_total = n_caps * n_win_per_cap                 # exact total

        # Vectorized power for first capture
        iq_t = iq_first[: n_win_per_cap * NIST_WINDOW_SIZE].reshape(
            n_win_per_cap, NIST_WINDOW_SIZE, 2
        )
        wins_first = np.stack([iq_t[:, :, 0], iq_t[:, :, 1]], axis=1)  # (n, 2, W)
        pwr_first = np.mean(wins_first[:, 0, :] ** 2 + wins_first[:, 1, :] ** 2, axis=1)
        pwr_db_first = 10.0 * np.log10(pwr_first + 1e-12)

        threshold_db = _auto_threshold(pwr_db_first, NIST_NOISE_PARAMS, NIST_MAX_SAMPLES)

        print(
            f"\n  {stem}.h5   ({n_caps} captures × {n_win_per_cap:,} win/cap = {file_total:,} total)"
        )
        print(
            f"    Noise floor from cap[0]: "
            f"{threshold_db - float(dict(NIST_NOISE_PARAMS).get('margin_db', 0)):.2f} dB"
            f"  +{float(dict(NIST_NOISE_PARAMS).get('margin_db', 0)):.0f} dB margin"
            f"  → threshold: {threshold_db:.2f} dB"
        )

        # ── Count pass / filtered ─────────────────────────────────────────────
        if n_sample == 0:
            # Exact: all captures
            file_filtered = 0
            with h5py.File(h5_path, "r") as f:
                for cap_key in capture_keys:
                    iq = _load_nist_capture_iq(f, cap_key)
                    n_w = len(iq) // NIST_WINDOW_SIZE
                    iq_t = iq[: n_w * NIST_WINDOW_SIZE].reshape(n_w, NIST_WINDOW_SIZE, 2)
                    wins = np.stack([iq_t[:, :, 0], iq_t[:, :, 1]], axis=1)
                    pwr = np.mean(wins[:, 0, :] ** 2 + wins[:, 1, :] ** 2, axis=1)
                    file_filtered += int(np.sum(10.0 * np.log10(pwr + 1e-12) < threshold_db))
            is_sampled = False
        else:
            sample_n = min(n_sample, n_caps)
            is_sampled = sample_n < n_caps
            if is_sampled:
                any_sampled = True

            idx_arr = np.linspace(0, n_caps - 1, sample_n, dtype=int)
            sample_filtered = 0
            sample_windows = 0

            with h5py.File(h5_path, "r") as f:
                for idx in idx_arr:
                    iq = _load_nist_capture_iq(f, capture_keys[int(idx)])
                    n_w = len(iq) // NIST_WINDOW_SIZE
                    iq_t = iq[: n_w * NIST_WINDOW_SIZE].reshape(n_w, NIST_WINDOW_SIZE, 2)
                    wins = np.stack([iq_t[:, :, 0], iq_t[:, :, 1]], axis=1)
                    pwr = np.mean(wins[:, 0, :] ** 2 + wins[:, 1, :] ** 2, axis=1)
                    sample_filtered += int(np.sum(10.0 * np.log10(pwr + 1e-12) < threshold_db))
                    sample_windows += n_w

            rate = sample_filtered / sample_windows if sample_windows > 0 else 0.0
            file_filtered = int(round(rate * file_total))

            est_tag = f"  (estimated from {sample_n}/{n_caps} caps, rate={100*rate:.1f}%)"
            print(f"    {est_tag.strip()}")

        file_pass = file_total - file_filtered
        flt_pct = 100 * file_filtered / file_total if file_total > 0 else 0.0
        est = " (est.)" if is_sampled else "       "
        print(
            f"    total={file_total:>12,}  "
            f"filtered={file_filtered:>10,}{est} ({flt_pct:5.1f}%)  "
            f"pass={file_pass:>10,}"
        )

        grand_total    += file_total
        grand_filtered += file_filtered
        per_file.append({
            "stem":          stem,
            "label":         label,
            "environment":   env,
            "n_captures":    n_caps,
            "win_per_cap":   n_win_per_cap,
            "total":         file_total,
            "filtered":      file_filtered,
            "pass":          file_pass,
            "threshold_db":  threshold_db,
            "sampled":       is_sampled,
        })

    return {
        "total":    grand_total,
        "filtered": grand_filtered,
        "pass":     grand_total - grand_filtered,
        "per_file": per_file,
        "sampled":  any_sampled,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nist-captures",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Captures to sample per NIST source file for filter-rate estimation "
            "(each file has 180 captures of ~120 MB each).  "
            "0 = exact full pass (slow).  Default: 10."
        ),
    )
    parser.add_argument(
        "--save-report", metavar="PATH", default=None,
        help="If set, also write a structured JSON audit report to PATH.",
    )
    args = parser.parse_args()

    W = 70
    print("=" * W)
    print("  Noise Filter Loss Audit")
    print("=" * W)
    print()

    # ── Collect disabled-dataset window counts ────────────────────────────────
    disabled_rows: list[tuple[str, int, str]] = []
    disabled_total = 0
    for name, (reason, h5_path) in DISABLED.items():
        n = _h5_window_count(h5_path) or 0
        disabled_total += n
        disabled_rows.append((name, n, reason))

    print(f"  Datasets with NOISE_POWER_THRESHOLD_DB = None  (0 windows filtered)")
    print(f"  {'─'*66}")
    for name, n, reason in disabled_rows:
        n_str = f"{n:,}" if n else "HDF5 not built"
        print(f"    {name:<48} {n_str:>10}  [{reason}]")
    print()

    # ── sub_ghz ───────────────────────────────────────────────────────────────
    print("─" * W)
    print("  sub_ghz  (auto-threshold, per .mat file, noise class skipped)")
    print("─" * W)
    sg = audit_subghz()
    print()

    # ── nist ──────────────────────────────────────────────────────────────────
    print("─" * W)
    print(
        f"  nist_wifi_bluetooth  (auto-threshold, per source file)\n"
        f"  Threshold from first capture; filter rate from "
        f"{'all' if args.nist_captures == 0 else args.nist_captures} "
        f"of 180 captures per file"
    )
    print("─" * W)
    nist = audit_nist(n_sample=args.nist_captures)

    # ── Grand summary ─────────────────────────────────────────────────────────
    print()
    print("=" * W)
    print("  Grand summary")
    print("=" * W)
    print()

    col = 44
    hdr = f"  {'Dataset':<{col}} {'Raw total':>12}  {'Filtered':>12}  {'Pass':>12}"
    sep = f"  {'─'*col} {'─'*12}  {'─'*12}  {'─'*12}"
    print(hdr)
    print(sep)

    grand_raw      = disabled_total
    grand_filtered = 0

    for name, n, _ in disabled_rows:
        print(f"  {name:<{col}} {n:>12,}  {'0':>12}  {n:>12,}")

    def _row(label, result, est_marker=""):
        nonlocal grand_raw, grand_filtered
        if result and "error" not in result:
            n, f, p = result["total"], result["filtered"], result["pass"]
            grand_raw      += n
            grand_filtered += f
            print(f"  {label:<{col}} {n:>12,}  {f:>12,}  {p:>12,}{est_marker}")

    sg_est   = "  (~est.)" if sg.get("sampled")   else ""
    nist_est = "  (~est.)" if nist.get("sampled") else ""
    _row("sub_ghz",             sg,   sg_est)
    _row("nist_wifi_bluetooth", nist, nist_est)

    print(sep)

    grand_pass = grand_raw - grand_filtered
    def pct(n, tot): return f"{100*n/tot:.1f}%" if tot else "n/a"

    print(
        f"  {'TOTAL':<{col}} {grand_raw:>12,}  "
        f"{grand_filtered:>12,}  {grand_pass:>12,}"
    )
    print(
        f"  {'':{ col}} {'':12}   ({pct(grand_filtered, grand_raw):>5})   "
        f"({pct(grand_pass, grand_raw):>5})"
    )

    # IQ sample counts
    subghz_w  = SUBGHZ_WINDOW_SIZE
    nist_w    = NIST_WINDOW_SIZE
    sg_flt_samp   = (sg.get("filtered",  0) * subghz_w) if "error" not in sg   else 0
    nist_flt_samp = (nist.get("filtered",0) * nist_w)   if "error" not in nist else 0
    total_flt_samp = sg_flt_samp + nist_flt_samp

    print()
    print(f"  IQ samples discarded (windows × window_size):")
    print(f"    sub_ghz             {sg_flt_samp:>16,}  ({SUBGHZ_WINDOW_SIZE} samples/window)")
    print(f"    nist_wifi_bluetooth {nist_flt_samp:>16,}  ({NIST_WINDOW_SIZE} samples/window)")
    print(f"    Total               {total_flt_samp:>16,}")

    if nist.get("sampled"):
        print()
        print(
            f"  {WARN} NIST counts are estimated from "
            f"{args.nist_captures}/180 captures per file.\n"
            f"     Run with --nist-captures 0 for exact numbers (slow, ~21 GB/file)."
        )

    print()
    print("=" * W)

    if args.save_report:
        # Strip non-serializable Path objects from the per_file lists.
        def _clean(result: dict) -> dict:
            if "error" in result:
                return result
            cleaned = {k: v for k, v in result.items() if k != "per_file"}
            cleaned["per_file"] = [
                {k: (str(v) if isinstance(v, Path) else v) for k, v in entry.items()}
                for entry in result.get("per_file", [])
            ]
            return cleaned

        report = {
            "audit": "noise_filtering",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "nist_captures_sampled": args.nist_captures,
            "summary": {
                "grand_raw":       grand_raw,
                "grand_filtered":  grand_filtered,
                "grand_pass":      grand_raw - grand_filtered,
                "iq_samples_discarded": {
                    "sub_ghz":             sg_flt_samp,
                    "nist_wifi_bluetooth": nist_flt_samp,
                    "total":               total_flt_samp,
                },
            },
            "disabled": [
                {"name": n, "n_windows": n_w, "reason": reason}
                for n, n_w, reason in disabled_rows
            ],
            "sub_ghz":             _clean(sg),
            "nist_wifi_bluetooth": _clean(nist),
        }
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
        print(f"\n  Audit report written to {report_path}")


if __name__ == "__main__":
    main()
