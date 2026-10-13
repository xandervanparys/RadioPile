"""
Verification script for NIST WiFi dataset noise filtering.

For each of the 3 included HDF5 files:
  1. Sample captures spread across all 180 to estimate true duty cycle
  2. Plot power distribution and show where the proposed threshold lands
  3. Compare pipeline auto-detection vs. the original amplitude threshold approach
  4. Report how many signal vs. noise windows each method preserves

Usage:
    python scripts/analysis/verify_nist_noise_filtering.py [--raw-root DIR] [--output-dir DIR] [--captures N]
"""

import argparse

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.preprocessing import (
    compute_window_power_db,
    detect_noise_floor_percentile,
)
from pipeline.windowing import iter_windows_from_samples

# -------------------------------------------------------------------------
# Config (mirrors datasets/nist_wifi_bluetooth/config.py)
# -------------------------------------------------------------------------

RAW_DATA_ROOT = Path("data/nist_wifi_bluetooth/raw")
OUTPUT_DIR = Path("data/nist_wifi_bluetooth/processed/plots")  # never write into raw/

FILE_CONFIG = {
    "2_4ghz_indoor": (2.437e9, "indoor"),
    "5ghz_indoor":   (5.825e9, "indoor"),
    "5ghz_outdoor":  (5.825e9, "outdoor"),
}

WINDOW_SIZE = 500
HOP_SIZE = 500
NOISE_AMPLITUDE_THRESHOLD = 0.05   # original approach: max(abs(I)) > threshold

# Pipeline approach parameters
PERCENTILE = 1.0
MARGIN_DB  = 6.0

# Default number of captures sampled per file for duty cycle estimation
# (180 in total; 20 spread evenly keeps the runtime reasonable)
N_CAPTURE_SAMPLES = 20
MAX_WINDOWS_FOR_NOISE_FLOOR = 10_000  # from NOISE_FLOOR_MAX_SAMPLES in config


def load_capture_iq(f: h5py.File, key: str) -> np.ndarray:
    """Load one capture and return float32 IQ array of shape (N, 2)."""
    raw = np.array(f[key], dtype=np.float32) / 32768.0
    return raw.reshape(-1, 2)


def windows_from_iq(iq: np.ndarray):
    """Yield (window, meta) tuples from a (N,2) IQ array via pipeline windowing."""
    yield from iter_windows_from_samples(
        samples=iq,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        pad_last=False,
    )


def analyze_file(filename_stem: str, raw_root: Path, n_samples: int):
    """
    Analyze one HDF5 file:
      - Sample n_samples captures spread across all 180
      - Compute per-window power for each sampled capture
      - Detect noise floor from first capture (as the build script does)
      - Compare amplitude-threshold and power-threshold approaches
    """
    h5_path = raw_root / f"{filename_stem}.h5"
    print(f"\n{'='*60}")
    print(f"File: {filename_stem}.h5")

    with h5py.File(h5_path, "r") as f:
        all_keys = list(f.keys())

    n_captures = len(all_keys)
    print(f"  Total captures: {n_captures}")

    # Pick captures spread evenly across all 180
    sample_indices = np.linspace(0, n_captures - 1, n_samples, dtype=int)
    sampled_keys = [all_keys[i] for i in sample_indices]

    # ------------------------------------------------------------------
    # Step 1: Compute window powers across sampled captures
    # ------------------------------------------------------------------
    all_powers_db = []
    amplitude_active_count = 0
    total_windows = 0

    with h5py.File(h5_path, "r") as f:
        for key in sampled_keys:
            iq = load_capture_iq(f, key)
            I_all = iq[:, 0]
            Q_all = iq[:, 1]

            n = len(I_all) // WINDOW_SIZE
            I_wins = I_all[:n * WINDOW_SIZE].reshape(n, WINDOW_SIZE)
            Q_wins = Q_all[:n * WINDOW_SIZE].reshape(n, WINDOW_SIZE)

            for idx in range(n):
                window = np.stack([I_wins[idx], Q_wins[idx]], axis=0)  # (2, 500)
                power_db = compute_window_power_db(window)
                all_powers_db.append(power_db)
                total_windows += 1

                # Original amplitude-threshold check
                if np.max(np.abs(I_wins[idx])) > NOISE_AMPLITUDE_THRESHOLD:
                    amplitude_active_count += 1

    all_powers_db = np.array(all_powers_db)

    # ------------------------------------------------------------------
    # Step 2: Noise floor detection from FIRST capture only
    # (as the build script does)
    # ------------------------------------------------------------------
    with h5py.File(h5_path, "r") as f:
        iq_first = load_capture_iq(f, all_keys[0])

    first_cap_powers = []
    for window, _ in windows_from_iq(iq_first):
        first_cap_powers.append(compute_window_power_db(window))
        if len(first_cap_powers) >= MAX_WINDOWS_FOR_NOISE_FLOOR:
            break

    first_cap_powers = np.array(first_cap_powers)
    noise_floor_db = detect_noise_floor_percentile(first_cap_powers, percentile=PERCENTILE)
    threshold_db   = noise_floor_db + MARGIN_DB

    # ------------------------------------------------------------------
    # Step 3: Count signal windows under each approach
    # ------------------------------------------------------------------
    power_active_count = int(np.sum(all_powers_db >= threshold_db))

    amp_duty  = amplitude_active_count / total_windows * 100
    pwr_duty  = power_active_count     / total_windows * 100
    print(f"  Sampled captures : {n_samples} of {n_captures}")
    print(f"  Windows sampled  : {total_windows:,}")
    print(f"  Noise floor (1st pct, first capture): {noise_floor_db:.2f} dB")
    print(f"  Threshold (noise floor + {MARGIN_DB} dB)  : {threshold_db:.2f} dB")
    print(f"  Power range of sampled windows: [{all_powers_db.min():.1f}, {all_powers_db.max():.1f}] dB")
    print()
    print(f"  Amplitude threshold (max|I| > {NOISE_AMPLITUDE_THRESHOLD}):")
    print(f"    Signal windows : {amplitude_active_count:,} / {total_windows:,}  ({amp_duty:.1f}%)")
    print()
    print(f"  Power threshold  (power >= {threshold_db:.1f} dB):")
    print(f"    Signal windows : {power_active_count:,} / {total_windows:,}  ({pwr_duty:.1f}%)")

    # Agreement between methods
    amp_mask = np.array([False] * total_windows)
    pwr_mask = all_powers_db >= threshold_db

    # Recompute amplitude mask per window
    amp_list = []
    with h5py.File(h5_path, "r") as f:
        for key in sampled_keys:
            iq = load_capture_iq(f, key)
            I_all = iq[:, 0]
            n = len(I_all) // WINDOW_SIZE
            I_wins = I_all[:n * WINDOW_SIZE].reshape(n, WINDOW_SIZE)
            for idx in range(n):
                amp_list.append(np.max(np.abs(I_wins[idx])) > NOISE_AMPLITUDE_THRESHOLD)
    amp_mask = np.array(amp_list)

    agree = np.sum(amp_mask == pwr_mask)
    both_signal   = np.sum(amp_mask & pwr_mask)
    amp_only      = np.sum(amp_mask & ~pwr_mask)   # amplitude says signal, power says noise
    pwr_only      = np.sum(~amp_mask & pwr_mask)   # power says signal, amplitude says noise
    both_noise    = np.sum(~amp_mask & ~pwr_mask)

    print()
    print(f"  Agreement between methods: {agree}/{total_windows} ({agree/total_windows*100:.1f}%)")
    print(f"    Both: signal  = {both_signal:,} | Both: noise = {both_noise:,}")
    print(f"    Amp=signal, Pwr=noise : {amp_only:,}  (power method MORE restrictive)")
    print(f"    Pwr=signal, Amp=noise : {pwr_only:,}  (amplitude method MORE restrictive)")

    return {
        "stem": filename_stem,
        "powers_db": all_powers_db,
        "noise_floor_db": noise_floor_db,
        "threshold_db": threshold_db,
        "amp_mask": amp_mask,
        "pwr_mask": pwr_mask,
        "first_cap_powers": first_cap_powers,
    }


def plot_results(results: list, output_dir: Path):
    fig, axes = plt.subplots(len(results), 2, figsize=(14, 4 * len(results)))
    fig.suptitle("NIST WiFi Noise Filtering Verification", fontsize=14, fontweight="bold")

    for row, res in enumerate(results):
        stem         = res["stem"]
        powers       = res["powers_db"]
        noise_floor  = res["noise_floor_db"]
        threshold    = res["threshold_db"]
        amp_mask     = res["amp_mask"]
        pwr_mask     = res["pwr_mask"]
        first_powers = res["first_cap_powers"]

        # ---- Left: power histogram with threshold lines ----
        ax = axes[row, 0]
        bins = np.linspace(powers.min(), powers.max(), 120)
        ax.hist(powers[~amp_mask], bins=bins, alpha=0.6, color="steelblue",
                label=f"Amp=noise ({(~amp_mask).sum():,})", density=True)
        ax.hist(powers[amp_mask],  bins=bins, alpha=0.6, color="tomato",
                label=f"Amp=signal ({amp_mask.sum():,})", density=True)
        ax.axvline(noise_floor, color="black",  linestyle="--", lw=1.5,
                   label=f"Noise floor {noise_floor:.1f} dB")
        ax.axvline(threshold,   color="purple", linestyle="-",  lw=2,
                   label=f"Threshold {threshold:.1f} dB")
        ax.set_xlabel("Window power (dB)")
        ax.set_ylabel("Density")
        ax.set_title(f"{stem}  — power distribution")
        ax.legend(fontsize=8)

        # ---- Right: first-capture power distribution used for noise estimation ----
        ax2 = axes[row, 1]
        ax2.hist(first_powers, bins=100, color="grey", alpha=0.7, density=True)
        ax2.axvline(noise_floor, color="black",  linestyle="--", lw=1.5,
                    label=f"1st-pct noise floor {noise_floor:.1f} dB")
        ax2.axvline(threshold,   color="purple", linestyle="-",  lw=2,
                    label=f"Threshold {threshold:.1f} dB")
        ax2.set_xlabel("Window power (dB)")
        ax2.set_ylabel("Density")
        ax2.set_title(f"{stem}  — first capture (noise floor source)")
        ax2.legend(fontsize=8)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "verify_noise_filtering.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, default=RAW_DATA_ROOT,
                        help=f"directory holding the NIST .h5 files (default: {RAW_DATA_ROOT})")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"directory for the plot (default: {OUTPUT_DIR})")
    parser.add_argument("--captures", type=int, default=N_CAPTURE_SAMPLES,
                        help=f"captures sampled per file (default: {N_CAPTURE_SAMPLES})")
    args = parser.parse_args()

    print("NIST WiFi Noise Filtering Verification")
    print(f"Noise floor method : percentile={PERCENTILE}, margin_db={MARGIN_DB}")
    print(f"Amplitude threshold: max(|I|) > {NOISE_AMPLITUDE_THRESHOLD}")
    print(f"Captures sampled   : {args.captures} of 180 per file (spread evenly)")

    results = []
    for stem in FILE_CONFIG:
        results.append(analyze_file(stem, args.raw_root, args.captures))

    plot_results(results, args.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
