"""
Visual verification of NIST noise filtering: signal windows vs. filtered windows.

For each of the 3 included files shows a grid of example windows:
  - Left columns: windows that PASS the power threshold (signal)
  - Right columns: windows that FAIL the power threshold (filtered/noise)
Each window is shown as a spectrogram (time-frequency).

Usage:
    python scripts/analysis/verify_nist_sample_quality.py [--raw-root DIR] [--output-dir DIR] [--examples N]
"""

import argparse

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.preprocessing import compute_window_power_db, detect_noise_floor_percentile
from pipeline.windowing import iter_windows_from_samples

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

RAW_DATA_ROOT = Path("data/nist_wifi_bluetooth/raw")
OUTPUT_DIR    = Path("data/nist_wifi_bluetooth/processed/plots")  # never write into raw/

FILE_CONFIG = {
    "2_4ghz_indoor": (2.437e9, "indoor"),
    "5ghz_indoor":   (5.825e9, "indoor"),
    "5ghz_outdoor":  (5.825e9, "outdoor"),
}

WINDOW_SIZE  = 500
HOP_SIZE     = 500
PERCENTILE   = 1.0
MARGIN_DB    = 6.0
SAMPLING_RATE_MSPS = 30

N_EXAMPLES   = 5   # default examples per category (signal / filtered)


def load_capture_iq(f: h5py.File, key: str) -> np.ndarray:
    raw = np.array(f[key], dtype=np.float32) / 32768.0
    return raw.reshape(-1, 2)  # (N, 2)


def get_threshold(h5_path: Path, all_keys: list) -> float:
    """Detect noise floor from first capture and return threshold."""
    with h5py.File(h5_path, "r") as f:
        iq = load_capture_iq(f, all_keys[0])
    powers = []
    for window, _ in iter_windows_from_samples(iq, window_size=WINDOW_SIZE,
                                               hop_size=HOP_SIZE, pad_last=False):
        powers.append(compute_window_power_db(window))
        if len(powers) >= 10_000:
            break
    noise_floor = detect_noise_floor_percentile(np.array(powers), percentile=PERCENTILE)
    return noise_floor + MARGIN_DB


def collect_examples(h5_path: Path, all_keys: list, threshold_db: float,
                     n: int = N_EXAMPLES):
    """
    Walk captures (spread across the file) and collect n signal and n noise windows.
    Returns two lists of (2, 500) float32 arrays.
    """
    signal_windows  = []
    noise_windows   = []

    # Use captures spread across the file to sample diverse conditions
    step = max(1, len(all_keys) // (n * 4))
    keys_to_scan = all_keys[::step]

    with h5py.File(h5_path, "r") as f:
        for key in keys_to_scan:
            if len(signal_windows) >= n and len(noise_windows) >= n:
                break
            iq = load_capture_iq(f, key)
            I_all = iq[:, 0]
            Q_all = iq[:, 1]
            nn = len(I_all) // WINDOW_SIZE
            I_wins = I_all[:nn * WINDOW_SIZE].reshape(nn, WINDOW_SIZE)
            Q_wins = Q_all[:nn * WINDOW_SIZE].reshape(nn, WINDOW_SIZE)
            for idx in range(nn):
                window = np.stack([I_wins[idx], Q_wins[idx]], axis=0)
                power_db = compute_window_power_db(window)
                if power_db >= threshold_db and len(signal_windows) < n:
                    signal_windows.append(window)
                elif power_db < threshold_db and len(noise_windows) < n:
                    noise_windows.append(window)
                if len(signal_windows) >= n and len(noise_windows) >= n:
                    break

    return signal_windows, noise_windows


def spectrogram_of_window(window: np.ndarray, ax: plt.Axes,
                           fs_mhz: float = SAMPLING_RATE_MSPS,
                           title: str = "", power_db: float = None):
    """
    Plot a short-time spectrogram of a (2, W) IQ window.
    Uses the complex I+jQ representation for FFT-based spectrogram.
    """
    iq_complex = window[0] + 1j * window[1]   # (W,)

    # STFT parameters: 32-sample FFT with 50% overlap gives ~30 time bins for W=500
    nperseg = 32
    noverlap = 16
    from scipy.signal import spectrogram as scipy_spec
    f, t, Sxx = scipy_spec(iq_complex, fs=fs_mhz, window="hann",
                            nperseg=nperseg, noverlap=noverlap,
                            return_onesided=False, scaling="density")

    # Shift zero-frequency to center
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)

    Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-12)
    vmin, vmax = np.percentile(Sxx_db, [5, 99])

    ax.imshow(Sxx_db, aspect="auto", origin="lower",
              extent=[t[0], t[-1], f[0], f[-1]],
              cmap="inferno", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Time (µs)", fontsize=7)
    ax.set_ylabel("Freq (MHz)", fontsize=7)
    ax.tick_params(labelsize=6)

    pwr_str = f"  [{power_db:.1f} dB]" if power_db is not None else ""
    ax.set_title(f"{title}{pwr_str}", fontsize=7, pad=2)


def plot_file(stem: str, signal_windows, noise_windows, threshold_db: float,
              output_dir: Path, n_examples: int):
    n = min(len(signal_windows), len(noise_windows), n_examples)
    ncols = n * 2   # left half: signal, right half: noise
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2.2, 3.2))
    fig.suptitle(
        f"{stem}   |   threshold = {threshold_db:.1f} dB   "
        f"(noise floor + {MARGIN_DB} dB)\n"
        f"Left {n} cols: SIGNAL windows (pass)   "
        f"|   Right {n} cols: FILTERED windows (fail)",
        fontsize=9, fontweight="bold"
    )

    for col in range(n):
        win   = signal_windows[col]
        pdb   = compute_window_power_db(win)
        ax    = axes[col]
        label = f"signal #{col+1}" if col > 0 else "▶ SIGNAL"
        spectrogram_of_window(win, ax, title=label, power_db=pdb)
        for spine in ax.spines.values():
            spine.set_edgecolor("green")
            spine.set_linewidth(1.5)

    for col in range(n):
        win   = noise_windows[col]
        pdb   = compute_window_power_db(win)
        ax    = axes[n + col]
        label = f"filtered #{col+1}" if col > 0 else "▶ FILTERED"
        spectrogram_of_window(win, ax, title=label, power_db=pdb)
        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(1.5)

    # Divider between signal and noise columns
    mid_x = (axes[n - 1].get_position().x1 + axes[n].get_position().x0) / 2
    fig.add_artist(plt.Line2D(
        [mid_x, mid_x], [0.08, 0.92],
        transform=fig.transFigure, color="grey", linewidth=1.5, linestyle="--"
    ))

    plt.tight_layout(rect=[0, 0, 1, 0.88])
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"sample_quality_{stem}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved {out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-root", type=Path, default=RAW_DATA_ROOT,
                        help=f"directory holding the NIST .h5 files (default: {RAW_DATA_ROOT})")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"directory for the plots (default: {OUTPUT_DIR})")
    parser.add_argument("--examples", type=int, default=N_EXAMPLES,
                        help=f"example windows per category (default: {N_EXAMPLES})")
    args = parser.parse_args()

    print("NIST WiFi Sample Quality Verification")
    print(f"Showing {args.examples} signal + {args.examples} filtered windows per file\n")

    for stem in FILE_CONFIG:
        h5_path = args.raw_root / f"{stem}.h5"
        print(f"Processing {stem}.h5 ...")

        with h5py.File(h5_path, "r") as f:
            all_keys = list(f.keys())

        threshold_db = get_threshold(h5_path, all_keys)
        print(f"  Threshold: {threshold_db:.1f} dB")

        signal_w, noise_w = collect_examples(h5_path, all_keys, threshold_db, args.examples)
        print(f"  Collected {len(signal_w)} signal + {len(noise_w)} noise windows")

        plot_file(stem, signal_w, noise_w, threshold_db, args.output_dir, args.examples)

    print("\nDone.")


if __name__ == "__main__":
    main()
