"""RadioPile v1.0 — Noise filtering validation figure.

Single-panel IEEE-style figure for the Noise Filtering Validation
subsection: per-window power histogram for the NIST 5 GHz outdoor
capture, marking the detected noise floor, the +6 dB threshold, and the
filtered (dropped) region. Of the three NIST files used, this one has the
smallest separation between its lowest and highest power modes, and a
third mode between them that the threshold discards; the figure makes no
claim about what that mode contains. Computed from a 20-of-180 capture sample
of data/nist_wifi_bluetooth/raw/5ghz_outdoor.h5; noise floor detected
from the first capture (matches the build pipeline).

The Sub-GHz LoRa SF7 distribution (textbook wide-gap case) is still
computed and saved into the .npz so the two-panel variant can be
regenerated from cached data if needed, but it is no longer plotted in
this figure.

Outputs (in figures/, or under --out-dir):
  noise_filtering_validation.pdf
  noise_filtering_validation.png
  noise_filtering_validation_data.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pipeline.preprocessing import detect_noise_floor_percentile

# ── Config ────────────────────────────────────────────────────────────────────

H5_PATH        = Path("data/nist_wifi_bluetooth/raw/5ghz_outdoor.h5")
SUBGHZ_PATH    = Path("data/sub_ghz/raw/867.4/Lora/lorasf7_g0.0dB_att12dB_freq867.4MHz_0.mat")
WINDOW_SIZE    = 500
NOISE_PERCENTILE = 1.0
NOISE_AMPLITUDE_REFERENCE = 0.05   # max(|I|) > 0.05 → "signal" reference
MARGINS_DB     = [3.0, 6.0, 10.0]
N_CAPTURE_SAMPLES = 20
MAX_WINDOWS_FOR_NOISE_FLOOR = 10_000

OUT_DIR = Path("figures")


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_capture_iq(f: h5py.File, key: str) -> np.ndarray:
    """One NIST capture → float32 (N, 2) IQ."""
    raw = np.array(f[key], dtype=np.float32) / 32768.0
    return raw.reshape(-1, 2)


def _windows_powers_and_amp(iq: np.ndarray):
    """Vectorized per-window power (dB) and max|I| for one capture."""
    n = len(iq) // WINDOW_SIZE
    iq_t = iq[: n * WINDOW_SIZE].reshape(n, WINDOW_SIZE, 2)
    I, Q = iq_t[:, :, 0], iq_t[:, :, 1]
    pwr_lin = np.mean(I ** 2 + Q ** 2, axis=1)
    pwr_db = 10.0 * np.log10(pwr_lin + 1e-12)
    amp_max = np.max(np.abs(I), axis=1)
    return pwr_db.astype(np.float32), amp_max.astype(np.float32)


def collect_data(h5_path: Path, n_samples: int):
    """Sample N captures from 5GHz outdoor; return powers, amps, noise floor."""
    print(f"Loading: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        all_keys = list(f.keys())
        n_caps = len(all_keys)
        print(f"  Total captures: {n_caps}; sampling {n_samples}")

        idx_arr = np.linspace(0, n_caps - 1, n_samples, dtype=int)
        sampled_keys = [all_keys[i] for i in idx_arr]

        # Noise floor from the first capture (matches build pipeline)
        iq_first = _load_capture_iq(f, all_keys[0])
        pwr_first, _ = _windows_powers_and_amp(iq_first)
        floor_sample = pwr_first[:MAX_WINDOWS_FOR_NOISE_FLOOR]
        noise_floor_db = detect_noise_floor_percentile(
            floor_sample, percentile=NOISE_PERCENTILE
        )
        print(f"  Noise floor (1st pct, first capture): {noise_floor_db:.2f} dB")

        # Powers + amplitudes across the sample
        all_powers, all_amps = [], []
        for i, key in enumerate(sampled_keys):
            iq = _load_capture_iq(f, key)
            pwr_db, amp_max = _windows_powers_and_amp(iq)
            all_powers.append(pwr_db)
            all_amps.append(amp_max)
            if (i + 1) % 5 == 0:
                print(f"  capture {i+1:>2}/{n_samples}: "
                      f"{len(pwr_db):,} windows")

    powers = np.concatenate(all_powers)
    amps = np.concatenate(all_amps)
    print(f"  Total windows sampled: {len(powers):,}")
    print(f"  Power range: [{powers.min():.1f}, {powers.max():.1f}] dB")
    return powers, amps, float(noise_floor_db)


def collect_subghz_data(mat_path: Path):
    """Load Sub-GHz LoRa file, compute per-window power, detect noise floor.

    Sub-GHz raw data is one .mat file per recording (no first-capture concept),
    so the noise floor is detected from the whole file using the same
    percentile-based method (`detect_noise_floor_percentile`, p=1) used in
    production. Math matches `compute_window_power_lin/_db` exactly:
    mean(I² + Q²) per window, followed by 10 · log10( . + 1e-12 ).
    """
    print(f"Loading: {mat_path}")
    m = sio.loadmat(mat_path)
    if "IQ_samples" not in m:
        raise KeyError(f"No IQ_samples field in {mat_path}")
    iq_complex = m["IQ_samples"].squeeze()
    if not np.iscomplexobj(iq_complex):
        raise TypeError(f"IQ_samples is not complex in {mat_path}")

    n = len(iq_complex) // WINDOW_SIZE
    iq_t = iq_complex[: n * WINDOW_SIZE].reshape(n, WINDOW_SIZE)
    pwr_lin = np.mean(iq_t.real ** 2 + iq_t.imag ** 2, axis=1)
    powers = (10.0 * np.log10(pwr_lin + 1e-12)).astype(np.float32)

    floor_sample = powers[:MAX_WINDOWS_FOR_NOISE_FLOOR]
    noise_floor_db = float(
        detect_noise_floor_percentile(floor_sample, percentile=NOISE_PERCENTILE)
    )
    print(f"  Total windows: {len(powers):,}")
    print(f"  Power range  : [{powers.min():.1f}, {powers.max():.1f}] dB")
    print(f"  Noise floor  : {noise_floor_db:.2f} dB")
    return powers, noise_floor_db


def margin_sweep(powers: np.ndarray, amps: np.ndarray, noise_floor_db: float):
    """For each margin, compute agreement (%) with the amplitude reference."""
    amp_signal = amps > NOISE_AMPLITUDE_REFERENCE  # reference labeling
    sweep = []
    for margin in MARGINS_DB:
        threshold = noise_floor_db + margin
        pwr_signal = powers >= threshold
        agreement = float(np.mean(amp_signal == pwr_signal)) * 100.0
        signal_pct = float(np.mean(pwr_signal)) * 100.0
        sweep.append({
            "margin_db": float(margin),
            "threshold_db": float(threshold),
            "agreement_pct": agreement,
            "signal_pct": signal_pct,
        })
        print(f"  margin {margin:>4.1f} dB → threshold {threshold:>5.1f} dB → "
              f"agreement {agreement:5.1f}%  (signal {signal_pct:5.1f}%)")
    return sweep


# ── Plot ──────────────────────────────────────────────────────────────────────

FILTERED_FACECOLOR = "#8B3A3A"   # same hue as the +6 dB threshold line
FILTERED_ALPHA     = 0.10        # very see-through


def make_figure(powers, noise_floor_db, out_pdf, out_png):
    """Single-panel histogram for the NIST 5 GHz outdoor capture.

    The +6 dB threshold is the production filter's decision boundary;
    windows with power < threshold are dropped (filtered region, shaded).
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # Single-column IEEE width ≈ 3.5". Height tuned so axis labels and the
    # legend block all fit without crowding the histogram.
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # X-axis cropped to [-75, -25] dB to drop log10-floor sentinels at -120 dB
    # (zero-power windows in the capture) without compressing the bimodal
    # region into a sliver.
    x_lo, x_hi = -75.0, -25.0
    threshold_db = noise_floor_db + 6.0
    bins = np.linspace(x_lo, x_hi, 100)

    ax.hist(powers, bins=bins, color="#3E5871", alpha=0.88, edgecolor="none")

    # Translucent shading for the filtered (dropped) region — left of the
    # threshold line. Drawn with low z-order so the threshold/floor lines
    # stay crisp on top.
    ax.axvspan(x_lo, threshold_db,
               facecolor=FILTERED_FACECOLOR, alpha=FILTERED_ALPHA,
               edgecolor="none", zorder=0)

    ax.axvline(noise_floor_db, color="#D4A017", linestyle="--", lw=1.2,
               label="noise floor")
    ax.axvline(threshold_db, color="#8B3A3A", linestyle="-", lw=1.2,
               label="+6 dB threshold")

    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("Per-window power (dB)")
    ax.set_ylabel("Window count")

    # Legend in the upper-left: that quadrant is empty (the noise/signal
    # modes occupy the right half of the cropped x-range) so it doesn't
    # collide with the histogram.
    line_handles, line_labels = ax.get_legend_handles_labels()
    overlay_patch = Patch(
        facecolor=FILTERED_FACECOLOR, alpha=FILTERED_ALPHA,
        edgecolor="none", label="filtered region",
    )
    ax.legend(
        handles=line_handles + [overlay_patch],
        labels=line_labels + ["filtered region"],
        loc="upper left", frameon=False,
    )

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.06, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nist-h5", type=Path, default=H5_PATH,
                        help=f"NIST capture file (default: {H5_PATH})")
    parser.add_argument("--subghz-mat", type=Path, default=SUBGHZ_PATH,
                        help=f"Sub-GHz LoRa recording (default: {SUBGHZ_PATH})")
    parser.add_argument("--captures", type=int, default=N_CAPTURE_SAMPLES,
                        help=f"NIST captures sampled (default: {N_CAPTURE_SAMPLES})")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help=f"output directory (default: {OUT_DIR})")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Panel (a): NIST 5 GHz outdoor.
    powers_nist, amps_nist, noise_floor_nist = collect_data(args.nist_h5, args.captures)
    print("\nMargin sweep (kept for reference; no longer plotted):")
    sweep = margin_sweep(powers_nist, amps_nist, noise_floor_nist)

    # Sub-GHz LoRa SF7: still computed and saved into the .npz so the
    # two-panel variant of this figure can be regenerated from cached
    # data, but it is no longer plotted in the published figure.
    print()
    powers_subghz, noise_floor_subghz = collect_subghz_data(args.subghz_mat)

    # Save underlying data — both panels' histograms plus the cached margin
    # sweep arrays from the previous figure version (kept since they don't
    # cost much and may be useful for follow-up analysis).
    npz_path = args.out_dir / "noise_filtering_validation_data.npz"
    np.savez(
        npz_path,
        # Panel (a) — NIST 5 GHz outdoor
        nist_powers_db=powers_nist,
        nist_amp_max=amps_nist,
        nist_noise_floor_db=np.float32(noise_floor_nist),
        nist_source_file=str(args.nist_h5),
        nist_n_capture_samples=np.int32(args.captures),
        # Panel (b) — Sub-GHz LoRa SF7
        subghz_powers_db=powers_subghz,
        subghz_noise_floor_db=np.float32(noise_floor_subghz),
        subghz_source_file=str(args.subghz_mat),
        # Cached from the previous figure version (no longer plotted).
        margins_db=np.array([s["margin_db"] for s in sweep], dtype=np.float32),
        thresholds_db=np.array([s["threshold_db"] for s in sweep], dtype=np.float32),
        agreement_pct=np.array([s["agreement_pct"] for s in sweep], dtype=np.float32),
        signal_pct=np.array([s["signal_pct"] for s in sweep], dtype=np.float32),
        amp_reference=np.float32(NOISE_AMPLITUDE_REFERENCE),
        # Common
        window_size=np.int32(WINDOW_SIZE),
    )
    print(f"\nWrote {npz_path}")

    make_figure(
        powers=powers_nist, noise_floor_db=noise_floor_nist,
        out_pdf=args.out_dir / "noise_filtering_validation.pdf",
        out_png=args.out_dir / "noise_filtering_validation.png",
    )


if __name__ == "__main__":
    main()
