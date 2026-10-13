"""
Visualize the same class as represented by different source datasets.

For a given class (technology_type or modulation_type) in the aggregated
RadioPile corpus, picks example windows from each contributing source (split by
sample rate when a source contributes at multiple rates) and renders three
side-by-side views per example: raw IQ time series, FFT magnitude spectrum,
STFT spectrogram.

Intended as an exploratory / paper-figure script — shows how "same class,
different source" actually looks across acquisition modalities.

Usage:
    python scripts/analysis/visualize_class_across_sources.py 64QAM
    python scripts/analysis/visualize_class_across_sources.py WiFi --label-type technology
    python scripts/analysis/visualize_class_across_sources.py BPSK --samples-per-source 2
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft


AGG_H5 = Path("data/aggregation/processed/aggregated.h5")
OUT_DIR = Path("data/aggregation/processed/plots/class_across_sources")

STFT_NPERSEG  = 64
STFT_NOVERLAP = 56
FFT_N         = 1024


def _decode(arr: np.ndarray) -> np.ndarray:
    if arr.dtype.kind == "O":
        return np.array([s.decode() if isinstance(s, bytes) else s for s in arr])
    return arr


def fmt_rate(sr: float | None) -> str:
    if sr is None or not np.isfinite(sr):
        return "simulated"
    if sr >= 1e6:
        return f"{sr/1e6:g} MHz"
    if sr >= 1e3:
        return f"{sr/1e3:g} kHz"
    return f"{sr:g} Hz"


def fmt_duration(window_size: int, sr: float | None) -> str:
    if sr is None or not np.isfinite(sr) or sr <= 0:
        return f"{window_size} samples (no physical rate)"
    secs = window_size / sr
    if secs >= 1e-3:
        return f"{secs*1e3:.3g} ms"
    return f"{secs*1e6:.3g} µs"


def compute_fft(iq: np.ndarray, sr: float | None, n: int = FFT_N):
    spec = np.fft.fftshift(np.fft.fft(iq, n=n))
    mag_db = 20 * np.log10(np.abs(spec) + 1e-12)
    if sr and np.isfinite(sr):
        freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1 / sr))
        return freqs, mag_db, "Hz"
    freqs = np.fft.fftshift(np.fft.fftfreq(n))  # cycles/sample
    return freqs, mag_db, "norm"


def compute_stft(iq: np.ndarray, sr: float | None):
    fs = sr if sr and np.isfinite(sr) else 1.0
    f, t, Zxx = stft(
        iq, fs=fs,
        nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
        return_onesided=False, boundary=None, padded=False,
    )
    order = np.argsort(np.fft.fftshift(f))
    f = np.fft.fftshift(f)
    Zxx = Zxx[order, :]
    mag_db = 20 * np.log10(np.abs(Zxx) + 1e-12)
    unit = "Hz" if (sr and np.isfinite(sr)) else "norm"
    return f, t, mag_db, unit


def plot_row(axes, iq: np.ndarray, sr: float | None, label: str,
             was_padded: bool, first_row: bool) -> None:
    window = iq.shape[0]
    if sr and np.isfinite(sr):
        t_axis = np.arange(window) / sr * 1e6   # µs
        t_unit = "µs"
    else:
        t_axis = np.arange(window)
        t_unit = "samples"

    ax = axes[0]
    ax.plot(t_axis, iq.real, color="C0", linewidth=0.7, label="I")
    ax.plot(t_axis, iq.imag, color="C3", linewidth=0.7, label="Q", alpha=0.8)
    if was_padded:
        # Find last non-zero sample to mark the native-content boundary.
        nz = np.where(np.abs(iq) > 0)[0]
        if nz.size and nz.max() < window - 1:
            ax.axvspan(t_axis[nz.max() + 1], t_axis[-1],
                       color="gray", alpha=0.15, label="padding")
    ax.set_ylabel(label, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(f"Time ({t_unit})", fontsize=8)
    if first_row:
        ax.set_title("IQ (time domain)", fontsize=11)
    ax.legend(loc="upper right", fontsize=7)

    ax = axes[1]
    freqs, mag_db, funit = compute_fft(iq, sr)
    x = freqs / 1e6 if funit == "Hz" else freqs
    xlabel = "Frequency (MHz)" if funit == "Hz" else "Normalized freq (cycles/sample)"
    ax.plot(x, mag_db, color="steelblue", linewidth=0.7)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Magnitude (dB)", fontsize=8)
    if first_row:
        ax.set_title("FFT", fontsize=11)

    ax = axes[2]
    f, t, S_db, funit = compute_stft(iq, sr)
    if funit == "Hz":
        x_t = t * 1e6; y_f = f / 1e6
        ax.set_xlabel("Time (µs)", fontsize=8)
        ax.set_ylabel("Frequency (MHz)", fontsize=8)
    else:
        x_t = t; y_f = f
        ax.set_xlabel("Time (samples)", fontsize=8)
        ax.set_ylabel("Norm freq", fontsize=8)
    im = ax.pcolormesh(
        x_t, y_f, S_db, shading="auto", cmap="viridis",
        vmin=np.percentile(S_db, 5), vmax=np.percentile(S_db, 99),
    )
    if first_row:
        ax.set_title("STFT", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("class_name", help="Class label to visualize (e.g. '64QAM', 'WiFi')")
    ap.add_argument("--label-type", choices=["modulation", "technology"],
                    default="modulation",
                    help="Which label field to query (default: modulation)")
    ap.add_argument("--h5", type=Path, default=AGG_H5)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--samples-per-source", type=int, default=1,
                    help="Example windows per (source, sample-rate) combo (default: 1)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.h5.exists():
        print(f"ERROR: {args.h5} not found")
        return 1

    label_field = f"meta/{args.label_type}_type"

    print(f"Reading {label_field} from {args.h5} ...")
    with h5py.File(args.h5, "r") as f:
        labels   = _decode(f[label_field][:])
        sources  = _decode(f["meta/source_dataset"][:])
        sr_arr   = f["meta/sample_rate_hz"][:].astype(np.float64)
        padded   = f["meta/was_padded"][:].astype(bool)

        mask = labels == args.class_name
        n_match = int(mask.sum())
        print(f"  '{args.class_name}': {n_match:,} windows")
        if n_match == 0:
            classes = sorted(set(labels.tolist()))
            print(f"  No matches. Available {args.label_type} classes: {classes}")
            return 1

        idx_all = np.where(mask)[0]
        sr_all  = sr_arr[idx_all]
        src_all = sources[idx_all]

        # Group by (source, rate-bucket).
        SENT = -1.0
        sr_bucket = np.where(np.isfinite(sr_all), sr_all, SENT)
        groups: dict[tuple[str, float], list[int]] = defaultdict(list)
        for i, src, r in zip(idx_all, src_all, sr_bucket):
            groups[(src, float(r))].append(int(i))
        # Sort groups by total count desc.
        group_order = sorted(groups.keys(), key=lambda k: -len(groups[k]))

        rng = np.random.default_rng(args.seed)
        chosen: list[tuple[str, float | None, np.ndarray, bool]] = []
        for (src, r) in group_order:
            pool = np.array(groups[(src, r)])
            n_take = min(args.samples_per_source, pool.size)
            picks = rng.choice(pool, size=n_take, replace=False)
            sr_label = None if r == SENT else r
            for p in picks:
                window = f["data"][p]  # (2, W)
                iq = window[0] + 1j * window[1]
                chosen.append((src, sr_label, iq, bool(padded[p])))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 2.7 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row, (src, sr, iq, was_pad) in enumerate(chosen):
        row_label = f"{src}\n{fmt_rate(sr)}\n({fmt_duration(iq.shape[0], sr)})"
        if was_pad:
            row_label += "\n[padded]"
        plot_row(axes[row], iq, sr, row_label, was_pad, first_row=(row == 0))

    fig.suptitle(
        f"RadioPile — class '{args.class_name}' ({args.label_type}) "
        f"across {len({s for s, _, _, _ in chosen})} source dataset(s)",
        fontsize=13,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    safe_name = args.class_name.replace(" ", "_").replace("/", "-")
    out_path = args.out_dir / f"{args.label_type}_{safe_name}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
