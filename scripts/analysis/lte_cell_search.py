"""
LTE cell search over the raw lte_wifi_dvbt captures.

One-off audit script (not part of the build or validation pipeline). It backs
the LOCATION_OVERRIDES entry in `datasets/lte_wifi_dvbt/config.py`: see
docs/DATASETS.md, "Misnamed Gentbrugge capture".

Every LTE base station broadcasts a physical cell ID (PCI, 0-503) in its
primary and secondary synchronization signals (PSS/SSS, 3GPP TS 36.211
§6.11). Those signals occupy the central 62 subcarriers (0.93 MHz) of the
carrier, so they are present in both the 1 Msps and the 10 Msps captures
centered on 806 MHz. The set of PCIs a recording hears depends on where the
receiver stood, which makes it a location fingerprint that is independent of
the filename and folder.

Method, per file:
  1. Resample to 1.92 Msps (128-point FFT at 15 kHz subcarrier spacing).
  2. Split into 0.1 s segments (a 10 Msps file is one segment, a 1 Msps
     file is eleven).
  3. Per segment and per PSS root, correlate against the PSS, normalize by
     signal energy, and average over the 5 ms PSS period. Keep up to five
     peaks that exceed six times the median.
  4. Per peak, estimate the frequency offset and channel from the PSS,
     equalize the SSS symbol that precedes it (FDD, normal cyclic prefix) and
     correlate against all 168 SSS candidates under both half-frame
     hypotheses, summed over the segment.
  5. Accept a detection when the winning candidate is more than 6 standard
     deviations above the other 167 and more than 1.3 times the runner-up.

Output: the accepted PCIs per file, the PCIs heard per location in the
1 Msps captures, and every 10 Msps capture checked against those sets.

Usage:
    python scripts/analysis/lte_cell_search.py
    python scripts/analysis/lte_cell_search.py --raw-root data/lte_wifi_dvbt/raw
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve, resample_poly

N_FFT = 128
FS = 1.92e6
HALF_FRAME = 9600          # 5 ms at 1.92 Msps: the PSS period
SEGMENT = 192000           # 0.1 s
SSS_TO_PSS = 137           # SSS symbol start to PSS symbol start, normal CP
MAX_PEAKS = 5
PSS_PEAK_OVER_MEDIAN = 6.0
MIN_Z = 6.0
MIN_RATIO = 1.3

# The 62 occupied subcarriers around DC, as FFT bin indices.
SUBCARRIERS = np.r_[np.arange(-31, 0), np.arange(1, 32)] % N_FFT


def _pss_freq(root: int) -> np.ndarray:
    n = np.arange(62)
    return np.where(
        n < 31,
        np.exp(-1j * np.pi * root * n * (n + 1) / 63),
        np.exp(-1j * np.pi * root * (n + 1) * (n + 2) / 63),
    )


def _to_time(d: np.ndarray) -> np.ndarray:
    spectrum = np.zeros(N_FFT, complex)
    spectrum[SUBCARRIERS] = d
    return np.fft.ifft(spectrum) * np.sqrt(N_FFT)


def _m_sequence(taps: tuple) -> np.ndarray:
    x = [0, 0, 0, 0, 1]
    for i in range(26):
        x.append(sum(x[i + t] for t in taps) % 2)
    return 1 - 2 * np.array(x)


def _sss(nid1: int, nid2: int) -> tuple:
    """SSS for subframe 0 and subframe 5 (TS 36.211 §6.11.2.1)."""
    qp = nid1 // 30
    q = (nid1 + qp * (qp + 1) // 2) // 30
    mp = nid1 + q * (q + 1) // 2
    m0 = mp % 31
    m1 = (m0 + mp // 31 + 1) % 31
    n = np.arange(31)
    s0, s1 = _S[(n + m0) % 31], _S[(n + m1) % 31]
    c0, c1 = _C[(n + nid2) % 31], _C[(n + nid2 + 3) % 31]
    z0, z1 = _Z[(n + m0 % 8) % 31], _Z[(n + m1 % 8) % 31]
    sf0, sf5 = np.zeros(62), np.zeros(62)
    sf0[0::2], sf0[1::2] = s0 * c0, s1 * c1 * z0
    sf5[0::2], sf5[1::2] = s1 * c0, s0 * c1 * z1
    return sf0, sf5


_S = _m_sequence((2, 0))
_C = _m_sequence((3, 0))
_Z = _m_sequence((4, 2, 1, 0))
PSS_F = {nid2: _pss_freq(root) for nid2, root in enumerate((25, 29, 34))}
PSS_T = {nid2: _to_time(d) for nid2, d in PSS_F.items()}
SSS = {nid2: np.array([_sss(g, nid2) for g in range(168)]) for nid2 in range(3)}


def load_resampled(path: Path, sample_rate_hz: float) -> np.ndarray:
    x = np.fromfile(path, dtype=np.complex64).astype(np.complex128)
    ratio = Fraction(int(FS), int(sample_rate_hz))
    return resample_poly(x, ratio.numerator, ratio.denominator)


def search(x: np.ndarray) -> list:
    """Return accepted detections as (segment, pci, z, ratio)."""
    found = []
    for start in range(0, len(x) - SEGMENT + 1, SEGMENT):
        seg = x[start:start + SEGMENT]
        energy = np.convolve(np.abs(seg) ** 2, np.ones(N_FFT), "valid") + 1e-20
        for nid2 in range(3):
            corr = fftconvolve(seg, np.conj(PSS_T[nid2][::-1]), "valid")
            metric = np.abs(corr) ** 2 / energy
            usable = (len(metric) // HALF_FRAME) * HALF_FRAME
            profile = metric[:usable].reshape(-1, HALF_FRAME).mean(0)
            median = np.median(profile)
            for _ in range(MAX_PEAKS):
                tau = int(np.argmax(profile))
                if profile[tau] < PSS_PEAK_OVER_MEDIAN * median:
                    break
                profile[max(0, tau - 20):tau + 21] = 0

                acc = np.zeros((168, 2))
                for k in range(SEGMENT // HALF_FRAME):
                    t = tau + k * HALF_FRAME
                    if t - SSS_TO_PSS < 0 or t + N_FFT > len(seg):
                        continue
                    # Frequency offset from the two halves of the PSS.
                    r = seg[t:t + N_FFT] * np.conj(PSS_T[nid2])
                    half = N_FFT // 2
                    w = np.angle(np.sum(r[half:]) * np.conj(np.sum(r[:half]))) / half
                    block = seg[t - SSS_TO_PSS:t + N_FFT]
                    block = block * np.exp(-1j * w * np.arange(-SSS_TO_PSS, N_FFT))
                    y_pss = np.fft.fft(block[SSS_TO_PSS:])[SUBCARRIERS]
                    y_sss = np.fft.fft(block[:N_FFT])[SUBCARRIERS]
                    channel = y_pss * np.conj(PSS_F[nid2])
                    scores = np.real(SSS[nid2] @ (y_sss * np.conj(channel)))  # (168, 2)
                    # Consecutive PSS occurrences alternate subframe 0 / 5.
                    acc[:, 0] += scores[:, k % 2]
                    acc[:, 1] += scores[:, 1 - k % 2]

                group, _ = np.unravel_index(np.argmax(acc), acc.shape)
                best = acc[group].max()
                others = np.delete(acc.max(1), group)
                z = (best - others.mean()) / (others.std() + 1e-30)
                ratio = best / (others.max() + 1e-30)
                if z > MIN_Z and ratio > MIN_RATIO:
                    found.append((start // SEGMENT, int(3 * group + nid2), float(z), float(ratio)))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-root", type=Path, default=Path("data/lte_wifi_dvbt/raw"))
    args = parser.parse_args()

    files = sorted(args.raw_root.glob("*/lte*.bin"))
    per_location = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # folder -> pci -> [files, segments]
    high_rate = []

    print(f"=== LTE cell search: {len(files)} files under {args.raw_root} ===")
    for path in files:
        is_10msps = "10Msps" in path.name
        x = load_resampled(path, 10e6 if is_10msps else 1e6)
        detections = search(x)
        segments = max(1, len(x) // SEGMENT)
        counts = Counter(pci for _, pci, _, _ in detections)
        listing = " ".join(f"{pci}x{n}" for pci, n in counts.most_common())
        print(f"{path.parent.name + '/' + path.name:52s} segments={segments:2d}  PCI: {listing}")
        if is_10msps:
            high_rate.append((path, detections))
        else:
            for pci, n in counts.items():
                per_location[path.parent.name][pci][0] += 1
                per_location[path.parent.name][pci][1] += n

    print("\n=== PCIs heard per location (1 Msps captures): pci(files, segments) ===")
    owner = defaultdict(set)
    for folder, cells in sorted(per_location.items()):
        ranked = sorted(cells.items(), key=lambda kv: -kv[1][1])
        print(f"{folder:11s}", " ".join(f"{pci}({f},{s})" for pci, (f, s) in ranked))
        for pci in cells:
            owner[pci].add(folder)
    shared = {pci: sorted(folders) for pci, folders in owner.items() if len(folders) > 1}
    print(f"PCIs heard at more than one location: {shared or 'none'}")

    print("\n=== 10 Msps captures against the per-location sets ===")
    for path, detections in high_rate:
        cells = sorted({pci for _, pci, _, _ in detections})
        matches = sorted({folder for pci in cells for folder in owner.get(pci, ())})
        token = path.stem.split("_")[2]
        scores = " ".join(f"{pci}(z={z:.1f})" for _, pci, z, _ in sorted(detections, key=lambda d: -d[2]))
        flag = ""
        if matches and matches != [token]:
            flag = "   <-- filename location disagrees"
        print(f"{path.parent.name + '/' + path.name:52s} filename={token:10s} cells match={','.join(matches) or '-':10s} {scores}{flag}")


if __name__ == "__main__":
    main()
