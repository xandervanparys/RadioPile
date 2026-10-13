"""
Dataset build script for ITS Band Technology Recognition (Girmay et al.).

All configuration and assumptions are in config.py.

Each .mat file contains one technology's IQ data, read by h5py as a
(2200, 7500) float64 array. The build:
  1. Transposes it to (7500, 2200), one row per observation
  2. Splits the 2200 values into I (first 1100) and Q (last 1100)
  3. Splits each 1100-sample observation into 2×500 windows (discard last 100)
  4. Applies unit_power normalization per window
  5. Yields each window with its technology label and recording_id metadata

Note: the source dataset adds synthetic AWGN at multiple SNR levels on top of
real-world captures (Girmay et al. §4.2). RadioPile stores raw IQ; the original
paper uses FFT of IQ as model input.
"""

from pathlib import Path

import h5py
import numpy as np

from pipeline.hdf5_writer import build_hdf5_dataset
from pipeline.preprocessing import normalize_window
from metadata.croissant import generate_croissant_from_hdf5
from datasets.identifiers import RELEASE_DATE, RELEASE_VERSION

from .config import (
    DATASET_NAME,
    SCRIPT_VERSION,
    RAW_DATA_ROOT,
    OUTPUT_H5_PATH,
    OUTPUT_CROISSANT_PATH,
    FILE_CONFIG,
    TECHNOLOGY_TYPES,
    LABEL_TO_WAVEFORM,
    LABEL_TO_WIFI_STANDARD,
    OBSERVATION_SIZE,
    WINDOW_SIZE,
    HOP_SIZE,
    PAD_LAST,
    NORMALIZE_MODE,
    DATA_DTYPE,
    GLOBAL_METADATA,
    WINDOW_METADATA_SCHEMA,
    SAMPLE_RATE_HZ,
    CITATIONS,
    LICENSE,
)


def iter_its_band_windows():
    """
    Yield (window, label_name, meta) for all samples across all technology files.

    For each file:
        mat variable shape: (2200, 7500) float64  [MATLAB column-major]
        Python transpose:   (7500, 2200) float64
        Per sample:         2200 floats → I = [:1100], Q = [1100:]
        Split:              2×500 windows per observation (last 100 samples discarded)
        Output window:      (2, 500) float32
    """
    n_splits = OBSERVATION_SIZE // WINDOW_SIZE  # 1100 // 500 = 2

    for filename_stem, (mat_var, label) in FILE_CONFIG.items():
        mat_path = RAW_DATA_ROOT / f"{filename_stem}.mat"

        if not mat_path.exists():
            raise FileNotFoundError(
                f"Expected file not found: {mat_path}\n"
                f"Download {filename_stem}.mat from the dataset authors."
            )

        print(f"  Loading {filename_stem}.mat → label '{label}'")

        with h5py.File(mat_path, "r") as f:
            # (2200, 7500) float64 → transpose to (7500, 2200)
            data = f[mat_var][:].T  # (7500, 2200) float64

        n_samples = data.shape[0]
        print(f"    {n_samples} observations × {n_splits} splits = {n_samples * n_splits} windows")

        for i in range(n_samples):
            sample = data[i]                               # (2200,) float64
            I = sample[:OBSERVATION_SIZE].astype(DATA_DTYPE)    # (1100,)
            Q = sample[OBSERVATION_SIZE:].astype(DATA_DTYPE)    # (1100,)

            obs_id = f"{filename_stem}_{i}"

            for s in range(n_splits):
                start = s * WINDOW_SIZE
                end = start + WINDOW_SIZE
                window = np.stack([I[start:end], Q[start:end]], axis=0)  # (2, 500)

                window = normalize_window(window, NORMALIZE_MODE)

                meta = {
                    "source_file":     str(mat_path),
                    "technology_type": label,
                    "modulation_type": "unknown",                          # MCS not stored per-sample in released data
                    "waveform_type":   LABEL_TO_WAVEFORM.get(label, "unknown"),
                    "wifi_standard":   LABEL_TO_WIFI_STANDARD.get(label, ""),
                    "center_frequency_hz":  5.9e9,       # V2X ITS band, fixed across all captures
                    "sample_rate_hz":  SAMPLE_RATE_HZ,
                    "sample_index":    i,
                    "recording_id":    obs_id,
                }

                yield window, label, meta


def main():
    print(f"=== Building dataset: {DATASET_NAME} ===")
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Sampling rate: {SAMPLE_RATE_HZ / 1e6:g} Msps → observation size: {OBSERVATION_SIZE} samples (44 µs) → {OBSERVATION_SIZE // WINDOW_SIZE}×{WINDOW_SIZE} windows")
    print(f"Data dtype: {DATA_DTYPE} (cast from float64)")
    print(f"Normalization mode: {NORMALIZE_MODE or 'none'}")
    print(f"Technologies ({len(FILE_CONFIG)}): {', '.join(TECHNOLOGY_TYPES)}")
    print()

    missing = [
        f"{stem}.mat"
        for stem in FILE_CONFIG
        if not (RAW_DATA_ROOT / f"{stem}.mat").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing source files in {RAW_DATA_ROOT}:\n"
            + "\n".join(f"  {f}" for f in missing)
        )

    summary = build_hdf5_dataset(
        output_path=OUTPUT_H5_PATH,
        window_iterator=iter_its_band_windows(),
        batch_size=64,
        data_dtype=DATA_DTYPE,
        global_metadata=GLOBAL_METADATA,
        window_metadata_schema=WINDOW_METADATA_SCHEMA,
    )

    print(f"\nCreated dataset with {summary['total_windows']:,} windows")
    print(f"Label names: {summary.get('label_names', [])}")
    print("Writing Croissant metadata...")

    generate_croissant_from_hdf5(
        h5_path=OUTPUT_H5_PATH,
        output_path=OUTPUT_CROISSANT_PATH,
        doi=GLOBAL_METADATA["doi"],
        related_identifiers=GLOBAL_METADATA["related_identifiers"],
        version=RELEASE_VERSION,
        date_published=RELEASE_DATE,
        dataset_name=DATASET_NAME,
        description=GLOBAL_METADATA["description"],
        task_type="classification",
        labels=summary.get("label_names", TECHNOLOGY_TYPES),
        citations=CITATIONS,
        license=LICENSE,
        preprocessing={
            "window_size": WINDOW_SIZE,
            "hop_size": HOP_SIZE,
            "pad_last": PAD_LAST,
            "normalization": NORMALIZE_MODE or "none",
        },
        provenance={
            "upstream_dataset": "ITS Band Technology Recognition (Girmay et al.)",
            "dataset_url": GLOBAL_METADATA["dataset_url"],
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "center_frequency_hz": GLOBAL_METADATA["center_frequency_hz"],
            "script_version": SCRIPT_VERSION,
        },
    )

    print(f"Croissant metadata written to {OUTPUT_CROISSANT_PATH}")
    print(f"Dataset written to {OUTPUT_H5_PATH}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
