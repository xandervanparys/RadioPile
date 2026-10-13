"""
Dataset build script for RadChar (Radar Signal Characterisation).

All dataset configuration and assumptions are defined in config.py.

The raw HDF5 file contains complex128 IQ data. The real and imaginary parts
are extracted to form 2-channel (I/Q) float32 windows of shape (2, 512).

RadChar-Baseline is ~8 GB, so it is read in chunks to keep memory usage bounded.
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
    RAW_H5_FILENAME,
    OUTPUT_H5_PATH,
    OUTPUT_CROISSANT_PATH,
    RADAR_SIGNAL_TYPES,
    RADAR_SIGNAL_TO_WAVEFORM,
    SIGNAL_TYPES,
    WINDOW_SIZE,
    HOP_SIZE,
    PAD_LAST,
    NORMALIZE_MODE,
    NOISE_POWER_THRESHOLD_DB,
    DATA_DTYPE,
    GLOBAL_METADATA,
    WINDOW_METADATA_SCHEMA,
    CITATIONS,
    LICENSE,
)


# ---------------------------------------------------------------------
# Chunked reading parameters
# ---------------------------------------------------------------------

# 4096 samples × 512 timesteps × 16 bytes (complex128) ≈ 32 MiB per chunk
CHUNK_SIZE = 4096


# ---------------------------------------------------------------------
# Dataset-specific window iterator
# ---------------------------------------------------------------------

def iter_radchar_windows():
    """
    Yield (window, label_name, meta) tuples for all samples in RadChar.

    Reads the raw HDF5 in chunks to keep memory bounded.

    Raw HDF5 layout:
        iq:     (N, 512) complex128 — complex baseband IQ samples
        labels: (N,) structured array — signal_type, pulse params, SNR, ...

    Output window shape: (2, 512) float32 — I channel, Q channel
    """
    h5_path = RAW_DATA_ROOT / RAW_H5_FILENAME

    with h5py.File(h5_path, "r") as f:
        iq_data = f["iq"]
        label_data = f["labels"]
        n_total = iq_data.shape[0]
        n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE

        print(f"  Total samples: {n_total:,} | Reading in {n_chunks} chunks of {CHUNK_SIZE}")

        for chunk_idx, start in enumerate(range(0, n_total, CHUNK_SIZE)):
            end = min(start + CHUNK_SIZE, n_total)

            if chunk_idx % 50 == 0 or chunk_idx == n_chunks - 1:
                pct = 100 * end / n_total
                print(f"  Chunk {chunk_idx + 1}/{n_chunks} (samples {start:,}–{end:,}, {pct:.0f}%)")

            # Read chunk into memory
            iq_chunk     = iq_data[start:end]     # (chunk, 512) complex128
            labels_chunk = label_data[start:end]   # (chunk,) structured

            for i in range(iq_chunk.shape[0]):
                # Split complex128 into I and Q channels → (2, 512) float32
                iq = iq_chunk[i]
                window = np.stack(
                    [np.real(iq).astype(DATA_DTYPE), np.imag(iq).astype(DATA_DTYPE)],
                    axis=0,
                )  # shape: (2, 512)

                # Apply normalization
                window = normalize_window(window, NORMALIZE_MODE)

                # Extract label fields
                label_row = labels_chunk[i]
                signal_type_idx = int(label_row["signal_type"])
                label_name = SIGNAL_TYPES[signal_type_idx]

                meta = {
                    "source_file":   str(h5_path),
                    # Fine radar target (parallel to technology_type / modulation_type)
                    "radar_signal_type": label_name,
                    # Coarse radar intra-pulse family one level above the fine type
                    "waveform_type": RADAR_SIGNAL_TO_WAVEFORM[label_name],
                    "snr_db":        float(label_row["signal_to_noise_ratio"]),
                    "number_of_pulses": int(label_row["number_of_pulses"]),
                    "pulse_width_s":    float(label_row["pulse_width"]),
                    "time_delay_s":     float(label_row["time_delay"]),
                    "pri_s":            float(label_row["pulse_repetition_interval"]),
                    "sample_index":     start + i,
                }

                yield window, label_name, meta


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"=== Building dataset: {DATASET_NAME} ===")
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Window size: {WINDOW_SIZE}, hop size: {HOP_SIZE}, pad_last: {PAD_LAST}")
    print(f"Data dtype: {DATA_DTYPE} (extracted from complex128 IQ)")
    print(f"Normalization mode: {NORMALIZE_MODE or 'none'}")
    print(f"Radar signal types: {len(RADAR_SIGNAL_TYPES)} ({', '.join(RADAR_SIGNAL_TYPES)})")
    print(f"Chunk size: {CHUNK_SIZE} samples")
    print("Noise filtering: DISABLED (SNR levels are part of dataset design)")
    print()

    h5_path = RAW_DATA_ROOT / RAW_H5_FILENAME
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Raw HDF5 file not found: {h5_path}. "
            f"Download RadChar-Baseline.h5 from https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023"
        )

    summary = build_hdf5_dataset(
        output_path=OUTPUT_H5_PATH,
        window_iterator=iter_radchar_windows(),
        batch_size=64,
        data_dtype=DATA_DTYPE,
        global_metadata=GLOBAL_METADATA,
        window_metadata_schema=WINDOW_METADATA_SCHEMA,
    )

    print(f"\nCreated dataset with {summary['total_windows']:,} windows")
    print(f"Label names: {summary.get('label_names', [])}")
    print("Writing Croissant metadata...")

    preprocessing = {
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "pad_last": PAD_LAST,
        "normalization": NORMALIZE_MODE or "none",
    }

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
        labels=summary.get("label_names", RADAR_SIGNAL_TYPES),
        citations=CITATIONS,
        license=LICENSE,
        preprocessing=preprocessing,
        provenance={
            "upstream_dataset": "RadChar",
            "dataset_url": GLOBAL_METADATA["dataset_url"],
            "dataset_type": "Radar signal characterization",
            "snr_range_db": GLOBAL_METADATA["snr_range_db"],
            "sample_rate_hz": GLOBAL_METADATA["sample_rate_hz"],
            "script_version": SCRIPT_VERSION,
        },
    )

    print(f"Croissant metadata written to {OUTPUT_CROISSANT_PATH}")
    print(f"Dataset written to {OUTPUT_H5_PATH}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
