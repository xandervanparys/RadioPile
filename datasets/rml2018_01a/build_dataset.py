"""
Dataset build script for RadioML 2018.01A.

All dataset configuration and assumptions are defined in config.py.

The raw HDF5 file is ~21 GB, so it is read in chunks to avoid loading
the entire dataset into memory.
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
    MODULATION_TYPES,
    MODULATION_TO_WAVEFORM,
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

# Number of samples to read per chunk from the 21 GB HDF5 file.
# 4096 samples × 1024 timesteps × 2 channels × 4 bytes ≈ 32 MiB per chunk.
CHUNK_SIZE = 4096


# ---------------------------------------------------------------------
# Dataset-specific window iterator
# ---------------------------------------------------------------------

def iter_rml2018_01a_windows():
    """
    Yield (window, label_name, meta) tuples for all samples in RadioML 2018.01A.

    Reads the raw HDF5 in chunks to keep memory usage bounded.
    Each sample is a pre-windowed 1024-point IQ capture.

    Raw HDF5 layout:
        X: (2555904, 1024, 2) float32 — IQ samples [time, I/Q]
        Y: (2555904, 24) int64 — one-hot labels (scrambled index order)
        Z: (2555904, 1) int64 — SNR in dB
    """
    h5_path = RAW_DATA_ROOT / RAW_H5_FILENAME

    with h5py.File(h5_path, "r") as f:
        X = f["X"]  # Do not load — access as h5py dataset
        Y = f["Y"]
        Z = f["Z"]
        n_total = X.shape[0]
        n_chunks = (n_total + CHUNK_SIZE - 1) // CHUNK_SIZE

        print(f"  Total samples: {n_total:,} | Reading in {n_chunks} chunks of {CHUNK_SIZE}")

        for chunk_idx, start in enumerate(range(0, n_total, CHUNK_SIZE)):
            end = min(start + CHUNK_SIZE, n_total)

            if chunk_idx % 50 == 0 or chunk_idx == n_chunks - 1:
                pct = 100 * end / n_total
                print(f"  Chunk {chunk_idx + 1}/{n_chunks} (samples {start:,}–{end:,}, {pct:.0f}%)")

            # Read chunk into memory
            x_chunk = X[start:end]   # (chunk, 1024, 2) float32
            y_chunk = Y[start:end]   # (chunk, 24) int64
            z_chunk = Z[start:end]   # (chunk, 1) int64

            # Decode one-hot labels to class indices
            label_indices = np.argmax(y_chunk, axis=1)

            for i in range(x_chunk.shape[0]):
                # Transpose from (1024, 2) to (2, 1024) — pipeline convention
                window = x_chunk[i].T.astype(DATA_DTYPE)

                # Apply normalization
                window = normalize_window(window, NORMALIZE_MODE)

                # Map label index to modulation name (corrected order)
                label_idx = int(label_indices[i])
                label_name = MODULATION_TYPES[label_idx]

                snr_db = float(z_chunk[i, 0])

                meta = {
                    "source_file":    str(h5_path),
                    "modulation_type": label_name,
                    "waveform_type":  MODULATION_TO_WAVEFORM.get(label_name, "unknown"),
                    "snr_db":         snr_db,
                    "sample_index":   start + i,
                }

                yield window, label_name, meta


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"=== Building dataset: {DATASET_NAME} ===")
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Window size: {WINDOW_SIZE}, hop size: {HOP_SIZE}, pad_last: {PAD_LAST}")
    print(f"Data dtype: {DATA_DTYPE}")
    print(f"Normalization mode: {NORMALIZE_MODE or 'none'}")
    print(f"Modulation classes: {len(MODULATION_TYPES)} ({', '.join(MODULATION_TYPES)})")
    print(f"Chunk size: {CHUNK_SIZE} samples")
    print("Noise filtering: DISABLED (SNR levels are part of dataset design)")
    print()

    h5_path = RAW_DATA_ROOT / RAW_H5_FILENAME
    if not h5_path.exists():
        raise FileNotFoundError(
            f"Raw HDF5 file not found: {h5_path}. "
            f"Download GOLD_XYZ_OSC.0001_1024.hdf5 from https://www.deepsig.ai/datasets"
        )

    summary = build_hdf5_dataset(
        output_path=OUTPUT_H5_PATH,
        window_iterator=iter_rml2018_01a_windows(),
        batch_size=64,
        data_dtype=DATA_DTYPE,
        global_metadata=GLOBAL_METADATA,
        window_metadata_schema=WINDOW_METADATA_SCHEMA,
    )

    print(f"\nCreated dataset with {summary['total_windows']} windows")
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
        labels=summary.get("label_names", MODULATION_TYPES),
        citations=CITATIONS,
        license=LICENSE,
        preprocessing=preprocessing,
        provenance={
            "upstream_dataset": "RadioML 2018.01A",
            "dataset_url": GLOBAL_METADATA["dataset_url"],
            "dataset_type": "Modulation classification",
            "snr_range_db": GLOBAL_METADATA["snr_range_db"],
            "script_version": SCRIPT_VERSION,
        },
    )

    print(f"Croissant metadata written to {OUTPUT_CROISSANT_PATH}")
    print(f"Dataset written to {OUTPUT_H5_PATH}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
