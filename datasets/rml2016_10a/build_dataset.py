"""
Dataset build script for RadioML 2016.10A.

All dataset configuration and assumptions are defined in config.py.
"""
from pathlib import Path
import pickle
import warnings

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
    LABEL_REMAP,
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
# Dataset-specific parsing
# ---------------------------------------------------------------------

def decode_bytes_to_str(value):
    """Convert bytes or byte strings to regular strings."""
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore')
    elif isinstance(value, tuple):
        return tuple(decode_bytes_to_str(v) for v in value)
    else:
        return value


def load_rml2016_10a_data(pickle_path: Path):
    """
    Load RadioML 2016.10A data from pickle file.

    Returns dictionary with keys (modulation_type_str, snr_int) -> numpy array
    of shape (1000, 2, 128).
    """
    with open(pickle_path, 'rb') as f:
        # Suppress numpy 2.4 deprecation warning caused by dtype serialization
        # in the original RML2016.10a pickle (third-party data, cannot be changed).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*dtype.*align.*")
            data = pickle.load(f, encoding='bytes')

    # Convert bytes keys to strings for easier handling
    decoded_data = {}
    for (mod_bytes, snr), value in data.items():
        decoded_data[(decode_bytes_to_str(mod_bytes), int(snr))] = value

    return decoded_data


# ---------------------------------------------------------------------
# Dataset-specific window iterator
# ---------------------------------------------------------------------

def iter_rml2016_10a_windows(pickle_path: Path):
    """
    Yield (window, label_name, meta) tuples for all samples in RadioML 2016.10A.

    This iterator processes a pickle file and yields windows from
    all (modulation, SNR) combinations.

    Parameters
    ----------
    pickle_path : Path
        Path to the pickle file (RML2016.10a_dict.pkl)
    """
    # Only process pickle files
    if not (pickle_path.suffix.lower() in ['.pkl', '.pickle']):
        return

    print(f"  Loading data from {pickle_path.name}")
    data = load_rml2016_10a_data(pickle_path)
    n_combinations = len(data)
    print(f"  Found {n_combinations} (modulation, SNR) combinations")

    # Process each (modulation, SNR) combination
    for combo_idx, ((mod_str, snr_db), samples_array) in enumerate(data.items()):
        if combo_idx % 20 == 0 or combo_idx == n_combinations - 1:
            print(f"  Processing {combo_idx + 1}/{n_combinations}: {mod_str} @ {snr_db} dB ({samples_array.shape[0]} samples)")
        # Normalize modulation name to match LABEL_REMAP keys
        mod_key = mod_str.lower()

        # Map to label name
        if mod_key not in LABEL_REMAP:
            print(f"Warning: Unknown modulation type '{mod_str}' (SNR={snr_db} dB), skipping...")
            continue

        label_name = LABEL_REMAP[mod_key]

        # Validate array shape: should be (1000, 2, 128)
        if samples_array.ndim != 3:
            print(f"Warning: Unexpected shape for ({mod_str}, {snr_db}): {samples_array.shape}, skipping...")
            continue

        num_samples, num_channels, num_points = samples_array.shape

        if num_channels != 2:
            print(f"Warning: Expected 2 channels (I/Q) but got {num_channels} for ({mod_str}, {snr_db}), skipping...")
            continue

        if num_points != WINDOW_SIZE:
            print(f"Warning: Expected {WINDOW_SIZE} time points but got {num_points} for ({mod_str}, {snr_db}), skipping...")
            continue

        # Process each of the 1000 samples
        for sample_idx in range(num_samples):
            # Extract I/Q window: shape (2, 128)
            # The source array has shape (1000, 2, 128), so [sample_idx] gives (2, 128),
            # the (2, W) format normalize_window expects
            window = samples_array[sample_idx].astype(DATA_DTYPE)

            # Apply per-window normalization
            window = normalize_window(window, NORMALIZE_MODE)

            # Build metadata
            meta = {
                "source_file":    str(pickle_path),
                "modulation_type": label_name,
                "waveform_type":  MODULATION_TO_WAVEFORM.get(label_name, "unknown"),
                "snr_db":         float(snr_db),
                "sample_index":   int(sample_idx),
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
    print("Noise filtering: DISABLED (SNR levels are part of dataset design)")

    # Check that pickle file exists
    pickle_files = list(RAW_DATA_ROOT.glob("*.pkl")) + list(RAW_DATA_ROOT.glob("*.pickle"))
    if len(pickle_files) == 0:
        raise FileNotFoundError(
            f"No pickle file found in {RAW_DATA_ROOT}. "
            f"Expected file: RML2016.10a_dict.pkl"
        )

    # build_hdf5_dataset will call window_iterator_fn for each file it finds
    # The iterator processes pickle files only and skips the rest
    summary = build_hdf5_dataset(
        data_dir=RAW_DATA_ROOT,
        output_path=OUTPUT_H5_PATH,
        window_iterator_fn=iter_rml2016_10a_windows,
        batch_size=64,
        data_dtype=DATA_DTYPE,
        global_metadata=GLOBAL_METADATA,
        window_metadata_schema=WINDOW_METADATA_SCHEMA,
    )

    print(f"Created dataset with {summary['total_windows']} windows")
    print(f"Label names: {summary.get('label_names', [])}")
    print("Writing Croissant metadata...")

    # Build preprocessing metadata
    preprocessing = {
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "pad_last": PAD_LAST,
        "normalization": NORMALIZE_MODE or "none",
    }

    # Add noise filtering metadata only if filtering is enabled
    if NOISE_POWER_THRESHOLD_DB is not None:
        noise_filtering = {"threshold_db": NOISE_POWER_THRESHOLD_DB}
        preprocessing["noise_filtering"] = noise_filtering

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
            "upstream_dataset": "RadioML 2016.10A",
            "dataset_url": GLOBAL_METADATA["dataset_url"],
            "dataset_type": "Modulation classification",
            "channel_characteristics": "Moderate LO drift, light fading",
            "snr_range_db": GLOBAL_METADATA["snr_range_db"],
            "script_version": SCRIPT_VERSION,
        },
    )

    print(f"Croissant metadata written to {OUTPUT_CROISSANT_PATH}")
    print(f"Dataset written to {OUTPUT_H5_PATH}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
