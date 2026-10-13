"""
Dataset build script.

All dataset configuration and assumptions are defined in config.py.

This script implements the dataset-specific logic for:
1. Inferring labels from filenames/paths
2. Parsing metadata from filenames/paths
3. Loading raw IQ data with the source dtype declared in config.py
4. Iterating over windows with proper metadata

The shared preprocessing pipeline handles:
- Windowing
- Normalization
- Noise filtering
- HDF5 writing
- Croissant metadata generation
"""

from pathlib import Path
import re
from typing import Iterator, Tuple, Dict, Any

import numpy as np

from pipeline.hdf5_writer import build_hdf5_dataset
from pipeline.io import iter_windows_for_file
from pipeline.preprocessing import (
    normalize_window,
    filter_window_by_power,
    determine_noise_threshold,
)
from metadata.croissant import generate_croissant_from_hdf5

from .config import (
    DATASET_NAME,
    SCRIPT_VERSION,
    RAW_DATA_ROOT,
    OUTPUT_H5_PATH,
    OUTPUT_CROISSANT_PATH,
    LABEL_REMAP,
    WINDOW_SIZE,
    HOP_SIZE,
    PAD_LAST,
    NORMALIZE_MODE,
    NOISE_POWER_THRESHOLD_DB,
    SOURCE_DTYPE,
    DATA_DTYPE,
    GLOBAL_METADATA,
    WINDOW_METADATA_SCHEMA,
    CITATIONS,
    LICENSE,
)

try:
    from .config import (
        NOISE_FLOOR_DETECTION_METHOD,
        NOISE_FLOOR_DETECTION_PARAMS,
        NOISE_FLOOR_MAX_SAMPLES,
    )
except ImportError:
    NOISE_FLOOR_DETECTION_METHOD = "percentile"
    NOISE_FLOOR_DETECTION_PARAMS = {"percentile": 1.0, "margin_db": 6.0}
    NOISE_FLOOR_MAX_SAMPLES = 10000

# -----------------------------------------------------------------------------
# Dataset-specific parsing functions
# -----------------------------------------------------------------------------

def infer_label_from_filename(path: Path) -> str:
    """
    Infer the raw label from a filename or path.

    This function should contain all dataset-specific logic for mapping
    filenames/paths to source labels.

    TODO: Implement label inference for the dataset.

    Examples:
    - Extract from filename: "lte_g30_uz_f806MHz_r1.bin" -> "lte"
    - Extract from directory: "raw/867.4/Sigfox/file.mat" -> "sigfox"
    - Use regex patterns to match technology names

    Returns:
        Raw label string (must be a key in LABEL_REMAP if LABEL_REMAP is not None)
    """
    # TODO: Implement the label inference logic
    # Example patterns:

    # Pattern 1: Extract from filename prefix
    # first = path.name.split("_")[0]
    # m = re.match(r"([A-Za-z]+)", first)
    # return (m.group(1) if m else first).lower()

    # Pattern 2: Extract from directory structure
    # parts_lower = [p.lower() for p in path.parts]
    # for key in LABEL_REMAP:
    #     if key in parts_lower:
    #         return key

    # Pattern 3: Use regex on filename
    # m = re.search(r"pattern", path.name.lower())
    # if m:
    #     return m.group(1)

    raise NotImplementedError("TODO: Implement label inference for this dataset")


def parse_filename_metadata(path: Path) -> Dict[str, Any]:
    """
    Parse additional metadata from filename or path.

    This function extracts dataset-specific metadata that may be encoded
    in filenames or directory structure (e.g., frequency, gain, location, etc.).

    TODO: Implement metadata parsing for the dataset.

    Returns:
        Dictionary with optional metadata keys. Common keys:
        - sample_rate_hz: float
        - center_frequency_hz: float
        - gain_db: float
        - location: str
        - run_id: int
        - Any other dataset-specific fields
    """
    # TODO: Implement the metadata parsing logic
    # Example:
    # meta = {}
    # stem = path.stem
    #
    # # Parse sample rate from filename
    # m = re.search(r"(\d+)Msps", stem)
    # if m:
    #     meta["sample_rate_hz"] = float(m.group(1)) * 1e6
    #
    # # Parse center frequency
    # m = re.search(r"f(\d+)MHz", stem)
    # if m:
    #     meta["center_frequency_hz"] = float(m.group(1)) * 1e6
    #
    # return meta

    return {}


# -----------------------------------------------------------------------------
# Dataset-specific window iterator
# -----------------------------------------------------------------------------

def iter_dataset_windows(path: Path) -> Iterator[Tuple[np.ndarray, str, Dict[str, Any]]]:
    """
    Iterate over (window, label_name, metadata) tuples for a single file.

    This is the main dataset-specific function that:
    1. Infers the label from the file
    2. Parses metadata from the filename/path
    3. Loads or streams the raw IQ data
    4. Iterates over windows with proper metadata
    5. Applies noise filtering (if configured)
    6. Applies normalization

    Yields:
        window: np.ndarray of shape (2, WINDOW_SIZE) - IQ data [I, Q]
        label_name: str - Semantic label name (from LABEL_REMAP or raw label)
        metadata: dict - Per-window metadata dict
    """
    # TODO: Add file format filtering if needed
    # Example: Skip non-data files
    # if not path.is_file() or path.suffix.lower() not in [".bin", ".mat", ".h5"]:
    #     return

    # Infer label
    raw_label = infer_label_from_filename(path)

    # Map to semantic label
    if LABEL_REMAP is not None:
        if raw_label not in LABEL_REMAP:
            raise ValueError(f"Label '{raw_label}' missing from LABEL_REMAP. Found in: {path}")
        label_name = LABEL_REMAP[raw_label]
    else:
        label_name = raw_label

    # Parse filename metadata (same for all windows from this file)
    file_meta = parse_filename_metadata(path)

    # Determine noise threshold for this file (if noise filtering is enabled)
    noise_threshold_db = None
    if NOISE_POWER_THRESHOLD_DB is not None:
        # For noise classes, skip filtering
        is_noise_class = (raw_label.lower() == "noise" or label_name.lower() == "noise")

        if not is_noise_class:
            # Determine threshold using auto-detection or fixed value
            noise_threshold_db = determine_noise_threshold(
                noise_power_threshold_db=NOISE_POWER_THRESHOLD_DB,
                window_iterator=iter_windows_for_file(
                    path=path,
                    window_size=WINDOW_SIZE,
                    hop_size=HOP_SIZE,
                    pad_last=PAD_LAST,
                    dtype=SOURCE_DTYPE,
                    output_dtype=DATA_DTYPE,
                ),
                detection_method=NOISE_FLOOR_DETECTION_METHOD,
                detection_params=NOISE_FLOOR_DETECTION_PARAMS,
                max_samples=NOISE_FLOOR_MAX_SAMPLES,
            )

    # Track filtering statistics
    total_windows = 0
    filtered_windows = 0

    # Iterate over windows from this file. dtype is the on-disk dtype of a .bin file:
    # a .bin carries no format information, and reading complex64 data with the
    # int16 default yields values that look like noise instead of raising an error.
    for window, meta in iter_windows_for_file(
        path=path,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        pad_last=PAD_LAST,
        dtype=SOURCE_DTYPE,
        output_dtype=DATA_DTYPE,
    ):
        total_windows += 1

        # Apply noise filtering (before normalization)
        if noise_threshold_db is not None:
            if not filter_window_by_power(window, noise_threshold_db):
                filtered_windows += 1
                continue

        # Build metadata dict
        meta_out = dict(meta)  # Start with windowing metadata (start_idx, padded, etc.)
        meta_out["source_file"] = str(path)  # Always include source file path

        # Add filename-derived metadata
        for key, value in file_meta.items():
            if value is not None:
                meta_out[key] = value

        # Apply per-window normalization
        window = normalize_window(window, NORMALIZE_MODE)

        # Ensure correct dtype
        window = window.astype(DATA_DTYPE)

        yield window, label_name, meta_out

    # Print filtering statistics
    if noise_threshold_db is not None and total_windows > 0:
        filter_rate = (filtered_windows / total_windows) * 100
        print(f"  {path.name}: Filtered {filtered_windows}/{total_windows} windows ({filter_rate:.1f}%)")


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

def main() -> None:
    print(f"=== Building dataset: {DATASET_NAME} ===")
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Window size: {WINDOW_SIZE}, hop size: {HOP_SIZE}, pad_last: {PAD_LAST}")
    print(f"Data dtype: {DATA_DTYPE}")
    print(f"Normalization mode: {NORMALIZE_MODE or 'none'}")

    if NOISE_POWER_THRESHOLD_DB == "auto":
        print(f"Noise filtering: AUTO (method={NOISE_FLOOR_DETECTION_METHOD})")
    elif NOISE_POWER_THRESHOLD_DB is not None:
        print(f"Noise power threshold (dB): {NOISE_POWER_THRESHOLD_DB}")
    else:
        print("Noise filtering: DISABLED")

    print(f"Raw data directory: {RAW_DATA_ROOT}")
    print(f"Output HDF5: {OUTPUT_H5_PATH}")
    print()

    # Build the HDF5 dataset
    summary = build_hdf5_dataset(
        data_dir=RAW_DATA_ROOT,
        output_path=OUTPUT_H5_PATH,
        window_iterator_fn=iter_dataset_windows,
        batch_size=64,  # TODO: Adjust if needed
        data_dtype=DATA_DTYPE,
        global_metadata=GLOBAL_METADATA,
        window_metadata_schema=WINDOW_METADATA_SCHEMA,
    )

    print(f"\nCreated dataset with {summary['total_windows']} windows")
    print(f"Label names: {summary.get('label_names', [])}")
    print("Writing Croissant metadata...")

    # Build preprocessing metadata for Croissant
    preprocessing = {
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "pad_last": PAD_LAST,
        "normalization": NORMALIZE_MODE or "none",
    }

    # Add noise filtering metadata if enabled
    if NOISE_POWER_THRESHOLD_DB is not None:
        noise_filtering = {"threshold_db": NOISE_POWER_THRESHOLD_DB}
        if NOISE_POWER_THRESHOLD_DB == "auto":
            noise_filtering["detection_method"] = NOISE_FLOOR_DETECTION_METHOD
            noise_filtering["detection_params"] = NOISE_FLOOR_DETECTION_PARAMS
            noise_filtering["max_samples"] = NOISE_FLOOR_MAX_SAMPLES
        preprocessing["noise_filtering"] = noise_filtering

    # Generate Croissant metadata
    generate_croissant_from_hdf5(
        h5_path=OUTPUT_H5_PATH,
        output_path=OUTPUT_CROISSANT_PATH,
        version=SCRIPT_VERSION,
        dataset_name=DATASET_NAME,
        description=GLOBAL_METADATA.get("description", DATASET_NAME),
        task_type="classification",
        labels=summary.get("label_names", []),
        citations=CITATIONS,
        license=LICENSE,
        preprocessing=preprocessing,
        provenance={
            "upstream_dataset": DATASET_NAME,  # TODO: Name of the original dataset
            "dataset_url": GLOBAL_METADATA["dataset_url"],
            "script_version": SCRIPT_VERSION,
            # TODO: Add dataset-specific provenance information
        },
    )

    print("=== Done ===")
    print(f"Dataset saved to: {OUTPUT_H5_PATH}")
    print(f"Metadata saved to: {OUTPUT_CROISSANT_PATH}")


if __name__ == "__main__":
    main()
