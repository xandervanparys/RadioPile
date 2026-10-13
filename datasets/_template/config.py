"""
Dataset configuration file.

This file defines all scientific and dataset-specific assumptions used when
constructing the dataset. It is intentionally declarative and should not
contain any data-loading or processing logic.

If this file changes, the resulting dataset should be considered a different
dataset version.

TODO: Fill in all sections marked with TODO comments.
"""

from pathlib import Path
import numpy as np

from pipeline.schemas import H5_STR_DTYPE, WindowMetaField

# -----------------------------------------------------------------------------
# Dataset identity
# -----------------------------------------------------------------------------

DATASET_NAME = "example_dataset"  # TODO: Set the dataset name
DATASET_URL = "https://example.org/dataset"  # TODO: Where the source data is published
SCRIPT_VERSION = "1.0.0"  # TODO: Increment when the build script logic changes

# Short human-readable description (used in logs / metadata)
DATASET_DESCRIPTION = (
    "Example RF dataset converted into a unified ML-ready representation. "
    "TODO: Describe your dataset."
)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

# TODO: Update these paths to match the dataset structure
# The raw data directory (relative to repository root)
RAW_DATA_ROOT = Path("data/_template/raw")
# The output directory for processed datasets (relative to repository root)
OUTPUT_ROOT = Path("data/_template/processed")

# Output file paths
OUTPUT_H5_PATH = OUTPUT_ROOT / f"{DATASET_NAME.replace(' ', '_').lower()}.h5"
OUTPUT_CROISSANT_PATH = OUTPUT_ROOT / f"{DATASET_NAME.replace(' ', '_').lower()}.croissant.json"

# -----------------------------------------------------------------------------
# Assumptions
# -----------------------------------------------------------------------------

"""
Assumptions
-----------
TODO: Document the dataset-specific assumptions here. Examples:

For .bin files:
- Raw .bin files contain interleaved int16 IQ samples (I,Q,I,Q,...)
- Sample rate is fixed at X Msps (or inferred from filename)
- Each file contains a single signal technology

For .mat files:
- Raw files are MATLAB .mat files
- IQ samples are stored in mat["IQ_samples"] (or another key)
- IQ samples are complex64 or complex128
- Sampling rate is fixed at X Msps

For .h5 files:
- Raw files are HDF5 files with IQ data in a specific structure
- TODO: Describe the structure
"""

# -----------------------------------------------------------------------------
# Labels (technology classes)
# -----------------------------------------------------------------------------

# TODO: Define the label mapping
# Option 1: Use label remapping (recommended for consistent naming)
LABEL_REMAP = {
    # "raw_label_from_file": "SemanticLabel",
    # Example:
    # "lte": "LTE",
    # "wf": "WiFi",
    # "dvbt": "DVB-T",
}

# Option 2: Keep the source labels. Use an identity mapping, not None: the
# aggregation reads the values of LABEL_REMAP as the set of valid classes.
# LABEL_REMAP = {"LTE": "LTE", "WiFi": "WiFi"}

SOURCE_LABELS = list(LABEL_REMAP.keys())

# -----------------------------------------------------------------------------
# Windowing
# -----------------------------------------------------------------------------

# SAMPLE_RATE_HZ = 2.048e6  # TODO: Define if one rate holds for every file

# Number of IQ samples per window
WINDOW_SIZE = 2048  # TODO: Set an appropriate window size for the dataset

# Hop size between consecutive windows (HOP_SIZE == WINDOW_SIZE means no overlap)
HOP_SIZE = 2048  # TODO: Set appropriate hop size (typically equal to WINDOW_SIZE for no overlap)

# If True, zero-pad the final window if the signal length is not divisible
PAD_LAST = True

# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------

# Normalization mode applied per window. Options:
#   None or "none": No normalization
#   "unit_power": Normalize to unit power (mean(I² + Q²) = 1) [default for RF]
#   "max_magnitude": Normalize by max absolute value across I and Q
#   "zero_mean_unit_variance": Standardization (zero mean, unit variance)
#   "mapminmax": Min-max scaling per channel to [-1, 1] range
NORMALIZE_MODE = "unit_power"  # TODO: Choose appropriate normalization

# Noise filtering configuration
#
# Option 1: Manual threshold (recommended for production)
#   Set NOISE_POWER_THRESHOLD_DB to a fixed value in dB.
#   Windows with power below this threshold will be filtered out.
#   Example: NOISE_POWER_THRESHOLD_DB = -70.0
#
# Option 2: Auto-detect noise floor per file
#   Set NOISE_POWER_THRESHOLD_DB = "auto" and configure:
#   - NOISE_FLOOR_DETECTION_METHOD: "percentile", "histogram", or "statistical"
#   - NOISE_FLOOR_DETECTION_PARAMS: dict with method-specific parameters
#   The noise floor will be detected from each file and used as threshold.
#
# Option 3: Disable noise filtering
#   Set NOISE_POWER_THRESHOLD_DB = None

NOISE_POWER_THRESHOLD_DB = None  # TODO: Set to None, float (dB), or "auto"

# Auto-detection configuration (only used if NOISE_POWER_THRESHOLD_DB == "auto")
NOISE_FLOOR_DETECTION_METHOD = "percentile"  # "percentile", "histogram", or "statistical"
NOISE_FLOOR_DETECTION_PARAMS = {
    # For "percentile" method:
    "percentile": 1.0,  # Percentile to use as noise floor (0-100), e.g., 1.0 = 1st percentile
    "margin_db": 6.0,  # Margin in dB to add above floor for threshold (recommended: 6.0)

    # For "histogram" method:
    "bins": 100,  # Number of histogram bins
    "noise_fraction": 0.1,  # Fraction of lowest-power windows to analyze

    # For "statistical" method:
    "std_multiplier": 2.0,  # Number of standard deviations below mean
}

# Maximum number of windows to sample per file for noise floor detection
# (only used for auto-detection)
NOISE_FLOOR_MAX_SAMPLES = 10000

# Source file dtype: the numpy dtype of the raw source data.
# For .bin files this is functional: build_dataset.py passes it to
# iter_windows_for_file, which reads the file with it. A .bin file carries no
# format information, so a wrong value does not raise; it produces windows
# that look like noise. Check a spectrogram of the first build.
# For .mat / .pkl / .h5 files the container knows its dtype and this is
# informational only.
# Common values (use string form, not np.* types):
#   "complex64"   — GNU Radio default (.bin)
#   "complex128"  — MATLAB double complex (.mat)
#   "float32"     — pre-normalized arrays (.pkl)
#   "int16"       — SDR integer captures (.bin)
SOURCE_DTYPE = "complex64"  # TODO: set to correct dtype for this dataset

# IQ storage dtype:
# - "float32" (default, for ML/efficiency)
# - "float64" (use only if extra precision is needed)
DATA_DTYPE = "float32"

# -----------------------------------------------------------------------------
# Global dataset metadata
# -----------------------------------------------------------------------------

GLOBAL_METADATA = {
    "dataset": DATASET_NAME,
    "description": DATASET_DESCRIPTION,
    "schema_version": "1.0.0",
    "script_version": SCRIPT_VERSION,
    "window_size": WINDOW_SIZE,
    "hop_size": HOP_SIZE,
    "pad_last": PAD_LAST,
    "normalization_mode": NORMALIZE_MODE or "none",
    "data_dtype": DATA_DTYPE,
    "dataset_url": DATASET_URL,
    # TODO: Add dataset-specific global metadata, for example:
    # "sample_rate_hz": SAMPLE_RATE_HZ,  # If one rate holds for every file
}

# Add noise filtering metadata only if filtering is enabled
if NOISE_POWER_THRESHOLD_DB is not None:
    GLOBAL_METADATA["noise_power_threshold_db"] = NOISE_POWER_THRESHOLD_DB
    if NOISE_POWER_THRESHOLD_DB == "auto":
        GLOBAL_METADATA["noise_floor_detection_method"] = NOISE_FLOOR_DETECTION_METHOD
        GLOBAL_METADATA["noise_floor_detection_params"] = NOISE_FLOOR_DETECTION_PARAMS
        GLOBAL_METADATA["noise_floor_max_samples"] = NOISE_FLOOR_MAX_SAMPLES

# -----------------------------------------------------------------------------
# Per-window metadata schema (stored as /meta/* datasets)
# -----------------------------------------------------------------------------
#
# The HDF5 writer always stores minimal per-window metadata:
#   - file_path (required, always stored)
#
# Optional windowing fields (start_idx, padded) can be included by adding
# WINDOWING_META_FIELDS to this schema if needed.
#
# This schema defines the custom per-window fields for this dataset.
# TODO: Add fields that are relevant to the dataset

WINDOW_METADATA_SCHEMA = [
    # --- Classification targets (include for aggregation compatibility) ---
    # WindowMetaField(key="technology_type",  h5_name="technology_type",  dtype=H5_STR_DTYPE, fill_value="unknown"),
    # WindowMetaField(key="modulation_type",  h5_name="modulation_type",  dtype=H5_STR_DTYPE, fill_value="unknown"),
    # WindowMetaField(key="waveform_type",    h5_name="waveform_type",    dtype=H5_STR_DTYPE, fill_value="unknown"),

    # --- Grouping key for recording-aware train/test splits ---
    # WindowMetaField(key="recording_id",     h5_name="recording_id",     dtype=H5_STR_DTYPE, fill_value=""),
    # Use string dtype even if the ID is numeric — avoids lossy int conversion for composite IDs.

    # --- RF capture parameters ---
    # WindowMetaField(key="sample_rate_hz",   h5_name="sample_rate_hz",   dtype="float64",    fill_value=np.nan),
    # WindowMetaField(key="center_frequency_hz",   h5_name="center_frequency_hz",   dtype="float64",    fill_value=np.nan),
    # WindowMetaField(key="gain_db",          h5_name="gain_db",          dtype="float32",    fill_value=np.nan),
    # WindowMetaField(key="snr_db",           h5_name="snr_db",           dtype="float32",    fill_value=np.nan),

    # --- Capture context ---
    # WindowMetaField(key="location",         h5_name="location",         dtype=H5_STR_DTYPE, fill_value=""),
]

# -----------------------------------------------------------------------------
# Croissant / provenance
# -----------------------------------------------------------------------------

# Citation(s) associated with the original dataset
# TODO: Add citations
CITATIONS = [
    # Example:
    # "Author et al., Journal, Year",
]

# Dataset license identifier (SPDX preferred)
# TODO: Set appropriate license
LICENSE = "CC-BY-4.0"

GLOBAL_METADATA["license"] = LICENSE
GLOBAL_METADATA["license_note"] = (
    "License of the original source dataset. This file was processed individually "
    "and never mixed, so no other dataset's terms apply to it. See LICENSES.md."
)
