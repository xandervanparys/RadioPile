"""
Dataset configuration for RadioML 2016.10A (Modulation Classification).

This dataset contains 11 modulation classes (8 digital + 3 analog) with
variable-SNR samples. The dataset features moderate LO drift, light fading,
and numerous different labeled SNR increments for measuring performance across
different signal and noise power scenarios.

Published by DeepSig; DATASET_URL and DATASET_MIRROR_URL below.
"""

from pathlib import Path
import numpy as np

from pipeline.schemas import H5_STR_DTYPE, WindowMetaField
from datasets.identifiers import RECORD_A_DOI, RECORD_B_DOI, CODE_DOI

# ---------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------

DATASET_NAME = "RadioML 2016.10A (Modulation Classification)"
DATASET_URL = "https://www.deepsig.ai/datasets"
DATASET_MIRROR_URL = "https://www.kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom"
SCRIPT_VERSION = "1.0.0"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_DATA_ROOT = Path("data/rml2016_10a/raw")
OUTPUT_ROOT = Path("data/rml2016_10a/processed")

OUTPUT_H5_PATH = OUTPUT_ROOT / "rml2016_10a.h5"
OUTPUT_CROISSANT_PATH = OUTPUT_ROOT / "rml2016_10a.croissant.json"

# ---------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------

"""
Assumptions
-----------
- Raw data is a single pickle file: RML2016.10a_dict.pkl
- Data structure: dictionary with keys (modulation_type_bytes, SNR_value)
- Each key maps to numpy array of shape (1000, 2, 128)
  - 1000 samples per (modulation, SNR) combination
  - 2 channels: I and Q
  - 128 time points per sample
- Each sample is already a fixed-size window (128 points)
- Modulation types: BPSK, QPSK, 8PSK, PAM4, QAM16, QAM64, GFSK, CPFSK,
  WBFM, AM-DSB, AM-SSB
- SNR values range from -20 to 18 dB in increments of 2 dB
"""

# ---------------------------------------------------------------------
# Labels (modulation classes)
# ---------------------------------------------------------------------

# Modulation types in the dataset
MODULATION_TYPES = [
    "BPSK",
    "QPSK",
    "8PSK",
    "PAM4",
    "16QAM",
    "64QAM",
    "GFSK",
    "CPFSK",
    "WBFM",
    "AM-DSB",
    "AM-SSB",
]

# Label remapping: lowercase canonical name → canonical name, plus
# original pickle key variants (QAM16 → 16QAM, QAM64 → 64QAM).
LABEL_REMAP = {mod.lower(): mod for mod in MODULATION_TYPES}
LABEL_REMAP.update({"qam16": "16QAM", "qam64": "64QAM"})

# Modulation → physical-layer waveform family.
# Groups specific modulation schemes into their parent family for cross-dataset
# waveform classification (e.g. BPSK + QPSK + 8PSK all belong to PSK family).
MODULATION_TO_WAVEFORM = {
    "BPSK":   "PSK",
    "QPSK":   "PSK",
    "8PSK":   "PSK",
    "PAM4":   "PAM",
    "16QAM":  "QAM",
    "64QAM":  "QAM",
    "GFSK":   "FSK",
    "CPFSK":  "FSK",
    "WBFM":   "FM",
    "AM-DSB": "AM",
    "AM-SSB": "AM",
}

# ---------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------

# Each sample is already a 128-point window and is used as is
# No overlapping windows - each sample is independent
WINDOW_SIZE = 128
HOP_SIZE = 128
PAD_LAST = False  # No padding needed since samples are fixed-size

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

# Source file dtype: raw .pkl file contains float32 arrays of shape (N, 2, W).
# Informational only — the pickle loader infers dtype automatically.
SOURCE_DTYPE = "float32"

# IQ storage dtype:
# - "float32" (default, for ML/efficiency)
# - "float64" (use only if extra precision is needed)
DATA_DTYPE = "float32"

# Normalization mode applied per window. Options:
#   None or "none": No normalization
#   "unit_power": Normalize to unit power (mean(I² + Q²) = 1) [default for RF]
#   "max_magnitude": Normalize by max absolute value across I and Q
#   "zero_mean_unit_variance": Standardization (zero mean, unit variance)
#   "mapminmax": Min-max scaling per channel to [-1, 1] range
NORMALIZE_MODE = "unit_power"

# Noise filtering configuration
# For this dataset, noise filtering is typically not needed since:
# 1. The dataset already includes controlled SNR levels
# 2. Noise characteristics are part of the dataset design
# 3. All samples are signal samples (no pure noise class)
NOISE_POWER_THRESHOLD_DB = None  # Disabled for this dataset

# Auto-detection configuration (not used since filtering is disabled)
NOISE_FLOOR_DETECTION_METHOD = "percentile"
NOISE_FLOOR_DETECTION_PARAMS = {
    "percentile": 1.0,
    "margin_db": 6.0,
    "bins": 100,
    "noise_fraction": 0.1,
    "std_multiplier": 2.0,
}
NOISE_FLOOR_MAX_SAMPLES = 10000

# ---------------------------------------------------------------------
# Global dataset metadata
# ---------------------------------------------------------------------

GLOBAL_METADATA = {
    "dataset": DATASET_NAME,
    "description": (
        "Variable-SNR modulation classification dataset containing 11 modulation "
        "classes (8 digital: BPSK, QPSK, 8PSK, PAM4, 16QAM, 64QAM, GFSK, CPFSK; "
        "3 analog: WBFM, AM-DSB, AM-SSB). Features moderate LO drift, light fading, "
        "and SNR values ranging from -20 to 18 dB in 2 dB increments. "
        "Each (modulation, SNR) combination contains 1000 samples of 128 IQ time points."
    ),
    "schema_version": "1.0.0",
    "script_version": SCRIPT_VERSION,
    "window_size": WINDOW_SIZE,
    "hop_size": HOP_SIZE,
    "pad_last": PAD_LAST,
    "normalization_mode": NORMALIZE_MODE or "none",
    "data_dtype": DATA_DTYPE,
    "modulation_classes": MODULATION_TYPES,
    "snr_range_db": "[-20, 18] in increments of 2",
    "samples_per_combination": 1000,
    "dataset_url": DATASET_URL,
}

# Add noise filtering metadata only if filtering is enabled
if NOISE_POWER_THRESHOLD_DB is not None:
    GLOBAL_METADATA["noise_power_threshold_db"] = NOISE_POWER_THRESHOLD_DB
    if NOISE_POWER_THRESHOLD_DB == "auto":
        GLOBAL_METADATA["noise_floor_detection_method"] = NOISE_FLOOR_DETECTION_METHOD
        GLOBAL_METADATA["noise_floor_detection_params"] = NOISE_FLOOR_DETECTION_PARAMS
        GLOBAL_METADATA["noise_floor_max_samples"] = NOISE_FLOOR_MAX_SAMPLES

# ---------------------------------------------------------------------
# Per-window metadata schema (stored as /meta/* datasets)
# ---------------------------------------------------------------------
#
# The HDF5 writer always stores minimal per-window metadata:
#   - file_path (required, always stored)
#
# Optional windowing fields (start_idx, padded) can be included by adding
# WINDOWING_META_FIELDS to this schema if needed.
#
# This schema defines the custom per-window fields for this dataset.

WINDOW_METADATA_SCHEMA = [
    WindowMetaField(key="modulation_type", h5_name="modulation_type", dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="waveform_type",   h5_name="waveform_type",   dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="snr_db",          h5_name="snr_db",          dtype="float32",    fill_value=np.nan),
    WindowMetaField(key="sample_index",    h5_name="sample_index",    dtype="int32",       fill_value=-1),
    # sample_rate_hz: not stored per-window — simulated dataset, no fixed physical sample rate
]

# ---------------------------------------------------------------------
# Croissant / provenance
# ---------------------------------------------------------------------

CITATIONS = [
    "T. J. O'Shea, N. West, 'Radio Machine Learning Dataset Generation with GNU Radio,' "
    "Proceedings of the 6th GNU Radio Conference, 2016.",
    "T. J. O'Shea, J. Corgan, T. C. Clancy, 'Convolutional Radio Modulation Recognition "
    "Networks,' Engineering Applications of Neural Networks (EANN), 2016.",
    f"Dataset: DeepSig Inc., RadioML 2016.10A. {DATASET_URL}",
    f"Mirror: {DATASET_MIRROR_URL}",
]

LICENSE = "CC-BY-NC-SA-4.0"

GLOBAL_METADATA["license"] = LICENSE
GLOBAL_METADATA["license_note"] = (
    "License of the original source dataset. This file was processed individually and never mixed, so no other dataset's terms apply to it. See LICENSES.md."
)

# Travels with the file: a per-dataset HDF5 separated from its README can still
# be traced to the record it was published in.
GLOBAL_METADATA["doi"] = RECORD_B_DOI
GLOBAL_METADATA["related_identifiers"] = f"aggregate={RECORD_A_DOI}; code={CODE_DOI}"
