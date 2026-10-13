"""
Dataset configuration for the
Technology-Recognition-dataset-of-real-life-LTE-WiFi-and-DVB-T.

This file defines all dataset identity, assumptions, and scientific
configuration. Changing any value in this file should be considered
a dataset change.
"""

from pathlib import Path
import numpy as np

from pipeline.schemas import H5_STR_DTYPE, WindowMetaField
from datasets.identifiers import RECORD_A_DOI, RECORD_B_DOI, CODE_DOI

# ---------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------

DATASET_NAME = "Technology Recognition (LTE / WiFi / DVB-T)"
DATASET_URL = "https://github.com/JaronFontaine/Technology-Recognition-dataset-of-real-life-LTE-WiFi-and-DVB-T"
SCRIPT_VERSION = "1.0.0"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_DATA_ROOT = Path("data/lte_wifi_dvbt/raw")
OUTPUT_ROOT = Path("data/lte_wifi_dvbt/processed")

OUTPUT_H5_PATH = OUTPUT_ROOT / "lte_wifi_dvbt.h5"
OUTPUT_CROISSANT_PATH = OUTPUT_ROOT / "lte_wifi_dvbt.croissant.json"

# ---------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------

"""
Assumptions
-----------
- Raw .bin files contain complex64 IQ samples (GNU Radio format: float32 I + float32 Q)
- Frequencies in filenames are expressed in MHz
- Sample rate is inferred from filename or defaults to 1 Msps if missing
- Each file contains a single signal technology (LTE, WiFi, or DVB-T)
"""

# ---------------------------------------------------------------------
# Location overrides
# ---------------------------------------------------------------------

# Capture location is parsed from the filename. One upstream file carries the
# wrong location token, so it is corrected here rather than by renaming the raw
# file (the local copy stays identical to upstream). Keys are "<folder>/<filename>"
# under RAW_DATA_ROOT.
#
# gentbrugge/lte10Msps_g30_uz_f806MHz_r1.bin was recorded at Gentbrugge: an LTE
# cell search finds physical cell IDs 297, 72, 159 and 61 in it, exactly the set
# heard in all ten 1 Msps Gentbrugge captures and disjoint from the UZ set
# (286, 25, 29). uz/ holds a different file of the same name that does hear the
# UZ cells. See docs/DATASETS.md, "Misnamed Gentbrugge capture".
LOCATION_OVERRIDES = {
    "gentbrugge/lte10Msps_g30_uz_f806MHz_r1.bin": "gentbrugge",
}

# ---------------------------------------------------------------------
# Labels (technology classes)
# ---------------------------------------------------------------------

# Labels (technology classes): ["dvbt", "wf", "lte"]

LABEL_REMAP = {
    "dvbt": "DVB-T",
    "wf": "WiFi",
    "lte": "LTE",
}

# Raw label -> physical-layer waveform type (fixed per technology, MCS-independent).
# All three technologies use OFDM as their physical layer waveform:
#   LTE: OFDMA (downlink), SC-FDMA (uplink) — both OFDM-family
#   WiFi: 802.11a/g/n/ac all use OFDM (only 802.11b uses DSSS, not present in this dataset)
#   DVB-T: COFDM (coded OFDM)
# Subcarrier modulation (BPSK/QPSK/16-QAM/64-QAM) is not resolvable without MCS metadata.
LABEL_TO_WAVEFORM = {
    "dvbt": "OFDM",
    "wf":   "OFDM",
    "lte":  "OFDM",
}

# ---------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------

WINDOW_SIZE = 500
HOP_SIZE = 500
PAD_LAST = True

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

# Source file dtype: raw .bin files are complex64 (GNU Radio default format).
# Passed to np.memmap when reading raw captures.
SOURCE_DTYPE = "complex64"

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
#   1: Manual threshold
#       Set NOISE_POWER_THRESHOLD_DB to a fixed value in dB.
#       Windows with power below this threshold will be filtered out.
#   2: Auto-detect noise floor per file
#       Set NOISE_POWER_THRESHOLD_DB = "auto" and configure:
#           - NOISE_FLOOR_DETECTION_METHOD: "percentile", "histogram", or "statistical"
#           - NOISE_FLOOR_DETECTION_PARAMS: dict with method-specific parameters
#       The noise floor will be detected from each file and used as threshold.
#   3: Disable noise filtering
#       Set NOISE_POWER_THRESHOLD_DB = None
NOISE_POWER_THRESHOLD_DB = None  # None, float (dB), or "auto"
# No filtering: the signals are continuous.

# Auto-detection configuration (only used if NOISE_POWER_THRESHOLD_DB == "auto")
NOISE_FLOOR_DETECTION_METHOD = "percentile"  # "percentile", "histogram", or "statistical"
NOISE_FLOOR_DETECTION_PARAMS = {
    # For "percentile":
    "percentile": 1.0,
    "margin_db": 6.0,
    # For "histogram":
    "bins": 100,
    # For "histogram" and "statistical":
    "noise_fraction": 0.1,
    # For "statistical":
    "std_multiplier": 2.0,
}
# Maximum windows to sample for auto-detection
NOISE_FLOOR_MAX_SAMPLES = 10000

# ---------------------------------------------------------------------
# Global dataset metadata
# ---------------------------------------------------------------------

GLOBAL_METADATA = {
    "dataset": DATASET_NAME,
    "description": (
        "Windowed IQ dataset derived from real-life LTE, WiFi and DVB-T "
        "over-the-air captures at six sites in and around Ghent, Belgium."
    ),
    "schema_version": "1.0.0",
    "dataset_url": DATASET_URL,
    "script_version": SCRIPT_VERSION,
    "window_size": WINDOW_SIZE,
    "hop_size": HOP_SIZE,
    "pad_last": PAD_LAST,
    "normalization_mode": NORMALIZE_MODE or "none",
    "data_dtype": DATA_DTYPE,
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
    WindowMetaField(key="sample_rate_hz",  h5_name="sample_rate_hz",  dtype="float64",    fill_value=np.nan),
    WindowMetaField(key="center_frequency_hz",  h5_name="center_frequency_hz",  dtype="float64",    fill_value=np.nan),
    WindowMetaField(key="gain_db",         h5_name="gain_db",         dtype="float32",    fill_value=np.nan),
    WindowMetaField(key="location",        h5_name="location",        dtype=H5_STR_DTYPE, fill_value=""),
    WindowMetaField(key="recording_id",    h5_name="recording_id",    dtype=H5_STR_DTYPE, fill_value=""),
    WindowMetaField(key="technology_type", h5_name="technology_type", dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="modulation_type", h5_name="modulation_type", dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="waveform_type",   h5_name="waveform_type",   dtype=H5_STR_DTYPE, fill_value="unknown"),
    # 802.11 variant. "unknown" for WiFi windows, because the variant is not
    # documented upstream; "" for LTE and DVB-T windows.
    WindowMetaField(key="wifi_standard",   h5_name="wifi_standard",   dtype=H5_STR_DTYPE, fill_value=""),
]

# ---------------------------------------------------------------------
# Croissant / provenance
# ---------------------------------------------------------------------

CITATIONS = [
    "J. Fontaine, E. Fonseca, A. Shahid, M. Kist, L. A. DaSilva, I. Moerman, E. De Poorter, "
    "'Towards low-complexity wireless technology classification across multiple environments,' "
    "Ad Hoc Networks, vol. 91, p. 101881, 2019.",
    "W. Liu, M. Kulin, T. Kazaz, A. Shahid, I. Moerman, E. De Poorter, "
    "'Wireless technology recognition based on RSSI distribution at sub-Nyquist sampling rate "
    "for constrained devices,' Sensors, vol. 17, no. 9, p. 2081, 2017.",
    f"Dataset repository: {DATASET_URL}",
    "Dataset files: https://cloud.ilabt.imec.be/index.php/s/qrJCWgzQaGPfHPr",
]

# Dataset license identifier (SPDX preferred)
LICENSE = "CC-BY-NC-SA-4.0"

GLOBAL_METADATA["license"] = LICENSE
GLOBAL_METADATA["license_note"] = (
    "License of the original source dataset. This file was processed individually and never mixed, so no other dataset's terms apply to it. See LICENSES.md."
)

# Travels with the file: a per-dataset HDF5 separated from its README can still
# be traced to the record it was published in.
GLOBAL_METADATA["doi"] = RECORD_B_DOI
GLOBAL_METADATA["related_identifiers"] = f"aggregate={RECORD_A_DOI}; code={CODE_DOI}"
