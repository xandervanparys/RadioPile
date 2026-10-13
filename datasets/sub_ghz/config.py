"""
Dataset configuration for the Sub-GHz Technology Recognition dataset.

This dataset contains IQ captures of multiple Sub-GHz wireless technologies
recorded at 2.048 Msps using an RTL-SDR. Two center frequencies (864.0 MHz
and 867.4 MHz) are included.
"""

from pathlib import Path
import numpy as np

from pipeline.schemas import H5_STR_DTYPE, WindowMetaField
from datasets.identifiers import RECORD_A_DOI, RECORD_B_DOI, CODE_DOI

# ---------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------

DATASET_NAME = "Sub-GHz Technology Recognition"
DATASET_URL = "https://github.com/JaronFontaine/Sub-GHz-IQ-signals-dataset"
SCRIPT_VERSION = "1.0.0"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_DATA_ROOT = Path("data/sub_ghz/raw")
OUTPUT_ROOT = Path("data/sub_ghz/processed")

OUTPUT_H5_PATH = OUTPUT_ROOT / "sub_ghz.h5"
OUTPUT_CROISSANT_PATH = OUTPUT_ROOT / "sub_ghz.croissant.json"

# ---------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------

"""
Assumptions
-----------
- Raw files are MATLAB .mat files
- IQ samples are stored in mat["IQ_samples"], a 1-D complex128 array
- Sampling rate is fixed at 2.048 Msps
- One signal technology per file
"""

# ---------------------------------------------------------------------
# Labels (technology classes)
# ---------------------------------------------------------------------

LABEL_REMAP = {
    "sigfox": "Sigfox",
    "lora": "LoRa",
    "80211ah": "WiFi",           # 802.11ah is a WiFi variant; standard stored in wifi_standard metadata
    "802154g": "IEEE 802.15.4",  # 802.15.4g is a 802.15.4 variant; stored in ieee802154_variant metadata
    "sunofdm": "IEEE 802.15.4",  # SUN-OFDM is a 802.15.4 variant; stored in ieee802154_variant metadata
    "noise": "Noise",
}

# Raw label -> subcarrier/data modulation type (classification-ready, compatible with RML).
# For OFDM-based technologies (80211ah, sunofdm) the subcarrier modulation depends on the
# MCS, so the per-label entry is "unknown" (used as fallback when no MCS in filename).
# Keyed by raw_label (not canonical label) because 802154g and sunofdm share one canonical
# label ("IEEE 802.15.4") but have different modulations (BFSK vs OQPSK/16-QAM).
LABEL_TO_MODULATION = {
    "sigfox":  "BPSK",
    "lora":    "CSS",
    "80211ah": "unknown",   # subcarrier modulation resolved per MCS (0/10→BPSK, 7→64QAM)
    "802154g": "BFSK",
    "sunofdm": "unknown",   # subcarrier modulation resolved per MCS (2→OQPSK, 6→16QAM)
    "noise":   "unknown",   # noise has no modulation — excluded from modulation recognition
}

# The Sub-GHz dataset captures LoRa transmissions at SF7 and SF12 only,
# not the full LoRa spec range of SF7-SF12.
LORA_SPREADING_FACTORS = {7, 12}

# Raw label -> physical-layer waveform type (fixed per technology, MCS-independent).
# Describes the physical layer scheme rather than the subcarrier modulation.
LABEL_TO_WAVEFORM = {
    "sigfox":  "PSK",    # BPSK is a PSK variant; waveform_type generalizes to family
    "lora":    "CSS",
    "80211ah": "OFDM",
    "802154g": "FSK",    # BFSK is an FSK variant; waveform_type generalizes to family
    "sunofdm": "OFDM",
    "noise":   "noise",
}

# Raw label -> WiFi standard (802.11 variant); empty string for non-WiFi
WIFI_STANDARD = {
    "80211ah": "802.11ah",
}

# Raw label -> IEEE 802.15.4 variant; empty string for non-802.15.4
IEEE802154_VARIANT = {
    "802154g": "g",
    "sunofdm": "SUN-OFDM",
}

# ---------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------

SAMPLE_RATE_HZ = 2.048e6
WINDOW_SIZE = 500
HOP_SIZE = 500
PAD_LAST = True

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

# Source file dtype: raw .mat files contain complex128 IQ arrays (MATLAB double complex).
# Informational only — scipy.io.loadmat infers dtype automatically.
SOURCE_DTYPE = "complex128"

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
NOISE_POWER_THRESHOLD_DB = "auto"  # None, float (dB), or "auto"

# Auto-detection configuration (only used if NOISE_POWER_THRESHOLD_DB == "auto")
NOISE_FLOOR_DETECTION_METHOD = "percentile"  # "percentile", "histogram", or "statistical"
NOISE_FLOOR_DETECTION_PARAMS = {
    "percentile": 1.0,
    "margin_db": 6.0,
}
# Maximum windows to sample for auto-detection
NOISE_FLOOR_MAX_SAMPLES = 10000

# ---------------------------------------------------------------------
# Global dataset metadata
# ---------------------------------------------------------------------

GLOBAL_METADATA = {
    "dataset": DATASET_NAME,
    "description": (
        "IQ dataset containing multiple Sub-GHz wireless technologies "
        "including Sigfox, LoRa, WiFi (802.11ah), IEEE 802.15.4 (variants: g, SUN-OFDM), "
        "and Noise. Captured over the air using an RTL-SDR at 2.048 Msps with center "
        "frequencies at 864.0 MHz and 867.4 MHz. The WiFi standard and the 802.15.4 "
        "variant are stored as per-window metadata fields (wifi_standard, ieee802154_variant)."
    ),
    "schema_version": "1.0.0",
    "sample_rate_hz": SAMPLE_RATE_HZ,
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
    WindowMetaField(key="technology_type", h5_name="technology_type", dtype=H5_STR_DTYPE,   fill_value="unknown"),
    WindowMetaField(key="center_frequency_hz",    h5_name="center_frequency_hz",    dtype="float64",    fill_value=np.nan),
    WindowMetaField(key="gain_db",           h5_name="gain_db",           dtype="float32",    fill_value=np.nan),
    WindowMetaField(key="attenuation_db",    h5_name="attenuation_db",    dtype="float32",    fill_value=np.nan),
    WindowMetaField(key="mcs",               h5_name="mcs",               dtype="int32",      fill_value=-1),
    WindowMetaField(key="channel",           h5_name="channel",           dtype="int32",      fill_value=-1),
    WindowMetaField(key="recording_id",       h5_name="recording_id",      dtype=H5_STR_DTYPE, fill_value=""),
    WindowMetaField(key="modulation_type",   h5_name="modulation_type",   dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="waveform_type",     h5_name="waveform_type",     dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="spreading_factor",  h5_name="spreading_factor",  dtype="int32",      fill_value=-1),
    WindowMetaField(key="wifi_standard",     h5_name="wifi_standard",     dtype=H5_STR_DTYPE, fill_value=""),
    WindowMetaField(key="ieee802154_variant", h5_name="ieee802154_variant", dtype=H5_STR_DTYPE, fill_value=""),
    # Fixed at 2.048 Msps across all captures; populated per-window so the
    # aggregated corpus is self-describing (consumers don't need per-source configs).
    WindowMetaField(key="sample_rate_hz",    h5_name="sample_rate_hz",    dtype="float64",    fill_value=SAMPLE_RATE_HZ),
]

# ---------------------------------------------------------------------
# Croissant / provenance
# ---------------------------------------------------------------------

CITATIONS = [
    "J. Fontaine, A. Shahid, R. Elsas, A. Seferagic, I. Moerman, E. De Poorter, "
    "'Multi-band sub-GHz technology recognition on NVIDIA's Jetson Nano,' "
    "IEEE 92nd Vehicular Technology Conference (VTC-Fall), 2020.",
    f"Dataset repository: {DATASET_URL}",
    "Dataset files: https://cloud.ilabt.imec.be/index.php/s/bqXtdp9QsfXLbb3",
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
