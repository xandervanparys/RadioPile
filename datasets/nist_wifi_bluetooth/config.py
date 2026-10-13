"""
Dataset configuration for NIST WiFi IQ Recordings.

Real over-the-air WiFi captures at 2.4 GHz and 5 GHz collected with USRP B200-mini-i
SDRs at the University of Colorado Boulder campus.

Dataset source:      https://doi.org/10.18434/mds2-2731
Technical note (TN): https://doi.org/10.6028/NIST.TN.2237

Notes on file selection:
- 3 of the 5 HDF5 files are used. Excluded files:
  - 2_4ghz_outdoor.h5: occasional Bluetooth FHSS bleedthrough visible in spectrograms
  - 2_4ghz_bluetooth.h5: intentional co-channel WiFi + Bluetooth; cannot cleanly label
    individual windows without a signal detector
- 5ghz_outdoor.h5 is included. Its noise floor is ~16 dB above 5ghz_indoor, consistent
  with its higher receive gain (65 vs 30 in the per-file metadata CSVs; TN 2237 Table 8).
  Below the main WiFi mode it also has lower-power activity consistent with WiFi, which
  the noise filter drops (see the margin_db note below).
"""

from pathlib import Path
import numpy as np

from pipeline.schemas import H5_STR_DTYPE, WindowMetaField
from datasets.identifiers import RECORD_A_DOI, RECORD_B_DOI, CODE_DOI

# ---------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------

DATASET_NAME = "NIST WiFi IQ Recordings (Tschimben et al.)"
DATASET_URL = "https://doi.org/10.18434/mds2-2731"
SCRIPT_VERSION = "1.0.0"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_DATA_ROOT = Path("data/nist_wifi_bluetooth/raw")
OUTPUT_ROOT = Path("data/nist_wifi_bluetooth/processed")

OUTPUT_H5_PATH = OUTPUT_ROOT / "nist_wifi_bluetooth.h5"
OUTPUT_CROISSANT_PATH = OUTPUT_ROOT / "nist_wifi_bluetooth.croissant.json"

# ---------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------

"""
Assumptions
-----------
- Raw files are HDF5, one per band and environment, each holding 180 captures
  keyed by a timestamp string
- Each capture is 60,000,000 int16 values: interleaved I/Q at 30 Msps for 1 second
- Samples are scaled by 1/32768 to float32 in [-1, 1]
- One technology (WiFi) per file; the label is fully determined by the filename
"""

# ---------------------------------------------------------------------
# Labels (technology classes)
#
# filename stem → (center_frequency_hz, environment, canonical_label)
#
# Files not included:
#   2_4ghz_outdoor  — occasional Bluetooth bleedthrough in spectrograms
#   2_4ghz_bluetooth — co-channel WiFi + Bluetooth, unlabelable at window level
# ---------------------------------------------------------------------

FILE_CONFIG = {
    "2_4ghz_indoor": (2.437e9, "indoor",  "WiFi"),
    "5ghz_indoor":   (5.825e9, "indoor",  "WiFi"),
    "5ghz_outdoor":  (5.825e9, "outdoor", "WiFi"),
}

# File-level MCS metadata from NIST TN 2237 (doi: 10.6028/NIST.TN.2237).
# MCS was monitored on the iPerf client's link once per second during capture
# (`iw dev wlan0 link`, TN 2237 §3.4) and remained stable throughout:
#   2_4ghz_indoor: MCS-15 (802.11n,  64-QAM) in the 2.4 GHz band
#   5ghz_indoor:   MCS-8  (802.11ac, 256-QAM) in the 5 GHz band
#   5ghz_outdoor:  MCS-8  (802.11ac, 256-QAM) in the 5 GHz band
# These are capture-level constants, not per-frame decoded values. Traffic ran
# iPerf client → AP → iPerf server; the note reports monitoring on the client link
# only, so the annotation is not verified for the AP's own transmissions.
FILE_METADATA = {
    "2_4ghz_indoor": {"wifi_standard": "802.11n",  "modulation_type": "64QAM"},
    "5ghz_indoor":   {"wifi_standard": "802.11ac", "modulation_type": "256QAM"},
    "5ghz_outdoor":  {"wifi_standard": "802.11ac", "modulation_type": "256QAM"},
}

TECHNOLOGY_TYPES = ["WiFi"]

# Alias for registry compatibility (extract_classification_metadata reads LABEL_REMAP)
LABEL_REMAP = {"WiFi": "WiFi"}

# ---------------------------------------------------------------------
# Windowing
#
# 30 Msps → 500-sample windows = 16.7 µs, the aggregation target size, so no
# padding or cropping is needed. Each 1-second capture yields 60,000 windows.
# ---------------------------------------------------------------------

SAMPLE_RATE_HZ = 30e6

WINDOW_SIZE = 500
HOP_SIZE = 500
PAD_LAST = False

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

# Source dtype: int16 interleaved I/Q. Informational only; h5py reports the dtype.
SOURCE_DTYPE = "int16"
DATA_DTYPE = "float32"
NORMALIZE_MODE = "unit_power"

# Noise / idle filtering
#
# WiFi is packet-based: many windows fall in inter-frame gaps (near-zero power).
# Noise floor is auto-detected per file from the first capture, then applied to
# all captures in that file. This mirrors the sub_ghz pattern.
#
# All windows at or above the threshold are kept (no cap). Of 10,800,000 windows
# per file, the v1.0 build keeps:
#   2_4ghz_indoor:  9,559,474 (88.5%)
#   5ghz_indoor:    7,407,438 (68.6%)
#   5ghz_outdoor:   3,660,915 (33.9%)
# Total: 20,627,827 windows, about 89 GB.
# Users of the processed dataset should subsample as needed.
NOISE_POWER_THRESHOLD_DB = "auto"  # None, float (dB), or "auto"
NOISE_FLOOR_DETECTION_METHOD = "percentile"
NOISE_FLOOR_DETECTION_PARAMS = {
    "percentile": 1.0,
    # margin_db=6 is empirically validated across all three files:
    # - Indoor files have a 40+ dB signal-to-noise gap; any margin 3-15 dB works.
    # - 5ghz_outdoor has only ~13 dB gap: its noise floor sits at ~-41 dB (vs ~-58 dB
    #   for 5ghz_indoor). margin=6 places the threshold at -35.5 dB — between a
    #   lower-power mode at -38 to -36 dB, whose activity is consistent with WiFi, and
    #   the main WiFi signal mode — yielding 98.1% agreement with the amplitude-threshold
    #   method (max|I|>0.05) on that file. Both methods drop the lower-power mode, so the
    #   agreement does not show that the dropped windows are idle.
    "margin_db": 6.0,
}
NOISE_FLOOR_MAX_SAMPLES = 10000

# ---------------------------------------------------------------------
# Global dataset metadata
# ---------------------------------------------------------------------

GLOBAL_METADATA = {
    "dataset": DATASET_NAME,
    "description": (
        "Real over-the-air WiFi IQ captures collected with USRP B200-mini-i SDRs at "
        "the University of Colorado Boulder campus (November 2021). "
        "Three capture conditions: 2.4 GHz indoor, 5 GHz indoor, 5 GHz outdoor. "
        "30 Msps, int16 interleaved, 1-second captures. "
        "All signal windows extracted (noise filtered per file using auto percentile method)."
    ),
    "schema_version": "1.0.0",
    "script_version": SCRIPT_VERSION,
    "window_size": WINDOW_SIZE,
    "hop_size": HOP_SIZE,
    "pad_last": PAD_LAST,
    "normalization_mode": NORMALIZE_MODE,
    "data_dtype": DATA_DTYPE,
    "technology_types": TECHNOLOGY_TYPES,
    "sample_rate_hz": SAMPLE_RATE_HZ,
    "noise_power_threshold_db": NOISE_POWER_THRESHOLD_DB,
    "noise_floor_detection_method": NOISE_FLOOR_DETECTION_METHOD,
    "noise_floor_detection_params": NOISE_FLOOR_DETECTION_PARAMS,
    "noise_floor_max_samples": NOISE_FLOOR_MAX_SAMPLES,
    "dataset_url": DATASET_URL,
}

# ---------------------------------------------------------------------
# Per-window metadata schema (stored as /meta/* datasets)
# ---------------------------------------------------------------------

WINDOW_METADATA_SCHEMA = [
    WindowMetaField(key="technology_type", h5_name="technology_type", dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="modulation_type", h5_name="modulation_type", dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="waveform_type",   h5_name="waveform_type",   dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="recording_id",    h5_name="recording_id",    dtype=H5_STR_DTYPE, fill_value=""),
    WindowMetaField(key="center_frequency_hz",  h5_name="center_frequency_hz",  dtype="float64",    fill_value=np.nan),
    WindowMetaField(key="environment",     h5_name="environment",     dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="sample_rate_hz",  h5_name="sample_rate_hz",  dtype="float64",    fill_value=np.nan),
    WindowMetaField(key="sample_index",    h5_name="sample_index",    dtype="int32",       fill_value=-1),
    # WiFi variant sub-classification, populated from FILE_METADATA (NIST TN 2237).
    # MCS was continuously monitored and remained stable: MCS-15 (802.11n) at 2.4 GHz,
    # MCS-8 (802.11ac) at 5 GHz. These are capture-level constants, not per-frame decoded values.
    WindowMetaField(key="wifi_standard",   h5_name="wifi_standard",   dtype=H5_STR_DTYPE, fill_value=""),
    # WindowMetaField(key="source_file",  h5_name="source_file",  dtype=H5_STR_DTYPE, fill_value=""),  # filename stem, add if needed
]

# ---------------------------------------------------------------------
# Croissant / provenance
# ---------------------------------------------------------------------

CITATIONS = [
    "S. Tschimben, S. Subray, A. Sanders, A. Wunderlich, "
    "'Wi-Fi and Bluetooth I/Q Recordings in the 2.4 GHz and 5 GHz Bands with Low-Cost "
    "Software Defined Radios,' NIST, 2022. https://doi.org/10.18434/mds2-2731",
    "S. Tschimben, S. Subray, A. Sanders, A. Wunderlich, "
    "'Wi-Fi and Bluetooth I/Q Recordings in the 2.4 GHz and 5 GHz Bands with Low-Cost "
    "Software Defined Radios,' NIST Technical Note 2237, 2022. https://doi.org/10.6028/NIST.TN.2237",
]

LICENSE = "NIST-public-domain"

GLOBAL_METADATA["license"] = LICENSE
GLOBAL_METADATA["license_note"] = (
    "License of the original source dataset. This file was processed individually and never mixed, so no other dataset's terms apply to it. See LICENSES.md."
)

# Travels with the file: a per-dataset HDF5 separated from its README can still
# be traced to the record it was published in.
GLOBAL_METADATA["doi"] = RECORD_B_DOI
GLOBAL_METADATA["related_identifiers"] = f"aggregate={RECORD_A_DOI}; code={CODE_DOI}"
