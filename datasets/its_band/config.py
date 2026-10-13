"""
Dataset configuration for ITS Band Technology Recognition (Girmay et al.).

Real over-the-air IQ captures of LTE, 5G NR, ITS-G5, C-V2X PC5, WiFi, and Noise,
collected at IDLab (Ghent University-IMEC) on the Belgian Smart Highway testbed.
Synthetic AWGN is added at SNR levels {-10, -5, 0, 5, 10, 15, 20, 25, 30} dB on top
of the real-world captures (Girmay et al. §4.2) — this is a hybrid dataset.

Paper: https://doi.org/10.1016/j.vehcom.2022.100563

Notes on data selection:
- Six sampling rates available (1, 5, 10, 15, 20, 25 Msps). 25 Msps is used for v1
  as it is the highest rate available. Downstream users can decimate to a lower rate;
  upsampling a lower rate cannot recover detail. See docs/DATASETS.md for the full
  rationale.
- WiFi class included: WiFi_25msps.mat was located and verified to match the format of
  all other 25 Msps files (shape (2200, 7500) float64, variable name 'WiFi').
  wifi_standard = "802.11n" (Girmay et al. §4.1.1: openWiFi IEEE 802.11a/g/n SDR
  implementation). modulation_type = "unknown" — MCS varied per observation (MCS 0–7,
  §4.2), not stored per-sample in the released data.
- The compiled IEEE Dataport file (dataset_noise_added.mat) was NOT used — unclear
  provenance, 18500 vs 7500 samples/class, unknown augmentation applied.
"""

from pathlib import Path
import numpy as np

from pipeline.schemas import H5_STR_DTYPE, WindowMetaField
from datasets.identifiers import RECORD_A_DOI, RECORD_B_DOI, CODE_DOI

# ---------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------

DATASET_NAME = "ITS Band Technology Recognition (Girmay et al.)"
DATASET_URL = "https://gitlab.ilabt.imec.be/mgirmay/tech-rec-its-band"
SCRIPT_VERSION = "1.0.0"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_DATA_ROOT = Path("data/its_band/raw")
OUTPUT_ROOT = Path("data/its_band/processed")

OUTPUT_H5_PATH = OUTPUT_ROOT / "its_band.h5"
OUTPUT_CROISSANT_PATH = OUTPUT_ROOT / "its_band.croissant.json"

# ---------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------

"""
Assumptions
-----------
- Raw files are MATLAB v7.3 .mat files, one per technology
- Shape: (2200, 7500) as read by h5py; transposed to (7500, 2200), one row per observation
- Layout: first 1100 values = I channel, last 1100 = Q channel
- dtype: float64 in source → cast to float32
- 7500 observations per technology at 25 Msps
- No separate label array — label is fully determined by filename
"""

# ---------------------------------------------------------------------
# Labels (technology classes)
#
# Each .mat file contains one variable named after the technology.
# Internal variable names differ from the canonical label names used
# in the aggregated dataset.
# ---------------------------------------------------------------------

# filename stem → (mat_variable_name, canonical_label)
FILE_CONFIG = {
    "LTE_25msps":   ("LTE",    "LTE"),
    "FiveG_25msps": ("Five_G", "5G NR"),
    "ITSG5_25msps": ("ITSG5",  "ITS-G5"),
    "CV2X_25msps":  ("CV2X",   "C-V2X PC5"),
    "WiFi_25msps":  ("WiFi",   "WiFi"),
    "noise_25msps": ("Noise",  "Noise"),
}

TECHNOLOGY_TYPES = [label for _, label in FILE_CONFIG.values()]

# Alias for registry compatibility (extract_classification_metadata reads LABEL_REMAP)
LABEL_REMAP = {var: label for var, label in FILE_CONFIG.values()}

# Canonical label → physical-layer waveform type.
# LTE and 5G NR use OFDM (OFDMA DL / SC-FDMA UL, and CP-OFDM respectively).
# ITS-G5 is based on IEEE 802.11p which also uses OFDM.
# C-V2X PC5 (LTE-V2X sidelink) uses SC-FDM (single-carrier FDM / SC-FDMA).
# WiFi (802.11n) uses OFDM.
LABEL_TO_WAVEFORM = {
    "LTE":       "OFDM",
    "5G NR":     "OFDM",
    "ITS-G5":    "OFDM",
    "C-V2X PC5": "SC-FDM",
    "WiFi":      "OFDM",
    "Noise":     "noise",
}

# Canonical label → wifi_standard metadata.
# Only populated for WiFi; all other labels use the empty-string fill value.
# Source: Girmay et al. §4.1.1 — openWiFi IEEE 802.11a/g/n SDR implementation.
LABEL_TO_WIFI_STANDARD = {
    "WiFi": "802.11n",
}

# ---------------------------------------------------------------------
# Windowing
#
# 25 Msps → 1100 IQ samples per observation = 44 µs.
# Each .mat column stores [I(0..1099), Q(0..1099)] → 2200 floats.
# Each 1100-sample observation is split into 2×500-sample output windows
# (discarding the last 100 samples, ~9%).
# A recording_id links the two windows back to the same source
# observation for downstream train/test split integrity.
#
# OBSERVATION_SIZE: raw observation length as stored in the .mat file
# WINDOW_SIZE:      output window size written to HDF5 (consistent with all other datasets)
# ---------------------------------------------------------------------

SAMPLE_RATE_HZ = 25e6

OBSERVATION_SIZE = 1100   # raw observation length from .mat file

WINDOW_SIZE = 500         # output window size
HOP_SIZE = WINDOW_SIZE
PAD_LAST = False

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

SOURCE_DTYPE = "float64"
DATA_DTYPE = "float32"
NORMALIZE_MODE = "unit_power"
NOISE_POWER_THRESHOLD_DB = None   # signals are continuous; no inter-frame gaps to filter

# ---------------------------------------------------------------------
# Global dataset metadata
# ---------------------------------------------------------------------

GLOBAL_METADATA = {
    "dataset": DATASET_NAME,
    "description": (
        "Real over-the-air IQ captures from the Belgian Smart Highway testbed (imec/IDLab), "
        "augmented with synthetic AWGN at SNR levels {-10,-5,0,5,10,15,20,25,30} dB "
        "(hybrid dataset; Girmay et al., Sec. 4.2). "
        "Technologies: LTE (srsRAN, USRP X310), 5G NR (OpenAirInterface, NSA mode), "
        "ITS-G5 (Cohda MK5), C-V2X PC5 (Cohda MK6c), WiFi (openwifi SDR, Xilinx Zynq, 802.11n), and Noise. "
        "Captured at 25 Msps, 5.9 GHz center frequency. "
        "7500 observations per technology, each split into two 500-sample windows (15000 windows per technology). "
        "Note: the original paper uses the FFT of IQ as model input; RadioPile stores raw IQ windows."
    ),
    "schema_version": "1.0.0",
    "script_version": SCRIPT_VERSION,
    "observation_size": OBSERVATION_SIZE,
    "window_size": WINDOW_SIZE,
    "hop_size": HOP_SIZE,
    "pad_last": PAD_LAST,
    "normalization_mode": NORMALIZE_MODE or "none",
    "data_dtype": DATA_DTYPE,
    "technology_types": TECHNOLOGY_TYPES,
    "sample_rate_hz": SAMPLE_RATE_HZ,
    "center_frequency_hz": 5.9e9,
    "observations_per_technology": 7500,
    "windows_per_technology": 15000,
    "dataset_paper": "https://doi.org/10.1016/j.vehcom.2022.100563",
    "dataset_url": DATASET_URL,
}

# ---------------------------------------------------------------------
# Per-window metadata schema (stored as /meta/* datasets)
# ---------------------------------------------------------------------

WINDOW_METADATA_SCHEMA = [
    WindowMetaField(key="technology_type", h5_name="technology_type", dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="modulation_type", h5_name="modulation_type", dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="waveform_type",   h5_name="waveform_type",   dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="wifi_standard",   h5_name="wifi_standard",   dtype=H5_STR_DTYPE, fill_value=""),
    WindowMetaField(key="center_frequency_hz",  h5_name="center_frequency_hz",  dtype="float64",    fill_value=np.nan),
    WindowMetaField(key="sample_rate_hz",  h5_name="sample_rate_hz",  dtype="float64",    fill_value=np.nan),
    WindowMetaField(key="sample_index",    h5_name="sample_index",    dtype="int32",       fill_value=-1),
    WindowMetaField(key="recording_id",    h5_name="recording_id",    dtype=H5_STR_DTYPE,  fill_value=""),
    # WindowMetaField(key="source_file",  h5_name="source_file",  dtype=H5_STR_DTYPE, fill_value=""),  # per-file, add if needed
]

# ---------------------------------------------------------------------
# Croissant / provenance
# ---------------------------------------------------------------------

CITATIONS = [
    "M. Girmay, V. Maglogiannis, D. Naudts, M. Aslam, A. Shahid, I. Moerman, "
    "'Technology recognition and traffic characterization for wireless technologies in ITS band,' "
    "Vehicular Communications, vol. 39, p. 100563, 2023. "
    "https://doi.org/10.1016/j.vehcom.2022.100563",
    f"Dataset repository: {DATASET_URL}",
]

LICENSE = "CC-BY-4.0"

GLOBAL_METADATA["license"] = LICENSE
GLOBAL_METADATA["license_note"] = (
    "License of the original source dataset. This file was processed individually and never mixed, so no other dataset's terms apply to it. See LICENSES.md."
)

# Travels with the file: a per-dataset HDF5 separated from its README can still
# be traced to the record it was published in.
GLOBAL_METADATA["doi"] = RECORD_B_DOI
GLOBAL_METADATA["related_identifiers"] = f"aggregate={RECORD_A_DOI}; code={CODE_DOI}"
