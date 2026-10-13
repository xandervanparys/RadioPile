"""
Dataset configuration for RadioML 2018.01A (Modulation Classification).

This dataset contains 24 modulation classes with variable-SNR samples generated
using GNU Radio. It is the successor to RML2016.10A with more modulation types,
longer windows (1024 samples), and a wider SNR range (-20 to +30 dB).

IMPORTANT: The label indices in the HDF5 file (Y one-hot vectors) correspond to
a scrambled class order. The corrected mapping is documented in classes-fixed.txt
and used in MODULATION_TYPES below.

Published by DeepSig; DATASET_URL and DATASET_MIRROR_URL below.
"""

from pathlib import Path
import numpy as np

from pipeline.schemas import H5_STR_DTYPE, WindowMetaField
from datasets.identifiers import RECORD_A_DOI, RECORD_B_DOI, CODE_DOI

# ---------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------

DATASET_NAME = "RadioML 2018.01A (Modulation Classification)"
DATASET_URL = "https://www.deepsig.ai/datasets"
DATASET_MIRROR_URL = "https://www.kaggle.com/datasets/pinxau1000/radioml2018"
SCRIPT_VERSION = "1.0.0"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_DATA_ROOT = Path("data/rml2018_01a/raw")
OUTPUT_ROOT = Path("data/rml2018_01a/processed")

OUTPUT_H5_PATH = OUTPUT_ROOT / "rml2018_01a.h5"
OUTPUT_CROISSANT_PATH = OUTPUT_ROOT / "rml2018_01a.croissant.json"

# Raw HDF5 file
RAW_H5_FILENAME = "GOLD_XYZ_OSC.0001_1024.hdf5"

# ---------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------

"""
Assumptions
-----------
- Raw data is a single HDF5 file: GOLD_XYZ_OSC.0001_1024.hdf5 (~21 GB)
- HDF5 structure:
    X: shape (2555904, 1024, 2), dtype float32 — IQ samples [time, I/Q]
    Y: shape (2555904, 24), dtype int64 — one-hot encoded labels
    Z: shape (2555904, 1), dtype int64 — SNR values in dB
- Each sample is a pre-windowed 1024-point IQ capture
- 2555904 total samples = 24 modulations × 26 SNR levels × 4096 samples each
- SNR range: -20 to +30 dB in 2 dB increments (26 levels)
- Label indices in Y are SCRAMBLED — the corrected order is in classes-fixed.txt
  and encoded in MODULATION_TYPES below (index 0 = OOK, not 32PSK)
"""

# ---------------------------------------------------------------------
# Labels (modulation classes)
# ---------------------------------------------------------------------

# The corrected class order from classes-fixed.txt.
# Index i in the one-hot Y vector corresponds to MODULATION_TYPES[i].
MODULATION_TYPES = [
    "OOK",
    "4ASK",
    "8ASK",
    "BPSK",
    "QPSK",
    "8PSK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "16QAM",
    "32QAM",
    "64QAM",
    "128QAM",
    "256QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-WC",
    "AM-DSB-SC",
    "FM",
    "GMSK",
    "OQPSK",
]

# Modulation → physical-layer waveform family.
# Groups specific modulation schemes into their parent family for cross-dataset
# waveform classification alongside RML2016.10A and real-capture datasets.
MODULATION_TO_WAVEFORM = {
    "OOK":      "ASK",    # OOK = 2-ASK (on-off keying)
    "4ASK":     "ASK",
    "8ASK":     "ASK",
    "BPSK":     "PSK",
    "QPSK":     "PSK",
    "8PSK":     "PSK",
    "16PSK":    "PSK",
    "32PSK":    "PSK",
    "16APSK":   "APSK",
    "32APSK":   "APSK",
    "64APSK":   "APSK",
    "128APSK":  "APSK",
    "16QAM":    "QAM",
    "32QAM":    "QAM",
    "64QAM":    "QAM",
    "128QAM":   "QAM",
    "256QAM":   "QAM",
    "AM-SSB-WC": "AM",
    "AM-SSB-SC": "AM",
    "AM-DSB-WC": "AM",
    "AM-DSB-SC": "AM",
    "FM":       "FM",
    "GMSK":     "FSK",    # Gaussian MSK is a continuous-phase FSK variant
    "OQPSK":    "PSK",    # Offset QPSK is a PSK variant
}

# ---------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------

# Each sample is already 1024 points — use directly
WINDOW_SIZE = 1024
HOP_SIZE = 1024
PAD_LAST = False

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

# Source dtype: HDF5 contains float32 arrays.
# Informational only — h5py handles dtype automatically.
SOURCE_DTYPE = "float32"

DATA_DTYPE = "float32"

NORMALIZE_MODE = "unit_power"

# Noise filtering disabled — SNR levels are part of the dataset design
NOISE_POWER_THRESHOLD_DB = None

# ---------------------------------------------------------------------
# Global dataset metadata
# ---------------------------------------------------------------------

GLOBAL_METADATA = {
    "dataset": DATASET_NAME,
    "description": (
        "Variable-SNR modulation classification dataset containing 24 modulation "
        "classes (digital: OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK, "
        "16APSK, 32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, "
        "GMSK, OQPSK; analog: AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM). "
        "SNR values range from -20 to +30 dB in 2 dB increments. "
        "Each (modulation, SNR) combination contains 4096 samples of 1024 IQ points."
    ),
    "schema_version": "1.0.0",
    "script_version": SCRIPT_VERSION,
    "window_size": WINDOW_SIZE,
    "hop_size": HOP_SIZE,
    "pad_last": PAD_LAST,
    "normalization_mode": NORMALIZE_MODE or "none",
    "data_dtype": DATA_DTYPE,
    "modulation_classes": MODULATION_TYPES,
    "snr_range_db": "[-20, 30] in increments of 2",
    "samples_per_combination": 4096,
    "dataset_url": DATASET_URL,
}

# ---------------------------------------------------------------------
# Per-window metadata schema (stored as /meta/* datasets)
# ---------------------------------------------------------------------

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
    "T. J. O'Shea, T. Roy, T. C. Clancy, 'Over-the-Air Deep Learning Based Radio Signal "
    "Classification,' IEEE Journal of Selected Topics in Signal Processing, vol. 12, no. 1, "
    "pp. 168-179, 2018.",
    f"Dataset: DeepSig Inc., RadioML 2018.01A. {DATASET_URL}",
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
