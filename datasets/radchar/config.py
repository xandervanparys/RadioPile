"""
Dataset configuration for RadChar (Radar Signal Characterisation).

RadChar is a synthetic radar signal dataset for multi-task learning.
It contains 5 radar signal types with variable SNR and pulse parameters.

Radar is modeled as a two-level classification task, symmetric with the
technology and modulation domains:
  - radar_signal_type (fine): the RadChar signal class, parallel to
    technology_type / modulation_type.
  - waveform_type (coarse): the radar intra-pulse family one level above it
    (frequency_modulated / phase_modulated / unmodulated), mapped via
    RADAR_SIGNAL_TO_WAVEFORM (= pipeline.schemas.RADAR_FINE_TO_COARSE).

Dataset source: https://github.com/abcxyzi/RadChar
Paper: Z. Huang et al., "Multi-Task Learning For Radar Signal Characterisation," ICASSPW 2023.
"""

from pathlib import Path
import numpy as np

from pipeline.schemas import H5_STR_DTYPE, WindowMetaField, RADAR_FINE_TO_COARSE
from datasets.identifiers import RECORD_A_DOI, RECORD_B_DOI, CODE_DOI

# ---------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------

DATASET_NAME = "RadChar (Radar Signal Characterisation)"
DATASET_URL = "https://github.com/abcxyzi/RadChar"
SCRIPT_VERSION = "1.0.0"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

RAW_DATA_ROOT = Path("data/radchar/raw")
OUTPUT_ROOT = Path("data/radchar/processed")

OUTPUT_H5_PATH = OUTPUT_ROOT / "radchar.h5"
OUTPUT_CROISSANT_PATH = OUTPUT_ROOT / "radchar.croissant.json"

# Raw HDF5 filename (RadChar-Baseline = 1M samples)
RAW_H5_FILENAME = "RadChar-Baseline.h5"

# ---------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------

"""
Assumptions
-----------
- Raw data is a single HDF5 file with two datasets:
    iq:     shape (1000000, 512), dtype complex128 — complex baseband IQ samples
    labels: shape (1000000,),     dtype structured array with fields:
              index, signal_type, number_of_pulses, pulse_width,
              time_delay, pulse_repetition_interval, signal_to_noise_ratio
- Each sample is a pre-windowed 512-point complex IQ capture
- signal_type is an integer 0–4 mapping to radar waveform types (see SIGNAL_TYPES below)
- SNR range: -20 to +20 dB
- Sampling rate: 3.2 MHz
- Radar signal parameters (number_of_pulses, pulse_width, etc.) are regression targets
  stored in per-window metadata for future multi-task learning experiments.
"""

# ---------------------------------------------------------------------
# Labels (radar signal types)
# ---------------------------------------------------------------------
#
# Two levels: the fine radar_signal_type below (these five classes) and the
# coarse waveform_type family above them (RADAR_SIGNAL_TO_WAVEFORM, defined
# further down).

# Integer → radar signal type mapping
SIGNAL_TYPES = {
    0: "coherent_pulse_train",
    1: "barker_code",
    2: "polyphase_barker_code",
    3: "frank_code",
    4: "linear_frequency_modulated",
}

# Ordered list for index-based access (matches integer encoding)
RADAR_SIGNAL_TYPES = [
    "coherent_pulse_train",
    "barker_code",
    "polyphase_barker_code",
    "frank_code",
    "linear_frequency_modulated",
]

# Fine radar signal type → coarse waveform_type family.
# Sourced from the canonical registry (pipeline.schemas.RADAR_FINE_TO_COARSE);
# aliased here under the *_TO_WAVEFORM name used by the other domains' configs
# (cf. MODULATION_TO_WAVEFORM in rml, LABEL_TO_WAVEFORM in sub_ghz) so the build
# script populates waveform_type the same way across all task families.
RADAR_SIGNAL_TO_WAVEFORM = RADAR_FINE_TO_COARSE

# ---------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------

SAMPLE_RATE_HZ = 3.2e6

# Each sample is already 512 points — use directly
WINDOW_SIZE = 512
HOP_SIZE = 512
PAD_LAST = False

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

# Source dtype: HDF5 iq dataset is complex128.
# The real and imaginary parts are extracted and cast to float32.
SOURCE_DTYPE = "complex128"

DATA_DTYPE = "float32"

# Note: RadChar applies unity average power normalization during dataset
# generation (Huang et al. 2023, Section 2.1). The unit_power normalization
# applied here is therefore idempotent but is retained for pipeline
# consistency across all datasets.
NORMALIZE_MODE = "unit_power"

NOISE_POWER_THRESHOLD_DB = None

# ---------------------------------------------------------------------
# Global dataset metadata
# ---------------------------------------------------------------------

GLOBAL_METADATA = {
    "dataset": DATASET_NAME,
    "description": (
        "Synthetic radar signal dataset for multi-task classification and regression. "
        "Contains 5 radar signal types: coherent pulse train, Barker code, "
        "polyphase Barker code, Frank code, and linear frequency-modulated (LFM) pulse. "
        "SNR range: -20 to +20 dB. Each waveform contains 512 complex baseband IQ samples "
        "at a sampling rate of 3.2 MHz. Labels include the signal type and pulse parameters "
        "(number of pulses, pulse width, time delay, PRI) for multi-task learning."
    ),
    "schema_version": "1.0.0",
    "script_version": SCRIPT_VERSION,
    "window_size": WINDOW_SIZE,
    "hop_size": HOP_SIZE,
    "pad_last": PAD_LAST,
    "normalization_mode": NORMALIZE_MODE or "none",
    "data_dtype": DATA_DTYPE,
    "radar_signal_types": RADAR_SIGNAL_TYPES,
    "snr_range_db": "[-20, 20] in increments of 1",
    "sample_rate_hz": SAMPLE_RATE_HZ,
    "dataset_url": DATASET_URL,
}

# ---------------------------------------------------------------------
# Per-window metadata schema (stored as /meta/* datasets)
# ---------------------------------------------------------------------

WINDOW_METADATA_SCHEMA = [
    # radar_signal_type: the fine per-window radar signal class (coherent_pulse_train,
    # barker_code, …). This is radar's fine classification target, parallel to
    # technology_type / modulation_type in the other domains.
    WindowMetaField(key="radar_signal_type", h5_name="radar_signal_type", dtype=H5_STR_DTYPE, fill_value="unknown"),
    # waveform_type: the coarse radar intra-pulse family one level above radar_signal_type
    # (frequency_modulated / phase_modulated / unmodulated), mapped via RADAR_SIGNAL_TO_WAVEFORM.
    # Symmetric with waveform_type = modulation family / physical-layer waveform in
    # the other domains.
    # See DESIGN.md §1.2 for the literature basis for the grouping.
    WindowMetaField(key="waveform_type",     h5_name="waveform_type",     dtype=H5_STR_DTYPE, fill_value="unknown"),
    WindowMetaField(key="snr_db",            h5_name="snr_db",            dtype="float32",    fill_value=np.nan),
    WindowMetaField(key="number_of_pulses",  h5_name="number_of_pulses",  dtype="int32",      fill_value=-1),
    WindowMetaField(key="pulse_width_s",     h5_name="pulse_width_s",     dtype="float32",    fill_value=np.nan),
    WindowMetaField(key="time_delay_s",      h5_name="time_delay_s",      dtype="float32",    fill_value=np.nan),
    WindowMetaField(key="pri_s",             h5_name="pri_s",             dtype="float32",    fill_value=np.nan),
    WindowMetaField(key="sample_index",      h5_name="sample_index",      dtype="int32",      fill_value=-1),

    WindowMetaField(key="sample_rate_hz",    h5_name="sample_rate_hz",    dtype="float64",    fill_value=SAMPLE_RATE_HZ),
]

# ---------------------------------------------------------------------
# Croissant / provenance
# ---------------------------------------------------------------------

CITATIONS = [
    "Z. Huang, A. Pemasiri, S. Denman, C. Fookes, T. Martin, "
    "'Multi-Task Learning For Radar Signal Characterisation,' "
    "2023 IEEE International Conference on Acoustics, Speech and Signal Processing Workshops "
    "(ICASSPW), 2023, pp. 1-5. https://doi.org/10.1109/ICASSPW59220.2023.10193318",
    f"Dataset repository: {DATASET_URL}",
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
