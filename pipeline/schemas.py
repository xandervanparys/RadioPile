"""
Schema definitions for HDF5 dataset metadata.

This module provides declarative schemas for defining per-window metadata
fields that are stored in HDF5 datasets, as well as controlled vocabularies
for task_type and label_represents used across the dataset registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import h5py

H5_STR_DTYPE = h5py.string_dtype(encoding="utf-8")


class TaskType(str, Enum):
    """
    Controlled vocabulary for the task_type field on DatasetEntry.

    Using str-Enum means values serialize as plain strings, so comparisons
    (entry.task_type == "modulation_recognition") and HDF5 metadata writes
    work on the value directly.
    """
    MODULATION_RECOGNITION = "modulation_recognition"
    TECHNOLOGY_RECOGNITION = "technology_recognition"
    RADAR_CLASSIFICATION   = "radar_classification"
    UNKNOWN                = "unknown"

    # Without this, str(TaskType.MODULATION_RECOGNITION) is
    # "TaskType.MODULATION_RECOGNITION". A str-mixin Enum inherits Enum.__str__,
    # not str.__str__ — the value only survives comparison, not stringification.
    # Written to HDF5 that way, task_type would read "TaskType.*" and the
    # documented filter (task_type == "technology_recognition") would match nothing.
    __str__ = str.__str__


class LabelRepresents(str, Enum):
    """
    Controlled vocabulary for the label_represents field on DatasetEntry.

    Drives routing in extract_classification_metadata:
    - MODULATION    → populates modulation_type (fine) from the primary label;
                      waveform_type carries the modulation family (coarse)
    - TECHNOLOGY    → populates technology_type (fine) from the primary label;
                      waveform_type carries the physical-layer waveform (coarse)
    - RADAR_WAVEFORM → populates radar_signal_type (fine) from the primary label;
                      waveform_type carries the radar intra-pulse family (coarse).
                      technology_type and modulation_type stay "unknown".
    """
    MODULATION     = "modulation"
    TECHNOLOGY     = "technology"
    # See DESIGN.md §1.2 for per-domain semantics. Radar is symmetric with the
    # other two domains: a fine target column (radar_signal_type) below a coarse
    # waveform_type family (see RADAR_FINE_TO_COARSE below).
    RADAR_WAVEFORM = "radar_waveform"

    # Same reason as TaskType above. Not currently written to HDF5, but it is
    # the same latent trap for anything that stringifies it.
    __str__ = str.__str__


# Canonical label names per task type.
#
# Purpose: single source of truth for label strings used across all dataset
# configs. When adding a new dataset, look up the canonical name here before
# defining LABEL_REMAP or MODULATION_TYPES — this prevents silent duplicates
# caused by inconsistent capitalization or abbreviation ("5GNR" vs "5G NR").
#
# This is a reference structure, not enforced at runtime.
#
# To add a new label: append it to the appropriate list and document its source.
CANONICAL_LABELS: dict[TaskType, list[str]] = {
    TaskType.TECHNOLOGY_RECOGNITION: [
        # Cellular
        "LTE",                      # lte_wifi_dvbt, its_band
        "5G NR",                    # its_band
        # Broadcast
        "DVB-T",                    # lte_wifi_dvbt
        # WiFi (all 802.11 variants — distinguish via wifi_standard metadata)
        "WiFi",                     # lte_wifi_dvbt, nist_wifi_bluetooth, sub_ghz (802.11ah)
        # V2X
        "ITS-G5",                   # its_band
        "C-V2X PC5",                # its_band
        # Sub-GHz IoT (sub_ghz)
        "LoRa",                     # spreading_factor metadata: SF7–SF12
        "Sigfox",
        "IEEE 802.15.4",            # ieee802154_variant metadata: "g", "SUN-OFDM"
        # Background
        "Noise",                    # sub_ghz, its_band
    ],
    TaskType.MODULATION_RECOGNITION: [
        # rml2016_10a
        "BPSK", "QPSK", "8PSK", "PAM4", "16QAM", "64QAM",
        "GFSK", "CPFSK", "WBFM", "AM-DSB", "AM-SSB",
        # rml2018_01a additions
        "OOK", "4ASK", "8ASK", "16PSK", "32PSK",
        "16APSK", "32APSK", "64APSK", "128APSK",
        "32QAM", "128QAM", "256QAM",
        "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
        "FM", "GMSK", "OQPSK",
    ],
    TaskType.RADAR_CLASSIFICATION: [
        # radchar — fine radar signal types. These appear as the primary label
        # (source_label) and in the radar_signal_type column, parallel to
        # modulation_type / technology_type. The coarse grouping above them is
        # waveform_type, mapped via RADAR_FINE_TO_COARSE below.
        "coherent_pulse_train",
        "barker_code",
        "polyphase_barker_code",
        "frank_code",
        "linear_frequency_modulated",
    ],
}


# Radar fine signal type → coarse waveform_type family.
#
# This is the radar counterpart to MODULATION_TO_WAVEFORM (rml configs) and
# LABEL_TO_WAVEFORM (sub_ghz config): it lifts the fine per-window radar signal
# type to the coarse waveform_type family that sits one level above it, giving
# radar the same two-level (coarse waveform_type → fine *_type) structure as the
# modulation and technology domains.
#
# Grouping is the standard radar intra-pulse modulation taxonomy:
#   - frequency_modulated: intra-pulse frequency modulation (LFM / chirp)
#   - phase_modulated:         intra-pulse phase coding (Barker, polyphase Barker, Frank)
#   - unmodulated:         no intra-pulse modulation (coherent pulse train of CW pulses)
# See DESIGN.md §1.2 for the literature basis. Spelled-out lowercase form
# matches the radar sub-domain's own value style (e.g. linear_frequency_modulated).
RADAR_FINE_TO_COARSE: dict[str, str] = {
    "linear_frequency_modulated": "frequency_modulated",
    "barker_code":                "phase_modulated",
    "polyphase_barker_code":      "phase_modulated",
    "frank_code":                 "phase_modulated",
    "coherent_pulse_train":       "unmodulated",
}

# The three coarse radar waveform_type values (canonical, deduplicated, order-stable).
RADAR_WAVEFORM_FAMILIES: list[str] = ["frequency_modulated", "phase_modulated", "unmodulated"]


@dataclass(frozen=True)
class WindowMetaField:
    """
    Declarative schema for one per-window metadata field stored in HDF5.

    key
        Key expected in the `meta` dict yielded by the dataset-specific iterator.
    h5_name
        Dataset name under /meta.
    dtype
        HDF5 dtype.
    fill_value
        Value used when the key is missing or None.
    """

    key: str
    h5_name: str
    dtype: object
    fill_value: object


# Minimal per-window metadata that is always written for every dataset.
# Only the truly essential field is required by default.
REQUIRED_WINDOW_META: Sequence[WindowMetaField] = (
    # Keep the on-disk name stable ("file_path") even though iterators use "source_file".
    WindowMetaField(key="source_file", h5_name="file_path", dtype=H5_STR_DTYPE, fill_value=""),
)

# Optional windowing metadata fields that are automatically generated by windowing functions.
# These are always available in the metadata dict but are only stored if explicitly
# included in the window_metadata_schema.
# They are useful for debugging or reconstruction but are omitted by default to save space.
#
# Usage: add them to a dataset's WINDOW_METADATA_SCHEMA to store them:
#   from pipeline.schemas import WINDOWING_META_FIELDS
#   WINDOW_METADATA_SCHEMA = list(WINDOWING_META_FIELDS) + [custom_fields...]
WINDOWING_META_FIELDS: Sequence[WindowMetaField] = (
    WindowMetaField(key="start_idx", h5_name="start_idx", dtype="int64", fill_value=-1),
    WindowMetaField(key="padded", h5_name="padded", dtype="bool", fill_value=False),
)
