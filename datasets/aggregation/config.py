"""
Dataset configuration for RadioPile, the aggregated multi-task RF IQ collection.

This configuration combines multiple processed datasets into a unified dataset
with multi-target metadata (technology + modulation + radar signal type).
"""

from pathlib import Path
import numpy as np
import importlib

from pipeline.schemas import H5_STR_DTYPE, WindowMetaField
from datasets.identifiers import RECORD_A_DOI, RECORD_B_DOI, CODE_DOI

# ---------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------

DATASET_NAME = "RadioPile: A Unified Multi-Task RF IQ Collection"
SCRIPT_VERSION = "1.0.0"

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

OUTPUT_ROOT = Path("data/aggregation/processed")

OUTPUT_H5_PATH = OUTPUT_ROOT / "aggregated.h5"
OUTPUT_CROISSANT_PATH = OUTPUT_ROOT / "aggregated.croissant.json"

# ---------------------------------------------------------------------
# Dataset Selection (using registry)
# ---------------------------------------------------------------------

# Use dataset registry - just specify which datasets to include
SELECTED_DATASETS = {
    "rml2016_10a": {
        "enabled": True,
        "use_existing": False,  # True reuses an existing processed file instead of rebuilding
    },
    "sub_ghz": {
        "enabled": True,
        "use_existing": False,
    },
    "lte_wifi_dvbt": {
        "enabled": True,
        "use_existing": False,
    },
    "rml2018_01a": {
        "enabled": True,
        "use_existing": False,
    },
    "radchar": {
        "enabled": True,
        "use_existing": False
    },
    "its_band": {
        "enabled": True,
        "use_existing": False
    },
    "nist_wifi_bluetooth": {
        "enabled": True,
        "use_existing": False,
    },
}

# For each dataset, datasets/registry.py provides the processed HDF5 path, the
# build module, the raw data root, the task type and what the label represents.
# Window size, normalization and the metadata schema are read from the dataset's
# own config.

# ---------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------

TARGET_WINDOW_SIZE = 500
HOP_SIZE = 500  # For datasets that need re-windowing
PAD_LAST = True

# ---------------------------------------------------------------------
# Window Size Handling Strategy
# ---------------------------------------------------------------------

# For source < target: padding strategy
PADDING_STRATEGY = "zero"  # Shorter windows are zero-padded at the end

# For source > target: crop policy
CROP_POLICY = "start"  # Longer windows keep their first TARGET_WINDOW_SIZE samples
# Cropping is a deterministic, lossy fallback used only when source windows exceed the target size.
# The source window size is read from each HDF5's data shape, falling back to its config.

# ---------------------------------------------------------------------
# Label Configuration (Multi-target Metadata)
# ---------------------------------------------------------------------

PRIMARY_LABEL_TYPE = "technology"  # Options: "technology" (v1 release), "modulation", "waveform"
# Primary label used for HDF5 structure (required by pipeline)
# All classification targets stored in metadata fields (technology_type, modulation_type, waveform_type)

# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

# IQ storage dtype:
# - "float32" (default, for ML/efficiency)
# - "float64" (use only if extra precision is needed)
DATA_DTYPE = "float32"
NORMALIZE_MODE = "unit_power"  # One mode for every source

# ---------------------------------------------------------------------
# v1 Frozen Defaults (for reproducibility)
# ---------------------------------------------------------------------
# Values below are those recorded in the shipped build's config snapshot,
# release/v1.0/aggregated/radiopile_v1.0.h5.config.json.
# - PADDING_STRATEGY   = "zero"
# - CROP_POLICY        = "start"
# - PRIMARY_LABEL_TYPE = "technology"
# - NORMALIZE_MODE     = "unit_power"
# - DATA_DTYPE         = "float32"
# - TARGET_WINDOW_SIZE = 500
# Everything else can remain configurable but unused in v1

# ---------------------------------------------------------------------
# Unified Metadata Schema
# ---------------------------------------------------------------------

# Fill value convention:
# - "unknown" for categorical strings (technology_type, modulation_type, waveform_type, etc.)
# - np.nan for numeric fields (sample_rate_hz, snr_db, etc.)
# - -1 for integer IDs (mcs, channel, etc.)
# - "" (empty string) for optional/grouping string fields (recording_id, location, etc.)

# Required fields (always present)
REQUIRED_METADATA_SCHEMA = [
    WindowMetaField("source_dataset",      "source_dataset",      H5_STR_DTYPE, "unknown"),
    WindowMetaField("source_label",        "source_label",        H5_STR_DTYPE, "unknown"),
    WindowMetaField("task_type",           "task_type",           H5_STR_DTYPE, "unknown"),
    WindowMetaField("was_padded",          "was_padded",          "bool",        False),
    WindowMetaField("padding_strategy",    "padding_strategy",    H5_STR_DTYPE, "none"),
    WindowMetaField("technology_type",     "technology_type",     H5_STR_DTYPE, "unknown"),
    WindowMetaField("modulation_type",     "modulation_type",     H5_STR_DTYPE, "unknown"),
    # Fine radar target, parallel to technology_type / modulation_type. Populated for
    # radar (radchar) windows; "unknown" for all other domains.
    WindowMetaField("radar_signal_type",   "radar_signal_type",   H5_STR_DTYPE, "unknown"),
    # Coarse family above the three fine targets above (modulation family /
    # physical-layer waveform / radar intra-pulse family). See DESIGN.md §1.2
    # for per-domain semantics.
    WindowMetaField("waveform_type",       "waveform_type",       H5_STR_DTYPE, "unknown"),
    WindowMetaField("recording_id",        "recording_id",        H5_STR_DTYPE, ""),
    WindowMetaField("original_window_size","original_window_size","int32",       -1),
    WindowMetaField("was_cropped", "was_cropped", "bool", False),
    WindowMetaField("crop_policy", "crop_policy", H5_STR_DTYPE, "none"),
    # WindowMetaField("source_file", "source_file", H5_STR_DTYPE, ""),
]

# Optional RF metadata fields (fill with NaN/-1/"" if missing)
OPTIONAL_METADATA_SCHEMA = [
    WindowMetaField("sample_rate_hz",       "sample_rate_hz",       "float64", np.nan),
    WindowMetaField("center_frequency_hz",  "center_frequency_hz",  "float64", np.nan),
    WindowMetaField("snr_db",               "snr_db",               "float32", np.nan),
    # WindowMetaField("gain_db",              "gain_db",              "float32", np.nan),
    # WindowMetaField("attenuation_db",       "attenuation_db",       "float32", np.nan),
    # WindowMetaField("location",             "location",             H5_STR_DTYPE, ""),
    # WindowMetaField("mcs",                  "mcs",                  "int32",   -1),
    # WindowMetaField("channel",              "channel",              "int32",   -1),
]

WINDOW_METADATA_SCHEMA = REQUIRED_METADATA_SCHEMA + OPTIONAL_METADATA_SCHEMA

# ---------------------------------------------------------------------
# Global dataset metadata
# ---------------------------------------------------------------------

# Collect source dataset information
SOURCE_DATASETS = [name for name, config in SELECTED_DATASETS.items() if config.get("enabled", True)]

# Single source of truth for the prose description: this feeds both the HDF5 root
# `description` attribute (via GLOBAL_METADATA) and the Croissant `description`,
# so the two can never drift.
DATASET_DESCRIPTION = (
    "RadioPile unifies seven publicly available RF IQ datasets into a single "
    "machine-learning-ready collection through a reproducible, extensible, open-source "
    "preprocessing pipeline. Version 1.0 comprises approximately 25.8 million labeled IQ "
    "windows (12.9 billion IQ samples) spanning ten technology classes, 32 modulation "
    "classes, and five radar signal types, mixing software-defined radio captures with "
    "simulated signals. Every window carries standardized metadata, including technology, "
    "modulation, waveform family, SNR, source dataset, and per-window provenance, under a "
    "single multi-target label schema."
)

# ---------------------------------------------------------------------
# Persistent identifiers
# ---------------------------------------------------------------------
# Defined in datasets/identifiers.py, shared with the per-dataset configs.

GLOBAL_METADATA = {
    "dataset": DATASET_NAME,
    "description": DATASET_DESCRIPTION,
    "schema_version": "1.0.0",
    "script_version": SCRIPT_VERSION,
    "source_datasets": SOURCE_DATASETS,
    "target_window_size": TARGET_WINDOW_SIZE,
    "padding_strategy": PADDING_STRATEGY,
    "crop_policy": CROP_POLICY,
    "primary_label_type": PRIMARY_LABEL_TYPE,
    "normalization_mode": NORMALIZE_MODE,
    "data_dtype": DATA_DTYPE,
    # Travels with the file. A 116 GB HDF5 gets copied onto clusters and into
    # scratch space without its README or Croissant sidecar; the root attributes
    # are the only thing a holder of the bare file can still read.
    "doi": RECORD_A_DOI,
    "related_identifiers": f"per_dataset={RECORD_B_DOI}; code={CODE_DOI}",
}

# ---------------------------------------------------------------------
# Croissant / provenance
# ---------------------------------------------------------------------

# Collect citations and licenses from source datasets
CITATIONS = []
LICENSES = []

for dataset_name in SOURCE_DATASETS:
    try:
        # Import the dataset's config module
        module = importlib.import_module(f"datasets.{dataset_name}.config")

        # Collect citations
        if hasattr(module, "CITATIONS"):
            CITATIONS.extend(module.CITATIONS)

        # Collect licenses
        if hasattr(module, "LICENSE"):
            license_value = module.LICENSE
            if license_value and license_value not in LICENSES:
                LICENSES.append(license_value)
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not load citations/license for dataset '{dataset_name}': {e}")

# How to cite the collection itself; written as `citeAs` in the Croissant sidecar.
COLLECTION_CITATION = (
    "Vanparys, X., Cheraghinia, M., De Poorter, E. & Shahid, A. RadioPile v1.0: "
    f"A Unified Multi-Task RF IQ Collection. Zenodo, 2026. https://doi.org/{RECORD_A_DOI}"
)

# With no source citations available, fall back to the collection itself.
if not CITATIONS:
    CITATIONS.append(COLLECTION_CITATION)

# Determine the combined license.
#
# The aggregate mixes windows from all source datasets into one artifact, so it
# inherits the *most restrictive* license among its inputs — the same rule
# stated in NOTICE and in release_templates/LICENSES.md.
#
# Higher rank = more restrictive. Extend when a source introduces a new license.
LICENSE_RESTRICTIVENESS = {
    "CC0-1.0":             0,
    "NIST-public-domain":  0,   # no US copyright; NIST terms grant reuse elsewhere
    "CC-BY-4.0":           1,   # attribution
    "CC-BY-SA-4.0":        2,   # + share-alike
    "CC-BY-NC-4.0":        3,   # + non-commercial
    "CC-BY-NC-SA-4.0":     4,   # + non-commercial and share-alike
}

_unranked = [lic for lic in LICENSES if lic not in LICENSE_RESTRICTIVENESS]
if _unranked:
    print(
        f"Warning: unranked source license(s) {_unranked}; falling back to "
        f"NOASSERTION. Add them to LICENSE_RESTRICTIVENESS to resolve."
    )
    LICENSE = "NOASSERTION"
elif LICENSES:
    LICENSE = max(LICENSES, key=lambda lic: LICENSE_RESTRICTIVENESS[lic])
else:
    LICENSE = "NOASSERTION"

GLOBAL_METADATA["license"] = LICENSE
GLOBAL_METADATA["license_note"] = (
    "Inherited: the most restrictive license among the source datasets. Per-dataset files are published separately under their own, less restrictive terms where applicable. See LICENSES.md."
)
