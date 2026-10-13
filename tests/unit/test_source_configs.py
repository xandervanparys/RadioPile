"""
Tests for the conventions every source config in datasets/<name>/config.py shares.

Two things are read from these configs by name, and nothing else checks them:

- the HDF5 root attributes a holder of a per-dataset file relies on
  (`dataset_url`, `sample_rate_hz`, `doi`, `license`), which come from GLOBAL_METADATA;
- the class list the aggregation reads to route a source's primary label
  (LABEL_REMAP values, MODULATION_TYPES or RADAR_SIGNAL_TYPES, depending on
  the registry's `label_represents`).

A config that drops or renames one of them still imports and still builds, so
the only symptom would be a missing attribute or a class routed to "unknown" in
a finished corpus. These tests import config modules only and need no data.
"""

import pytest

from datasets.aggregation.build_dataset import (
    _load_dataset_config,
    extract_classification_metadata,
)
from datasets.identifiers import CODE_DOI, RECORD_A_DOI, RECORD_B_DOI
from datasets.registry import DATASET_REGISTRY


SOURCES = sorted(DATASET_REGISTRY.items())
SOURCE_NAMES = [name for name, _ in SOURCES]

# Sources recorded at one sample rate. lte_wifi_dvbt mixes 1 and 10 Msps and
# stores the rate per window; the RadioML sources are simulated.
FIXED_RATE = {"sub_ghz", "its_band", "nist_wifi_bluetooth", "radchar"}

CLASS_LIST_CONSTANT = {
    "technology": "LABEL_REMAP",
    "modulation": "MODULATION_TYPES",
    "radar_waveform": "RADAR_SIGNAL_TYPES",
}


def class_list(entry, config):
    constant = getattr(config, CLASS_LIST_CONSTANT[entry.label_represents])
    return sorted(set(constant.values())) if isinstance(constant, dict) else list(constant)


# ── Root attributes ───────────────────────────────────────────────────────────

class TestSharedRootAttributes:

    @pytest.mark.parametrize("name,entry", SOURCES, ids=SOURCE_NAMES)
    def test_dataset_url(self, name, entry):
        config = _load_dataset_config(entry)
        assert config.DATASET_URL.startswith("https://")
        assert config.GLOBAL_METADATA["dataset_url"] == config.DATASET_URL

    @pytest.mark.parametrize("name,entry", SOURCES, ids=SOURCE_NAMES)
    def test_record_identifiers(self, name, entry):
        """A per-dataset file points at its own record and names its siblings."""
        config = _load_dataset_config(entry)
        assert config.GLOBAL_METADATA["doi"] == RECORD_B_DOI
        related = config.GLOBAL_METADATA["related_identifiers"]
        assert RECORD_A_DOI in related and CODE_DOI in related

    @pytest.mark.parametrize("name,entry", SOURCES, ids=SOURCE_NAMES)
    def test_license(self, name, entry):
        config = _load_dataset_config(entry)
        assert config.GLOBAL_METADATA["license"] == config.LICENSE
        assert config.GLOBAL_METADATA["license_note"]

    @pytest.mark.parametrize("name,entry", SOURCES, ids=SOURCE_NAMES)
    def test_sample_rate_only_where_fixed(self, name, entry):
        config = _load_dataset_config(entry)
        if name in FIXED_RATE:
            assert config.GLOBAL_METADATA["sample_rate_hz"] == config.SAMPLE_RATE_HZ
            assert config.SAMPLE_RATE_HZ > 0
        else:
            assert "sample_rate_hz" not in config.GLOBAL_METADATA

    @pytest.mark.parametrize("name,entry", SOURCES, ids=SOURCE_NAMES)
    def test_no_second_name_for_a_shared_attribute(self, name, entry):
        """One concept, one attribute name, in every per-dataset file."""
        config = _load_dataset_config(entry)
        other_spellings = {"sampling_rate_hz", "dataset_source", "source_dataset"}
        assert not other_spellings & set(config.GLOBAL_METADATA)


# ── Label routing ─────────────────────────────────────────────────────────────

class TestEverySourceRoutesItsClasses:

    @pytest.mark.parametrize("name,entry", SOURCES, ids=SOURCE_NAMES)
    def test_class_list_defined(self, name, entry):
        config = _load_dataset_config(entry)
        assert class_list(entry, config)

    @pytest.mark.parametrize("name,entry", SOURCES, ids=SOURCE_NAMES)
    def test_each_class_lands_in_its_own_field(self, name, entry):
        config = _load_dataset_config(entry)
        for label in class_list(entry, config):
            technology, modulation, _, radar_signal = extract_classification_metadata(
                label, {}, entry, config
            )
            routed = {
                "technology": technology,
                "modulation": modulation,
                "radar_waveform": radar_signal,
            }
            for field, value in routed.items():
                expected = label if field == entry.label_represents else "unknown"
                assert value == expected, f"{name}: {label!r} gave {field}={value!r}"

    @pytest.mark.parametrize("name,entry", SOURCES, ids=SOURCE_NAMES)
    def test_label_outside_the_class_list_is_not_routed(self, name, entry):
        config = _load_dataset_config(entry)
        technology, modulation, _, radar_signal = extract_classification_metadata(
            "not-a-class", {}, entry, config
        )
        assert (technology, modulation, radar_signal) == ("unknown", "unknown", "unknown")
