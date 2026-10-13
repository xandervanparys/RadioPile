"""
Tests for metadata schema correctness.

Verifies that:
- unify_metadata produces all required schema fields
- fill values match the project convention ("unknown" for strings, False for bools)
- source_dataset and classification fields are set from arguments
"""

import pytest

from datasets.aggregation.build_dataset import unify_metadata
from datasets.aggregation.config import WINDOW_METADATA_SCHEMA
from datasets.registry import DatasetEntry
from pathlib import Path


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_registry_entry(task_type: str = "technology_recognition") -> DatasetEntry:
    """Minimal registry entry with no filesystem dependencies."""
    return DatasetEntry(
        name="test_dataset",
        processed_h5_path=Path("/dev/null"),
        build_module=None,
        task_type=task_type,
        label_represents="technology",
    )


def call_unify_minimal(source_dataset="test_ds", source_label="LTE",
                       technology="LTE", modulation="unknown",
                       waveform="unknown", radar_signal="unknown"):
    """Call unify_metadata with minimal inputs."""
    entry = make_registry_entry()
    return unify_metadata(
        # original_meta carries source-side fields that unify_metadata maps
        # into the optional RF schema (sample_rate_hz, center_frequency_hz,
        # snr_db). NaN values are filtered out by unify_metadata, so
        # these must be concrete numbers for the optional fields to land in
        # the output.
        original_meta={
            "sample_rate_hz":      1.0e6,
            "center_frequency_hz": 2.4e9,
            "snr_db":              10.0,
        },
        source_dataset=source_dataset,
        source_label=source_label,
        technology=technology,
        modulation=modulation,
        waveform=waveform,
        window_meta={
            "was_padded": False,
            "padding_strategy": "none",
            "was_cropped": False,
            "crop_policy": "none",
            "original_window_size": 500,
        },
        normalize_mode="unit_power",
        registry_entry=entry,
        radar_signal=radar_signal,
    )


# ── Required fields are present ───────────────────────────────────────────────

class TestRequiredFieldsPresent:

    def test_all_schema_keys_present(self):
        """Every field defined in WINDOW_METADATA_SCHEMA must appear in output."""
        result = call_unify_minimal()
        for field in WINDOW_METADATA_SCHEMA:
            assert field.key in result, (
                f"Required field '{field.key}' missing from unify_metadata output"
            )

    def test_source_dataset_set(self):
        result = call_unify_minimal(source_dataset="my_dataset")
        assert result["source_dataset"] == "my_dataset"

    def test_source_label_set(self):
        result = call_unify_minimal(source_label="WiFi")
        assert result["source_label"] == "WiFi"

    def test_technology_type_set(self):
        result = call_unify_minimal(technology="DVB-T")
        assert result["technology_type"] == "DVB-T"

    def test_modulation_type_set(self):
        result = call_unify_minimal(modulation="OFDM")
        assert result["modulation_type"] == "OFDM"

    def test_radar_signal_type_set(self):
        result = call_unify_minimal(radar_signal="barker_code")
        assert result["radar_signal_type"] == "barker_code"

    def test_radar_signal_type_defaults_unknown(self):
        result = call_unify_minimal()
        assert result["radar_signal_type"] == "unknown"

    def test_task_type_from_registry(self):
        result = call_unify_minimal()
        assert result["task_type"] == "technology_recognition"


# ── Fill value conventions ────────────────────────────────────────────────────

class TestFillValues:

    def test_missing_technology_fills_unknown(self):
        """technology="unknown" should propagate as "unknown", not empty string."""
        result = call_unify_minimal(technology="unknown")
        assert result["technology_type"] == "unknown"
        assert result["technology_type"] != ""
        assert result["technology_type"] is not None

    def test_missing_modulation_fills_unknown(self):
        result = call_unify_minimal(modulation="unknown")
        assert result["modulation_type"] == "unknown"
        assert result["modulation_type"] != ""
        assert result["modulation_type"] is not None

    def test_was_padded_bool_type(self):
        """was_padded should always be a boolean, not a string."""
        result = call_unify_minimal()
        assert isinstance(result["was_padded"], bool)

    def test_padding_strategy_default(self):
        result = call_unify_minimal()
        assert result["padding_strategy"] == "none"


# ── No extra junk in output ───────────────────────────────────────────────────

class TestOutputCompleteness:

    def test_no_none_values_for_schema_fields(self):
        """Schema fields must never be None — use fill values instead."""
        result = call_unify_minimal()
        for field in WINDOW_METADATA_SCHEMA:
            assert result[field.key] is not None, (
                f"Field '{field.key}' is None — should use fill value '{field.fill_value}'"
            )
