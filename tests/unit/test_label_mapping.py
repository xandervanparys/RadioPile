"""
Tests for datasets/aggregation/build_dataset.py :: extract_classification_metadata

Verifies that technology, modulation, and waveform labels are mapped correctly
for each source dataset, and that unknown/missing labels produce "unknown"
fill values.

These unit tests cover the same routing semantics that the runtime validator
checks (the label-routing checks in `validate_datasets.py`), but at the function
level — so they catch regressions without requiring a full aggregated build.
"""

import pytest

from datasets.aggregation.build_dataset import (
    _load_dataset_config,
    extract_classification_metadata,
    get_primary_label,
)
from datasets.registry import DATASET_REGISTRY


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_tech_entry():
    """
    Real DatasetEntry for lte_wifi_dvbt.
    Valid labels: "DVB-T", "WiFi", "LTE"
    """
    return DATASET_REGISTRY["lte_wifi_dvbt"]


def make_mod_entry():
    """
    Real DatasetEntry for rml2016_10a.
    Valid labels: BPSK, QPSK, 8PSK, PAM4, 16QAM, 64QAM, GFSK, CPFSK, WBFM,
    AM-DSB, AM-SSB
    """
    return DATASET_REGISTRY["rml2016_10a"]


def make_radar_entry():
    """
    Real DatasetEntry for radchar (label_represents=radar_waveform).
    """
    return DATASET_REGISTRY["radchar"]


# ── Technology-recognition datasets ──────────────────────────────────────────

class TestTechnologyLabels:

    def test_lte_maps_to_technology(self):
        entry = make_tech_entry()
        cfg = _load_dataset_config(entry)
        tech, mod, wf, rst = extract_classification_metadata("LTE", {}, entry, cfg)
        assert tech == "LTE"

    def test_wifi_maps_to_technology(self):
        entry = make_tech_entry()
        cfg = _load_dataset_config(entry)
        tech, mod, wf, rst = extract_classification_metadata("WiFi", {}, entry, cfg)
        assert tech == "WiFi"

    def test_dvbt_maps_to_technology(self):
        entry = make_tech_entry()
        cfg = _load_dataset_config(entry)
        tech, mod, wf, rst = extract_classification_metadata("DVB-T", {}, entry, cfg)
        assert tech == "DVB-T"

    def test_unknown_label_gives_unknown_technology(self):
        entry = make_tech_entry()
        cfg = _load_dataset_config(entry)
        tech, mod, wf, rst = extract_classification_metadata("Bluetooth", {}, entry, cfg)
        assert tech == "unknown"

    def test_empty_label_gives_unknown_technology(self):
        entry = make_tech_entry()
        cfg = _load_dataset_config(entry)
        tech, mod, wf, rst = extract_classification_metadata("", {}, entry, cfg)
        assert tech == "unknown"

    def test_modulation_is_unknown_when_not_in_metadata(self):
        """Technology datasets without per-window modulation info → modulation unknown."""
        entry = make_tech_entry()
        cfg = _load_dataset_config(entry)
        _, mod, _, _ = extract_classification_metadata("LTE", {}, entry, cfg)
        assert mod == "unknown"

    def test_modulation_from_metadata_field(self):
        """If per-window metadata contains modulation_type, it should be used."""
        entry = make_tech_entry()
        cfg = _load_dataset_config(entry)
        _, mod, _, _ = extract_classification_metadata(
            "LTE", {"modulation_type": "OFDM"}, entry, cfg
        )
        assert mod == "OFDM"

    def test_waveform_defaults_unknown(self):
        entry = make_tech_entry()
        cfg = _load_dataset_config(entry)
        _, _, wf, _ = extract_classification_metadata("LTE", {}, entry, cfg)
        assert wf == "unknown"


# ── Modulation-recognition datasets ──────────────────────────────────────────

class TestModulationLabels:

    def test_bpsk_maps_to_modulation(self):
        entry = make_mod_entry()
        cfg = _load_dataset_config(entry)
        tech, mod, wf, rst = extract_classification_metadata("BPSK", {}, entry, cfg)
        assert mod == "BPSK"

    def test_qpsk_maps_to_modulation(self):
        entry = make_mod_entry()
        cfg = _load_dataset_config(entry)
        tech, mod, wf, rst = extract_classification_metadata("QPSK", {}, entry, cfg)
        assert mod == "QPSK"

    def test_technology_is_unknown_for_modulation_dataset(self):
        """Pure modulation datasets have no technology info."""
        entry = make_mod_entry()
        cfg = _load_dataset_config(entry)
        tech, _, _, _ = extract_classification_metadata("BPSK", {}, entry, cfg)
        assert tech == "unknown"

    def test_unknown_modulation_gives_unknown(self):
        # "LoRa" is a technology label, not a modulation in rml2016_10a
        entry = make_mod_entry()
        cfg = _load_dataset_config(entry)
        _, mod, _, _ = extract_classification_metadata("LoRa", {}, entry, cfg)
        assert mod == "unknown"


# ── Radar datasets (label_represents=radar_waveform) ─────────────────────────

class TestRadarLabels:

    def test_radar_tech_and_modulation_unknown(self):
        """Radar datasets: technology and modulation both stay 'unknown'."""
        entry = make_radar_entry()
        cfg = _load_dataset_config(entry)
        tech, mod, _, _ = extract_classification_metadata(
            "barker_code", {"waveform_type": "phase_modulated"}, entry, cfg
        )
        assert tech == "unknown"
        assert mod == "unknown"

    def test_radar_signal_type_from_label(self):
        """Fine radar target comes from the primary label, symmetric with tech/mod."""
        entry = make_radar_entry()
        cfg = _load_dataset_config(entry)
        _, _, _, rst = extract_classification_metadata(
            "barker_code", {"waveform_type": "phase_modulated"}, entry, cfg
        )
        assert rst == "barker_code"

    def test_radar_waveform_is_coarse_from_metadata(self):
        """waveform_type (coarse family) comes from per-window metadata, not the label."""
        entry = make_radar_entry()
        cfg = _load_dataset_config(entry)
        _, _, wf, _ = extract_classification_metadata(
            "barker_code", {"waveform_type": "phase_modulated"}, entry, cfg
        )
        assert wf == "phase_modulated"

    def test_unknown_radar_label_gives_unknown_signal_type(self):
        """A label outside RADAR_SIGNAL_TYPES does not populate radar_signal_type."""
        entry = make_radar_entry()
        cfg = _load_dataset_config(entry)
        _, _, _, rst = extract_classification_metadata(
            "not_a_radar_class", {"waveform_type": "unknown"}, entry, cfg
        )
        assert rst == "unknown"


# ── get_primary_label ─────────────────────────────────────────────────────────

class TestGetPrimaryLabel:

    def test_technology_label_type(self):
        assert get_primary_label("LTE", "OFDM", "unknown", "technology") == "LTE"

    def test_modulation_label_type(self):
        assert get_primary_label("LTE", "OFDM", "unknown", "modulation") == "OFDM"

    def test_waveform_label_type(self):
        assert get_primary_label(
            "unknown", "unknown", "barker_code", "waveform"
        ) == "barker_code"

    def test_unknown_label_type_raises(self):
        with pytest.raises(ValueError, match="Unknown label_type"):
            get_primary_label("LTE", "OFDM", "unknown", "invalid")


# ── Aggregation loader rejects unlabeled input ───────────────────────────────

class TestAggregationLoaderRequiresLabelNames:
    """
    load_dataset_h5 feeds the aggregation build directly. If it fell back to
    numeric names, a per-dataset HDF5 missing /label_names would contribute
    windows whose class names were the strings "0", "1", "2": a corrupt
    aggregate that still passes every shape, dtype, and count check. It must
    raise instead.
    """

    def _write(self, path, label_names):
        import h5py
        import numpy as np
        from pipeline.schemas import H5_STR_DTYPE
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=np.zeros((3, 2, 20), dtype=np.float32))
            f.create_dataset("labels", data=np.array([0, 1, 2], dtype=np.int64))
            if label_names is not None:
                f.create_dataset("label_names", data=list(label_names),
                                 dtype=H5_STR_DTYPE)

    def test_missing_label_names_raises(self, tmp_path):
        from datasets.aggregation.build_dataset import load_dataset_h5

        p = tmp_path / "unlabeled.h5"
        self._write(p, None)

        with pytest.raises(ValueError, match="no /label_names"):
            list(load_dataset_h5(p))

    def test_label_id_beyond_label_names_raises(self, tmp_path):
        """A truncated label map must not IndexError opaquely mid-build."""
        from datasets.aggregation.build_dataset import load_dataset_h5

        p = tmp_path / "short_map.h5"
        self._write(p, ["only_one"])

        with pytest.raises(ValueError, match="holds only 1 entries"):
            list(load_dataset_h5(p))

    def test_well_formed_file_still_loads(self, tmp_path):
        from datasets.aggregation.build_dataset import load_dataset_h5

        p = tmp_path / "ok.h5"
        self._write(p, ["a", "b", "c"])

        got = [label_name for _, label_name, _ in load_dataset_h5(p)]
        assert got == ["a", "b", "c"]
