"""
Tests for capture-location resolution in the lte_wifi_dvbt reader.

One upstream file sits in the gentbrugge folder but carries "uz" in its name.
It was recorded at Gentbrugge (see docs/DATASETS.md), so the reader
corrects it through LOCATION_OVERRIDES while every other file keeps the
location parsed from its filename.
"""
from pathlib import Path

from datasets.lte_wifi_dvbt.build_dataset import parse_lte_wifi_dvbt_filename
from datasets.lte_wifi_dvbt.config import LOCATION_OVERRIDES

RAW = Path("data/lte_wifi_dvbt/raw")


class TestLocationOverride:
    def test_misnamed_gentbrugge_capture_is_relabeled(self):
        meta = parse_lte_wifi_dvbt_filename(RAW / "gentbrugge" / "lte10Msps_g30_uz_f806MHz_r1.bin")
        assert meta["location"] == "gentbrugge"

    def test_same_filename_in_uz_folder_stays_uz(self):
        meta = parse_lte_wifi_dvbt_filename(RAW / "uz" / "lte10Msps_g30_uz_f806MHz_r1.bin")
        assert meta["location"] == "uz"

    def test_override_leaves_other_fields_untouched(self):
        meta = parse_lte_wifi_dvbt_filename(RAW / "gentbrugge" / "lte10Msps_g30_uz_f806MHz_r1.bin")
        assert meta["tech_label"] == "lte"
        assert meta["sample_rate_hz"] == 10e6
        assert meta["center_frequency_hz"] == 806e6
        assert meta["gain_db"] == 30.0
        assert meta["recording_id"] == "1"

    def test_ordinary_file_uses_filename_location(self):
        meta = parse_lte_wifi_dvbt_filename(RAW / "rabot" / "wf10Msps_g76_rabot_f5240MHz_r1.bin")
        assert meta["location"] == "rabot"

    def test_unlisted_mismatch_keeps_filename_and_warns(self, capsys):
        meta = parse_lte_wifi_dvbt_filename(RAW / "reep" / "dvbt_g30_rabot_f482MHz_r1.bin")
        assert meta["location"] == "rabot"
        assert "disagrees" in capsys.readouterr().out

    def test_overrides_are_keyed_by_folder_and_filename(self):
        assert all(key.count("/") == 1 for key in LOCATION_OVERRIDES)
