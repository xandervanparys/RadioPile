"""
Tests for the source-specific mappings that decide a window's label.

Each of these is a place where a wrong value would not fail a build: the file
would have the right shape and the right number of windows, carrying the wrong
class or the wrong modulation. docs/DATASETS.md describes each one.
"""
import pickle
from pathlib import Path

import numpy as np
import pytest

from datasets.nist_wifi_bluetooth import config as nist
from datasets.rml2016_10a import config as rml2016
from datasets.rml2016_10a.build_dataset import (
    iter_rml2016_10a_windows,
    load_rml2016_10a_data,
)
from datasets.rml2018_01a import config as rml2018
from datasets.sub_ghz import config as sub_ghz
from datasets.sub_ghz.build_dataset import (
    MCS_TO_MODULATION,
    infer_label_from_path,
    parse_metadata_filename,
)


# ── sub_ghz ───────────────────────────────────────────────────────────────────

SUB_GHZ_RAW = Path("data/sub_ghz/raw")

# (raw label, filename, modulation_type, waveform_type)
SUB_GHZ_FILES = [
    ("sigfox",  "sigfox_g0.0dB_att25dB_freq867.4MHz_0.mat",             "BPSK",    "PSK"),
    ("lora",    "lorasf7_g0.0dB_att12dB_freq867.4MHz_0.mat",            "CSS",     "CSS"),
    ("80211ah", "80211ah_mcs0_chan1_g0.0dB_att10dB_freq864.0MHz_0.mat", "BPSK",    "OFDM"),
    ("80211ah", "80211ah_mcs7_chan3_g0.0dB_att10dB_freq864.0MHz_1.mat", "64QAM",   "OFDM"),
    ("80211ah", "80211ah_mcs10_chan6_g0.0dB_att3dB_freq867.4MHz_2.mat", "BPSK",    "OFDM"),
    ("802154g", "802154g_g0.0dB_att20dB_freq867.4MHz_0.mat",            "BFSK",    "FSK"),
    ("sunofdm", "sunofdm_mcs2_chan1_g0.0dB_att22dB_freq867.4MHz_0.mat", "OQPSK",   "OFDM"),
    ("sunofdm", "sunofdm_mcs6_chan2_g0.0dB_att23dB_freq867.4MHz_0.mat", "16QAM",   "OFDM"),
    ("noise",   "noise_g0.0dB_att10dB_freq867.4MHz_0.mat",              "unknown", "noise"),
]


class TestSubGhzModulation:

    @pytest.mark.parametrize("label,filename,modulation,waveform", SUB_GHZ_FILES)
    def test_modulation_and_waveform(self, label, filename, modulation, waveform):
        meta = parse_metadata_filename(SUB_GHZ_RAW / filename, label)
        assert meta["modulation_type"] == modulation
        assert meta["waveform_type"] == waveform

    def test_mcs_table_covers_exactly_the_values_in_the_source(self):
        """
        One flat table serves 802.11ah (MCS 0, 7, 10) and SUN-OFDM (MCS 2, 6).
        That is only safe while the two sets stay disjoint, so a new key here
        needs a look at which technology it belongs to.
        """
        assert set(MCS_TO_MODULATION) == {0, 2, 6, 7, 10}

    def test_unlisted_mcs_is_unknown_rather_than_guessed(self):
        meta = parse_metadata_filename(
            SUB_GHZ_RAW / "80211ah_mcs3_chan1_g0.0dB_att10dB_freq864.0MHz_0.mat", "80211ah"
        )
        assert meta["modulation_type"] == "unknown"


class TestSubGhzFilenameFields:

    def test_all_tokens_parsed(self):
        meta = parse_metadata_filename(
            SUB_GHZ_RAW / "80211ah_mcs7_chan3_g0.0dB_att10dB_freq864.0MHz_4.mat", "80211ah"
        )
        assert meta["mcs"] == 7
        assert meta["channel"] == 3
        assert meta["gain_db"] == 0.0
        assert meta["attenuation_db"] == 10.0
        assert meta["center_frequency_hz"] == pytest.approx(864.0e6)
        assert meta["recording_id"] == "4"

    def test_absent_tokens_are_absent_not_defaulted(self):
        meta = parse_metadata_filename(
            SUB_GHZ_RAW / "sigfox_g0.0dB_att25dB_freq867.4MHz_0.mat", "sigfox"
        )
        assert "mcs" not in meta
        assert "channel" not in meta
        assert "spreading_factor" not in meta

    @pytest.mark.parametrize("sf", sorted(sub_ghz.LORA_SPREADING_FACTORS))
    def test_lora_spreading_factor(self, sf):
        meta = parse_metadata_filename(
            SUB_GHZ_RAW / f"lorasf{sf}_g0.0dB_att12dB_freq867.4MHz_0.mat", "lora"
        )
        assert meta["spreading_factor"] == sf


class TestSubGhzLabels:

    def test_six_raw_labels_map_to_five_classes(self):
        assert len(sub_ghz.LABEL_REMAP) == 6
        assert set(sub_ghz.LABEL_REMAP.values()) == {
            "WiFi", "IEEE 802.15.4", "LoRa", "Noise", "Sigfox",
        }

    def test_merged_labels_keep_their_variant(self):
        """The variant field is the only way to tell merged technologies apart."""
        assert sub_ghz.WIFI_STANDARD == {"80211ah": "802.11ah"}
        assert sub_ghz.IEEE802154_VARIANT == {"802154g": "g", "sunofdm": "SUN-OFDM"}

    @pytest.mark.parametrize("relative,label", [
        ("864/80211ah/80211ah_mcs0_chan1_g0.0dB_att10dB_freq864.0MHz_0.mat", "80211ah"),
        ("867.4/Lora/lorasf7_g0.0dB_att12dB_freq867.4MHz_0.mat",             "lora"),
        ("867.4/Noise/noise_g0.0dB_att10dB_freq867.4MHz_0.mat",              "noise"),
        ("867.4/Sigfox/sigfox_g0.0dB_att25dB_freq867.4MHz_0.mat",            "sigfox"),
    ])
    def test_label_inferred_from_directory(self, relative, label):
        assert infer_label_from_path(SUB_GHZ_RAW / relative) == label


# ── rml2018_01a ───────────────────────────────────────────────────────────────

# raw/classes-fixed.txt. The order distributed with the dataset starts
# "32PSK", "16APSK", "32QAM" and mislabels every class.
RML2018_CORRECTED_ORDER = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]


class TestRml2018ClassOrder:

    def test_one_hot_index_order_is_the_corrected_one(self):
        """Index i of the one-hot Y vector is MODULATION_TYPES[i]."""
        assert rml2018.MODULATION_TYPES == RML2018_CORRECTED_ORDER

    def test_every_class_has_a_waveform_family(self):
        assert set(rml2018.MODULATION_TO_WAVEFORM) == set(rml2018.MODULATION_TYPES)


# ── rml2016_10a ───────────────────────────────────────────────────────────────

def _write_py2_style_pickle(path: Path, entries: dict) -> None:
    """The upstream pickle holds byte-string keys; reproduce that."""
    with open(path, "wb") as f:
        pickle.dump(entries, f, protocol=2)


def _examples(n: int = 3) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.standard_normal((n, 2, rml2016.WINDOW_SIZE)).astype(np.float32)


class TestRml2016Pickle:

    def test_byte_string_keys_are_decoded(self, tmp_path):
        path = tmp_path / "rml.pkl"
        _write_py2_style_pickle(path, {(b"BPSK", -4): _examples(), (b"QAM16", 2): _examples()})

        assert set(load_rml2016_10a_data(path)) == {("BPSK", -4), ("QAM16", 2)}

    @pytest.mark.parametrize("upstream,canonical", [
        (b"QAM16", "16QAM"), (b"QAM64", "64QAM"), (b"AM-DSB", "AM-DSB"), (b"8PSK", "8PSK"),
    ])
    def test_upstream_spelling_maps_to_canonical_class(self, tmp_path, upstream, canonical):
        path = tmp_path / "rml.pkl"
        _write_py2_style_pickle(path, {(upstream, 6): _examples()})

        labels = {label for _, label, _ in iter_rml2016_10a_windows(path)}
        assert labels == {canonical}

    def test_snr_and_position_carried_per_window(self, tmp_path):
        path = tmp_path / "rml.pkl"
        _write_py2_style_pickle(path, {(b"BPSK", -18): _examples(3)})

        metas = [meta for _, _, meta in iter_rml2016_10a_windows(path)]
        assert [m["snr_db"] for m in metas] == [-18.0] * 3
        assert [m["sample_index"] for m in metas] == [0, 1, 2]

    def test_eleven_classes(self):
        assert len(rml2016.MODULATION_TYPES) == 11
        assert set(rml2016.LABEL_REMAP.values()) == set(rml2016.MODULATION_TYPES)


# ── nist_wifi_bluetooth ───────────────────────────────────────────────────────

class TestNistFileConstants:
    """Modulation and standard are per-file constants from NIST TN 2237."""

    @pytest.mark.parametrize("stem,frequency_hz,environment,standard,modulation", [
        ("2_4ghz_indoor", 2.437e9, "indoor",  "802.11n",  "64QAM"),
        ("5ghz_indoor",   5.825e9, "indoor",  "802.11ac", "256QAM"),
        ("5ghz_outdoor",  5.825e9, "outdoor", "802.11ac", "256QAM"),
    ])
    def test_file_constants(self, stem, frequency_hz, environment, standard, modulation):
        assert nist.FILE_CONFIG[stem] == (frequency_hz, environment, "WiFi")
        assert nist.FILE_METADATA[stem] == {
            "modulation_type": modulation,
            "wifi_standard": standard,
        }

    def test_only_the_three_wifi_only_files_are_built(self):
        """The outdoor 2.4 GHz and Bluetooth files contain mixed signals."""
        assert set(nist.FILE_CONFIG) == set(nist.FILE_METADATA) == {
            "2_4ghz_indoor", "5ghz_indoor", "5ghz_outdoor",
        }
