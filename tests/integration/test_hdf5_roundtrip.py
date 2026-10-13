"""
Integration tests: build_hdf5_dataset → load_processed_h5_dataset round-trip.

Uses only synthetic in-memory data written to tmp_path — no real dataset files.
Verifies that data, labels, and metadata survive the write/read cycle intact.
"""

import numpy as np
import pytest

from pipeline.hdf5_writer import build_hdf5_dataset
from pipeline.io import load_processed_h5_dataset
from pipeline.labels import reset_label_registry
from pipeline.schemas import WindowMetaField, H5_STR_DTYPE


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_synthetic_windows(n: int = 10, window_size: int = 500):
    """
    Generate n synthetic (2, window_size) windows with known values.
    Window i has all samples set to float(i+1) for easy identity checking.
    """
    return [np.full((2, window_size), float(i + 1), dtype=np.float32) for i in range(n)]


def synthetic_iterator(windows, labels, metadata_list=None):
    """
    Yield (window, label_name, metadata) tuples as required by build_hdf5_dataset.
    """
    for i, (window, label) in enumerate(zip(windows, labels)):
        meta = metadata_list[i] if metadata_list else {}
        yield window, label, meta


CUSTOM_SCHEMA = [
    WindowMetaField("source_dataset", "source_dataset", H5_STR_DTYPE, "unknown"),
]


# ── Data round-trip ───────────────────────────────────────────────────────────

class TestDataRoundtrip:

    def test_data_shape(self, tmp_path):
        """Written (N, 2, W) should come back as (N, W, 2)."""
        windows = make_synthetic_windows(10, 500)
        labels = ["classA"] * 5 + ["classB"] * 5

        reset_label_registry()
        build_hdf5_dataset(
            output_path=tmp_path / "test.h5",
            window_iterator=synthetic_iterator(windows, labels),
        )

        x, y, label_names, _ = load_processed_h5_dataset(tmp_path / "test.h5")
        assert x.shape == (10, 500, 2)

    def test_data_values_preserved(self, tmp_path):
        """Each window's values should survive the round-trip exactly."""
        windows = make_synthetic_windows(5, 100)
        labels = ["A"] * 5

        reset_label_registry()
        build_hdf5_dataset(
            output_path=tmp_path / "test.h5",
            window_iterator=synthetic_iterator(windows, labels),
        )

        x, _, _, _ = load_processed_h5_dataset(tmp_path / "test.h5")
        for i in range(5):
            # Window i was filled with float(i+1); after transpose shape is (W, 2)
            expected_val = float(i + 1)
            np.testing.assert_array_equal(x[i], expected_val)

    def test_data_dtype_is_float32(self, tmp_path):
        windows = make_synthetic_windows(4, 50)
        labels = ["A"] * 4

        reset_label_registry()
        build_hdf5_dataset(
            output_path=tmp_path / "test.h5",
            window_iterator=synthetic_iterator(windows, labels),
        )

        x, _, _, _ = load_processed_h5_dataset(tmp_path / "test.h5")
        assert x.dtype == np.float32


# ── Label round-trip ──────────────────────────────────────────────────────────

class TestLabelRoundtrip:

    def test_label_names_preserved(self, tmp_path):
        windows = make_synthetic_windows(6, 50)
        labels = ["LTE", "WiFi", "DVB-T", "LTE", "WiFi", "DVB-T"]

        reset_label_registry()
        build_hdf5_dataset(
            output_path=tmp_path / "test.h5",
            window_iterator=synthetic_iterator(windows, labels),
        )

        _, _, label_names, _ = load_processed_h5_dataset(tmp_path / "test.h5")
        assert set(label_names) == {"LTE", "WiFi", "DVB-T"}

    def test_label_indices_contiguous(self, tmp_path):
        """Label indices should be 0-based and contiguous."""
        windows = make_synthetic_windows(6, 50)
        labels = ["A", "B", "C", "A", "B", "C"]

        reset_label_registry()
        build_hdf5_dataset(
            output_path=tmp_path / "test.h5",
            window_iterator=synthetic_iterator(windows, labels),
        )

        _, y, label_names, _ = load_processed_h5_dataset(tmp_path / "test.h5")
        assert set(y.tolist()) == set(range(len(label_names)))

    def test_label_count_matches_windows(self, tmp_path):
        windows = make_synthetic_windows(8, 50)
        labels = ["X"] * 8

        reset_label_registry()
        build_hdf5_dataset(
            output_path=tmp_path / "test.h5",
            window_iterator=synthetic_iterator(windows, labels),
        )

        _, y, _, _ = load_processed_h5_dataset(tmp_path / "test.h5")
        assert len(y) == 8


# ── Metadata round-trip ───────────────────────────────────────────────────────

class TestMetadataRoundtrip:

    def test_string_metadata_preserved(self, tmp_path):
        """String metadata fields should survive encoding/decoding."""
        windows = make_synthetic_windows(3, 50)
        labels = ["A", "A", "A"]
        metadata_list = [
            {"source_dataset": "ds_one"},
            {"source_dataset": "ds_two"},
            {"source_dataset": "ds_one"},
        ]

        reset_label_registry()
        build_hdf5_dataset(
            output_path=tmp_path / "test.h5",
            window_iterator=synthetic_iterator(windows, labels, metadata_list),
            window_metadata_schema=CUSTOM_SCHEMA,
        )

        _, _, _, metadata = load_processed_h5_dataset(tmp_path / "test.h5")
        source_datasets = metadata["source_dataset"]
        decoded = [
            v.decode("utf-8") if isinstance(v, bytes) else str(v)
            for v in source_datasets
        ]
        assert decoded[0] == "ds_one"
        assert decoded[1] == "ds_two"

    def test_window_count_attribute_matches(self, tmp_path):
        """The total_windows HDF5 attribute must match the actual array length."""
        import h5py
        windows = make_synthetic_windows(7, 50)
        labels = ["A"] * 7

        reset_label_registry()
        build_hdf5_dataset(
            output_path=tmp_path / "test.h5",
            window_iterator=synthetic_iterator(windows, labels),
        )

        with h5py.File(tmp_path / "test.h5", "r") as f:
            total_windows_attr = f.attrs.get("total_windows")
            actual_count = f["data"].shape[0]

        assert total_windows_attr == actual_count == 7


# ── Malformed input is rejected, not silently accepted ────────────────────────

class TestMalformedInputRaises:
    """
    load_processed_h5_dataset must not degrade silently on malformed input.
    Passing an unrecognized IQ shape through unchanged, or turning a file with
    no label names into numeric class names, would both produce output with the
    right shape, dtype, and record count but the wrong contents. Both must
    raise instead.
    """

    def _write_raw(self, path, data, labels, label_names=("A", "B")):
        """Write an HDF5 by hand, bypassing build_hdf5_dataset's guarantees."""
        import h5py
        with h5py.File(path, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("labels", data=np.asarray(labels, dtype=np.int64))
            if label_names is not None:
                f.create_dataset("label_names", data=list(label_names),
                                 dtype=H5_STR_DTYPE)

    def test_iq_shape_with_no_axis_of_length_two_raises(self, tmp_path):
        """(N, 3, W) has no identifiable real/imaginary axis."""
        p = tmp_path / "bad_shape.h5"
        self._write_raw(p, np.zeros((4, 3, 50), dtype=np.float32), [0] * 4)

        with pytest.raises(ValueError, match="neither.*trailing axis|cannot be identified"):
            load_processed_h5_dataset(p)

    def test_non_3d_iq_data_raises(self, tmp_path):
        """A 2-D /data array is not windowed IQ."""
        p = tmp_path / "flat.h5"
        self._write_raw(p, np.zeros((4, 50), dtype=np.float32), [0] * 4)

        with pytest.raises(ValueError, match="expected a 3-D array"):
            load_processed_h5_dataset(p)

    def test_missing_label_names_raises(self, tmp_path):
        """Neither /label_names nor meta/label_name present."""
        p = tmp_path / "no_labels.h5"
        self._write_raw(p, np.zeros((4, 2, 50), dtype=np.float32), [0] * 4,
                        label_names=None)

        with pytest.raises(ValueError, match="no label names"):
            load_processed_h5_dataset(p)

    def test_missing_required_dataset_raises(self, tmp_path):
        """A clear error, rather than a bare KeyError from h5py."""
        import h5py
        p = tmp_path / "no_data.h5"
        with h5py.File(p, "w") as f:
            f.create_dataset("labels", data=np.zeros(4, dtype=np.int64))

        with pytest.raises(ValueError, match="missing required dataset"):
            load_processed_h5_dataset(p)

    def test_valid_file_still_loads(self, tmp_path):
        """The guards must not reject well-formed pipeline output."""
        windows = make_synthetic_windows(6, 40)
        reset_label_registry()
        build_hdf5_dataset(
            output_path=tmp_path / "ok.h5",
            window_iterator=synthetic_iterator(windows, ["A", "B"] * 3),
        )

        x, y, label_names, _ = load_processed_h5_dataset(tmp_path / "ok.h5")
        assert x.shape == (6, 40, 2)
        assert sorted(label_names) == ["A", "B"]
