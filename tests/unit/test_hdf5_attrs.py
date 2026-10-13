"""
Tests for pipeline/hdf5_writer.py coercion internals.

_sanitize_hdf5_attr guards global attribute writing; _coerce_window_meta_value
applies the per-window fill-value conventions. Both sit between a dataset's
config and what actually lands on disk, so a regression in either silently
changes the contents of every released HDF5 rather than failing the build.

Synthetic data only — no dataset files required.
"""

from pathlib import Path

import numpy as np
import pytest

from pipeline.hdf5_writer import _coerce_window_meta_value, _sanitize_hdf5_attr
from pipeline.schemas import H5_STR_DTYPE, WindowMetaField


SOURCE = Path("/data/example/raw/capture_001.bin")


def coerce(field: WindowMetaField, meta: dict, source: Path = SOURCE):
    return _coerce_window_meta_value(field, meta, source)


# ── _sanitize_hdf5_attr ───────────────────────────────────────────────────────

class TestSanitizeScalars:

    def test_none_becomes_sentinel_string(self):
        """HDF5 attrs cannot hold None, so it is encoded rather than dropped."""
        assert _sanitize_hdf5_attr(None) == "__NONE__"

    @pytest.mark.parametrize("value", ["text", 42, 3.5, True, False, 0, ""])
    def test_supported_scalars_pass_through_unchanged(self, value):
        result = _sanitize_hdf5_attr(value)
        assert result == value
        assert type(result) is type(value)

    @pytest.mark.parametrize(
        "value", [np.int32(7), np.int64(7), np.float32(1.5), np.float64(1.5)]
    )
    def test_numpy_scalars_pass_through(self, value):
        assert _sanitize_hdf5_attr(value) == value


class TestSanitizeContainers:

    def test_list_is_comma_joined(self):
        assert _sanitize_hdf5_attr(["LTE", "WiFi", "DVB-T"]) == "LTE, WiFi, DVB-T"

    def test_tuple_is_comma_joined(self):
        assert _sanitize_hdf5_attr((1, 2, 3)) == "1, 2, 3"

    def test_empty_list_becomes_empty_string(self):
        assert _sanitize_hdf5_attr([]) == ""

    def test_dict_is_semicolon_joined_key_value(self):
        result = _sanitize_hdf5_attr({"percentile": 1.0, "margin_db": 6.0})
        assert result == "percentile=1.0; margin_db=6.0"

    def test_empty_dict_becomes_empty_string(self):
        assert _sanitize_hdf5_attr({}) == ""

    def test_nested_container_is_stringified_not_recursed(self):
        """Nesting is flattened by str(), not descended into."""
        assert _sanitize_hdf5_attr([{"a": 1}]) == "{'a': 1}"


class TestSanitizeFallback:

    def test_path_is_stringified(self):
        assert _sanitize_hdf5_attr(Path("data/x/raw")) == "data/x/raw"

    def test_arbitrary_object_is_stringified(self):
        class Thing:
            def __str__(self):
                return "a-thing"

        assert _sanitize_hdf5_attr(Thing()) == "a-thing"

    def test_result_is_always_hdf5_writable(self):
        """Every branch must yield a str or a scalar h5py can store."""
        for value in [None, "s", 1, 1.5, True, np.int64(2), [1, 2],
                      (1, 2), {"k": "v"}, Path("p")]:
            assert isinstance(_sanitize_hdf5_attr(value), (str, int, float, bool, np.number))


# ── _coerce_window_meta_value: the source_file special case ───────────────────

class TestSourceFileFallback:

    FIELD = WindowMetaField("source_file", "file_path", H5_STR_DTYPE, "")

    def test_missing_source_file_falls_back_to_current_path(self):
        """
        source_file is the provenance field; when the iterator omits it the
        writer substitutes the file being read rather than the empty fill.
        """
        assert coerce(self.FIELD, {}) == str(SOURCE)

    def test_provided_source_file_wins(self):
        assert coerce(self.FIELD, {"source_file": "other.bin"}) == "other.bin"

    def test_explicit_none_falls_back_to_fill_not_the_path(self):
        """
        An explicit None is not the same as absence: absence means "the writer
        knows better", None means the iterator deliberately had nothing.
        """
        assert coerce(self.FIELD, {"source_file": None}) == ""


# ── Fill-value conventions, one per dtype family ──────────────────────────────

class TestStringFillConventions:

    UNKNOWN = WindowMetaField("technology_type", "technology_type", H5_STR_DTYPE, "unknown")
    EMPTY = WindowMetaField("location", "location", H5_STR_DTYPE, "")

    def test_missing_key_fills_unknown(self):
        assert coerce(self.UNKNOWN, {}) == "unknown"

    def test_none_value_fills_unknown(self):
        assert coerce(self.UNKNOWN, {"technology_type": None}) == "unknown"

    def test_present_value_preserved(self):
        assert coerce(self.UNKNOWN, {"technology_type": "LTE"}) == "LTE"

    def test_missing_key_fills_empty_string(self):
        assert coerce(self.EMPTY, {}) == ""

    def test_none_value_fills_empty_string(self):
        assert coerce(self.EMPTY, {"location": None}) == ""

    def test_non_string_value_is_stringified(self):
        assert coerce(self.EMPTY, {"location": 42}) == "42"

    def test_result_type_is_str(self):
        assert isinstance(coerce(self.UNKNOWN, {"technology_type": "LTE"}), str)


class TestFloatFillConventions:

    FIELD = WindowMetaField("center_frequency_hz", "center_frequency_hz", "float64", np.nan)

    def test_missing_key_fills_nan(self):
        assert np.isnan(coerce(self.FIELD, {}))

    def test_none_value_fills_nan(self):
        assert np.isnan(coerce(self.FIELD, {"center_frequency_hz": None}))

    def test_present_value_preserved(self):
        assert coerce(self.FIELD, {"center_frequency_hz": 8.06e8}) == pytest.approx(8.06e8)

    def test_numeric_string_is_parsed(self):
        assert coerce(self.FIELD, {"center_frequency_hz": "806000000"}) == pytest.approx(8.06e8)

    def test_uncoercible_value_falls_back_to_fill(self):
        """A garbage value must not crash the build; it degrades to the fill."""
        assert np.isnan(coerce(self.FIELD, {"center_frequency_hz": "not-a-number"}))

    def test_result_type_is_float(self):
        assert isinstance(coerce(self.FIELD, {"center_frequency_hz": 1}), float)

    def test_float32_family_behaves_the_same(self):
        field = WindowMetaField("gain_db", "gain_db", "float32", np.nan)
        assert coerce(field, {"gain_db": "30"}) == pytest.approx(30.0)
        assert np.isnan(coerce(field, {}))


class TestIntFillConventions:

    FIELD = WindowMetaField("start_idx", "start_idx", "int64", -1)

    def test_missing_key_fills_sentinel(self):
        assert coerce(self.FIELD, {}) == -1

    def test_none_value_fills_sentinel(self):
        assert coerce(self.FIELD, {"start_idx": None}) == -1

    def test_present_value_preserved(self):
        assert coerce(self.FIELD, {"start_idx": 5000}) == 5000

    def test_numeric_string_is_parsed(self):
        assert coerce(self.FIELD, {"start_idx": "5000"}) == 5000

    def test_uncoercible_value_falls_back_to_sentinel(self):
        assert coerce(self.FIELD, {"start_idx": "abc"}) == -1

    def test_float_value_truncates_toward_zero(self):
        assert coerce(self.FIELD, {"start_idx": 7.9}) == 7

    def test_result_type_is_int(self):
        assert isinstance(coerce(self.FIELD, {"start_idx": 3}), int)

    def test_int32_family_behaves_the_same(self):
        field = WindowMetaField("recording", "recording", "int32", -1)
        assert coerce(field, {}) == -1
        assert coerce(field, {"recording": 12}) == 12


class TestBoolFillConventions:

    FIELD = WindowMetaField("padded", "padded", "bool", False)

    def test_missing_key_fills_false(self):
        assert coerce(self.FIELD, {}) is False

    def test_none_value_fills_false(self):
        assert coerce(self.FIELD, {"padded": None}) is False

    def test_true_preserved(self):
        assert coerce(self.FIELD, {"padded": True}) is True

    def test_result_type_is_bool(self):
        assert isinstance(coerce(self.FIELD, {"padded": True}), bool)

    @pytest.mark.parametrize(
        "value,expected",
        [(1, True), (0, False), ("", False), ("no", True), ([], False), ([0], True)],
    )
    def test_python_truthiness_applies(self, value, expected):
        """
        Coercion is bool(v), so the usual truthiness rules hold — note that the
        non-empty string "no" is True. Datasets should yield real booleans.
        """
        assert coerce(self.FIELD, {"padded": value}) is expected


class TestSchemaRoundTrip:

    def test_every_fill_convention_survives_an_empty_meta_dict(self):
        """
        A window whose iterator supplied no metadata at all must still produce
        one well-typed value per schema field — this is what keeps /meta/*
        rectangular across a whole dataset.
        """
        schema = [
            WindowMetaField("technology_type", "technology_type", H5_STR_DTYPE, "unknown"),
            WindowMetaField("location", "location", H5_STR_DTYPE, ""),
            WindowMetaField("center_frequency_hz", "center_frequency_hz", "float64", np.nan),
            WindowMetaField("start_idx", "start_idx", "int64", -1),
            WindowMetaField("padded", "padded", "bool", False),
        ]
        values = [coerce(field, {}) for field in schema]

        assert values[0] == "unknown"
        assert values[1] == ""
        assert np.isnan(values[2])
        assert values[3] == -1
        assert values[4] is False


# ── Enum metadata must serialize as its value, not its repr ───────────────────

class TestEnumMetadataSerialization:
    """
    A str-mixin Enum inherits Enum.__str__, so str(member) returns the repr
    ("TaskType.TECHNOLOGY_RECOGNITION") and not the value
    ("technology_recognition"). The value survives comparison but not
    stringification, and _coerce_window_meta_value stringifies on the way to disk.
    Written as the repr, the documented filter (task_type == "technology_recognition")
    would match nothing in a built corpus.
    """

    def test_task_type_stringifies_to_value(self):
        from pipeline.schemas import TaskType
        for member in TaskType:
            assert str(member) == member.value, f"{member!r} stringifies to {str(member)!r}"
            assert not str(member).startswith("TaskType."), f"{member!r} leaks its repr"

    def test_label_represents_stringifies_to_value(self):
        from pipeline.schemas import LabelRepresents
        for member in LabelRepresents:
            assert str(member) == member.value
            assert not str(member).startswith("LabelRepresents.")

    def test_enum_comparison_still_works(self):
        """The str mixin must keep value equality — routing code relies on it."""
        from pipeline.schemas import TaskType, LabelRepresents
        assert TaskType.TECHNOLOGY_RECOGNITION == "technology_recognition"
        assert TaskType.RADAR_CLASSIFICATION == "radar_classification"
        assert LabelRepresents.MODULATION == "modulation"

    def test_coerce_unwraps_enum_for_string_field(self):
        """The HDF5 boundary must not depend on how the enum declares __str__."""
        from pipeline.hdf5_writer import _coerce_window_meta_value
        from pipeline.schemas import WindowMetaField, H5_STR_DTYPE, TaskType
        from pathlib import Path

        field = WindowMetaField("task_type", "task_type", H5_STR_DTYPE, "unknown")
        got = _coerce_window_meta_value(
            field, {"task_type": TaskType.TECHNOLOGY_RECOGNITION}, Path("x"))
        assert got == "technology_recognition"
        assert isinstance(got, str)

    def test_enum_survives_a_real_hdf5_roundtrip(self, tmp_path):
        """End to end: what lands on disk must be the value."""
        import h5py
        import numpy as np
        from pipeline.hdf5_writer import build_hdf5_dataset
        from pipeline.labels import reset_label_registry
        from pipeline.schemas import WindowMetaField, H5_STR_DTYPE, TaskType

        schema = [WindowMetaField("task_type", "task_type", H5_STR_DTYPE, "unknown")]
        windows = [np.zeros((2, 16), dtype=np.float32) for _ in range(4)]

        def it():
            for w in windows:
                yield w, "A", {"task_type": TaskType.RADAR_CLASSIFICATION}

        reset_label_registry()
        build_hdf5_dataset(
            output_path=tmp_path / "e.h5",
            window_iterator=it(),
            window_metadata_schema=schema,
        )
        with h5py.File(tmp_path / "e.h5", "r") as f:
            vals = set(f["meta/task_type"].asstr()[:])
        assert vals == {"radar_classification"}, f"on-disk values were {vals}"
