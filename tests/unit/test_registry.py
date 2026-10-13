"""
Tests for datasets/registry.py.

The registry is the only authoritative list of buildable datasets: run.py, the
validator, and the aggregation pipeline all resolve through it. A registry entry
that points at a module that no longer imports, or at a path its own config
disagrees with, breaks the build in a way no per-dataset test would catch.

These tests import each dataset's config module but never touch raw or processed
data, so they run without any dataset files present.
"""

import importlib
from pathlib import Path

import pytest

from datasets.registry import (
    DATASET_REGISTRY,
    DatasetEntry,
    get_dataset,
    list_datasets,
)
from pipeline.schemas import LabelRepresents, TaskType


REGISTERED = sorted(DATASET_REGISTRY.items())
REGISTERED_NAMES = [name for name, _ in REGISTERED]


# ── Structural invariants over every entry ────────────────────────────────────

class TestRegistryEntries:

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_key_matches_entry_name(self, name, entry):
        """A mismatch here makes lookups and error messages disagree."""
        assert entry.name == name

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_build_module_declared(self, name, entry):
        assert entry.build_module, f"{name} has no build_module"

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_build_module_imports(self, name, entry):
        importlib.import_module(entry.build_module)

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_build_dataset_module_exposes_main(self, name, entry):
        """run.py calls module.main() — every registered dataset must have one."""
        module = importlib.import_module(f"{entry.build_module}.build_dataset")
        assert callable(getattr(module, "main", None))

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_config_module_imports(self, name, entry):
        importlib.import_module(f"{entry.build_module}.config")

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_processed_path_matches_config_output(self, name, entry):
        """
        The registry path and the config's OUTPUT_H5_PATH must name the same
        file. If they drift, the build writes one path and the validator and
        aggregation read another.
        """
        config = importlib.import_module(f"{entry.build_module}.config")
        assert Path(config.OUTPUT_H5_PATH) == Path(entry.processed_h5_path)

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_task_type_is_enum_member(self, name, entry):
        assert isinstance(entry.task_type, TaskType)

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_label_represents_is_enum_member(self, name, entry):
        assert isinstance(entry.label_represents, LabelRepresents)

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_paths_are_relative_to_repo_root(self, name, entry):
        """
        Absolute paths would be machine-specific and break reproducibility for
        anyone who clones the repository elsewhere.
        """
        assert not Path(entry.processed_h5_path).is_absolute()
        if entry.raw_data_root is not None:
            assert not Path(entry.raw_data_root).is_absolute()

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_processed_path_lives_under_data(self, name, entry):
        assert Path(entry.processed_h5_path).parts[0] == "data"

    def test_processed_paths_are_unique(self):
        """Two datasets writing the same file would silently overwrite."""
        paths = [Path(e.processed_h5_path) for _, e in REGISTERED]
        assert len(paths) == len(set(paths))

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_config_exposes_window_metadata_schema(self, name, entry):
        """Aggregation reads this schema off every registered dataset."""
        config = importlib.import_module(f"{entry.build_module}.config")
        assert isinstance(getattr(config, "WINDOW_METADATA_SCHEMA"), list)


# ── Dynamic accessors on DatasetEntry ─────────────────────────────────────────

class TestEntryAccessors:

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_preprocessing_config_reports_window_size(self, name, entry):
        config = entry.get_default_preprocessing_config()
        assert config["window_size"] is not None

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_metadata_fields_non_empty(self, name, entry):
        assert entry.get_available_metadata_fields()

    @pytest.mark.parametrize("name,entry", REGISTERED, ids=REGISTERED_NAMES)
    def test_metadata_schema_keys_match_field_list(self, name, entry):
        keys = [f.key for f in entry.get_available_metadata_schema()]
        assert keys == entry.get_available_metadata_fields()

    def test_accessors_are_empty_without_a_build_module(self):
        entry = DatasetEntry(name="x", processed_h5_path=Path("data/x/p.h5"))
        assert entry.get_default_preprocessing_config() == {}
        assert entry.get_available_metadata_fields() == []
        assert entry.get_available_metadata_schema() == []

    def test_accessors_degrade_gracefully_on_bad_module(self):
        """A missing module returns empty rather than exploding at import time."""
        entry = DatasetEntry(
            name="x",
            processed_h5_path=Path("data/x/p.h5"),
            build_module="datasets.does_not_exist",
        )
        assert entry.get_default_preprocessing_config() == {}
        assert entry.get_available_metadata_fields() == []


# ── Enum coercion in __post_init__ ────────────────────────────────────────────

class TestEntryConstruction:

    def test_string_task_type_coerced(self):
        entry = DatasetEntry(
            name="x",
            processed_h5_path=Path("data/x/p.h5"),
            task_type="radar_classification",
        )
        assert entry.task_type is TaskType.RADAR_CLASSIFICATION

    def test_string_label_represents_coerced(self):
        entry = DatasetEntry(
            name="x",
            processed_h5_path=Path("data/x/p.h5"),
            label_represents="modulation",
        )
        assert entry.label_represents is LabelRepresents.MODULATION

    def test_enum_member_passes_through(self):
        entry = DatasetEntry(
            name="x",
            processed_h5_path=Path("data/x/p.h5"),
            task_type=TaskType.MODULATION_RECOGNITION,
        )
        assert entry.task_type is TaskType.MODULATION_RECOGNITION

    def test_invalid_task_type_rejected_at_construction(self):
        """A typo must fail immediately, not surface later as a bad label."""
        with pytest.raises(ValueError):
            DatasetEntry(
                name="x", processed_h5_path=Path("data/x/p.h5"), task_type="typo"
            )

    def test_invalid_label_represents_rejected_at_construction(self):
        with pytest.raises(ValueError):
            DatasetEntry(
                name="x",
                processed_h5_path=Path("data/x/p.h5"),
                label_represents="typo",
            )


# ── Module-level lookup helpers ───────────────────────────────────────────────

class TestLookupHelpers:

    def test_list_datasets_is_sorted(self):
        assert list_datasets() == sorted(list_datasets())

    def test_list_datasets_covers_the_registry(self):
        assert set(list_datasets()) == set(DATASET_REGISTRY)

    def test_get_dataset_returns_the_entry(self):
        name = REGISTERED_NAMES[0]
        assert get_dataset(name) is DATASET_REGISTRY[name]

    def test_get_dataset_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError):
            get_dataset("no_such_dataset")

    def test_get_dataset_error_lists_available_names(self):
        """The error is a user's main discovery path for the correct name."""
        with pytest.raises(KeyError) as excinfo:
            get_dataset("no_such_dataset")
        assert REGISTERED_NAMES[0] in str(excinfo.value)
