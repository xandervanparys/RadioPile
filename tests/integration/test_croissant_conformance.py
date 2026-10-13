"""
Integration tests: build_hdf5_dataset → generate_croissant_from_hdf5.

Croissant sidecars are a headline feature of the release and are consumed by
third parties, so a malformed descriptor fails on someone else's machine after
publication. These tests build a small synthetic HDF5, emit a descriptor from
it, and check the result against metadata/schemas/croissant_minimal.json.

The full-schema check uses jsonschema when it is installed and skips otherwise;
the required-field checks below it are pure stdlib and always run, so the suite
retains protection with no new hard dependency. Conformance with the Croissant
specification itself is checked by mlcroissant, which is not a dependency of
this repository; that test runs where it is installed and skips elsewhere.

Synthetic data only — no dataset files required.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from metadata.croissant import generate_croissant_from_hdf5
from pipeline.hdf5_writer import build_hdf5_dataset
from pipeline.labels import reset_label_registry


SCHEMA_PATH = Path("metadata/schemas/croissant_minimal.json")

LABELS = ["LTE", "WiFi", "DVB-T"]

PREPROCESSING = {
    "window_size": 500,
    "hop_size": 500,
    "pad_last": True,
    "normalization": "unit_power",
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

def synthetic_iterator(n=12, window_size=500):
    """Yield (window, label_name, meta) triples across the three label classes."""
    for i in range(n):
        window = np.full((2, window_size), float(i + 1), dtype=np.float32)
        yield window, LABELS[i % len(LABELS)], {}


@pytest.fixture
def built_h5(tmp_path):
    """A small, valid processed HDF5 written from synthetic windows."""
    path = tmp_path / "synthetic.h5"
    reset_label_registry()
    build_hdf5_dataset(output_path=path, window_iterator=synthetic_iterator())
    return path


@pytest.fixture
def croissant(built_h5, tmp_path):
    """The emitted descriptor, parsed."""
    out = tmp_path / "synthetic.croissant.json"
    generate_croissant_from_hdf5(
        h5_path=built_h5,
        output_path=out,
        dataset_name="Synthetic Test Dataset",
        description="Synthetic windows used to exercise Croissant emission.",
        task_type="classification",
        labels=LABELS,
        citations=["Example et al., Journal of Examples, 2026"],
        license="CC-BY-4.0",
        preprocessing=PREPROCESSING,
        provenance={"script_version": "1.0.0"},
    )
    return json.loads(out.read_text())


@pytest.fixture
def schema():
    return json.loads(SCHEMA_PATH.read_text())


# ── Schema conformance ────────────────────────────────────────────────────────

class TestSchemaConformance:

    def test_schema_file_is_a_json_schema(self, schema):
        """
        Guards against the schema regressing to a placeholder. Validating against
        an object with no constraints passes vacuously, so the schema itself must
        assert something before the conformance test below means anything.
        """
        assert schema.get("type") == "object"
        assert schema.get("required"), "schema declares no required fields"
        assert "properties" in schema

    def test_emitted_croissant_validates(self, croissant, schema):
        jsonschema = pytest.importorskip("jsonschema")
        jsonschema.validate(instance=croissant, schema=schema)

    def test_schema_rejects_a_missing_required_field(self, croissant, schema):
        """The negative control: proof the schema can actually fail something."""
        jsonschema = pytest.importorskip("jsonschema")
        broken = dict(croissant)
        broken.pop("distribution")
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=broken, schema=schema)

    def test_all_schema_required_fields_present(self, croissant, schema):
        """Stdlib-only equivalent of the above, so coverage survives without jsonschema."""
        missing = [key for key in schema["required"] if key not in croissant]
        assert missing == []

    def test_passes_the_official_validator(self, built_h5, tmp_path):
        """`mlcroissant validate` raises on any error in the descriptor."""
        mlc = pytest.importorskip("mlcroissant")
        out = tmp_path / "official.croissant.json"
        generate_croissant_from_hdf5(
            h5_path=built_h5,
            output_path=out,
            dataset_name="Synthetic Test Dataset",
            description="Synthetic windows used to exercise Croissant emission.",
            task_type="classification",
            labels=LABELS,
            citations=["Example et al., Journal of Examples, 2026"],
            doi="10.5281/zenodo.0000000",
            version="1.0.0",
            date_published="2026-01-01",
            license="CC-BY-4.0",
            preprocessing=PREPROCESSING,
        )
        mlc.Dataset(jsonld=str(out))


# ── distribution[] ────────────────────────────────────────────────────────────

class TestDistribution:

    def test_distribution_is_non_empty(self, croissant):
        assert len(croissant["distribution"]) >= 1

    def test_content_url_populated(self, croissant, built_h5):
        """
        contentUrl is how a consumer locates the actual HDF5. An empty or missing
        value produces a descriptor that resolves to nothing.
        """
        for item in croissant["distribution"]:
            assert item.get("contentUrl")
            assert item["contentUrl"].strip() != ""
        assert croissant["distribution"][0]["contentUrl"] == str(built_h5)

    def test_name_matches_the_hdf5_filename(self, croissant, built_h5):
        assert croissant["distribution"][0]["name"] == built_h5.name

    def test_encoding_format_is_hdf5(self, croissant):
        assert croissant["distribution"][0]["encodingFormat"] == "application/x-hdf5"

    def test_file_object_type(self, croissant):
        assert croissant["distribution"][0]["@type"] == "cr:FileObject"

    def test_checksum_is_the_sha256_of_the_file(self, croissant, built_h5):
        import hashlib

        assert croissant["distribution"][0]["sha256"] == hashlib.sha256(built_h5.read_bytes()).hexdigest()

    def test_every_field_reads_from_the_file_object(self, croissant):
        file_id = croissant["distribution"][0]["@id"]
        for field in croissant["recordSet"][0]["field"]:
            assert field["source"]["fileObject"] == {"@id": file_id}


# ── recordSet and task description ────────────────────────────────────────────

class TestRecordSet:

    def test_declares_data_and_labels_fields(self, croissant):
        names = {f["name"] for f in croissant["recordSet"][0]["field"]}
        assert {"data", "labels"} <= names

    def test_label_map_covers_every_label(self, croissant):
        assert croissant["radiopile:labels"] == {"0": "LTE", "1": "WiFi", "2": "DVB-T"}

    def test_metadata_columns_are_declared(self, croissant, built_h5):
        import h5py

        with h5py.File(built_h5, "r") as f:
            columns = {f"meta/{key}" for key in f["meta"]}
        declared = {f["source"]["extract"]["column"] for f in croissant["recordSet"][0]["field"]}
        assert columns <= declared

    def test_data_field_records_on_disk_window_shape(self, croissant):
        """The descriptor reports the HDF5 layout, which is (channels, samples)."""
        data_field = next(
            f for f in croissant["recordSet"][0]["field"] if f["name"] == "data"
        )
        assert data_field["radiopile:arrayShape"] == [2, 500]

    def test_declared_shape_matches_the_file_but_not_the_loader(self, croissant, built_h5):
        """
        Pins a real trap for consumers. /data is stored as (N, 2, W) and the
        Croissant descriptor faithfully reports that. The repository's own
        loader, load_processed_h5_dataset, transposes to (N, W, 2) for ML use.

        So a consumer who reads the array shape from the descriptor and a consumer who
        calls the loader see different axis orders for the same file. Both are
        internally correct; the descriptor is the one describing the bytes.
        """
        import h5py

        from pipeline.io import load_processed_h5_dataset

        data_field = next(
            f for f in croissant["recordSet"][0]["field"] if f["name"] == "data"
        )
        with h5py.File(built_h5, "r") as f:
            on_disk_window_shape = list(f["data"].shape[1:])

        assert data_field["radiopile:arrayShape"] == on_disk_window_shape == [2, 500]

        x, _, _, _ = load_processed_h5_dataset(built_h5)
        assert list(x.shape[1:]) == [500, 2]

    def test_label_count_matches_labels(self, croissant):
        assert croissant["radiopile:task"]["labelCount"] == len(LABELS)

    def test_task_names_input_and_target(self, croissant):
        task = croissant["radiopile:task"]
        assert (task["type"], task["input"], task["target"]) == ("classification", "data", "labels")


# ── Descriptive fields carried through from the caller ────────────────────────

class TestPassthroughFields:

    def test_license_preserved(self, croissant):
        assert croissant["license"] == "CC-BY-4.0"

    def test_citations_preserved(self, croissant):
        assert croissant["citation"] == ["Example et al., Journal of Examples, 2026"]

    def test_preprocessing_block_preserved(self, croissant):
        assert croissant["radiopile:preprocessing"] == PREPROCESSING

    def test_provenance_included_when_given(self, croissant):
        assert croissant["radiopile:provenance"] == {"script_version": "1.0.0"}

    def test_provenance_omitted_when_absent(self, built_h5, tmp_path):
        out = tmp_path / "no_provenance.croissant.json"
        generate_croissant_from_hdf5(
            h5_path=built_h5,
            output_path=out,
            dataset_name="No Provenance",
            description="d",
            task_type="classification",
            labels=LABELS,
            citations=[],
            license="CC-BY-4.0",
            preprocessing=PREPROCESSING,
        )
        assert "radiopile:provenance" not in json.loads(out.read_text())

    def test_date_created_is_iso_format(self, croissant):
        from datetime import date

        assert croissant["dateCreated"] == str(date.today())


# ── Failure modes ─────────────────────────────────────────────────────────────

class TestEmitterErrors:

    def test_missing_hdf5_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            generate_croissant_from_hdf5(
                h5_path=tmp_path / "absent.h5",
                output_path=tmp_path / "out.json",
                dataset_name="x",
                description="d",
                task_type="classification",
                labels=[],
                citations=[],
                license="CC-BY-4.0",
                preprocessing={},
            )

    def test_hdf5_without_required_datasets_raises(self, tmp_path):
        """A descriptor must not be emitted for a file that lacks /data or /labels."""
        import h5py

        path = tmp_path / "incomplete.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("something_else", data=np.zeros(4))

        with pytest.raises(ValueError, match="/data and /labels"):
            generate_croissant_from_hdf5(
                h5_path=path,
                output_path=tmp_path / "out.json",
                dataset_name="x",
                description="d",
                task_type="classification",
                labels=[],
                citations=[],
                license="CC-BY-4.0",
                preprocessing={},
            )

    def test_output_directory_created_if_absent(self, built_h5, tmp_path):
        out = tmp_path / "nested" / "deeper" / "x.croissant.json"
        generate_croissant_from_hdf5(
            h5_path=built_h5,
            output_path=out,
            dataset_name="x",
            description="d",
            task_type="classification",
            labels=LABELS,
            citations=[],
            license="CC-BY-4.0",
            preprocessing={},
        )
        assert out.exists()
