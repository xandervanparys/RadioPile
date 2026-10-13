# Test suite

Run from the repository root:

```bash
pytest tests/
```

Every test uses synthetic, in-memory data written to `tmp_path`. **No raw or
processed dataset files are required**, so the suite runs on a fresh clone in a few
seconds. One provenance test skips outside a git checkout, for example in the
source archive on Zenodo.

## Scope

The suite is not exhaustive by design. It covers the paths where a regression
would corrupt released artifacts silently: the output still has the right shape,
dtype and record count, but the wrong contents.

| File | Covers |
|---|---|
| `unit/test_windowing.py` | `window_iq_signal`, `iter_windows_from_bin`: segmentation, hop, padding, source dtype |
| `unit/test_normalization.py` | All five `normalize_window` modes; `unit_power` hardest, as the frozen v1.0 default |
| `unit/test_noise_filtering.py` | The three noise-floor detectors, `determine_noise_threshold`, `filter_window_by_power` |
| `unit/test_hdf5_attrs.py` | `_sanitize_hdf5_attr` and `_coerce_window_meta_value` fill-value conventions |
| `unit/test_label_mapping.py` | Technology / modulation / radar label routing in aggregation |
| `unit/test_metadata_schema.py` | Unified metadata schema completeness |
| `unit/test_window_adaptation.py` | Pad / crop / passthrough at aggregation time |
| `unit/test_registry.py` | Every registry entry imports, and agrees with its own config |
| `unit/test_source_configs.py` | Shared root attributes (`dataset_url`, `sample_rate_hz`, license) and label routing for every registered source |
| `unit/test_source_mappings.py` | Per-source label traps: Sub-GHz MCS and filename parsing, RML2018 class order, RML2016 byte-string keys and spellings, NIST per-file constants |
| `unit/test_location_override.py` | The `lte_wifi_dvbt` capture-location override |
| `unit/test_provenance.py` | The git commit and `-dirty` stamp written into config snapshots |
| `unit/test_validation_report.py` | File fingerprints and code provenance in the validation report |
| `unit/test_raw_manifest.py` | `docs/RAW_INPUT_CHECKSUMS.txt` against its generator, and its internal consistency |
| `integration/test_hdf5_roundtrip.py` | `build_hdf5_dataset` → `load_processed_h5_dataset`, plus the loader's rejection of malformed input |
| `integration/test_croissant_conformance.py` | Emitted Croissant against `metadata/schemas/croissant_minimal.json`, and against `mlcroissant` where installed |

## Dependencies

`pytest` and `jsonschema` are both in `requirements.txt`. `jsonschema` runs the
minimal-schema validation of the Croissant descriptor; without it those two tests
skip and only the required-field checks beside them run. One further test runs the
descriptor through `mlcroissant`, the official Croissant validator, and skips unless
that package is installed (`pip install mlcroissant`).

## Adding a dataset

`unit/test_registry.py` is parametrized over `DATASET_REGISTRY`, so a new dataset
is picked up automatically: it must import, expose `main()`, have a
`config.OUTPUT_H5_PATH` matching its registry `processed_h5_path`, and declare at
least one field in `WINDOW_METADATA_SCHEMA`. `unit/test_source_configs.py` is
parametrized the same way and requires `DATASET_URL`, the license attributes and the
class list the aggregation routes on. If the source has
a label trap of its own, add its case to `unit/test_source_mappings.py`.
