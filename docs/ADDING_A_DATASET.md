# Adding a New Dataset

Step-by-step guide for integrating a new RF dataset into the pipeline.

## Prerequisites

- Raw data downloaded and placed under `data/<dataset_name>/raw/`
- You know the file format (`.bin`, `.mat`, `.h5`, `.pkl`), the source dtype, and how labels are encoded (filename, directory, internal structure)

## Steps

### 1. Copy the template

```bash
cp -r datasets/_template datasets/<dataset_name>
```

### 2. Fill in `config.py`

Open `datasets/<dataset_name>/config.py` and set every field marked `TODO`:

| Field | What to set |
|---|---|
| `DATASET_NAME` | Human-readable name, written as the `dataset` root attribute (e.g. `"RadioML 2018.01A (Modulation Classification)"`) |
| `DATASET_URL` | Where the source data is published; written as the `dataset_url` root attribute |
| `RAW_DATA_ROOT` | Path to raw data (e.g. `Path("data/rml2018_01a/raw")`) |
| `OUTPUT_ROOT` | Path for processed output (e.g. `Path("data/rml2018_01a/processed")`) |
| `SOURCE_DTYPE` | Numpy dtype of the raw files, as a string (e.g. `"complex64"`, `"int16"`) |
| `LABEL_REMAP` | Dict mapping raw labels to class names; an identity mapping if nothing is renamed. Aggregation reads its values as the valid technology labels (`MODULATION_TYPES` for a modulation source, `RADAR_SIGNAL_TYPES` for radar) |
| `WINDOW_SIZE` / `HOP_SIZE` | Window parameters for this dataset |
| `NORMALIZE_MODE` | Usually `"unit_power"` |
| `WINDOW_METADATA_SCHEMA` | List of `WindowMetaField` entries for per-window metadata you want to track |
| `CITATIONS` / `LICENSE` | Croissant metadata; `LICENSE` is also written as the `license` root attribute |

See existing datasets for examples:
- `datasets/rml2016_10a/config.py` — pickle format, modulation labels
- `datasets/sub_ghz/config.py` — MATLAB format, technology labels, custom metadata fields
- `datasets/lte_wifi_dvbt/config.py` — binary format, filename-parsed metadata

### 3. Implement `build_dataset.py`

Your build script must yield `(window, label_name, metadata_dict)` tuples. How you get there depends entirely on the dataset's format. There are two common patterns:

**Pattern A: File-based datasets** (one or more raw files per class)
The template is set up for this. Implement:
- `infer_label_from_filename(path)` — returns the label for a given file
- `parse_filename_metadata(path)` — returns metadata extracted from filename/path (center frequency, sample rate, gain, location, etc.)

The template's `iter_dataset_windows()` handles the rest (windowing, normalization, noise filtering). Examples: `datasets/sub_ghz/`, `datasets/lte_wifi_dvbt/`.

**Pattern B: Self-contained datasets** (single file with all samples + labels inside)
Replace `iter_dataset_windows()` entirely with custom loading logic. Labels and metadata come from inside the file, not from filenames. Example: `datasets/rml2016_10a/` loads a pickle where labels are dictionary keys.

In both cases, `main()` feeds the iterator into `build_hdf5_dataset()` — that part stays the same.

### 4. Add a registry entry

Open `datasets/registry.py` and add your dataset to `DATASET_REGISTRY`:

```python
"your_dataset": DatasetEntry(
    name="your_dataset",
    processed_h5_path=Path("data/your_dataset/processed/your_dataset.h5"),
    build_module="datasets.your_dataset",
    raw_data_root=Path("data/your_dataset/raw"),
    task_type="modulation_recognition",      # or "technology_recognition", "radar_classification", etc.
    label_represents="modulation",           # or "technology", "radar_waveform" — drives aggregation label mapping
),
```

### 5. Keep `__init__.py`

`datasets/<dataset_name>/` must contain an `__init__.py` (an empty file; the template
has one) so the directory is an importable package. That is all the packaging the
dataset needs: `run.py` is the single entry point for every dataset and resolves your
build module through the registry entry you added in step 4.

### 6. Build the dataset

```bash
python run.py <dataset_name>
```

### 7. Validate

Three checks, in order:

```bash
# 1. Structural integrity and metadata completeness
python scripts/validate_datasets.py --dataset <dataset_name>

# 2. Regression tests (confirms pipeline functions still work)
pytest tests/

# 3. Visual inspection — open notebooks/validation.ipynb, point it at
#    data/<dataset_name>/processed/<dataset_name>.h5
#    and check the time-domain / spectrogram plots
```

### 8. Rebuild aggregation

If this dataset should be included in the aggregated corpus:

1. Add it to `SELECTED_DATASETS` in `datasets/aggregation/config.py`
2. If its license is not yet ranked in `LICENSE_RESTRICTIVENESS` in that file, rank it;
   an unranked license makes the aggregate's license `NOASSERTION`
3. Rebuild: `python run.py aggregation`
4. Validate: `python scripts/validate_datasets.py --dataset aggregation`

## Checklist

- [ ] `config.py` — all `TODO` fields filled in
- [ ] `build_dataset.py` — yields `(window, label_name, metadata_dict)` tuples correctly
- [ ] `__init__.py` exists
- [ ] Copied `README.md` deleted; dataset documented in `docs/DATASETS.md`
- [ ] Registry entry added in `datasets/registry.py`
- [ ] `python run.py <dataset_name>` runs without errors
- [ ] `python scripts/validate_datasets.py --dataset <dataset_name>` passes
- [ ] `pytest tests/` still passes (no regressions)
- [ ] Visual inspection in `notebooks/validation.ipynb` looks correct
- [ ] Aggregation rebuilt and validated (if applicable)

## Common pitfalls

| Symptom | Likely cause |
|---|---|
| I channel near-zero, Q dominant | Wrong `SOURCE_DTYPE` — check if raw files are `complex64` vs `int16` |
| All windows look like noise | OFDM signals (LTE, WiFi, DVB-T) genuinely look noise-like in time domain — check spectrogram instead |
| Metadata field absent from the built file, or holding only fill values | Field not in `WINDOW_METADATA_SCHEMA`, or its `key` differs from the key the iterator yields |
| Label not found in `LABEL_REMAP` | `infer_label_from_filename()` returns a string not in `LABEL_REMAP` keys |
| Wrong window count | Check `WINDOW_SIZE` and `HOP_SIZE` — large hop = fewer windows |
