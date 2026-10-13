# RadioPile
![Static Badge](https://img.shields.io/badge/License-Apache_2.0-blue?link=https%3A%2F%2Fwww.apache.org%2Flicenses%2FLICENSE-2.0)

RadioPile unifies seven publicly available RF IQ datasets into one machine-learning-ready
collection, through a reproducible preprocessing pipeline. This repository is that
pipeline: it turns raw IQ captures in heterogeneous formats into standardized HDF5
files with per-window metadata and Croissant descriptors, and builds the aggregated
corpus from them.

Version 1.0 holds 25,836,227 labeled IQ windows spanning ten technology classes (nine
wireless technologies and noise), 32 modulation classes and five radar signal types.

---

## Published release

RadioPile v1.0 is published on Zenodo as two linked records. The full bundle is
240 GB, which exceeds Zenodo's per-record ceiling, so it is split by artifact type.

| Record | Contents | Size | DOI |
|---|---|---|---|
| **Aggregated** | `radiopile_v1.0.h5`: all seven sources unified at 500 samples per window | 117.46 GB | 10.5281/zenodo.22876704 |
| **Per-dataset** | The seven sources processed individually at their native window sizes | 122.13 GB | 10.5281/zenodo.22877108 |
| Source code | This repository at the tagged release commit | — | 10.5281/zenodo.23174413 |

Each HDF5 ships with its build-configuration snapshot, its Croissant descriptor and
the configuration module that produced it. Both records also carry `README.md`,
`LICENSES.md`, a byte-identical `validation_report.json` covering all eight files,
and `RAW_INPUT_CHECKSUMS.txt`.

Most users want the **aggregated** record. Take the per-dataset record for one
specific source without aggregation, or for a license more permissive than the
aggregate's (see [License](#license)).

---

## Repository structure

```text
RadioPile/
├── data/<name>/raw|processed # Not tracked: raw inputs and built HDF5s
├── datasets/
│   ├── _template/            # Starting point for a new source
│   ├── aggregation/          # Builds the unified corpus from the per-dataset files
│   ├── <name>/               # One package per source: config.py + build_dataset.py
│   └── registry.py           # Every buildable dataset: paths, build module, task
├── docs/                     # DATASETS, DESIGN, ADDING_A_DATASET, NOISE_FILTERING
├── figures/                  # Figures and supporting figures for the paper
├── metadata/                 # Croissant generation and its minimal schema
├── ml-baselines/             # Reference baselines reported in the paper
├── notebooks/                # Visual inspection of built files and noise floors
├── pipeline/                 # Dataset-agnostic core: io, windowing, preprocessing,
│                             #   noise_analysis, hdf5_writer, labels, schemas
├── release/                  # Not tracked: make release output and Zenodo staging
├── release_templates/        # README and LICENSES shipped with the bundle and records
├── scripts/                  # Validation, raw-input verification, audits, release tooling
├── tests/                    # pytest, synthetic data only, no datasets needed
├── Makefile                  # make release: validate, bundle, write the report
└── run.py                    # Single entry point: python run.py <dataset>
```

---

## Setup

Requires Python 3.11 to 3.13. The v1.0 corpus was built and validated on CPython
3.11.15; versions in `requirements.txt` are pinned to that environment.

```bash
./setup_env.sh
source .venv_pipeline/bin/activate
pytest tests/                 # verifies the install; needs no data
```

`requirements.txt` covers the pipeline, the validator, the figure scripts and the
tests. TensorFlow is needed only for the baselines and installs separately from
`requirements-ml-baselines.txt`.

---

## Building a dataset

Raw data is not included. Where to download each source, and which files are used,
is in [`docs/DATASETS.md`](docs/DATASETS.md).

A full build needs about 160 GB of disk for the raw inputs and 240 GB for the built
files; `make release` copies the built files, which takes another 240 GB. Building
every source and the aggregate took about one hour on the machine used for v1.0 (NVMe
storage, 32 GB of RAM).

1. **Place the raw files** under `data/<dataset>/raw/`, at the paths listed in
   [`docs/RAW_INPUT_CHECKSUMS.txt`](docs/RAW_INPUT_CHECKSUMS.txt). Filenames carry
   capture parameters and must be kept.
2. **[OPTIONAL] Check them** against the checksums recorded for v1.0:

   ```bash
   python scripts/verify_raw_inputs.py --dataset lte_wifi_dvbt
   ```

3. **Build:**

   ```bash
   python run.py lte_wifi_dvbt       # one source
   python run.py aggregation         # every source, then the unified corpus
   ```

   `python run.py --help` lists the available datasets.

4. **Validate:**

   ```bash
   python scripts/validate_datasets.py --save-report validation_report.json
   ```

A build writes three files to `data/<dataset>/processed/` (for the aggregate:
`data/aggregation/processed/aggregated.*`):

```text
<dataset>.h5                # /data, /labels, /label_names, /meta/*
<dataset>.h5.config.json    # build configuration, timestamp and git commit
<dataset>.croissant.json    # Croissant 1.0 metadata
```

Each dataset is defined by two files: `config.py` states what the dataset is (window
size, normalization, label mapping, metadata schema) and `build_dataset.py` holds the
source-specific parsing. Changing `config.py` changes the dataset.

---

## Frozen v1 defaults

These are fixed for v1 so that results stay comparable; do not change them when
reproducing the release.

| Setting | Value | Meaning |
|---|---|---|
| `TARGET_WINDOW_SIZE` | `500` | Window size of the aggregated corpus |
| `PADDING_STRATEGY` | `"zero"` | Shorter windows are zero-padded |
| `CROP_POLICY` | `"start"` | Longer windows keep their first 500 samples |
| `NORMALIZE_MODE` | `"unit_power"` | Per-window unit-power normalization |
| `PRIMARY_LABEL_TYPE` | `"technology"` | `/labels` of the aggregate holds the technology class |
| `DATA_DTYPE` | `"float32"` | Storage dtype of the IQ samples |

`/labels` is one view of a multi-task corpus. Every target is stored per window under
`/meta/` (`technology_type`, `modulation_type`, `radar_signal_type`, `waveform_type`);
read the field for your task and drop the windows whose value is `"unknown"`.

---

## Documentation

* [`docs/DATASETS.md`](docs/DATASETS.md) — everything per source: provenance and
  license, raw format, parsing, classes, metadata fields, decisions, caveats and the
  group key for splitting.
* [`docs/DESIGN.md`](docs/DESIGN.md) — everything cross-cutting: the label schema, the
  pipeline, the HDF5 layout, train/test splitting, the decisions taken and the open
  questions.
* [`docs/ADDING_A_DATASET.md`](docs/ADDING_A_DATASET.md) — how to integrate a new
  source, starting from `datasets/_template/`.
* [`docs/NOISE_FILTERING.md`](docs/NOISE_FILTERING.md) — how to choose a noise
  threshold.
* [`ml-baselines/README.md`](ml-baselines/README.md) — the reference baselines and how to
  reproduce them.

---

## How to cite

If you use the RadioPile collection, cite the dataset record:

```bibtex
@dataset{radiopile_v1_0,
  title        = {RadioPile v1.0: A Unified Multi-Task RF IQ Collection},
  author       = {Vanparys, Xander and Cheraghinia, Mohammad and De Poorter, Eli and Shahid, Adnan},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.22876704},
  note         = {Per-dataset files: 10.5281/zenodo.22877108. Pipeline: 10.5281/zenodo.23174413},
  url          = {https://doi.org/10.5281/zenodo.22876704}
}
```

If you use this pipeline, [`CITATION.cff`](CITATION.cff) has the software citation.

RadioPile is built from seven third-party datasets. Please also cite the sources you
use: their references are in [`docs/DATASETS.md`](docs/DATASETS.md) and in each
file's `*.croissant.json`.

---

## License

Code is released under the [Apache License 2.0](LICENSE). See [`NOTICE`](NOTICE) for
attribution requirements.

Generated datasets are **not** covered by that license. Each source carries its own
terms (CC-BY-NC-SA-4.0, CC-BY-4.0, or the NIST data terms), recorded in the `LICENSE`
constant of `datasets/<name>/config.py` and written into every HDF5 and Croissant
file. The aggregated file mixes sources, so it inherits the most restrictive license
among them (CC-BY-NC-SA-4.0). The per-dataset files were never mixed, so each keeps
its own: `nist_wifi_bluetooth` and `its_band` are usable commercially as standalone
files. The `LICENSES.md` inside each published record gives the terms that apply to
what you downloaded.
