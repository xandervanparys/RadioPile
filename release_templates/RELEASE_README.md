# RadioPile v1.0 — Release Bundle

## What is RadioPile?

RadioPile is a harmonized collection of seven publicly available radio-frequency
(RF) IQ datasets, unified through a single reproducible preprocessing
pipeline. The corpus contains approximately **25.8 million IQ windows** at
a uniform window size, normalized and re-labeled into a consistent multi-task
metadata schema (technology, modulation, radar waveform). It is intended for
training and evaluating RF foundation models and for cross-dataset benchmarks.

For methodology, source-dataset details, and design rationale, see the
accompanying data descriptor, linked from the Zenodo records once published.

---

## Which file should I download?

* **I want a single ready-to-train corpus spanning all RF data** →
  use [`aggregated/`](aggregated/). One HDF5 file with all sources unified at a
  fixed window size and a consistent metadata schema.

* **I want one specific source dataset with RadioPile's preprocessing applied** →
  use [`per_dataset/<name>/`](per_dataset/). Each subdirectory contains the
  preprocessed HDF5 for one source dataset, at its native window size, with
  source-specific metadata preserved.

* **I need commercial-friendly licensing** → see [`LICENSES.md`](LICENSES.md)
  for the per-file license breakdown. The aggregated corpus inherits the most
  restrictive source license; some per-dataset files are more permissive.

* **I want to verify the release was built correctly** → see
  [`validation_report.json`](validation_report.json). A top-level
  `overall_pass: true` means every per-file structural and metadata check
  passed.

---

## Bundle structure

```
release/v1.0/
├── README.md                 ← this file
├── LICENSES.md               ← per-file license summary
├── validation_report.json    ← machine-readable validation report
├── aggregated/
│   ├── radiopile_v1.0.h5
│   ├── radiopile_v1.0.h5.config.json
│   ├── radiopile_v1.0.croissant.json
│   └── aggregation_config.py   ← Python source of the build config
└── per_dataset/
    ├── rml2016_10a/
    │   ├── *.h5
    │   ├── *.h5.config.json
    │   ├── *.croissant.json
    │   └── rml2016_10a_config.py
    ├── rml2018_01a/
    ├── sub_ghz/
    ├── lte_wifi_dvbt/
    ├── radchar/
    ├── its_band/
    └── nist_wifi_bluetooth/
```

Each artifact directory is self-contained: the HDF5, its two sidecars, and the
Python config module that produced it all sit together. Croissant
`contentUrl` values are bare filenames, so every descriptor resolves relative
to its own directory.

---

## File contents per HDF5

Each `.h5` file ships alongside two sidecar files:

* **`<name>.h5.config.json`** — the build-time configuration snapshot. The same
  configuration that is stored as HDF5 root attributes, plus a UTC timestamp and
  the build-time git commit SHA. Lets you inspect the configuration without
  opening the HDF5 file.

* **`<name>.croissant.json`** — an [MLCommons Croissant](https://github.com/mlcommons/croissant)
  1.0 descriptor: name, license, citations, the SHA-256 of the HDF5 file, and one
  field per dataset in it (`/data`, `/labels`, every `/meta` column). It passes
  `mlcroissant validate`. Preprocessing, label names and provenance, for which
  Croissant has no vocabulary, are under the `radiopile:` prefix. The data itself is
  read with h5py, as shown below.

The HDF5 itself contains:
* `/data` — windowed IQ samples, shape `(N, 2, W)`, dtype `float32`
* `/labels` — integer class indices, shape `(N,)`
* `/label_names` — string label vocabulary
* `/meta/*` — per-window metadata fields (one HDF5 dataset per field)
* root attributes — dataset-level configuration (the contents of
  `.h5.config.json` minus the snapshot timestamp/git_commit, plus `total_windows`;
  lists and dicts are stored as strings)

---

## Important: `/labels` is one view, and it contains an `unknown` class

**Read this before training on `/labels`.**

RadioPile is a multi-task corpus. Every classification target is stored in
`/meta/*` — `technology_type`, `modulation_type`, `radar_signal_type`,
`waveform_type`. The `/labels` and `/label_names` datasets are a *convenience
view* of one of them, selected at build time by `primary_label_type` in the
config snapshot. **For the aggregated v1.0 file that is `technology`.**

Two consequences:

**1. `/labels` is not the modulation task.** If you want modulation
classification, read `/meta/modulation_type` — do not rebuild, and do not use
`/labels`. Nothing about the corpus is specific to the technology view.

**2. `/labels` contains an `unknown` class that is not a class.** Windows with
no label for the *selected* task are assigned the primary label `"unknown"`,
which becomes an ordinary entry in `/label_names` rather than a null. In the
aggregated file this is **3,775,904 windows — 14.6% of the corpus, the second
largest of the 11 entries.** It is the three sources with no technology label:
RadioML 2016.10A and 2018.01A, whose `modulation_type` is perfectly well
populated, and RadChar, whose `radar_signal_type` is.

Training on all 11 entries as-is means training on a class that is not one. Exclude it.
The same holds in the modulation view, where 1,587,264 windows carry `unknown`.
Noise windows are among them: `Noise` is a capture class in the technology view
only.

### Loading either task correctly

Labels are small and can be read whole; `/data` cannot — the file is 117.46 GB, so
select indices first and read IQ in batches.

```python
import h5py, numpy as np

TASK = "technology"          # or "modulation"

with h5py.File("radiopile_v1.0.h5", "r") as f:
    if TASK == "technology":
        y_all = f["meta/technology_type"].asstr()[:]      # 10 classes
        drop  = {"unknown"}
    else:
        y_all = f["meta/modulation_type"].asstr()[:]      # 32 classes
        drop  = {"unknown"}

    idx = np.flatnonzero(~np.isin(y_all, list(drop)))     # keep these windows
    y   = y_all[idx]
    classes = sorted(set(y))
    print(f"{len(idx):,} windows, {len(classes)} classes")

    # Read IQ in batches. h5py needs the index list sorted and ascending.
    for start in range(0, len(idx), 8192):
        batch = idx[start:start + 8192]
        x = f["data"][batch]        # (B, 2, 500) float32
        yield_batch = y[start:start + 8192]
        ...                          # train / featurize here
```

Expected output — `22,060,323 windows, 10 classes` for technology,
`24,248,963 windows, 32 classes` for modulation. If you see 25,836,227 windows,
or 11 or 33 classes, the `unknown` class has not been excluded.

Read labels from `/meta/*` rather than `/labels` and both tasks work from the
same file with no rebuild. This is exactly what the reference baselines do —
see `ml-baselines/_common.py`.

---

## Rebuilding the release

The preprocessing pipeline that produced this bundle is open source; see the
source repository for the build code, raw-data acquisition instructions, and
configuration details: https://github.com/xandervanparys/RadioPile (10.5281/zenodo.23174413).

Within the source repository, `python run.py aggregation` followed by
`make release` reproduces this bundle (assuming the same raw inputs are present
locally).

---

## How to cite

If you use RadioPile in academic work, please cite the dataset record below (and the
data descriptor once it is published) and credit the original source-dataset authors as listed in each file's
`*.croissant.json`.

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

---

## Contact / Issues

For questions, bug reports, or contributions, use the issue tracker at
https://github.com/xandervanparys/RadioPile/issues or contact the maintainer at xander.vanparys@ugent.be.
