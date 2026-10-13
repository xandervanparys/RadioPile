# RadioPile v1.0 — Aggregated Collection (Record 1 of 2)

RadioPile v1.0 unifies seven publicly available radio-frequency (RF) in-phase and
quadrature (IQ) datasets into a single machine-learning-ready collection, produced by
a reproducible, open-source preprocessing pipeline.

**This release spans two Zenodo records.** The full bundle exceeds Zenodo's per-record
ceiling, so it is split by artifact type:

| Record | Contents | DOI |
|---|---|---|
| **This record** | The aggregated collection — one HDF5 with all seven sources unified | 10.5281/zenodo.22876704 |
| Per-dataset record | The seven source datasets individually, at their native window sizes | 10.5281/zenodo.22877108 |

Most users want this record. Take the per-dataset record only if you need one specific
source with RadioPile's preprocessing applied but without aggregation.

---

## Contents

| File | Size | Description |
|---|---|---|
| `radiopile_v1.0.h5` | 117.5 GB | The aggregated collection |
| `radiopile_v1.0.h5.config.json` | — | Build configuration snapshot |
| `radiopile_v1.0.croissant.json` | — | MLCommons Croissant 1.0 descriptor |
| `aggregation_config.py` | — | The configuration module used to build this release |
| `validation_report.json` | — | Automated validation results, 146/146 checks, with the size, SHA-256 and MD5 of each file checked. Covers **both records** — all seven per-dataset files and this aggregate. The same file ships in the per-dataset record. |
| `RAW_INPUT_CHECKSUMS.txt` | — | SHA-256 of every raw third-party input, for anyone rebuilding from the original sources (see below). The same file ships in the per-dataset record. |
| `README.md`, `LICENSES.md` | — | This file and the per-source license breakdown |

## What is in it

25,836,227 labeled IQ windows — 12,918,113,500 IQ samples — spanning ten technology
classes (nine wireless technologies and noise), 32 modulation classes, and five radar signal
types, mixing software-defined-radio captures with simulated signals.

Real captures span **482 MHz to 5.9 GHz**. The three simulated sources (RadioML
2016.10A, RadioML 2018.01A, RadChar) carry no center frequency, so
`center_frequency_hz` is `NaN` for 3,775,904 windows (14.6%).

Every window carries standardized metadata under a single multi-target label schema:
technology, modulation, waveform family, radar signal type, SNR, source dataset, and
per-window provenance.

## Structure

IQ data is in `/data` with shape `(25836227, 2, 500)` as 32-bit float — one
500-sample window per row, real and imaginary parts on the second axis. Integer labels
are in `/labels` with class names in `/label_names`, and **17** per-window metadata
fields are stored under `/meta/`. The label and metadata arrays are small and can be
read without touching the IQ payload.

```python
import h5py

with h5py.File("radiopile_v1.0.h5", "r") as f:
    x = f["data"][:1000]                          # (1000, 2, 500) float32

    # String metadata is stored as variable-length bytes. Use .asstr() —
    # without it you get b'rml2016_10a', which never equals 'rml2016_10a'.
    src  = f["meta/source_dataset"].asstr()[:1000]
    task = f["meta/task_type"].asstr()[:1000]
```

## Usage notes

* **Filter to the target task before supervised training.** Windows with no label for
  that task carry `"unknown"`, which is an absence of a label rather than a class.
  `/labels` reflects `primary_label_type = "technology"`, so its 11 entries include
  `unknown` — 3,775,904 windows, 14.6% of the corpus, and the second-largest entry.
  Read the task-specific field from `/meta/` instead:

  ```python
  import numpy as np
  y   = f["meta/technology_type"].asstr()[:]       # or modulation_type
  idx = np.flatnonzero(y != "unknown")             # 22,060,323 windows, 10 classes
  ```

  Expect 22,060,323 windows across 10 technology classes, or 24,248,963 across 32
  modulation classes once `unknown` is excluded. Noise windows carry `unknown`
  in `modulation_type`; `Noise` is a class in the technology view only.

* **NIST WiFi contributes 79.8% of windows.** Subsample or reweight.

* **Technology is partly confounded with capture domain** — most technology classes
  come from a single source. Prefer source-aware evaluation.

* **Windows from one capture are strongly correlated.** Do not split at random, and
  do not group by `recording_id` alone: `sub_ghz` and `lte_wifi_dvbt` reuse the same
  run indices in every file. Group by `source_dataset` + `file_path` +
  `recording_id` and keep each group on one side of the split. What a group is
  differs by source: a raw capture file (`sub_ghz`, `lte_wifi_dvbt`), a one-second
  capture (`nist_wifi_bluetooth`), or one 1,100-sample observation, i.e. at most
  two windows (`its_band`). The simulated sources (`rml2016_10a`, `rml2018_01a`,
  `radchar`) have an empty `recording_id` and no grouping to respect.

* **The 5 GHz indoor and outdoor NIST files were recorded simultaneously.** Their
  windows share `recording_id` values and a center frequency and differ in
  `file_path`. Grouping as above puts an indoor capture and the outdoor capture
  taken at the same moment in separate groups; to keep them together, group the
  NIST windows on `recording_id` alone, as the paper's baselines do. The NIST
  windows are rows 5,208,400 to 25,836,226: the first 9,559,474 are 2.4 GHz indoor,
  the next 7,407,438 are 5 GHz indoor and the last 3,660,915 are 5 GHz outdoor.

## Sources

Sub-GHz technology recognition (Fontaine et al.), LTE/WiFi/DVB-T technology
recognition (Fontaine et al.), NIST WiFi recordings (Tschimben et al.), ITS band
technology recognition (Girmay et al.), RadioML 2016.10A and 2018.01A (O'Shea et al.),
and RadChar (Huang et al.). The original datasets remain available from their
respective repositories; full citations are in `radiopile_v1.0.croissant.json`.

## Verifying the source data (only if you rebuild)

`RAW_INPUT_CHECKSUMS.txt` lists the SHA-256 of every raw third-party file this
collection was built from. You do not need it to use the data in this record —
it is for anyone re-running the pipeline from the original sources.

Two of the seven sources are not versioned upstream, so "the same dataset" is
not necessarily the same bytes. To check yours:

```bash
git clone https://github.com/xandervanparys/RadioPile && cd RadioPile
# fetch the sources (see docs/DATASETS.md) into data/<name>/raw/
python scripts/verify_raw_inputs.py
```

It reports any file that is missing, differs, or is present but unrecorded. A
mismatch means a rebuild will not reproduce this corpus.

If your raw files already sit in that layout, the manifest also works on its
own, without the repository:

```bash
sha256sum -c RAW_INPUT_CHECKSUMS.txt
```

## Related resources

* Per-dataset files: 10.5281/zenodo.22877108
* Preprocessing pipeline: https://github.com/xandervanparys/RadioPile (10.5281/zenodo.23174413)
* Data descriptor: listed under this record's related works on Zenodo once published

## License

**CC-BY-NC-SA-4.0**, inherited from the most restrictive source license among the
inputs. The aggregate mixes sources, so the strongest constraint present governs the
whole file. Per-source licenses are listed in `LICENSES.md`. If you need a more
permissive posture, take the individual source from the per-dataset record instead —
two of the seven are commercially usable.
