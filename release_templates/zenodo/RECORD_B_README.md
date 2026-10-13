# RadioPile v1.0 — Per-Dataset Files (Record 2 of 2)

The seven source datasets of RadioPile v1.0, each processed individually by the same
pipeline but **not** aggregated: every file keeps its own native window size, its own
label space, and **its own original source license**.

**This release spans two Zenodo records.** The full bundle exceeds Zenodo's per-record
ceiling, so it is split by artifact type:

| Record | Contents | DOI |
|---|---|---|
| Aggregated record | The aggregated collection — one HDF5, 25.8M windows, all seven sources unified | 10.5281/zenodo.22876704 |
| **This record** | The seven source datasets individually, at native window sizes | 10.5281/zenodo.22877108 |

Most users want the **aggregated** record. Take this one if you need a single source
with RadioPile's preprocessing but without aggregation, or if you need a license more
permissive than the aggregate's.

---

## Contents

Four files per dataset — the HDF5, its build-configuration snapshot, its Croissant
descriptor, and the configuration module that produced it — for 28 files, plus this
README, `LICENSES.md`, `validation_report.json` and `RAW_INPUT_CHECKSUMS.txt`.

| File | Source | Windows | Window size | Size | License |
|---|---|---|---|---|---|
| `nist_wifi_bluetooth` | NIST WiFi recordings (Tschimben et al.) | 20,627,827 | 500 | 89.9 GB | **NIST data terms** |
| `rml2018_01a` | RadioML 2018.01A (O'Shea et al.) | 2,555,904 | 1024 | 21.4 GB | CC-BY-NC-SA-4.0 |
| `radchar` | RadChar (Huang et al.) | 1,000,000 | 512 | 4.3 GB | CC-BY-NC-SA-4.0 |
| `sub_ghz` | Sub-GHz technology recognition (Fontaine et al.) | 927,152 | 500 | 4.1 GB | CC-BY-NC-SA-4.0 |
| `lte_wifi_dvbt` | LTE/WiFi/DVB-T technology recognition (Fontaine et al.) | 415,344 | 500 | 1.8 GB | CC-BY-NC-SA-4.0 |
| `rml2016_10a` | RadioML 2016.10A (O'Shea & West) | 220,000 | 128 | 0.3 GB | CC-BY-NC-SA-4.0 |
| `its_band` | ITS band technology recognition (Girmay et al.) | 90,000 | 500 | 0.4 GB | **CC-BY-4.0** |

Each source has four files: `<name>.h5` (the data), `<name>.h5.config.json` (build
configuration snapshot), `<name>.croissant.json` (Croissant 1.0 descriptor), and
`<name>_config.py` (the configuration module used to build it). Zenodo records are
flat, so these sit alongside one another rather than in per-dataset folders; the
shared filename stem is what groups them. `<name>` is the dataset's registry name —
the same value the aggregated corpus stores in `source_dataset`.

Also included:

* `README.md` and `LICENSES.md`.
* `RAW_INPUT_CHECKSUMS.txt` — the SHA-256 of every raw third-party input, for anyone
  rebuilding from the original sources (see below). The same file ships in the
  aggregated record.
* `validation_report.json` — the automated validation results. **One report covers
  the whole release**: all seven files here and the aggregated collection, 146 checks,
  `overall_pass: true`, with the size, SHA-256 and MD5 of each file it checked. The byte-identical file is in both Zenodo records, so you have
  the evidence whichever one you downloaded. Each dataset appears under its registry
  name, the same `<name>` used by the files above.

## Licensing — read this before redistributing

**Each file keeps the license of its original source dataset.** There is no single
license covering this record. Unlike the aggregated record, which inherits the most
restrictive license among its inputs, these files were never mixed, so each one
carries only its own terms:

* `nist_wifi_bluetooth.h5` — **NIST data terms**. Not subject to
  copyright in the United States; elsewhere NIST grants a royalty-free right to use,
  modify and redistribute. Commercial use permitted. Acknowledgment of NIST required.
* `its_band.h5` — **CC-BY-4.0**. Commercial use permitted with attribution.
* The remaining five — **CC-BY-NC-SA-4.0**. Non-commercial, share-alike, attribution.

So two of the seven are usable commercially as standalone files, which is the main
reason this record exists separately from the aggregate. `LICENSES.md` gives the
per-file breakdown and the reasoning; each `*.croissant.json` carries the canonical
citation for its source.

## Structure

Every HDF5 follows the same layout as the aggregate: `/data` with shape
`(N, 2, W)` as 32-bit float, integer labels in `/labels` with class names in
`/label_names`, and per-window metadata under `/meta/`. **Window size `W` differs per
dataset** — 128, 500, 512 and 1024 all appear — because each is kept at its native
size. The aggregated record is where they are adapted to a common 500.

Unlike the aggregate, these files retain **dataset-specific metadata** that the
unified schema omits: `sub_ghz` carries `spreading_factor` (LoRa SF) and
`wifi_standard`, and `radchar` carries its four regression targets —
`number_of_pulses`, `pulse_width_s`, `time_delay_s` and `pri_s` (pulse repetition
interval; the `_s` suffix marks seconds).

`lte_wifi_dvbt` carries `location`, the capture site. For one capture it differs from
the upstream filename: `gentbrugge/lte10Msps_g30_uz_f806MHz_r1.bin` is named `uz` but
labeled `gentbrugge`, because the LTE cell IDs it contains are the Gentbrugge site's
and none of UZ's. `file_path` keeps the upstream name. The evidence is in the code
repository, `docs/DATASETS.md`, "Misnamed Gentbrugge capture".

```python
import h5py

with h5py.File("radchar.h5", "r") as f:
    x      = f["data"][:1000]                   # (1000, 2, 512) float32
    labels = f["label_names"].asstr()[:]        # decode to str, not bytes
    pri    = f["meta/pri_s"][:1000]             # a RadChar-only regression target
```

## When to use these files instead of the aggregate

* Single-source experiments at the source's native window size.
* Building your own variant of the aggregated collection.
* Dataset-specific metadata and regression targets the unified schema omits.
* Less restrictive licensing: **`nist_wifi_bluetooth` and `its_band`** are not bound
  by the NonCommercial terms that govern the aggregate. The other five are — RadChar
  included, despite being simulated.

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

## Splitting into train and test

Windows cut from one capture are strongly correlated, so do not split at random.
Where a file has a `recording_id`, group by `file_path` + `recording_id` and keep
each group on one side of the split; `recording_id` alone is not unique, because
`sub_ghz` and `lte_wifi_dvbt` reuse the same run indices in every source file. In
`its_band` a group is one 1,100-sample observation, i.e. at most two windows. The
simulated files (`rml2016_10a`, `rml2018_01a`, `radchar`) have no `recording_id`.

In `nist_wifi_bluetooth.h5` the 5 GHz indoor and outdoor files were recorded
simultaneously and share `recording_id` values; `file_path` and `environment` tell
them apart.

## Related resources

* Aggregated collection: 10.5281/zenodo.22876704
* Preprocessing pipeline: https://github.com/xandervanparys/RadioPile (10.5281/zenodo.23174413)
* Data descriptor: listed under this record's related works on Zenodo once published
