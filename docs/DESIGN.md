# Design

Everything that is true across sources: the label schema, the pipeline, the HDF5
layout, the decisions taken and why, and the questions still open. This file should
change rarely.

Statements about code were checked against `pipeline/`, `datasets/aggregation/` and
the built files of 2026-10-12. When the code and this document disagree, the code is
right.

Sections: [1. Schema](#1-schema) · [2. Pipeline](#2-pipeline) ·
[3. HDF5 format](#3-hdf5-format) · [4. Train/test splitting](#4-traintest-splitting) ·
[5. Licensing](#5-licensing) · [6. Open questions](#6-open-questions)

Related: [`ADDING_A_DATASET.md`](ADDING_A_DATASET.md) is the procedure for integrating
a new source; [`NOISE_FILTERING.md`](NOISE_FILTERING.md) is the how-to for choosing a
noise threshold.

---

## 1. Schema

### 1.1 Schema minimalism

RadioPile's unified schema does not add fields to enforce cross-domain coherence when
the information is recoverable from existing ones. Where a field means something
slightly different per task, the per-domain meaning is documented and users who need
one coherent representation reconstruct it from `task_type` plus the field value.

Three reasons:

1. **RF taxonomy has no coherent cross-domain grouping.** Modulation datasets group
   into modulation families, technology datasets into physical-layer waveforms, radar
   into intra-pulse families. One schema-level interpretation would need an invented
   parent taxonomy or would leave most of the corpus `"unknown"`.
2. **Integration burden scales with schema size.** Every field is another value a
   future contributor must determine, justify and validate. For the intended uses
   (per-task supervised classification, IQ pre-training, cross-dataset evaluation) a
   finer schema adds little.
3. **The information is recoverable.** A consumer that needs
   `(field_name, field_value)` pairs with one meaning across the corpus can map the
   field name per task in a few lines:

   ```python
   SEMANTIC_KEY_BY_TASK = {
       "modulation_recognition": "modulation_family",
       "technology_recognition": "phy_layer",
       "radar_classification":   "radar_intrapulse_family",
   }
   semantic_key = SEMANTIC_KEY_BY_TASK[task_type]
   semantic_value = waveform_type
   ```

**When to reconsider.** A split becomes appropriate when a concrete use case cannot
tolerate the reconstruction step: a production pipeline without `task_type` at
inference time, or a metadata query interface whose users cannot be expected to do
it. Until then, minimalism holds.

### 1.2 Two-level labels: a fine target below `waveform_type`

Sources label along different axes, so there are three fine targets, one per task
family. `waveform_type` is the coarse family one level above whichever applies, and is
set for every window, so every task has the same two-level shape:

| Domain | Fine target | Coarse `waveform_type` | Example |
|---|---|---|---|
| Modulation | `modulation_type` | modulation family | 16QAM → QAM |
| Technology | `technology_type` | physical-layer waveform | LTE → OFDM, LoRa → CSS |
| Radar | `radar_signal_type` | intra-pulse modulation family | barker_code → phase_modulated |

The per-domain meaning differs (a true superclass for modulation, a correlated
physical-layer attribute for technology, the intra-pulse family for radar), but the
field never holds a fine label in any domain. `waveform_type` describes the signal as
transmitted: a 64QAM window from an OFDM technology is `OFDM`, not `QAM`.

A fine target a window has no label for is `"unknown"`. v1.0 holds 10 technology
classes, 32 modulation classes and 5 radar signal types; `unknown` is the not-labeled
value on each axis, not a class.

**`waveform_type` values in v1.0** (15): `OFDM`, `SC-FDM`, `PSK`, `QAM`, `FSK`, `APSK`,
`ASK`, `PAM`, `AM`, `FM`, `CSS`, `noise`, and the three radar families below. Which
classes of which source map to each is in that source's section of `DATASETS.md`.

**Radar grouping.** The radar literature groups pulse-compression waveforms by how
the carrier is modulated within a pulse: frequency-modulated (LFM / chirp),
phase-modulated (Barker, polyphase Barker, Frank and related codes) and unmodulated
(for example a coherent train of constant-carrier pulses). See Levanon & Mozeson,
*Radar Signals*, and Richards, *Fundamentals of Radar Signal Processing*. The mapping
is `RADAR_FINE_TO_COARSE` in `pipeline/schemas.py`:

```text
linear_frequency_modulated → frequency_modulated
barker_code                → phase_modulated
polyphase_barker_code      → phase_modulated
frank_code                 → phase_modulated
coherent_pulse_train       → unmodulated
```

The fine class stays in `radar_signal_type` and `source_label`.

### 1.3 Two modulation fields and the OFDM asymmetry

One field cannot carry both the access scheme and the constellation. For an OFDM
technology the multiplexing scheme (OFDM) is what the air interface uses, while the
subcarrier modulation (BPSK to 256QAM) follows the MCS and is usually not recorded.
So:

- `modulation_type` holds the specific data modulation (`"BPSK"`, `"64QAM"`, `"CSS"`),
  in the same label space as the modulation datasets. For OFDM technologies it is
  `"unknown"` unless MCS metadata exists.
- `waveform_type` holds the family and does not depend on MCS (`"OFDM"`, `"PSK"`,
  `"QAM"`, `"FSK"`, …).

**Consistency rule.** `waveform_type` always holds the family, never the variant:
`"BPSK"` → `"PSK"`, `"16QAM"` → `"QAM"`. A field whose abstraction level shifts
between datasets corrupts aggregation silently.

**The asymmetry.** For single-carrier signals the two fields form a hierarchy
(PSK → BPSK, QAM → 16QAM). For OFDM technologies they do not: `"OFDM"` is the
multiplexing scheme and `"64QAM"` the modulation on each subcarrier, two orthogonal
properties of one signal.

**Alternative considered.** Split into `multiplexing_scheme` (OFDM, OFDMA, SC-FDM,
DSSS, FHSS, single-carrier), `modulation_family` and `modulation_type`, each
coherent across domains.

**Why not in v1.0.** Subcarrier modulation follows the MCS index, which varies per
packet, and most technology sources do not record MCS per sample, so
`modulation_type` is `"unknown"` for most OFDM windows anyway. A `multiplexing_scheme`
field correctly set to `"OFDM"` would not make the other two fields less unknown; the
richer schema only pays off when MCS is known, which is rare in v1.0.

**Triggers for adding it.** A source that provides per-window MCS for OFDM captures
at scale, or a downstream use that must query "OFDM with unknown subcarrier
modulation" versus "single-carrier with known modulation" directly at field level.
Adding the field later is purely additive: existing fields stay valid and datasets
that omit it fill `"unknown"`. Until one of these applies, the two-field design
stands.

### 1.4 Multi-target metadata, and why `/labels` must not mix taxonomies

**Decision.** Every target is stored in per-window metadata (`technology_type`,
`modulation_type`, `radar_signal_type`, `waveform_type`). The primary `/labels` column
of the aggregate is chosen at build time by `PRIMARY_LABEL_TYPE`, which is
`"technology"` in v1.0.

**Rule.** A window with no label for the primary task gets `"unknown"`, never a
fallback to its own `source_label`. Falling back would put `"WiFi"`, `"BPSK"` and
`"barker_code"` as peer classes in one head. `"unknown"` is the honest statement and
is excluded or masked in the loss.

**Routing.** Each source declares in `datasets/registry.py` what its class label
represents; the aggregation uses that to decide which fine target the label fills.

| `task_type` | Label fills | Other targets |
|---|---|---|
| `technology_recognition` | `technology_type` | `modulation_type` from the source's own field where set |
| `modulation_recognition` | `modulation_type` | `technology_type = "unknown"` |
| `radar_classification` | `radar_signal_type` | `technology_type` and `modulation_type` are `"unknown"` |

`waveform_type` is never derived at aggregation: it is read from the field each
source's build already wrote.

**Implication.** For multi-task training, route each window by `task_type` and read
the label from the matching metadata field. `/labels` is only meaningful when all
sources share a task; the baselines read the metadata fields directly.

### 1.5 A canonical label registry

Class names drift across sources (`"LTE"` / `"lte"`, `"QAM16"` / `"16QAM"` /
`"16-QAM"`), and aggregation routes on exact string match, so drift produces
duplicate classes. `CANONICAL_LABELS` in `pipeline/schemas.py` lists the canonical
class names of each task family's label space; in v1 it is reference-only, not
enforced at runtime. Modulation values that a technology source sets as metadata
(`BFSK` and `CSS`, from `sub_ghz`) are not in it. For the QAM case above the
canonical names are `"16QAM"` and `"64QAM"`.

Related names that are not identical are deliberately kept apart:

| Names | Why kept separate |
|---|---|
| `PAM4` / `4ASK` | both 4-level amplitude modulation, different channel models and parameters |
| `WBFM` / `FM` | different simulation bandwidth assumptions |
| `GFSK` / `GMSK` | GMSK is a special case of GFSK (BT = 0.3, h = 0.5) |

Formatting duplicates get fixed; semantic near-duplicates get documented and left to
the user.

### 1.6 Technology label versus protocol variant

**Rule.** A variant goes in a metadata field when it is a parameter of a technology
already present and a recognizer would be expected to classify all variants as one
class. It becomes a top-level label only when a practitioner would call it a
different technology.

| Family | Label | Variant field |
|---|---|---|
| LoRa spreading factors | `"LoRa"` | `spreading_factor` |
| IEEE 802.11 | `"WiFi"` | `wifi_standard` (`802.11ah`, `802.11n`, `802.11ac`) |
| IEEE 802.15.4 | `"IEEE 802.15.4"` | `ieee802154_variant` (`g`, `SUN-OFDM`) |

A source that emits `"IEEE 802.11ah"` as its own class while others emit `"WiFi"`
has misclassified a variant as a technology.

**Cost.** The grouped label is internally diverse: 802.11ah (sub-GHz, ~1 MHz) and
802.11n (2.4/5 GHz, 20 MHz) look very different after windowing, and 802.15.4g
(BFSK) and SUN-OFDM (OFDM) are different modulation families. Whether that helps or
hurts a classifier is an open question (§6). The variant fields are stored in the
per-dataset files, so per-variant subsets remain constructible.

### 1.7 Fill values

Fields a source cannot populate are still written, so every `/meta` dataset has
length N. The fill is declared per field in the schema.

| Kind | Fill | Meaning | Examples |
|---|---|---|---|
| Categorical string | `"unknown"` | no label for this axis | `technology_type`, `modulation_type`, `waveform_type`, `radar_signal_type` |
| Free-text string | `""` | not recorded | `recording_id`, `location`, `wifi_standard`, `ieee802154_variant` |
| Float | `NaN` | not measured | `snr_db`, `center_frequency_hz`, `gain_db` |
| Integer | `-1` | sentinel | `mcs`, `channel`, `spreading_factor`, `sample_index` |
| Boolean | `False` | did not occur | `was_padded`, `was_cropped` |

A fill must not read as a legitimate value: `0` for a frequency or gain would be
silently plausible, `NaN` and `-1` are not. Store uncertainty as `"unknown"` rather
than forcing a plausible value. Where a source departs from the convention, its
section in `DATASETS.md` says so.

### 1.8 Controlled vocabularies and identifiers

**Enums for anything that drives routing.** `task_type` and `label_represents` are
`TaskType` and `LabelRepresents` enums in `pipeline/schemas.py`, subclassing `str` so
they serialize unchanged. As plain strings, a typo like `"radar_clasification"` would
be accepted. `DatasetEntry.__post_init__` coerces valid strings and raises
on anything else, at construction rather than hours into a build.

**`recording_id` is a string.** Unified to `H5_STR_DTYPE`. Grouping works on any
hashable type and arithmetic on a recording ID is never meaningful. String is the
lossless direction: integer run indices convert trivially, but capture keys and
composite keys do not become integers without a lookup table.

---

## 2. Pipeline

### 2.1 Architecture

```
raw files (.bin / .mat / .pkl / .h5)
  → load                 per-dataset iterator in datasets/<name>/build_dataset.py,
                         using pipeline/io.py and pipeline/windowing.py
  → window               hop = window size, no overlap; pre-windowed sources pass through
  → noise filter         optional, on raw power (§2.6)
  → normalize            unit power (§2.5)
  → write                pipeline/hdf5_writer.py → data/<name>/processed/<name>.h5
per-dataset files
  → aggregate            datasets/aggregation/build_dataset.py: adapt window size (§2.4),
                         route labels (§1.4), unify metadata → one file
```

**The order window → noise filter → normalize is fixed.** The filter thresholds raw
power in dB against a detected noise floor, which is meaningless after unit-power
normalization, so the order is written into each dataset's window iterator and is not
configurable.

**Registry with dynamic config loading.** `datasets/registry.py` holds only registry
facts: paths, build module, task type. Preprocessing parameters and metadata schemas
are read from each dataset's `config.py` on demand, so adding a dataset changes no
aggregation code and there is one source of truth per dataset. The registry does not
remove the need for dataset-specific extraction logic (modulation from MCS for one
source, fixed for another); a plugin pattern would be needed for full extensibility.

**A per-dataset file sits between raw data and the aggregate.** Per-dataset files are
built at native window size and the aggregate is assembled from them, so re-aggregating
with a different target or a different selection of sources needs no rebuild from raw.
The cost is that window sizes are adapted at aggregation time (§2.4).

### 2.2 Frozen v1 defaults, and explicit before generic

Padding strategy (`zero`), crop policy (`start`), normalization (`unit_power`),
storage dtype (`float32`) and target window size (500) are fixed for v1. Every
artifact carries `script_version` and `schema_version`, and each build writes a
`.h5.config.json` snapshot with the git commit, suffixed `-dirty` if the tree had
uncommitted changes. Less flexibility, in exchange for results comparable across time
and users.

v1 favors explicit, readable code over premature abstraction. Config-driven
metadata extraction, re-windowing from raw during aggregation and advanced padding
strategies were deferred rather than half-built.

### 2.3 Reading raw data

**Source dtype is load-critical for `.bin`.** `.bin` carries no format information;
GNU Radio writes complex64 by default, int16 is common for cheaper SDRs, and the
extension tells you nothing. A dtype mismatch does not raise: it yields data that
looks like noise, and I-channel excursions beyond the expected range are its
signature. Every config declares `SOURCE_DTYPE`. For `.bin` it is functional (passed
to `np.memmap`); for `.mat`, `.pkl` and `.h5` the container knows its dtype and the
constant is documentation. The case that taught this is in `DATASETS.md`,
`lte_wifi_dvbt`.

**Reinterpret complex dtypes instead of splitting them.**
`np.column_stack([raw.real, raw.imag])` on a memmap builds two intermediate arrays
before the final one. `raw.view(np.dtype(f"f{itemsize // 2}")).reshape(-1, 2)`
reinterprets the bytes in place, valid because complex64/128 store adjacent `[I, Q]`
pairs, so the only allocation is the single conversion to the output dtype. That
conversion does read the whole file into memory; windows are not streamed from disk.

**Never infer capture parameters from file properties.** Sample rate is independent
of file size, and files at different rates can be the same size. Take capture
parameters from filename conventions, capture logs or explicit config, never from
size or location on disk.

**Exclude classes whose core metadata cannot be verified.** Window duration sets the
physical meaning of every axis. Label noise is recoverable; wrong axis calibration
corrupts everything downstream silently. So a class whose sampling rate cannot be
confirmed is excluded rather than assumed, and the evidence that would resolve the
exclusion is recorded so the recovery path is mechanical. The case is in
`DATASETS.md`, `its_band`.

**Verify public documentation, and read the companion report.** Public RF datasets
often describe what was captured without the signal structure needed to integrate
safely. Filenames are not content: a file named for one technology may contain two.
Inspect spectrograms and I/Q plots before integrating, and exclude mixed-signal files
where a single label is assumed. The reverse also applies: a dataset DOI points at a
repository, while the method usually lives in a separate report with its own DOI.
Exhaust it before declaring a parameter undocumented, since an `"unknown"` that was
knowable is a correctness gap. Both cases are in `DATASETS.md`, `nist_wifi_bluetooth`.

### 2.4 Window-size adaptation

Sources are windowed at different lengths; the aggregate needs one.

**Decision.** Per-dataset files keep their native window length. Adaptation to
`TARGET_WINDOW_SIZE = 500` happens at aggregation: `PADDING_STRATEGY = "zero"` appends
zeros to shorter windows, `CROP_POLICY = "start"` keeps the first 500 samples of longer
ones. Each source's native length and what happens to it are in its section of
`DATASETS.md`.

**Trade-off.** Padding adds artificial zeros, cropping discards signal. That is why
every adapted window records `was_padded`, `padding_strategy`, `was_cropped`,
`crop_policy` and `original_window_size`. A mask over the real samples of a padded
window follows from the last of these:

```python
mask = np.arange(500) < original_window_size
```

**Two sources of padding, one tracked.**

1. *Per-dataset build:* the tail of a raw capture that does not fill a window is
   zero-padded when `PAD_LAST` is set. At most one window per source file. This is not
   recorded in the built files: the windowing step reports a `padded` flag to the
   iterator, but no v1 schema stores it.
2. *Aggregation:* every window from a shorter-native-size dataset is padded. This is
   the one tracked, through the fields above, because sequence models need the mask.

**Padded windows are not at unit power.** Normalization runs at each source's native
window length, and padded windows are deliberately not renormalized afterward,
because that would amplify the real samples to compensate for the zeros. A padded
window therefore has mean power `original_window_size / 500` over all 500 samples and
1.0 over its real samples. Cropped windows are renormalized and sit at unit power. The
validator's unit-power check runs on the per-dataset files only and is skipped for the
aggregate for this reason.

**Prefer the highest available sampling rate.** Where a source offers several rates,
take the highest. Decimation is well defined and anyone can apply it downstream;
upsampling recovers nothing. Choosing a lower rate because its observations happen to
need less padding optimizes for a preprocessing artifact at the cost of signal detail.

**Prefer splitting to cropping.** When an observation is longer than the target,
cutting it into several windows loses less than cropping it to one. Emit the metadata
that lets users regroup what was split (§4).

### 2.5 Normalization

`normalize_window()` in `pipeline/preprocessing.py` implements five modes, selected
per dataset by `NORMALIZE_MODE`.

| Mode | Operation | Scope |
|---|---|---|
| `unit_power` | `x / sqrt(mean(I² + Q²))` | whole window; preserves phase |
| `max_magnitude` | `x / max(abs(x))` | whole window, I and Q together |
| `zero_mean_unit_variance` | `(x − mean) / std` | per channel |
| `mapminmax` | min-max scaling to [−1, 1] | per channel |
| `none` | identity | |

**Decision: every window is unit-power normalized, across all datasets.** Per-dataset
normalization would preserve each source's amplitude characteristics; unified
normalization is what makes the corpus usable as one training set and makes
cross-dataset comparison fair. It also removes absolute power differences that only
reflect arbitrary hardware gain.

**Normalization does not remove SNR information.** SNR is a ratio; dividing a window
by one scalar scales signal and noise alike. SNR lives in the structure of the window,
which is scale-invariant. The measurement that checks absolute power was not an extra
cue is in `DATASETS.md`, `rml2016_10a`.

**Normalization at aggregation.** The aggregation applies `unit_power` a second time
to windows already normalized in their own build. For unchanged windows that second
pass has no effect; cropped windows are brought back to unit power over the 500
samples kept; zero-padded windows are skipped (§2.4).

### 2.6 Noise-floor detection

Per-window power is `mean(I² + Q²)`, compared in dB against a threshold.
`NOISE_POWER_THRESHOLD_DB` is `None` (no filtering), a number (fixed threshold) or
`"auto"` (threshold = detected floor + `margin_db`). Three detection methods exist:

| Method | Floor estimate |
|---|---|
| `percentile` | the Nth percentile of window power |
| `histogram` | the mode of the lowest `noise_fraction` of window powers |
| `statistical` | mean − `std_multiplier` × std of the lowest `noise_fraction` |

Filtering is for packet-based signals with idle gaps; sources with continuous signals
or with SNR as a design variable set the threshold to `None`. The sources that filter
use `percentile` with `percentile = 1.0`, `margin_db = 6.0` and at most 10,000 windows
for detection. The function's own default percentile is 5.0, so the config value
matters.

**Filtering parameters do not transfer between datasets without checking.** The auto
method was developed on one kind of capture and then applied to packet-based 802.11
captures with inter-frame gaps. The same margin transferred, but only by
verification: where the gap between floor and signal is wide almost any margin works,
and where it is narrow a few dB either way keeps a different population of windows.
Before a full build of any packet-based protocol, sweep `margin_db` on a sample and
compare against an independent reference. The sweep that established this, with its
numbers, is in `DATASETS.md`, `nist_wifi_bluetooth`; how to run one is in
[`NOISE_FILTERING.md`](NOISE_FILTERING.md).

---

## 3. HDF5 format

### 3.1 Per-dataset files

```
<dataset>.h5
├── data          [N, 2, W]  float32   IQ windows; data[:, 0] is I, data[:, 1] is Q
├── labels        [N]        int64     class index into label_names
├── label_names   [C]        string    class names
├── meta/
│   ├── file_path [N]        string    raw file the window came from; written for every dataset
│   └── ...                            the fields of the dataset's WINDOW_METADATA_SCHEMA
└── attributes
    ├── dataset, description           name and free-text description
    ├── schema_version                 "1.0.0"
    ├── script_version                 build script version
    ├── window_size, hop_size          samples
    ├── normalization_mode             "unit_power"
    ├── data_dtype                     "float32"
    ├── total_windows                  N
    ├── license, license_note          license identifier (SPDX where one exists) and its scope
    ├── dataset_url                    where the source data is published
    ├── doi, related_identifiers       DOI of the record the file is published in, and
    │                                  of its sibling records
    ├── sample_rate_hz                 Hz; only where one rate holds for the whole file
    └── ...                            dataset-specific
```

Attributes that describe the same thing carry the same name in every file.
`sample_rate_hz` is absent from `lte_wifi_dvbt`, which mixes 1 and 10 Msps captures and
stores the rate per window, and from the two simulated RadioML sources, which have no
physical rate.

The iterators call the source field `source_file`; on disk it is always `file_path`.
Which `/meta` fields exist differs per dataset; each section of `DATASETS.md` lists
them.

### 3.2 Aggregated file

Same top-level layout, with one metadata schema for all sources (17 fields):

| Field | Type | Content |
|---|---|---|
| `source_dataset` | string | `"sub_ghz"`, `"rml2016_10a"`, … |
| `source_label` | string | the window's class in its source file |
| `task_type` | string | `"technology_recognition"`, `"modulation_recognition"`, `"radar_classification"` |
| `technology_type` | string | fine target, or `"unknown"` |
| `modulation_type` | string | fine target, or `"unknown"` |
| `radar_signal_type` | string | fine target, or `"unknown"` |
| `waveform_type` | string | coarse family; set for every window |
| `recording_id` | string | as in the source file; `""` where the source has none |
| `file_path` | string | as in the source file |
| `original_window_size` | int32 | window length before adaptation |
| `was_padded` | bool | |
| `padding_strategy` | string | `"zero"` or `"none"` |
| `was_cropped` | bool | |
| `crop_policy` | string | `"start"` or `"none"` |
| `sample_rate_hz` | float64 | NaN where the source has no physical rate |
| `center_frequency_hz` | float64 | NaN for simulated sources |
| `snr_db` | float32 | NaN where the source has none |

Attributes: the common ones of §3.1 except `window_size`, `hop_size`, `dataset_url`
and `sample_rate_hz`, which describe a single source; `doi` is the aggregate's own
record. In addition: `source_datasets` (comma-separated
string), `target_window_size` (500), `padding_strategy` (`"zero"`), `crop_policy`
(`"start"`), `primary_label_type` (`"technology"`) and `total_windows` (25,836,227).

`/labels` follows `PRIMARY_LABEL_TYPE` (§1.4): `label_names` holds the 10 technology
classes plus `unknown`.

Source-specific fields such as `environment`, `location`, `gain_db`, `mcs` or
`wifi_standard` are not carried into the aggregate. They can be recovered by row
position, because the aggregate keeps each source's rows in the order of its own file.

### 3.3 Using the aggregated file

The aggregated HDF5 is a store of all sources with shared metadata, not a single
benchmark. Pick the task by filtering on a `/meta` field; `/labels` is only a
convenience view of one of them.

| Task | Field | Keep windows where | Result in v1.0 |
|---|---|---|---|
| Technology recognition | `technology_type` | value ≠ `"unknown"` | 22,060,323 windows, 10 classes |
| Modulation classification | `modulation_type` | value ≠ `"unknown"` | 24,248,963 windows, 32 classes |
| Radar signal classification | `radar_signal_type` | value ≠ `"unknown"` | 1,000,000 windows, 5 classes |
| Waveform family | `waveform_type` | always populated | 15 values (§1.2) |

The reference baselines in `ml-baselines/` use the first two filters. The `"unknown"`
values are expected: modulation and radar sources carry no technology label, radar and
most OFDM captures carry no modulation label.

`/data` is 103 GB of the file's 117.5 GB. Select indices from the metadata first, then read IQ in sorted
batches:

```python
import h5py, numpy as np

with h5py.File("radiopile_v1.0.h5", "r") as f:
    y = f["meta/technology_type"].asstr()[:]
    idx = np.flatnonzero(y != "unknown")
    for s in range(0, len(idx), 8192):
        x = f["data"][idx[s:s + 8192]]        # (B, 2, 500) float32
```

### 3.4 Chunking: why it exists and how the chunk shape is chosen

**Chunking is required, not a tuning choice.** `pipeline/hdf5_writer.py` creates
`data`, `labels` and every metadata dataset empty with `maxshape=(None, ...)` and
grows them with `resize()` on each `flush()`, because files are assembled by
appending batches as windows are produced. HDF5 only allows extendable datasets with
chunked storage, so the incremental build forces chunking. No compression filter is
applied anywhere.

**The chunk shape is the write-batch size.** `data` uses
`chunks=(batch_size, 2, window_size)`, where `batch_size` is the same literal that
sets the flush threshold at each `build_hdf5_dataset` call site: 64 for the seven
per-dataset builds, 4096 for aggregation. `labels` and metadata use
`chunks=(batch_size,)`. They match only because they reuse one parameter.

**Known suboptimality of the aggregate chunk.** With `batch_size=4096` and 500-sample
windows the data chunk is 4096 × 2 × 500 × 4 B ≈ 16 MB. That is large for shuffled
per-window reads during training: one window costs a 16 MB chunk read. The value is
justified on the write side, where it cuts the number of resize-and-write operations
on a 116 GB file. Changing it means rebuilding the aggregate, so it is documented
here rather than altered. Readers should sort their indices and read in batches (§3.3).

**Not to be confused with raw-read chunking.** The `CHUNK_SIZE` / `chunk_size_mb`
constants in the `rml2018_01a`, `radchar` and `aggregation` builders bound how much
of a source file is held in RAM at a time. They are unrelated to the HDF5 `chunks=`
parameter.

---

## 4. Train/test splitting

**Correlated windows are a leakage risk.** Adjacent windows from one recording share
channel conditions, noise floor, hardware state and often the same transmission. A
random split puts some in train and some in test, and accuracy is inflated. Windows
from one recording must stay on one side of the split.

**The pipeline stores the grouping metadata; it does not enforce the split.**
Splitting is a training-time concern. Every dataset that windows continuous recordings
needs a documented group key, and each source's key is in its section of
`DATASETS.md`. Simulated sources, whose examples are independent by construction, have
none.

**`recording_id` alone is never the key.** It is only unique within a source file:
run indices such as `"1"` repeat in every file of a source. In the aggregated file the
group key is the composite

```
source_dataset | file_path | recording_id
```

and a window whose `recording_id` is empty is its own group.

```python
import h5py, numpy as np
from sklearn.model_selection import GroupShuffleSplit

with h5py.File("radiopile_v1.0.h5", "r") as f:
    src = f["meta/source_dataset"].asstr()[:]
    path = f["meta/file_path"].asstr()[:]
    rec = f["meta/recording_id"].asstr()[:]

groups = np.array([
    f"{s}|{p}|{r}" if r else f"{s}|window|{i}"
    for i, (s, p, r) in enumerate(zip(src, path, rec))
])

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(groups, groups=groups))
```

The example needs scikit-learn, which is not in `requirements.txt`.
`GroupShuffleSplit` keeps groups whole but does not stratify, so a class with few
recordings can land entirely on one side. In `ml-baselines/_common.py`,
`load_group_keys()` builds the same key, except that NIST captures recorded
simultaneously in two files share one group, and `group_split_indices()` splits on
it, stratified by class and source; it is what the paper's baselines use.

---

## 5. Licensing

Sources carry different licenses: CC-BY-NC-SA-4.0, CC-BY-4.0 and the NIST data
terms (`NIST-public-domain` in the config). Each dataset's `config.py` declares `LICENSE`, which is written into the
HDF5 root attributes, the config snapshot and the Croissant sidecar. The aggregate's
license is the most restrictive among its inputs, chosen by the explicit ranking
`LICENSE_RESTRICTIVENESS` in `datasets/aggregation/config.py`.
`source_dataset` is a per-window field, not a dataset attribute, so any subset of the
corpus can be traced to the sources it draws on and attributed. The evidence for each
source's license is in its section of `DATASETS.md`.

---

## 6. Open questions

1. How does zero-padding to a common window size affect model performance relative
   to re-windowing from raw?
2. Does grouping protocol variants under one technology label (§1.6) help or hurt?
3. Is unified or per-dataset normalization better for cross-dataset transfer?
4. Should a `multiplexing_scheme` field separate OFDM multiplexing from subcarrier
   modulation (§1.3)?
5. What is the right trade-off between metadata completeness and storage overhead as
   the corpus grows?
6. Should aggregation renormalize at all? With `unit_power` everywhere the second
   pass is harmless (§2.5), but that would not hold if a source used another mode.
