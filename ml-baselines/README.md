# RadioPile baselines

Two single-task 1D CNN baselines on the aggregated corpus, there to show that the
released schema trains directly, not to characterize what can be learned from it.
The figures reported in the RadioPile paper use a **group-aware split**:

```bash
pip install -r requirements.txt -r requirements-ml-baselines.txt

python ml-baselines/group_split.py technology     # → 94.8 % over 10 classes
python ml-baselines/group_split.py modulation     # → 67.3 % over 32 classes
```

The training configuration is written into the scripts as literal constants.
`--h5` and `--output-dir` override where the aggregate is read from and where the
results are written.

The pinned `tensorflow==2.21.0` wheel does not bring the NVIDIA libraries with
it, so on a machine with a GPU it logs `Cannot dlopen some GPU libraries` and
trains on CPU. To train on an NVIDIA GPU under Linux, install the same version
with its CUDA extra instead:

```bash
pip install -r requirements.txt 'tensorflow[and-cuda]==2.21.0'
```

The reported figures are CPU runs; a GPU run should land within their range, not
on them (see [On exact reproducibility](#on-exact-reproducibility)).

These are **baselines, not pipeline code**. Nothing under `pipeline/`,
`datasets/`, `metadata/`, or `scripts/` imports them, and the repository's test
suite does not need TensorFlow to run.

## Results

Three runs per task (CPU, TensorFlow 2.21). All three use the same split, so the
spread is training nondeterminism, not a different draw of the data.

| Task | Test windows | Accuracy, per run | Mean | Macro recall, per run | Mean |
|---|---|---|---|---|---|
| technology (10 classes) | 13,433 | 94.08 / 96.62 / 93.76 % | 94.82 % | 91.83 / 96.78 / 91.16 % | 93.26 % |
| modulation (32 classes) | 38,781 | 66.68 / 67.68 / 67.56 % | 67.31 % | 53.10 / 54.07 / 53.91 % | 53.69 % |

Read the two modulation columns together. 64QAM and 256QAM, which come almost
entirely from the NIST WiFi captures, are 39 % of the test windows and are
recalled at 92 % and 96 %; several high-order constellations that exist only in
the synthetic sources are near chance. Accuracy weights the former, macro recall
does not.

Test accuracy by source dataset, mean of the three runs:

| Task | `nist_wifi_bluetooth` | `lte_wifi_dvbt` | `its_band` | `sub_ghz` | `rml2016_10a` | `rml2018_01a` |
|---|---|---|---|---|---|---|
| technology | 99.49 % | 92.69 % | 89.72 % | 87.11 % | | |
| modulation | 96.13 % | | | 76.01 % | 56.19 % | 41.90 % |

The modulation test set holds 19,243 windows with an SNR label (the two RML
sources, all 26 levels from −20 to +30 dB) and 19,538 without one (NIST and
`sub_ghz`). No SNR floor is applied.

Row-normalized confusion matrices, averaged over the three runs, are in
`figures/cm_technology_grouped.*` and `figures/cm_modulation_grouped.*`, with the
per-run count matrices beside them in `*_grouped_data.npz`.

## What the split holds out

Windows are assigned to train, validation and test in whole groups, stratified by
class and source dataset, so that no group is divided. A group is the finest unit
of provenance a source records:

| Source | A group is | Consequence |
|---|---|---|
| `sub_ghz`, `lte_wifi_dvbt` | one raw capture file | whole recordings are held out |
| `nist_wifi_bluetooth` | one one-second capture; the 5 GHz indoor and outdoor captures taken at the same moment form one group and stay together | whole captures are held out, but every capture in a file comes from one recording session |
| `its_band` | one 1,100-sample observation (one or two windows) | the two windows of an observation stay together; each class still comes from a single source file |
| `rml2016_10a`, `rml2018_01a` | one window | equivalent to a random split; these sources record no grouping |

The group key is `source_dataset` + `file_path` + `recording_id`. `recording_id`
alone is not enough: `sub_ghz` and `lte_wifi_dvbt` reuse the same run indices in
every file.

So the result supports this much: labels are learnable when no test window shares
a capture or observation with a training window. It does **not** estimate
performance on an unseen recording session, site, device or source dataset —
every source appears in all three partitions.

Two further cautions:

* **Some strata are very small.** Sigfox, the `sub_ghz` Noise class and BFSK have
  five capture files each, split 3 / 1 / 1. Sigfox test recall was 50 %, 99 % and
  42 % across the three technology runs, all on the same single held-out file.
* **One draw of the split.** All figures come from seed 42. With one file held
  out per small class, a different draw can move them by more than the
  run-to-run spread shown here.

## Configuration

| Setting | Value |
|---|---|
| Model | 1D CNN — Conv1D(32,7) → MaxPool(2) → Conv1D(64,5) → MaxPool(2) → Conv1D(128,3) → GAP → Dense(128) → Dropout(0.3) → softmax |
| Parameters | 53,290 (10 classes) / 56,128 (32 classes) |
| Subsample | 5% per class, then capped at 50,000 windows per class |
| Split | group-aware, target 70 / 15 / 15; actual 69.1 / 15.3 / 15.6 (technology) and 70.1 / 14.9 / 15.0 (modulation) |
| Epochs | up to 60; early stopping on validation accuracy, patience 10 |
| Checkpoint | weights with the best validation accuracy |
| Batch size | 256 |
| Optimizer | Adam, lr 1e-3, halved after 3 epochs without validation-loss improvement |
| Class weights | Inverse frequency |
| Seed | 42, for subsampling, splitting and TensorFlow |
| SNR floor | None |

Subsampling and capping come first and the split second, so the split divides the
selected windows, not the corpus. The technology runs stopped early at 27 to 29
epochs; the modulation runs ran all 60.

Labels are read from `meta/technology_type` and `meta/modulation_type`
directly, so neither baseline depends on the `PRIMARY_LABEL_TYPE` chosen when
the aggregate was built. Windows whose label is `unknown` are excluded (in the
modulation task this covers the Noise windows), which leaves four sources per task.

The per-class cap matters: without it a single source (NIST WiFi contributes
roughly 20M windows) dominates the sample. It does not remove the imbalance
inside a class — after capping, 97 % of the WiFi windows are still NIST.

## Random-window split

`technology_recognition.py` and `modulation_classification.py` run the same
configuration with windows divided at random. Three runs each gave 97.79 %
(technology) and 68.99 % (modulation). Under that split 95.7 % of the technology
test windows share a recording with the training set, so those figures are
within-recording numbers and are not the ones the paper reports. The scripts are
kept because the group-aware runs reuse their configuration and because the
difference between the two splits is itself informative.

## Input

The aggregated HDF5 is located in this order:

1. `--h5`, if given
2. `$RADIOPILE_H5`, if set
3. `release/v1.0/aggregated/radiopile_v1.0.h5` — the released artifact
4. `data/aggregation/processed/aggregated.h5` — a local build

To run against a copy elsewhere:

```bash
python ml-baselines/group_split.py technology --h5 /mnt/data/radiopile_v1.0.h5
```

## Output

Written to `--output-dir` if given (used as is), else to
`$RADIOPILE_BASELINE_OUT/<baseline_name>/` if that is set, else to
`ml-baselines/results/<baseline_name>/`. Each run writes `best_model.keras`,
`confusion_matrix.png`, `confusion_matrix.npy`, `loss_curves.png`,
`label_info.json`, `training_history.json`, and `training_summary.json`. For a
group-aware run the summary also records the groups and windows per (class,
source) in each partition and the test accuracy per source dataset.

## On exact reproducibility

Seeds are fixed for data subsampling, the split, and TensorFlow's global RNG, so
the *data* each run sees is deterministic. The split sizes are a useful first
check that a reproduction is on track: 59,664 / 13,217 / 13,433 (technology) and
180,954 / 38,344 / 38,781 (modulation).

Training itself is only bitwise reproducible on identical hardware and library
versions: kernel non-determinism and cuDNN algorithm selection on GPU,
non-deterministic reduction order across threads on CPU. The technology task is
the more sensitive of the two, because one small class can swing on a single
held-out file; expect a reproduction to land within the range of the three runs
above rather than on their mean.
