#!/usr/bin/env python3
"""
RadioPile baseline — modulation classification, random-window split.

Reference result: 69.0% test accuracy (mean of three runs) over 32 modulation
classes, using a ~56k-parameter 1D CNN trained on a 5%
stratified per-class subsample of the released aggregate, capped at 50,000
windows per class.

Run it with no arguments for the reference configuration:

    python ml-baselines/modulation_classification.py

Windows are divided at random here, so windows cut from one recording fall on
both sides of the train/test boundary. The figures the paper reports come from
group_split.py, which reuses every constant below and changes only the split.

The 32 classes are the distinct non-"unknown" modulation_type values across the
whole corpus; "noise" windows are excluded alongside "unknown". No SNR floor is
applied — MIN_SNR is left at None so the low-SNR RML
windows, where classes are close to indistinguishable, remain in the task.

Input resolution, in order:
  1. the --h5 argument, if given
  2. the RADIOPILE_H5 environment variable, if set
  3. release/v1.0/aggregated/radiopile_v1.0.h5
  4. data/aggregation/processed/aggregated.h5

Outputs (in ml-baselines/results/modulation_classification/, or under
RADIOPILE_BASELINE_OUT if set, or --output-dir):
    best_model.keras        best checkpoint by validation accuracy
    confusion_matrix.png    row-normalized confusion matrix
    confusion_matrix.npy    raw counts
    loss_curves.png         train/validation loss and accuracy
    label_info.json         per-class counts and class weights
    training_history.json   per-epoch metrics
    training_summary.json   scalars, configuration, and provenance

Requires TensorFlow, which is an optional dependency of this repository:
    pip install -r requirements-ml-baselines.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import run_baseline  # noqa: E402

# ── Baseline configuration (random-window split) ─────────────────────────────
TASK = "modulation"
REFERENCE_ACCURACY = 0.690           # 69.0% test accuracy, 32 classes (mean of three runs)
NUM_CLASSES = 32

SUBSAMPLE = 0.05                 # 5% stratified sample per class
MAX_PER_CLASS = 50_000           # cap so large sources cannot dominate
EPOCHS = 60
BATCH_SIZE = 256
VAL_SPLIT = 0.15                 # 70/15/15 train/val/test
TEST_SPLIT = 0.15
SEED = 42
MIN_SNR = None                   # no SNR floor
# ─────────────────────────────────────────────────────────────────────────────


def main(h5_path=None, output_dir=None) -> float:
    """
    Run the random-window baseline. Both arguments are optional; passing
    neither resolves input and output as documented above.
    """
    return run_baseline(
        task=TASK,
        model_name="modulation_recog_cnn",
        output_name="modulation_classification",
        reference_accuracy=REFERENCE_ACCURACY,
        expected_num_classes=NUM_CLASSES,
        subsample=SUBSAMPLE,
        max_per_class=MAX_PER_CLASS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        seed=SEED,
        min_snr=MIN_SNR,
        h5_path=h5_path,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h5", type=Path, default=None,
                        help="aggregated HDF5 to train on (default: resolved as described above)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="directory for the outputs (default: as described above)")
    args = parser.parse_args()
    main(h5_path=args.h5, output_dir=args.output_dir)
