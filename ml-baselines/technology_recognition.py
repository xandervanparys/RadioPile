#!/usr/bin/env python3
"""
RadioPile baseline — technology recognition, random-window split.

Reference result: 97.8% test accuracy (mean of three runs) over 10 technology
classes, using a ~53k-parameter 1D CNN trained on a 5%
stratified per-class subsample of the released aggregate, capped at 50,000
windows per class.

Run it with no arguments for the reference configuration:

    python ml-baselines/technology_recognition.py

Windows are divided at random here, so windows cut from one recording fall on
both sides of the train/test boundary. The figures the paper reports come from
group_split.py, which reuses every constant below and changes only the split.

Input resolution, in order:
  1. the --h5 argument, if given
  2. the RADIOPILE_H5 environment variable, if set
  3. release/v1.0/aggregated/radiopile_v1.0.h5
  4. data/aggregation/processed/aggregated.h5

Outputs (in ml-baselines/results/technology_recognition/, or under
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
TASK = "technology"
REFERENCE_ACCURACY = 0.978           # 97.8% test accuracy, 10 classes (mean of three runs)
NUM_CLASSES = 10

SUBSAMPLE = 0.05                 # 5% stratified sample per class
MAX_PER_CLASS = 50_000           # cap so large sources cannot dominate
EPOCHS = 60
BATCH_SIZE = 256
VAL_SPLIT = 0.15                 # 70/15/15 train/val/test
TEST_SPLIT = 0.15
SEED = 42
# ─────────────────────────────────────────────────────────────────────────────


def main(h5_path=None, output_dir=None) -> float:
    """
    Run the random-window baseline. Both arguments are optional; passing
    neither resolves input and output as documented above.
    """
    return run_baseline(
        task=TASK,
        model_name="technology_recog_cnn",
        output_name="technology_recognition",
        reference_accuracy=REFERENCE_ACCURACY,
        expected_num_classes=NUM_CLASSES,
        subsample=SUBSAMPLE,
        max_per_class=MAX_PER_CLASS,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        seed=SEED,
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
