#!/usr/bin/env python3
"""
RadioPile baselines under a group-aware split — the figures the paper reports.

The random-window scripts divide windows at random, so windows cut from one
recording land on both sides of the train/test boundary. This script runs
either baseline with that one thing changed: whole groups are assigned to
train, validation or test, stratified by class and source dataset. A group is
the finest unit of provenance a source records — a capture file, a capture, an
observation, or for the simulated sources a single window (see README.md). The
windows selected, the model and every training setting are imported from the
random-window scripts, so the difference in test accuracy is attributable to
the split.

    python ml-baselines/group_split.py technology
    python ml-baselines/group_split.py modulation

Input is resolved as for the random-window scripts (or --h5). Outputs go to
ml-baselines/results/<baseline_name>_group_split/ (or under
RADIOPILE_BASELINE_OUT if set, or --output-dir) and match those scripts' outputs;
training_summary.json additionally records the split per (class, source) and
the test accuracy per source dataset.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modulation_classification as modulation  # noqa: E402
import technology_recognition as technology  # noqa: E402
from _common import run_baseline  # noqa: E402

# Mean test accuracy of three runs under the group-aware split (seed 42).
REFERENCE_ACCURACY = {"technology": 0.948, "modulation": 0.673}

BASELINES = {
    "technology": (technology, "technology_recog_cnn", "technology_recognition"),
    "modulation": (modulation, "modulation_recog_cnn", "modulation_classification"),
}


def main(task: str, h5_path=None, output_dir=None) -> float:
    config, model_name, output_name = BASELINES[task]
    return run_baseline(
        task=config.TASK,
        model_name=model_name,
        output_name=f"{output_name}_group_split",
        reference_accuracy=REFERENCE_ACCURACY[task],
        expected_num_classes=config.NUM_CLASSES,
        subsample=config.SUBSAMPLE,
        max_per_class=config.MAX_PER_CLASS,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        val_split=config.VAL_SPLIT,
        test_split=config.TEST_SPLIT,
        seed=config.SEED,
        min_snr=getattr(config, "MIN_SNR", None),
        h5_path=h5_path,
        output_dir=output_dir,
        split="group",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=sorted(BASELINES), help="baseline to run")
    parser.add_argument("--h5", type=Path, default=None,
                        help="aggregated HDF5 to train on (default: resolved as described above)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="directory for the outputs (default: as described above)")
    args = parser.parse_args()
    main(args.task, h5_path=args.h5, output_dir=args.output_dir)
