"""Render publication-ready confusion matrix PDFs from saved .npy + label_info.

Usage:
    python scripts/paper_figures/confusion_matrix_pdf.py \\
        <model_dir> [<model_dir> ...] <out_pdf> <out_npz>

Reads:
    <model_dir>/confusion_matrix.npy  — int64 (C, C)
    <model_dir>/label_info.json       — dict keyed by class name (any order)

Several model directories are combined as the mean of their row-normalized
matrices. They must be runs evaluated on the same test set (identical labels
and per-class supports — true for runs sharing a split seed). Repeated runs
re-predict the same test windows, so they are averaged as rates and never
presented as a larger test set: the figure describes `supports.sum()` windows,
not that number times the run count.

Writes:
    <out_pdf>                         — row-normalized CM, IEEE single-column
    <out_pdf>, suffix .png            — same, 300 dpi raster
    <out_npz>                         — cm_norm (mean over runs of the per-run
                                        row-normalized matrices), cm_norm_std,
                                        cm_per_run (counts, one matrix per run),
                                        supports, accuracy_per_run,
                                        macro_recall_per_run, label_names,
                                        source_dir (";"-joined), n_runs, and
                                        cm_raw (summed counts)

Plotting style mirrors ml-baselines/_common.py :: plot_confusion_matrix:
    matplotlib, Blues colormap, white/black auto-contrast cell values, true on
    Y, predicted on X, rotated X tick labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_run(model_dir: Path) -> tuple[np.ndarray, list[str]]:
    cm_path = model_dir / "confusion_matrix.npy"
    label_path = model_dir / "label_info.json"

    if not cm_path.exists():
        raise FileNotFoundError(cm_path)
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    cm = np.load(cm_path)
    with open(label_path) as f:
        label_info = json.load(f)

    # label_info.json keys are class names; the order must match training_summary
    # ordering. The training script builds `unique_labels = sorted(set(...))`,
    # so both confusion_matrix.npy axes and label_info keys are alphabetical.
    label_names = sorted(label_info.keys())
    if len(label_names) != cm.shape[0] or cm.shape[0] != cm.shape[1]:
        raise ValueError(
            f"Label/CM size mismatch: cm={cm.shape}, labels={len(label_names)}"
        )
    return cm, label_names


def render(model_dirs: list[Path], out_pdf: Path, out_npz: Path) -> None:
    cm, label_names = load_run(model_dirs[0])
    supports = cm.sum(axis=1)
    per_run = [cm]
    for model_dir in model_dirs[1:]:
        other, other_names = load_run(model_dir)
        if other_names != label_names:
            raise ValueError(f"Label mismatch: {model_dirs[0]} vs {model_dir}")
        # Equal per-class supports mean the same test set; averaging rates
        # across different test sets would not describe either of them.
        if not np.array_equal(other.sum(axis=1), supports):
            raise ValueError(
                f"Per-class supports differ: {model_dirs[0]} vs {model_dir}"
            )
        per_run.append(other)

    per_run = np.stack(per_run).astype(np.int64)                  # (R, C, C)
    num_classes = len(label_names)
    norm_per_run = per_run / per_run.sum(axis=2, keepdims=True).clip(min=1)
    cm_norm = norm_per_run.mean(axis=0)
    accuracy_per_run = per_run.trace(axis1=1, axis2=2) / per_run.sum(axis=(1, 2))
    macro_recall_per_run = norm_per_run.diagonal(axis1=1, axis2=2).mean(axis=1)

    # Single-column IEEE figure ≈ 3.5 in wide. Height scales with class count
    # so tick labels remain legible for both 10-class and 32-class CMs.
    width  = 3.5
    height = max(3.0, 0.18 * num_classes + 1.6)
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)

    # Match the matplotlib default sans-serif used by
    # ml-baselines/_common.py :: plot_confusion_matrix so the publication PDF and the
    # in-place training PNG have the same typeface.
    plt.rcParams.update({"font.size": 7})

    im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=6)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    if num_classes > 20:
        tick_fontsize = 4.5
    else:
        tick_fontsize = 7
    # Slanted labels overlap their neighbors once the columns get narrow, so
    # the 32-class matrix sets them upright.
    if num_classes > 20:
        ax.set_xticklabels(label_names, rotation=90, ha="center", fontsize=tick_fontsize)
    else:
        ax.set_xticklabels(label_names, rotation=60, ha="right", fontsize=tick_fontsize)
    ax.set_yticklabels(label_names, fontsize=tick_fontsize)
    ax.tick_params(axis="both", which="major", pad=1.0, length=2)
    ax.set_xlabel("Predicted class", fontsize=7)
    ax.set_ylabel("True class", fontsize=7)
    cbar.set_label("Fraction of true class", fontsize=6)

    # Cell values — only printed when the class count is small enough that
    # text remains readable; for the 32-class modulation matrix the text
    # would be unreadably small, so it is skipped.
    if num_classes <= 12:
        for i in range(num_classes):
            for j in range(num_classes):
                v = cm_norm[i, j]
                ax.text(
                    j, i, f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=5.0,
                    color="white" if v > 0.5 else "black",
                )

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_pdf.with_suffix(".png"),
                bbox_inches="tight", pad_inches=0.04, dpi=300)
    plt.close(fig)

    np.savez(
        out_npz,
        cm_norm=cm_norm.astype(np.float64),
        cm_norm_std=norm_per_run.std(axis=0).astype(np.float64),
        cm_per_run=per_run,
        supports=supports.astype(np.int64),
        accuracy_per_run=accuracy_per_run,
        macro_recall_per_run=macro_recall_per_run,
        # Plain unicode, so the file loads without allow_pickle.
        label_names=np.array(label_names, dtype=str),
        source_dir=";".join(str(d) for d in model_dirs),
        n_runs=len(model_dirs),
        cm_raw=per_run.sum(axis=0),
    )

    print(f"Accuracy: mean {100.0 * accuracy_per_run.mean():.2f}% "
          f"[{', '.join(f'{100.0 * a:.2f}' for a in accuracy_per_run)}]")
    print(f"Macro recall: mean {100.0 * macro_recall_per_run.mean():.2f}% "
          f"[{', '.join(f'{100.0 * r:.2f}' for r in macro_recall_per_run)}]")
    print(f"{len(model_dirs)} run(s) on one test set of {supports.sum()} windows")

    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_pdf.with_suffix('.png')}")
    print(f"Wrote {out_npz}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dirs", type=Path, nargs="+", metavar="model_dir",
                        help="baseline output directory; several are averaged")
    parser.add_argument("out_pdf", type=Path, help="PDF to write")
    parser.add_argument("out_npz", type=Path, help="NPZ with the plotted data")
    args = parser.parse_args()
    render(model_dirs=args.model_dirs, out_pdf=args.out_pdf, out_npz=args.out_npz)


if __name__ == "__main__":
    main()
