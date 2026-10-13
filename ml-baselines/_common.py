"""
Shared machinery for the RadioPile baselines.

The runnable scripts beside this module — technology_recognition.py and
modulation_classification.py for the random-window split, group_split.py for
the group-aware split the paper reports — carry the configuration as literal
constants and call run_baseline() below. All task-independent logic lives here
so the scripts differ only in their configuration, not their code path.

Numerical behavior (subsampling, splitting, model, class weights, callbacks)
is preserved verbatim from the training script that produced the published
numbers. Do not "clean up" the arithmetic here without re-running
both baselines: the reported accuracies are reproduction targets.

TensorFlow is an OPTIONAL dependency of this repository. It is imported lazily
by require_tensorflow() so that importing this module — or running the rest of
the pipeline and its test suite — never requires it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.io import _detect_git_commit  # noqa: E402


# Where to look for the aggregate, in order. The released bundle wins: these
# baselines are meant to reproduce the paper from the published artifact, and
# falling back to the in-tree build keeps them usable before a release is cut.
H5_CANDIDATES = (
    Path("release/v1.0/aggregated/radiopile_v1.0.h5"),
    Path("data/aggregation/processed/aggregated.h5"),
)

H5_ENV_VAR = "RADIOPILE_H5"
OUT_ENV_VAR = "RADIOPILE_BASELINE_OUT"


# ── Optional-dependency handling ──────────────────────────────────────────────

def require_tensorflow():
    """
    Import TensorFlow, or exit with an actionable message.

    TensorFlow is deliberately not in requirements.txt: it is needed only by
    these baselines, not by the preprocessing pipeline, the validator, or the
    test suite.
    """
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras  # noqa: F401
    except ImportError as exc:
        sys.exit(
            "ERROR: TensorFlow is required to run the baselines but is not installed.\n"
            f"       ({exc})\n\n"
            "       TensorFlow is an optional dependency of this repository — the\n"
            "       preprocessing pipeline, validator, and tests do not need it.\n\n"
            "       Install the baseline extras:\n"
            "           pip install -r requirements-ml-baselines.txt\n"
        )
    return tf, keras


# ── Input resolution ──────────────────────────────────────────────────────────

def resolve_h5_path(explicit: Optional[Path] = None) -> Path:
    """Locate the aggregated HDF5: explicit argument, then env var, then defaults."""
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            sys.exit(f"ERROR: aggregate not found at {path}")
        return path

    env_value = os.environ.get(H5_ENV_VAR)
    if env_value:
        path = Path(env_value)
        if not path.exists():
            sys.exit(f"ERROR: {H5_ENV_VAR} points at a missing file: {path}")
        return path

    for candidate in H5_CANDIDATES:
        resolved = candidate if candidate.is_absolute() else ROOT / candidate
        if resolved.exists():
            return resolved

    tried = "\n".join(f"         - {ROOT / c}" for c in H5_CANDIDATES)
    sys.exit(
        "ERROR: could not find the aggregated dataset. Looked in:\n"
        f"{tried}\n\n"
        "       Build it with `python run.py aggregation`, or point at an\n"
        f"       existing copy with {H5_ENV_VAR}=/path/to/radiopile_v1.0.h5\n"
    )


def resolve_output_dir(default_name: str, explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    env_value = os.environ.get(OUT_ENV_VAR)
    if env_value:
        return Path(env_value) / default_name
    return ROOT / "ml-baselines" / "results" / default_name


# ── Data loading ──────────────────────────────────────────────────────────────

def _read_coalesced(ds, sorted_indices, max_gap: int = 512):
    """
    Read selected rows from a chunked HDF5 dataset efficiently.

    Fancy indexing reads one whole HDF5 chunk per touched chunk, even when only
    one row is needed from it. For large sorted index sets that means thousands
    of separate reads. This groups nearby indices into contiguous slice reads,
    trading a little extra I/O for far fewer calls. Raise max_gap for spinning
    disk, lower it for NVMe.
    """
    if len(sorted_indices) == 0:
        return np.empty((0, *ds.shape[1:]), dtype=ds.dtype)

    parts = []
    seg_start = int(sorted_indices[0])
    seg_local = [0]
    prev = seg_start

    for raw in sorted_indices[1:]:
        cur = int(raw)
        if cur - prev <= max_gap:
            seg_local.append(cur - seg_start)
        else:
            parts.append(ds[seg_start: prev + 1][seg_local])
            seg_start = cur
            seg_local = [0]
        prev = cur

    parts.append(ds[seg_start: prev + 1][seg_local])
    return np.concatenate(parts, axis=0)


def _decode(values) -> np.ndarray:
    return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values])


def select_task_indices(
    f,
    task: str,
    subsample: float,
    max_per_class: Optional[int],
    seed: int,
    min_snr: Optional[float] = None,
    verbose: bool = True,
):
    """
    Choose which windows a task trains on, without reading any IQ data.

    Split out from load_task_data so that tooling needing only the selection —
    the release-verification harness re-deriving a run's exact test rows, for
    instance — can reuse it instead of re-implementing the arithmetic. Any
    divergence between a reimplementation and this function would silently
    evaluate a model on rows it was trained on.

    `f` is an open h5py.File. Returns (global_idx, labels_sel, unique_labels)
    where global_idx indexes into the full dataset and is sorted ascending.
    """
    assert task in ("technology", "modulation"), f"Unknown task: {task!r}"

    meta_field = "meta/technology_type" if task == "technology" else "meta/modulation_type"
    exclude_vals = {"unknown"} if task == "technology" else {"unknown", "noise"}

    types = _decode(f[meta_field][:])
    known_mask = ~np.isin(types, list(exclude_vals))

    # SNR filter (modulation only): keep NaN (real-world captures) OR >= min_snr.
    if task == "modulation" and min_snr is not None:
        snr_raw = f["meta/snr_db"][:]
        known_mask = known_mask & (np.isnan(snr_raw) | (snr_raw >= min_snr))
        if verbose:
            print(f"  SNR filter: >= {min_snr} dB (NaN = real-world, always included)")

    known_indices = np.where(known_mask)[0]
    known_types = types[known_mask]
    if verbose:
        print(f"  {known_mask.sum():,} / {len(types):,} windows pass filters")

        src_known = _decode(f["meta/source_dataset"][:])[known_mask]
        unique_sources, src_counts = np.unique(src_known, return_counts=True)
        print("  Contributing source datasets:")
        for source, count in zip(unique_sources, src_counts):
            print(f"    {source:45s}  {count:>8,}")

    unique_labels = sorted(set(known_types))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    labels_all = np.array([label_to_id[t] for t in known_types], dtype=np.int64)

    # Stratified random subsample at a fixed seed. A random draw rather than a
    # head slice because RML windows are stored in (modulation, SNR) order:
    # head-slicing would take only the lowest SNR levels, where every class
    # looks like noise.
    rng = np.random.default_rng(seed)
    if subsample < 1.0 or max_per_class is not None:
        idx_parts = []
        for cls_id in range(len(unique_labels)):
            cls_idx = np.where(labels_all == cls_id)[0]
            n_take = int(len(cls_idx) * subsample) if subsample < 1.0 else len(cls_idx)
            if max_per_class is not None:
                n_take = min(n_take, max_per_class)
            n_take = max(1, n_take)
            chosen = (
                rng.choice(cls_idx, size=n_take, replace=False)
                if n_take < len(cls_idx)
                else cls_idx
            )
            idx_parts.append(chosen)
        local_idx = np.sort(np.concatenate(idx_parts))
    else:
        local_idx = np.arange(len(labels_all))

    return known_indices[local_idx], labels_all[local_idx], unique_labels


def load_task_data(
    h5_path: Path,
    task: str,
    subsample: float,
    max_per_class: Optional[int],
    seed: int,
    min_snr: Optional[float] = None,
):
    """
    Load IQ windows and labels for one task from the aggregated HDF5.

    Labels come from meta/technology_type or meta/modulation_type directly, so
    the task is independent of the PRIMARY_LABEL_TYPE chosen at aggregation
    build time — no rebuild is needed to switch tasks.

    Returns (x, y, label_names) with x shaped (N, window, 2) for Conv1D.
    """
    meta_field = "meta/technology_type" if task == "technology" else "meta/modulation_type"

    print(f"Loading dataset from {h5_path} ...")
    print(f"  Task: {task}  |  Label field: {meta_field}")

    with h5py.File(h5_path, "r") as f:
        global_idx, labels_sel, unique_labels = select_task_indices(
            f, task, subsample, max_per_class, seed, min_snr
        )
        approx_gb = len(global_idx) * 2 * 500 * 4 / 1e9
        print(f"  Reading {len(global_idx):,} windows (~{approx_gb:.1f} GB) from HDF5 ...")
        data_sel = _read_coalesced(f["data"], global_idx)

    # (N, 2, W) on disk → (N, W, 2) for Conv1D.
    data_sel = data_sel.transpose(0, 2, 1)

    print(f"  Loaded {len(labels_sel):,} windows  |  {len(unique_labels)} classes")
    for value, count in zip(*np.unique(labels_sel, return_counts=True)):
        print(f"    [{value}] {unique_labels[value]:30s}  {count:>8,}")

    return data_sel, labels_sel, unique_labels


def split_indices(n: int, val_split: float, test_split: float, seed: int):
    """Return (train_idx, val_idx, test_idx) from one seeded permutation."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]


# Sources whose files were recorded at the same time. Captures that share a
# recording_id across those files are the same moment seen by two receivers
# (the NIST 5 GHz indoor and outdoor files), so they form one group and never
# fall on both sides of a split.
SIMULTANEOUS_CAPTURE_SOURCES = {"nist_wifi_bluetooth"}


def load_group_keys(f, global_idx):
    """
    Return (groups, sources) for the selected windows: the recording each one
    was cut from, and its source dataset.

    A recording is one source file, narrowed by recording_id where a file holds
    several (NIST capture keys, ITS Band observations). recording_id alone is
    not enough: sub_ghz and lte_wifi_dvbt reuse the same run indices across
    every file. For the sources in SIMULTANEOUS_CAPTURE_SOURCES it is the
    recording_id alone that identifies a recording. The simulated sources (RML,
    RadChar) leave recording_id empty because their windows are i.i.d., so each
    of those windows is its own group.
    """
    sources = _decode(f["meta/source_dataset"][:][global_idx])
    files = _decode(f["meta/file_path"][:][global_idx])
    rec_ids = _decode(f["meta/recording_id"][:][global_idx])
    groups = np.array([
        f"{src}|window|{idx}" if not rec
        else f"{src}|{rec}" if src in SIMULTANEOUS_CAPTURE_SOURCES
        else f"{src}|{path}|{rec}"
        for src, path, rec, idx in zip(sources, files, rec_ids, global_idx)
    ])
    return groups, sources


def group_split_indices(y, groups, sources, val_split: float, test_split: float, seed: int):
    """
    Group-aware counterpart of split_indices: no group is divided between
    train, validation and test.

    The split is stratified by (class, source dataset). Within each stratum the
    groups are shuffled and whole groups are dealt to test, then validation,
    until each holds as close to its share of the stratum's windows as whole
    groups allow, with at least one group each. A stratum of single-window
    groups therefore gets an ordinary stratified random split, while one with
    five recordings can do no better than 3/1/1.

    Returns (train_idx, val_idx, test_idx, report), where report lists the
    group and window counts per stratum.
    """
    rng = np.random.default_rng(seed)
    strata = np.array([f"{cls}|{src}" for cls, src in zip(y, sources)])
    part = np.zeros(len(y), dtype=np.int8)          # 0 train, 1 val, 2 test
    report = []

    for stratum in np.unique(strata):
        rows = np.where(strata == stratum)[0]
        group_ids, inverse, sizes = np.unique(
            groups[rows], return_inverse=True, return_counts=True
        )
        n_groups = len(group_ids)
        if n_groups < 3:
            raise ValueError(
                f"stratum {stratum!r} has {n_groups} group(s); a group split "
                "needs at least three to fill train, validation and test"
            )

        order = rng.permutation(n_groups)
        cumulative = np.cumsum(sizes[order])
        n = len(rows)
        k_test = int(np.argmin(np.abs(cumulative - n * test_split))) + 1
        k_test = min(max(k_test, 1), n_groups - 2)
        val_target = cumulative[k_test - 1] + n * val_split
        k_val = int(np.argmin(np.abs(cumulative - val_target))) + 1
        k_val = min(max(k_val, k_test + 1), n_groups - 1)

        group_part = np.zeros(n_groups, dtype=np.int8)
        group_part[order[:k_test]] = 2
        group_part[order[k_test:k_val]] = 1
        part[rows] = group_part[inverse]

        report.append({
            "class_id": int(y[rows[0]]),
            "source_dataset": str(sources[rows[0]]),
            "groups": [int((group_part == p).sum()) for p in (0, 1, 2)],
            "windows": [int((part[rows] == p).sum()) for p in (0, 1, 2)],
        })

    # A group that carried two classes would have been dealt once per stratum.
    for a, b in ((0, 1), (0, 2), (1, 2)):
        shared = np.intersect1d(groups[part == a], groups[part == b])
        if len(shared):
            raise ValueError(f"group {shared[0]!r} spans more than one class")

    # Shuffled so that row order carries no stratum structure into training.
    train_idx, val_idx, test_idx = (rng.permutation(np.where(part == p)[0]) for p in (0, 1, 2))
    return train_idx, val_idx, test_idx, report


# ── Model ─────────────────────────────────────────────────────────────────────

def build_cnn(keras, num_classes: int, window_size: int, name: str):
    """
    The paper's 1D CNN: three Conv1D blocks → global average pool → dense head.

    Input is (window_size, 2) — time steps by IQ channel. About 53k parameters
    for the 10-class technology task and 56k for the 32-class modulation task.
    """
    inputs = keras.Input(shape=(window_size, 2), name="iq")

    x = keras.layers.Conv1D(32, 7, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling1D()(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name=name)


# ── Reporting ─────────────────────────────────────────────────────────────────

def compute_confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, label_names: Sequence[str],
                          save_path: Path, title: str):
    """Row-normalized confusion matrix (row = true, column = predicted)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_classes = len(label_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(8, num_classes), max(6, num_classes)))
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {save_path}")


def plot_loss_curves(history, save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"], label="Train")
    ax1.plot(history.history["val_loss"], label="Validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["accuracy"], label="Train")
    ax2.plot(history.history["val_accuracy"], label="Validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy Curves")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Loss curves saved → {save_path}")


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_baseline(
    *,
    task: str,
    model_name: str,
    output_name: str,
    reference_accuracy: float,
    expected_num_classes: int,
    subsample: float,
    max_per_class: Optional[int],
    epochs: int,
    batch_size: int,
    val_split: float,
    test_split: float,
    seed: int,
    min_snr: Optional[float] = None,
    h5_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    split: str = "random",
) -> float:
    """
    Run one baseline end to end. Returns the test accuracy.

    split="random" divides windows at random. split="group", the split the paper
    reports, keeps the same
    windows, model and training setup but divides them with
    group_split_indices, so that no recording contributes to more than one of
    train, validation and test.
    """
    assert split in ("random", "group"), f"Unknown split: {split!r}"
    tf, keras = require_tensorflow()

    h5_path = resolve_h5_path(h5_path)
    output_dir = resolve_output_dir(output_name, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"RadioPile baseline — {task} ({expected_num_classes} classes)")
    print(f"Reference result ({split} split): {reference_accuracy * 100:.1f}% test accuracy")
    print("=" * 72)
    print(f"  aggregate : {h5_path}")
    print(f"  output    : {output_dir}")
    print(f"  config    : subsample={subsample} max_per_class={max_per_class} "
          f"epochs={epochs} batch={batch_size} seed={seed}")
    print(f"  split     : {int((1 - val_split - test_split) * 100)}/"
          f"{int(val_split * 100)}/{int(test_split * 100)} train/val/test ({split})")
    print()

    tf.random.set_seed(seed)

    x, y, label_names = load_task_data(
        h5_path, task, subsample, max_per_class, seed, min_snr
    )
    num_classes = len(label_names)
    window_size = x.shape[1]

    if num_classes != expected_num_classes:
        print(f"\nWARNING: found {num_classes} classes, paper reports "
              f"{expected_num_classes}. The aggregate may differ from the released one.")

    sources = None
    split_report = None
    if split == "group":
        with h5py.File(h5_path, "r") as f:
            global_idx, _, _ = select_task_indices(
                f, task, subsample, max_per_class, seed, min_snr, verbose=False
            )
            groups, sources = load_group_keys(f, global_idx)
        train_idx, val_idx, test_idx, split_report = group_split_indices(
            y, groups, sources, val_split, test_split, seed
        )
        print("\nGroup split — groups and windows per (class, source), train/val/test:")
        for row in split_report:
            row["class"] = label_names[row.pop("class_id")]
            g, w = row["groups"], row["windows"]
            print(f"  {row['class']:15s} {row['source_dataset']:20s} "
                  f"groups {g[0]:>6,}/{g[1]:>5,}/{g[2]:>5,}   "
                  f"windows {w[0]:>6,}/{w[1]:>6,}/{w[2]:>6,}")
    else:
        train_idx, val_idx, test_idx = split_indices(len(y), val_split, test_split, seed)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]
    print(f"\nSplit — train: {len(y_train):,}  val: {len(y_val):,}  test: {len(y_test):,}")

    # Inverse-frequency class weights.
    unique, counts = np.unique(y_train, return_counts=True)
    weights = len(y_train) / (num_classes * counts.astype(float))
    class_weight = {int(cls): float(w) for cls, w in zip(unique, weights)}
    print("\nClass weights:")
    for cls, weight in class_weight.items():
        print(f"  [{cls}] {label_names[cls]:30s}  weight={weight:.3f}")

    label_info = {
        label_names[cls]: {"count": int(counts[i]), "class_weight": float(weights[i])}
        for i, cls in enumerate(unique)
    }
    (output_dir / "label_info.json").write_text(json.dumps(label_info, indent=2))

    model = build_cnn(keras, num_classes, window_size, model_name)
    model.summary()

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    checkpoint_path = output_dir / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_accuracy", save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1,
        ),
    ]

    started = time.time()
    history = model.fit(
        x_train, y_train,
        class_weight=class_weight,
        validation_data=(x_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
    )
    duration = time.time() - started

    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)

    plot_loss_curves(history, output_dir / "loss_curves.png")

    y_pred = model.predict(x_test, batch_size=batch_size, verbose=0).argmax(axis=1)
    cm = compute_confusion_matrix(y_test, y_pred, num_classes)
    np.save(output_dir / "confusion_matrix.npy", cm)
    plot_confusion_matrix(
        cm, label_names, output_dir / "confusion_matrix.png",
        f"Confusion Matrix — {task.capitalize()} ({num_classes} classes, row-normalized)",
    )

    (output_dir / "training_history.json").write_text(
        json.dumps({k: [float(v) for v in vs] for k, vs in history.history.items()},
                   indent=2, sort_keys=True)
    )

    per_source = None
    if sources is not None:
        src_test = sources[test_idx]
        per_source = {
            str(src): {
                "test_samples": int((src_test == src).sum()),
                "accuracy": float((y_pred[src_test == src] == y_test[src_test == src]).mean()),
            }
            for src in np.unique(src_test)
        }
        print("\nTest accuracy by source dataset:")
        for src, row in per_source.items():
            print(f"  {src:25s} {row['accuracy'] * 100:6.2f}%   n={row['test_samples']:,}")

    val_acc_history = history.history.get("val_accuracy", [])
    snapshot_path = h5_path.with_name(h5_path.name + ".config.json")
    h5_config_snapshot = None
    if snapshot_path.exists():
        try:
            h5_config_snapshot = json.loads(snapshot_path.read_text())
        except (OSError, json.JSONDecodeError):
            h5_config_snapshot = None

    summary = {
        "task": task,
        "n_classes": int(num_classes),
        "training_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "final_test_accuracy": float(acc),
        "final_test_loss": float(loss),
        "best_val_accuracy": float(np.max(val_acc_history)) if val_acc_history else 0.0,
        "best_val_epoch": int(np.argmax(val_acc_history)) + 1 if val_acc_history else 0,
        "reference_accuracy": reference_accuracy,
        "training_duration_seconds": float(duration),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _detect_git_commit(),
        "aggregate_path": str(h5_path),
        "h5_config_snapshot": h5_config_snapshot,
        "group_split": split_report,
        "per_source_test_accuracy": per_source,
        "training_config": {
            "TASK": task,
            "BATCH_SIZE": batch_size,
            "EPOCHS": epochs,
            "VAL_SPLIT": val_split,
            "TEST_SPLIT": test_split,
            "SPLIT": split,
            "SEED": seed,
            "SUBSAMPLE": subsample,
            "MAX_PER_CLASS": max_per_class,
            "MIN_SNR": min_snr,
        },
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )

    delta = (acc - reference_accuracy) * 100
    print()
    print("=" * 72)
    print(f"TEST ACCURACY: {acc * 100:.2f}%   (test loss {loss:.4f})")
    print(f"Reference ({split} split): {reference_accuracy * 100:.1f}%   difference {delta:+.2f} pp")
    print("=" * 72)
    print(f"Artifacts written to {output_dir}")

    return float(acc)
