"""
RF data I/O utilities.

Responsibilities:
- Detect file formats
- Load raw IQ data from MAT and HDF5 files, and window BIN files through a memory map
- Convert raw arrays into standardized IQ format [N, 2]
- Load processed ML-ready HDF5 datasets with metadata
- Write standalone configuration snapshots alongside processed HDF5 files
"""

import json
import subprocess
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import h5py
import scipy.io as sio
from pipeline.windowing import window_iq_signal, iter_windows_from_bin

def detect_format(path: Path) -> Optional[str]:
    """Detect raw file format based on extension."""
    ext = path.suffix.lower()
    if ext == ".bin":
        return "bin"
    if ext == ".mat":
        return "mat"
    if ext == ".h5":
        return "h5"
    return None


def _array_to_iq(array: np.ndarray) -> np.ndarray:
    """
    Convert various array layouts into standardized IQ format:
    shape [N_samples, 2], dtype float32.
    """
    array = np.asarray(array)

    if np.iscomplexobj(array):
        flat = array.reshape(-1)
        return np.column_stack((flat.real, flat.imag)).astype(np.float32)

    if array.ndim == 1:
        if array.size % 2 != 0:
            raise ValueError("Odd-length real array cannot be interleaved IQ")
        return array.reshape(-1, 2).astype(np.float32)

    if array.ndim == 2:
        if array.shape[1] == 2:
            return array.astype(np.float32)
        if array.shape[0] == 2:
            return array.T.astype(np.float32)

    raise ValueError(f"Unsupported IQ array shape {array.shape}")


def load_mat(path: Path) -> np.ndarray:
    mat = sio.loadmat(path)
    arrays = [
        v for k, v in mat.items()
        if not k.startswith("__") and isinstance(v, np.ndarray)
    ]
    arrays.sort(key=lambda a: a.size, reverse=True)
    for arr in arrays:
        try:
            return _array_to_iq(arr)
        except Exception:
            continue
    raise ValueError(f"No valid IQ array in MAT file {path}")


def _iter_h5_datasets(group):
    for value in group.values():
        if isinstance(value, h5py.Dataset):
            yield value
        elif isinstance(value, h5py.Group):
            yield from _iter_h5_datasets(value)


def load_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        datasets = list(_iter_h5_datasets(f))
        datasets.sort(key=lambda d: d.size, reverse=True)
        for ds in datasets:
            try:
                return _array_to_iq(ds[()])
            except Exception:
                continue
    raise ValueError(f"No valid IQ dataset in H5 file {path}")


def load_raw_file(path: Path) -> np.ndarray:
    """Load a whole MAT or HDF5 file as an [N, 2] IQ array.

    BIN files are not loaded here: iter_windows_for_file windows them through a
    memory map, because their dtype cannot be inferred from the file.
    """
    fmt = detect_format(path)
    if fmt == "mat":
        return load_mat(path)
    if fmt == "h5":
        return load_h5(path)
    raise ValueError(f"Unknown raw format: {path}")

def iter_windows_for_file(
    path: Path,
    window_size: int,
    hop_size: int,
    pad_last: bool,
    dtype=np.int16,
    output_dtype=np.float32,
):
    """
    Yield (window, metadata) tuples for a single raw RF file.

    The strategy depends on the format:
    - BIN files: read through a memory map, converted to output_dtype, then windowed
    - MAT/H5 files: loaded into memory, then windowed

    Parameters
    ----------
    path : Path
        Path to the raw RF data file
    window_size : int
        Size of each window in samples
    hop_size : int
        Number of samples to advance between windows
    pad_last : bool
        Whether to pad the last window if it's incomplete
    dtype : type, optional
        Source on-disk data type for BIN files (default: np.int16). Ignored for other formats.
    output_dtype : type, optional
        Dtype the emitted IQ windows are cast to (default: np.float32). Wire this to
        the dataset's DATA_DTYPE config parameter. Applies to both BIN and MAT/H5 paths.

    Yields
    ------
    window : np.ndarray
        Windowed IQ data, shape (2, window_size), dtype output_dtype (default float32)
    metadata : dict
        Metadata dict with 'start_idx' and 'padded' keys
    """
    path = Path(path)

    fmt = detect_format(path)
    if fmt is None:
        raise ValueError(f"Unsupported file format: {path}")

    # ------------------------------------------------------------------
    # BIN files: read through a memory map (the source dtype must be given)
    # ------------------------------------------------------------------
    if fmt == "bin":
        for window, meta in iter_windows_from_bin(
            path=path,
            window_size=window_size,
            hop_size=hop_size,
            pad_last=pad_last,
            dtype=dtype,
            output_dtype=output_dtype,
        ):
            yield window, meta
        return

    # ------------------------------------------------------------------
    # MAT/H5 files: Load into memory, then window
    # ------------------------------------------------------------------
    iq = load_raw_file(path).astype(output_dtype, copy=False)

    windows, meta = window_iq_signal(
        iq,
        window_size=window_size,
        hop_size=hop_size,
        pad_last=pad_last,
    )

    start_indices = meta.get("start_idx", [])
    padded_flags = meta.get("padded", [])

    for i in range(windows.shape[0]):
        yield windows[i], {
            "start_idx": int(start_indices[i]) if len(start_indices) else -1,
            "padded": bool(padded_flags[i]) if len(padded_flags) else False,
        }


# ---------------------------------------------------------------------
# Processed dataset loading (ML-ready HDF5 datasets)
# ---------------------------------------------------------------------

def load_processed_h5_dataset(h5_path: Path):
    """
    Load processed ML-ready HDF5 dataset and return data, labels, and combined metadata.

    The metadata dictionary combines:
    - Per-window metadata from /meta/* datasets
    - Global attributes (for fields missing in per-window metadata)

    This ensures that fields like sample_rate_hz are available even if they
    were not persisted as per-window metadata (e.g., if they were removed from
    the schema but are available as global attributes).

    Parameters
    ----------
    h5_path : Path
        Path to the processed HDF5 dataset file.

    Returns
    -------
    x : np.ndarray
        Windowed IQ data, shape (B, W, 2) where B is batch size, W is window size.
    y : np.ndarray
        Label indices, shape (B,), normalized to 0..C-1.
    label_names : List[str]
        List of class label names.
    metadata : Dict[str, np.ndarray]
        Combined metadata dictionary with per-window fields and global attributes.
        Global attributes are stored under the "_global_attrs" key.
    """
    with h5py.File(h5_path, "r") as f:
        missing = [k for k in ("data", "labels") if k not in f]
        if missing:
            raise ValueError(
                f"{h5_path} is missing required dataset(s): {', '.join(missing)}. "
                f"Present at root: {sorted(f.keys())}. Expected a file written by "
                f"pipeline.hdf5_writer.build_hdf5_dataset."
            )

        x_raw = f["data"][:]  # (B, 2, W) or (B, W, 2)
        y_raw = f["labels"][:].astype(np.int64)
        num_windows = len(y_raw)

        # Get label names. build_hdf5_dataset always writes /label_names, so a
        # file lacking both it and meta/label_name is corrupt or was not produced
        # by this pipeline. Raise rather than fall back to numeric class names,
        # which no shape, dtype or record-count check would catch.
        if "label_names" in f:
            label_names = [s.decode("utf-8") if isinstance(s, bytes) else str(s)
                          for s in f["label_names"][:]]
        elif "meta" in f and "label_name" in f["meta"]:
            label_names = sorted(set([s.decode("utf-8") if isinstance(s, bytes) else str(s)
                                     for s in f["meta/label_name"][:]
                                     if s and s not in (b"__NONE__", "__NONE__", "None", None)]))
        else:
            raise ValueError(
                f"{h5_path} has no label names: neither /label_names nor "
                f"meta/label_name is present. Present at root: {sorted(f.keys())}."
            )

        original_dtype = x_raw.dtype
        # Fall back to float32 if the stored dtype is not a float type
        if not np.issubdtype(original_dtype, np.floating):
            original_dtype = np.float32

        # Convert to (B, W, 2) format, preserving the original dtype
        # Data is always stored as (B, 2, W) by build_hdf5_dataset, but both formats are
        # handled for robustness. Check both dimensions to avoid ambiguity when window_size=2.
        if x_raw.ndim == 3:
            if x_raw.shape[1] == 2 and x_raw.shape[2] != 2:
                # Definitely (B, 2, W) format: shape[1]=2, shape[2]=W (where W != 2)
                x = np.moveaxis(x_raw, 1, 2).astype(original_dtype, copy=False)
            elif x_raw.shape[2] == 2 and x_raw.shape[1] != 2:
                # Definitely (B, W, 2) format: shape[1]=W (where W != 2), shape[2]=2
                x = x_raw.astype(original_dtype, copy=False)
            elif x_raw.shape[1] == 2 and x_raw.shape[2] == 2:
                # Ambiguous case: (B, 2, 2) could be either format
                # Since build_hdf5_dataset always writes (B, 2, W), assume (B, 2, W) format
                x = np.moveaxis(x_raw, 1, 2).astype(original_dtype, copy=False)
            else:
                raise ValueError(
                    f"{h5_path} has IQ data of shape {x_raw.shape}, which is "
                    f"neither (B, 2, W) nor (B, W, 2) — neither trailing axis has "
                    f"length 2, so the real/imaginary axis cannot be identified."
                )
        else:
            raise ValueError(
                f"{h5_path} has IQ data of shape {x_raw.shape}; expected a 3-D "
                f"array of (B, 2, W) or (B, W, 2)."
            )

        # Normalize labels to 0..C-1
        unique_labels = np.unique(y_raw)
        label_map = {int(old): int(i) for i, old in enumerate(unique_labels)}
        y = np.array([label_map[int(v)] for v in y_raw], dtype=np.int64)

        # Load per-window metadata
        metadata = {}
        if "meta" in f:
            for key in f["meta"].keys():
                metadata[key] = f[f"meta/{key}"][:]

        # Get global attributes
        global_attrs = dict(f.attrs)

        # Create combined metadata: fill missing per-window fields from global attributes
        # Common fields that might be global-only:
        global_to_per_window = {
            "sample_rate_hz": "sample_rate_hz",  # Often stored as global if fixed
        }

        for global_key, meta_key in global_to_per_window.items():
            # Only add if not already in per-window metadata
            if meta_key not in metadata and global_key in global_attrs:
                global_value = global_attrs[global_key]
                # Convert to per-window array by broadcasting the global value
                if isinstance(global_value, (int, float, np.number)):
                    # Create array filled with the global value
                    metadata[meta_key] = np.full(num_windows, float(global_value), dtype=np.float64)
                elif isinstance(global_value, str):
                    # For strings, create array of the same string
                    metadata[meta_key] = np.array([global_value] * num_windows, dtype=object)

        # Store global attributes separately for reference
        metadata["_global_attrs"] = global_attrs

    return x, y, label_names, metadata


# ---------------------------------------------------------------------
# Configuration snapshot
# ---------------------------------------------------------------------

class _ConfigJSONEncoder(json.JSONEncoder):
    """JSON encoder for build-config dicts.

    Handles types that appear in GLOBAL_METADATA across datasets but are not
    natively JSON-serializable: Path, Enum, numpy scalars/arrays, sets.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, (set, frozenset)):
            return sorted(obj, key=str)
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="replace")
        return super().default(obj)


def _git(*args: str) -> Optional[subprocess.CompletedProcess]:
    """Run a git command, returning None if git is unavailable or errors."""
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _detect_git_commit() -> Optional[str]:
    """Return the current git commit (12-char short SHA) or None.

    A "-dirty" suffix is appended when the working tree has uncommitted
    changes to tracked files, following the convention of `git describe
    --dirty`. Without it a build run from a modified tree records a commit
    that does not reproduce the artifact. Ignored files (data/, release/)
    are not counted, so a normal build does not read as dirty.
    """
    result = _git("rev-parse", "--short=12", "HEAD")
    if result is None or result.returncode != 0:
        return None
    commit = result.stdout.strip()
    if not commit:
        return None

    status = _git("status", "--porcelain", "--untracked-files=no")
    if status is not None and status.returncode == 0 and status.stdout.strip():
        commit += "-dirty"
    return commit


def write_config_snapshot(output_path: Path, config: Dict[str, Any]) -> Path:
    """Write a JSON snapshot of the build-time configuration next to the HDF5.

    The snapshot is written to ``f"{output_path}.config.json"`` and contains
    the supplied ``config`` dict augmented with:

    - ``snapshot_written_at``: UTC ISO-8601 timestamp of the snapshot.
    - ``git_commit``: short SHA of the repository at build time, when
      available (omitted when not in a git repo).

    The output is deterministic (``sort_keys=True``, ``indent=2``) so that
    re-running the build with identical configuration produces a byte-identical
    snapshot — apart from the timestamp/commit fields above.

    Returns the path written.
    """
    snapshot: Dict[str, Any] = dict(config)  # shallow copy; do not mutate caller
    snapshot["snapshot_written_at"] = datetime.now(timezone.utc).isoformat()
    commit = _detect_git_commit()
    if commit is not None:
        snapshot["git_commit"] = commit

    output_path = Path(output_path)
    snapshot_path = output_path.with_name(output_path.name + ".config.json")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True, cls=_ConfigJSONEncoder)
        f.write("\n")

    return snapshot_path
