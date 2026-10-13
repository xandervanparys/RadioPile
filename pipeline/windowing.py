"""
IQ windowing utilities.

Responsibilities:
- Window in-memory IQ arrays
- Window BIN files through a memory map
- Produce per-window metadata (start_idx, padded)
"""

from pathlib import Path
from typing import Iterator, Tuple, Dict
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def window_iq_signal(
    iq: np.ndarray,
    window_size: int,
    hop_size: int,
    pad_last: bool,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Slice a [N, 2] IQ signal into [num_windows, 2, window_size] windows.

    Full windows are extracted with a strided view and returned as a new array. If pad_last is True
    and the signal tail doesn't fill a window, one zero-padded window is appended.
    If the signal is shorter than window_size, returns empty arrays or a single
    padded window depending on pad_last.

    Raises ValueError for non-[N,2] input or non-positive window_size / hop_size.
    """
    if iq.ndim != 2 or iq.shape[1] != 2:
        raise ValueError(f"Expected IQ shape [N, 2], got {iq.shape}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if hop_size <= 0:
        raise ValueError(f"hop_size must be positive, got {hop_size}")

    n = iq.shape[0]
    output_dtype = iq.dtype
    empty = (
        np.zeros((0, 2, window_size), dtype=output_dtype),
        {"start_idx": np.array([], dtype=np.int64),
         "padded": np.array([], dtype=bool)},
    )

    if n < window_size:
        if not pad_last:
            return empty
        window = np.zeros((window_size, 2), dtype=output_dtype)
        window[:n] = iq
        return (
            window.T[np.newaxis].astype(output_dtype),
            {"start_idx": np.array([0], dtype=np.int64),
             "padded": np.array([True], dtype=bool)},
        )

    full_windows = sliding_window_view(iq, window_size, axis=0)[
        ::hop_size].astype(output_dtype)
    num_full = full_windows.shape[0]
    start_indices = np.arange(num_full, dtype=np.int64) * hop_size
    padded_flags = np.zeros(num_full, dtype=bool)

    if pad_last:
        tail_start = num_full * hop_size
        if tail_start < n:
            tail = np.zeros((window_size, 2), dtype=output_dtype)
            tail[: n - tail_start] = iq[tail_start:]
            full_windows = np.concatenate(
                [full_windows, tail.T[np.newaxis].astype(output_dtype)], axis=0)
            start_indices = np.append(start_indices, np.int64(tail_start))
            padded_flags = np.append(padded_flags, True)

    if full_windows.shape[0] == 0:
        return empty

    return (full_windows, {"start_idx": start_indices, "padded": padded_flags})


def iter_windows_from_bin(
    path: Path,
    window_size: int,
    hop_size: int,
    pad_last: bool,
    dtype: np.dtype = np.int16,
    output_dtype: np.dtype = np.float32,
) -> Iterator[Tuple[np.ndarray, Dict]]:
    """Yield windows from a BIN file.

    ``dtype`` is the *source* on-disk dtype (int16 or complex64/128).
    ``output_dtype`` is the dtype the emitted IQ windows are cast to (float32 by
    default); pass the dataset's ``DATA_DTYPE`` config parameter.

    The file is read through a memory map and converted to ``output_dtype`` in one
    step, so the converted samples are held in memory while the file is windowed.
    """
    raw = np.memmap(path, dtype=dtype, mode="r")
    if np.issubdtype(dtype, np.complexfloating):
        # complex64/complex128: reinterpret the bytes as the component float dtype and
        # reshape to (N, 2) [I, Q] pairs. The view itself copies nothing; astype then
        # reads the file into an array of output_dtype.
        float_view_dtype = np.dtype(f"f{np.dtype(dtype).itemsize // 2}")
        iq = raw.view(float_view_dtype).reshape(-1, 2).astype(output_dtype)
    else:
        # Integer dtype (e.g. int16): interleaved I,Q pairs, scaled by the dtype's maximum
        iq = (raw.reshape(-1, 2) / np.iinfo(dtype).max).astype(output_dtype)
    yield from iter_windows_from_samples(iq, window_size, hop_size, pad_last)


def iter_windows_from_samples(
    samples: np.ndarray,
    window_size: int,
    hop_size: int,
    pad_last: bool,
) -> Iterator[Tuple[np.ndarray, Dict[str, object]]]:
    """Yield (window, metadata) tuples from in-memory [N, 2] IQ samples."""
    windows, meta = window_iq_signal(
        samples, window_size=window_size, hop_size=hop_size, pad_last=pad_last
    )
    for i in range(windows.shape[0]):
        yield windows[i], {
            "start_idx": int(meta["start_idx"][i]),
            "padded": bool(meta["padded"][i]),
        }
