"""
Signal preprocessing utilities.

Responsibilities:
- Compute window power
- Normalize windows
- Detect noise floor
- Filter windows based on noise threshold
"""

import numpy as np
from typing import Optional, Iterator, Tuple, Dict, List
from pathlib import Path


def compute_window_power_lin(window: np.ndarray) -> float:
    return float(np.mean(window[0] ** 2 + window[1] ** 2))


def compute_window_power_db(window: np.ndarray) -> float:
    return 10.0 * np.log10(compute_window_power_lin(window) + 1e-12)


def normalize_window(window: np.ndarray, mode: Optional[str]) -> np.ndarray:
    """
    Normalize an IQ window using the specified mode.

    Window format: shape (2, W) where window[0] is I and window[1] is Q.

    Supported modes:
    - None or "none": No normalization (returns copy with preserved dtype)
    - "unit_power": Normalize to unit power (mean(I² + Q²) = 1) [default]
    - "max_magnitude": Normalize by maximum absolute value across I and Q
    - "zero_mean_unit_variance": Standardization (zero mean, unit variance)
    - "mapminmax": Min-max scaling per channel to [-1, 1] range
    """
    if mode in (None, "none"):
        return window.astype(window.dtype, copy=False)

    if mode == "unit_power":
        # Normalize to unit power: mean(I² + Q²) = 1
        p = compute_window_power_lin(window)
        if p <= 0:
            # Zero power window: window is already all zeros (normalized state)
            return window
        # Ensure consistent dtype by casting result to original dtype
        return (window / np.sqrt(p + 1e-12)).astype(window.dtype, copy=False)

    if mode == "max_magnitude":
        # Normalize by maximum absolute value across both I and Q channels
        max_abs = np.max(np.abs(window))
        if max_abs <= 0:
            # Zero magnitude window: window is already all zeros (normalized state)
            return window
        # Ensure consistent dtype by casting result to original dtype
        return (window / (max_abs + 1e-12)).astype(window.dtype, copy=False)

    if mode == "zero_mean_unit_variance":
        # Standardization: zero mean, unit variance
        # Compute mean and std per channel
        mean = np.mean(window, axis=1, keepdims=True)
        std = np.std(window, axis=1, keepdims=True)
        # Avoid division by zero
        std = np.where(std <= 0, 1.0, std)
        # Ensure consistent dtype by casting result to original dtype
        return ((window - mean) / (std + 1e-12)).astype(window.dtype, copy=False)

    if mode == "mapminmax":
        # Min-max scaling per channel to [-1, 1] range
        # For each channel: scale from [min, max] to [lo, hi] = [-1, 1]
        lo, hi = -1.0, 1.0
        normalized = window.copy().astype(window.dtype)

        for ch_idx in range(2):  # I and Q channels
            ch = window[ch_idx]
            ch_min = np.min(ch)
            ch_max = np.max(ch)
            span = ch_max - ch_min

            if span <= 0:
                # Constant column: set to midpoint
                normalized[ch_idx] = np.full_like(ch, (lo + hi) / 2.0, dtype=window.dtype)
            else:
                # Min-max scaling: (x - min) / span maps to [0, 1], then scale to [lo, hi]
                normalized[ch_idx] = (lo + (ch - ch_min) / span * (hi - lo)).astype(window.dtype, copy=False)

        return normalized

    raise ValueError(
        f"Unknown normalize mode {mode}. "
        f"Supported modes: none, unit_power, max_magnitude, "
        f"zero_mean_unit_variance, mapminmax"
    )


# -----------------------------------------------------------------------------
# Noise floor detection and filtering
# -----------------------------------------------------------------------------

def detect_noise_floor_percentile(
    window_powers_db: np.ndarray,
    percentile: float = 5.0,
) -> float:
    """
    Detect noise floor using a percentile-based method.

    This method assumes that the lowest percentile of window powers
    represents the noise floor. This works well when most windows contain
    signal and only a small fraction are pure noise.

    Parameters
    ----------
    window_powers_db : np.ndarray
        Array of window powers in dB, shape (N,)
    percentile : float, default=5.0
        Percentile to use for noise floor estimation (0-100)

    Returns
    -------
    float
        Estimated noise floor in dB
    """
    if len(window_powers_db) == 0:
        raise ValueError("Cannot detect noise floor from empty array")
    return float(np.percentile(window_powers_db, percentile))


def detect_noise_floor_histogram(
    window_powers_db: np.ndarray,
    bins: int = 100,
    noise_fraction: float = 0.1,
) -> float:
    """
    Detect noise floor using histogram-based method.

    This method finds the mode (peak) of the power distribution in the
    lower power region, which typically represents the noise floor.

    Parameters
    ----------
    window_powers_db : np.ndarray
        Array of window powers in dB, shape (N,)
    bins : int, default=100
        Number of bins for histogram
    noise_fraction : float, default=0.1
        Fraction of lowest-power windows to analyze (0-1)

    Returns
    -------
    float
        Estimated noise floor in dB
    """
    if len(window_powers_db) == 0:
        raise ValueError("Cannot detect noise floor from empty array")

    # Focus on the lower power region
    sorted_powers = np.sort(window_powers_db)
    n_noise = max(1, int(len(sorted_powers) * noise_fraction))
    noise_region = sorted_powers[:n_noise]

    # Find the mode (most common value) in the noise region
    hist, bin_edges = np.histogram(noise_region, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mode_idx = np.argmax(hist)

    return float(bin_centers[mode_idx])


def detect_noise_floor_statistical(
    window_powers_db: np.ndarray,
    noise_fraction: float = 0.1,
    std_multiplier: float = 2.0,
) -> float:
    """
    Detect noise floor using statistical method.

    This method analyzes the lower power region and estimates noise floor
    as mean - N*std of the noise region.

    Parameters
    ----------
    window_powers_db : np.ndarray
        Array of window powers in dB, shape (N,)
    noise_fraction : float, default=0.1
        Fraction of lowest-power windows to analyze (0-1)
    std_multiplier : float, default=2.0
        Number of standard deviations below mean to use as threshold

    Returns
    -------
    float
        Estimated noise floor in dB
    """
    if len(window_powers_db) == 0:
        raise ValueError("Cannot detect noise floor from empty array")

    sorted_powers = np.sort(window_powers_db)
    n_noise = max(1, int(len(sorted_powers) * noise_fraction))
    noise_region = sorted_powers[:n_noise]

    mean_noise = np.mean(noise_region)
    std_noise = np.std(noise_region)

    return float(mean_noise - std_multiplier * std_noise)


def detect_noise_floor_from_windows(
    windows: Iterator[Tuple[np.ndarray, Dict]],
    method: str = "percentile",
    max_samples: int = 10000,
    **kwargs,
) -> float:
    """
    Detect noise floor from a sample of windows.

    This function samples windows from an iterator, computes their powers,
    and applies a noise floor detection method.

    Parameters
    ----------
    windows : Iterator[Tuple[np.ndarray, Dict]]
        Iterator yielding (window, metadata) tuples
        Window format: shape (2, W) where window[0] is I and window[1] is Q
    method : str, default="percentile"
        Detection method: "percentile", "histogram", or "statistical"
    max_samples : int, default=10000
        Maximum number of windows to sample for noise floor estimation
    **kwargs
        Additional arguments passed to the detection method

    Returns
    -------
    float
        Estimated noise floor in dB
    """
    powers_db = []

    for i, (window, _) in enumerate(windows):
        if i >= max_samples:
            break
        power_db = compute_window_power_db(window)
        powers_db.append(power_db)

    if len(powers_db) == 0:
        raise ValueError("No windows available for noise floor detection")

    powers_db = np.array(powers_db)

    if method == "percentile":
        percentile = kwargs.get("percentile", 5.0)
        return detect_noise_floor_percentile(powers_db, percentile=percentile)
    elif method == "histogram":
        bins = kwargs.get("bins", 100)
        noise_fraction = kwargs.get("noise_fraction", 0.1)
        return detect_noise_floor_histogram(
            powers_db, bins=bins, noise_fraction=noise_fraction
        )
    elif method == "statistical":
        noise_fraction = kwargs.get("noise_fraction", 0.1)
        std_multiplier = kwargs.get("std_multiplier", 2.0)
        return detect_noise_floor_statistical(
            powers_db, noise_fraction=noise_fraction, std_multiplier=std_multiplier
        )
    else:
        raise ValueError(
            f"Unknown noise floor detection method: {method}. "
            f"Supported methods: percentile, histogram, statistical"
        )


def determine_noise_threshold(
    noise_power_threshold_db: Optional[float | str],
    window_iterator: Iterator[Tuple[np.ndarray, Dict]],
    detection_method: str = "percentile",
    detection_params: Optional[Dict] = None,
    max_samples: int = 10000,
    verbose: bool = True,
) -> Optional[float]:
    """
    Determine the noise power threshold for filtering windows.

    This function handles both manual thresholds and auto-detection, including
    margin extraction and error handling.

    Parameters
    ----------
    noise_power_threshold_db : float, str, or None
        Noise filtering configuration:
        - None: No filtering (returns None)
        - float: Manual threshold in dB (returns as-is)
        - "auto": Auto-detect noise floor per file
    window_iterator : Iterator[Tuple[np.ndarray, Dict]]
        Iterator yielding (window, metadata) tuples for noise floor detection.
        Only used when noise_power_threshold_db == "auto".
    detection_method : str, default="percentile"
        Method for auto-detection: "percentile", "histogram", or "statistical"
    detection_params : dict, optional
        Parameters for the detection method. If "margin_db" is present, it will
        be extracted and added to the detected noise floor.
    max_samples : int, default=10000
        Maximum number of windows to sample for auto-detection
    verbose : bool, default=True
        If True, print detection results and warnings

    Returns
    -------
    float or None
        Noise power threshold in dB, or None if filtering is disabled or detection failed

    Examples
    --------
    >>> # Manual threshold
    >>> threshold = determine_noise_threshold(-70.0, None)
    >>> # threshold == -70.0
    >>>
    >>> # Auto-detection
    >>> windows = iter_windows_from_samples(samples, window_size=500, hop_size=500)
    >>> threshold = determine_noise_threshold(
    ...     "auto",
    ...     windows,
    ...     detection_method="percentile",
    ...     detection_params={"percentile": 1.0, "margin_db": 6.0}
    ... )
    """
    if noise_power_threshold_db is None:
        return None

    if isinstance(noise_power_threshold_db, (int, float)):
        # Manual threshold
        return float(noise_power_threshold_db)

    if noise_power_threshold_db == "auto":
        # Auto-detect noise floor
        try:
            # Extract margin_db if provided, otherwise use 0
            params = dict(detection_params or {})
            margin_db = params.pop("margin_db", 0.0)

            noise_floor_db = detect_noise_floor_from_windows(
                window_iterator,
                method=detection_method,
                max_samples=max_samples,
                **params,
            )

            # Add margin to get the threshold
            noise_threshold_db = noise_floor_db + margin_db

            if verbose:
                print(f"  Auto-detected noise floor: {noise_floor_db:.2f} dB")
                if margin_db > 0:
                    print(f"  Added margin: {margin_db:.1f} dB → threshold: {noise_threshold_db:.2f} dB")
                else:
                    print(f"  Using as threshold: {noise_threshold_db:.2f} dB")

            return noise_threshold_db

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to auto-detect noise floor: {e}")
                print(f"  Skipping noise filtering for this file")
            return None

    raise ValueError(
        f"Invalid noise_power_threshold_db value: {noise_power_threshold_db}. "
        f"Expected None, float, or 'auto'"
    )


def filter_window_by_power(
    window: np.ndarray,
    threshold_db: float,
) -> bool:
    """
    Check if a window passes the power threshold filter.

    Parameters
    ----------
    window : np.ndarray
        Window to check, shape (2, W)
    threshold_db : float
        Power threshold in dB. Windows below this threshold are filtered out.

    Returns
    -------
    bool
        True if window power >= threshold_db, False otherwise
    """
    power_db = compute_window_power_db(window)
    return power_db >= threshold_db
