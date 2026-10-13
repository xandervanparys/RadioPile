"""
Noise floor analysis and visualization utilities.

This module provides tools for:
- Visualizing power distributions to identify noise floors
- Interactive noise floor detection
- Comparing different detection methods
"""

import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, Dict, Optional, List
import matplotlib.pyplot as plt

from pipeline.preprocessing import (
    compute_window_power_db,
    detect_noise_floor_percentile,
    detect_noise_floor_histogram,
    detect_noise_floor_statistical,
    detect_noise_floor_from_windows,
)


def plot_power_distribution(
    window_powers_db: np.ndarray,
    noise_floor_db: Optional[float] = None,
    title: str = "Window Power Distribution",
    save_path: Optional[Path] = None,
    show: bool = True,
) -> None:
    """
    Plot histogram of window powers with optional noise floor threshold.

    Parameters
    ----------
    window_powers_db : np.ndarray
        Array of window powers in dB
    noise_floor_db : float, optional
        Noise floor threshold to display as vertical line
    title : str, default="Window Power Distribution"
        Plot title
    save_path : Path, optional
        If provided, save the plot to this path
    show : bool, default=True
        If True, display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(window_powers_db, bins=100, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Power (dB)")
    ax.set_ylabel("Number of Windows")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if noise_floor_db is not None:
        ax.axvline(
            noise_floor_db,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Noise Floor: {noise_floor_db:.2f} dB",
        )
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def analyze_file_noise_floor(
    file_path: Path,
    window_iterator_fn: callable,
    max_samples: int = 10000,
    methods: Optional[List[str]] = None,
    plot: bool = True,
    save_plot: Optional[Path] = None,
    show: bool = True,
) -> Dict[str, float]:
    """
    Analyze noise floor for a single file using multiple methods.

    Parameters
    ----------
    file_path : Path
        Path to the raw data file
    window_iterator_fn : callable
        Function that takes a Path and yields (window, metadata) tuples
    max_samples : int, default=10000
        Maximum number of windows to sample
    methods : List[str], optional
        List of methods to use. If None, uses all methods.
        Options: "percentile", "histogram", "statistical"
    plot : bool, default=True
        If True, draw the power distribution plot
    save_plot : Path, optional
        If provided, save the plot to this path (the plot is drawn even if plot is False)
    show : bool, default=True
        If True, display the plot in a window

    Returns
    -------
    Dict[str, float]
        Dictionary mapping method names to estimated noise floor (dB)
    """
    if methods is None:
        methods = ["percentile", "histogram", "statistical"]

    # Collect window powers
    powers_db = []
    windows = window_iterator_fn(file_path)

    for i, (window, _) in enumerate(windows):
        if i >= max_samples:
            break
        power_db = compute_window_power_db(window)
        powers_db.append(power_db)

    if len(powers_db) == 0:
        raise ValueError(f"No windows found in file: {file_path}")

    powers_db = np.array(powers_db)

    # Apply all methods
    results = {}

    if "percentile" in methods:
        results["percentile"] = detect_noise_floor_percentile(powers_db, percentile=5.0)

    if "histogram" in methods:
        results["histogram"] = detect_noise_floor_histogram(
            powers_db, bins=100, noise_fraction=0.1
        )

    if "statistical" in methods:
        results["statistical"] = detect_noise_floor_statistical(
            powers_db, noise_fraction=0.1, std_multiplier=2.0
        )

    # Plot if requested, or if a file to save it to was given
    if plot or save_plot is not None:
        # Use average of all methods as representative threshold
        avg_noise_floor = np.mean(list(results.values()))
        plot_power_distribution(
            powers_db,
            noise_floor_db=avg_noise_floor,
            title=f"Power Distribution: {file_path.name}",
            save_path=save_plot,
            show=show and plot,
        )

    return results


def compare_noise_floor_methods(
    window_powers_db: np.ndarray,
    save_path: Optional[Path] = None,
    show: bool = True,
) -> Dict[str, float]:
    """
    Compare different noise floor detection methods on the same data.

    Parameters
    ----------
    window_powers_db : np.ndarray
        Array of window powers in dB
    save_path : Path, optional
        If provided, save the comparison plot
    show : bool, default=True
        If True, display the plot

    Returns
    -------
    Dict[str, float]
        Dictionary mapping method names to estimated noise floor (dB)
    """
    methods = {
        "percentile (5%)": detect_noise_floor_percentile(window_powers_db, percentile=5.0),
        "percentile (10%)": detect_noise_floor_percentile(window_powers_db, percentile=10.0),
        "histogram": detect_noise_floor_histogram(window_powers_db),
        "statistical": detect_noise_floor_statistical(window_powers_db),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Histogram with all thresholds
    ax1.hist(window_powers_db, bins=100, alpha=0.7, edgecolor="black")
    colors = ["red", "orange", "green", "blue"]
    for (name, threshold), color in zip(methods.items(), colors):
        ax1.axvline(
            threshold,
            color=color,
            linestyle="--",
            linewidth=2,
            label=f"{name}: {threshold:.2f} dB",
        )
    ax1.set_xlabel("Power (dB)")
    ax1.set_ylabel("Number of Windows")
    ax1.set_title("Power Distribution with Noise Floor Estimates")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bar chart comparing methods
    names = list(methods.keys())
    values = list(methods.values())
    ax2.barh(names, values, alpha=0.7)
    ax2.set_xlabel("Noise Floor (dB)")
    ax2.set_title("Comparison of Detection Methods")
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return methods
