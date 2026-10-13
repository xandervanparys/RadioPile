#!/usr/bin/env python3
"""
Utility script for inspecting noise floor in RF data files.

It can be used to:
1. Visualize power distributions to identify noise floors
2. Compare different noise floor detection methods
3. Determine appropriate thresholds for noise filtering

Usage:
    python scripts/inspect_noise_floor.py <file_path> [options]

Example:
    python scripts/inspect_noise_floor.py data/lte_wifi_dvbt/raw/gentbrugge/lte_g30_gentbrugge_f806MHz_r1.bin
"""

import argparse
from pathlib import Path
import sys

import numpy as np

# Add parent directory to path to import pipeline modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.io import iter_windows_for_file
from pipeline.noise_analysis import (
    analyze_file_noise_floor,
    compare_noise_floor_methods,
    plot_power_distribution,
)
from pipeline.preprocessing import compute_window_power_db


def main():
    parser = argparse.ArgumentParser(
        description="Inspect noise floor in RF data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "file_path",
        type=Path,
        help="Path to raw RF data file (.bin, .mat, or .h5)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=500,
        help="Window size in samples (default: 500)",
    )
    parser.add_argument(
        "--hop-size",
        type=int,
        default=500,
        help="Hop size in samples (default: 500)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of windows to sample (default: 10000)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["percentile", "histogram", "statistical"],
        default=["percentile", "histogram", "statistical"],
        help="Noise floor detection methods to use (default: all)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all detection methods side-by-side",
    )
    parser.add_argument(
        "--save-plot",
        type=Path,
        help="Save plot to file instead of displaying",
    )
    parser.add_argument(
        "--source-dtype",
        default="complex64",
        help="On-disk dtype of a .bin file, e.g. complex64 or int16 (default: complex64). "
             "Ignored for .mat and .h5 files.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't display plots (only print results)",
    )

    args = parser.parse_args()

    if not args.file_path.exists():
        print(f"Error: File not found: {args.file_path}")
        sys.exit(1)

    # Create window iterator function
    def window_iterator(path):
        for window, meta in iter_windows_for_file(
            path=path,
            window_size=args.window_size,
            hop_size=args.hop_size,
            pad_last=True,
            dtype=np.dtype(args.source_dtype),
        ):
            yield window, meta

    print(f"Analyzing: {args.file_path.name}")
    print(f"Window size: {args.window_size}, Hop size: {args.hop_size}")
    print(f"Max samples: {args.max_samples}")
    print()

    if args.compare:
        # Collect all window powers
        powers_db = []
        for i, (window, _) in enumerate(window_iterator(args.file_path)):
            if i >= args.max_samples:
                break
            power_db = compute_window_power_db(window)
            powers_db.append(power_db)

        if len(powers_db) == 0:
            print("Error: No windows found in file")
            sys.exit(1)

        powers_db = np.array(powers_db)

        print("Comparing detection methods:")
        print("-" * 50)
        results = compare_noise_floor_methods(
            powers_db,
            save_path=args.save_plot,
            show=not args.no_plot and args.save_plot is None,
        )

        print("\nResults:")
        for method, threshold in results.items():
            print(f"  {method:20s}: {threshold:8.2f} dB")
    else:
        # Analyze with specified methods
        results = analyze_file_noise_floor(
            args.file_path,
            window_iterator,
            max_samples=args.max_samples,
            methods=args.methods,
            plot=not args.no_plot,
            save_plot=args.save_plot,
            show=args.save_plot is None,
        )

        print("Noise floor estimates:")
        print("-" * 50)
        for method, threshold in results.items():
            print(f"  {method:15s}: {threshold:8.2f} dB")

        if len(results) > 1:
            avg_threshold = np.mean(list(results.values()))
            print(f"\n  Average:         {avg_threshold:8.2f} dB")
            print(f"\n  Suggested threshold: {avg_threshold:.1f} dB")


if __name__ == "__main__":
    main()
