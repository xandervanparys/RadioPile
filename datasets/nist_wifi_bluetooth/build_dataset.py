"""
Dataset build script for NIST WiFi IQ Recordings (Tschimben et al.).

All configuration and assumptions are in config.py.

Source format per HDF5 file:
  - 180 captures, each keyed by a timestamp string
  - Each capture: 60,000,000 int16 values (interleaved I/Q at 30 Msps for 1 second)
  - Conversion: int16 / 32768 → float32 in [-1, 1], then reshape to (-1, 2) for [I, Q] pairs

Window extraction per capture:
  1. Read int16, normalize to float32 in [-1, 1]
  2. Split interleaved into I and Q channels
  3. Window into WINDOW_SIZE=500 non-overlapping chunks
  4. Discard windows below the per-file noise power threshold
  5. Yield all remaining signal windows — no cap applied

Noise floor detection:
  - Run once per HDF5 file using the first capture as the sample source
  - Method: percentile (1st percentile + 6 dB margin), same as sub_ghz
  - All subsequent captures in the file use the same threshold

All signal windows are extracted. Users of the processed dataset should subsample as needed.
"""

import h5py
import numpy as np

from pipeline.hdf5_writer import build_hdf5_dataset
from pipeline.windowing import iter_windows_from_samples
from pipeline.preprocessing import normalize_window, filter_window_by_power, determine_noise_threshold
from metadata.croissant import generate_croissant_from_hdf5
from datasets.identifiers import RELEASE_DATE, RELEASE_VERSION

from .config import (
    DATASET_NAME,
    SCRIPT_VERSION,
    RAW_DATA_ROOT,
    OUTPUT_H5_PATH,
    OUTPUT_CROISSANT_PATH,
    FILE_CONFIG,
    FILE_METADATA,
    TECHNOLOGY_TYPES,
    WINDOW_SIZE,
    HOP_SIZE,
    PAD_LAST,
    NORMALIZE_MODE,
    DATA_DTYPE,
    GLOBAL_METADATA,
    WINDOW_METADATA_SCHEMA,
    SAMPLE_RATE_HZ,
    CITATIONS,
    LICENSE,
    NOISE_POWER_THRESHOLD_DB,
    NOISE_FLOOR_DETECTION_METHOD,
    NOISE_FLOOR_DETECTION_PARAMS,
    NOISE_FLOOR_MAX_SAMPLES,
)


def _load_capture_iq(f: h5py.File, capture_key: str) -> np.ndarray:
    """Read one capture from an open HDF5 file and return float32 IQ array (N, 2)."""
    raw = np.array(f[capture_key], dtype=np.float32) / 32768.0
    return raw.reshape(-1, 2)  # (30M, 2): column 0 = I, column 1 = Q


def iter_nist_windows():
    """
    Yield (window, label, meta) for all signal windows across the three NIST HDF5 files.

    For each file:
      1. Detect noise floor from the first capture (power-based, same method as sub_ghz)
      2. Iterate all 180 captures, filtering windows below the noise threshold
    """
    for filename_stem, (center_frequency_hz, environment, label) in FILE_CONFIG.items():
        h5_path = RAW_DATA_ROOT / f"{filename_stem}.h5"

        if not h5_path.exists():
            raise FileNotFoundError(
                f"Expected file not found: {h5_path}\n"
                f"Download {filename_stem}.h5 from https://doi.org/10.18434/mds2-2731"
            )

        # File-level MCS metadata from NIST TN 2237 — constant for all windows in this file.
        file_meta = FILE_METADATA[filename_stem]

        print(f"  Loading {filename_stem}.h5 → label '{label}' "
              f"({center_frequency_hz / 1e6:.0f} MHz, {environment}, "
              f"{file_meta['wifi_standard']}, {file_meta['modulation_type']})")

        with h5py.File(h5_path, "r") as f:
            capture_keys = list(f.keys())

        # ------------------------------------------------------------------
        # Step 1: Detect noise floor from the first capture
        # Using the same pattern as sub_ghz: iter_windows_from_samples on a
        # loaded IQ array. Reading only the first capture (~120 MB) avoids
        # having to load all 180 captures (~21.6 GB) at once.
        # ------------------------------------------------------------------
        with h5py.File(h5_path, "r") as f:
            iq_sample = _load_capture_iq(f, capture_keys[0])

        noise_threshold_db = determine_noise_threshold(
            noise_power_threshold_db=NOISE_POWER_THRESHOLD_DB,
            window_iterator=iter_windows_from_samples(
                samples=iq_sample,
                window_size=WINDOW_SIZE,
                hop_size=HOP_SIZE,
                pad_last=PAD_LAST,
            ),
            detection_method=NOISE_FLOOR_DETECTION_METHOD,
            detection_params=NOISE_FLOOR_DETECTION_PARAMS,
            max_samples=NOISE_FLOOR_MAX_SAMPLES,
        )

        # ------------------------------------------------------------------
        # Step 2: Iterate all captures, filter, normalize, and yield
        # ------------------------------------------------------------------
        file_yielded = 0
        file_filtered = 0

        with h5py.File(h5_path, "r") as f:
            for i, capture_key in enumerate(capture_keys):
                iq = _load_capture_iq(f, capture_key)
                I_all = iq[:, 0]
                Q_all = iq[:, 1]

                n = len(I_all) // WINDOW_SIZE
                I_wins = I_all[:n * WINDOW_SIZE].reshape(n, WINDOW_SIZE)
                Q_wins = Q_all[:n * WINDOW_SIZE].reshape(n, WINDOW_SIZE)

                for idx in range(n):
                    window = np.stack(
                        [I_wins[idx].astype(DATA_DTYPE),
                         Q_wins[idx].astype(DATA_DTYPE)],
                        axis=0,
                    )  # (2, 500)

                    if noise_threshold_db is not None:
                        if not filter_window_by_power(window, noise_threshold_db):
                            file_filtered += 1
                            continue

                    window = normalize_window(window, NORMALIZE_MODE)

                    meta = {
                        "source_file":     str(h5_path),
                        "technology_type": label,
                        "modulation_type": file_meta["modulation_type"],  # from FILE_METADATA (NIST TN 2237)
                        "waveform_type":   "OFDM",                        # all 802.11 variants in this dataset are OFDM-based
                        "recording_id":    capture_key,
                        "center_frequency_hz":  center_frequency_hz,
                        "environment":     environment,
                        "sample_rate_hz":  SAMPLE_RATE_HZ,
                        "sample_index":    int(idx),
                        "wifi_standard":   file_meta["wifi_standard"],    # from FILE_METADATA (NIST TN 2237)
                    }
                    yield window, label, meta
                    file_yielded += 1

                if (i + 1) % 20 == 0:
                    print(f"    [{i+1}/{len(capture_keys)} captures] "
                          f"{file_yielded:,} signal / {file_filtered:,} filtered so far")

        total = file_yielded + file_filtered
        filter_rate = (file_filtered / total * 100) if total > 0 else 0.0
        print(f"    {file_yielded:,} signal windows kept, "
              f"{file_filtered:,} filtered ({filter_rate:.1f}%) "
              f"from {len(capture_keys)} captures")


def main():
    print(f"=== Building dataset: {DATASET_NAME} ===")
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Sampling rate: {SAMPLE_RATE_HZ / 1e6:g} Msps → window size: {WINDOW_SIZE} samples "
          f"({WINDOW_SIZE / SAMPLE_RATE_HZ * 1e6:.1f} µs)")
    print(f"Data dtype: {DATA_DTYPE}")
    print(f"Normalization mode: {NORMALIZE_MODE}")
    print(f"Noise filtering: AUTO (method={NOISE_FLOOR_DETECTION_METHOD}, "
          f"params={NOISE_FLOOR_DETECTION_PARAMS})")
    print(f"Source files ({len(FILE_CONFIG)}): {', '.join(FILE_CONFIG.keys())}")
    print("Extracting all signal windows (no cap) — expected ~20.6M total, ~89 GB")
    print()

    missing = [
        f"{stem}.h5"
        for stem in FILE_CONFIG
        if not (RAW_DATA_ROOT / f"{stem}.h5").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing source files in {RAW_DATA_ROOT}:\n"
            + "\n".join(f"  {f}" for f in missing)
            + f"\n\nDownload from https://doi.org/10.18434/mds2-2731"
        )

    summary = build_hdf5_dataset(
        output_path=OUTPUT_H5_PATH,
        window_iterator=iter_nist_windows(),
        batch_size=64,
        data_dtype=DATA_DTYPE,
        global_metadata=GLOBAL_METADATA,
        window_metadata_schema=WINDOW_METADATA_SCHEMA,
    )

    print(f"\nCreated dataset with {summary['total_windows']:,} windows")
    print(f"Label names: {summary.get('label_names', [])}")
    print("Writing Croissant metadata...")

    generate_croissant_from_hdf5(
        h5_path=OUTPUT_H5_PATH,
        output_path=OUTPUT_CROISSANT_PATH,
        doi=GLOBAL_METADATA["doi"],
        related_identifiers=GLOBAL_METADATA["related_identifiers"],
        version=RELEASE_VERSION,
        date_published=RELEASE_DATE,
        dataset_name=DATASET_NAME,
        description=GLOBAL_METADATA["description"],
        task_type="classification",
        labels=summary.get("label_names", TECHNOLOGY_TYPES),
        citations=CITATIONS,
        license=LICENSE,
        preprocessing={
            "window_size": WINDOW_SIZE,
            "hop_size": HOP_SIZE,
            "pad_last": PAD_LAST,
            "normalization": NORMALIZE_MODE,
            "noise_filtering": {
                "threshold_db": NOISE_POWER_THRESHOLD_DB,
                "detection_method": NOISE_FLOOR_DETECTION_METHOD,
                "detection_params": NOISE_FLOOR_DETECTION_PARAMS,
                "max_samples": NOISE_FLOOR_MAX_SAMPLES,
            },
        },
        provenance={
            "upstream_dataset": "Wi-Fi and Bluetooth I/Q Recordings in the 2.4 GHz and 5 GHz Bands with Low-Cost Software Defined Radios",
            "dataset_url": GLOBAL_METADATA["dataset_url"],
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "hardware": "USRP B200-mini-i",
            "script_version": SCRIPT_VERSION,
        },
    )

    print(f"Croissant metadata written to {OUTPUT_CROISSANT_PATH}")
    print(f"Dataset written to {OUTPUT_H5_PATH}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
