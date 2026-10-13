"""
Dataset build script for Sub-GHz Technology Recognition (Fontaine et al.).

All dataset configuration and assumptions are defined in config.py.
"""
from pathlib import Path
import re

from pipeline.hdf5_writer import build_hdf5_dataset
from pipeline.windowing import iter_windows_from_samples
from pipeline.preprocessing import (
    normalize_window,
    filter_window_by_power,
    determine_noise_threshold,
)
from metadata.croissant import generate_croissant_from_hdf5
from datasets.identifiers import RELEASE_DATE, RELEASE_VERSION

from .config import (
    DATASET_NAME,
    SCRIPT_VERSION,
    RAW_DATA_ROOT,
    OUTPUT_H5_PATH,
    OUTPUT_CROISSANT_PATH,
    LABEL_REMAP,
    LABEL_TO_MODULATION,
    LABEL_TO_WAVEFORM,
    WIFI_STANDARD,
    IEEE802154_VARIANT,
    WINDOW_SIZE,
    HOP_SIZE,
    PAD_LAST,
    NORMALIZE_MODE,
    NOISE_POWER_THRESHOLD_DB,
    DATA_DTYPE,
    GLOBAL_METADATA,
    WINDOW_METADATA_SCHEMA,
    CITATIONS,
    LICENSE,
)

import scipy.io
import numpy as np

try:
    from .config import (
        NOISE_FLOOR_DETECTION_METHOD,
        NOISE_FLOOR_DETECTION_PARAMS,
        NOISE_FLOOR_MAX_SAMPLES,
    )
except ImportError:
    NOISE_FLOOR_DETECTION_METHOD = "percentile"
    NOISE_FLOOR_DETECTION_PARAMS = {"percentile": 1.0, "margin_db": 6.0}
    NOISE_FLOOR_MAX_SAMPLES = 10000

# ---------------------------------------------------------------------
# Dataset-specific parsing
# ---------------------------------------------------------------------

def infer_label_from_path(path: Path) -> str:
    """
    Infer technology label from directory name.
    Example: raw/867.4/Sigfox/*.mat -> sigfox
    """
    parts_lower = [p.lower() for p in path.parts]

    # 1) Exact match on any path component (directory names etc.)
    for key in LABEL_REMAP:
        if key in parts_lower:
            return key

    # 2) Substring match on any component (handles e.g. 'Sigfox_files')
    for part in parts_lower:
        for key in LABEL_REMAP:
            if key in part:
                return key

    # 3) Fallback: first token of the filename stem (before '_')
    stem_first = path.stem.split("_")[0].lower()
    if stem_first in LABEL_REMAP:
        return stem_first

    raise ValueError(f"Could not infer label from path: {path}")


MCS_TO_MODULATION = {
    0: "BPSK",
    2: "OQPSK",
    6: "16QAM",
    7: "64QAM",
    10: "BPSK",
}


def parse_metadata_filename(path: Path, label: str) -> dict:
    """
    Parse Sub-GHz filename for additional metadata.

    Expected patterns (examples):
      80211ah_mcs0_chan6_g0.0dB_att3dB_freq867.4MHz_0.mat
      sigfox_g0.0dB_att25dB_freq867.4MHz_0.mat

    Returns a dict with optional keys:
      - mcs
      - channel
      - gain_db
      - attenuation_db
      - center_frequency_hz (if present)
      - modulation_type
      - spreading_factor (LoRa only; -1 if not found in filename)
    """
    meta = {}

    name = path.stem.lower()

    # MCS
    m = re.search(r"mcs(?P<mcs>\d+)", name)
    if m:
        meta["mcs"] = int(m.group("mcs"))

    # Channel
    m = re.search(r"chan(?P<chan>\d+)", name)
    if m:
        meta["channel"] = int(m.group("chan"))

    # Gain (dB)
    m = re.search(r"g(?P<gain>-?\d+(\.\d+)?)db", name)
    if m:
        meta["gain_db"] = float(m.group("gain"))

    # Attenuation (dB)
    m = re.search(r"att(?P<att>\d+(\.\d+)?)db", name)
    if m:
        meta["attenuation_db"] = float(m.group("att"))

    # Center frequency from filename
    m = re.search(r"freq(?P<freq>\d+(\.\d+)?)mhz", name)
    if m:
        meta["center_frequency_hz"] = float(m.group("freq")) * 1e6

    # Run ID
    m = re.search(r"_(?P<run_id>\d+)$", name)
    if m:
        meta["recording_id"] = m.group("run_id")   # stored as string for cross-dataset dtype consistency

    # Subcarrier modulation: resolved per MCS when available, else falls back to
    # LABEL_TO_MODULATION (returns "unknown" for OFDM-based technologies without MCS).
    # MCS_TO_MODULATION is safe here because 80211ah (MCS 0, 7, 10) and sunofdm
    # (MCS 2, 6) use non-overlapping MCS values in this dataset.
    if "mcs" in meta:
        modulation = MCS_TO_MODULATION.get(meta["mcs"], "unknown")
    else:
        modulation = LABEL_TO_MODULATION.get(label, "unknown")
    meta["modulation_type"] = modulation

    # Waveform type: fixed per technology, MCS-independent (OFDM, CSS, FSK, PSK, noise).
    meta["waveform_type"] = LABEL_TO_WAVEFORM.get(label, "unknown")

    # Spreading factor: LoRa only (SF7 or SF12, encoded in filename, e.g. loraSF7_...)
    # This is a transmission parameter separate from the modulation type (CSS).
    if label == "lora":
        m = re.search(r"sf(?P<sf>\d+)", path.stem.lower())
        meta["spreading_factor"] = int(m.group("sf")) if m else -1

    return meta


# ---------------------------------------------------------------------
# Dataset-specific window iterator
# ---------------------------------------------------------------------

def iter_subghz_windows(path: Path):
    """
    Yield (window, label_name, meta) for one Sub-GHz .mat file.
    """
    if not path.is_file() or path.suffix.lower() != ".mat":
        return

    raw_label = infer_label_from_path(path)

    if raw_label not in LABEL_REMAP:
        raise ValueError(f"Label '{raw_label}' missing from LABEL_REMAP")

    label_name = LABEL_REMAP[raw_label]
    filename_meta = parse_metadata_filename(path, raw_label)

    mat = scipy.io.loadmat(path)
    if "IQ_samples" not in mat:
        raise ValueError(f"'IQ_samples' not found in {path}")

    iq = mat["IQ_samples"].squeeze()

    if iq.ndim != 1:
        raise ValueError(f"Expected 1-D complex IQ array in {path}, got shape {iq.shape}")

    if not np.iscomplexobj(iq):
        raise ValueError(f"Expected complex-valued IQ samples in {path}")

    # Convert complex IQ → [N, 2] real-valued array
    iq = np.stack([iq.real, iq.imag], axis=1).astype(DATA_DTYPE)

    is_noise_class = (raw_label == "noise" or label_name == "Noise")

    noise_threshold_db = None
    if not is_noise_class and NOISE_POWER_THRESHOLD_DB is not None:
        noise_threshold_db = determine_noise_threshold(
            noise_power_threshold_db=NOISE_POWER_THRESHOLD_DB,
            window_iterator=iter_windows_from_samples(
                samples=iq,
                window_size=WINDOW_SIZE,
                hop_size=HOP_SIZE,
                pad_last=PAD_LAST,
            ),
            detection_method=NOISE_FLOOR_DETECTION_METHOD,
            detection_params=NOISE_FLOOR_DETECTION_PARAMS,
            max_samples=NOISE_FLOOR_MAX_SAMPLES,
        )

    # Track filtering statistics
    total_windows = 0
    filtered_windows = 0

    for window, meta in iter_windows_from_samples(
        samples=iq,
        window_size=WINDOW_SIZE,
        hop_size=HOP_SIZE,
        pad_last=PAD_LAST,
    ):
        total_windows += 1

        # Apply noise filtering (before normalization)
        if noise_threshold_db is not None:
            if not filter_window_by_power(window, noise_threshold_db):
                filtered_windows += 1
                continue

        meta_out = dict(meta)

        # Base metadata
        meta_out.update({
            "source_file": str(path),
        })

        # Canonical technology label
        meta_out["technology_type"] = label_name

        # Technology variant sub-classification (empty string = not applicable)
        meta_out["wifi_standard"]      = WIFI_STANDARD.get(raw_label, "")
        meta_out["ieee802154_variant"] = IEEE802154_VARIANT.get(raw_label, "")

        # Filename-derived metadata
        for key in [field.key for field in WINDOW_METADATA_SCHEMA]:
            if key in filename_meta:
                meta_out[key] = filename_meta[key]

        # Apply per-window normalization
        window = normalize_window(window, NORMALIZE_MODE)

        yield window, label_name, meta_out

    # Print filtering statistics
    if is_noise_class:
        print(f"  Skipping noise filtering (noise class file - all windows kept)")
    elif noise_threshold_db is not None and total_windows > 0:
        filter_rate = (filtered_windows / total_windows) * 100
        print(f"  Filtered {filtered_windows}/{total_windows} windows ({filter_rate:.1f}%)")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print(f"=== Building dataset: {DATASET_NAME} ===")
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Window size: {WINDOW_SIZE}, hop size: {HOP_SIZE}, pad_last: {PAD_LAST}")
    print(f"Data dtype: {DATA_DTYPE}")
    print(f"Normalization mode: {NORMALIZE_MODE or 'none'}")
    if NOISE_POWER_THRESHOLD_DB == "auto":
        print(f"Noise filtering: AUTO (method={NOISE_FLOOR_DETECTION_METHOD})")
    elif NOISE_POWER_THRESHOLD_DB is not None:
        print(f"Noise power threshold (dB): {NOISE_POWER_THRESHOLD_DB}")
    else:
        print("Noise filtering: DISABLED")

    summary = build_hdf5_dataset(
        data_dir=RAW_DATA_ROOT,
        output_path=OUTPUT_H5_PATH,
        window_iterator_fn=iter_subghz_windows,
        batch_size=64,
        data_dtype=DATA_DTYPE,
        global_metadata=GLOBAL_METADATA,
        window_metadata_schema=WINDOW_METADATA_SCHEMA,
    )

    print(f"Created dataset with {summary['total_windows']} windows")
    print("Writing Croissant metadata...")

    # Build preprocessing metadata
    preprocessing = {
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "pad_last": PAD_LAST,
        "normalization": NORMALIZE_MODE or "none",
    }

    # Add noise filtering metadata only if filtering is enabled
    if NOISE_POWER_THRESHOLD_DB is not None:
        noise_filtering = {"threshold_db": NOISE_POWER_THRESHOLD_DB}
        if NOISE_POWER_THRESHOLD_DB == "auto":
            noise_filtering["detection_method"] = NOISE_FLOOR_DETECTION_METHOD
            noise_filtering["detection_params"] = NOISE_FLOOR_DETECTION_PARAMS
            noise_filtering["max_samples"] = NOISE_FLOOR_MAX_SAMPLES
        preprocessing["noise_filtering"] = noise_filtering

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
        labels=summary.get("label_names", []),
        citations=CITATIONS,
        license=LICENSE,
        preprocessing=preprocessing,
        provenance={
            "upstream_dataset": "Sub-GHz IQ Signals Dataset",
            "dataset_url": GLOBAL_METADATA["dataset_url"],
            "capture_setup": "RTL-SDR, over the air, 2.048 Msps",
            "script_version": SCRIPT_VERSION,
        },
    )

    print(f"Croissant metadata written to {OUTPUT_CROISSANT_PATH}")
    print(f"Dataset written to {OUTPUT_H5_PATH}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
