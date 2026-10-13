"""
Dataset build script for Technology Recognition (LTE / WiFi / DVB-T) (Fontaine et al.).

All dataset configuration and assumptions are defined in config.py.
"""
from pathlib import Path
import re

from pipeline.hdf5_writer import build_hdf5_dataset
from pipeline.io import iter_windows_for_file
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
    LABEL_TO_WAVEFORM,
    LOCATION_OVERRIDES,
    WINDOW_SIZE,
    HOP_SIZE,
    PAD_LAST,
    NORMALIZE_MODE,
    NOISE_POWER_THRESHOLD_DB,
    SOURCE_DTYPE,
    DATA_DTYPE,
    GLOBAL_METADATA,
    WINDOW_METADATA_SCHEMA,
    CITATIONS,
    LICENSE,
)

try:
    from .config import (
        NOISE_FLOOR_DETECTION_METHOD,
        NOISE_FLOOR_DETECTION_PARAMS,
        NOISE_FLOOR_MAX_SAMPLES,
    )
except ImportError:
    NOISE_FLOOR_DETECTION_METHOD = "percentile"
    NOISE_FLOOR_DETECTION_PARAMS = {"percentile": 5.0}
    NOISE_FLOOR_MAX_SAMPLES = 10000

# ---------------------------------------------------------------------
# Dataset-specific parsing
# ---------------------------------------------------------------------

def infer_label_from_filename(path: Path) -> str:
    """
    Filename examples:
      dvbt10Msps_g30_rabot_f482MHz_r1.bin -> dvbt
      wf10Msps_g76_rabot_f5240MHz_r1.bin  -> wf
      lte_g30_uz_f806MHz_r1.bin           -> lte
    """
    first = path.name.split("_")[0]
    m = re.match(r"([A-Za-z]+)", first)
    return (m.group(1) if m else first).lower()


def parse_lte_wifi_dvbt_filename(path: Path) -> dict:
    """
    Parse metadata from filename:
      <tech><optional sps>_g<gain>_<location>_f<freqMHz>MHz_r<run>.bin
    """
    stem = path.stem
    parts = stem.split("_")

    # Robust fallback: at minimum tech + sample rate
    first = parts[0] if parts else stem
    m = re.match(r"(?P<tech>[A-Za-z]+)(?P<sps>\d+Msps)?", first)
    tech = (m.group("tech") if m else "").lower()
    sps_str = (m.group("sps") if m else None)

    if sps_str is None:
        sample_rate_hz = 1e6
    else:
        num = int(sps_str.replace("Msps", ""))
        sample_rate_hz = float(num) * 1e6

    gain_db = None
    location = None
    center_frequency_hz = None
    run_id = None

    if len(parts) >= 5:
        _, gain_part, location, freq_part, run_part = parts[:5]

        mg = re.match(r"g(?P<gain>\d+)", gain_part)
        if mg:
            gain_db = float(mg.group("gain"))

        mf = re.match(r"f(?P<freq_mhz>\d+)MHz", freq_part)
        if mf:
            center_frequency_hz = float(mf.group("freq_mhz")) * 1e6

        mr = re.match(r"r(?P<run>\d+)", run_part)
        if mr:
            run_id = mr.group("run")   # stored as string for cross-dataset dtype consistency

    return {
        "tech_label": tech,
        "sample_rate_hz": sample_rate_hz,
        "gain_db": gain_db,
        "location": resolve_location(path, location),
        "center_frequency_hz": center_frequency_hz,
        "recording_id": run_id,
    }


def resolve_location(path: Path, parsed_location):
    """
    Return the capture location for a file.

    The filename token is authoritative unless the file is listed in
    LOCATION_OVERRIDES (keyed by "<folder>/<filename>"). A folder/filename
    disagreement that is not covered by an override is reported, not corrected.
    """
    key = f"{path.parent.name}/{path.name}"
    if key in LOCATION_OVERRIDES:
        return LOCATION_OVERRIDES[key]
    if parsed_location is not None and path.parent.name != parsed_location:
        print(
            f"WARNING: {key}: folder '{path.parent.name}' disagrees with filename "
            f"location '{parsed_location}'; using the filename"
        )
    return parsed_location


# ---------------------------------------------------------------------
# Dataset-specific window iterator
# ---------------------------------------------------------------------

def iter_lte_wifi_dvbt_windows(path: Path):
    """
    Yield (window, label_name, meta) for one file.

    Uses reusable mechanics from pipeline.io and pipeline.preprocessing,
    and only adds:
    - label inference
    - filename-derived RF metadata
    - noise filtering (if configured)
    """
    # Skip non-data files (e.g. .gitkeep, README, directories)
    if not path.is_file() or path.suffix.lower() != ".bin":
        print(f"skipping {path.parts[-1]}")
        return

    raw_label = infer_label_from_filename(path)

    if LABEL_REMAP is not None:
        if raw_label not in LABEL_REMAP:
            raise ValueError(f"Label '{raw_label}' missing from LABEL_REMAP")
        label_name = LABEL_REMAP[raw_label]
    else:
        label_name = raw_label

    # Parse filename metadata once (same for all windows from this file)
    file_meta = parse_lte_wifi_dvbt_filename(path)

    # Determine noise threshold for this file
    noise_threshold_db = None
    if NOISE_POWER_THRESHOLD_DB is not None:
        noise_threshold_db = determine_noise_threshold(
            noise_power_threshold_db=NOISE_POWER_THRESHOLD_DB,
            window_iterator=iter_windows_for_file(
                path=path,
                window_size=WINDOW_SIZE,
                hop_size=HOP_SIZE,
                pad_last=PAD_LAST,
                dtype=SOURCE_DTYPE,
                output_dtype=DATA_DTYPE,
            ),
            detection_method=NOISE_FLOOR_DETECTION_METHOD,
            detection_params=NOISE_FLOOR_DETECTION_PARAMS,
            max_samples=NOISE_FLOOR_MAX_SAMPLES,
        )

    # Track filtering statistics
    total_windows = 0
    filtered_windows = 0

    for window, meta in iter_windows_for_file(
            path=path,
            window_size=WINDOW_SIZE,
            hop_size=HOP_SIZE,
            pad_last=PAD_LAST,
            dtype=SOURCE_DTYPE,
            output_dtype=DATA_DTYPE,
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
            "sample_rate_hz": file_meta.get("sample_rate_hz"),
            "center_frequency_hz": file_meta.get("center_frequency_hz"),
        })

        # Filename-derived metadata (optional per file)
        for key in ("gain_db", "location", "recording_id"):
            if key in file_meta and file_meta[key] is not None:
                # Handle location: use empty string if None, otherwise use the value
                if key == "location":
                    meta_out[key] = file_meta[key] or ""
                else:
                    meta_out[key] = file_meta[key]

        # Technology label and modulation — set explicitly for aggregation / multi-target use
        raw_label = file_meta.get("tech_label", "")
        meta_out["technology_type"] = LABEL_REMAP.get(raw_label, "unknown")  # canonical name
        meta_out["modulation_type"] = "unknown"  # subcarrier modulation not resolvable without MCS
        meta_out["waveform_type"]   = LABEL_TO_WAVEFORM.get(raw_label, "unknown")
        # 802.11 variant is not documented in the source dataset; other
        # technologies have no such variant
        meta_out["wifi_standard"] = "unknown" if meta_out["technology_type"] == "WiFi" else ""

        # Apply per-window normalization
        window = normalize_window(window, NORMALIZE_MODE)

        yield window, label_name, meta_out

    # Print filtering statistics
    if noise_threshold_db is not None and total_windows > 0:
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
        window_iterator_fn=iter_lte_wifi_dvbt_windows,
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
            "upstream_dataset": "Technology-Recognition-dataset-of-real-life-LTE-WiFi-and-DVB-T",
            "dataset_url": GLOBAL_METADATA["dataset_url"],
            "locations": ["gentbrugge", "igent", "merelbeke", "rabot", "reep", "uz"],
            "script_version": SCRIPT_VERSION,
        },
    )

    print(f"Croissant metadata written to {OUTPUT_CROISSANT_PATH}")
    print(f"Dataset written to {OUTPUT_H5_PATH}")

    print("=== Done ===")


if __name__ == "__main__":
    main()
