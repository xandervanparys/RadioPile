"""
Dataset build script for RadioPile, the aggregated multi-task RF IQ collection.

Combines multiple processed datasets into a unified dataset with:
- Unified window sizes
- Multi-target metadata (technology, modulation, radar signal type, waveform family)
- Consistent preprocessing
"""

from pathlib import Path
from typing import Iterator, Tuple, Dict, Any, Optional
import importlib

import h5py
import numpy as np

from pipeline.hdf5_writer import build_hdf5_dataset
from pipeline.labels import reset_label_registry
from pipeline.preprocessing import normalize_window
from metadata.croissant import generate_croissant_from_hdf5
from datasets.identifiers import RELEASE_DATE, RELEASE_VERSION
from datasets.registry import get_dataset, DATASET_REGISTRY

from .config import (
    DATASET_NAME,
    DATASET_DESCRIPTION,
    SCRIPT_VERSION,
    OUTPUT_H5_PATH,
    OUTPUT_CROISSANT_PATH,
    SELECTED_DATASETS,
    SOURCE_DATASETS,
    TARGET_WINDOW_SIZE,
    PADDING_STRATEGY,
    CROP_POLICY,
    PRIMARY_LABEL_TYPE,
    NORMALIZE_MODE,
    DATA_DTYPE,
    GLOBAL_METADATA,
    WINDOW_METADATA_SCHEMA,
    CITATIONS,
    COLLECTION_CITATION,
    LICENSE,
    RECORD_A_DOI,
    RECORD_B_DOI,
    CODE_DOI,
)


def _decode_str_array(arr: np.ndarray) -> list[str]:
    """Decode string array from HDF5 (handles bytes and strings)."""
    out = []
    for x in arr.tolist():
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def build_individual_datasets(selected_datasets: Dict[str, Dict]) -> Dict[str, Path]:
    """
    Build individual datasets from raw data if needed.

    Returns dictionary mapping dataset names to processed HDF5 paths.
    """
    dataset_paths = {}

    for dataset_name, config in selected_datasets.items():
        if not config.get("enabled", True):
            continue

        try:
            entry = get_dataset(dataset_name)
        except KeyError:
            print(
                f"Warning: Dataset '{dataset_name}' not found in registry, skipping...")
            continue

        use_existing = config.get("use_existing", True)

        # Check if processed dataset exists
        if use_existing and entry.processed_h5_path.exists():
            print(
                f"Using existing dataset: {dataset_name} at {entry.processed_h5_path}")
            dataset_paths[dataset_name] = entry.processed_h5_path
            continue

        # Build from raw data
        if entry.build_module is None:
            raise ValueError(
                f"Dataset '{dataset_name}' has no build_module specified")

        print(f"Building dataset: {dataset_name} from raw data...")
        try:
            # Import and run build script
            module = importlib.import_module(
                f"{entry.build_module}.build_dataset")
            if hasattr(module, "main"):
                module.main()
                if entry.processed_h5_path.exists():
                    dataset_paths[dataset_name] = entry.processed_h5_path
                else:
                    raise FileNotFoundError(
                        f"Build completed but {entry.processed_h5_path} not found")
            else:
                raise AttributeError(
                    f"Module {entry.build_module}.build_dataset has no main() function")
        except Exception as e:
            print(f"Error building dataset '{dataset_name}': {e}")
            raise

    return dataset_paths


def load_dataset_h5(
    h5_path: Path,
    chunk_size_mb: float = 64.0,
) -> Iterator[Tuple[np.ndarray, str, Dict[str, Any]]]:
    """
    Load windows, labels, and metadata from processed HDF5.

    Yields (window, label_name, metadata) tuples. IQ data is read in
    chunks to avoid loading the full dataset into memory, and metadata is
    read per chunk with the same slicing.

    chunk_size_mb controls the approximate memory used per IQ data chunk.
    The actual chunk size in windows is derived from the window size and
    dtype, so the memory footprint is consistent regardless of configuration.
    """
    with h5py.File(h5_path, "r") as f:
        if "data" not in f or "labels" not in f:
            raise ValueError(f"HDF5 file {h5_path} missing required datasets")

        data = f["data"]  # (N, 2, W) — not loaded yet
        labels = f["labels"][:]  # small, load upfront

        n_windows = data.shape[0]
        window_size = data.shape[2]
        data_dtype = data.dtype

        if "label_names" not in f:
            raise ValueError(
                f"HDF5 file {h5_path} has no /label_names. Every file written by "
                f"pipeline.hdf5_writer.build_hdf5_dataset has one, so this file is "
                f"either corrupt or was not produced by this pipeline. Refusing to "
                f"fall back to numeric label names, which would silently place "
                f"meaningless class names in the aggregate."
            )
        label_names = _decode_str_array(f["label_names"][:])

        # Metadata is read per-chunk (same slicing as IQ data) to avoid
        # loading millions of variable-length strings upfront, which is
        # extremely slow for large datasets like NIST (20M+ vlen strings).
        meta_group = f.get("meta", {})
        meta_keys = list(meta_group.keys()) if meta_group else []

        # Derive chunk size in windows from target memory budget
        bytes_per_window = 2 * window_size * data_dtype.itemsize  # IQ dominates
        chunk_windows = max(
            1, int((chunk_size_mb * 1024 * 1024) // bytes_per_window))

        for chunk_start in range(0, n_windows, chunk_windows):
            chunk_end = min(chunk_start + chunk_windows, n_windows)
            data_chunk = data[chunk_start:chunk_end]  # (chunk, 2, W)

            # Read metadata for this chunk only
            meta_chunks = {key: meta_group[key][chunk_start:chunk_end]
                for key in meta_keys}

            for i in range(len(data_chunk)):
                global_i = chunk_start + i
                window = data_chunk[i]
                label_id = int(labels[global_i])
                if not 0 <= label_id < len(label_names):
                    raise ValueError(
                        f"HDF5 file {h5_path} has label id {label_id} at window "
                        f"{global_i}, but /label_names holds only "
                        f"{len(label_names)} entries."
                    )
                label_name = label_names[label_id]

                meta = {}
                for key, arr in meta_chunks.items():
                    value = arr[i]
                    if isinstance(value, bytes):
                        meta[key] = value.decode("utf-8")
                    elif isinstance(value, np.ndarray) and value.dtype == object:
                        meta[key] = str(value.item())
                    else:
                        meta[key] = value.item() if hasattr(
                            value, "item") else value

                yield window, label_name, meta


def adapt_window_size(
    window: np.ndarray,
    source_size: int,
    target_size: int,
    padding_strategy: str,
    crop_policy: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Adapt window size to target size.

    Returns (adapted_window, metadata_dict) tuple.
    """
    metadata = {
        "was_padded": False,
        "was_cropped": False,
        "padding_strategy": "none",
        "crop_policy": "none",
        "original_window_size": source_size,
    }

    if source_size == target_size:
        return window, metadata

    if source_size < target_size:
        # Apply padding strategy
        metadata["was_padded"] = True
        metadata["padding_strategy"] = padding_strategy

        if padding_strategy == "zero":
            padded = np.zeros((2, target_size), dtype=window.dtype)
            padded[:, :source_size] = window
            return padded, metadata

        else:
            raise ValueError(f"Unknown padding strategy: {padding_strategy}")

    else:  # source_size > target_size
        # Apply crop policy
        metadata["was_cropped"] = True
        metadata["crop_policy"] = crop_policy

        if crop_policy == "start":
            cropped = window[:, :target_size]
            return cropped, metadata

        else:
            raise ValueError(f"Unknown crop policy: {crop_policy}")


def _load_dataset_config(registry_entry):
    """Load a dataset's config module. Returns None if not available."""
    if registry_entry.build_module:
        try:
            return importlib.import_module(f"{registry_entry.build_module}.config")
        except ImportError:
            return None
    return None


def extract_classification_metadata(
    label: str,
    metadata: Dict[str, Any],
    registry_entry,
    config_module=None,
) -> Tuple[str, str, str, str]:
    """
    Extract technology, modulation, radar-signal and waveform information from label/metadata.

    Returns (technology_type, modulation_type, waveform_type, radar_signal_type) tuple.

    The fine target is driven by registry_entry.label_represents, populated from the
    primary label symmetrically across all three task families:
    - "modulation"     → modulation_type; valid labels read from config MODULATION_TYPES
    - "technology"     → technology_type; valid labels read from config LABEL_REMAP values
    - "radar_waveform" → radar_signal_type; valid labels read from config RADAR_SIGNAL_TYPES;
                         technology_type and modulation_type stay "unknown"

    waveform_type (coarse) is always read directly from per-window metadata — every dataset
    populates it in its own build (modulation family / physical-layer waveform / radar
    intra-pulse family), so no routing logic is needed here. It is a classification target alongside the
    fine fields, not a grouping key.

    config_module should be pre-loaded via _load_dataset_config() once per dataset,
    not per window.
    """
    technology = "unknown"
    modulation = "unknown"
    radar_signal = "unknown"

    # Try to extract modulation from per-window metadata first (available for some datasets)
    if "modulation_type" in metadata and metadata["modulation_type"] not in ["", "unknown", None]:
        modulation = str(metadata["modulation_type"])

    if registry_entry.label_represents == "modulation":
        valid = list(getattr(config_module, "MODULATION_TYPES", [])) if config_module else []
        if not valid or label in valid:
            modulation = label

    elif registry_entry.label_represents == "technology":
        valid = list(getattr(config_module, "LABEL_REMAP", {}).values()) if config_module else []
        if not valid or label in valid:
            technology = label

    elif registry_entry.label_represents == "radar_waveform":
        # Fine radar target from the primary label — symmetric with modulation/technology.
        # RADAR_SIGNAL_TYPES doubles as the valid-label set (cf. MODULATION_TYPES). The
        # coarse waveform_type family is mapped in the radchar build and passed through below.
        valid = list(getattr(config_module, "RADAR_SIGNAL_TYPES", [])) if config_module else []
        if not valid or label in valid:
            radar_signal = label

    waveform = str(metadata.get("waveform_type", "unknown")) or "unknown"

    return technology, modulation, waveform, radar_signal


def get_primary_label(technology: str, modulation: str, waveform: str, label_type: str) -> str:
    """Get primary label based on label_type ("technology", "modulation", or "waveform")."""
    if label_type == "technology":
        return technology
    elif label_type == "modulation":
        return modulation
    elif label_type == "waveform":
        return waveform
    else:
        raise ValueError(f"Unknown label_type: {label_type}")


def unify_metadata(
    original_meta: Dict[str, Any],
    source_dataset: str,
    source_label: str,
    technology: str,
    modulation: str,
    waveform: str,
    window_meta: Dict[str, Any],
    normalize_mode: str,
    registry_entry=None,
    radar_signal: str = "unknown",
) -> Dict[str, Any]:
    """
    Unify metadata from source with unified schema.

    Applies fill values for missing fields.
    """

    unified = {
        "source_dataset":      source_dataset,
        "source_file":         original_meta.get("file_path", original_meta.get("source_file", "")),
        "recording_id":        original_meta.get("recording_id", ""),
        "source_label":        source_label,
        "task_type":           registry_entry.task_type if registry_entry is not None else "unknown",
        "technology_type":     technology,
        "modulation_type":     modulation,
        "radar_signal_type":   radar_signal,
        "waveform_type":       waveform,
        "original_window_size": -1,  # Set later by adapt_window_size
        **window_meta,
    }

    # Map common metadata fields (with consistent naming)
    field_mapping = {
        "center_frequency_hz": "center_frequency_hz",
        "sample_rate_hz": "sample_rate_hz",
        "snr_db": "snr_db",
        "gain_db": "gain_db",
        "attenuation_db": "attenuation_db",
        "location": "location",
        "mcs": "mcs",
        "channel": "channel",
    }

    for source_key, target_key in field_mapping.items():
        if source_key in original_meta and target_key not in unified:
            value = original_meta[source_key]
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                unified[target_key] = value

    return unified


def _iter_aggregated_windows() -> Iterator[Tuple[np.ndarray, str, Dict[str, Any]]]:
    """
    Main iterator that yields (window, primary_label, unified_metadata) tuples.

    This function:
    1. Loads dataset registry
    2. Builds individual datasets from raw data (if needed)
    3. Loops through selected datasets
    4. Gets dataset metadata from registry
    5. Loads windows from each processed HDF5
    6. Adapts window sizes to target
    7. Extracts/constructs classification metadata
    8. Gets primary label
    9. Unifies metadata schema
    10. Applies normalization
    11. Yields windows
    """
    # Build individual datasets if needed
    dataset_paths = build_individual_datasets(SELECTED_DATASETS)

    # Clear label registry so it is not polluted by constituent builds.
    reset_label_registry()

    # Iterate through selected datasets
    for dataset_name, config in SELECTED_DATASETS.items():
        if not config.get("enabled", True):
            continue

        if dataset_name not in dataset_paths:
            print(
                f"Warning: No path found for dataset '{dataset_name}', skipping...")
            continue

        h5_path = dataset_paths[dataset_name]
        registry_entry = get_dataset(dataset_name)
        config_module = _load_dataset_config(registry_entry)

        print(f"Processing dataset: {dataset_name}")

        norm_mode = NORMALIZE_MODE

        # Get source window size from actual HDF5 data (more robust than trusting config)
        # Read actual window size from the data shape: (N, 2, W) -> W is window size
        try:
            with h5py.File(h5_path, "r") as f:
                if "data" in f:
                    data_shape = f["data"].shape
                    if len(data_shape) == 3 and data_shape[1] == 2:
                        source_window_size = int(
                            data_shape[2])  # Get W dimension
                        print(
                            f"  Detected window size from HDF5: {source_window_size}")
                    else:
                        raise ValueError(
                            f"Unexpected data shape: {data_shape}, expected (N, 2, W)")
                else:
                    raise ValueError(
                        f"HDF5 file {h5_path} missing 'data' dataset")
        except Exception as e:
            # Fall back to the config if the HDF5 cannot be read
            print(
                f"  Warning: Could not read window size from HDF5 ({e}), falling back to config")
            preprocessing_config = registry_entry.get_default_preprocessing_config()
            source_window_size = preprocessing_config.get(
                "window_size", TARGET_WINDOW_SIZE)
            print(f"  Using window size from config: {source_window_size}")

        # Load and process windows
        window_count = 0
        for window, label_name, original_meta in load_dataset_h5(h5_path):
            # Adapt window size
            adapted_window, window_meta = adapt_window_size(
                window,
                source_window_size,
                TARGET_WINDOW_SIZE,
                PADDING_STRATEGY,
                CROP_POLICY,
            )

            # Extract classification metadata
            technology, modulation, waveform, radar_signal = extract_classification_metadata(
                label_name,
                original_meta,
                registry_entry,
                config_module,
            )

            # Get primary label
            primary_label = get_primary_label(
                technology, modulation, waveform, PRIMARY_LABEL_TYPE)

            # Unify metadata
            unified_meta = unify_metadata(
                original_meta,
                dataset_name,
                label_name,
                technology,
                modulation,
                waveform,
                window_meta,
                norm_mode,
                registry_entry,
                radar_signal=radar_signal,
            )

            # Source windows are already normalized in their own build. Re-applying
            # the mode here leaves unchanged windows as they are and brings cropped
            # windows back to unit power. Zero-padded windows are not renormalized:
            # scaling real signal up to compensate for artificial zeros would
            # distort its characteristics.
            if window_meta["was_padded"]:
                normalized_window = adapted_window
            else:
                normalized_window = normalize_window(adapted_window, norm_mode)

            # Ensure correct dtype
            normalized_window = normalized_window.astype(DATA_DTYPE)

            yield normalized_window, primary_label, unified_meta

            window_count += 1
            if window_count % 100_000 == 0:
                print(
                    f"  Processed {window_count} windows from {dataset_name}...")

        print(f"  Completed {dataset_name}: {window_count} windows")


def main():
    """Main entry point for building aggregated dataset."""
    print(f"=== Building {DATASET_NAME} ===")
    print(f"Script version: {SCRIPT_VERSION}")
    print(f"Target window size: {TARGET_WINDOW_SIZE}")
    print(f"Padding strategy: {PADDING_STRATEGY}")
    print(f"Crop policy: {CROP_POLICY}")
    print(f"Primary label type: {PRIMARY_LABEL_TYPE}")
    print(f"Normalization mode: {NORMALIZE_MODE}")
    print(f"Selected datasets: {SOURCE_DATASETS}")
    print()

    summary = build_hdf5_dataset(
        output_path=OUTPUT_H5_PATH,
        window_iterator=_iter_aggregated_windows(),
        batch_size=4096,
        data_dtype=DATA_DTYPE,
        global_metadata=GLOBAL_METADATA,
        window_metadata_schema=WINDOW_METADATA_SCHEMA,
    )

    print(
        f"\nCreated aggregated dataset with {summary['total_windows']} windows")
    print(f"Label names: {summary.get('label_names', [])}")
    print("Writing Croissant metadata...")

    # Build preprocessing metadata
    preprocessing = {
        "target_window_size": TARGET_WINDOW_SIZE,
        "padding_strategy": PADDING_STRATEGY,
        "crop_policy": CROP_POLICY,
        "normalization": NORMALIZE_MODE,
        "primary_label_type": PRIMARY_LABEL_TYPE,
    }

    generate_croissant_from_hdf5(
        h5_path=OUTPUT_H5_PATH,
        output_path=OUTPUT_CROISSANT_PATH,
        doi=RECORD_A_DOI,
        related_identifiers=f"per_dataset={RECORD_B_DOI}; code={CODE_DOI}",
        cite_as=COLLECTION_CITATION,
        version=RELEASE_VERSION,
        date_published=RELEASE_DATE,
        dataset_name=DATASET_NAME,
        # Same constant as the HDF5 root `description` attribute — see config.py.
        description=DATASET_DESCRIPTION,
        task_type="classification",
        labels=summary.get("label_names", []),
        citations=CITATIONS,
        license=LICENSE,
        preprocessing=preprocessing,
        provenance={
            "script_version": SCRIPT_VERSION,
            # Same list as the root `source_datasets` attribute — both derive from
            # SOURCE_DATASETS so the Croissant cannot claim a source the file lacks.
            "source_datasets": SOURCE_DATASETS,
            "aggregation_method": "multi-dataset combination with unified preprocessing",
        },
    )

    print(f"Croissant metadata written to {OUTPUT_CROISSANT_PATH}")
    print(f"Dataset written to {OUTPUT_H5_PATH}")
    print("=== Done ===")


if __name__ == "__main__":
    main()
