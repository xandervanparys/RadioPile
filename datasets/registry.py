"""
Dataset Registry System

Centralized registry that exposes dataset metadata for aggregation and other tools.
Each dataset entry provides name, paths, and build module. Preprocessing config and metadata
fields are fetched dynamically from the dataset's config.py file.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import importlib

from pipeline.schemas import TaskType, LabelRepresents, WindowMetaField


@dataclass
class DatasetEntry:
    """Entry for a single dataset in the registry."""

    name: str
    processed_h5_path: Path
    build_module: Optional[str] = None  # e.g., "datasets.rml2016_10a"
    raw_data_root: Optional[Path] = None
    task_type: TaskType = TaskType.UNKNOWN
    label_represents: LabelRepresents = LabelRepresents.TECHNOLOGY

    def __post_init__(self):
        # Coerce plain strings to enum members; raises ValueError on invalid values.
        # Accepts both DatasetEntry(..., task_type="radar_classification") and
        # DatasetEntry(..., task_type=TaskType.RADAR_CLASSIFICATION), but
        # DatasetEntry(..., task_type="typo") fails immediately at construction.
        if not isinstance(self.task_type, TaskType):
            self.task_type = TaskType(self.task_type)
        if not isinstance(self.label_represents, LabelRepresents):
            self.label_represents = LabelRepresents(self.label_represents)

    def get_default_preprocessing_config(self) -> Dict:
        """Fetch preprocessing config from the dataset's config module."""
        if not self.build_module:
            return {}
        try:
            module = importlib.import_module(f"{self.build_module}.config")
            return {
                "window_size": getattr(module, "WINDOW_SIZE", None),
                "normalization_mode": getattr(module, "NORMALIZE_MODE", None),
                "noise_filtering": getattr(module, "NOISE_POWER_THRESHOLD_DB", None),
                "data_dtype": getattr(module, "DATA_DTYPE", None),
            }
        except (ImportError, AttributeError):
            return {}

    def get_available_metadata_fields(self) -> List[str]:
        """Extract metadata field names from the dataset's WINDOW_METADATA_SCHEMA."""
        if not self.build_module:
            return []
        try:
            module = importlib.import_module(f"{self.build_module}.config")
            schema = getattr(module, "WINDOW_METADATA_SCHEMA", [])
            return [field.key for field in schema]
        except (ImportError, AttributeError):
            return []

    def get_available_metadata_schema(self) -> List[WindowMetaField]:
        """Return the dataset's WINDOW_METADATA_SCHEMA entries (full WindowMetaField objects).

        Used by tooling that needs dtype/fill_value in addition to the field key
        (e.g. the validator's generic -1-sentinel sweep). Returns [] if the
        dataset has no build_module or the schema cannot be imported.
        """
        if not self.build_module:
            return []
        try:
            module = importlib.import_module(f"{self.build_module}.config")
            return list(getattr(module, "WINDOW_METADATA_SCHEMA", []))
        except (ImportError, AttributeError):
            return []


# Registry of all available datasets
# Only registry-specific information is stored here (paths, build modules).
# Preprocessing config and metadata fields are fetched dynamically from each
# dataset's config.py file.
DATASET_REGISTRY: Dict[str, DatasetEntry] = {
    "rml2016_10a": DatasetEntry(
        name="rml2016_10a",
        processed_h5_path=Path("data/rml2016_10a/processed/rml2016_10a.h5"),
        build_module="datasets.rml2016_10a",
        raw_data_root=Path("data/rml2016_10a/raw"),
        task_type=TaskType.MODULATION_RECOGNITION,
        label_represents=LabelRepresents.MODULATION,
    ),
    "sub_ghz": DatasetEntry(
        name="sub_ghz",
        processed_h5_path=Path("data/sub_ghz/processed/sub_ghz.h5"),
        build_module="datasets.sub_ghz",
        raw_data_root=Path("data/sub_ghz/raw"),
        task_type=TaskType.TECHNOLOGY_RECOGNITION,
        label_represents=LabelRepresents.TECHNOLOGY,
    ),
    "rml2018_01a": DatasetEntry(
        name="rml2018_01a",
        processed_h5_path=Path("data/rml2018_01a/processed/rml2018_01a.h5"),
        build_module="datasets.rml2018_01a",
        raw_data_root=Path("data/rml2018_01a/raw"),
        task_type=TaskType.MODULATION_RECOGNITION,
        label_represents=LabelRepresents.MODULATION,
    ),
    "lte_wifi_dvbt": DatasetEntry(
        name="lte_wifi_dvbt",
        processed_h5_path=Path("data/lte_wifi_dvbt/processed/lte_wifi_dvbt.h5"),
        build_module="datasets.lte_wifi_dvbt",
        raw_data_root=Path("data/lte_wifi_dvbt/raw"),
        task_type=TaskType.TECHNOLOGY_RECOGNITION,
        label_represents=LabelRepresents.TECHNOLOGY,
    ),
    "radchar": DatasetEntry(
        name="radchar",
        processed_h5_path=Path("data/radchar/processed/radchar.h5"),
        build_module="datasets.radchar",
        raw_data_root=Path("data/radchar/raw"),
        task_type=TaskType.RADAR_CLASSIFICATION,
        label_represents=LabelRepresents.RADAR_WAVEFORM,
    ),
    "its_band": DatasetEntry(
        name="its_band",
        processed_h5_path=Path("data/its_band/processed/its_band.h5"),
        build_module="datasets.its_band",
        raw_data_root=Path("data/its_band/raw"),
        task_type=TaskType.TECHNOLOGY_RECOGNITION,
        label_represents=LabelRepresents.TECHNOLOGY,
    ),
    "nist_wifi_bluetooth": DatasetEntry(
        name="nist_wifi_bluetooth",
        processed_h5_path=Path("data/nist_wifi_bluetooth/processed/nist_wifi_bluetooth.h5"),
        build_module="datasets.nist_wifi_bluetooth",
        raw_data_root=Path("data/nist_wifi_bluetooth/raw"),
        task_type=TaskType.TECHNOLOGY_RECOGNITION,
        label_represents=LabelRepresents.TECHNOLOGY,
    ),
}


def get_dataset(name: str) -> DatasetEntry:
    """Get dataset entry by name."""
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Dataset '{name}' not found in registry. Available datasets: {list_datasets()}")
    return DATASET_REGISTRY[name]


def list_datasets() -> List[str]:
    """List all registered dataset names."""
    return sorted(DATASET_REGISTRY.keys())
