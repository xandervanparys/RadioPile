"""
Croissant metadata utilities.

This module writes an MLCommons Croissant 1.0 descriptor for an ML-ready RF
dataset stored in HDF5.

Design principles:
- Croissant is descriptive, not operational
- No dataset is loaded into memory
- Semantics are explicit, not inferred

The descriptor passes `mlcroissant validate`. Croissant has no vocabulary for
the windowing, normalization and provenance of an RF dataset, so those are
written under the `radiopile:` prefix declared in the context.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import date
from typing import Dict, List, Optional

import h5py
import numpy as np


CROISSANT_VERSION = "http://mlcommons.org/croissant/1.0"

# The JSON-LD context of Croissant 1.0, as mlcroissant emits it, plus the
# prefix under which the RF-specific blocks are written.
CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
    "radiopile": "https://github.com/xandervanparys/RadioPile#",
}

# Identifier of the HDF5 file inside the descriptor. Fields refer to it, so it
# stays fixed when the bundler renames the file.
FILE_OBJECT_ID = "hdf5-file"
RECORD_SET_ID = "windows"

_HASH_CHUNK = 16 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _croissant_data_type(dtype) -> str:
    """Map an HDF5 dataset dtype to a Croissant data type."""
    if h5py.check_string_dtype(dtype) is not None or dtype.kind in "SUO":
        return "sc:Text"
    if dtype.kind == "b":
        return "sc:Boolean"
    if dtype.kind in "iu":
        return "sc:Integer"
    if dtype.kind == "f":
        return "sc:Float"
    return "sc:Text"


def _field(name: str, column: str, dtype, description: str, **extra) -> Dict:
    return {
        "@type": "cr:Field",
        "@id": f"{RECORD_SET_ID}/{name}",
        "name": name,
        "description": description,
        "dataType": _croissant_data_type(dtype),
        "source": {
            "fileObject": {"@id": FILE_OBJECT_ID},
            "extract": {"column": column},
        },
        **extra,
    }


def generate_croissant_from_hdf5(
    *,
    h5_path: Path,
    output_path: Path,
    dataset_name: str,
    description: str,
    task_type: str,
    labels: List[str],
    citations: List[str],
    doi: Optional[str] = None,
    related_identifiers: Optional[str] = None,
    cite_as: Optional[str] = None,
    version: Optional[str] = None,
    date_published: Optional[str] = None,
    license: str,
    preprocessing: Dict,
    provenance: Optional[Dict] = None,
) -> Path:
    """
    Generate a Croissant 1.0 metadata JSON file for an ML-ready dataset.

    Parameters
    ----------
    h5_path : Path
        Path to the processed HDF5 dataset.
    output_path : Path
        Where to write the croissant.json file.
    dataset_name : str
        Human-readable dataset name.
    description : str
        Short description of the dataset.
    task_type : str
        ML task type (e.g. "classification").
    labels : list[str]
        Ordered list mapping label IDs to label names.
    doi : str, optional
        Persistent identifier for this artifact. Emitted as `url`, so a
        consumer holding only the descriptor can still locate the dataset.
    related_identifiers : str, optional
        Free-form pointer to sibling records (e.g. the per-dataset record and
        the code DOI). Emitted as `radiopile:relatedIdentifiers`; the same
        value is the `related_identifiers` HDF5 root attribute.
    cite_as : str, optional
        Full citation for this artifact, emitted as `citeAs` when a DOI is
        given. Defaults to the dataset name followed by the DOI.
    version : str, optional
        Version of the dataset, emitted as `version`.
    date_published : str, optional
        Publication date (YYYY-MM-DD), emitted as `datePublished`. `dateCreated`
        is always the day the descriptor is written.
    citations : list[str]
        Bibliographic references for the dataset.
    license : str
        Dataset license.
    preprocessing : dict
        High-level preprocessing description (windowing, normalization, etc.).
    provenance : dict, optional
        Information about source datasets and capture context.

    Returns
    -------
    Path
        Path to the written Croissant JSON file.
    """

    h5_path = Path(h5_path)
    output_path = Path(output_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        if "data" not in f or "labels" not in f:
            raise ValueError("HDF5 must contain at least /data and /labels datasets")

        data_dtype = f["data"].dtype
        window_shape = list(f["data"].shape[1:])

        fields = [
            _field(
                "data", "data", data_dtype,
                "Windowed baseband IQ samples, shape [channels=2, window_length]; "
                "channel 0 is I, channel 1 is Q.",
                **{
                    "radiopile:dtype": str(data_dtype),
                    "radiopile:arrayShape": window_shape,
                },
            ),
            _field(
                "labels", "labels", f["labels"].dtype,
                "Integer class label ID; radiopile:labels maps each ID to its name.",
                **{"radiopile:dtype": str(f["labels"].dtype)},
            ),
        ]
        if "meta" in f:
            for key in sorted(f["meta"].keys()):
                column = f["meta"][key]
                dtype = column.dtype
                is_string = _croissant_data_type(dtype) == "sc:Text"
                fields.append(_field(
                    key, f"meta/{key}", dtype,
                    f"Per-window metadata field {key}.",
                    **{"radiopile:dtype": "string" if is_string else str(np.dtype(dtype))},
                ))

    has_doi = bool(doi) and not doi.startswith("[")

    croissant = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": CROISSANT_VERSION,
        "name": dataset_name,
        "description": description,
        "dateCreated": str(date.today()),
        **({"datePublished": date_published} if date_published else {}),
        **({"version": version} if version else {}),
        "license": license,
        "citation": citations,
        **({"url": f"https://doi.org/{doi}"} if has_doi else {}),
        **({"citeAs": cite_as or f"{dataset_name}. DOI: {doi}"} if has_doi else {}),

        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": FILE_OBJECT_ID,
                "name": h5_path.name,
                "contentUrl": str(h5_path),
                "encodingFormat": "application/x-hdf5",
                "sha256": _sha256(h5_path),
            }
        ],

        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": RECORD_SET_ID,
                "name": RECORD_SET_ID,
                "description": "Windowed RF IQ samples with per-window labels and metadata.",
                "field": fields,
            }
        ],

        "radiopile:task": {
            "type": task_type,
            "input": "data",
            "target": "labels",
            "labelCount": len(labels),
        },
        "radiopile:labels": {str(i): name for i, name in enumerate(labels)},
        "radiopile:preprocessing": preprocessing,
    }

    if provenance:
        croissant["radiopile:provenance"] = provenance
    if related_identifiers and "[" not in related_identifiers:
        croissant["radiopile:relatedIdentifiers"] = related_identifiers

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(croissant, f, indent=2)

    return output_path
