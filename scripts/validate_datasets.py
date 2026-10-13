"""
Validate processed HDF5 datasets for structural integrity and metadata completeness.

Checks per dataset:
  - File exists and can be opened
  - total_windows attribute matches actual data/labels array lengths
  - All meta/* fields have length == n_windows
  - Data dtype matches config
  - IQ data is finite (no NaN/inf) in a spot-check sample
  - unit_power normalization sanity: mean(I²+Q²) ≈ 1 on a sample (individual datasets only)
  - Label indices are contiguous (0..C-1)
  - Expected metadata fields (from the dataset's WINDOW_METADATA_SCHEMA) are present
  - No empty-string values in string metadata fields (sampled for large datasets)
  - recording_id dtype is string (not int) when present
  - Dataset-specific checks (e.g. sub_ghz modulation routing from LABEL_TO_MODULATION,
    LoRa spreading_factor)

Aggregated-only checks (all config-driven — no hardcoded source names):
  - Data shape is (N, 2, TARGET_WINDOW_SIZE)
  - original_window_size populated and consistent with was_padded / was_cropped
  - snr_db field: present, float dtype, non-NaN for sources whose per-dataset schema
    includes snr_db (derived from each source's WINDOW_METADATA_SCHEMA)
  - Label routing semantics per source (derived from label_represents in registry):
      MODULATION     → technology_type == "unknown", modulation_type != "unknown"
      TECHNOLOGY     → technology_type != "unknown"
      RADAR_WAVEFORM → technology_type and modulation_type both "unknown";
                       radar_signal_type (fine) populated from the primary label;
                       waveform_type populated with the coarse radar family
                       (frequency_modulated / phase_modulated / unmodulated)

The report can also identify exactly which bytes were checked (--fingerprint):
each file's path, size, SHA-256 and MD5 are recorded before and after the checks
run, together with the commit and any uncommitted changes of the code that ran
them. A report without that says only that *some* file at that path passed.

Usage:
    python scripts/validate_datasets.py
    python scripts/validate_datasets.py --dataset sub_ghz
    python scripts/validate_datasets.py --dataset sub_ghz aggregation
    python scripts/validate_datasets.py --fingerprint --save-report report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import importlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import numpy as np
import h5py

# ── Project root on sys.path ──────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.registry import DATASET_REGISTRY, DatasetEntry
from pipeline.schemas import LabelRepresents, RADAR_FINE_TO_COARSE, RADAR_WAVEFORM_FAMILIES


# ── Constants ─────────────────────────────────────────────────────────────────

# Version of the report format, recorded in every saved report.
VALIDATOR_VERSION = "1.1"

REPO_ROOT = Path(__file__).resolve().parent.parent

# Bytes per read when hashing. Large enough that hashlib releases the GIL for
# almost all of the work, so files hash in parallel across threads.
HASH_CHUNK = 16 << 20

# Source files the checks take their expectations from, beyond this script and
# the dataset configs: label vocabularies and registry entries are imported,
# not restated here.
EXPECTATION_SOURCES = ("datasets/registry.py", "pipeline/schemas.py")

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

# Maximum rows scanned per string field for the empty-string check.
# Empty-string bugs are typically systematic (all windows or none), so a
# 500k-row prefix is sufficient to catch them reliably without deserializing
# millions of vlen strings on large datasets like NIST (~20M rows).
EMPTY_CHECK_SAMPLE = 500_000

# Rows per read when decoding a vlen string column (see _decode_str_field).
# Bounds the transient cost of the raw bytes objects; the decoded column is
# built incrementally rather than all at once.
DECODE_CHUNK = 1_000_000

# String metadata fields where "" is a valid fill value.
# These are whitelisted so they produce a warning rather than a failure.
EMPTY_OK_FIELDS = {
    "recording_id",        # simulated datasets have no physical recording
    "padding_strategy",    # "none" sentinel stored as ""
    "wifi_standard",       # only populated for WiFi windows
    "ieee802154_variant",  # only populated for 802.15.4 windows
}

# Integer fields where -1 is a legitimate "not applicable" fill rather than a
# data-integrity bug. The generic sentinel sweep skips these; domain-specific
# checks (e.g. spreading_factor for LoRa-only windows) still cover them.
SENTINEL_SWEEP_SKIP_FIELDS = {
    "spreading_factor",    # LoRa-only; -1 expected for non-LoRa windows
    "mcs",                 # only populated when filename carries an MCS token
    "channel",             # only populated when filename carries a channel token
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _decode_str_field(ds: h5py.Dataset) -> np.ndarray:
    """Read an HDF5 string dataset and return a numpy array of Python str.

    Handles both fixed-length bytes (dtype.kind='S') and variable-length
    (dtype.kind='O', as produced by h5py for vlen UTF-8 strings).

    The vlen path reads in chunks and reuses one str object per distinct
    value. Reading a whole column at once materializes every row's raw bytes
    object and a separate decoded str, about 3 GB per column on the aggregated
    file. These columns are low-cardinality (dataset names, class labels,
    policy strings), so sharing references collapses the decoded column to one
    pointer per row.

    Returns an object array of Python str with the same values in the same order.
    """
    if ds.dtype.kind == "S":
        return np.char.decode(ds[:], "utf-8")

    n = ds.shape[0]
    out = np.empty(n, dtype=object)
    cache: dict = {}
    for start in range(0, n, DECODE_CHUNK):
        stop = min(start + DECODE_CHUNK, n)
        for i, raw in enumerate(ds[start:stop], start):
            value = cache.get(raw)
            if value is None:
                value = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                cache[raw] = value
            out[i] = value
    return out


def _count_empty(ds: h5py.Dataset) -> tuple[int, int]:
    """Count empty strings in at most EMPTY_CHECK_SAMPLE rows.

    Returns (n_empty, n_checked). When n_checked < ds.shape[0] the result
    is from a prefix sample, not the full dataset.
    """
    n_check = min(ds.shape[0], EMPTY_CHECK_SAMPLE)
    arr = ds[:n_check]
    if arr.dtype.kind == "S":
        return int(np.sum(arr == b"")), n_check
    return int(np.sum((arr == b"") | (arr == ""))), n_check


def _is_int_dtype(dtype_obj: object) -> bool:
    """Return True if the WindowMetaField dtype is a numpy integer dtype."""
    try:
        return np.dtype(dtype_obj).kind in ("i", "u")
    except TypeError:
        return False


def _sentinel_sweep(
    f: h5py.File,
    schema_fields,
    results: list,
    *,
    source_label: str,
    mask: "np.ndarray | None" = None,
) -> None:
    """Generic -1-sentinel sweep over every integer field in a schema.

    For each WindowMetaField with an integer dtype and fill_value == -1
    that is not on the SENTINEL_SWEEP_SKIP_FIELDS opt-out list, count
    occurrences of the -1 sentinel in /meta/<h5_name> and append a
    structured pass/fail record.

    source_label : appears in the check name and detail text. Use the
        dataset name for per-dataset files; use "<aggregation>:<source>"
        when sweeping a source's slice of the aggregated file.
    mask : optional boolean array selecting the rows to inspect. Used by
        the aggregated sweep to scope each source's check to its own
        windows. None means "all rows".
    """
    for field in schema_fields:
        if field.key in SENTINEL_SWEEP_SKIP_FIELDS:
            continue
        if not _is_int_dtype(field.dtype):
            continue
        if field.fill_value != -1:
            continue
        if "meta" not in f or field.h5_name not in f["meta"]:
            # A mask means this is a source's slice of the aggregated file;
            # the aggregator only writes its own schema, so source-specific
            # fields are intentionally absent and are skipped silently. Without a mask (per-dataset path), missing schema
            # fields are a real bug and produce a failure.
            if mask is None:
                results.append(_check(
                    f"sentinel_check_{source_label}_{field.key}",
                    False,
                    f"/meta/{field.h5_name} missing",
                ))
            continue

        arr = f[f"meta/{field.h5_name}"][:]
        if mask is not None:
            arr = arr[mask]
        n_total = int(arr.size)
        if n_total == 0:
            results.append(_warn(
                f"sentinel_check_{source_label}_{field.key}: no windows in scope"
            ))
            continue
        n_bad = int(np.sum(arr == -1))
        results.append(_check(
            f"sentinel_check_{source_label}_{field.key}",
            n_bad == 0,
            f"{n_bad:,}/{n_total:,} windows carry -1 sentinel" if n_bad
            else f"{n_total:,} windows OK",
        ))


def _check(label: str, ok: bool, detail: str = "") -> dict:
    """Return a structured record for a named check.

    Keys: name, passed, detail, level ("check"), line (preformatted for stdout).
    """
    status = PASS if ok else FAIL
    line = f"  {status} {label}"
    if detail:
        line += f": {detail}"
    return {"name": label, "passed": bool(ok), "detail": detail, "level": "check", "line": line}


def _info(msg: str) -> dict:
    """Return a structured record for an always-passing informational message."""
    return {"name": msg, "passed": True, "detail": "", "level": "info", "line": f"  {PASS} {msg}"}


def _warn(msg: str) -> dict:
    """Return a structured record for an always-passing warning message."""
    return {"name": msg, "passed": True, "detail": "", "level": "warn", "line": f"  {WARN} {msg}"}


# ── File fingerprints and code provenance ─────────────────────────────────────

def fingerprint_file(path: Path) -> dict | None:
    """Size, mtime, SHA-256 and MD5 of one file, or None if it does not exist.

    Both digests come from a single pass. MD5 is recorded alongside SHA-256
    because it is what Zenodo reports for an uploaded file.
    """
    path = Path(path)
    if not path.exists():
        return None
    stat = path.stat()
    sha256, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(HASH_CHUNK), b""):
            sha256.update(block)
            md5.update(block)
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256.hexdigest(),
        "md5": md5.hexdigest(),
    }


def fingerprint_files(paths: dict[str, Path]) -> dict[str, dict | None]:
    """Fingerprint several files concurrently, keyed as given."""
    if not paths:
        return {}
    with ThreadPoolExecutor(max_workers=min(8, len(paths))) as pool:
        futures = {name: pool.submit(fingerprint_file, p) for name, p in paths.items()}
        return {name: fut.result() for name, fut in futures.items()}


def _git(*args: str) -> str | None:
    """Run git in the repository holding this script; None if unavailable."""
    try:
        out = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


def _sha256_of(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def code_provenance(dataset_names) -> dict:
    """Identify the code that produced a report.

    The commit alone is not enough: the checks read their expected values from
    the dataset configs, so a report is only as trustworthy as the working tree
    it ran in. Uncommitted changes to tracked files are listed, and the diff is
    hashed so that two reports from one dirty tree can be told apart from two
    reports from different ones.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    diff = _git("diff", "HEAD")
    changed = [line[3:] for line in (status or "").splitlines() if line.strip()]

    sources = [Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()]
    sources += list(EXPECTATION_SOURCES)
    sources += [f"datasets/{name}/config.py" for name in dataset_names]

    return {
        "validator_version": VALIDATOR_VERSION,
        "git_commit": commit.strip() if commit else None,
        "git_dirty": bool(changed) if status is not None else None,
        "uncommitted_changes": changed if status is not None else None,
        "uncommitted_diff_sha256": (
            hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else None
        ),
        "source_sha256": {
            rel: _sha256_of(REPO_ROOT / rel) for rel in sorted(set(sources))
        },
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "hdf5": h5py.version.hdf5_version,
    }


def _file_record(path: Path, before: dict | None, after: dict | None) -> dict:
    """Per-file block of the report: what was checked, and whether it moved."""
    record = {
        "path": str(path),
        "before": before,
        "after": after,
    }
    if before is None or after is None:
        record["unchanged_during_validation"] = None
    else:
        record["unchanged_during_validation"] = all(
            before[k] == after[k] for k in ("size_bytes", "mtime_ns", "sha256", "md5")
        )
    if after is not None:
        for key in ("size_bytes", "sha256", "md5"):
            record[key] = after[key]
    return record


# ── Per-dataset validation ─────────────────────────────────────────────────────

def validate_dataset(name: str, entry: DatasetEntry) -> tuple[bool, list[dict]]:
    """Validate one processed HDF5 dataset. Prints results to stdout.

    Returns (all_passed, results) where results is a list of structured
    record dicts (see ``_check``/``_info``/``_warn``).
    """
    h5_path = entry.processed_h5_path
    print(f"\nValidating {name} ({h5_path})")

    # ── 1. File existence ─────────────────────────────────────────────────────
    file_record = _check("file exists", h5_path.exists())
    print(file_record["line"])
    if not file_record["passed"]:
        return False, [file_record]

    all_passed = True
    results: list[dict] = [file_record]
    preprocessing_config = entry.get_default_preprocessing_config()

    with h5py.File(h5_path, "r") as f:

        # ── 2. Shape consistency ──────────────────────────────────────────────
        n_data   = f["data"].shape[0]   if "data"   in f else None
        n_labels = f["labels"].shape[0] if "labels" in f else None
        n_attr   = f.attrs.get("total_windows")

        if n_data is None:
            results.append(_check("data dataset present", False))
        if n_labels is None:
            results.append(_check("labels dataset present", False))

        if n_data is not None and n_labels is not None:
            consistent = (n_data == n_labels)
            results.append(_check(
                "window count consistent",
                consistent,
                f"data={n_data}, labels={n_labels}" if not consistent
                else f"{n_data:,} windows",
            ))

            if n_attr is not None:
                results.append(_check(
                    "total_windows attr matches",
                    int(n_attr) == n_data,
                    f"attr={int(n_attr)}, actual={n_data}" if int(n_attr) != n_data else "",
                ))

        # ── 2b. Meta field length consistency ─────────────────────────────────
        if "meta" in f and n_data is not None:
            short = [
                f"{key}: {f[f'meta/{key}'].shape[0]} ≠ {n_data}"
                for key in f["meta"].keys()
                if f[f"meta/{key}"].shape[0] != n_data
            ]
            results.append(_check(
                "all meta fields length == n_windows",
                len(short) == 0,
                "; ".join(short) if short else f"{len(list(f['meta'].keys()))} fields",
            ))

        # ── 3. Data dtype ─────────────────────────────────────────────────────
        if "data" in f:
            dtype = f["data"].dtype
            expected_dtype_str = preprocessing_config.get("data_dtype") or "float32"
            expected_dtype = np.dtype(expected_dtype_str)
            results.append(_check(
                f"data dtype {expected_dtype_str}",
                dtype == expected_dtype,
                str(dtype) if dtype != expected_dtype else "",
            ))
            results.append(_info(f"data shape: {f['data'].shape}"))

        # ── 3b. NaN/inf check on IQ data sample ───────────────────────────────
        if "data" in f and n_data is not None and n_data > 0:
            sample_size = min(1000, n_data)
            sample = f["data"][:sample_size]
            n_bad = int(np.sum(~np.isfinite(sample)))
            results.append(_check(
                f"IQ data finite (sample of {sample_size:,})",
                n_bad == 0,
                f"{n_bad} non-finite values" if n_bad else "",
            ))

        # ── 3c. Normalization sanity (individual datasets only) ───────────────
        # Skipped for aggregated: zero-padded windows (e.g. RML2016 128→500)
        # have reduced mean power by design; renormalization is intentionally
        # skipped after zero-padding to avoid amplifying artificial zeros.
        if "data" in f and n_data is not None and n_data > 0 and name != "aggregation":
            norm_mode = preprocessing_config.get("normalization_mode") or ""
            if norm_mode == "unit_power":
                mid = n_data // 2
                sample = f["data"][mid:min(mid + 500, n_data)]  # (N, 2, W)
                power = np.mean(
                    sample[:, 0, :].astype(float) ** 2
                    + sample[:, 1, :].astype(float) ** 2,
                    axis=1,
                )
                mean_p = float(np.mean(power))
                n_out = int(np.sum(np.abs(power - 1.0) > 0.15))
                results.append(_check(
                    "unit_power normalization sanity (mean power ≈ 1)",
                    n_out == 0,
                    f"mean={mean_p:.3f}, {n_out} windows outside [0.85, 1.15]"
                    if n_out else f"mean={mean_p:.3f}",
                ))

        # ── 4. Label names ────────────────────────────────────────────────────
        if "label_names" in f:
            label_names = _decode_str_field(f["label_names"]).tolist()
            results.append(_check("label_names present", True, ", ".join(label_names)))
        else:
            label_names = []
            results.append(_check("label_names present", False))

        # ── 5. Label indices contiguous ───────────────────────────────────────
        if "labels" in f and label_names:
            y = f["labels"][:]
            unique = sorted(np.unique(y).tolist())
            expected = list(range(len(label_names)))
            results.append(_check(
                "label indices contiguous (0..C-1)",
                unique == expected,
                f"got {unique}, expected {expected}" if unique != expected else "",
            ))

        # ── 5b. Controlled-vocabulary fields hold values, not enum reprs ──────
        # A str-mixin Enum stringifies to its repr, so a window can end up with
        # task_type = "TaskType.TECHNOLOGY_RECOGNITION" instead of
        # "technology_recognition", and the documented filter then matches
        # nothing. Check the vocabulary explicitly.
        if "meta" in f and "task_type" in f["meta"]:
            from pipeline.schemas import TaskType
            allowed = {m.value for m in TaskType}
            n = f["meta/task_type"].shape[0]
            step = max(1, n // 100_000)
            seen = set(f["meta/task_type"].asstr()[::step])
            bad = sorted(seen - allowed)
            results.append(_check(
                "task_type values in controlled vocabulary",
                not bad,
                f"unexpected: {bad[:4]}" if bad else f"{sorted(seen)}",
            ))

        # ── 6. Metadata fields present ────────────────────────────────────────
        schema_fields = entry.get_available_metadata_fields()
        if schema_fields:
            present = set(f["meta"].keys()) if "meta" in f else set()
            missing = [k for k in schema_fields if k not in present]
            results.append(_check(
                "schema metadata fields present",
                len(missing) == 0,
                f"missing: {missing}" if missing else f"{len(schema_fields)} fields",
            ))
        else:
            results.append(_warn("no WINDOW_METADATA_SCHEMA found in config (skipping field check)"))

        # ── 7. No empty strings in string metadata ────────────────────────────
        if "meta" in f:
            bad_fields = []
            warn_fields = []
            sampled = False
            for key in f["meta"].keys():
                ds = f[f"meta/{key}"]
                if ds.dtype.kind in ("S", "O"):
                    n_empty, n_checked = _count_empty(ds)
                    if n_checked < ds.shape[0]:
                        sampled = True
                    if n_empty:
                        if key in EMPTY_OK_FIELDS:
                            warn_fields.append(f"{key}: {n_empty} empty (fill value, expected)")
                        else:
                            bad_fields.append(f"{key}: {n_empty} empty strings")
            label = "no empty strings in string metadata"
            if sampled:
                label += f" (sampled {EMPTY_CHECK_SAMPLE:,} rows)"
            results.append(_check(label, len(bad_fields) == 0, "; ".join(bad_fields) if bad_fields else ""))
            for msg in warn_fields:
                results.append(_warn(msg))

        # ── 8. recording_id dtype is string ──────────────────────────────────
        if "meta" in f and "recording_id" in f["meta"]:
            rid_dtype = f["meta/recording_id"].dtype
            is_str = rid_dtype.kind in ("S", "O")
            results.append(_check(
                "recording_id dtype is string",
                is_str,
                f"got {rid_dtype} (expected bytes or object)" if not is_str else str(rid_dtype),
            ))

        # ── 9. Dataset-specific checks ────────────────────────────────────────
        if name == "sub_ghz" and "meta" in f:
            _validate_sub_ghz(f, results)

        if name == "its_band" and "meta" in f:
            _validate_its_band(f, results)

        if name == "nist_wifi_bluetooth" and "meta" in f:
            _validate_nist_wifi_bluetooth(f, results)

        if name == "radchar" and "meta" in f:
            _validate_radchar(f, results)

        if name == "aggregation" and "meta" in f:
            _validate_aggregated(f, results)

        # ── 10. Generic schema-driven -1 sentinel sweep ───────────────────────
        # Per-dataset path only. The aggregated path runs its own per-source
        # sweep inside _validate_aggregated so each source is checked against
        # its own schema with the correct source_dataset mask.
        if name != "aggregation":
            schema_fields = entry.get_available_metadata_schema()
            if schema_fields:
                _sentinel_sweep(f, schema_fields, results, source_label=name)

    # Skip the file-existence record (already printed above) when re-printing.
    for r in results[1:]:
        print(r["line"])
        if not r["passed"]:
            all_passed = False

    return all_passed, results


def _validate_radchar(f: h5py.File, results: list[tuple[bool, str]]):
    """RadChar-specific checks.

    Radar is symmetric with the other two task families: a fine target column
    (radar_signal_type, one of the five canonical signal types) sits below a
    coarse waveform_type family (the radar intra-pulse taxonomy:
    frequency_modulated / phase_modulated / unmodulated). Asserts both columns are
    populated with canonical values for every window, and that each fine type
    maps to the expected coarse family.
    """
    valid_signal_types = set(RADAR_FINE_TO_COARSE)

    # ── Fine target: radar_signal_type ───────────────────────────────────────
    if "radar_signal_type" not in f.get("meta", {}):
        results.append(_check("radar_signal_type present", False, "field missing from /meta"))
        return

    rst_vals = _decode_str_field(f["meta/radar_signal_type"])
    n = len(rst_vals)

    n_unknown = int(np.sum(rst_vals == "unknown"))
    results.append(_check(
        "radchar: radar_signal_type populated (no 'unknown')",
        n_unknown == 0,
        f"{n_unknown:,} windows with 'unknown' radar_signal_type" if n_unknown
        else f"{n:,} windows",
    ))
    n_invalid = int(np.sum(~np.isin(rst_vals, list(valid_signal_types))))
    results.append(_check(
        "radchar: radar_signal_type values are canonical radar signal types",
        n_invalid == 0,
        f"{n_invalid:,} windows with unexpected value" if n_invalid else "",
    ))

    # ── Coarse family: waveform_type ─────────────────────────────────────────
    if "waveform_type" not in f.get("meta", {}):
        results.append(_check("waveform_type present", False, "field missing from /meta"))
        return

    wf_vals = _decode_str_field(f["meta/waveform_type"])

    n_unknown = int(np.sum(wf_vals == "unknown"))
    results.append(_check(
        "radchar: waveform_type populated (no 'unknown')",
        n_unknown == 0,
        f"{n_unknown:,} windows with 'unknown' waveform_type" if n_unknown
        else f"{n:,} windows",
    ))
    n_invalid = int(np.sum(~np.isin(wf_vals, RADAR_WAVEFORM_FAMILIES)))
    results.append(_check(
        "radchar: waveform_type values are coarse radar families "
        f"({'/'.join(RADAR_WAVEFORM_FAMILIES)})",
        n_invalid == 0,
        f"{n_invalid:,} windows with unexpected value" if n_invalid else "",
    ))

    # ── Fine → coarse mapping consistency ────────────────────────────────────
    expected_coarse = np.array([RADAR_FINE_TO_COARSE.get(s, "") for s in rst_vals])
    n_mismatch = int(np.sum(expected_coarse != wf_vals))
    results.append(_check(
        "radchar: waveform_type matches RADAR_FINE_TO_COARSE[radar_signal_type]",
        n_mismatch == 0,
        f"{n_mismatch:,} windows where coarse family != mapping" if n_mismatch else "",
    ))


def _validate_sub_ghz(f: h5py.File, results: list[tuple[bool, str]]):
    """Sub-GHz specific checks.

    Modulation routing is validated against LABEL_TO_MODULATION from the
    sub_ghz config — the check derives its expected values from the same
    config that drives the build, so it stays correct when the mapping changes.
    """
    try:
        cfg = importlib.import_module("datasets.sub_ghz.config")
        label_to_mod  = getattr(cfg, "LABEL_TO_MODULATION", {})
        lora_sf_set   = getattr(cfg, "LORA_SPREADING_FACTORS", None)
        wifi_standard = getattr(cfg, "WIFI_STANDARD", {})
    except ImportError:
        label_to_mod  = {}
        lora_sf_set   = None
        wifi_standard = {}
        results.append(_warn("could not import sub_ghz config — skipping modulation routing check"))

    # Read label_names and labels once — reused by both checks below
    label_names = _decode_str_field(f["label_names"]).tolist() if "label_names" in f else []
    labels      = f["labels"][:] if "labels" in f else None

    # ── Modulation routing: check every label against LABEL_TO_MODULATION ────
    if label_to_mod and "modulation_type" in f["meta"] and label_names and labels is not None:
        mod_vals = _decode_str_field(f["meta/modulation_type"])

        for label in label_names:
            expected = label_to_mod.get(label.lower())
            if expected is None:
                continue  # label not in mapping — no expectation to check

            mask = labels == label_names.index(label)
            if not np.any(mask):
                continue

            n_bad = int(np.sum(mod_vals[mask] != expected))
            results.append(_check(
                f"sub_ghz '{label}': modulation_type == '{expected}' (from LABEL_TO_MODULATION)",
                n_bad == 0,
                f"{n_bad} windows with unexpected modulation" if n_bad
                else f"{int(np.sum(mask)):,} windows",
            ))

    # ── LoRa spreading_factor ─────────────────────────────────────────────────
    if "spreading_factor" in f["meta"] and label_names and labels is not None:
        sf_values = f["meta/spreading_factor"][:]

        if "LoRa" not in label_names:
            results.append(_warn("LoRa label not found in label_names — skipping spreading_factor check"))
        else:
            lora_mask = labels == label_names.index("LoRa")
            sf_lora_values = sf_values[lora_mask]
            sf_counts = Counter(sf_lora_values.tolist())
            n_unknown = sf_counts.get(-1, 0)
            known_sfs = {f"SF{k}": v for k, v in sf_counts.items() if k != -1}
            results.append(_check(
                "LoRa spreading_factor parsed",
                n_unknown == 0,
                f"{n_unknown:,} windows with unknown SF (-1)" if n_unknown
                else ", ".join(f"{k}={v:,}" for k, v in sorted(known_sfs.items())),
            ))

            # Membership: every LoRa SF must be in the documented capture set.
            # Driven by LORA_SPREADING_FACTORS in the sub_ghz config so adding
            # new SFs only requires a config edit.
            if lora_sf_set is not None:
                sf_in_set = np.isin(sf_lora_values, list(lora_sf_set))
                n_out_of_set = int(np.sum(~sf_in_set))
                results.append(_check(
                    f"sub_ghz_lora_sf_membership (expected {sorted(lora_sf_set)})",
                    n_out_of_set == 0,
                    f"{n_out_of_set:,} LoRa windows with SF outside expected set"
                    if n_out_of_set else f"{int(np.sum(lora_mask)):,} LoRa windows",
                ))

    # ── 802.11ah wifi_standard routing ────────────────────────────────────────
    # Sub-GHz's only WiFi-canonical class is 802.11ah, so every window with
    # canonical label "WiFi" must carry wifi_standard == "802.11ah" (sourced
    # from WIFI_STANDARD["80211ah"] in the sub_ghz config).
    if (wifi_standard and "wifi_standard" in f["meta"]
            and label_names and labels is not None and "WiFi" in label_names):
        expected_wifi_std = wifi_standard.get("80211ah")
        if expected_wifi_std:
            ws_vals  = _decode_str_field(f["meta/wifi_standard"])
            wifi_mask = labels == label_names.index("WiFi")
            n_wifi   = int(np.sum(wifi_mask))
            n_bad    = int(np.sum(ws_vals[wifi_mask] != expected_wifi_std))
            results.append(_check(
                f"sub_ghz WiFi: wifi_standard == '{expected_wifi_std}' (from WIFI_STANDARD)",
                n_bad == 0,
                f"{n_bad} windows with unexpected wifi_standard" if n_bad
                else f"{n_wifi:,} windows",
            ))


def _validate_its_band(f: h5py.File, results: list[tuple[bool, str]]):
    """ITS Band-specific checks.

    Validates that WiFi windows carry the correct waveform_type and wifi_standard
    derived from Girmay et al. §4.1.1 (openWiFi IEEE 802.11a/g/n SDR implementation).
    """
    label_names = _decode_str_field(f["label_names"]).tolist() if "label_names" in f else []
    labels      = f["labels"][:] if "labels" in f else None

    if "WiFi" not in label_names or labels is None:
        results.append(_warn("its_band: WiFi label not found — skipping WiFi-specific checks"))
        return

    wifi_mask = labels == label_names.index("WiFi")
    n_wifi = int(np.sum(wifi_mask))

    # WiFi windows must have waveform_type == "OFDM"
    if "waveform_type" in f["meta"]:
        wf_vals = _decode_str_field(f["meta/waveform_type"])
        n_bad = int(np.sum(wf_vals[wifi_mask] != "OFDM"))
        results.append(_check(
            "its_band WiFi: waveform_type == 'OFDM'",
            n_bad == 0,
            f"{n_bad} windows with unexpected waveform_type" if n_bad else f"{n_wifi:,} windows",
        ))

    # WiFi windows must have wifi_standard == "802.11n" (Girmay et al. §4.1.1)
    if "wifi_standard" in f["meta"]:
        ws_vals = _decode_str_field(f["meta/wifi_standard"])
        n_bad = int(np.sum(ws_vals[wifi_mask] != "802.11n"))
        results.append(_check(
            "its_band WiFi: wifi_standard == '802.11n' (Girmay et al. §4.1.1)",
            n_bad == 0,
            f"{n_bad} windows with unexpected wifi_standard" if n_bad else f"{n_wifi:,} windows",
        ))


def _validate_nist_wifi_bluetooth(f: h5py.File, results: list[tuple[bool, str]]):
    """NIST WiFi-specific checks.

    Validates that modulation_type and wifi_standard are fully populated and
    per-file consistent with the MCS metadata documented in NIST TN 2237
    (doi: 10.6028/NIST.TN.2237). MCS was continuously monitored during capture
    and remained stable throughout:
      2_4ghz_indoor: MCS-15 → 802.11n,  64QAM
      5ghz_indoor:   MCS-8  → 802.11ac, 256QAM
      5ghz_outdoor:  MCS-8  → 802.11ac, 256QAM
    These are capture-level constants, not per-frame decoded values.
    """
    try:
        cfg = importlib.import_module("datasets.nist_wifi_bluetooth.config")
        file_config   = getattr(cfg, "FILE_CONFIG",   {})
        file_metadata = getattr(cfg, "FILE_METADATA", {})
    except ImportError:
        results.append(_warn("could not import nist_wifi_bluetooth config — skipping modulation/wifi_standard checks"))
        return

    # ── Overall: no "unknown" values ──────────────────────────────────────────
    for field_name, valid_values in [
        ("modulation_type", {"64QAM", "256QAM"}),
        ("wifi_standard",   {"802.11n", "802.11ac"}),
    ]:
        if field_name not in f["meta"]:
            results.append(_check(f"{field_name} present", False, "field missing from /meta"))
            continue
        vals  = _decode_str_field(f[f"meta/{field_name}"])
        n_bad = int(np.sum(~np.isin(vals, list(valid_values))))
        results.append(_check(
            f"nist: {field_name} in {sorted(valid_values)} (no 'unknown')",
            n_bad == 0,
            f"{n_bad} windows with unexpected value" if n_bad else f"{len(vals):,} windows OK",
        ))

    # ── Per-file consistency (config-driven via FILE_CONFIG + FILE_METADATA) ──
    # Join FILE_CONFIG and FILE_METADATA by filename stem to build the expected
    # (modulation_type, wifi_standard) for each (center_frequency_hz, environment) group.
    if "center_frequency_hz" not in f["meta"] or "environment" not in f["meta"]:
        results.append(_warn("center_frequency_hz or environment field missing — skipping per-file consistency check"))
        return
    if not file_config or not file_metadata:
        results.append(_warn("FILE_CONFIG or FILE_METADATA missing from config — skipping per-file consistency check"))
        return

    center_freq = f["meta/center_frequency_hz"][:]
    env_vals    = _decode_str_field(f["meta/environment"])
    mod_vals    = _decode_str_field(f["meta/modulation_type"]) if "modulation_type" in f["meta"] else None
    wifi_vals   = _decode_str_field(f["meta/wifi_standard"])   if "wifi_standard"   in f["meta"] else None

    group_expected = {
        (freq, env): file_metadata[stem]
        for stem, (freq, env, _label) in file_config.items()
        if stem in file_metadata
    }

    for (freq, env), expected in sorted(group_expected.items()):
        mask = (center_freq == freq) & (env_vals == env)
        if not np.any(mask):
            results.append(_warn(f"nist {env} {freq/1e9:.1f}GHz: no windows found"))
            continue
        n = int(np.sum(mask))

        if mod_vals is not None:
            exp_mod = expected["modulation_type"]
            n_bad   = int(np.sum(mod_vals[mask] != exp_mod))
            results.append(_check(
                f"nist {env} {freq/1e9:.1f}GHz: modulation_type == '{exp_mod}'",
                n_bad == 0,
                f"{n_bad} unexpected" if n_bad else f"{n:,} windows",
            ))

        if wifi_vals is not None:
            exp_std = expected["wifi_standard"]
            n_bad   = int(np.sum(wifi_vals[mask] != exp_std))
            results.append(_check(
                f"nist {env} {freq/1e9:.1f}GHz: wifi_standard == '{exp_std}'",
                n_bad == 0,
                f"{n_bad} unexpected" if n_bad else f"{n:,} windows",
            ))


def _validate_aggregated(f: h5py.File, results: list[tuple[bool, str]]):
    """Aggregated dataset checks.

    All source-specific checks are derived from the registry (label_represents)
    and per-source configs (WINDOW_METADATA_SCHEMA) — no hardcoded source names.
    """
    from datasets.aggregation.config import SELECTED_DATASETS, TARGET_WINDOW_SIZE

    # Pre-load source/routing fields once — reused across distribution prints,
    # snr_db checks, and routing checks to avoid redundant vlen string reads.
    src  = _decode_str_field(f["meta/source_dataset"])  if "source_dataset"  in f["meta"] else None
    tech = _decode_str_field(f["meta/technology_type"]) if "technology_type" in f["meta"] else None
    mod  = _decode_str_field(f["meta/modulation_type"]) if "modulation_type" in f["meta"] else None
    wav  = _decode_str_field(f["meta/waveform_type"])   if "waveform_type"   in f["meta"] else None
    rst  = _decode_str_field(f["meta/radar_signal_type"]) if "radar_signal_type" in f["meta"] else None

    # ── Source / tech / mod distributions (informational) ────────────────────
    if src is not None:
        counts = Counter(src.tolist())
        results.append(_info("source_dataset distribution: "
            + ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items()))))
    if tech is not None:
        counts = Counter(tech.tolist())
        results.append(_info("technology_type distribution: "
            + ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items()))))
    if mod is not None:
        counts = Counter(mod.tolist())
        results.append(_info("modulation_type distribution: "
            + ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items()))))

    # ── Cross-file: aggregated total == sum of per-dataset window counts ──────
    # If any enabled source's processed HDF5 is missing on disk, downgrade to
    # an info record (e.g. validating the aggregated file in isolation).
    n_aggregated = f["data"].shape[0] if "data" in f else None
    if n_aggregated is not None:
        per_source_counts: dict[str, int] = {}
        missing: list[str] = []
        for ds_name, ds_cfg in SELECTED_DATASETS.items():
            if not ds_cfg.get("enabled", True):
                continue
            entry = DATASET_REGISTRY.get(ds_name)
            if entry is None:
                missing.append(f"{ds_name} (not in registry)")
                continue
            ds_path = entry.processed_h5_path
            if not ds_path.exists():
                missing.append(f"{ds_name} ({ds_path})")
                continue
            try:
                with h5py.File(ds_path, "r") as df:
                    per_source_counts[ds_name] = int(df["data"].shape[0])
            except (OSError, KeyError) as e:
                missing.append(f"{ds_name} ({type(e).__name__})")

        if missing:
            results.append(_info(
                "aggregated_total_equals_per_dataset_sum: skipped — "
                f"per-dataset file(s) unavailable: {', '.join(missing)}"
            ))
        else:
            sum_per_dataset = sum(per_source_counts.values())
            results.append(_check(
                "aggregated_total_equals_per_dataset_sum",
                sum_per_dataset == n_aggregated,
                f"per_dataset_sum={sum_per_dataset:,}, aggregated={n_aggregated:,}",
            ))

    # ── Data shape: (N, 2, TARGET_WINDOW_SIZE) ────────────────────────────────
    if "data" in f:
        shape = f["data"].shape
        ok = len(shape) == 3 and shape[1] == 2 and shape[2] == TARGET_WINDOW_SIZE
        results.append(_check(
            f"data shape (N, 2, {TARGET_WINDOW_SIZE})",
            ok,
            f"got {shape}" if not ok else f"{shape[0]:,} × 2 × {shape[2]}",
        ))

    # ── original_window_size populated ────────────────────────────────────────
    # The -1 sentinel check itself is covered by the generic sentinel sweep
    # below (run on the aggregation REQUIRED_METADATA_SCHEMA). This block only
    # reports the per-size distribution as info, plus the consistency check
    # below which has aggregated-specific semantics.
    orig = f["meta/original_window_size"][:] if "original_window_size" in f["meta"] else None
    if orig is not None:
        size_counts = Counter(orig.tolist())
        detail = ", ".join(f"{k}samples={v:,}" for k, v in sorted(size_counts.items()))
        results.append(_info(f"original_window_size distribution: {detail}"))
    else:
        results.append(_check("original_window_size present", False, "field missing from /meta"))

    # ── original_window_size consistency with was_padded / was_cropped ────────
    if orig is not None and "was_padded" in f["meta"]:
        padded  = f["meta/was_padded"][:]
        cropped = f["meta/was_cropped"][:] if "was_cropped" in f["meta"] else np.zeros(len(orig), dtype=bool)

        parts = []
        n = int(np.sum(padded & (orig >= TARGET_WINDOW_SIZE)))
        if n:
            parts.append(f"{n} padded but orig >= {TARGET_WINDOW_SIZE}")
        n = int(np.sum(cropped & (orig <= TARGET_WINDOW_SIZE)))
        if n:
            parts.append(f"{n} cropped but orig <= {TARGET_WINDOW_SIZE}")
        neither = ~padded & ~cropped
        n = int(np.sum(neither & (orig != TARGET_WINDOW_SIZE)))
        if n:
            parts.append(f"{n} unmodified but orig != {TARGET_WINDOW_SIZE}")
        results.append(_check(
            "original_window_size consistent with was_padded/was_cropped",
            len(parts) == 0,
            "; ".join(parts) if parts else "",
        ))

    # ── snr_db: presence, dtype, non-NaN for sources that provide it ─────────
    # "Provides snr_db" is determined by whether the source's own
    # WINDOW_METADATA_SCHEMA includes snr_db — derived from config, not hardcoded.
    if "snr_db" in f["meta"]:
        snr_ds = f["meta/snr_db"]
        results.append(_info(f"snr_db present (dtype={snr_ds.dtype})"))

        ok_float = snr_ds.dtype.kind == "f"
        results.append(_check(
            "snr_db dtype is float",
            ok_float,
            str(snr_ds.dtype) if not ok_float else "",
        ))

        if ok_float and src is not None:
            snr = snr_ds[:]
            for ds_name, ds_cfg in SELECTED_DATASETS.items():
                if not ds_cfg.get("enabled", True):
                    continue
                entry = DATASET_REGISTRY.get(ds_name)
                if entry is None or "snr_db" not in entry.get_available_metadata_fields():
                    continue  # source does not provide snr_db → NaN fill is expected
                mask = src == ds_name
                if not np.any(mask):
                    continue
                n_nan = int(np.sum(np.isnan(snr[mask])))
                results.append(_check(
                    f"{ds_name}: snr_db not NaN (snr_db in source schema)",
                    n_nan == 0,
                    f"{n_nan} NaN values" if n_nan
                    else f"all {int(np.sum(mask)):,} windows have SNR",
                ))
    else:
        results.append(_check(
            "snr_db present",
            False,
            "field missing from /meta — check WINDOW_METADATA_SCHEMA in aggregation config",
        ))

    # ── Label routing semantics: derived from label_represents in registry ────
    # Each source dataset's label_represents determines what technology_type and
    # modulation_type should contain for its windows in the aggregated file.
    # Adding a new source dataset to the registry automatically adds its check.
    if src is None:
        results.append(_check("routing checks", False, "source_dataset field missing"))
        return

    for ds_name, ds_cfg in SELECTED_DATASETS.items():
        if not ds_cfg.get("enabled", True):
            continue
        entry = DATASET_REGISTRY.get(ds_name)
        if entry is None:
            results.append(_warn(f"{ds_name}: not in registry, skipping routing check"))
            continue

        mask = src == ds_name
        if not np.any(mask):
            results.append(_warn(f"{ds_name}: no windows found in aggregated file"))
            continue

        n   = int(np.sum(mask))
        lr  = entry.label_represents

        if lr == LabelRepresents.MODULATION:
            if tech is not None:
                n_bad = int(np.sum(tech[mask] != "unknown"))
                results.append(_check(
                    f"{ds_name}: technology_type == 'unknown' (label_represents=modulation)",
                    n_bad == 0,
                    f"{n_bad} windows with unexpected technology" if n_bad else f"{n:,} windows",
                ))
            if mod is not None:
                n_bad = int(np.sum(mod[mask] == "unknown"))
                results.append(_check(
                    f"{ds_name}: modulation_type populated (label_represents=modulation)",
                    n_bad == 0,
                    f"{n_bad} windows with 'unknown' modulation" if n_bad else "",
                ))
            if rst is not None:
                n_bad = int(np.sum(rst[mask] != "unknown"))
                results.append(_check(
                    f"{ds_name}: radar_signal_type == 'unknown' (label_represents=modulation)",
                    n_bad == 0,
                    f"{n_bad} windows with unexpected radar_signal_type" if n_bad else f"{n:,} windows",
                ))

        elif lr == LabelRepresents.TECHNOLOGY:
            if tech is not None:
                n_bad = int(np.sum(tech[mask] == "unknown"))
                results.append(_check(
                    f"{ds_name}: technology_type populated (label_represents=technology)",
                    n_bad == 0,
                    f"{n_bad} windows with 'unknown' technology" if n_bad else f"{n:,} windows",
                ))
            if rst is not None:
                n_bad = int(np.sum(rst[mask] != "unknown"))
                results.append(_check(
                    f"{ds_name}: radar_signal_type == 'unknown' (label_represents=technology)",
                    n_bad == 0,
                    f"{n_bad} windows with unexpected radar_signal_type" if n_bad else f"{n:,} windows",
                ))

        else:  # RADAR_WAVEFORM or other domain-specific
            bad = 0
            if tech is not None: bad += int(np.sum(tech[mask] != "unknown"))
            if mod  is not None: bad += int(np.sum(mod[mask]  != "unknown"))
            results.append(_check(
                f"{ds_name}: technology_type and modulation_type both 'unknown'"
                f" (label_represents={lr.value})",
                bad == 0,
                f"{bad} windows with unexpected non-'unknown' values" if bad
                else f"{n:,} windows",
            ))
            # waveform_type must be populated with a coarse radar family (not "unknown",
            # and not the raw fine class) for radar sources.
            if wav is not None:
                n_unknown = int(np.sum(wav[mask] == "unknown"))
                results.append(_check(
                    f"{ds_name}: waveform_type populated (label_represents={lr.value})",
                    n_unknown == 0,
                    f"{n_unknown:,} windows with 'unknown' waveform_type" if n_unknown
                    else f"{n:,} windows",
                ))
                n_noncoarse = int(np.sum(~np.isin(wav[mask], RADAR_WAVEFORM_FAMILIES)))
                results.append(_check(
                    f"{ds_name}: waveform_type is a coarse radar family "
                    f"({'/'.join(RADAR_WAVEFORM_FAMILIES)})",
                    n_noncoarse == 0,
                    f"{n_noncoarse:,} windows with non-coarse waveform_type" if n_noncoarse
                    else "",
                ))
            # radar_signal_type (fine target) must be populated for radar sources.
            if rst is not None:
                n_bad = int(np.sum(rst[mask] == "unknown"))
                results.append(_check(
                    f"{ds_name}: radar_signal_type populated (label_represents={lr.value})",
                    n_bad == 0,
                    f"{n_bad:,} windows with 'unknown' radar_signal_type" if n_bad
                    else f"{n:,} windows",
                ))

    # ── Generic -1 sentinel sweep on the aggregated file ──────────────────────
    # Two-part sweep:
    #   (a) Aggregation REQUIRED_METADATA_SCHEMA — covers fields the aggregator
    #       writes itself (e.g. original_window_size). Applied to all rows.
    #   (b) Per-source sweep — for each enabled source dataset, run the sweep
    #       on the slice of the aggregated file where source_dataset == ds_name,
    #       using that source's own WINDOW_METADATA_SCHEMA. This catches per-
    #       source integer fields (e.g. sample_index, number_of_pulses) that
    #       the aggregator copies through from the source.
    from datasets.aggregation.config import REQUIRED_METADATA_SCHEMA
    _sentinel_sweep(f, REQUIRED_METADATA_SCHEMA, results,
                    source_label="aggregation")

    if src is not None:
        for ds_name, ds_cfg in SELECTED_DATASETS.items():
            if not ds_cfg.get("enabled", True):
                continue
            entry = DATASET_REGISTRY.get(ds_name)
            if entry is None:
                continue
            schema_fields = entry.get_available_metadata_schema()
            if not schema_fields:
                continue
            mask = src == ds_name
            if not np.any(mask):
                continue
            _sentinel_sweep(f, schema_fields, results,
                            source_label=f"aggregation:{ds_name}", mask=mask)


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_config_snapshot(h5_path: Path) -> tuple[Path | None, dict | None]:
    """Look for ``{h5_path}.config.json`` and return (path, parsed_dict).

    Returns (None, None) if the snapshot file does not exist or cannot be parsed.
    """
    snapshot_path = h5_path.with_name(h5_path.name + ".config.json")
    if not snapshot_path.exists():
        return None, None
    try:
        with open(snapshot_path) as f:
            return snapshot_path, json.load(f)
    except (OSError, json.JSONDecodeError):
        return snapshot_path, None


def _build_report(
    dataset_results: dict[str, tuple[bool, list[dict], DatasetEntry]],
    fingerprints: tuple[dict, dict] | None = None,
    provenance: dict | None = None,
    started_at: str | None = None,
) -> dict:
    """Assemble the JSON report from per-dataset validation results.

    fingerprints is (before, after), each keyed by dataset name, when the run
    was asked to identify the files it checked; None otherwise.
    """
    datasets_section: dict[str, dict] = {}
    total_checks = 0
    total_passed = 0
    total_failed = 0

    for name, (passed, records, entry) in dataset_results.items():
        h5_path = entry.processed_h5_path
        snapshot_path, snapshot_dict = _load_config_snapshot(h5_path)

        n_checks = sum(1 for r in records if r["level"] == "check")
        n_check_passed = sum(1 for r in records if r["level"] == "check" and r["passed"])
        n_check_failed = n_checks - n_check_passed
        n_warnings = sum(1 for r in records if r["level"] == "warn")

        total_checks += n_checks
        total_passed += n_check_passed
        total_failed += n_check_failed

        datasets_section[name] = {
            "passed": passed,
            "h5_path": str(h5_path),
            "config_snapshot_path": str(snapshot_path) if snapshot_path else None,
            "config_snapshot": snapshot_dict,
            "file": (
                _file_record(h5_path, fingerprints[0].get(name), fingerprints[1].get(name))
                if fingerprints is not None else None
            ),
            "summary": {
                "total_checks": n_checks,
                "passed": n_check_passed,
                "failed": n_check_failed,
                "warnings": n_warnings,
            },
            # Drop the preformatted "line" field — it is stdout-only sugar.
            "checks": [
                {k: v for k, v in r.items() if k != "line"}
                for r in records
            ],
        }

    overall_pass = all(p for p, _, _ in dataset_results.values())
    if fingerprints is None:
        files_unchanged = None
    else:
        files_unchanged = all(
            d["file"]["unchanged_during_validation"] is True
            for d in datasets_section.values()
        )
    return {
        "validator_version": VALIDATOR_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "overall_pass": overall_pass,
        # None when the run did not fingerprint; False if any file was missing
        # or differed between the two passes. Independent of overall_pass,
        # which reports the checks alone.
        "files_fingerprinted": fingerprints is not None,
        "files_unchanged_during_validation": files_unchanged,
        "provenance": provenance,
        "summary": {
            "total_checks": total_checks,
            "passed": total_passed,
            "failed": total_failed,
            "datasets_total": len(dataset_results),
            "datasets_passed": sum(1 for p, _, _ in dataset_results.values() if p),
            "datasets_failed": sum(1 for p, _, _ in dataset_results.values() if not p),
        },
        "datasets": datasets_section,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate processed RF HDF5 datasets.")
    parser.add_argument(
        "--dataset", nargs="*",
        help="Dataset name(s) to validate. Defaults to all registered datasets.",
    )
    parser.add_argument(
        "--save-report", metavar="PATH", default=None,
        help="If set, also write a structured JSON validation report to PATH.",
    )
    parser.add_argument(
        "--fingerprint", action="store_true",
        help="Record each file's size, SHA-256 and MD5 in the report, before "
             "and after the checks, and fail if a file changed in between. "
             "Reads every file twice in full; requires --save-report.",
    )
    args = parser.parse_args()
    if args.fingerprint and not args.save_report:
        parser.error("--fingerprint has nowhere to record its result without --save-report")

    from datasets.aggregation.config import OUTPUT_H5_PATH as AGG_H5_PATH

    entries = dict(DATASET_REGISTRY)
    entries["aggregation"] = DatasetEntry(
        name="aggregation",
        processed_h5_path=AGG_H5_PATH,
        build_module="datasets.aggregation",
    )

    names = args.dataset if args.dataset else sorted(entries.keys())
    unknown = [n for n in names if n not in entries]
    if unknown:
        print(f"Unknown datasets: {unknown}")
        print(f"Available: {sorted(entries.keys())}")
        sys.exit(1)

    started_at = datetime.now(timezone.utc).isoformat()
    paths = {name: entries[name].processed_h5_path for name in names}
    before = None
    if args.fingerprint:
        print(f"Fingerprinting {len(paths)} file(s) before validation ...", flush=True)
        before = fingerprint_files(paths)

    passed = []
    failed = []
    dataset_results: dict[str, tuple[bool, list[dict], DatasetEntry]] = {}
    for name in names:
        ok, records = validate_dataset(name, entries[name])
        dataset_results[name] = (ok, records, entries[name])
        (passed if ok else failed).append(name)

    fingerprints = None
    if args.fingerprint:
        print(f"\nFingerprinting {len(paths)} file(s) after validation ...", flush=True)
        fingerprints = (before, fingerprint_files(paths))

    files_changed = False
    if args.save_report:
        report = _build_report(
            dataset_results,
            fingerprints=fingerprints,
            provenance=code_provenance(names),
            started_at=started_at,
        )
        files_changed = report["files_unchanged_during_validation"] is False
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
        print()
        print(f"Validation report written to {report_path}")

    print()
    if files_changed:
        moved = [
            name for name, d in report["datasets"].items()
            if d["file"]["unchanged_during_validation"] is not True
        ]
        print(f"{FAIL} file(s) missing or modified while being validated: {', '.join(moved)}")
        print("  The checks above do not describe a single state of these files.")
    if failed:
        print(f"{FAIL} {len(failed)} dataset(s) failed: {', '.join(failed)}")
        sys.exit(1)
    elif files_changed:
        sys.exit(1)
    else:
        print(f"{PASS} All {len(passed)} dataset(s) passed validation.")


if __name__ == "__main__":
    main()
