"""
Disk-backed DataFrame cache.

Stores DataFrames as Parquet files on disk instead of holding them in RAM.
This prevents memory explosions when loading large datasets (e.g. PPMI
biospecimen data) that can crash the host machine via the OOM killer.

All API modules (data, analysis, statistics) share a single cache instance
so that a cache_key produced by one module is readable by the others.

Two storage modes:
  1. Single-file  — one .parquet per cache_key (used by engineered/train/test)
  2. Modular      — one directory per cache_key with a .parquet per modality
                     plus a _manifest.json catalog (used by merged data loads)
"""

import gc
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(tempfile.gettempdir()) / "pie_workbench_cache"
_CACHE_DIR.mkdir(exist_ok=True)

# Lightweight in-memory stores (these are small — only metadata, not data)
_column_meta: Dict[str, Dict[str, Any]] = {}   # Pre-computed column stats
_cached_paths: Dict[str, Path] = {}             # Maps cache_key -> path on disk
_modular_keys: Set[str] = set()                 # cache_keys using multi-file format


# ---------------------------------------------------------------------------
# Dtype optimization
# ---------------------------------------------------------------------------

def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce DataFrame memory usage by downcasting numeric types and
    converting low-cardinality object columns to categoricals.

    Typically saves 40-60 % of memory on PPMI-style wide DataFrames.
    """
    n_rows = len(df)
    for col in df.columns:
        col_dtype = df[col].dtype

        if col_dtype == np.float64:
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif col_dtype in (np.int64, np.int32):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif col_dtype == object and n_rows > 0:
            n_unique = df[col].nunique()
            if n_unique < n_rows * 0.5:
                try:
                    # Ensure all values are strings before converting to
                    # category — mixed float/str values from pipe-separation
                    # can cause Parquet serialization failures.
                    df[col] = df[col].where(df[col].isna(), df[col].astype(str))
                    df[col] = df[col].astype("category")
                except Exception:
                    pass  # Leave as object if conversion fails
    return df


# ---------------------------------------------------------------------------
# Column metadata (tiny — safe to keep in memory)
# ---------------------------------------------------------------------------

def _compute_column_meta(df: pd.DataFrame) -> Dict[str, Any]:
    """Pre-compute column statistics so /columns never reloads the full DF."""
    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        is_numeric = (
            np.issubdtype(df[col].dtype, np.number)
            if not isinstance(df[col].dtype, pd.CategoricalDtype)
            else np.issubdtype(df[col].cat.categories.dtype, np.number)
        )
        n_unique = int(df[col].nunique())
        n_null = int(df[col].isnull().sum())
        null_pct = round(n_null / len(df) * 100, 2) if len(df) > 0 else 0

        columns.append({
            "name": col,
            "dtype": dtype,
            "is_numeric": is_numeric,
            "is_categorical": not is_numeric and n_unique < 50,
            "unique_count": n_unique,
            "null_count": n_null,
            "null_pct": null_pct,
        })
    return {"columns": columns, "total_rows": len(df)}


def get_column_meta(cache_key: str) -> Optional[Dict[str, Any]]:
    """Return pre-computed column metadata, or None if unavailable."""
    return _column_meta.get(cache_key)


# ---------------------------------------------------------------------------
# Core cache operations (single-file mode)
# ---------------------------------------------------------------------------

def store(cache_key: str, data: Any, *, compute_meta: bool = True) -> None:
    """Persist *data* to disk and free it from RAM.

    Supports ``pd.DataFrame`` and ``dict[str, pd.DataFrame]``.
    """
    if isinstance(data, pd.DataFrame):
        # Pre-compute lightweight metadata before downcast
        if compute_meta:
            _column_meta[cache_key] = _compute_column_meta(data)

        data = downcast_dataframe(data)
        path = _CACHE_DIR / f"{cache_key}.parquet"
        data.to_parquet(path, engine="pyarrow", index=False)
        _cached_paths[cache_key] = path
        logger.info("Cached DataFrame (%s) to %s", cache_key, path)

    elif isinstance(data, dict):
        dir_path = _CACHE_DIR / cache_key
        dir_path.mkdir(parents=True, exist_ok=True)
        for name, df in data.items():
            if isinstance(df, pd.DataFrame):
                df = downcast_dataframe(df)
                df.to_parquet(dir_path / f"{name}.parquet", engine="pyarrow", index=False)
        # Sentinel so we know it's a dict-of-DataFrames
        (dir_path / "_is_dict").touch()
        _cached_paths[cache_key] = dir_path
        logger.info("Cached dict (%s) to %s", cache_key, dir_path)
    else:
        raise TypeError(f"Unsupported cache type: {type(data)}")

    # Free the Python objects — the data now lives on disk
    del data
    gc.collect()


def load(cache_key: str) -> Any:
    """Load cached data back from disk.

    Categorical columns created by ``downcast_dataframe`` are converted back
    to their base dtype (same as ``_decat`` does for modular loads).
    """
    path = _cached_paths.get(cache_key)
    if path is None:
        raise KeyError(f"Cache key not found: {cache_key}")

    if path.is_file() and path.suffix == ".parquet":
        return _decat(pd.read_parquet(path, engine="pyarrow"))

    if path.is_dir() and (path / "_is_dict").exists():
        result = {}
        for f in sorted(path.glob("*.parquet")):
            result[f.stem] = _decat(pd.read_parquet(f, engine="pyarrow"))
        return result

    raise KeyError(f"Cache path invalid for key: {cache_key}")


def load_sample(cache_key: str, n: int = 100) -> pd.DataFrame:
    """Load only the first *n* rows — useful for heatmaps / previews."""
    path = _cached_paths.get(cache_key)
    if path is None or not (path.is_file() and path.suffix == ".parquet"):
        raise KeyError(f"Cache key not found or not a DataFrame: {cache_key}")

    table = pq.read_table(path).slice(0, n)
    return _decat(table.to_pandas())


def exists(cache_key: str) -> bool:
    return cache_key in _cached_paths


def delete(cache_key: str) -> None:
    path = _cached_paths.pop(cache_key, None)
    _column_meta.pop(cache_key, None)
    _modular_keys.discard(cache_key)
    if path is None:
        return
    if path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path, ignore_errors=True)


def clear() -> None:
    """Remove all cached data from disk and metadata from memory."""
    _cached_paths.clear()
    _column_meta.clear()
    _modular_keys.clear()
    if _CACHE_DIR.exists():
        shutil.rmtree(_CACHE_DIR, ignore_errors=True)
    _CACHE_DIR.mkdir(exist_ok=True)
    gc.collect()


# ---------------------------------------------------------------------------
# Modular cache operations (multi-file mode)
# ---------------------------------------------------------------------------

_JOIN_KEYS = ("PATNO", "EVENT_ID")


def _safe_filename(name: str) -> str:
    """Sanitise a modality/table name into a safe filename stem."""
    return re.sub(r"[^\w\-]", "_", name)


def store_modality(
    cache_key: str,
    modality_name: str,
    df: pd.DataFrame,
    total_rows: int = 0,
) -> int:
    """Persist one modality DataFrame to the modular cache directory.

    Returns the number of non-key columns stored (for progress tracking).
    """
    dir_path = _CACHE_DIR / cache_key
    dir_path.mkdir(parents=True, exist_ok=True)

    # Pre-compute per-column metadata before downcast
    meta_entries = []
    for col in df.columns:
        if col in _JOIN_KEYS:
            continue
        dtype = str(df[col].dtype)
        is_numeric = (
            np.issubdtype(df[col].dtype, np.number)
            if not isinstance(df[col].dtype, pd.CategoricalDtype)
            else np.issubdtype(df[col].cat.categories.dtype, np.number)
        )
        n_unique = int(df[col].nunique())
        n_null = int(df[col].isnull().sum())
        n_rows = len(df)
        null_pct = round(n_null / n_rows * 100, 2) if n_rows > 0 else 0

        meta_entries.append({
            "name": col,
            "dtype": dtype,
            "is_numeric": is_numeric,
            "is_categorical": not is_numeric and n_unique < 50,
            "unique_count": n_unique,
            "null_count": n_null,
            "null_pct": null_pct,
            "_source_file": _safe_filename(modality_name),
        })

    # Accumulate metadata
    if cache_key not in _column_meta:
        _column_meta[cache_key] = {"columns": [], "total_rows": 0, "_catalog": {}}
    existing = _column_meta[cache_key]
    catalog = existing["_catalog"]

    for entry in meta_entries:
        col_name = entry["name"]
        source = entry["_source_file"]
        # Only record the first occurrence in the catalog
        if col_name not in catalog:
            catalog[col_name] = source
            existing["columns"].append(entry)

    # Downcast and write parquet
    df = downcast_dataframe(df)
    safe = _safe_filename(modality_name)
    df.to_parquet(dir_path / f"{safe}.parquet", engine="pyarrow", index=False)
    _cached_paths[cache_key] = dir_path
    logger.info("Stored modality '%s' (%d cols) for %s", modality_name, len(meta_entries), cache_key)
    return len(meta_entries)


def finalize_modular_cache(cache_key: str, total_rows: int) -> None:
    """Write the _manifest.json and mark this cache_key as modular."""
    dir_path = _CACHE_DIR / cache_key
    meta = _column_meta.get(cache_key, {"columns": [], "_catalog": {}})
    meta["total_rows"] = total_rows

    # Build catalog (col -> file) and rename _source_file to source_modality
    catalog = meta.pop("_catalog", {})
    for entry in meta["columns"]:
        source = entry.pop("_source_file", None)
        if source is not None:
            entry["source_modality"] = source

    manifest = {
        "total_rows": total_rows,
        "catalog": catalog,
        "columns": meta["columns"],
    }
    manifest_path = dir_path / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    logger.info("Finalized modular cache %s: %d rows, %d columns, manifest at %s",
                cache_key, total_rows, len(meta["columns"]), manifest_path)

    _modular_keys.add(cache_key)
    _cached_paths[cache_key] = dir_path


def is_modular(cache_key: str) -> bool:
    """Check whether *cache_key* uses multi-file modular storage."""
    if cache_key in _modular_keys:
        return True
    # Fallback: check disk
    manifest = _CACHE_DIR / cache_key / "_manifest.json"
    if manifest.exists():
        _modular_keys.add(cache_key)
        _cached_paths.setdefault(cache_key, _CACHE_DIR / cache_key)
        return True
    return False


# ---------------------------------------------------------------------------
# Modular loading helpers
# ---------------------------------------------------------------------------

def _read_manifest(cache_key: str) -> dict:
    """Read _manifest.json from disk."""
    manifest_path = _CACHE_DIR / cache_key / "_manifest.json"
    if not manifest_path.exists():
        raise KeyError(f"No manifest for cache key: {cache_key}")
    return json.loads(manifest_path.read_text())


def _load_manifest_into_meta(cache_key: str) -> None:
    """Repopulate _column_meta from disk manifest (process-restart recovery)."""
    manifest = _read_manifest(cache_key)
    catalog = manifest.get("catalog", {})
    # Re-attach source_modality for backward compat with old manifests
    for entry in manifest["columns"]:
        if "source_modality" not in entry:
            entry["source_modality"] = catalog.get(entry["name"], "unknown")
    _column_meta[cache_key] = {
        "columns": manifest["columns"],
        "total_rows": manifest["total_rows"],
    }
    _modular_keys.add(cache_key)
    _cached_paths.setdefault(cache_key, _CACHE_DIR / cache_key)
    logger.info("Loaded manifest into memory for %s", cache_key)


def load_columns(cache_key: str, columns: List[str]) -> pd.DataFrame:
    """Load only the requested columns from a modular cache.

    Reads the manifest to determine which Parquet files contain the
    requested columns, then loads just those columns (plus PATNO/EVENT_ID)
    using PyArrow column projection.  Merges with outer join on join keys.
    """
    manifest = _read_manifest(cache_key)
    catalog = manifest["catalog"]
    dir_path = _CACHE_DIR / cache_key

    # Group requested columns by source file
    file_cols: Dict[str, List[str]] = {}
    for col in columns:
        source_file = catalog.get(col)
        if source_file is None:
            logger.warning("Column '%s' not found in manifest for %s", col, cache_key)
            continue
        file_cols.setdefault(source_file, []).append(col)

    if not file_cols:
        raise KeyError(f"None of the requested columns found in cache: {columns}")

    frames = []
    for source_file, cols in file_cols.items():
        parquet_path = dir_path / f"{source_file}.parquet"
        if not parquet_path.exists():
            logger.warning("Parquet file not found: %s", parquet_path)
            continue
        # Always include join keys + requested columns
        load_cols = list(_JOIN_KEYS) + cols
        # Only request columns that actually exist in the file
        schema = pq.read_schema(parquet_path)
        available = set(schema.names)
        load_cols = [c for c in load_cols if c in available]
        df = _decat(pd.read_parquet(parquet_path, columns=load_cols, engine="pyarrow"))
        frames.append(df)

    if len(frames) == 1:
        return frames[0]

    # Merge frames on join keys
    result = frames[0]
    for other in frames[1:]:
        result = pd.merge(result, other, on=list(_JOIN_KEYS), how="outer")
    return result


def _decat(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all Categorical columns back to their base dtypes.

    Parquet round-trips preserve Categorical dtypes from ``downcast_dataframe``.
    Merging frames with outer joins (or later feature engineering that inserts
    new values like ``_OTHER_``) fails on Categoricals, so we undo the
    optimisation here where it matters.
    """
    cat_cols = df.select_dtypes(include=["category"]).columns
    if len(cat_cols):
        df[cat_cols] = df[cat_cols].astype(object)
    return df


def load_modalities(cache_key: str, modality_names: List[str]) -> pd.DataFrame:
    """Load entire Parquet files for named modalities and merge on join keys."""
    dir_path = _CACHE_DIR / cache_key
    if not dir_path.is_dir():
        raise KeyError(f"Modular cache dir not found: {cache_key}")

    frames = []
    for name in modality_names:
        safe = _safe_filename(name)
        parquet_path = dir_path / f"{safe}.parquet"
        if not parquet_path.exists():
            logger.warning("Modality file not found: %s", parquet_path)
            continue
        frames.append(_decat(pd.read_parquet(parquet_path, engine="pyarrow")))

    if not frames:
        raise KeyError(f"No modality files found for: {modality_names}")

    if len(frames) == 1:
        return frames[0]

    result = frames[0]
    for other in frames[1:]:
        result = pd.merge(result, other, on=list(_JOIN_KEYS), how="outer")
    return result


def list_modality_files(cache_key: str) -> List[str]:
    """Return the stems of all .parquet files in a modular cache directory."""
    dir_path = _CACHE_DIR / cache_key
    if not dir_path.is_dir():
        return []
    return [f.stem for f in sorted(dir_path.glob("*.parquet"))]
