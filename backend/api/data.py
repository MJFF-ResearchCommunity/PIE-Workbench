"""
Data ingestion API endpoints.

Handles data loading, preview, validation, and processing using PIE-clean.
"""

import gc
import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import numpy as np
import psutil

from . import cache as disk_cache

logger = logging.getLogger(__name__)

# Import PIE-clean constants (loader functions imported lazily inside the task)
try:
    from pie_clean import ALL_MODALITIES
except ImportError:
    ALL_MODALITIES = [
        "subject_characteristics",
        "medical_history",
        "motor_assessments",
        "non_motor_assessments",
        "biospecimen"
    ]

router = APIRouter()


# Task store for long-running operations
_tasks: Dict[str, Dict[str, Any]] = {}


class DataLoadRequest(BaseModel):
    """Request model for data loading."""
    data_path: str
    modalities: List[str] = []
    merge_output: bool = False
    clean_data: bool = True


class ProcessingStatus(BaseModel):
    """Status of a long-running task."""
    task_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.get("/modalities")
async def get_available_modalities():
    """Get list of available data modalities."""
    return {
        "modalities": [
            {"id": "subject_characteristics", "name": "Subject Characteristics", "description": "Demographics, family history, cohort info"},
            {"id": "medical_history", "name": "Medical History", "description": "Medications, vital signs, physical exams"},
            {"id": "motor_assessments", "name": "Motor Assessments (MDS-UPDRS)", "description": "Motor function scores and evaluations"},
            {"id": "non_motor_assessments", "name": "Non-Motor Assessments", "description": "Cognitive, communication, quality of life"},
            {"id": "biospecimen", "name": "Biospecimen", "description": "Biological samples and lab results"},
        ]
    }


@router.post("/detect_modalities")
async def detect_modalities(data_path: str):
    """Detect available modalities in the given data directory."""
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"Data path not found: {data_path}")

    folder_mappings = {
        "subject_characteristics": "_Subject_Characteristics",
        "medical_history": "Medical_History",
        "motor_assessments": "Motor___MDS-UPDRS",
        "non_motor_assessments": "Non-motor_Assessments",
        "biospecimen": "Biospecimen"
    }

    detected = []
    for modality, folder in folder_mappings.items():
        folder_path = os.path.join(data_path, folder)
        if os.path.exists(folder_path):
            # Count CSV files in the folder
            csv_files = list(Path(folder_path).glob("*.csv"))
            detected.append({
                "id": modality,
                "available": True,
                "file_count": len(csv_files),
                "folder": folder
            })
        else:
            detected.append({
                "id": modality,
                "available": False,
                "file_count": 0,
                "folder": folder
            })

    return {"modalities": detected, "data_path": data_path}


@router.post("/preview")
async def preview_data(data_path: str, modality: Optional[str] = None, limit: int = 50):
    """Preview data from a CSV file or modality folder."""
    if not os.path.exists(data_path):
        raise HTTPException(status_code=404, detail=f"Path not found: {data_path}")

    try:
        if os.path.isfile(data_path):
            df = pd.read_csv(data_path, nrows=limit)
        else:
            # Find first CSV in the folder
            csv_files = list(Path(data_path).glob("*.csv"))
            if not csv_files:
                raise HTTPException(status_code=404, detail="No CSV files found in directory")
            df = pd.read_csv(csv_files[0], nrows=limit)

        # Calculate missingness info
        null_counts = df.isnull().sum()
        total_rows = len(df)

        columns_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            null_pct = (null_counts[col] / total_rows * 100) if total_rows > 0 else 0
            columns_info.append({
                "name": col,
                "dtype": dtype,
                "null_count": int(null_counts[col]),
                "null_pct": round(null_pct, 2),
                "sample_values": df[col].dropna().head(5).tolist()
            })

        return {
            "columns": columns_info,
            "row_count": total_rows,
            "data": df.replace({np.nan: None}).to_dict(orient="records"),
            "shape": list(df.shape)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load")
async def start_data_loading(request: DataLoadRequest, background_tasks: BackgroundTasks):
    """Start a background task to load data."""
    task_id = str(uuid.uuid4())

    _tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Initializing data loading...",
        "result": None,
        "error": None,
        "started_at": datetime.now().isoformat()
    }

    background_tasks.add_task(
        _load_data_task,
        task_id,
        request.data_path,
        request.modalities or ALL_MODALITIES,
        request.merge_output,
        request.clean_data
    )

    return {"task_id": task_id, "status": "started"}


_MAX_TASK_LOGS = 2000  # Cap log entries per task to prevent unbounded memory growth


def _update_task(task_id: str, **kwargs):
    """Helper to update task state and append to log history."""
    task = _tasks[task_id]
    for key, value in kwargs.items():
        task[key] = value
    # Append message to log history so the frontend can display all messages
    if "message" in kwargs and kwargs["message"]:
        if "logs" not in task:
            task["logs"] = []
        if len(task["logs"]) < _MAX_TASK_LOGS:
            task["logs"].append({
                "timestamp": datetime.now().isoformat(),
                "message": kwargs["message"]
            })


class _TaskLogHandler(logging.Handler):
    """Logging handler that forwards Python log records into a task's log array."""

    def __init__(self, task_id: str):
        super().__init__()
        self.task_id = task_id

    def emit(self, record: logging.LogRecord):
        try:
            module = record.module or record.name
            msg = f"{module} [{record.levelname}] {record.getMessage()}"
            _update_task(self.task_id, message=msg)
        except Exception:
            pass


def _count_csv_files(folder_path: str) -> tuple:
    """Count CSV files and estimate total size in a folder."""
    csv_files = list(Path(folder_path).glob("*.csv"))
    total_size = sum(f.stat().st_size for f in csv_files)
    return csv_files, total_size


def _cleanup_completed_tasks():
    """Remove completed/failed tasks from _tasks to free memory (especially logs)."""
    to_remove = [
        tid for tid, t in _tasks.items()
        if t.get("status") in ("completed", "failed")
    ]
    for tid in to_remove:
        del _tasks[tid]


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def _collect_pairs(df: pd.DataFrame, target_set: set) -> None:
    """Efficiently collect unique (PATNO, EVENT_ID) tuples into target_set."""
    if df.empty or "PATNO" not in df.columns or "EVENT_ID" not in df.columns:
        return
    unique = df[["PATNO", "EVENT_ID"]].drop_duplicates()
    target_set.update(
        zip(unique["PATNO"].astype(str), unique["EVENT_ID"])
    )


# Mapping from modality IDs to folder names
_MODALITY_FOLDERS = {
    "subject_characteristics": "_Subject_Characteristics",
    "medical_history": "Medical_History",
    "motor_assessments": "Motor___MDS-UPDRS",
    "non_motor_assessments": "Non-motor_Assessments",
    "biospecimen": "Biospecimen"
}

_MODALITY_DISPLAY_NAMES = {
    "subject_characteristics": "Subject Characteristics",
    "medical_history": "Medical History",
    "motor_assessments": "Motor Assessments (MDS-UPDRS)",
    "non_motor_assessments": "Non-Motor Assessments",
    "biospecimen": "Biospecimen"
}


def _prefix_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Rename non-key columns with a prefix to avoid name collisions."""
    rename = {
        col: f"{prefix}_{col}"
        for col in df.columns
        if col not in ("PATNO", "EVENT_ID")
    }
    return df.rename(columns=rename)


_ID_DATE_PATTERNS = frozenset(["ID", "DATE", "TIME", "PATNO", "EVENT", "REC_ID", "ORIG_ENTRY"])


def _resolve_pipe_separated_values(df: pd.DataFrame, modality: str = "") -> pd.DataFrame:
    """Resolve pipe-separated values produced by PIE-clean's deduplication.

    PIE-clean joins conflicting values with '|' when consolidating duplicate
    (PATNO, EVENT_ID) rows.  This mirrors the resolution logic from the PIE
    library's pipeline (pipeline.py lines 270-319):

      * Numeric pipe values (e.g. '51.6|53.0') are averaged → 52.3
      * Non-numeric pipe values are left as the first element
      * If >90% of a column's non-null values convert to numeric, the whole
        column is cast to float64
    """
    obj_cols = df.select_dtypes(include=["object"]).columns
    if obj_cols.empty:
        return df

    resolved_count = 0

    for col in obj_cols:
        upper_col = col.upper()
        if any(pat in upper_col for pat in _ID_DATE_PATTERNS):
            continue

        has_pipe = df[col].astype(str).str.contains(r"\|", na=False)
        if not has_pipe.any():
            continue

        def _average_pipe(val):
            if isinstance(val, str) and "|" in val:
                try:
                    return np.mean([float(x) for x in val.split("|")])
                except (ValueError, TypeError):
                    return val.split("|")[0]
            return val

        converted = df[col].apply(_average_pipe)
        numeric = pd.to_numeric(converted, errors="coerce")
        n_nonnull = df[col].notna().sum()
        n_numeric = numeric.notna().sum()

        if n_nonnull > 0 and (n_numeric / n_nonnull) > 0.90:
            df[col] = numeric
            resolved_count += 1
        else:
            df[col] = converted
            resolved_count += 1

    if resolved_count > 0:
        logger.info(
            "%s: resolved pipe-separated values in %d column(s)",
            modality or "unknown", resolved_count,
        )
    return df


async def _load_data_task(
    task_id: str,
    data_path: str,
    modalities: List[str],
    merge_output: bool,
    clean_data: bool
):
    """Background task for loading data.

    When merge_output is True (the normal case from the frontend), we use
    modular storage: each modality is loaded via its specific loader, persisted
    to its own Parquet file immediately, and freed from RAM.  The mega-merged
    DataFrame is never created.
    """
    import time
    loop = asyncio.get_event_loop()
    task_start = time.time()

    try:
        _update_task(task_id, status="running", message="Initializing data loader...", progress=0.05)
        await asyncio.sleep(0.1)

        # Scan the data directory and report what we find
        _update_task(task_id, message=f"Scanning data directory: {data_path}", progress=0.06)
        await asyncio.sleep(0.1)

        total_files = 0
        total_size = 0
        for modality in modalities:
            display_name = _MODALITY_DISPLAY_NAMES.get(modality, modality)
            folder = _MODALITY_FOLDERS.get(modality, modality)
            folder_path = os.path.join(data_path, folder)
            if os.path.exists(folder_path):
                csv_files, size = _count_csv_files(folder_path)
                file_names = [f.name for f in csv_files]
                total_files += len(csv_files)
                total_size += size
                _update_task(
                    task_id,
                    message=f"  Found {display_name}: {len(csv_files)} CSV files ({_format_size(size)}) in {folder}/",
                    progress=0.07
                )
                await asyncio.sleep(0.05)
                # Log individual file names for verbose output
                for fname in file_names:
                    _update_task(task_id, message=f"    - {fname}")
                    await asyncio.sleep(0.01)
            else:
                _update_task(
                    task_id,
                    message=f"  {display_name}: folder '{folder}/' not found — skipping"
                )
                await asyncio.sleep(0.05)

        _update_task(
            task_id,
            message=f"Directory scan complete: {total_files} total CSV files ({_format_size(total_size)}) across {len(modalities)} modalities",
            progress=0.08
        )
        await asyncio.sleep(0.1)

        # Import PIE-clean loaders
        try:
            _update_task(task_id, message="Importing PIE-clean library...", progress=0.09)
            await asyncio.sleep(0.05)

            from pie_clean.sub_char_loader import load_ppmi_subject_characteristics
            from pie_clean.med_hist_loader import load_ppmi_medical_history
            from pie_clean.motor_loader import load_ppmi_motor_assessments
            from pie_clean.non_motor_loader import load_ppmi_non_motor_assessments
            from pie_clean.biospecimen_loader import load_biospecimen_data
            from pie_clean.data_preprocessor import DataPreprocessor
            from pie_clean.constants import FOLDER_PATHS

            _update_task(task_id, message="PIE-clean library loaded successfully", progress=0.10)
            await asyncio.sleep(0.1)

            # Capture PIE-clean library log output into the task log
            log_handler = _TaskLogHandler(task_id)
            log_handler.setLevel(logging.INFO)
            pie_root_logger = logging.getLogger("PIE")
            pie_root_logger.addHandler(log_handler)

            # Free previously cached data and completed tasks to prevent
            # unbounded memory growth when the user loads data multiple times.
            disk_cache.clear()
            _cleanup_completed_tasks()

            # Log memory baseline
            proc = psutil.Process()
            mem_before = proc.memory_info().rss
            total_mem = psutil.virtual_memory().total
            _update_task(
                task_id,
                message=f"Memory baseline: {_format_size(mem_before)} used / {_format_size(total_mem)} total"
            )

            # Always include subject_characteristics for PPMI — it contains
            # essential metadata (COHORT, demographics) needed for classification.
            if "subject_characteristics" not in modalities:
                sc_folder = os.path.join(data_path, _MODALITY_FOLDERS["subject_characteristics"])
                if os.path.exists(sc_folder):
                    modalities = ["subject_characteristics"] + list(modalities)
                    _update_task(task_id, message="Auto-including Subject Characteristics (contains COHORT & demographics)")

            total_modalities = len(modalities)
            cache_key = f"data_{task_id}"

            if merge_output:
                # -------------------------------------------------------
                # MODULAR STORAGE PATH — never builds the mega-merge
                # -------------------------------------------------------
                all_pairs: set = set()
                total_columns = 0

                for idx, modality in enumerate(modalities):
                    mod_start = time.time()
                    display_name = _MODALITY_DISPLAY_NAMES.get(modality, modality)
                    base_progress = 0.10 + (idx / total_modalities) * 0.70
                    _update_task(
                        task_id,
                        progress=base_progress,
                        message=f"[{idx + 1}/{total_modalities}] Loading {display_name}..."
                    )
                    await asyncio.sleep(0.1)

                    folder_path = os.path.join(data_path, FOLDER_PATHS.get(modality, _MODALITY_FOLDERS.get(modality, modality)))

                    try:
                        if modality == "subject_characteristics":
                            df = await loop.run_in_executor(
                                None, load_ppmi_subject_characteristics, folder_path
                            )
                            df = _resolve_pipe_separated_values(df, display_name)
                            _collect_pairs(df, all_pairs)
                            n = disk_cache.store_modality(cache_key, modality, df)
                            total_columns += n
                            elapsed = time.time() - mod_start
                            _update_task(task_id, message=f"[{idx+1}/{total_modalities}] {display_name}: {df.shape[0]:,} rows x {df.shape[1]} cols [{elapsed:.1f}s]")
                            del df; gc.collect()

                        elif modality == "medical_history":
                            med_dict = await loop.run_in_executor(
                                None, load_ppmi_medical_history, folder_path
                            )
                            if clean_data and med_dict:
                                med_dict = DataPreprocessor.clean_medical_history(med_dict)
                            for table_name, tdf in med_dict.items():
                                if not isinstance(tdf, pd.DataFrame) or tdf.empty:
                                    continue
                                tdf = _resolve_pipe_separated_values(tdf, f"med_hist/{table_name}")
                                _collect_pairs(tdf, all_pairs)
                                prefixed = _prefix_columns(tdf, table_name)
                                store_name = f"medical_history__{table_name}"
                                n = disk_cache.store_modality(cache_key, store_name, prefixed)
                                total_columns += n
                                del prefixed
                            elapsed = time.time() - mod_start
                            _update_task(task_id, message=f"[{idx+1}/{total_modalities}] {display_name}: {len(med_dict)} tables [{elapsed:.1f}s]")
                            del med_dict; gc.collect()

                        elif modality == "motor_assessments":
                            df = await loop.run_in_executor(
                                None, load_ppmi_motor_assessments, folder_path
                            )
                            df = _resolve_pipe_separated_values(df, display_name)
                            _collect_pairs(df, all_pairs)
                            n = disk_cache.store_modality(cache_key, modality, df)
                            total_columns += n
                            elapsed = time.time() - mod_start
                            _update_task(task_id, message=f"[{idx+1}/{total_modalities}] {display_name}: {df.shape[0]:,} rows x {df.shape[1]} cols [{elapsed:.1f}s]")
                            del df; gc.collect()

                        elif modality == "non_motor_assessments":
                            df = await loop.run_in_executor(
                                None, load_ppmi_non_motor_assessments, folder_path
                            )
                            df = _resolve_pipe_separated_values(df, display_name)
                            _collect_pairs(df, all_pairs)
                            n = disk_cache.store_modality(cache_key, modality, df)
                            total_columns += n
                            elapsed = time.time() - mod_start
                            _update_task(task_id, message=f"[{idx+1}/{total_modalities}] {display_name}: {df.shape[0]:,} rows x {df.shape[1]} cols [{elapsed:.1f}s]")
                            del df; gc.collect()

                        elif modality == "biospecimen":
                            bio_dict = await loop.run_in_executor(
                                None, lambda: load_biospecimen_data(data_path, "PPMI")
                            )
                            for source_name, sdf in bio_dict.items():
                                if not isinstance(sdf, pd.DataFrame) or sdf.empty:
                                    continue
                                sdf = _resolve_pipe_separated_values(sdf, f"biospecimen/{source_name}")
                                _collect_pairs(sdf, all_pairs)
                                prefixed = _prefix_columns(sdf, source_name)
                                store_name = f"biospecimen__{source_name}"
                                n = disk_cache.store_modality(cache_key, store_name, prefixed)
                                total_columns += n
                                del prefixed
                            elapsed = time.time() - mod_start
                            _update_task(task_id, message=f"[{idx+1}/{total_modalities}] {display_name}: {len(bio_dict)} sub-datasets [{elapsed:.1f}s]")
                            del bio_dict; gc.collect()

                        else:
                            _update_task(task_id, message=f"[{idx+1}/{total_modalities}] Unknown modality '{modality}' — skipping")

                    except Exception as e:
                        elapsed = time.time() - mod_start
                        _update_task(
                            task_id,
                            message=f"[{idx+1}/{total_modalities}] WARNING: Failed to load {display_name} after {elapsed:.1f}s — {e}"
                        )
                    await asyncio.sleep(0.05)

                # Finalize modular cache
                total_rows = len(all_pairs)
                disk_cache.finalize_modular_cache(cache_key, total_rows)
                del all_pairs

                # Log memory after loading
                mem_after = proc.memory_info().rss
                _update_task(
                    task_id,
                    progress=0.90,
                    message=f"Memory after loading: {_format_size(mem_after)} (delta +{_format_size(mem_after - mem_before)})"
                )
                await asyncio.sleep(0.05)

                meta = disk_cache.get_column_meta(cache_key)
                col_names = [c["name"] for c in meta["columns"]] if meta else []

                summary = {
                    "type": "merged_dataframe",
                    "shape": [total_rows, total_columns],
                    "columns": col_names,
                    "cache_key": cache_key,
                }

                total_elapsed = time.time() - task_start
                _update_task(
                    task_id,
                    progress=0.95,
                    message=f"Final dataset: {total_rows:,} rows x {total_columns} columns (modular storage)"
                )
                await asyncio.sleep(0.05)
                _update_task(
                    task_id,
                    message=f"Total processing time: {total_elapsed:.1f}s — memory: {_format_size(mem_after)}"
                )
                await asyncio.sleep(0.1)

            else:
                # -------------------------------------------------------
                # Non-merge path: load each modality individually
                # -------------------------------------------------------
                loaded_data = {}

                for idx, modality in enumerate(modalities):
                    mod_start = time.time()
                    display_name = _MODALITY_DISPLAY_NAMES.get(modality, modality)
                    base_progress = 0.10 + (idx / total_modalities) * 0.50
                    _update_task(
                        task_id,
                        progress=base_progress,
                        message=f"[{idx + 1}/{total_modalities}] Loading {display_name}..."
                    )
                    await asyncio.sleep(0.1)

                    try:
                        from pie_clean import DataLoader
                        mod_data = await loop.run_in_executor(
                            None,
                            lambda m=modality: DataLoader.load(
                                data_path=data_path,
                                modalities=[m],
                                merge_output=False,
                                clean_data=clean_data
                            )
                        )

                        elapsed = time.time() - mod_start

                        if isinstance(mod_data, dict):
                            loaded_data.update(mod_data)
                            for key, df in mod_data.items():
                                if isinstance(df, pd.DataFrame):
                                    _update_task(
                                        task_id,
                                        message=f"[{idx+1}/{total_modalities}] {display_name} loaded: {df.shape[0]:,} rows x {df.shape[1]} columns [{elapsed:.1f}s]"
                                    )
                        else:
                            loaded_data[modality] = mod_data
                            if isinstance(mod_data, pd.DataFrame):
                                _update_task(
                                    task_id,
                                    message=f"[{idx+1}/{total_modalities}] {display_name} loaded: {mod_data.shape[0]:,} rows x {mod_data.shape[1]} columns [{elapsed:.1f}s]"
                                )

                        await asyncio.sleep(0.05)
                    except Exception as e:
                        elapsed = time.time() - mod_start
                        _update_task(
                            task_id,
                            message=f"[{idx+1}/{total_modalities}] WARNING: Failed to load {display_name} after {elapsed:.1f}s — {e}"
                        )
                        await asyncio.sleep(0.1)

                result = loaded_data

                mem_after_load = proc.memory_info().rss
                _update_task(
                    task_id,
                    progress=0.88,
                    message=f"Memory after loading: {_format_size(mem_after_load)} (delta +{_format_size(mem_after_load - mem_before)})"
                )
                await asyncio.sleep(0.05)

                _update_task(task_id, progress=0.90, message="Downcasting dtypes and caching data to disk...")
                await asyncio.sleep(0.1)

                summary = {
                    "type": "dict",
                    "modalities": list(result.keys()) if isinstance(result, dict) else [],
                    "cache_key": cache_key
                }

                disk_cache.store(cache_key, result)
                del result
                gc.collect()

                total_elapsed = time.time() - task_start
                _update_task(
                    task_id,
                    progress=0.95,
                    message=f"Loaded {len(summary.get('modalities', []))} modalities. Total processing time: {total_elapsed:.1f}s"
                )
                await asyncio.sleep(0.1)

            # Done with PIE-clean calls — remove the log capture handler
            pie_root_logger.removeHandler(log_handler)

            _update_task(
                task_id,
                status="completed",
                progress=1.0,
                message="Data loading completed successfully",
                result=summary
            )

        except ImportError as e:
            _update_task(
                task_id,
                status="failed",
                error=f"PIE-clean library not available: {str(e)}",
                message=f"PIE-clean import failed: {str(e)}"
            )

    except Exception as e:
        # Ensure the log handler is cleaned up if it was attached
        try:
            logging.getLogger("PIE").removeHandler(log_handler)
        except NameError:
            pass
        _update_task(
            task_id,
            status="failed",
            error=str(e),
            message=f"Failed: {str(e)}"
        )


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background task, including full log history."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = _tasks[task_id]
    return {
        "status": task.get("status"),
        "progress": task.get("progress", 0.0),
        "message": task.get("message", ""),
        "result": task.get("result"),
        "error": task.get("error"),
        "logs": task.get("logs", []),
        "started_at": task.get("started_at")
    }


@router.get("/columns")
async def get_data_columns(cache_key: str):
    """Get column information from cached data.

    Returns pre-computed metadata so the full DataFrame never needs to be
    reloaded into memory just to answer this request.
    """
    # Try pre-computed metadata first (fast, zero memory)
    meta = disk_cache.get_column_meta(cache_key)
    if meta is not None:
        return meta

    # For modular caches after process restart, reload manifest into memory
    if disk_cache.is_modular(cache_key):
        disk_cache._load_manifest_into_meta(cache_key)
        meta = disk_cache.get_column_meta(cache_key)
        if meta is not None:
            return meta

    # Fallback: load from disk, compute, then free
    if not disk_cache.exists(cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {cache_key}")

    data = disk_cache.load(cache_key)
    if not isinstance(data, pd.DataFrame):
        raise HTTPException(status_code=400, detail="Cached data is not a DataFrame")

    columns = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        is_numeric = np.issubdtype(data[col].dtype, np.number)
        unique_count = data[col].nunique()

        columns.append({
            "name": col,
            "dtype": dtype,
            "is_numeric": is_numeric,
            "is_categorical": not is_numeric and unique_count < 50,
            "unique_count": unique_count,
            "null_count": int(data[col].isnull().sum()),
            "null_pct": round(data[col].isnull().sum() / len(data) * 100, 2) if len(data) > 0 else 0,
            "source_modality": "unknown",
        })

    result = {"columns": columns, "total_rows": len(data)}
    del data
    gc.collect()
    return result


@router.post("/missingness_heatmap")
async def get_missingness_heatmap(cache_key: str, sample_size: int = 100):
    """Generate missingness heatmap data for visualization.

    For modular caches, uses pre-computed column metadata for column_summary
    and samples from a small modality file for the matrix.
    """
    if not disk_cache.exists(cache_key) and not disk_cache.is_modular(cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {cache_key}")

    # Use pre-computed column metadata for the full-dataset summary
    meta = disk_cache.get_column_meta(cache_key)
    if meta is None and disk_cache.is_modular(cache_key):
        disk_cache._load_manifest_into_meta(cache_key)
        meta = disk_cache.get_column_meta(cache_key)

    if disk_cache.is_modular(cache_key):
        # For modular caches, sample from a non-biospecimen modality file
        mod_files = disk_cache.list_modality_files(cache_key)
        # Prefer a small modality for sampling
        sample_file = None
        for preferred in ["subject_characteristics", "motor_assessments", "non_motor_assessments"]:
            if preferred in mod_files:
                sample_file = preferred
                break
        if sample_file is None and mod_files:
            sample_file = mod_files[0]

        if sample_file:
            import pyarrow.parquet as pq
            dir_path = disk_cache._CACHE_DIR / cache_key
            table = pq.read_table(dir_path / f"{sample_file}.parquet").slice(0, sample_size)
            sampled = table.to_pandas()
        else:
            sampled = pd.DataFrame()

        if meta is not None:
            col_missing = [
                {"column": c["name"], "missing_pct": c["null_pct"]}
                for c in meta["columns"]
            ]
            columns = [c["name"] for c in meta["columns"]]
        else:
            columns = list(sampled.columns) if not sampled.empty else []
            col_missing = []

        missing_matrix = sampled.isnull().astype(int).values.tolist() if not sampled.empty else []

        result = {
            "matrix": missing_matrix,
            "columns": columns,
            "row_count": len(sampled),
            "column_summary": col_missing,
        }
        del sampled
        gc.collect()
        return result

    # Legacy single-file path
    try:
        sampled = disk_cache.load_sample(cache_key, n=sample_size)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Data not found: {cache_key}")

    if not isinstance(sampled, pd.DataFrame):
        raise HTTPException(status_code=400, detail="Cached data is not a DataFrame")

    # Create missingness matrix from the sample
    missing_matrix = sampled.isnull().astype(int).values.tolist()

    if meta is not None:
        col_missing = [
            {"column": c["name"], "missing_pct": c["null_pct"]}
            for c in meta["columns"]
        ]
        columns = [c["name"] for c in meta["columns"]]
    else:
        columns = list(sampled.columns)
        col_missing = []
        for col in columns:
            null_pct = sampled[col].isnull().sum() / len(sampled) * 100 if len(sampled) > 0 else 0
            col_missing.append({"column": col, "missing_pct": round(null_pct, 2)})

    result = {
        "matrix": missing_matrix,
        "columns": columns,
        "row_count": len(sampled),
        "column_summary": col_missing,
    }
    del sampled
    gc.collect()
    return result
