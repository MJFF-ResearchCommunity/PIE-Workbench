"""
Data ingestion API endpoints.

Handles data loading, preview, validation, and processing using PIE-clean.
"""

import os
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import numpy as np

# Import PIE-clean
try:
    from pie_clean import DataLoader, DataPreprocessor, ALL_MODALITIES
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
_cached_data: Dict[str, Any] = {}


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


async def _load_data_task(
    task_id: str,
    data_path: str,
    modalities: List[str],
    merge_output: bool,
    clean_data: bool
):
    """Background task for loading data."""
    try:
        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["message"] = "Initializing data loader..."
        _tasks[task_id]["progress"] = 0.05
        await asyncio.sleep(0.1)  # Allow status to be polled
        
        # Import and use PIE-clean DataLoader
        try:
            from pie_clean import DataLoader
            
            _tasks[task_id]["progress"] = 0.1
            _tasks[task_id]["message"] = "PIE-clean library loaded successfully"
            await asyncio.sleep(0.1)
            
            total_modalities = len(modalities)
            loaded_data = {}
            
            # Load each modality with progress updates
            for idx, modality in enumerate(modalities):
                progress = 0.1 + (idx / total_modalities) * 0.5
                _tasks[task_id]["progress"] = progress
                _tasks[task_id]["message"] = f"Loading modality: {modality} ({idx + 1}/{total_modalities})"
                await asyncio.sleep(0.1)  # Allow status to be polled
                
                try:
                    # Load single modality
                    mod_data = DataLoader.load(
                        data_path=data_path,
                        modalities=[modality],
                        merge_output=False,
                        clean_data=clean_data
                    )
                    if isinstance(mod_data, dict):
                        loaded_data.update(mod_data)
                    else:
                        loaded_data[modality] = mod_data
                    
                    _tasks[task_id]["message"] = f"Loaded {modality} successfully"
                    await asyncio.sleep(0.05)
                except Exception as e:
                    _tasks[task_id]["message"] = f"Warning: Could not load {modality}: {str(e)[:50]}"
                    await asyncio.sleep(0.1)
            
            _tasks[task_id]["progress"] = 0.65
            _tasks[task_id]["message"] = "All modalities loaded. Starting merge process..."
            await asyncio.sleep(0.1)
            
            # If merge requested, do the full load with merge
            if merge_output:
                _tasks[task_id]["progress"] = 0.7
                _tasks[task_id]["message"] = "Merging data across modalities..."
                await asyncio.sleep(0.1)
                
                result = DataLoader.load(
                    data_path=data_path,
                    modalities=modalities,
                    merge_output=True,
                    clean_data=clean_data
                )
                
                _tasks[task_id]["progress"] = 0.85
                _tasks[task_id]["message"] = "Merge complete. Calculating statistics..."
                await asyncio.sleep(0.1)
            else:
                result = loaded_data
            
            _tasks[task_id]["progress"] = 0.9
            _tasks[task_id]["message"] = "Caching data and generating summary..."
            await asyncio.sleep(0.1)
            
            # Store the loaded data
            cache_key = f"data_{task_id}"
            _cached_data[cache_key] = result
            
            # Generate summary
            if isinstance(result, pd.DataFrame):
                null_total = result.isnull().sum().sum()
                total_cells = result.shape[0] * result.shape[1]
                null_pct = round(null_total / total_cells * 100, 2) if total_cells > 0 else 0
                
                summary = {
                    "type": "merged_dataframe",
                    "shape": list(result.shape),
                    "columns": list(result.columns),
                    "null_pct": null_pct,
                    "cache_key": cache_key
                }
                
                _tasks[task_id]["progress"] = 0.95
                _tasks[task_id]["message"] = f"Loaded {result.shape[0]:,} rows × {result.shape[1]} columns ({null_pct}% missing)"
                await asyncio.sleep(0.1)
            else:
                summary = {
                    "type": "dict",
                    "modalities": list(result.keys()),
                    "cache_key": cache_key
                }
            
            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["progress"] = 1.0
            _tasks[task_id]["message"] = "Data loading completed successfully"
            _tasks[task_id]["result"] = summary
            
        except ImportError as e:
            # Fallback if PIE-clean not available
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = f"PIE-clean library not available: {str(e)}"
            
    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)
        _tasks[task_id]["message"] = f"Failed: {str(e)}"


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    return _tasks[task_id]


@router.get("/columns")
async def get_data_columns(cache_key: str):
    """Get column information from cached data."""
    if cache_key not in _cached_data:
        raise HTTPException(status_code=404, detail=f"Data not found: {cache_key}")
    
    data = _cached_data[cache_key]
    
    if isinstance(data, pd.DataFrame):
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
                "null_pct": round(data[col].isnull().sum() / len(data) * 100, 2) if len(data) > 0 else 0
            })
        
        return {"columns": columns, "total_rows": len(data)}
    
    raise HTTPException(status_code=400, detail="Cached data is not a DataFrame")


@router.post("/missingness_heatmap")
async def get_missingness_heatmap(cache_key: str, sample_size: int = 100):
    """Generate missingness heatmap data for visualization."""
    if cache_key not in _cached_data:
        raise HTTPException(status_code=404, detail=f"Data not found: {cache_key}")
    
    data = _cached_data[cache_key]
    
    if not isinstance(data, pd.DataFrame):
        raise HTTPException(status_code=400, detail="Cached data is not a DataFrame")
    
    # Sample rows for visualization
    if len(data) > sample_size:
        sampled = data.sample(n=sample_size, random_state=42)
    else:
        sampled = data
    
    # Create missingness matrix
    missing_matrix = sampled.isnull().astype(int).values.tolist()
    
    # Column-wise missingness summary
    col_missing = []
    for col in data.columns:
        null_pct = data[col].isnull().sum() / len(data) * 100 if len(data) > 0 else 0
        col_missing.append({
            "column": col,
            "missing_pct": round(null_pct, 2)
        })
    
    return {
        "matrix": missing_matrix,
        "columns": list(data.columns),
        "row_count": len(sampled),
        "column_summary": col_missing
    }
