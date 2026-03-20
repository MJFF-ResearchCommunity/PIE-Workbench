"""
Project management API endpoints.

Handles saving and loading project state, project configuration, etc.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

_REGISTRY_DIR = Path.home() / ".pie_workbench"
_REGISTRY_FILE = _REGISTRY_DIR / "recent_projects.json"


class ProjectConfig(BaseModel):
    """Project configuration model."""
    name: str
    disease_context: str = "parkinsons"  # parkinsons, alzheimers, etc.
    data_path: str
    output_path: Optional[str] = None
    target_column: Optional[str] = None
    leakage_features: List[str] = []
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class DataStateModel(BaseModel):
    """Persisted data pipeline state."""
    loaded: bool = False
    cache_key: Optional[str] = None
    shape: Optional[List[int]] = None
    columns: List[str] = []
    modalities: List[str] = []


class AnalysisStateModel(BaseModel):
    """Persisted analysis pipeline state."""
    engineered_cache_key: Optional[str] = None
    train_cache_key: Optional[str] = None
    test_cache_key: Optional[str] = None
    model_id: Optional[str] = None
    selected_features: List[str] = []
    calibrated_model_id: Optional[str] = None
    ensemble_model_id: Optional[str] = None
    drift_result: Optional[str] = None


class ProjectState(BaseModel):
    """Full project state including processed data info."""
    config: ProjectConfig
    data: DataStateModel = DataStateModel()
    analysis: AnalysisStateModel = AnalysisStateModel()
    # Legacy flags kept for backward compat with existing .pie files
    data_loaded: bool = False
    data_processed: bool = False
    modalities_selected: List[str] = []
    features_engineered: bool = False
    features_selected: bool = False
    model_trained: bool = False
    current_step: str = "project_hub"


# In-memory project store (in production, use proper persistence)
_current_project: Optional[ProjectState] = None


def _auto_save():
    """Auto-save project to <output_path>/project.pie after every mutation."""
    if _current_project is None or not _current_project.config.output_path:
        return
    try:
        save_path = Path(_current_project.config.output_path) / "project.pie"
        os.makedirs(_current_project.config.output_path, exist_ok=True)
        _current_project.config.updated_at = datetime.now().isoformat()
        with open(save_path, 'w') as f:
            json.dump(_current_project.model_dump(), f, indent=2)
        _register_recent_project(str(save_path), _current_project.config.name)
    except Exception:
        pass  # Auto-save is best-effort


def _register_recent_project(file_path: str, name: str) -> None:
    """Track a project in the user-level recent-projects registry."""
    try:
        _REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        entries: list = []
        if _REGISTRY_FILE.exists():
            entries = json.loads(_REGISTRY_FILE.read_text())

        entries = [e for e in entries if e.get("path") != file_path]
        entries.insert(0, {
            "path": file_path,
            "name": name,
            "last_opened": datetime.now().isoformat(),
        })
        entries = entries[:20]  # Keep at most 20

        _REGISTRY_FILE.write_text(json.dumps(entries, indent=2))
    except Exception:
        pass


def _get_recent_projects() -> list:
    """Return the recent-projects list, pruning entries whose files no longer exist."""
    if not _REGISTRY_FILE.exists():
        return []
    try:
        entries = json.loads(_REGISTRY_FILE.read_text())
        valid = [e for e in entries if os.path.isfile(e.get("path", ""))]
        if len(valid) != len(entries):
            _REGISTRY_FILE.write_text(json.dumps(valid, indent=2))
        return valid
    except Exception:
        return []


@router.post("/create")
async def create_project(config: ProjectConfig):
    """Create a new project."""
    global _current_project
    
    # Validate data path exists
    if not os.path.exists(config.data_path):
        raise HTTPException(status_code=400, detail=f"Data path does not exist: {config.data_path}")
    
    # Set timestamps
    config.created_at = datetime.now().isoformat()
    config.updated_at = config.created_at
    
    # Set default output path if not provided
    if not config.output_path:
        config.output_path = str(Path(config.data_path).parent / f"pie_output_{config.name}")
    
    # Create output directory
    os.makedirs(config.output_path, exist_ok=True)
    
    _current_project = ProjectState(config=config)
    _auto_save()

    return {"status": "success", "project": _current_project.model_dump()}


@router.get("/current")
async def get_current_project():
    """Get the current project state."""
    if _current_project is None:
        raise HTTPException(status_code=404, detail="No project loaded")
    return _current_project.model_dump()


@router.post("/save")
async def save_project(file_path: Optional[str] = None):
    """Save the current project to a file. If no path given, auto-saves to output dir."""
    if _current_project is None:
        raise HTTPException(status_code=404, detail="No project to save")

    _current_project.config.updated_at = datetime.now().isoformat()

    if file_path is None:
        if not _current_project.config.output_path:
            raise HTTPException(status_code=400, detail="No output path configured")
        file_path = str(Path(_current_project.config.output_path) / "project.pie")

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(_current_project.model_dump(), f, indent=2)

    return {"status": "success", "path": file_path}


@router.get("/recent")
async def get_recent_projects():
    """Return the list of recently-opened project files."""
    return {"projects": _get_recent_projects()}


@router.post("/load")
async def load_project(file_path: str):
    """Load a project from a file."""
    global _current_project
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Project file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    _current_project = ProjectState(**data)
    _register_recent_project(file_path, _current_project.config.name)
    return {"status": "success", "project": _current_project.model_dump()}


@router.post("/update_state")
async def update_project_state(updates: dict):
    """Update the project state."""
    global _current_project
    
    if _current_project is None:
        raise HTTPException(status_code=404, detail="No project loaded")
    
    for key, value in updates.items():
        if key == "config" and isinstance(value, dict):
            for ck, cv in value.items():
                if hasattr(_current_project.config, ck):
                    setattr(_current_project.config, ck, cv)
        elif key == "data" and isinstance(value, dict):
            for dk, dv in value.items():
                if hasattr(_current_project.data, dk):
                    setattr(_current_project.data, dk, dv)
        elif key == "analysis" and isinstance(value, dict):
            for ak, av in value.items():
                if hasattr(_current_project.analysis, ak):
                    setattr(_current_project.analysis, ak, av)
        elif hasattr(_current_project, key):
            setattr(_current_project, key, value)

    _auto_save()

    return {"status": "success", "project": _current_project.model_dump()}


@router.get("/disease_contexts")
async def get_disease_contexts():
    """Get available disease contexts."""
    return {
        "contexts": [
            {"id": "parkinsons", "name": "Parkinson's Disease (PPMI)", "available": True},
            {"id": "alzheimers", "name": "Alzheimer's Disease", "available": False, "note": "Coming soon"},
        ]
    }
