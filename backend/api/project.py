"""
Project management API endpoints.

Handles saving and loading project state, project configuration, etc.
"""

import os
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


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


class ProjectState(BaseModel):
    """Full project state including processed data info."""
    config: ProjectConfig
    data_loaded: bool = False
    data_processed: bool = False
    modalities_selected: List[str] = []
    features_engineered: bool = False
    features_selected: bool = False
    model_trained: bool = False
    current_step: str = "project_hub"


# In-memory project store (in production, use proper persistence)
_current_project: Optional[ProjectState] = None


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
    
    return {"status": "success", "project": _current_project.model_dump()}


@router.get("/current")
async def get_current_project():
    """Get the current project state."""
    if _current_project is None:
        raise HTTPException(status_code=404, detail="No project loaded")
    return _current_project.model_dump()


@router.post("/save")
async def save_project(file_path: str):
    """Save the current project to a file."""
    if _current_project is None:
        raise HTTPException(status_code=404, detail="No project to save")
    
    _current_project.config.updated_at = datetime.now().isoformat()
    
    with open(file_path, 'w') as f:
        json.dump(_current_project.model_dump(), f, indent=2)
    
    return {"status": "success", "path": file_path}


@router.post("/load")
async def load_project(file_path: str):
    """Load a project from a file."""
    global _current_project
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Project file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    _current_project = ProjectState(**data)
    return {"status": "success", "project": _current_project.model_dump()}


@router.post("/update_state")
async def update_project_state(updates: dict):
    """Update the project state."""
    global _current_project
    
    if _current_project is None:
        raise HTTPException(status_code=404, detail="No project loaded")
    
    for key, value in updates.items():
        if hasattr(_current_project, key):
            setattr(_current_project, key, value)
    
    _current_project.config.updated_at = datetime.now().isoformat()
    
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
