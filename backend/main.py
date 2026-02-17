"""
PIE Workbench Backend - FastAPI Server

This module serves as the main entry point for the PIE Workbench backend.
It provides REST API endpoints for data loading, processing, ML operations, and statistics.
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add lib directory to path for importing PIE and PIE-clean
lib_path = Path(__file__).parent.parent / "lib"
sys.path.insert(0, str(lib_path / "PIE"))
sys.path.insert(0, str(lib_path / "PIE-clean"))

from api import project, data, analysis, statistics

app = FastAPI(
    title="PIE Workbench API",
    description="Backend API for the Parkinson's Insight Engine Workbench",
    version="1.0.0"
)

# Configure CORS for Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(project.router, prefix="/api/project", tags=["Project"])
app.include_router(data.router, prefix="/api/data", tags=["Data"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(statistics.router, prefix="/api/statistics", tags=["Statistics"])


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {"status": "ok", "message": "PIE Workbench API is running"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
