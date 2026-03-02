"""
PIE Workbench Backend - FastAPI Server

This module serves as the main entry point for the PIE Workbench backend.
It provides REST API endpoints for data loading, processing, ML operations, and statistics.
"""

import os
import sys
import resource
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psutil

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory safety: cap the process address space so that runaway allocations
# raise MemoryError instead of triggering the Linux OOM killer (which can
# crash or reboot the entire machine).
# ---------------------------------------------------------------------------
_MEMORY_LIMIT_FRACTION = 0.75  # use at most 75 % of physical RAM

try:
    _total_ram = psutil.virtual_memory().total
    _mem_limit = int(_total_ram * _MEMORY_LIMIT_FRACTION)
    # RLIMIT_AS limits the total virtual address space (data + stack + mmap)
    resource.setrlimit(resource.RLIMIT_AS, (_mem_limit, _mem_limit))
    logger.info(
        "Memory limit set to %d MB (%.0f%% of %d MB total RAM)",
        _mem_limit // (1024 * 1024),
        _MEMORY_LIMIT_FRACTION * 100,
        _total_ram // (1024 * 1024),
    )
except Exception as exc:
    # Non-fatal: the limit is a safety net, not a hard requirement
    logger.warning("Could not set memory limit: %s", exc)

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
