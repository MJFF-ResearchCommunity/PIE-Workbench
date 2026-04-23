"""
PIE Workbench Backend - FastAPI Server

This module serves as the main entry point for the PIE Workbench backend.
It provides REST API endpoints for data loading, processing, ML operations, and statistics.
"""

import os
import sys
import resource
import logging
import faulthandler
import signal
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psutil

logger = logging.getLogger(__name__)

# Route logger.info() calls throughout the backend to stderr. Without this,
# background-task progress messages and warnings disappear because Python's
# root logger defaults to WARNING with no visible handler, leaving only
# uvicorn's access log in the terminal.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)

# Dump a Python traceback on fatal signals (SIGSEGV/SIGABRT/SIGFPE/SIGBUS/SIGILL)
# so native-extension crashes and RLIMIT_AS trips leave a trace on stderr instead
# of dying silently. SIGUSR1 prints all thread stacks on demand — use
# `kill -SIGUSR1 <pid>` to diagnose a hang without killing the process.
faulthandler.enable()
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except (AttributeError, ValueError):
    pass

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

# Configure CORS for the local frontend (Vite dev server + packaged Electron).
# The backend has no authentication and is bound to 127.0.0.1, so we scope
# origins tightly rather than using "*". Electron's packaged renderer loads
# via file:// which sends Origin: null.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "null",
    ],
    allow_credentials=False,
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
    uvicorn.run(app, host="127.0.0.1", port=8100)
