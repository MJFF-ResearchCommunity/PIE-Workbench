"""
pytest configuration for the backend.

Adds the `backend/` directory to sys.path so tests under `backend/tests/`
can do `from main import app`, `from api import cache`, etc. without the
caller having to set PYTHONPATH.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
