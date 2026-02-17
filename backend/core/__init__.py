"""
Core adapter layer for PIE Workbench.

This module provides abstract interfaces and concrete implementations
for data loading and analysis, enabling modularity and extensibility.
"""

from .abstract_loader import AbstractDataLoader
from .ppmi_loader import PPMIDataLoader

__all__ = ['AbstractDataLoader', 'PPMIDataLoader']
