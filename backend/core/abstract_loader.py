"""
Abstract data loader interface.

This module defines the abstract base class for data loaders,
enabling support for different data sources (PPMI, Alzheimer's datasets, etc.)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
import pandas as pd


class AbstractDataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    This interface defines the contract that all data source adapters must implement.
    This enables the GUI to work with different datasets without hardcoding PPMI-specific logic.
    """
    
    @abstractmethod
    def detect_modalities(self, path: str) -> List[Dict[str, Any]]:
        """
        Detect available data modalities in the given path.
        
        Args:
            path: Path to the data directory
            
        Returns:
            List of dictionaries with modality information:
            [
                {
                    "id": "modality_name",
                    "name": "Human-readable name",
                    "available": True/False,
                    "file_count": int,
                    "description": "Description of the modality"
                },
                ...
            ]
        """
        pass
    
    @abstractmethod
    def validate_schema(self, path: str) -> Dict[str, Any]:
        """
        Validate the data schema in the given path.
        
        Args:
            path: Path to the data directory
            
        Returns:
            Dictionary with validation results:
            {
                "valid": True/False,
                "errors": ["list of error messages"],
                "warnings": ["list of warning messages"],
                "summary": "Overall validation summary"
            }
        """
        pass
    
    @abstractmethod
    def load_data(
        self,
        path: str,
        modalities: Optional[List[str]] = None,
        merge_output: bool = False,
        clean_data: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """
        Load and optionally clean/merge data from the specified modalities.
        
        Args:
            path: Path to the data directory
            modalities: List of modality IDs to load (None = all)
            merge_output: If True, return merged DataFrame; if False, return dict
            clean_data: If True, apply data cleaning functions
            **kwargs: Additional loader-specific arguments
            
        Returns:
            Either a dictionary of DataFrames (one per modality) or a single merged DataFrame
        """
        pass
    
    @abstractmethod
    def get_modality_info(self, modality_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific modality.
        
        Args:
            modality_id: The ID of the modality
            
        Returns:
            Dictionary with modality information:
            {
                "id": "modality_id",
                "name": "Human-readable name",
                "description": "Detailed description",
                "expected_files": ["list of expected file patterns"],
                "key_columns": ["PATNO", "EVENT_ID", ...],
                "documentation_url": "optional URL to documentation"
            }
        """
        pass
    
    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of the data source (e.g., 'PPMI', 'ADNI')."""
        pass
    
    @property
    @abstractmethod
    def supported_modalities(self) -> List[str]:
        """Return list of supported modality IDs."""
        pass


class AbstractAnalysisAdapter(ABC):
    """
    Abstract base class for analysis adapters.
    
    This interface defines the contract for ML/Analysis operations.
    """
    
    @abstractmethod
    def engineer_features(
        self,
        data: pd.DataFrame,
        scale_numeric: bool = True,
        one_hot_encode: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Apply feature engineering to the data."""
        pass
    
    @abstractmethod
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'fdr',
        **kwargs
    ) -> pd.DataFrame:
        """Perform feature selection."""
        pass
    
    @abstractmethod
    def train_model(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str,
        task_type: str = 'classification',
        **kwargs
    ) -> Dict[str, Any]:
        """Train and evaluate models."""
        pass
    
    @abstractmethod
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> str:
        """Generate an analysis report."""
        pass
