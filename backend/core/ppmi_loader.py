"""
PPMI-specific data loader implementation.

This module provides the concrete implementation of AbstractDataLoader
for the Parkinson's Progression Markers Initiative (PPMI) dataset.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd

from .abstract_loader import AbstractDataLoader, AbstractAnalysisAdapter


class PPMIDataLoader(AbstractDataLoader):
    """
    Data loader for PPMI (Parkinson's Progression Markers Initiative) dataset.
    
    This adapter wraps the PIE-clean library to provide data loading functionality
    through the abstract interface defined by AbstractDataLoader.
    """
    
    MODALITY_INFO = {
        "subject_characteristics": {
            "name": "Subject Characteristics",
            "description": "Demographics, family history, cohort information, and participant status",
            "folder": "_Subject_Characteristics",
            "expected_files": ["Age_at_visit*.csv", "Family_History*.csv", "Participant_Status*.csv"],
            "key_columns": ["PATNO", "EVENT_ID"]
        },
        "medical_history": {
            "name": "Medical History",
            "description": "Concomitant medications, vital signs, physical exams, and features of parkinsonism",
            "folder": "Medical_History",
            "expected_files": ["Concomitant_Medication*.csv", "Vital_Signs*.csv", "Features_of_Parkinsonism*.csv"],
            "key_columns": ["PATNO", "EVENT_ID"]
        },
        "motor_assessments": {
            "name": "Motor Assessments (MDS-UPDRS)",
            "description": "MDS-UPDRS Parts I-IV scores and motor function evaluations",
            "folder": "Motor___MDS-UPDRS",
            "expected_files": ["MDS-UPDRS_Part_I*.csv", "MDS-UPDRS_Part_II*.csv", "MDS-UPDRS_Part_III*.csv"],
            "key_columns": ["PATNO", "EVENT_ID"]
        },
        "non_motor_assessments": {
            "name": "Non-Motor Assessments",
            "description": "Cognitive assessments (MoCA), quality of life measures, and other non-motor evaluations",
            "folder": "Non-motor_Assessments",
            "expected_files": ["Montreal_Cognitive_Assessment*.csv", "Neuro_QoL*.csv"],
            "key_columns": ["PATNO", "EVENT_ID"]
        },
        "biospecimen": {
            "name": "Biospecimen",
            "description": "Biological samples, lab results, and biomarker data",
            "folder": "Biospecimen",
            "expected_files": ["*.csv"],
            "key_columns": ["PATNO", "EVENT_ID"]
        }
    }
    
    def __init__(self):
        """Initialize the PPMI data loader."""
        self._pie_clean_available = False
        try:
            from pie_clean import DataLoader, ALL_MODALITIES
            self._pie_clean_loader = DataLoader
            self._all_modalities = ALL_MODALITIES
            self._pie_clean_available = True
        except ImportError:
            self._all_modalities = list(self.MODALITY_INFO.keys())
    
    @property
    def source_name(self) -> str:
        return "PPMI"
    
    @property
    def supported_modalities(self) -> List[str]:
        return list(self.MODALITY_INFO.keys())
    
    def detect_modalities(self, path: str) -> List[Dict[str, Any]]:
        """Detect available PPMI modalities in the given path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data path not found: {path}")
        
        detected = []
        for modality_id, info in self.MODALITY_INFO.items():
            folder_path = os.path.join(path, info["folder"])
            
            if os.path.exists(folder_path):
                csv_files = list(Path(folder_path).glob("*.csv"))
                detected.append({
                    "id": modality_id,
                    "name": info["name"],
                    "description": info["description"],
                    "available": True,
                    "file_count": len(csv_files),
                    "folder": info["folder"]
                })
            else:
                detected.append({
                    "id": modality_id,
                    "name": info["name"],
                    "description": info["description"],
                    "available": False,
                    "file_count": 0,
                    "folder": info["folder"]
                })
        
        return detected
    
    def validate_schema(self, path: str) -> Dict[str, Any]:
        """Validate the PPMI data schema."""
        errors = []
        warnings = []
        
        if not os.path.exists(path):
            return {
                "valid": False,
                "errors": [f"Data path does not exist: {path}"],
                "warnings": [],
                "summary": "Validation failed - path not found"
            }
        
        detected = self.detect_modalities(path)
        available_count = sum(1 for m in detected if m["available"])
        
        if available_count == 0:
            errors.append("No PPMI modalities found in the specified path")
        
        for modality in detected:
            if not modality["available"]:
                warnings.append(f"Modality '{modality['name']}' folder not found")
            elif modality["file_count"] == 0:
                warnings.append(f"No CSV files found in '{modality['name']}' folder")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "summary": f"Found {available_count}/{len(detected)} modalities" if len(errors) == 0 else "Validation failed"
        }
    
    def load_data(
        self,
        path: str,
        modalities: Optional[List[str]] = None,
        merge_output: bool = False,
        clean_data: bool = True,
        **kwargs
    ) -> Union[Dict[str, Any], pd.DataFrame]:
        """Load PPMI data using PIE-clean."""
        if not self._pie_clean_available:
            raise ImportError("PIE-clean library is not available. Please install it.")
        
        # Use PIE-clean DataLoader
        return self._pie_clean_loader.load(
            data_path=path,
            modalities=modalities,
            merge_output=merge_output,
            clean_data=clean_data,
            **kwargs
        )
    
    def get_modality_info(self, modality_id: str) -> Dict[str, Any]:
        """Get detailed information about a PPMI modality."""
        if modality_id not in self.MODALITY_INFO:
            raise ValueError(f"Unknown modality: {modality_id}")
        
        info = self.MODALITY_INFO[modality_id]
        return {
            "id": modality_id,
            "name": info["name"],
            "description": info["description"],
            "expected_files": info["expected_files"],
            "key_columns": info["key_columns"],
            "documentation_url": "https://www.ppmi-info.org/access-data-specimens/download-data"
        }


class PIEAnalysisAdapter(AbstractAnalysisAdapter):
    """
    Analysis adapter that wraps the PIE library.
    
    This adapter provides access to PIE's feature engineering,
    feature selection, and classification capabilities.
    """
    
    def __init__(self):
        """Initialize the PIE analysis adapter."""
        self._pie_available = False
        try:
            from pie.feature_engineer import FeatureEngineer
            from pie.feature_selector import FeatureSelector
            from pie.classifier import Classifier
            self._feature_engineer = FeatureEngineer
            self._feature_selector = FeatureSelector
            self._classifier = Classifier
            self._pie_available = True
        except ImportError:
            pass
    
    def engineer_features(
        self,
        data: pd.DataFrame,
        scale_numeric: bool = True,
        one_hot_encode: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """Apply PIE feature engineering."""
        if not self._pie_available:
            raise ImportError("PIE library is not available")
        
        engineer = self._feature_engineer(data.copy())
        
        if one_hot_encode:
            engineer.one_hot_encode(
                auto_identify_threshold=kwargs.get('auto_identify_threshold', 20),
                max_categories_to_encode=kwargs.get('max_categories', 25),
                min_frequency_for_category=kwargs.get('min_frequency', 0.01)
            )
        
        if scale_numeric:
            engineer.scale_numeric_features(
                scaler_type=kwargs.get('scaler_type', 'standard')
            )
        
        return engineer.get_dataframe()
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'fdr',
        **kwargs
    ) -> pd.DataFrame:
        """Perform PIE feature selection."""
        if not self._pie_available:
            raise ImportError("PIE library is not available")
        
        selector = self._feature_selector(
            method=method,
            task_type=kwargs.get('task_type', 'classification'),
            k_or_frac=kwargs.get('k_or_frac', 0.5),
            alpha_fdr=kwargs.get('alpha_fdr', 0.05)
        )
        
        selector.fit(X, y)
        return selector.transform(X)
    
    def train_model(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target_column: str,
        task_type: str = 'classification',
        **kwargs
    ) -> Dict[str, Any]:
        """Train models using PIE Classifier."""
        if not self._pie_available:
            raise ImportError("PIE library is not available")
        
        classifier = self._classifier()
        
        classifier.setup_experiment(
            data=train_data,
            target=target_column,
            test_data=test_data,
            session_id=kwargs.get('session_id', 42),
            verbose=kwargs.get('verbose', False)
        )
        
        best_model = classifier.compare_models(
            n_select=1,
            budget_time=kwargs.get('budget_time_minutes', 30),
            verbose=kwargs.get('verbose', False)
        )
        
        if kwargs.get('tune_best', False):
            best_model = classifier.tune_model(verbose=kwargs.get('verbose', False))
        
        return {
            "model": best_model,
            "model_name": type(best_model).__name__,
            "comparison_results": classifier.comparison_results,
            "classifier": classifier
        }
    
    def generate_report(
        self,
        results: Dict[str, Any],
        output_path: str
    ) -> str:
        """Generate analysis report."""
        classifier = results.get("classifier")
        if classifier:
            return classifier.generate_report(output_path=output_path)
        return output_path
