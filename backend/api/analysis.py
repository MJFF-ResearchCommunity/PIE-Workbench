"""
ML Analysis API endpoints.

Handles feature engineering, selection, and model training using PIE + endgame.

The analysis pipeline mirrors PIE's ``pipeline.py``:

  1. Data Reduction   — DataReducer (analyze → drop → merge → consolidate COHORT)
  2. Feature Engineer — FeatureEngineer (OHE + scale)
  3. Feature Select   — pipe-value resolution, drop non-numeric, split, FeatureSelector
  4. Classification   — Classifier (compare → tune → predict)
"""

import asyncio
import gc
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import numpy as np

from . import cache as disk_cache

logger = logging.getLogger(__name__)

router = APIRouter()

# Task and data stores
_tasks: Dict[str, Dict[str, Any]] = {}
_models: Dict[str, Any] = {}

_MAX_TASK_LOGS = 2000  # Cap log entries per task


def _update_task(task_id: str, **kwargs):
    """Helper to update task state and append to log history."""
    task = _tasks[task_id]
    for key, value in kwargs.items():
        task[key] = value
    if "message" in kwargs and kwargs["message"]:
        if "logs" not in task:
            task["logs"] = []
        if len(task["logs"]) < _MAX_TASK_LOGS:
            task["logs"].append({
                "timestamp": datetime.now().isoformat(),
                "message": kwargs["message"],
            })


class FeatureEngineeringRequest(BaseModel):
    """Request for feature engineering."""
    cache_key: str
    target_column: Optional[str] = None
    scale_numeric: bool = True
    one_hot_encode: bool = True
    max_categories: int = 25
    min_frequency: float = 0.01
    modalities: Optional[List[str]] = None


class FeatureSelectionRequest(BaseModel):
    """Request for feature selection."""
    cache_key: str
    target_column: str
    method: str = "fdr"  # fdr, k_best, select_from_model, rfe, boruta, shap, mrmr, etc.
    param_value: float = 0.05  # alpha for fdr, fraction for k_best
    leakage_features: List[str] = []
    test_size: float = 0.2


class TrainModelRequest(BaseModel):
    """Request for model training."""
    train_cache_key: str
    test_cache_key: str
    target_column: str
    task_type: str = "classification"  # classification, regression
    models_to_compare: List[str] = []
    n_models: int = 5
    tune_best: bool = False
    time_budget_minutes: float = 30.0


class AutoMLRequest(BaseModel):
    """Request for endgame AutoML."""
    train_cache_key: str
    test_cache_key: str
    target_column: str
    time_limit: int = 3600
    presets: str = "good_quality"


class CalibrateRequest(BaseModel):
    """Request for model calibration."""
    model_id: str
    method: str = "conformal"  # conformal, temperature_scaling, venn_abers, platt, isotonic


class DriftValidationRequest(BaseModel):
    """Request for adversarial drift validation."""
    train_cache_key: str
    test_cache_key: str


class DetectLeakageRequest(BaseModel):
    """Request for leakage detection scan."""
    cache_key: str
    target_column: str


class EnsembleRequest(BaseModel):
    """Request for creating an ensemble model."""
    model_id: str
    method: str = "super_learner"  # super_learner, bma, blending, bagging, boosting


class PipelineRequest(BaseModel):
    """Request for running the full pipeline."""
    data_path: str
    output_dir: str
    target_column: str
    modalities: List[str] = []
    leakage_features_path: Optional[str] = None
    fs_method: str = "fdr"
    fs_param: float = 0.05
    n_models: int = 5
    tune_best: bool = False
    generate_plots: bool = True
    budget_minutes: float = 30.0


@router.get("/task_types")
async def get_task_types():
    """Get available ML task types."""
    return {
        "task_types": [
            {"id": "classification", "name": "Classification", "description": "Predict categorical outcomes"},
            {"id": "regression", "name": "Regression", "description": "Predict continuous values"},
        ]
    }


@router.get("/feature_selection_methods")
async def get_feature_selection_methods():
    """Get available feature selection methods (sklearn + endgame)."""
    methods = [
        # sklearn methods (always available)
        {"id": "fdr", "name": "False Discovery Rate (FDR)", "description": "Select features with statistical significance", "requires_endgame": False},
        {"id": "k_best", "name": "K-Best", "description": "Select top K features by score", "requires_endgame": False},
        {"id": "select_from_model", "name": "Model-Based", "description": "Use a model to rank features", "requires_endgame": False},
        {"id": "rfe", "name": "Recursive Feature Elimination", "description": "Iteratively remove least important features", "requires_endgame": False},
        # endgame methods
        {"id": "boruta", "name": "Boruta", "description": "All-relevant feature selection using shadow features", "requires_endgame": True},
        {"id": "shap", "name": "SHAP Selection", "description": "Select features based on SHAP importance", "requires_endgame": True},
        {"id": "mrmr", "name": "mRMR", "description": "Minimum Redundancy Maximum Relevance", "requires_endgame": True},
        {"id": "relief", "name": "ReliefF", "description": "Instance-based feature weighting", "requires_endgame": True},
        {"id": "adversarial", "name": "Adversarial", "description": "Remove features that distinguish train from test", "requires_endgame": True},
        {"id": "permutation", "name": "Permutation Importance", "description": "Select by permutation importance scores", "requires_endgame": True},
        {"id": "genetic", "name": "Genetic Algorithm", "description": "Evolutionary feature subset selection", "requires_endgame": True},
        {"id": "stability", "name": "Stability Selection", "description": "Select features stable across subsamples", "requires_endgame": True},
        {"id": "knockoff", "name": "Knockoff Filter", "description": "FDR-controlled feature selection via knockoffs", "requires_endgame": True},
        {"id": "null_importance", "name": "Null Importance", "description": "Compare feature importance to shuffled-target baseline", "requires_endgame": True},
        {"id": "tree_importance", "name": "Tree Importance", "description": "Select features using tree-based importance", "requires_endgame": True},
        {"id": "correlation", "name": "Correlation Filter", "description": "Remove highly correlated features", "requires_endgame": True},
    ]
    return {"methods": methods}


@router.get("/available_models")
async def get_available_models(task_type: str = "classification"):
    """Get available ML models from endgame's dynamic catalog."""
    try:
        from pie.classifier import get_model_catalog
        catalog = get_model_catalog(task_type)
        models = [
            {"id": k, "name": v.get("name", k) if isinstance(v, dict) else getattr(v, "name", k)}
            for k, v in catalog.items()
        ]
        return {"models": models}
    except ImportError:
        # Fallback static catalog
        if task_type == "classification":
            return {
                "models": [
                    {"id": "lr", "name": "Logistic Regression"},
                    {"id": "rf", "name": "Random Forest"},
                    {"id": "xgboost", "name": "XGBoost"},
                    {"id": "catboost", "name": "CatBoost"},
                    {"id": "lightgbm", "name": "LightGBM"},
                    {"id": "svm", "name": "Support Vector Machine"},
                    {"id": "knn", "name": "K-Nearest Neighbors"},
                    {"id": "dt", "name": "Decision Tree"},
                    {"id": "nb", "name": "Naive Bayes"},
                    {"id": "et", "name": "Extra Trees"},
                ]
            }
        else:
            return {
                "models": [
                    {"id": "lr", "name": "Linear Regression"},
                    {"id": "rf", "name": "Random Forest"},
                    {"id": "xgboost", "name": "XGBoost"},
                    {"id": "catboost", "name": "CatBoost"},
                    {"id": "lightgbm", "name": "LightGBM"},
                    {"id": "svm", "name": "Support Vector Regression"},
                    {"id": "knn", "name": "K-Nearest Neighbors"},
                    {"id": "dt", "name": "Decision Tree"},
                    {"id": "ridge", "name": "Ridge Regression"},
                    {"id": "lasso", "name": "Lasso Regression"},
                ]
            }


@router.post("/suggest_task_type")
async def suggest_task_type(cache_key: str, target_column: str):
    """Suggest classification vs regression based on target column."""
    if not disk_cache.exists(cache_key) and not disk_cache.is_modular(cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {cache_key}")

    # Try column metadata first (fast, always available for modular caches)
    col_meta = disk_cache.get_column_meta(cache_key)
    if col_meta is None and disk_cache.is_modular(cache_key):
        try:
            disk_cache._load_manifest_into_meta(cache_key)
            col_meta = disk_cache.get_column_meta(cache_key)
        except Exception:
            pass

    if col_meta is not None:
        col_info = next((c for c in col_meta["columns"] if c["name"] == target_column), None)
        if col_info is not None:
            is_numeric = col_info.get("is_numeric", False)
            unique_count = col_info.get("unique_count", 0)

            if is_numeric and unique_count > 20:
                suggestion = "regression"
                confidence = 0.85
            elif unique_count <= 10:
                suggestion = "classification"
                confidence = 0.95
            else:
                suggestion = "classification"
                confidence = 0.6

            return {
                "suggestion": suggestion,
                "confidence": confidence,
                "target_info": {
                    "dtype": col_info.get("dtype", "unknown"),
                    "unique_values": unique_count,
                    "is_numeric": is_numeric,
                }
            }

    # Fallback: load actual data
    if disk_cache.is_modular(cache_key):
        try:
            data = disk_cache.load_columns(cache_key, [target_column])
        except (KeyError, Exception):
            raise HTTPException(status_code=400, detail=f"Column not found: {target_column}")
    else:
        data = disk_cache.load(cache_key)
        if not isinstance(data, pd.DataFrame):
            raise HTTPException(status_code=400, detail="Cached data is not a DataFrame")

    if target_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Column not found: {target_column}")

    col = data[target_column]
    is_numeric = np.issubdtype(col.dtype, np.number)
    unique_count = col.nunique()

    if is_numeric and unique_count > 20:
        suggestion = "regression"
        confidence = 0.85
    elif unique_count <= 10:
        suggestion = "classification"
        confidence = 0.95
    else:
        suggestion = "classification"
        confidence = 0.6

    return {
        "suggestion": suggestion,
        "confidence": confidence,
        "target_info": {
            "dtype": str(col.dtype),
            "unique_values": unique_count,
            "is_numeric": is_numeric
        }
    }


@router.post("/feature_engineering")
async def start_feature_engineering(request: FeatureEngineeringRequest, background_tasks: BackgroundTasks):
    """Start feature engineering process."""
    if not disk_cache.exists(request.cache_key) and not disk_cache.is_modular(request.cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {request.cache_key}")

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Initializing feature engineering...",
        "result": None,
        "error": None
    }

    background_tasks.add_task(
        _feature_engineering_task,
        task_id,
        request
    )

    return {"task_id": task_id, "status": "started"}


async def _feature_engineering_task(task_id: str, request: FeatureEngineeringRequest):
    """Background task for feature engineering.

    Follows PIE's ``pipeline.py`` steps 1-2:
      1. Load raw data → DataReducer (analyze → drop → merge → consolidate COHORT)
      2. FeatureEngineer (OHE + scale)

    DataReducer is the critical step: it drops hundreds of low-signal columns
    (>95% missing, metadata, single-value, near-zero-variance) **before** merging,
    keeping memory manageable.
    """
    loop = asyncio.get_event_loop()

    try:
        _update_task(task_id, status="running", progress=0.05,
                     message="Starting data reduction & feature engineering (PIE pipeline)...")

        # Determine the data path from the active project
        from .project import _current_project
        data_path = _current_project.config.data_path if _current_project else None
        if not data_path or not os.path.isdir(data_path):
            raise ValueError(
                "No project loaded or data path is invalid. "
                "Create/open a project first."
            )

        # ---------------------------------------------------------------
        # Step 1a: Load raw data via PIE-clean (same as PIE's pipeline)
        # ---------------------------------------------------------------
        _update_task(task_id, progress=0.08,
                     message="Loading raw PPMI data via PIE-clean DataLoader...")

        from pie_clean import DataLoader, ALL_MODALITIES

        modalities = request.modalities or [
            "subject_characteristics", "medical_history",
            "motor_assessments", "non_motor_assessments",
        ]

        def _load():
            return DataLoader.load(
                data_path=data_path,
                merge_output=False,
                modalities=modalities,
                clean_data=True,
            )

        data_dict = await loop.run_in_executor(None, _load)

        table_count = sum(
            len(v) if isinstance(v, dict) else 1
            for v in data_dict.values()
            if isinstance(v, (pd.DataFrame, dict))
        )
        _update_task(task_id, progress=0.25,
                     message=f"Loaded {table_count} tables across {len(modalities)} modalities")

        # ---------------------------------------------------------------
        # Step 1b: DataReducer — analyze, drop, merge, consolidate COHORT
        # ---------------------------------------------------------------
        _update_task(task_id, message="Running DataReducer (analyzing columns)...")

        from pie.data_reducer import DataReducer

        reducer = DataReducer(data_dict)
        analysis = await loop.run_in_executor(None, reducer.analyze)
        drops = reducer.get_drop_suggestions(analysis)

        total_drops = sum(len(v) for v in drops.values())
        _update_task(task_id, progress=0.35,
                     message=f"DataReducer: dropping {total_drops} low-signal columns")

        reduced_dict = reducer.apply_drops(drops)
        del data_dict
        gc.collect()

        _update_task(task_id, progress=0.40, message="Merging reduced data...")

        merged_df = await loop.run_in_executor(
            None, lambda: reducer.merge_reduced_data(reduced_dict, output_filename=None)
        )
        del reduced_dict
        gc.collect()

        if merged_df.empty:
            raise ValueError("DataReducer produced an empty DataFrame after merge — check your data path")

        _update_task(task_id, progress=0.45,
                     message=f"Merged: {merged_df.shape[0]:,} rows x {merged_df.shape[1]} cols")

        # Consolidate COHORT columns (PIE pipeline step)
        target = request.target_column or "COHORT"
        cohort_cols = [c for c in merged_df.columns if "COHORT" in c.upper()]
        if cohort_cols:
            _update_task(task_id, message=f"Consolidating {len(cohort_cols)} COHORT columns...")
            merged_df = reducer.consolidate_cohort_columns(merged_df, target_cohort_col_name=target)
            _update_task(task_id, progress=0.50,
                         message=f"After COHORT consolidation: {merged_df.shape[0]:,} rows x {merged_df.shape[1]} cols")

        if target and target not in merged_df.columns:
            raise ValueError(
                f"Target column '{target}' not found after reduction & COHORT consolidation. "
                f"Available COHORT-like columns were: {cohort_cols}"
            )

        # ---------------------------------------------------------------
        # Step 2: Feature Engineering
        # ---------------------------------------------------------------
        original_shape = list(merged_df.shape)

        try:
            from pie.feature_engineer import FeatureEngineer

            _update_task(task_id, progress=0.55, message="Applying FeatureEngineer...")

            engineer = FeatureEngineer(merged_df)

            if request.one_hot_encode:
                _update_task(task_id, message="One-hot encoding categorical features...")
                ignore_cols = [target] if target else []
                engineer.one_hot_encode(
                    auto_identify_threshold=20,
                    max_categories_to_encode=request.max_categories,
                    min_frequency_for_category=request.min_frequency,
                    ignore_for_ohe=ignore_cols,
                )

            if request.scale_numeric:
                _update_task(task_id, progress=0.75, message="Scaling numeric features...")
                engineer.scale_numeric_features(scaler_type="standard")

            engineered_df = engineer.get_dataframe()
            summary = engineer.get_engineered_feature_summary()

        except ImportError:
            _update_task(task_id, message="PIE FeatureEngineer not available — basic OHE fallback...")
            engineered_df = merged_df
            cat_cols = engineered_df.select_dtypes(include=["object", "category"]).columns
            for col in cat_cols:
                if col == target:
                    continue
                if engineered_df[col].nunique() <= request.max_categories:
                    dummies = pd.get_dummies(engineered_df[col], prefix=col)
                    engineered_df = pd.concat([engineered_df.drop(col, axis=1), dummies], axis=1)
            summary = {}

        # ---------------------------------------------------------------
        # Cache result
        # ---------------------------------------------------------------
        new_cache_key = f"engineered_{task_id}"
        new_shape = list(engineered_df.shape)
        _update_task(task_id, progress=0.90,
                     message=f"Caching engineered data ({new_shape[0]:,} rows x {new_shape[1]} cols)...")
        disk_cache.store(new_cache_key, engineered_df)
        del engineered_df, merged_df
        gc.collect()

        _update_task(
            task_id,
            status="completed", progress=1.0,
            message="Feature engineering completed",
            result={
                "cache_key": new_cache_key,
                "original_shape": original_shape,
                "new_shape": new_shape,
                "summary": summary,
            },
        )

    except Exception as e:
        _update_task(task_id, status="failed", error=str(e) or repr(e),
                     message=f"Failed: {e!r}")


@router.post("/feature_selection")
async def start_feature_selection(request: FeatureSelectionRequest, background_tasks: BackgroundTasks):
    """Start feature selection process."""
    if not disk_cache.exists(request.cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {request.cache_key}")

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Initializing feature selection...",
        "result": None,
        "error": None
    }

    background_tasks.add_task(
        _feature_selection_task,
        task_id,
        request
    )

    return {"task_id": task_id, "status": "started"}


async def _feature_selection_task(task_id: str, request: FeatureSelectionRequest):
    """Background task for feature selection.

    Mirrors PIE ``pipeline.py`` step 3:
      - Remove leakage features
      - Resolve remaining pipe-separated values (average numeric ones)
      - Drop non-numeric columns
      - fillna(0)
      - train/test split (stratified)
      - FeatureSelector fit/transform
    """
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        _update_task(task_id, status="running", progress=0.1, message="Loading engineered data...")

        data = disk_cache.load(request.cache_key)
        df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
        del data
        gc.collect()

        _update_task(task_id, message=f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

        # Remove leakage features (never drop the target column)
        if request.leakage_features:
            cols_to_drop = [c for c in request.leakage_features if c in df.columns and c != request.target_column]
            df = df.drop(columns=cols_to_drop)
            _update_task(task_id, message=f"Removed {len(cols_to_drop)} leakage features")

        _update_task(task_id, progress=0.2, message="Preparing features and target...")

        if request.target_column not in df.columns:
            ohe_cols = [c for c in df.columns if c.startswith(f"{request.target_column}_")]
            if ohe_cols:
                raise ValueError(
                    f"Target column '{request.target_column}' was one-hot encoded during feature engineering "
                    f"(found: {ohe_cols[:5]}). Re-run feature engineering with target_column set to preserve it."
                )
            raise ValueError(
                f"Target column '{request.target_column}' not found in engineered data. "
                f"Available columns ({len(df.columns)}): {list(df.columns[:20])}"
            )

        df = df.dropna(subset=[request.target_column])

        if "PATNO" in df.columns:
            try:
                df["PATNO"] = df["PATNO"].astype(int)
            except (ValueError, TypeError):
                pass

        id_cols = ["PATNO", "EVENT_ID"]
        feature_cols = [c for c in df.columns if c not in [request.target_column] + id_cols]

        X = df[feature_cols].copy()
        y = df[request.target_column]

        # ----- Resolve remaining pipe-separated values (PIE pipeline.py L270-319) -----
        _ID_DATE_PATS = ["ID", "DATE", "TIME", "PATNO", "EVENT"]
        pipe_resolved = 0
        for col in X.select_dtypes(include=["object"]).columns:
            if any(p in col.upper() for p in _ID_DATE_PATS):
                continue
            if not X[col].astype(str).str.contains(r"\|", na=False).any():
                continue

            def _avg_pipe(val):
                if isinstance(val, str) and "|" in val:
                    try:
                        return np.mean([float(x) for x in val.split("|")])
                    except (ValueError, TypeError):
                        return np.nan
                return val

            converted = X[col].apply(_avg_pipe)
            numeric = pd.to_numeric(converted, errors="coerce")
            n_nonnull = X[col].notna().sum()
            n_numeric = numeric.notna().sum()
            if n_nonnull > 0 and (n_numeric / n_nonnull) > 0.9:
                X[col] = numeric
                pipe_resolved += 1

        if pipe_resolved:
            _update_task(task_id, message=f"Resolved pipe-separated values in {pipe_resolved} columns")

        # Drop remaining non-numeric columns
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric):
            _update_task(task_id, message=f"Dropping {len(non_numeric)} non-numeric columns")
            X = X.drop(columns=non_numeric)

        X = X.fillna(0)

        _update_task(task_id, progress=0.35,
                     message=f"Numeric features: {X.shape[1]}, target classes: {y.nunique()}")

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        _update_task(task_id, progress=0.4, message="Splitting train/test...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=42, stratify=y_encoded
        )

        _update_task(task_id, message=f"Train: {X_train.shape[0]:,} rows, Test: {X_test.shape[0]:,} rows")

        if request.method == "none":
            _update_task(task_id, message="Skipping feature selection — keeping all features")
            X_train_selected = X_train
            X_test_selected = X_test
            selected_features = list(X_train.columns)
        else:
            try:
                from pie.feature_selector import FeatureSelector

                _update_task(task_id, progress=0.55,
                             message=f"Applying {request.method} feature selection...")

                selector = FeatureSelector(
                    method=request.method,
                    task_type="classification",
                    k_or_frac=request.param_value if request.method == "k_best" else None,
                    alpha_fdr=request.param_value if request.method == "fdr" else 0.05,
                )

                y_train_enc = le.fit_transform(y_train)
                selector.fit(X_train, y_train_enc)
                X_train_selected = selector.transform(X_train)
                X_test_selected = selector.transform(X_test)
                selected_features = selector.selected_feature_names_

            except ImportError:
                from sklearn.feature_selection import VarianceThreshold
                _update_task(task_id, message="PIE FeatureSelector not available, using VarianceThreshold fallback...")

                vt = VarianceThreshold(threshold=0.01)
                X_train_selected = pd.DataFrame(
                    vt.fit_transform(X_train), columns=X_train.columns[vt.get_support()]
                )
                X_test_selected = pd.DataFrame(
                    vt.transform(X_test), columns=X_train.columns[vt.get_support()]
                )
                selected_features = list(X_train_selected.columns)

        _update_task(task_id, progress=0.8,
                     message=f"Selected {len(selected_features)} features from {X.shape[1]}")

        train_df = pd.concat([X_train_selected.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([X_test_selected.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        train_cache_key = f"train_{task_id}"
        test_cache_key = f"test_{task_id}"
        _update_task(task_id, message="Caching train/test splits to disk...")
        disk_cache.store(train_cache_key, train_df)
        disk_cache.store(test_cache_key, test_df)

        train_shape = list(train_df.shape)
        test_shape = list(test_df.shape)
        del train_df, test_df, X, df
        gc.collect()

        _update_task(
            task_id,
            status="completed", progress=1.0,
            message="Feature selection completed",
            result={
                "train_cache_key": train_cache_key,
                "test_cache_key": test_cache_key,
                "original_features": len(feature_cols),
                "selected_features": len(selected_features),
                "selected_feature_names": selected_features[:50],
                "train_shape": train_shape,
                "test_shape": test_shape,
            },
        )

    except Exception as e:
        _update_task(task_id, status="failed", error=str(e) or repr(e),
                     message=f"Failed: {e!r}")


@router.post("/train")
async def start_model_training(request: TrainModelRequest, background_tasks: BackgroundTasks):
    """Start model training process."""
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Initializing model training...",
        "result": None,
        "error": None
    }

    background_tasks.add_task(
        _train_model_task,
        task_id,
        request
    )

    return {"task_id": task_id, "status": "started"}


async def _train_model_task(task_id: str, request: TrainModelRequest):
    """Background task for model training."""
    try:
        _update_task(task_id, status="running", progress=0.1, message="Loading training data...")

        if not disk_cache.exists(request.train_cache_key) or not disk_cache.exists(request.test_cache_key):
            raise ValueError("Training or test data not found in cache")

        train_data = disk_cache.load(request.train_cache_key)
        test_data = disk_cache.load(request.test_cache_key)

        _update_task(task_id, message=f"Train data: {train_data.shape[0]:,} rows x {train_data.shape[1]} columns")

        try:
            from pie.classifier import Classifier

            _update_task(task_id, progress=0.2, message="Setting up classifier...")

            classifier = Classifier()
            classifier.setup_experiment(
                data=train_data,
                target=request.target_column,
                test_data=test_data,
                session_id=42,
                verbose=False
            )

            compare_kwargs = dict(
                n_select=1,
                budget_time=request.time_budget_minutes,
                verbose=False,
            )
            if request.models_to_compare:
                compare_kwargs["include"] = request.models_to_compare
                _update_task(task_id, progress=0.4, message=f"Comparing {len(request.models_to_compare)} selected models...")
            else:
                _update_task(task_id, progress=0.4, message="Comparing models (auto-pilot)...")

            best_model = classifier.compare_models(**compare_kwargs)

            comparison_results = classifier.comparison_results
            _update_task(task_id, message=f"Best model: {type(best_model).__name__}")

            if request.tune_best:
                _update_task(task_id, progress=0.7, message=f"Tuning {type(best_model).__name__}...")
                best_model = classifier.tune_model(verbose=False)
                _update_task(task_id, message="Tuning complete")

            _update_task(task_id, progress=0.9, message="Generating predictions on test set...")

            predictions = classifier.predict_model(data=test_data, verbose=False)

            # Store the model
            model_id = f"model_{task_id}"
            _models[model_id] = {
                "classifier": classifier,
                "model": best_model,
                "comparison_results": comparison_results.to_dict() if comparison_results is not None else None
            }

            _update_task(
                task_id,
                status="completed", progress=1.0,
                message="Model training completed",
                result={
                    "model_id": model_id,
                    "model_name": type(best_model).__name__,
                    "comparison_results": comparison_results.head(10).to_dict() if comparison_results is not None else None,
                    "test_predictions_shape": list(predictions.shape),
                },
            )

        except ImportError:
            _update_task(task_id, status="failed", error="endgame/PIE Classifier not available",
                         message="Failed: endgame/PIE Classifier not available")

    except Exception as e:
        _update_task(task_id, status="failed", error=str(e) or repr(e), message=f"Failed: {e!r}")


@router.post("/auto_ml")
async def start_auto_ml(request: AutoMLRequest, background_tasks: BackgroundTasks):
    """Start endgame AutoML pipeline."""
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Initializing AutoML...",
        "result": None,
        "error": None,
    }

    background_tasks.add_task(_auto_ml_task, task_id, request)
    return {"task_id": task_id, "status": "started"}


async def _auto_ml_task(task_id: str, request: AutoMLRequest):
    """Background task for AutoML."""
    try:
        _update_task(task_id, status="running", progress=0.1, message="Loading data...")

        if not disk_cache.exists(request.train_cache_key) or not disk_cache.exists(request.test_cache_key):
            raise ValueError("Training or test data not found in cache")

        train_data = disk_cache.load(request.train_cache_key)
        test_data = disk_cache.load(request.test_cache_key)

        from pie.classifier import Classifier

        classifier = Classifier()
        classifier.setup_experiment(
            data=train_data,
            target=request.target_column,
            test_data=test_data,
            session_id=42,
            verbose=False,
        )

        _update_task(task_id, progress=0.2, message="Running AutoML pipeline...")

        predictor = classifier.auto_ml(
            time_limit=request.time_limit,
            presets=request.presets,
        )

        model_id = f"model_{task_id}"
        _models[model_id] = {
            "classifier": classifier,
            "model": predictor,
            "comparison_results": classifier.comparison_results.to_dict() if classifier.comparison_results is not None else None,
        }

        _update_task(
            task_id,
            status="completed", progress=1.0,
            message="AutoML completed",
            result={
                "model_id": model_id,
                "leaderboard": classifier.comparison_results.head(10).to_dict() if classifier.comparison_results is not None else None,
            },
        )

    except ImportError:
        _update_task(task_id, status="failed", error="endgame-ml not installed",
                     message="Failed: endgame-ml is required for AutoML. Install with: pip install endgame-ml[tabular]")
    except Exception as e:
        _update_task(task_id, status="failed", error=str(e) or repr(e), message=f"Failed: {e!r}")


@router.post("/calibrate")
async def start_calibration(request: CalibrateRequest, background_tasks: BackgroundTasks):
    """Calibrate a trained model's probability estimates."""
    if request.model_id not in _models:
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Initializing calibration...",
        "result": None,
        "error": None,
    }

    background_tasks.add_task(_calibrate_task, task_id, request)
    return {"task_id": task_id, "status": "started"}


async def _calibrate_task(task_id: str, request: CalibrateRequest):
    """Background task for model calibration."""
    try:
        _update_task(task_id, status="running", progress=0.3, message=f"Calibrating with {request.method}...")

        model_info = _models[request.model_id]
        classifier = model_info["classifier"]
        model = model_info["model"]

        calibrated = classifier.calibrate_model(estimator=model, method=request.method)

        calibrated_model_id = f"calibrated_{request.model_id}"
        _models[calibrated_model_id] = {
            "classifier": classifier,
            "model": calibrated,
            "comparison_results": model_info.get("comparison_results"),
        }

        _update_task(
            task_id,
            status="completed", progress=1.0,
            message=f"Calibration ({request.method}) completed",
            result={"model_id": calibrated_model_id, "method": request.method},
        )

    except Exception as e:
        _update_task(task_id, status="failed", error=str(e) or repr(e), message=f"Failed: {e!r}")


@router.post("/validate_drift")
async def start_drift_validation(request: DriftValidationRequest, background_tasks: BackgroundTasks):
    """Run adversarial validation to detect dataset drift."""
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Initializing drift validation...",
        "result": None,
        "error": None,
    }

    background_tasks.add_task(_drift_validation_task, task_id, request)
    return {"task_id": task_id, "status": "started"}


async def _drift_validation_task(task_id: str, request: DriftValidationRequest):
    """Background task for drift validation."""
    try:
        _update_task(task_id, status="running", progress=0.2, message="Loading data...")

        train_data = disk_cache.load(request.train_cache_key)
        test_data = disk_cache.load(request.test_cache_key)

        from pie.classifier import Classifier
        classifier = Classifier()
        result = classifier.validate_drift(train_data=train_data, test_data=test_data)

        _update_task(
            task_id,
            status="completed", progress=1.0,
            message="Drift validation completed",
            result={"drift_result": str(result)},
        )

    except ImportError:
        _update_task(task_id, status="failed", error="endgame-ml required for drift validation",
                     message="Failed: endgame-ml is required for drift validation")
    except Exception as e:
        _update_task(task_id, status="failed", error=str(e) or repr(e), message=f"Failed: {e!r}")


@router.post("/detect_leakage")
async def detect_leakage(request: DetectLeakageRequest):
    """Scan features for potential data leakage using statistical heuristics.

    Uses pre-computed column metadata (from the manifest / column cache) for
    name-based and cardinality checks so it doesn't need to reload the full
    dataset.  Only falls back to raw data loading for correlation analysis,
    and gracefully skips that step if the data isn't loadable.
    """
    if not disk_cache.exists(request.cache_key) and not disk_cache.is_modular(request.cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {request.cache_key}")

    start = time.time()

    # --- Obtain column metadata (fast, already in memory) -----------------
    col_meta = disk_cache.get_column_meta(request.cache_key)

    # If metadata isn't in memory yet (e.g. server restart), reload from manifest
    if col_meta is None and disk_cache.is_modular(request.cache_key):
        try:
            disk_cache._load_manifest_into_meta(request.cache_key)
            col_meta = disk_cache.get_column_meta(request.cache_key)
        except Exception:
            pass

    if col_meta is None:
        raise HTTPException(status_code=400, detail="Column metadata not available for this cache key")

    columns_info: List[Dict[str, Any]] = col_meta.get("columns", [])
    total_rows: int = col_meta.get("total_rows", 0)
    col_names = {c["name"] for c in columns_info}

    if request.target_column not in col_names:
        raise HTTPException(status_code=400, detail=f"Target column not found: {request.target_column}")

    feature_columns = [c for c in columns_info if c["name"] != request.target_column]
    total_scanned = len(feature_columns)
    suspicious: List[Dict[str, str]] = []

    # 1. Known PPMI leakage list -------------------------------------------
    try:
        from config.constants import LEAKAGE_FEATURES as KNOWN_LEAKAGE
    except ImportError:
        try:
            import sys
            pie_config = Path(__file__).resolve().parent.parent.parent / "lib" / "PIE"
            if str(pie_config) not in sys.path:
                sys.path.insert(0, str(pie_config))
            from config.constants import LEAKAGE_FEATURES as KNOWN_LEAKAGE
        except Exception:
            KNOWN_LEAKAGE = []

    known_set = set(KNOWN_LEAKAGE)
    for col_info in feature_columns:
        if col_info["name"] in known_set:
            suspicious.append({
                "name": col_info["name"],
                "reason": "known_leakage",
                "detail": "In PPMI known leakage list",
            })

    # 2. ID-like columns ---------------------------------------------------
    id_pattern = re.compile(r'(?:_ID$|^PATNO$|^EVENT_ID$|^SUBJECT_ID$|^SAMPLE_ID$)', re.IGNORECASE)
    already_flagged = {s["name"] for s in suspicious}
    for col_info in feature_columns:
        name = col_info["name"]
        if name in already_flagged:
            continue
        unique_count = col_info.get("unique_count", 0)
        is_id_name = bool(id_pattern.search(name))
        is_id_cardinality = total_rows > 0 and unique_count == total_rows
        if is_id_name or is_id_cardinality:
            if is_id_cardinality:
                detail = f"Unique count ({unique_count}) equals row count ({total_rows})"
            else:
                detail = "Column name matches ID pattern"
            suspicious.append({"name": name, "reason": "identifier", "detail": detail})

    # 3. Near-zero variance ------------------------------------------------
    already_flagged = {s["name"] for s in suspicious}
    for col_info in feature_columns:
        name = col_info["name"]
        if name in already_flagged:
            continue
        unique_count = col_info.get("unique_count", 0)
        # If only 1 unique value, it's constant (100 % same value)
        if unique_count <= 1 and total_rows > 0:
            suspicious.append({
                "name": name,
                "reason": "near_zero_variance",
                "detail": "Constant column (1 unique value)",
            })
        # Very low cardinality relative to rows → likely near-zero variance
        elif total_rows > 100 and unique_count == 2:
            null_pct = col_info.get("null_pct", 0)
            if null_pct > 99.5:
                suspicious.append({
                    "name": name,
                    "reason": "near_zero_variance",
                    "detail": f"{null_pct}% null values",
                })

    # 4. High target correlation (optional — needs raw data) ---------------
    # Try to load actual data for correlation analysis.  If the underlying
    # parquet files are unavailable (e.g. modular cache with missing files),
    # we skip this step rather than failing the whole scan.
    try:
        df: Optional[pd.DataFrame] = None
        if disk_cache.is_modular(request.cache_key):
            # Try loading target + numeric columns via load_columns
            numeric_names = [
                c["name"] for c in feature_columns
                if c.get("is_numeric") and c["name"] not in {s["name"] for s in suspicious}
            ]
            cols_to_load = [request.target_column] + numeric_names[:200]  # Cap to avoid OOM
            try:
                df = disk_cache.load_columns(request.cache_key, cols_to_load)
            except (KeyError, Exception):
                df = None
        else:
            raw = disk_cache.load(request.cache_key)
            df = raw if isinstance(raw, pd.DataFrame) else None

        if df is not None and request.target_column in df.columns:
            target = df[request.target_column]
            if not np.issubdtype(target.dtype, np.number):
                from sklearn.preprocessing import LabelEncoder
                try:
                    target_encoded = pd.Series(
                        LabelEncoder().fit_transform(target.dropna()),
                        index=target.dropna().index,
                    )
                except Exception:
                    target_encoded = None
            else:
                target_encoded = target

            if target_encoded is not None:
                already_flagged = {s["name"] for s in suspicious}
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col == request.target_column or col in already_flagged:
                        continue
                    try:
                        valid = df[col].notna() & target_encoded.reindex(df.index).notna()
                        if valid.sum() < 10:
                            continue
                        x = df.loc[valid, col].values.astype(float)
                        y = target_encoded.reindex(df.index).loc[valid].values.astype(float)
                        corr = np.abs(np.corrcoef(x, y)[0, 1])
                        if np.isnan(corr):
                            continue
                        if corr > 0.95:
                            suspicious.append({
                                "name": col,
                                "reason": "high_target_correlation",
                                "detail": f"Absolute correlation with target: {corr:.3f}",
                            })
                    except Exception:
                        continue
            del df
            gc.collect()
    except Exception:
        pass  # Correlation check is best-effort

    elapsed = time.time() - start

    return {
        "suspicious_features": suspicious,
        "total_scanned": total_scanned,
        "scan_time_seconds": round(elapsed, 2),
    }


@router.post("/create_ensemble")
async def create_ensemble(request: EnsembleRequest, background_tasks: BackgroundTasks):
    """Create an ensemble from a trained model."""
    if request.model_id not in _models:
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Initializing ensemble creation...",
        "result": None,
        "error": None,
    }

    background_tasks.add_task(_create_ensemble_task, task_id, request)
    return {"task_id": task_id, "status": "started"}


async def _create_ensemble_task(task_id: str, request: EnsembleRequest):
    """Background task for ensemble creation."""
    try:
        _update_task(task_id, status="running", progress=0.3, message=f"Creating {request.method} ensemble...")

        model_info = _models[request.model_id]
        classifier = model_info["classifier"]

        ensemble = classifier.create_ensemble(method=request.method)

        ensemble_model_id = f"ensemble_{request.model_id}"
        _models[ensemble_model_id] = {
            "classifier": classifier,
            "model": ensemble,
            "comparison_results": model_info.get("comparison_results"),
        }

        _update_task(
            task_id,
            status="completed", progress=1.0,
            message=f"Ensemble ({request.method}) created",
            result={"model_id": ensemble_model_id, "method": request.method},
        )

    except Exception as e:
        _update_task(task_id, status="failed", error=str(e) or repr(e), message=f"Failed: {e!r}")


@router.post("/run_pipeline")
async def start_full_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Start the full PIE pipeline."""
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Initializing pipeline...",
        "result": None,
        "error": None
    }

    background_tasks.add_task(
        _run_pipeline_task,
        task_id,
        request
    )

    return {"task_id": task_id, "status": "started"}


async def _run_pipeline_task(task_id: str, request: PipelineRequest):
    """Background task for running the full pipeline."""
    try:
        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["progress"] = 0.05
        _tasks[task_id]["message"] = "Starting PIE pipeline..."

        try:
            from pie.pipeline import run_pipeline
            from pie_clean import ALL_MODALITIES

            modalities = request.modalities if request.modalities else ALL_MODALITIES

            _tasks[task_id]["progress"] = 0.1
            _tasks[task_id]["message"] = "Running data reduction..."

            # This will run the full pipeline
            run_pipeline(
                data_dir=request.data_path,
                output_dir=request.output_dir,
                target_column=request.target_column,
                leakage_features_path=request.leakage_features_path,
                modalities=modalities,
                fs_method=request.fs_method,
                fs_param_value=request.fs_param,
                n_models_to_compare=request.n_models,
                tune_best_model=request.tune_best,
                generate_plots=request.generate_plots,
                budget_time_minutes=request.budget_minutes
            )

            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["progress"] = 1.0
            _tasks[task_id]["message"] = "Pipeline completed successfully"
            _tasks[task_id]["result"] = {
                "output_dir": request.output_dir,
                "report_path": f"{request.output_dir}/pipeline_report.html"
            }

        except ImportError as e:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = f"PIE library not available: {str(e)}"

    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)


@router.get("/task/{task_id}")
async def get_analysis_task_status(task_id: str):
    """Get the status of an analysis task, including full log history."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = _tasks[task_id]
    return {
        "status": task.get("status"),
        "progress": task.get("progress", 0.0),
        "message": task.get("message", ""),
        "result": task.get("result"),
        "error": task.get("error"),
        "logs": task.get("logs", []),
    }


@router.get("/model/{model_id}/feature_importance")
async def get_feature_importance(model_id: str, top_n: int = 20):
    """Get feature importance from a trained model."""
    if model_id not in _models:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    model_info = _models[model_id]
    model = model_info["model"]

    try:
        # Try to get feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
        else:
            return {"error": "Model does not support feature importance"}

        # Get feature names from the classifier
        classifier = model_info["classifier"]
        feature_names = classifier.get_config('X_train').columns.tolist()

        # Sort by importance
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)

        return {
            "features": importance_df['feature'].tolist(),
            "importances": importance_df['importance'].tolist()
        }

    except Exception as e:
        return {"error": str(e)}
