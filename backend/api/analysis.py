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
from fastapi.responses import HTMLResponse
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
# Bound the number of trained classifiers we keep resident. Each entry pins
# the classifier's training data, comparison results, and best-model object;
# letting the dict grow unboundedly across runs is the single largest source
# of slow RAM creep observed in the workbench.
_MAX_RETAINED_MODELS = 3


def _free_classifier(model_entry: Dict[str, Any]) -> None:
    """Release the heavy state hanging off a stored classifier so the next
    training run starts from a clean baseline."""
    classifier = model_entry.get("classifier")
    if classifier is None:
        return
    for attr in ("_X_train", "_X_test", "_y_train", "_y_test"):
        try:
            setattr(classifier, attr, None)
        except Exception:
            pass
    try:
        classifier.models_dict = {}
    except Exception:
        pass
    try:
        classifier.setup_params = None
    except Exception:
        pass


def _evict_old_models(keep: int = _MAX_RETAINED_MODELS) -> None:
    """Drop the oldest stored classifiers to keep RAM bounded across runs."""
    while len(_models) >= keep:
        oldest_id = next(iter(_models))
        _free_classifier(_models[oldest_id])
        _models.pop(oldest_id, None)
        gc.collect()


class TaskCancelled(Exception):
    """Raised when a task has been cancelled by the user."""
    pass


def _check_cancelled(task_id: str):
    """Raise TaskCancelled if the user requested cancellation."""
    task = _tasks.get(task_id)
    if task and task.get("cancel_requested"):
        raise TaskCancelled(f"Task {task_id} cancelled by user")


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
    # Per-ensemble base-learner configs: {ensemble_id: [base_model_ids]}. Only
    # populated when the user selects a meta-ensemble (bagging, stacking, etc.)
    # in the Model Arena and picks its underlying estimators.
    ensemble_configs: Dict[str, List[str]] = {}
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


# Fallback family assignments for models sourced from the classifier.py static
# catalog (which lacks a `family` field). Used only when endgame's ModelInfo
# isn't available for a given id.
_STATIC_FAMILY_BY_ID: Dict[str, str] = {
    # Classification + regression shared ids
    "lr": "linear",
    "ridge": "linear",
    "lasso": "linear",
    "elastic_net": "linear",
    "lda": "linear",
    "qda": "linear",
    "rf": "tree",
    "et": "tree",
    "dt": "tree",
    "gbc": "gbdt",
    "ada": "ensemble",
    "xgboost": "gbdt",
    "lightgbm": "gbdt",
    "catboost": "gbdt",
    "knn": "kernel",
    "svm": "kernel",
    "nb": "bayesian",
    "ebm": "ensemble",
    "tabnet": "neural",
    "saint": "neural",
    "ft_transformer": "neural",
    "node": "neural",
    "rule_fit": "rules",
}


def _family_for(model_id: str, info: Any) -> str:
    """Derive the model family (e.g. 'tree', 'linear') from a catalog entry."""
    if isinstance(info, dict):
        return info.get("family") or _STATIC_FAMILY_BY_ID.get(model_id, "other")
    # endgame ModelInfo dataclass
    return getattr(info, "family", None) or _STATIC_FAMILY_BY_ID.get(model_id, "other")


def _interpretable_for(info: Any) -> bool:
    if isinstance(info, dict):
        return bool(info.get("interpretable", False))
    return bool(getattr(info, "interpretable", False))


def _display_name_for(model_id: str, info: Any) -> str:
    """Prefer display_name for endgame ModelInfo, else name/id."""
    if isinstance(info, dict):
        return info.get("name", model_id)
    return getattr(info, "display_name", None) or getattr(info, "name", model_id)


# Ensemble meta-methods from endgame.ensemble — these wrap user-selected base
# learners and are surfaced as individually-selectable models in the Ensemble
# family. Each entry carries `accepts_base_learners=True` so the UI renders a
# sub-picker for the underlying estimators.
_ENSEMBLE_META_METHODS: List[Dict[str, Any]] = [
    {"id": "bagging", "name": "Bagging", "class_path": "endgame.ensemble.bagging.BaggingClassifier"},
    {"id": "stacking", "name": "Stacking", "class_path": "endgame.ensemble.stacking.StackingEnsemble"},
    {"id": "super_learner", "name": "Super Learner", "class_path": "endgame.ensemble.super_learner.SuperLearner"},
    {"id": "blending", "name": "Blending", "class_path": "endgame.ensemble.blending.BlendingEnsemble"},
    {"id": "voting", "name": "Voting", "class_path": "endgame.ensemble.voting.VotingClassifier"},
    {"id": "bma", "name": "Bayesian Model Averaging", "class_path": "endgame.ensemble.bayesian_averaging.BayesianModelAveraging"},
    {"id": "hill_climbing", "name": "Hill Climbing", "class_path": "endgame.ensemble.hill_climbing.HillClimbingEnsemble"},
    {"id": "snapshot", "name": "Snapshot Ensemble", "class_path": "endgame.ensemble.snapshot.SnapshotEnsemble"},
    {"id": "cascade", "name": "Cascade Ensemble", "class_path": "endgame.ensemble.cascade.CascadeEnsemble"},
    {"id": "neg_corr", "name": "Negative Correlation", "class_path": "endgame.ensemble.negative_correlation.NegativeCorrelationEnsemble"},
]

_ENSEMBLE_METHOD_IDS = frozenset(m["id"] for m in _ENSEMBLE_META_METHODS)

# Map UI ensemble ids → methods understood by Classifier.create_ensemble.
# create_ensemble currently implements: super_learner, bma, blending, bagging,
# boosting. Ids not in this map fall back to super_learner (a safe default for
# generic meta-ensembles like hill_climbing / snapshot / cascade that the
# Classifier doesn't yet support natively).
_ENSEMBLE_ID_TO_PIE_METHOD: Dict[str, str] = {
    "bagging": "bagging",
    "stacking": "super_learner",
    "super_learner": "super_learner",
    "blending": "blending",
    "voting": "super_learner",
    "bma": "bma",
    "hill_climbing": "super_learner",
    "snapshot": "super_learner",
    "cascade": "super_learner",
    "neg_corr": "super_learner",
}


def _ensemble_is_importable(class_path: str) -> bool:
    """Check whether an ensemble class exists on disk without instantiating it
    (ensemble meta-learners require base_estimators, so we can't just call ())."""
    if "." not in class_path:
        return False
    module_name, cls_name = class_path.rsplit(".", 1)
    try:
        import importlib
        mod = importlib.import_module(module_name)
        return hasattr(mod, cls_name)
    except Exception:
        return False


@router.get("/available_models")
async def get_available_models(task_type: str = "classification"):
    """Get available ML models from endgame's dynamic catalog.

    Deduplicates by display_name so that models registered under multiple
    ids (e.g. 'xgboost' from the sklearn catalog and 'xgb' from the endgame
    registry) only appear once. Shorter ids win (proxy for canonical).
    Ensemble meta-methods (bagging, stacking, super learner, etc.) are
    appended from the static `_ENSEMBLE_META_METHODS` list and carry an
    `accepts_base_learners` flag so the UI can render a base-learner picker.
    """
    try:
        from pie.classifier import get_model_catalog
        catalog = get_model_catalog(task_type)
        by_name: Dict[str, Dict[str, Any]] = {}
        for k, v in catalog.items():
            name = _display_name_for(k, v)
            entry = {
                "id": k,
                "name": name,
                "family": _family_for(k, v),
                "interpretable": _interpretable_for(v),
                "accepts_base_learners": False,
            }
            existing = by_name.get(name)
            if existing is None or len(k) < len(existing["id"]):
                by_name[name] = entry

        # Append ensemble meta-methods (only those importable on this env)
        for meta in _ENSEMBLE_META_METHODS:
            if not _ensemble_is_importable(meta["class_path"]):
                continue
            # Don't clobber a same-name entry if one already exists
            if meta["name"] in by_name:
                continue
            by_name[meta["name"]] = {
                "id": meta["id"],
                "name": meta["name"],
                "family": "ensemble",
                "interpretable": False,
                "accepts_base_learners": True,
            }

        return {"models": list(by_name.values())}
    except ImportError:
        # Fallback static catalog
        if task_type == "classification":
            fallback = [
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
        else:
            fallback = [
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
        for m in fallback:
            m["family"] = _STATIC_FAMILY_BY_ID.get(m["id"], "other")
            m["interpretable"] = False
        return {"models": fallback}


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
            # FeatureEngineer takes its own .copy() in __init__; release ours
            # so the post-OHE working set isn't double-resident.
            del merged_df
            gc.collect()

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
            # get_dataframe() returns yet another .copy(); drop the engineer's
            # internal frame so we're not pinning the same data twice.
            del engineer
            gc.collect()

        except ImportError:
            _update_task(task_id, message="PIE FeatureEngineer not available — basic OHE fallback...")
            engineered_df = merged_df
            del merged_df  # we now reach the engineered frame via engineered_df
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
        del engineered_df
        # merged_df was already released after FeatureEngineer init (or never
        # created in the ImportError fallback path).
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

    except TaskCancelled:
        logger.info("Task %s cancelled by user", task_id)
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

    Each CPU-bound step is dispatched through ``loop.run_in_executor`` so the
    asyncio loop stays responsive — otherwise the status-poll endpoint serves
    stale data and the UI looks "stuck" even when work is progressing.
    """
    loop = asyncio.get_event_loop()
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        _update_task(task_id, status="running", progress=0.1, message="Loading engineered data from cache...")

        data = await loop.run_in_executor(None, disk_cache.load, request.cache_key)
        df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
        del data
        gc.collect()

        # Undo Categorical dtypes from the cache/parquet layer
        def _strip_categoricals(frame: pd.DataFrame) -> pd.DataFrame:
            for col in frame.columns:
                if isinstance(frame[col].dtype, pd.CategoricalDtype):
                    frame[col] = frame[col].astype(frame[col].cat.categories.dtype)
            return frame

        df = await loop.run_in_executor(None, _strip_categoricals, df)

        _update_task(task_id, progress=0.15,
                     message=f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

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
        # Off-thread: this can be many million Python-level apply() calls on
        # PPMI's pipe-encoded medical-history fields and otherwise blocks the
        # event loop for tens of seconds.
        _ID_DATE_PATS = ["ID", "DATE", "TIME", "PATNO", "EVENT"]

        def _resolve_pipes(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
            resolved = 0
            for col in frame.select_dtypes(include=["object"]).columns:
                if any(p in col.upper() for p in _ID_DATE_PATS):
                    continue
                if not frame[col].astype(str).str.contains(r"\|", na=False).any():
                    continue

                def _avg_pipe(val):
                    if isinstance(val, str) and "|" in val:
                        try:
                            return np.mean([float(x) for x in val.split("|")])
                        except (ValueError, TypeError):
                            return np.nan
                    return val

                converted = frame[col].apply(_avg_pipe)
                numeric = pd.to_numeric(converted, errors="coerce")
                n_nonnull = frame[col].notna().sum()
                n_numeric = numeric.notna().sum()
                if n_nonnull > 0 and (n_numeric / n_nonnull) > 0.9:
                    frame[col] = numeric
                    resolved += 1
            return frame, resolved

        _update_task(task_id, progress=0.25, message="Resolving pipe-separated columns...")
        X, pipe_resolved = await loop.run_in_executor(None, _resolve_pipes, X)
        if pipe_resolved:
            _update_task(task_id, message=f"Resolved pipe-separated values in {pipe_resolved} columns")

        # Drop remaining non-numeric columns
        non_numeric = X.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric):
            _update_task(task_id, message=f"Dropping {len(non_numeric)} non-numeric columns")
            X = X.drop(columns=non_numeric)

        _update_task(task_id, progress=0.30, message="Filling missing values...")
        X = await loop.run_in_executor(None, lambda: X.fillna(0))

        _update_task(task_id, progress=0.35,
                     message=f"Numeric features: {X.shape[1]}, target classes: {y.nunique()}")

        _update_task(task_id, progress=0.40, message="Encoding target labels...")
        le = LabelEncoder()
        y_encoded = await loop.run_in_executor(None, le.fit_transform, y)

        _update_task(task_id, progress=0.45, message="Splitting train/test (stratified)...")

        def _split():
            return train_test_split(
                X, y, test_size=request.test_size, random_state=42, stratify=y_encoded,
            )

        X_train, X_test, y_train, y_test = await loop.run_in_executor(None, _split)
        _update_task(task_id, progress=0.50,
                     message=f"Train: {X_train.shape[0]:,} rows, Test: {X_test.shape[0]:,} rows")

        if request.method == "none":
            _update_task(task_id, message="Skipping feature selection — keeping all features")
            X_train_selected = X_train
            X_test_selected = X_test
            selected_features = list(X_train.columns)
        else:
            try:
                from pie.feature_selector import FeatureSelector

                _update_task(task_id, progress=0.55,
                             message=f"Applying {request.method} feature selection on {X_train.shape[0]:,} rows × {X_train.shape[1]} features...")

                selector = FeatureSelector(
                    method=request.method,
                    task_type="classification",
                    k_or_frac=request.param_value if request.method == "k_best" else None,
                    alpha_fdr=request.param_value if request.method == "fdr" else 0.05,
                )

                y_train_enc = await loop.run_in_executor(None, le.fit_transform, y_train)

                _update_task(task_id, progress=0.60,
                             message="Scoring features (this is the longest step)...")

                def _fit_and_transform():
                    selector.fit(X_train, y_train_enc)
                    return (
                        selector.transform(X_train),
                        selector.transform(X_test),
                        selector.selected_feature_names_,
                    )

                X_train_selected, X_test_selected, selected_features = await loop.run_in_executor(
                    None, _fit_and_transform
                )

            except ImportError:
                from sklearn.feature_selection import VarianceThreshold
                _update_task(task_id, message="PIE FeatureSelector not available, using VarianceThreshold fallback...")

                def _vt_fit():
                    vt = VarianceThreshold(threshold=0.01)
                    Xtr = pd.DataFrame(
                        vt.fit_transform(X_train), columns=X_train.columns[vt.get_support()]
                    )
                    Xte = pd.DataFrame(
                        vt.transform(X_test), columns=X_train.columns[vt.get_support()]
                    )
                    return Xtr, Xte, list(Xtr.columns)

                X_train_selected, X_test_selected, selected_features = await loop.run_in_executor(
                    None, _vt_fit
                )

        _update_task(task_id, progress=0.85,
                     message=f"Selected {len(selected_features)} features from {X.shape[1]}")

        def _build_split_frames():
            train = pd.concat([X_train_selected.reset_index(drop=True),
                               y_train.reset_index(drop=True)], axis=1)
            test = pd.concat([X_test_selected.reset_index(drop=True),
                              y_test.reset_index(drop=True)], axis=1)
            return train, test

        train_df, test_df = await loop.run_in_executor(None, _build_split_frames)

        train_cache_key = f"train_{task_id}"
        test_cache_key = f"test_{task_id}"
        _update_task(task_id, progress=0.92, message="Caching train/test splits to disk...")
        await loop.run_in_executor(None, disk_cache.store, train_cache_key, train_df)
        await loop.run_in_executor(None, disk_cache.store, test_cache_key, test_df)

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

    except TaskCancelled:
        logger.info("Task %s cancelled by user", task_id)
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
    """Background task for model training.

    Heavy CPU work (cache load, ``compare_models``, tuning, prediction) is
    dispatched through ``loop.run_in_executor`` so the asyncio loop can keep
    serving status polls — otherwise the UI looks "stuck" mid-training.
    """
    loop = asyncio.get_event_loop()
    try:
        _check_cancelled(task_id)
        _update_task(task_id, status="running", progress=0.1, message="Loading training data...")

        if not disk_cache.exists(request.train_cache_key) or not disk_cache.exists(request.test_cache_key):
            raise ValueError("Training or test data not found in cache")

        # Free retained classifiers from prior runs *before* loading new data.
        # Each retained entry can hold gigabytes (training matrix + compared
        # models), so this is the largest single RAM win between runs.
        _evict_old_models()

        train_data = await loop.run_in_executor(None, disk_cache.load, request.train_cache_key)
        test_data = await loop.run_in_executor(None, disk_cache.load, request.test_cache_key)

        # Undo ALL Categorical dtypes from the cache/parquet layer.
        # numpy/sklearn cannot interpret CategoricalDtype as a data type.
        target = request.target_column

        def _normalize_dtypes(frames):
            for frame in frames:
                for col in frame.columns:
                    if isinstance(frame[col].dtype, pd.CategoricalDtype):
                        frame[col] = frame[col].astype(frame[col].cat.categories.dtype)
                if target in frame.columns and isinstance(frame[target].dtype, pd.CategoricalDtype):
                    frame[target] = frame[target].astype(str)
            return frames

        train_data, test_data = await loop.run_in_executor(
            None, _normalize_dtypes, (train_data, test_data)
        )

        _check_cancelled(task_id)
        _update_task(task_id, message=f"Train data: {train_data.shape[0]:,} rows x {train_data.shape[1]} columns")

        try:
            from pie.classifier import Classifier

            _update_task(task_id, progress=0.2, message="Setting up classifier...")

            classifier = Classifier()

            def _setup():
                classifier.setup_experiment(
                    data=train_data,
                    target=request.target_column,
                    test_data=test_data,
                    session_id=42,
                    verbose=False,
                )

            await loop.run_in_executor(None, _setup)

            # Classifier has now copied what it needs (_X_train / _X_test).
            # Drop our local refs so the loaded frames can be reclaimed before
            # CV begins — otherwise we hold the data twice for the entire run.
            del train_data, test_data
            gc.collect()

            _check_cancelled(task_id)

            # Split the user's model selection into regular models (trained via
            # compare_models) and ensemble meta-methods (built separately from
            # user-specified base learners, after the regular CV loop).
            ensemble_ids = set(_ENSEMBLE_METHOD_IDS)
            regular_models = [m for m in request.models_to_compare if m not in ensemble_ids]
            ensemble_models = [m for m in request.models_to_compare if m in ensemble_ids]

            compare_kwargs = dict(
                n_select=1,
                budget_time=request.time_budget_minutes,
                verbose=False,
            )
            if regular_models:
                compare_kwargs["include"] = regular_models
                _update_task(task_id, progress=0.4,
                             message=f"Comparing {len(regular_models)} selected models (CV)...")
            elif request.models_to_compare and not regular_models:
                # Only ensemble methods were picked — compare_models still
                # needs at least one base comparison, so use endgame's default
                # short list to pick the "best" non-ensemble baseline.
                _update_task(task_id, progress=0.4,
                             message="No regular models selected — running default comparison for best baseline...")
            else:
                _update_task(task_id, progress=0.4, message="Comparing models (auto-pilot)...")

            best_model = await loop.run_in_executor(
                None, lambda: classifier.compare_models(**compare_kwargs)
            )

            _check_cancelled(task_id)
            comparison_results = classifier.comparison_results
            _update_task(task_id, progress=0.65,
                         message=f"Best model: {type(best_model).__name__}")

            if request.tune_best:
                _check_cancelled(task_id)
                _update_task(task_id, progress=0.7, message=f"Tuning {type(best_model).__name__}...")
                best_model = await loop.run_in_executor(
                    None, lambda: classifier.tune_model(verbose=False)
                )
                _update_task(task_id, message="Tuning complete")

            _check_cancelled(task_id)
            _update_task(task_id, progress=0.85, message="Generating predictions on test set...")

            # Use the classifier's stored _X_test (already a copy made during
            # setup_experiment) rather than re-passing test_data — that local
            # has been freed to keep RAM bounded during CV.
            predictions = await loop.run_in_executor(
                None, lambda: classifier.predict_model(verbose=False)
            )
            predictions_shape = list(predictions.shape)
            del predictions  # large per-row score frame, no longer needed

            # -------- Ensemble meta-methods ----------------------------------
            # For each ensemble the user selected, build it from their chosen
            # base learners and fit. Each failure is logged and skipped so one
            # broken ensemble can't sink the whole run.
            ensemble_results: List[Dict[str, Any]] = []
            if ensemble_models:
                from pie.classifier import _instantiate_model  # type: ignore
                for i, ens_id in enumerate(ensemble_models):
                    _check_cancelled(task_id)
                    base_ids = request.ensemble_configs.get(ens_id, [])
                    if not base_ids:
                        _update_task(
                            task_id,
                            message=f"Skipping {ens_id}: no base learners configured",
                        )
                        continue
                    progress = 0.85 + 0.1 * ((i + 1) / max(len(ensemble_models), 1))
                    _update_task(
                        task_id, progress=progress,
                        message=f"Building {ens_id} over {len(base_ids)} base learner(s)...",
                    )

                    def _build_and_eval(ens_id=ens_id, base_ids=base_ids):
                        base_instances = []
                        for bid in base_ids:
                            try:
                                base_instances.append(_instantiate_model(bid, request.task_type))
                            except Exception as exc:
                                logger.warning("Ensemble %s: failed to instantiate base '%s': %s", ens_id, bid, exc)
                        if not base_instances:
                            raise ValueError(f"No usable base learners for {ens_id}")
                        method = _ENSEMBLE_ID_TO_PIE_METHOD.get(ens_id, "super_learner")
                        ens = classifier.create_ensemble(base_models=base_instances, method=method)
                        # Score on the held-out test set so results are
                        # comparable to the regular compare_models output.
                        import numpy as np
                        X_test = classifier._X_test
                        y_test = classifier._encode_target(classifier._y_test) if classifier._y_test is not None else None
                        try:
                            preds = ens.predict(X_test)
                        except Exception as exc:
                            logger.warning("Ensemble %s predict failed: %s", ens_id, exc)
                            return None
                        entry: Dict[str, Any] = {
                            "ensemble_id": ens_id,
                            "method": method,
                            "n_base_learners": len(base_instances),
                        }
                        if y_test is not None:
                            if request.task_type == "classification":
                                entry["accuracy"] = float(np.mean(preds == y_test))
                            else:
                                entry["rmse"] = float(np.sqrt(np.mean((preds - y_test) ** 2)))
                        return entry

                    try:
                        result = await loop.run_in_executor(None, _build_and_eval)
                        if result:
                            ensemble_results.append(result)
                            _update_task(task_id, message=f"{ens_id} trained: {result}")
                    except Exception as exc:
                        logger.exception("Ensemble %s failed", ens_id)
                        _update_task(task_id, message=f"{ens_id} failed: {exc}")

            # Store the model. comparison_results is converted to a plain dict
            # so we don't keep a live pandas object around just for inspection.
            model_id = f"model_{task_id}"
            _models[model_id] = {
                "classifier": classifier,
                "model": best_model,
                "comparison_results": comparison_results.to_dict() if comparison_results is not None else None,
                "ensemble_results": ensemble_results,
            }

            _update_task(
                task_id,
                status="completed", progress=1.0,
                message="Model training completed",
                result={
                    "model_id": model_id,
                    "model_name": type(best_model).__name__,
                    "comparison_results": comparison_results.head(10).to_dict() if comparison_results is not None else None,
                    "ensemble_results": ensemble_results,
                    "test_predictions_shape": predictions_shape,
                },
            )

        except ImportError:
            _update_task(task_id, status="failed", error="endgame/PIE Classifier not available",
                         message="Failed: endgame/PIE Classifier not available")

    except TaskCancelled:
        logger.info("Task %s cancelled by user", task_id)
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

        # Same eviction discipline as _train_model_task: prior runs can hold
        # gigabytes of training state. Free them before loading new frames.
        _evict_old_models()

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
        del train_data, test_data
        gc.collect()

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


def _calibration_metrics(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """Compute Brier score, log loss, and expected calibration error.

    Works for both binary (n_classes=2) and multiclass probability arrays.
    ECE uses equal-width confidence bins on the predicted class probability.
    """
    from sklearn.metrics import log_loss

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    if y_proba.ndim == 1:
        y_proba = np.column_stack([1.0 - y_proba, y_proba])

    n_samples, n_classes = y_proba.shape
    one_hot = np.zeros_like(y_proba)
    one_hot[np.arange(n_samples), y_true] = 1.0
    brier = float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))

    labels = list(range(n_classes))
    ll = float(log_loss(y_true, y_proba, labels=labels))

    pred_class = np.argmax(y_proba, axis=1)
    confidence = y_proba[np.arange(n_samples), pred_class]
    correct = (pred_class == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi) if hi < 1.0 else (confidence > lo) & (confidence <= hi + 1e-12)
        if not mask.any():
            continue
        bin_conf = float(confidence[mask].mean())
        bin_acc = float(correct[mask].mean())
        ece += (mask.sum() / n_samples) * abs(bin_conf - bin_acc)

    return {"brier": brier, "log_loss": ll, "ece": float(ece)}


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

        _update_task(task_id, status="running", progress=0.8, message="Evaluating calibration quality...")

        diagnostics: Dict[str, Any] = {}
        try:
            X_test = classifier._X_test
            y_test = classifier._y_test
            if X_test is not None and y_test is not None and hasattr(model, "predict_proba") and hasattr(calibrated, "predict_proba"):
                y_true = classifier._encode_target(y_test)
                before = _calibration_metrics(y_true, model.predict_proba(X_test))
                after = _calibration_metrics(y_true, calibrated.predict_proba(X_test))
                diagnostics = {
                    "before": before,
                    "after": after,
                    "delta": {k: after[k] - before[k] for k in before},
                    "n_test_samples": int(len(y_true)),
                }
        except Exception as diag_exc:
            logger.warning(f"Calibration diagnostics failed: {diag_exc!r}")
            diagnostics = {"error": str(diag_exc)}

        _update_task(
            task_id,
            status="completed", progress=1.0,
            message=f"Calibration ({request.method}) completed",
            result={
                "model_id": calibrated_model_id,
                "method": request.method,
                "diagnostics": diagnostics,
            },
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

        severity = getattr(result, "drift_severity", "unknown")
        auc = getattr(result, "auc_score", None)
        if severity == "none":
            summary = f"No drift detected (AUC {auc:.3f})" if auc is not None else "No drift detected"
        else:
            summary = (
                f"Drift detected — severity: {severity} (AUC {auc:.3f})"
                if auc is not None
                else f"Drift detected — severity: {severity}"
            )

        _update_task(
            task_id,
            status="completed", progress=1.0,
            message="Drift validation completed",
            result={"drift_result": summary},
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


@router.post("/task/{task_id}/cancel")
async def cancel_task(task_id: str):
    """Request cancellation of a running task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    task = _tasks[task_id]
    if task.get("status") in ("completed", "failed", "cancelled"):
        return {"status": task["status"], "message": "Task already finished"}

    task["cancel_requested"] = True
    task["status"] = "cancelled"
    task["message"] = "Cancelled by user"
    logger.info("Task %s: cancellation requested", task_id)
    return {"status": "cancelled", "message": "Cancellation requested"}


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


@router.get("/model/{model_id}/results")
async def get_model_results(model_id: str):
    """Get full model results: best model info, metrics, comparison table, confusion matrix."""
    if model_id not in _models:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    model_info = _models[model_id]
    classifier = model_info["classifier"]
    best_model = model_info["model"]
    comparison_results = model_info.get("comparison_results")

    # --- Best model metrics ---
    from sklearn.metrics import confusion_matrix as sk_confusion_matrix

    X_test = classifier._X_test
    y_test = classifier._y_test
    y_test_enc = classifier._encode_target(y_test)
    y_pred = best_model.predict(X_test)

    # Compute metrics
    try:
        y_proba = best_model.predict_proba(X_test) if hasattr(best_model, 'predict_proba') else None
    except Exception:
        y_proba = None

    from pie.classifier import _compute_metrics
    metrics = _compute_metrics(y_test_enc, y_pred, y_proba)

    # --- Confusion matrix ---
    labels = sorted(y_test_enc.unique()) if hasattr(y_test_enc, 'unique') else sorted(set(y_test_enc))
    cm = sk_confusion_matrix(y_test_enc, y_pred, labels=labels)

    # Map encoded labels back to original names if label encoder exists
    if classifier._label_encoder is not None:
        label_names = [str(classifier._label_encoder.inverse_transform([l])[0]) for l in labels]
    else:
        label_names = [str(l) for l in labels]

    confusion = []
    for i, actual in enumerate(label_names):
        for j, predicted in enumerate(label_names):
            confusion.append({
                "actual": actual,
                "predicted": predicted,
                "value": int(cm[i][j]),
            })

    # Clean NaN from metrics for JSON
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, float) and np.isnan(v):
            clean_metrics[k] = None
        else:
            clean_metrics[k] = float(v) if isinstance(v, (float, np.floating)) else v

    return {
        "best_model_name": type(best_model).__name__,
        "metrics": clean_metrics,
        "confusion_matrix": confusion,
        "class_labels": label_names,
        "comparison": comparison_results,
    }


@router.get("/model/{model_id}/report", response_class=HTMLResponse)
async def get_classification_report(model_id: str):
    """Generate and return the full classification report HTML."""
    if model_id not in _models:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    model_info = _models[model_id]
    classifier = model_info["classifier"]

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        classifier.generate_report(output_path=tmp_path)
        with open(tmp_path, "r", encoding="utf-8") as f:
            html = f.read()
        return HTMLResponse(content=html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
