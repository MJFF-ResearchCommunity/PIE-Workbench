"""
ML Analysis API endpoints.

Handles feature engineering, selection, and model training using PIE.
"""

import gc
import os
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import pandas as pd
import numpy as np

from . import cache as disk_cache

router = APIRouter()

# Task and data stores
_tasks: Dict[str, Dict[str, Any]] = {}
_models: Dict[str, Any] = {}


class FeatureEngineeringRequest(BaseModel):
    """Request for feature engineering."""
    cache_key: str
    scale_numeric: bool = True
    one_hot_encode: bool = True
    max_categories: int = 25
    min_frequency: float = 0.01
    modalities: Optional[List[str]] = None


class FeatureSelectionRequest(BaseModel):
    """Request for feature selection."""
    cache_key: str
    target_column: str
    method: str = "fdr"  # fdr, k_best, select_from_model, rfe
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
    """Get available feature selection methods."""
    return {
        "methods": [
            {"id": "fdr", "name": "False Discovery Rate (FDR)", "description": "Select features with statistical significance"},
            {"id": "k_best", "name": "K-Best", "description": "Select top K features by score"},
            {"id": "select_from_model", "name": "Model-Based", "description": "Use a model to rank features"},
            {"id": "rfe", "name": "Recursive Feature Elimination", "description": "Iteratively remove least important features"},
        ]
    }


@router.get("/available_models")
async def get_available_models(task_type: str = "classification"):
    """Get available ML models for the given task type."""
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

    # For modular caches, load only the target column
    if disk_cache.is_modular(cache_key):
        try:
            data = disk_cache.load_columns(cache_key, [target_column])
        except KeyError:
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

    # Heuristic: if numeric with many unique values, suggest regression
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
    """Background task for feature engineering."""
    try:
        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["progress"] = 0.1

        # Load data — modular-aware
        if disk_cache.is_modular(request.cache_key):
            _tasks[task_id]["message"] = "Loading modalities from modular cache..."

            if request.modalities:
                # Load specific requested modalities
                df = disk_cache.load_modalities(request.cache_key, request.modalities)
            else:
                # Default: load all non-biospecimen modalities (manageable column count)
                all_files = disk_cache.list_modality_files(request.cache_key)
                non_bio = [f for f in all_files if not f.startswith("biospecimen__")]
                if non_bio:
                    df = disk_cache.load_modalities(request.cache_key, non_bio)
                else:
                    df = disk_cache.load_modalities(request.cache_key, all_files)
        else:
            data = disk_cache.load(request.cache_key)
            df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
            del data
            gc.collect()

        try:
            from pie.feature_engineer import FeatureEngineer

            _tasks[task_id]["message"] = "Applying feature engineering..."
            _tasks[task_id]["progress"] = 0.3

            engineer = FeatureEngineer(df)

            if request.one_hot_encode:
                _tasks[task_id]["message"] = "One-hot encoding categorical features..."
                engineer.one_hot_encode(
                    auto_identify_threshold=20,
                    max_categories_to_encode=request.max_categories,
                    min_frequency_for_category=request.min_frequency
                )

            _tasks[task_id]["progress"] = 0.6

            if request.scale_numeric:
                _tasks[task_id]["message"] = "Scaling numeric features..."
                engineer.scale_numeric_features(scaler_type='standard')

            _tasks[task_id]["progress"] = 0.8

            engineered_df = engineer.get_dataframe()
            summary = engineer.get_engineered_feature_summary()

            # Cache the engineered data to disk
            new_cache_key = f"engineered_{task_id}"
            original_shape = list(df.shape)
            new_shape = list(engineered_df.shape)
            disk_cache.store(new_cache_key, engineered_df)
            del engineered_df, df
            gc.collect()

            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["progress"] = 1.0
            _tasks[task_id]["message"] = "Feature engineering completed"
            _tasks[task_id]["result"] = {
                "cache_key": new_cache_key,
                "original_shape": original_shape,
                "new_shape": new_shape,
                "summary": summary
            }

        except ImportError:
            # Fallback: basic feature engineering
            _tasks[task_id]["message"] = "Using basic feature engineering (PIE not available)..."

            # Simple one-hot encoding
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df[col].nunique() <= request.max_categories:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df.drop(col, axis=1), dummies], axis=1)

            new_cache_key = f"engineered_{task_id}"
            new_shape = list(df.shape)
            disk_cache.store(new_cache_key, df)
            del df
            gc.collect()

            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["progress"] = 1.0
            _tasks[task_id]["result"] = {
                "cache_key": new_cache_key,
                "new_shape": new_shape
            }

    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)


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
    """Background task for feature selection."""
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["progress"] = 0.1

        data = disk_cache.load(request.cache_key)
        df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
        del data
        gc.collect()

        # Remove leakage features
        if request.leakage_features:
            cols_to_drop = [c for c in request.leakage_features if c in df.columns]
            df = df.drop(columns=cols_to_drop)
            _tasks[task_id]["message"] = f"Removed {len(cols_to_drop)} leakage features"

        _tasks[task_id]["progress"] = 0.3

        # Prepare features and target
        df = df.dropna(subset=[request.target_column])

        id_cols = ['PATNO', 'EVENT_ID']
        feature_cols = [c for c in df.columns if c not in [request.target_column] + id_cols]

        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df[request.target_column]

        # Encode target if categorical
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        _tasks[task_id]["progress"] = 0.5

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=request.test_size, random_state=42, stratify=y_encoded
        )

        try:
            from pie.feature_selector import FeatureSelector

            _tasks[task_id]["message"] = f"Applying {request.method} feature selection..."

            selector = FeatureSelector(
                method=request.method,
                task_type='classification',
                k_or_frac=request.param_value if request.method == 'k_best' else None,
                alpha_fdr=request.param_value if request.method == 'fdr' else 0.05
            )

            selector.fit(X_train.fillna(0), le.fit_transform(y_train))
            X_train_selected = selector.transform(X_train.fillna(0))
            X_test_selected = selector.transform(X_test.fillna(0))

            selected_features = selector.selected_feature_names_

        except ImportError:
            # Fallback: variance threshold
            from sklearn.feature_selection import VarianceThreshold

            selector = VarianceThreshold(threshold=0.01)
            X_train_selected = pd.DataFrame(
                selector.fit_transform(X_train.fillna(0)),
                columns=X_train.columns[selector.get_support()]
            )
            X_test_selected = pd.DataFrame(
                selector.transform(X_test.fillna(0)),
                columns=X_train.columns[selector.get_support()]
            )
            selected_features = list(X_train_selected.columns)

        _tasks[task_id]["progress"] = 0.8

        # Combine with target
        train_df = pd.concat([X_train_selected.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([X_test_selected.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        # Cache results to disk
        train_cache_key = f"train_{task_id}"
        test_cache_key = f"test_{task_id}"
        train_shape = list(train_df.shape)
        test_shape = list(test_df.shape)
        disk_cache.store(train_cache_key, train_df)
        disk_cache.store(test_cache_key, test_df)
        del train_df, test_df
        gc.collect()

        _tasks[task_id]["status"] = "completed"
        _tasks[task_id]["progress"] = 1.0
        _tasks[task_id]["message"] = "Feature selection completed"
        _tasks[task_id]["result"] = {
            "train_cache_key": train_cache_key,
            "test_cache_key": test_cache_key,
            "original_features": len(feature_cols),
            "selected_features": len(selected_features),
            "selected_feature_names": selected_features[:50],  # Limit for response size
            "train_shape": train_shape,
            "test_shape": test_shape
        }

    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)


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
        _tasks[task_id]["status"] = "running"
        _tasks[task_id]["progress"] = 0.1
        _tasks[task_id]["message"] = "Loading training data..."

        if not disk_cache.exists(request.train_cache_key) or not disk_cache.exists(request.test_cache_key):
            raise ValueError("Training or test data not found in cache")

        train_data = disk_cache.load(request.train_cache_key)
        test_data = disk_cache.load(request.test_cache_key)

        try:
            from pie.classifier import Classifier

            _tasks[task_id]["progress"] = 0.2
            _tasks[task_id]["message"] = "Setting up classifier..."

            classifier = Classifier()
            classifier.setup_experiment(
                data=train_data,
                target=request.target_column,
                test_data=test_data,
                session_id=42,
                verbose=False
            )

            _tasks[task_id]["progress"] = 0.4
            _tasks[task_id]["message"] = "Comparing models..."

            best_model = classifier.compare_models(
                n_select=1,
                budget_time=request.time_budget_minutes,
                verbose=False
            )

            comparison_results = classifier.comparison_results

            if request.tune_best:
                _tasks[task_id]["progress"] = 0.7
                _tasks[task_id]["message"] = "Tuning best model..."
                best_model = classifier.tune_model(verbose=False)

            _tasks[task_id]["progress"] = 0.9
            _tasks[task_id]["message"] = "Generating predictions..."

            predictions = classifier.predict_model(data=test_data, verbose=False)

            # Store the model
            model_id = f"model_{task_id}"
            _models[model_id] = {
                "classifier": classifier,
                "model": best_model,
                "comparison_results": comparison_results.to_dict() if comparison_results is not None else None
            }

            _tasks[task_id]["status"] = "completed"
            _tasks[task_id]["progress"] = 1.0
            _tasks[task_id]["message"] = "Model training completed"
            _tasks[task_id]["result"] = {
                "model_id": model_id,
                "model_name": type(best_model).__name__,
                "comparison_results": comparison_results.head(10).to_dict() if comparison_results is not None else None,
                "test_predictions_shape": list(predictions.shape)
            }

        except ImportError:
            _tasks[task_id]["status"] = "failed"
            _tasks[task_id]["error"] = "PyCaret/PIE Classifier not available"

    except Exception as e:
        _tasks[task_id]["status"] = "failed"
        _tasks[task_id]["error"] = str(e)


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
    """Get the status of an analysis task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return _tasks[task_id]


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
