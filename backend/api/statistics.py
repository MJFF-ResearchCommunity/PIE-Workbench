"""
Statistical Analysis API endpoints.

Provides traditional statistical tests and survival analysis.
"""

import gc
import os
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np

from . import cache as disk_cache

router = APIRouter()


class StatTestRequest(BaseModel):
    """Request for statistical test."""
    cache_key: str
    x_variable: str
    y_variable: str
    grouping_variable: Optional[str] = None


class SurvivalAnalysisRequest(BaseModel):
    """Request for survival analysis."""
    cache_key: str
    time_variable: str
    event_variable: str
    grouping_variable: Optional[str] = None


class CorrelationRequest(BaseModel):
    """Request for correlation analysis."""
    cache_key: str
    variables: List[str]
    method: str = "pearson"  # pearson, spearman, kendall


def _load_for_stats(cache_key: str, columns: List[str]) -> pd.DataFrame:
    """Load only the needed columns, using modular column projection when available."""
    if disk_cache.is_modular(cache_key):
        return disk_cache.load_columns(cache_key, columns)
    # Legacy single-file path
    return disk_cache.load(cache_key)


@router.post("/auto_test")
async def auto_statistical_test(request: StatTestRequest):
    """
    Automatically select and run the appropriate statistical test
    based on variable types.
    """
    if not disk_cache.exists(request.cache_key) and not disk_cache.is_modular(request.cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {request.cache_key}")

    data = _load_for_stats(request.cache_key, [request.x_variable, request.y_variable])
    if not isinstance(data, pd.DataFrame):
        raise HTTPException(status_code=400, detail="Cached data is not a DataFrame")

    if request.y_variable not in data.columns:
        raise HTTPException(status_code=400, detail=f"Y variable not found: {request.y_variable}")
    if request.x_variable not in data.columns:
        raise HTTPException(status_code=400, detail=f"X variable not found: {request.x_variable}")

    # Determine variable types
    y_col = data[request.y_variable].dropna()
    x_col = data[request.x_variable].dropna()

    y_is_numeric = np.issubdtype(y_col.dtype, np.number)
    x_is_numeric = np.issubdtype(x_col.dtype, np.number)
    x_unique = x_col.nunique()

    try:
        from scipy import stats

        # Determine and run appropriate test
        if y_is_numeric and not x_is_numeric:
            # Continuous Y vs Categorical X
            groups = [group[request.y_variable].dropna().values
                     for name, group in data.groupby(request.x_variable)]

            if x_unique == 2:
                # T-test for 2 groups
                stat, p_value = stats.ttest_ind(groups[0], groups[1])
                test_name = "Independent T-Test"
                description = f"Comparing {request.y_variable} between 2 groups of {request.x_variable}"
            else:
                # ANOVA for >2 groups
                stat, p_value = stats.f_oneway(*groups)
                test_name = "One-Way ANOVA"
                description = f"Comparing {request.y_variable} across {x_unique} groups of {request.x_variable}"

            # Calculate group statistics
            group_stats = data.groupby(request.x_variable)[request.y_variable].agg(['mean', 'std', 'count']).to_dict()

            return {
                "test_name": test_name,
                "description": description,
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "group_statistics": group_stats,
                "interpretation": _interpret_p_value(p_value, test_name)
            }

        elif y_is_numeric and x_is_numeric:
            # Continuous Y vs Continuous X -> Correlation
            # Use aligned data
            aligned = data[[request.x_variable, request.y_variable]].dropna()

            stat, p_value = stats.pearsonr(aligned[request.x_variable], aligned[request.y_variable])
            test_name = "Pearson Correlation"
            description = f"Correlation between {request.x_variable} and {request.y_variable}"

            return {
                "test_name": test_name,
                "description": description,
                "correlation": float(stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "interpretation": _interpret_correlation(stat, p_value)
            }

        elif not y_is_numeric and not x_is_numeric:
            # Categorical Y vs Categorical X -> Chi-Square
            contingency = pd.crosstab(data[request.x_variable], data[request.y_variable])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

            test_name = "Chi-Square Test"
            description = f"Association between {request.x_variable} and {request.y_variable}"

            return {
                "test_name": test_name,
                "description": description,
                "chi2_statistic": float(chi2),
                "p_value": float(p_value),
                "degrees_of_freedom": int(dof),
                "significant": p_value < 0.05,
                "contingency_table": contingency.to_dict(),
                "interpretation": _interpret_p_value(p_value, test_name)
            }

        else:
            raise HTTPException(
                status_code=400,
                detail="Cannot determine appropriate test for this variable combination"
            )

    except ImportError:
        raise HTTPException(status_code=500, detail="scipy not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ttest")
async def run_ttest(cache_key: str, variable: str, grouping_variable: str):
    """Run independent samples t-test."""
    if not disk_cache.exists(cache_key) and not disk_cache.is_modular(cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {cache_key}")

    data = _load_for_stats(cache_key, [variable, grouping_variable])

    try:
        from scipy import stats

        groups = data.groupby(grouping_variable)[variable].apply(list).values
        if len(groups) != 2:
            raise HTTPException(status_code=400, detail="T-test requires exactly 2 groups")

        stat, p_value = stats.ttest_ind(groups[0], groups[1])

        return {
            "test_name": "Independent T-Test",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/anova")
async def run_anova(cache_key: str, variable: str, grouping_variable: str):
    """Run one-way ANOVA."""
    if not disk_cache.exists(cache_key) and not disk_cache.is_modular(cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {cache_key}")

    data = _load_for_stats(cache_key, [variable, grouping_variable])

    try:
        from scipy import stats

        groups = [group[variable].dropna().values
                 for name, group in data.groupby(grouping_variable)]

        stat, p_value = stats.f_oneway(*groups)

        return {
            "test_name": "One-Way ANOVA",
            "f_statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "n_groups": len(groups)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/correlation")
async def run_correlation(request: CorrelationRequest):
    """Calculate correlation matrix."""
    if not disk_cache.exists(request.cache_key) and not disk_cache.is_modular(request.cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {request.cache_key}")

    data = _load_for_stats(request.cache_key, request.variables)

    try:
        subset = data[request.variables].select_dtypes(include=[np.number])

        if request.method == "pearson":
            corr_matrix = subset.corr(method='pearson')
        elif request.method == "spearman":
            corr_matrix = subset.corr(method='spearman')
        elif request.method == "kendall":
            corr_matrix = subset.corr(method='kendall')
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")

        return {
            "method": request.method,
            "variables": list(corr_matrix.columns),
            "matrix": corr_matrix.values.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/survival")
async def run_survival_analysis(request: SurvivalAnalysisRequest):
    """Run Kaplan-Meier survival analysis."""
    if not disk_cache.exists(request.cache_key) and not disk_cache.is_modular(request.cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {request.cache_key}")

    cols = [request.time_variable, request.event_variable]
    if request.grouping_variable:
        cols.append(request.grouping_variable)
    data = _load_for_stats(request.cache_key, cols)

    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test

        # Prepare data
        T = data[request.time_variable].dropna()
        E = data[request.event_variable].dropna()

        # Align data
        aligned_idx = T.index.intersection(E.index)
        T = T.loc[aligned_idx]
        E = E.loc[aligned_idx]

        results = {"curves": [], "statistics": {}}

        if request.grouping_variable:
            # Stratified analysis
            groups = data.loc[aligned_idx, request.grouping_variable].dropna()
            aligned_idx = aligned_idx.intersection(groups.index)
            T = T.loc[aligned_idx]
            E = E.loc[aligned_idx]
            groups = groups.loc[aligned_idx]

            unique_groups = groups.unique()

            for group_val in unique_groups:
                mask = groups == group_val
                kmf = KaplanMeierFitter()
                kmf.fit(T[mask], E[mask], label=str(group_val))

                results["curves"].append({
                    "group": str(group_val),
                    "timeline": kmf.survival_function_.index.tolist(),
                    "survival": kmf.survival_function_.iloc[:, 0].tolist(),
                    "median_survival": float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else None
                })

            # Log-rank test
            if len(unique_groups) == 2:
                group_a = groups == unique_groups[0]
                group_b = groups == unique_groups[1]
                lr_result = logrank_test(T[group_a], T[group_b], E[group_a], E[group_b])
                results["statistics"]["logrank"] = {
                    "test_statistic": float(lr_result.test_statistic),
                    "p_value": float(lr_result.p_value),
                    "significant": lr_result.p_value < 0.05
                }
        else:
            # Overall survival
            kmf = KaplanMeierFitter()
            kmf.fit(T, E, label="Overall")

            results["curves"].append({
                "group": "Overall",
                "timeline": kmf.survival_function_.index.tolist(),
                "survival": kmf.survival_function_.iloc[:, 0].tolist(),
                "median_survival": float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else None
            })

        return results

    except ImportError:
        raise HTTPException(status_code=500, detail="lifelines library not available")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/descriptive")
async def get_descriptive_statistics(cache_key: str, variables: List[str]):
    """Get descriptive statistics for specified variables."""
    if not disk_cache.exists(cache_key) and not disk_cache.is_modular(cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {cache_key}")

    data = _load_for_stats(cache_key, variables)

    results = {}
    for var in variables:
        if var not in data.columns:
            continue

        col = data[var]
        is_numeric = np.issubdtype(col.dtype, np.number)

        if is_numeric:
            results[var] = {
                "type": "numeric",
                "count": int(col.count()),
                "mean": float(col.mean()) if not col.empty else None,
                "std": float(col.std()) if not col.empty else None,
                "min": float(col.min()) if not col.empty else None,
                "max": float(col.max()) if not col.empty else None,
                "median": float(col.median()) if not col.empty else None,
                "q25": float(col.quantile(0.25)) if not col.empty else None,
                "q75": float(col.quantile(0.75)) if not col.empty else None,
                "missing": int(col.isnull().sum()),
                "missing_pct": round(col.isnull().sum() / len(col) * 100, 2) if len(col) > 0 else 0
            }
        else:
            value_counts = col.value_counts().head(10).to_dict()
            results[var] = {
                "type": "categorical",
                "count": int(col.count()),
                "unique": int(col.nunique()),
                "top_values": value_counts,
                "missing": int(col.isnull().sum()),
                "missing_pct": round(col.isnull().sum() / len(col) * 100, 2) if len(col) > 0 else 0
            }

    return {"statistics": results}


def _interpret_p_value(p_value: float, test_name: str) -> str:
    """Generate interpretation text for p-value."""
    if p_value < 0.001:
        strength = "very strong"
    elif p_value < 0.01:
        strength = "strong"
    elif p_value < 0.05:
        strength = "moderate"
    elif p_value < 0.1:
        strength = "weak"
    else:
        strength = "no"

    if p_value < 0.05:
        return f"The {test_name} shows {strength} evidence (p = {p_value:.4f}) of a statistically significant difference."
    else:
        return f"The {test_name} shows {strength} evidence (p = {p_value:.4f}) of a statistically significant difference. The null hypothesis cannot be rejected."


def _interpret_correlation(r: float, p_value: float) -> str:
    """Generate interpretation text for correlation."""
    abs_r = abs(r)
    direction = "positive" if r > 0 else "negative"

    if abs_r >= 0.8:
        strength = "very strong"
    elif abs_r >= 0.6:
        strength = "strong"
    elif abs_r >= 0.4:
        strength = "moderate"
    elif abs_r >= 0.2:
        strength = "weak"
    else:
        strength = "very weak"

    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

    return f"There is a {strength} {direction} correlation (r = {r:.4f}), which is {significance} (p = {p_value:.4f})."
