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


class ScatterRequest(BaseModel):
    """Request for a paired-variable scatter analysis."""
    cache_key: str
    x_variable: str
    y_variable: str
    method: str = "pearson"  # pearson, spearman, kendall
    max_points: int = 2000   # downsample cap for the returned point cloud


def _load_for_stats(cache_key: str, columns: List[str]) -> pd.DataFrame:
    """Load only the needed columns, using modular column projection when available."""
    if disk_cache.is_modular(cache_key):
        return disk_cache.load_columns(cache_key, columns)
    # Legacy single-file path
    return disk_cache.load(cache_key)


def _to_jsonable(obj):
    """Recursively coerce numpy scalars to Python natives for FastAPI.

    Specifically: NumPy 2.x dropped ``np.bool_`` as a Python ``bool`` subclass,
    so FastAPI's ``jsonable_encoder`` blows up with
    ``TypeError: 'numpy.bool' object is not iterable`` whenever a comparison
    result like ``p_value < 0.05`` reaches the response. We also coerce
    numeric numpy scalars and walk dicts/lists/tuples for safety.
    """
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if not (np.isnan(v) or np.isinf(v)) else None
    if isinstance(obj, float):
        # pie.stats returns Python floats; NaN/Inf are not JSON-compliant.
        return obj if not (obj != obj or obj in (float("inf"), float("-inf"))) else None
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {_to_jsonable(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


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
                "significant": bool(p_value < 0.05),
                "group_statistics": _to_jsonable(group_stats),
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
                "significant": bool(p_value < 0.05),
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
                "significant": bool(p_value < 0.05),
                "contingency_table": _to_jsonable(contingency.to_dict()),
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
            "significant": bool(p_value < 0.05),
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
            "significant": bool(p_value < 0.05),
            "n_groups": len(groups),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scatter")
async def run_scatter_correlation(request: ScatterRequest):
    """
    Compute correlation between two numeric variables AND return the (x, y)
    points + best-fit line so the frontend can plot a scatter with regression
    overlay. Points are downsampled to ``max_points`` to keep payloads light.
    """
    if not disk_cache.exists(request.cache_key) and not disk_cache.is_modular(request.cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {request.cache_key}")

    data = _load_for_stats(request.cache_key, [request.x_variable, request.y_variable])
    if not isinstance(data, pd.DataFrame):
        raise HTTPException(status_code=400, detail="Cached data is not a DataFrame")

    if request.x_variable not in data.columns:
        raise HTTPException(status_code=400, detail=f"X variable not found: {request.x_variable}")
    if request.y_variable not in data.columns:
        raise HTTPException(status_code=400, detail=f"Y variable not found: {request.y_variable}")

    aligned = data[[request.x_variable, request.y_variable]].apply(pd.to_numeric, errors="coerce").dropna()
    n = int(len(aligned))
    if n < 3:
        raise HTTPException(status_code=400, detail=f"Need at least 3 paired observations, got {n}")

    try:
        from scipy import stats

        x = aligned[request.x_variable].to_numpy(dtype=float)
        y = aligned[request.y_variable].to_numpy(dtype=float)

        if request.method == "pearson":
            r, p_value = stats.pearsonr(x, y)
            method_name = "Pearson"
        elif request.method == "spearman":
            r, p_value = stats.spearmanr(x, y)
            method_name = "Spearman"
        elif request.method == "kendall":
            r, p_value = stats.kendalltau(x, y)
            method_name = "Kendall"
        else:
            raise HTTPException(status_code=400, detail=f"Unknown method: {request.method}")

        # OLS regression line (always Pearson-style fit, regardless of method)
        slope_obj = stats.linregress(x, y)
        slope = float(slope_obj.slope)
        intercept = float(slope_obj.intercept)
        r_squared = float(slope_obj.rvalue ** 2)

        # Downsample point cloud for the chart
        max_points = max(50, min(int(request.max_points), 10_000))
        if n > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(n, size=max_points, replace=False)
            x_sample = x[idx]
            y_sample = y[idx]
        else:
            x_sample = x
            y_sample = y

        # Endpoints for the regression line on the downsampled domain
        x_min, x_max = float(x.min()), float(x.max())
        line_endpoints = [
            {"x": x_min, "y": intercept + slope * x_min},
            {"x": x_max, "y": intercept + slope * x_max},
        ]

        return _to_jsonable({
            "test_name": f"{method_name} Correlation",
            "method": request.method,
            "x_variable": request.x_variable,
            "y_variable": request.y_variable,
            "n": n,
            "n_plotted": int(len(x_sample)),
            "correlation": float(r),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "regression": {
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_squared,
                "endpoints": line_endpoints,
            },
            "points": [{"x": float(xv), "y": float(yv)} for xv, yv in zip(x_sample, y_sample)],
            "interpretation": _interpret_correlation(float(r), float(p_value)),
        })

    except HTTPException:
        raise
    except ImportError:
        raise HTTPException(status_code=500, detail="scipy not available")
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
                T_g = T[mask]
                E_g = E[mask]
                kmf = KaplanMeierFitter()
                kmf.fit(T_g, E_g, label=str(group_val))

                n_subjects = int(mask.sum())
                n_events = int(E_g.fillna(0).astype(bool).sum())
                results["curves"].append({
                    "group": str(group_val),
                    "timeline": kmf.survival_function_.index.tolist(),
                    "survival": kmf.survival_function_.iloc[:, 0].tolist(),
                    "median_survival": float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else None,
                    "n_subjects": n_subjects,
                    "n_events": n_events,
                    "n_censored": n_subjects - n_events,
                    "follow_up_max": float(T_g.max()) if len(T_g) else None,
                })

            # Log-rank test
            if len(unique_groups) == 2:
                group_a = groups == unique_groups[0]
                group_b = groups == unique_groups[1]
                lr_result = logrank_test(T[group_a], T[group_b], E[group_a], E[group_b])
                results["statistics"]["logrank"] = {
                    "test_statistic": float(lr_result.test_statistic),
                    "p_value": float(lr_result.p_value),
                    "significant": bool(lr_result.p_value < 0.05),
                }
        else:
            # Overall survival
            kmf = KaplanMeierFitter()
            kmf.fit(T, E, label="Overall")

            n_subjects = int(len(T))
            n_events = int(E.fillna(0).astype(bool).sum())
            results["curves"].append({
                "group": "Overall",
                "timeline": kmf.survival_function_.index.tolist(),
                "survival": kmf.survival_function_.iloc[:, 0].tolist(),
                "median_survival": float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else None,
                "n_subjects": n_subjects,
                "n_events": n_events,
                "n_censored": n_subjects - n_events,
                "follow_up_max": float(T.max()) if len(T) else None,
            })

        return _to_jsonable(results)

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
                "top_values": _to_jsonable(value_counts),
                "missing": int(col.isnull().sum()),
                "missing_pct": round(col.isnull().sum() / len(col) * 100, 2) if len(col) > 0 else 0
            }

    return {"statistics": _to_jsonable(results)}


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


# ===========================================================================
# Classical statistics suite — thin adapters over pie.stats
# ===========================================================================

def _load_df(cache_key: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load cached data as a DataFrame, optionally restricted to ``columns``.

    404s on unknown cache; raises HTTPException instead of silently returning
    an empty frame so the UI can show a proper error.
    """
    if not disk_cache.exists(cache_key) and not disk_cache.is_modular(cache_key):
        raise HTTPException(status_code=404, detail=f"Data not found: {cache_key}")
    if columns and disk_cache.is_modular(cache_key):
        return disk_cache.load_columns(cache_key, columns)
    return disk_cache.load(cache_key)


# -------- Describe ---------------------------------------------------------

class DescribeRequest(BaseModel):
    cache_key: str
    variables: List[str]


class NormalityRequest(BaseModel):
    cache_key: str
    variable: str
    test: str = "shapiro"


@router.post("/describe/summary")
async def describe_summary(request: DescribeRequest):
    from pie.stats import summary_statistics
    df = _load_df(request.cache_key, request.variables)
    try:
        return _to_jsonable(summary_statistics(df, request.variables))
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/describe/normality")
async def describe_normality(request: NormalityRequest):
    from pie.stats import normality_test
    df = _load_df(request.cache_key, [request.variable])
    if request.variable not in df.columns:
        raise HTTPException(status_code=404, detail=f"column {request.variable!r} not found")
    try:
        return _to_jsonable(normality_test(df[request.variable], test=request.test))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/describe/missingness")
async def describe_missingness(request: DescribeRequest):
    from pie.stats import missingness_report
    df = _load_df(request.cache_key, request.variables if request.variables else None)
    return _to_jsonable(missingness_report(df, request.variables or None))


# -------- Compare ----------------------------------------------------------

class TwoGroupRequest(BaseModel):
    cache_key: str
    variable: str
    grouping_variable: str
    test: str = "auto"


class MultiGroupRequest(BaseModel):
    cache_key: str
    variable: str
    grouping_variable: str
    test: str = "auto"
    posthoc: Optional[str] = None


class CategoricalRequest(BaseModel):
    cache_key: str
    variable_a: str
    variable_b: str
    test: str = "auto"


@router.post("/compare/two_group")
async def compare_two_group(request: TwoGroupRequest):
    from pie.stats import compare as C
    df = _load_df(request.cache_key, [request.variable, request.grouping_variable])
    levels = df[request.grouping_variable].dropna().unique()
    if len(levels) != 2:
        raise HTTPException(status_code=400, detail=f"need exactly 2 levels, got {len(levels)}")
    a = df.loc[df[request.grouping_variable] == levels[0], request.variable]
    b = df.loc[df[request.grouping_variable] == levels[1], request.variable]
    table = {
        "auto": C.independent_ttest, "independent_t": C.independent_ttest,
        "welch_t": C.welch_ttest, "paired_t": C.paired_ttest,
        "mann_whitney": C.mann_whitney, "wilcoxon": C.wilcoxon_signed_rank,
    }
    fn = table.get(request.test)
    if fn is None:
        raise HTTPException(status_code=400, detail=f"unknown test {request.test}")
    try:
        result = fn(a.values, b.values)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result["group_labels"] = [str(levels[0]), str(levels[1])]
    return _to_jsonable(result)


@router.post("/compare/multi_group")
async def compare_multi_group(request: MultiGroupRequest):
    from pie.stats import compare as C
    df = _load_df(request.cache_key, [request.variable, request.grouping_variable])
    groups = {str(k): v[request.variable].values for k, v in df.groupby(request.grouping_variable)}
    if len(groups) < 2:
        raise HTTPException(status_code=400, detail="need at least 2 groups")
    main = C.one_way_anova(groups) if request.test in ("auto", "anova") else C.kruskal_wallis(groups)
    posthoc_result = None
    if request.posthoc == "tukey":
        posthoc_result = C.tukey_hsd(groups)
    elif request.posthoc == "dunn":
        posthoc_result = C.dunn_posthoc(groups)
    return _to_jsonable({"main": main, "posthoc": posthoc_result})


@router.post("/compare/categorical")
async def compare_categorical(request: CategoricalRequest):
    from pie.stats import compare as C
    df = _load_df(request.cache_key, [request.variable_a, request.variable_b])
    ct = pd.crosstab(df[request.variable_a], df[request.variable_b])
    base = {
        "contingency": ct.values.tolist(),
        "row_labels": [str(x) for x in ct.index],
        "col_labels": [str(x) for x in ct.columns],
    }
    if request.test == "fisher" or (
        request.test == "auto" and ct.shape == (2, 2) and (ct < 5).any().any()
    ):
        base.update(C.fisher_exact(ct.values.tolist()))
    else:
        base.update(C.chi_square(ct.values.tolist()))
    return _to_jsonable(base)


# -------- Correlate --------------------------------------------------------

class PartialCorrRequest(BaseModel):
    cache_key: str
    x: str
    y: str
    covariates: List[str]
    method: str = "pearson"


class MatrixCorrRequest(BaseModel):
    cache_key: str
    variables: List[str]
    method: str = "pearson"


@router.post("/correlate/partial")
async def correlate_partial(request: PartialCorrRequest):
    from pie.stats import partial_correlation
    df = _load_df(request.cache_key, [request.x, request.y, *request.covariates])
    return _to_jsonable(partial_correlation(df, request.x, request.y,
                                            request.covariates, method=request.method))


@router.post("/correlate/matrix")
async def correlate_matrix(request: MatrixCorrRequest):
    from pie.stats import correlation_matrix
    df = _load_df(request.cache_key, request.variables)
    return _to_jsonable(correlation_matrix(df, request.variables, method=request.method))


# -------- Regress ----------------------------------------------------------

class LinearRegRequest(BaseModel):
    cache_key: str
    outcome: str
    predictors: List[str]
    standardize: bool = False


class LogisticRegRequest(BaseModel):
    cache_key: str
    outcome: str
    predictors: List[str]


class ANCOVARequest(BaseModel):
    cache_key: str
    outcome: str
    group: str
    covariates: List[str]


@router.post("/regress/linear")
async def regress_linear(request: LinearRegRequest):
    from pie.stats import linear_regression
    df = _load_df(request.cache_key, [request.outcome, *request.predictors])
    try:
        return _to_jsonable(linear_regression(df, request.outcome, request.predictors,
                                              standardize=request.standardize))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/regress/logistic")
async def regress_logistic(request: LogisticRegRequest):
    from pie.stats import logistic_regression
    df = _load_df(request.cache_key, [request.outcome, *request.predictors])
    try:
        return _to_jsonable(logistic_regression(df, request.outcome, request.predictors))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/regress/ancova")
async def regress_ancova(request: ANCOVARequest):
    from pie.stats import ancova
    df = _load_df(request.cache_key, [request.outcome, request.group, *request.covariates])
    try:
        return _to_jsonable(ancova(df, request.outcome, request.group, request.covariates))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# -------- Longitudinal -----------------------------------------------------

class LMMRequest(BaseModel):
    cache_key: str
    outcome: str
    fixed_effects: List[str]
    group: str
    random_slopes: Optional[List[str]] = None


class ChangeRequest(BaseModel):
    cache_key: str
    subject: str
    time: str
    outcome: str
    baseline_time: Any = 0


@router.post("/longitudinal/mixed_model")
async def longitudinal_lmm(request: LMMRequest):
    from pie.stats import linear_mixed_model
    cols = [request.outcome, request.group, *request.fixed_effects]
    if request.random_slopes:
        cols.extend(request.random_slopes)
    df = _load_df(request.cache_key, cols)
    try:
        return _to_jsonable(linear_mixed_model(
            df, request.outcome, request.fixed_effects, request.group, request.random_slopes
        ))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/longitudinal/change_from_baseline")
async def longitudinal_change(request: ChangeRequest):
    from pie.stats import change_from_baseline
    df = _load_df(request.cache_key, [request.subject, request.time, request.outcome])
    try:
        return _to_jsonable(change_from_baseline(
            df, request.subject, request.time, request.outcome, request.baseline_time
        ))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# -------- Survive (new endpoints — old /survival remains for back-compat) --

class SurviveKMRequest(BaseModel):
    cache_key: str
    time: str
    event: str
    group: Optional[str] = None


class LogRankRequest(BaseModel):
    cache_key: str
    time: str
    event: str
    group: str


class CoxRequest(BaseModel):
    cache_key: str
    time: str
    event: str
    covariates: List[str]


@router.post("/survive/km")
async def survive_km(request: SurviveKMRequest):
    from pie.stats import kaplan_meier
    cols = [request.time, request.event] + ([request.group] if request.group else [])
    df = _load_df(request.cache_key, cols)
    return _to_jsonable(kaplan_meier(df, request.time, request.event, request.group))


@router.post("/survive/logrank")
async def survive_logrank(request: LogRankRequest):
    from pie.stats import logrank_test
    df = _load_df(request.cache_key, [request.time, request.event, request.group])
    return _to_jsonable(logrank_test(df, request.time, request.event, request.group))


@router.post("/survive/cox")
async def survive_cox(request: CoxRequest):
    from pie.stats import cox_regression
    df = _load_df(request.cache_key, [request.time, request.event, *request.covariates])
    try:
        return _to_jsonable(cox_regression(df, request.time, request.event, request.covariates))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# -------- Multitest --------------------------------------------------------

class MultitestRequest(BaseModel):
    p_values: List[float]
    method: str = "fdr_bh"
    alpha: float = 0.05


@router.post("/multitest/adjust")
async def multitest_adjust(request: MultitestRequest):
    from pie.stats import adjust_pvalues
    try:
        return _to_jsonable(adjust_pvalues(request.p_values, method=request.method,
                                           alpha=request.alpha))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# -------- PD helpers -------------------------------------------------------

class LEDDRequest(BaseModel):
    doses_mg: Dict[str, float]


class UPDRSRequest(BaseModel):
    cache_key: str
    part1_cols: Optional[List[str]] = None
    part2_cols: Optional[List[str]] = None
    part3_cols: Optional[List[str]] = None
    part4_cols: Optional[List[str]] = None


class HoehnYahrRequest(BaseModel):
    cache_key: str
    variable: str


@router.post("/pd/ledd")
async def pd_ledd(request: LEDDRequest):
    from pie.stats import compute_ledd
    return _to_jsonable(compute_ledd(request.doses_mg))


@router.get("/pd/ledd_factors")
async def pd_ledd_factors():
    """Return the supported drug→factor map so the UI can render a dropdown."""
    from pie.stats.pd_helpers import LEDD_FACTORS
    return {"factors": LEDD_FACTORS}


@router.post("/pd/updrs")
async def pd_updrs(request: UPDRSRequest):
    from pie.stats import aggregate_updrs
    parts = [c for c in (request.part1_cols, request.part2_cols,
                         request.part3_cols, request.part4_cols) if c]
    all_cols = [c for grp in parts for c in grp]
    df = _load_df(request.cache_key, all_cols)
    result = aggregate_updrs(
        df,
        part1_cols=request.part1_cols,
        part2_cols=request.part2_cols,
        part3_cols=request.part3_cols,
        part4_cols=request.part4_cols,
    )
    # Return a head preview + summary stats so the UI doesn't receive a huge payload.
    preview = result.head(20).to_dict(orient="records")
    summary = result.describe().to_dict() if len(result) else {}
    return _to_jsonable({
        "n_rows": int(len(result)),
        "columns": list(result.columns),
        "preview": preview,
        "summary": summary,
    })


@router.post("/pd/hoehn_yahr")
async def pd_hoehn_yahr(request: HoehnYahrRequest):
    from pie.stats import hoehn_yahr_summary
    df = _load_df(request.cache_key, [request.variable])
    return _to_jsonable(hoehn_yahr_summary(df[request.variable]))
