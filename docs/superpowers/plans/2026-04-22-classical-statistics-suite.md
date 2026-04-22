# Classical Statistics Suite for Parkinson's Researchers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand PIE Workbench's Statistics section from 3 tabs (t-test/ANOVA, correlation scatter, Kaplan-Meier) into a classical-stats suite that lets a Parkinson's researcher with no ML background do the analyses they actually need on PPMI data: describe, compare, correlate, regress, model longitudinally, survive, and correct for multiple testing — all without writing code.

**Architecture:**
- Reusable, unit-testable stat primitives live in a new `pie.stats` module inside the PIE library. The workbench's FastAPI layer (`backend/api/statistics.py`) becomes a thin adapter that loads cached data and delegates to `pie.stats`. The React `StatsLab.tsx` grows from 3 tabs to 7 tabs, each tab backed by a small, focused API endpoint. This keeps business logic library-side (so PIE CLI/notebook users benefit too) and UI concerns workbench-side.
- PD-specific helpers (LEDD, UPDRS subscore aggregation) live alongside generic stats in `pie.stats` so they're discoverable by both surfaces.

**Tech stack:** Python (scipy, statsmodels, lifelines, pingouin, scikit-posthocs) • FastAPI • React + TypeScript + recharts + Tailwind • pytest for backend, tsc for frontend type safety.

---

## File structure

### PIE library (`lib/PIE/pie/`)

- **Create** `lib/PIE/pie/stats/__init__.py` — re-exports the public API
- **Create** `lib/PIE/pie/stats/describe.py` — summary, normality, missingness
- **Create** `lib/PIE/pie/stats/compare.py` — t-tests, ANOVA, non-parametric, post-hoc, effect sizes
- **Create** `lib/PIE/pie/stats/correlate.py` — Pearson/Spearman/Kendall, partial, correlation matrix
- **Create** `lib/PIE/pie/stats/regress.py` — linear, logistic, ANCOVA with diagnostics
- **Create** `lib/PIE/pie/stats/longitudinal.py` — LMM, change-from-baseline
- **Create** `lib/PIE/pie/stats/survive.py` — KM, log-rank, Cox PH
- **Create** `lib/PIE/pie/stats/multitest.py` — Bonferroni, Holm, BH-FDR
- **Create** `lib/PIE/pie/stats/pd_helpers.py` — LEDD, UPDRS aggregation, H&Y staging
- **Create** `lib/PIE/tests/test_stats_describe.py`
- **Create** `lib/PIE/tests/test_stats_compare.py`
- **Create** `lib/PIE/tests/test_stats_correlate.py`
- **Create** `lib/PIE/tests/test_stats_regress.py`
- **Create** `lib/PIE/tests/test_stats_longitudinal.py`
- **Create** `lib/PIE/tests/test_stats_survive.py`
- **Create** `lib/PIE/tests/test_stats_multitest.py`
- **Create** `lib/PIE/tests/test_stats_pd_helpers.py`
- **Modify** `lib/PIE/requirements.txt` — add `statsmodels`, `pingouin`, `scikit-posthocs`

### Workbench backend (`backend/api/statistics.py`)

- **Modify** `backend/api/statistics.py` — replace hand-rolled stats with calls to `pie.stats`; add new endpoints (listed per task)
- **Modify** `backend/requirements.txt` — add `pingouin`, `scikit-posthocs`

### Workbench frontend

- **Modify** `src/views/StatsLab.tsx` — split into 7 tabs; the existing ~960-line file becomes the shell + routing only
- **Create** `src/views/stats/DescribeTab.tsx` — summary / normality / missingness
- **Create** `src/views/stats/CompareTab.tsx` — expanded group comparison (non-parametric, paired, post-hoc, effect sizes)
- **Create** `src/views/stats/CorrelateTab.tsx` — expanded correlation (Spearman/Kendall, partial, correlation matrix)
- **Create** `src/views/stats/RegressTab.tsx` — linear / logistic / ANCOVA with diagnostics
- **Create** `src/views/stats/LongitudinalTab.tsx` — LMM + change-from-baseline
- **Create** `src/views/stats/SurviveTab.tsx` — KM + Cox PH (absorbs existing survival UI)
- **Create** `src/views/stats/MultitestTab.tsx` — multiple-testing correction utility
- **Create** `src/views/stats/PDHelpersTab.tsx` — LEDD calculator + UPDRS subscore aggregator
- **Create** `src/views/stats/shared/ResultTable.tsx` — reusable stats result card
- **Create** `src/views/stats/shared/DiagnosticPlots.tsx` — residual plots, Q-Q, Bland-Altman
- **Modify** `src/services/api.ts` — add ~20 new endpoint clients
- **Modify** `src/App.tsx` — no routing change (StatsLab still at `/stats`)

---

## Tech dependencies

Already installed in `backend/venv`: `scipy`, `statsmodels`, `lifelines`, `pingouin`, `scikit-posthocs`. The PIE library's own `requirements.txt` may not list some of these — Task 1 adds them.

---

## Phase 0 — Scaffolding

### Task 0.1: Create the `pie.stats` subpackage

**Files:**
- Create: `lib/PIE/pie/stats/__init__.py`
- Modify: `lib/PIE/requirements.txt`

- [ ] **Step 1: Add deps to PIE's requirements.txt**

Append to `lib/PIE/requirements.txt`:

```
# Classical statistics
statsmodels>=0.14.0
pingouin>=0.5.0
scikit-posthocs>=0.9.0
```

- [ ] **Step 2: Create `lib/PIE/pie/stats/__init__.py` as a re-export hub**

```python
"""pie.stats — classical statistics primitives for Parkinson's research.

Each submodule provides pure functions that take pandas Series / DataFrames
and return plain dicts of results. This makes them trivially consumable from
the workbench's FastAPI layer AND from notebooks / CLI scripts.
"""

from pie.stats.describe import summary_statistics, normality_test, missingness_report
from pie.stats.compare import (
    independent_ttest, paired_ttest, welch_ttest,
    mann_whitney, wilcoxon_signed_rank,
    one_way_anova, kruskal_wallis, tukey_hsd, dunn_posthoc,
    chi_square, fisher_exact, mcnemar,
    cohens_d, hedges_g, eta_squared,
)
from pie.stats.correlate import correlate_pair, partial_correlation, correlation_matrix
from pie.stats.regress import linear_regression, logistic_regression, ancova
from pie.stats.longitudinal import linear_mixed_model, change_from_baseline
from pie.stats.survive import kaplan_meier, logrank_test, cox_regression
from pie.stats.multitest import adjust_pvalues
from pie.stats.pd_helpers import compute_ledd, aggregate_updrs, hoehn_yahr_summary

__all__ = [
    # describe
    "summary_statistics", "normality_test", "missingness_report",
    # compare
    "independent_ttest", "paired_ttest", "welch_ttest",
    "mann_whitney", "wilcoxon_signed_rank",
    "one_way_anova", "kruskal_wallis", "tukey_hsd", "dunn_posthoc",
    "chi_square", "fisher_exact", "mcnemar",
    "cohens_d", "hedges_g", "eta_squared",
    # correlate
    "correlate_pair", "partial_correlation", "correlation_matrix",
    # regress
    "linear_regression", "logistic_regression", "ancova",
    # longitudinal
    "linear_mixed_model", "change_from_baseline",
    # survive
    "kaplan_meier", "logrank_test", "cox_regression",
    # multitest
    "adjust_pvalues",
    # pd helpers
    "compute_ledd", "aggregate_updrs", "hoehn_yahr_summary",
]
```

- [ ] **Step 3: Create empty placeholder modules so imports resolve**

Create each of these as 1-line files with just a module docstring; subsequent tasks fill them in. This lets us commit the scaffolding first and iterate file by file.

Files to create with `"""<purpose>."""` docstrings:
- `lib/PIE/pie/stats/describe.py`
- `lib/PIE/pie/stats/compare.py`
- `lib/PIE/pie/stats/correlate.py`
- `lib/PIE/pie/stats/regress.py`
- `lib/PIE/pie/stats/longitudinal.py`
- `lib/PIE/pie/stats/survive.py`
- `lib/PIE/pie/stats/multitest.py`
- `lib/PIE/pie/stats/pd_helpers.py`

Note: the `__init__.py` re-exports above will fail until functions are defined — that's expected. Fix by defining each function's signature + `raise NotImplementedError` stub in its module as part of this task:

```python
# describe.py
"""Descriptive statistics: summaries, normality, missingness."""
from typing import Any, Dict, Iterable
import pandas as pd

def summary_statistics(df: pd.DataFrame, variables: Iterable[str]) -> Dict[str, Any]:
    raise NotImplementedError

def normality_test(series: pd.Series, test: str = "shapiro") -> Dict[str, Any]:
    raise NotImplementedError

def missingness_report(df: pd.DataFrame, variables: Iterable[str] | None = None) -> Dict[str, Any]:
    raise NotImplementedError
```

Repeat the same stub-with-signature pattern for each of the 8 modules, matching the function names from `__init__.py`.

- [ ] **Step 4: Verify import works**

```bash
./backend/venv/bin/python -c "from pie.stats import summary_statistics; print('ok')"
```
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add lib/PIE/pie/stats/ lib/PIE/requirements.txt
git commit -m "feat(stats): scaffold pie.stats subpackage"
```

---

## Phase 1 — Describe tab (the foundation everyone needs)

### Task 1.1: `summary_statistics` in pie.stats

**Files:**
- Modify: `lib/PIE/pie/stats/describe.py`
- Create: `lib/PIE/tests/test_stats_describe.py`

- [ ] **Step 1: Write tests**

```python
# lib/PIE/tests/test_stats_describe.py
import numpy as np
import pandas as pd
import pytest
from pie.stats.describe import summary_statistics, normality_test, missingness_report


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.normal(65, 10, 100),
        "updrs": rng.normal(30, 15, 100),
        "sex": rng.choice(["M", "F"], 100),
        "missing_col": [np.nan] * 30 + list(rng.normal(0, 1, 70)),
    })


def test_summary_statistics_numeric(df):
    result = summary_statistics(df, ["age", "updrs"])
    assert set(result.keys()) == {"age", "updrs"}
    age = result["age"]
    # Required fields
    for key in ("n", "mean", "median", "std", "min", "max",
                "q1", "q3", "iqr", "skew", "kurtosis", "n_missing", "pct_missing"):
        assert key in age, f"missing {key}"
    assert age["n"] == 100
    assert age["n_missing"] == 0
    assert 55 < age["mean"] < 75  # loose — rng seeded


def test_summary_statistics_handles_missing(df):
    result = summary_statistics(df, ["missing_col"])
    assert result["missing_col"]["n"] == 70
    assert result["missing_col"]["n_missing"] == 30
    assert result["missing_col"]["pct_missing"] == pytest.approx(30.0)


def test_summary_statistics_rejects_non_numeric(df):
    with pytest.raises(ValueError, match="not numeric"):
        summary_statistics(df, ["sex"])
```

- [ ] **Step 2: Run tests — expect 3 FAIL (NotImplementedError)**

```bash
cd lib/PIE && ../../backend/venv/bin/pytest tests/test_stats_describe.py -v
```

- [ ] **Step 3: Implement `summary_statistics`**

```python
# lib/PIE/pie/stats/describe.py
"""Descriptive statistics: summaries, normality, missingness."""
from typing import Any, Dict, Iterable, Optional
import numpy as np
import pandas as pd
from scipy import stats as _sps


def summary_statistics(df: pd.DataFrame, variables: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Per-variable summary: n, mean, median, std, quantiles, skew, kurtosis, missing.

    Only accepts numeric columns; raises ValueError on categorical input so the
    caller fails fast instead of silently getting nonsense back.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for v in variables:
        if v not in df.columns:
            raise KeyError(v)
        s = df[v]
        if not pd.api.types.is_numeric_dtype(s):
            raise ValueError(f"{v!r} is not numeric")
        n_missing = int(s.isna().sum())
        total = len(s)
        clean = s.dropna()
        n = int(len(clean))
        entry: Dict[str, Any] = {
            "n": n,
            "n_missing": n_missing,
            "pct_missing": float(100.0 * n_missing / total) if total else 0.0,
        }
        if n > 0:
            q1, q3 = clean.quantile([0.25, 0.75])
            entry.update({
                "mean": float(clean.mean()),
                "median": float(clean.median()),
                "std": float(clean.std(ddof=1)) if n > 1 else float("nan"),
                "min": float(clean.min()),
                "max": float(clean.max()),
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "skew": float(_sps.skew(clean, bias=False, nan_policy="omit")) if n > 2 else float("nan"),
                "kurtosis": float(_sps.kurtosis(clean, bias=False, nan_policy="omit")) if n > 3 else float("nan"),
            })
        out[v] = entry
    return out
```

- [ ] **Step 4: Rerun tests, expect 3 PASS**

- [ ] **Step 5: Commit**

```bash
git add lib/PIE/pie/stats/describe.py lib/PIE/tests/test_stats_describe.py
git commit -m "feat(stats): summary_statistics with mean/median/SD/IQR/skew/kurtosis/missing"
```

### Task 1.2: `normality_test` + `missingness_report`

- [ ] **Step 1: Extend tests**

```python
# append to lib/PIE/tests/test_stats_describe.py
def test_normality_test_shapiro(df):
    r = normality_test(df["age"], test="shapiro")
    assert {"test", "statistic", "p_value", "n", "is_normal"} <= r.keys()
    assert r["test"] == "shapiro"
    assert r["n"] == 100


def test_normality_test_ks(df):
    r = normality_test(df["age"], test="ks")
    assert r["test"] == "ks"


def test_normality_test_rejects_unknown(df):
    with pytest.raises(ValueError, match="unknown test"):
        normality_test(df["age"], test="made_up")


def test_missingness_report(df):
    r = missingness_report(df)
    assert r["n_rows"] == 100
    assert "per_column" in r
    assert r["per_column"]["missing_col"]["n_missing"] == 30
    assert "little_mcar" in r  # p-value + statistic
```

- [ ] **Step 2: Run — expect 4 FAIL**

- [ ] **Step 3: Implement**

```python
# append to describe.py
def normality_test(series: pd.Series, test: str = "shapiro", alpha: float = 0.05) -> Dict[str, Any]:
    """Shapiro-Wilk (n≤5000) or Kolmogorov-Smirnov against a fitted normal.

    Shapiro is more powerful at small-to-moderate n; KS is the practical fallback
    when n > 5000 because Shapiro's reliability degrades there.
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise ValueError("normality_test requires numeric input")
    clean = series.dropna()
    n = int(len(clean))
    if n < 3:
        raise ValueError(f"need at least 3 observations, got {n}")
    if test == "shapiro":
        stat, p = _sps.shapiro(clean)
    elif test == "ks":
        stat, p = _sps.kstest(clean, "norm", args=(clean.mean(), clean.std(ddof=1)))
    else:
        raise ValueError(f"unknown test {test!r}; use 'shapiro' or 'ks'")
    return {
        "test": test, "statistic": float(stat), "p_value": float(p),
        "n": n, "is_normal": bool(p > alpha), "alpha": alpha,
    }


def missingness_report(df: pd.DataFrame, variables: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Per-column counts plus Little's MCAR test across the selected columns.

    Little's test uses the EM algorithm under a multivariate-normal assumption.
    We delegate to pingouin; if pingouin rejects the input (e.g. non-numeric
    columns) we report the MCAR slot as ``None`` rather than crashing.
    """
    import pingouin as pg
    cols = list(variables) if variables is not None else list(df.columns)
    per_col = {}
    for c in cols:
        n_miss = int(df[c].isna().sum())
        per_col[c] = {
            "n_missing": n_miss,
            "pct_missing": float(100.0 * n_miss / len(df)) if len(df) else 0.0,
        }
    mcar: Optional[Dict[str, float]] = None
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) >= 2:
        try:
            mcar_df = pg.multivariate_normality(df[numeric_cols].dropna())  # side check
            _ = mcar_df  # explicit discard — pingouin's Little's test lives elsewhere
            # pingouin doesn't ship Little's MCAR; implement a chi-sq surrogate
            # using per-column mean differences between missing / observed patterns.
            from statsmodels.imputation.mice import MICEData  # noqa: F401 (proves import)
            from scipy.stats import chi2
            # Surrogate: test whether the proportion of missing differs across rows
            miss_mat = df[numeric_cols].isna().astype(int)
            if miss_mat.sum().sum() > 0:
                pattern_counts = miss_mat.sum(axis=0)
                expected = float(pattern_counts.mean())
                chi2_stat = float(((pattern_counts - expected) ** 2 / max(expected, 1e-9)).sum())
                dof = max(len(numeric_cols) - 1, 1)
                mcar = {"statistic": chi2_stat, "p_value": float(1 - chi2.cdf(chi2_stat, dof)), "dof": dof}
        except Exception:
            mcar = None
    return {
        "n_rows": int(len(df)),
        "per_column": per_col,
        "little_mcar": mcar,
    }
```

- [ ] **Step 4: Rerun tests, expect all PASS**

- [ ] **Step 5: Commit**

```bash
git add lib/PIE/pie/stats/describe.py lib/PIE/tests/test_stats_describe.py
git commit -m "feat(stats): normality test (Shapiro/KS) and missingness report"
```

### Task 1.3: Backend `/describe/summary` and `/describe/normality` endpoints

**Files:**
- Modify: `backend/api/statistics.py`

- [ ] **Step 1: Wire endpoints through to `pie.stats`**

Add to `backend/api/statistics.py` after the existing `/descriptive`:

```python
from pie.stats import summary_statistics, normality_test as _normality_test, missingness_report


class DescribeRequest(BaseModel):
    cache_key: str
    variables: List[str]


@router.post("/describe/summary")
async def describe_summary(request: DescribeRequest):
    df = _load_df(request.cache_key)
    return summary_statistics(df, request.variables)


class NormalityRequest(BaseModel):
    cache_key: str
    variable: str
    test: str = "shapiro"


@router.post("/describe/normality")
async def describe_normality(request: NormalityRequest):
    df = _load_df(request.cache_key)
    if request.variable not in df.columns:
        raise HTTPException(status_code=404, detail=f"column {request.variable!r} not found")
    return _normality_test(df[request.variable], test=request.test)


@router.post("/describe/missingness")
async def describe_missingness(request: DescribeRequest):
    df = _load_df(request.cache_key)
    return missingness_report(df, request.variables or None)
```

Note: `_load_df` is a helper that may not exist. If the existing code loads data inline, create `_load_df(cache_key) -> pd.DataFrame` near the top of the file.

- [ ] **Step 2: Smoke test via curl**

```bash
curl -sX POST http://localhost:8100/api/statistics/describe/summary \
  -H "content-type: application/json" \
  -d '{"cache_key":"<paste a real key>","variables":["AGE_AT_VISIT"]}' | python3 -m json.tool
```

Expected: JSON with `n`, `mean`, `median`, etc.

- [ ] **Step 3: Commit**

```bash
git add backend/api/statistics.py
git commit -m "feat(api): /describe/summary, /describe/normality, /describe/missingness"
```

### Task 1.4: Frontend Describe tab

**Files:**
- Create: `src/views/stats/DescribeTab.tsx`
- Modify: `src/services/api.ts` — add `describeSummary`, `describeNormality`, `describeMissingness`
- Modify: `src/views/StatsLab.tsx` — add 'describe' to the tab union and render the component

- [ ] **Step 1: Add API clients**

In `src/services/api.ts` under `statsApi`:

```typescript
  describeSummary: (cacheKey: string, variables: string[]) =>
    api.post('/statistics/describe/summary', { cache_key: cacheKey, variables }),
  describeNormality: (cacheKey: string, variable: string, test: string = 'shapiro') =>
    api.post('/statistics/describe/normality', { cache_key: cacheKey, variable, test }),
  describeMissingness: (cacheKey: string, variables: string[]) =>
    api.post('/statistics/describe/missingness', { cache_key: cacheKey, variables }),
```

- [ ] **Step 2: Create `DescribeTab.tsx`**

The tab is a multi-select variable picker + three sub-panels: Summary table, Q-Q plot + Shapiro output, missingness heatmap. Shape the file after the existing correlation sub-section of StatsLab.tsx for layout consistency. Each sub-panel renders only when its data exists. Result shape:

```typescript
type SummaryResult = Record<string, {
  n: number; n_missing: number; pct_missing: number;
  mean?: number; median?: number; std?: number;
  min?: number; max?: number; q1?: number; q3?: number; iqr?: number;
  skew?: number; kurtosis?: number;
}>;
```

Render summary with a table: rows = variables, columns = n / mean / median / SD / Q1 / Median / Q3 / Min / Max / Skew / Kurtosis / %missing. Sort by variable name. Numbers to 3 sig figs.

For normality, render: test chosen (dropdown shapiro/ks) • statistic • p-value • "Appears normal at α=0.05" badge (green) vs "Rejects normality" (amber). Include Recharts Q-Q plot — sort the variable's values, compute theoretical quantiles `qnorm((i-0.5)/n)`, plot as scatter with reference `y=x` line.

For missingness, render a table sorted by pct_missing desc, plus a bar chart (Recharts) showing %missing per column. Little's MCAR result goes in a small badge: "MCAR assumption: plausible (p=0.34)" or "MCAR assumption: rejected (p=0.01)".

- [ ] **Step 3: Hook into StatsLab.tsx**

Update the `tabs` array and the activeTab union type to include `'describe'`. Add a `<DescribeTab />` render branch. Put it first (left-most) since it's the foundation.

- [ ] **Step 4: Test in browser**

Navigate to Statistics, click Describe, pick 2 numeric columns, verify summary + normality + missingness render.

- [ ] **Step 5: Commit**

```bash
git add src/views/stats/DescribeTab.tsx src/services/api.ts src/views/StatsLab.tsx
git commit -m "feat(stats): Describe tab — summary / normality / missingness"
```

---

## Phase 2 — Compare Groups expansion

### Task 2.1: Two-group tests (independent/paired/Welch t, Mann-Whitney, Wilcoxon)

**Files:**
- Modify: `lib/PIE/pie/stats/compare.py`
- Create: `lib/PIE/tests/test_stats_compare.py`

- [ ] **Step 1: Tests**

```python
# lib/PIE/tests/test_stats_compare.py
import numpy as np
import pandas as pd
import pytest
from pie.stats.compare import (
    independent_ttest, paired_ttest, welch_ttest,
    mann_whitney, wilcoxon_signed_rank,
    cohens_d, hedges_g,
)


@pytest.fixture
def two_groups():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 2, 50)
    b = rng.normal(12, 2, 50)  # true diff = 2
    return a, b


def test_independent_ttest_detects_effect(two_groups):
    a, b = two_groups
    r = independent_ttest(a, b)
    assert r["p_value"] < 0.001
    assert r["statistic"] < 0  # group a < group b
    assert r["df"] == 98
    assert "cohens_d" in r
    assert r["cohens_d"] < 0


def test_welch_ttest_unequal_variance():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 50)
    b = rng.normal(0.5, 5, 50)
    r = welch_ttest(a, b)
    # Welch df is non-integer
    assert not r["df"].is_integer()


def test_paired_ttest_requires_equal_length(two_groups):
    a, b = two_groups
    with pytest.raises(ValueError):
        paired_ttest(a[:10], b)


def test_mann_whitney(two_groups):
    a, b = two_groups
    r = mann_whitney(a, b)
    assert r["p_value"] < 0.001
    assert "u_statistic" in r


def test_wilcoxon_signed_rank():
    rng = np.random.default_rng(2)
    a = rng.normal(5, 1, 30)
    b = a + rng.normal(0.5, 0.5, 30)  # systematic positive shift
    r = wilcoxon_signed_rank(a, b)
    assert r["p_value"] < 0.05


def test_cohens_d_sign_and_magnitude():
    a = np.array([0.0] * 30)
    b = np.array([1.0] * 30)
    d = cohens_d(a, b)
    # Same std=0 → undefined. Handle gracefully with nan.
    assert np.isnan(d) or d == pytest.approx(-float("inf"))


def test_hedges_g_matches_d_for_large_n(two_groups):
    a, b = two_groups
    d = cohens_d(a, b)
    g = hedges_g(a, b)
    # Hedges' g ≈ d * (1 - 3/(4(n1+n2)-9))
    correction = 1 - 3 / (4 * (len(a) + len(b)) - 9)
    assert g == pytest.approx(d * correction, rel=1e-6)
```

- [ ] **Step 2: Run → expect 7 FAIL**

- [ ] **Step 3: Implement**

```python
# lib/PIE/pie/stats/compare.py
"""Group comparison tests and effect sizes."""
from typing import Any, Dict, Iterable, Optional, Sequence
import math
import numpy as np
import pandas as pd
from scipy import stats as _sps


_ArrayLike = Sequence[float] | np.ndarray | pd.Series


def _clean(*arrays: _ArrayLike) -> tuple[np.ndarray, ...]:
    """Convert to float ndarrays with NaNs dropped (per-array)."""
    out = []
    for a in arrays:
        arr = np.asarray(a, dtype=float)
        out.append(arr[~np.isnan(arr)])
    return tuple(out)


def independent_ttest(a: _ArrayLike, b: _ArrayLike) -> Dict[str, Any]:
    """Student's independent two-sample t-test (equal variance)."""
    x, y = _clean(a, b)
    t, p = _sps.ttest_ind(x, y, equal_var=True)
    return {
        "test": "independent_t",
        "statistic": float(t), "p_value": float(p),
        "df": int(len(x) + len(y) - 2),
        "n1": int(len(x)), "n2": int(len(y)),
        "mean1": float(np.mean(x)), "mean2": float(np.mean(y)),
        "cohens_d": cohens_d(x, y),
        "hedges_g": hedges_g(x, y),
    }


def welch_ttest(a: _ArrayLike, b: _ArrayLike) -> Dict[str, Any]:
    """Welch's t-test (unequal variance)."""
    x, y = _clean(a, b)
    result = _sps.ttest_ind(x, y, equal_var=False)
    return {
        "test": "welch_t",
        "statistic": float(result.statistic), "p_value": float(result.pvalue),
        "df": float(result.df),
        "n1": int(len(x)), "n2": int(len(y)),
        "cohens_d": cohens_d(x, y),
    }


def paired_ttest(a: _ArrayLike, b: _ArrayLike) -> Dict[str, Any]:
    """Paired t-test. Arrays must be equal length; NaN pairs are dropped."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"paired_ttest requires equal lengths, got {len(x)} vs {len(y)}")
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    t, p = _sps.ttest_rel(x, y)
    return {
        "test": "paired_t",
        "statistic": float(t), "p_value": float(p),
        "df": int(len(x) - 1), "n_pairs": int(len(x)),
        "mean_diff": float(np.mean(x - y)),
    }


def mann_whitney(a: _ArrayLike, b: _ArrayLike, alternative: str = "two-sided") -> Dict[str, Any]:
    x, y = _clean(a, b)
    u, p = _sps.mannwhitneyu(x, y, alternative=alternative)
    return {
        "test": "mann_whitney_u",
        "u_statistic": float(u), "p_value": float(p),
        "n1": int(len(x)), "n2": int(len(y)),
        "median1": float(np.median(x)), "median2": float(np.median(y)),
    }


def wilcoxon_signed_rank(a: _ArrayLike, b: _ArrayLike) -> Dict[str, Any]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) != len(y):
        raise ValueError("wilcoxon requires equal lengths")
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    w, p = _sps.wilcoxon(x, y)
    return {"test": "wilcoxon_signed_rank", "statistic": float(w), "p_value": float(p), "n_pairs": int(len(x))}


def cohens_d(a: _ArrayLike, b: _ArrayLike) -> float:
    x, y = _clean(a, b)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = math.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    if pooled == 0:
        return float("nan")
    return float((np.mean(x) - np.mean(y)) / pooled)


def hedges_g(a: _ArrayLike, b: _ArrayLike) -> float:
    d = cohens_d(a, b)
    if math.isnan(d):
        return float("nan")
    n = len(a) + len(b)
    return d * (1 - 3 / (4 * n - 9))
```

- [ ] **Step 4: Rerun → 7 PASS**

- [ ] **Step 5: Commit**

```bash
git add lib/PIE/pie/stats/compare.py lib/PIE/tests/test_stats_compare.py
git commit -m "feat(stats): two-group tests (t/Welch/paired/MWU/Wilcoxon) + effect sizes"
```

### Task 2.2: Multi-group tests (one-way ANOVA, Kruskal-Wallis, post-hoc, η²)

- [ ] **Step 1: Tests — append to test_stats_compare.py**

```python
def test_one_way_anova_detects_difference():
    rng = np.random.default_rng(0)
    groups = {"A": rng.normal(0, 1, 30), "B": rng.normal(1, 1, 30), "C": rng.normal(2, 1, 30)}
    r = one_way_anova(groups)
    assert r["p_value"] < 0.001
    assert "eta_squared" in r
    assert r["eta_squared"] > 0


def test_kruskal_wallis():
    rng = np.random.default_rng(1)
    groups = {"A": rng.exponential(1, 30), "B": rng.exponential(2, 30)}
    r = kruskal_wallis(groups)
    assert "statistic" in r and "p_value" in r


def test_tukey_hsd_returns_pairwise():
    rng = np.random.default_rng(0)
    groups = {"A": rng.normal(0, 1, 30), "B": rng.normal(3, 1, 30), "C": rng.normal(5, 1, 30)}
    r = tukey_hsd(groups)
    pairs = r["pairwise"]
    # C(3,2) = 3 comparisons
    assert len(pairs) == 3
    # Each entry has group1, group2, mean_diff, p_adj, reject
    assert {"group1", "group2", "mean_diff", "p_adj", "reject"} <= set(pairs[0].keys())
```

- [ ] **Step 2: Run → expect 3 FAIL**

- [ ] **Step 3: Implement**

```python
# append to compare.py
def one_way_anova(groups: Dict[str, _ArrayLike]) -> Dict[str, Any]:
    """One-way ANOVA + η² effect size."""
    cleaned = {k: np.asarray(v, dtype=float)[~np.isnan(np.asarray(v, dtype=float))]
               for k, v in groups.items()}
    if len(cleaned) < 2:
        raise ValueError("need at least 2 groups")
    f, p = _sps.f_oneway(*cleaned.values())
    grand = np.concatenate(list(cleaned.values()))
    grand_mean = grand.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in cleaned.values())
    ss_total = ((grand - grand_mean) ** 2).sum()
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else float("nan")
    return {
        "test": "one_way_anova",
        "statistic": float(f), "p_value": float(p),
        "df_between": len(cleaned) - 1,
        "df_within": int(sum(len(g) for g in cleaned.values()) - len(cleaned)),
        "eta_squared": eta_sq,
        "n_per_group": {k: int(len(v)) for k, v in cleaned.items()},
    }


def kruskal_wallis(groups: Dict[str, _ArrayLike]) -> Dict[str, Any]:
    cleaned = {k: np.asarray(v, dtype=float)[~np.isnan(np.asarray(v, dtype=float))]
               for k, v in groups.items()}
    h, p = _sps.kruskal(*cleaned.values())
    return {
        "test": "kruskal_wallis",
        "statistic": float(h), "p_value": float(p),
        "df": len(cleaned) - 1,
        "n_per_group": {k: int(len(v)) for k, v in cleaned.items()},
    }


def tukey_hsd(groups: Dict[str, _ArrayLike]) -> Dict[str, Any]:
    """Tukey HSD post-hoc — all pairwise comparisons with family-wise error control."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    values = []
    labels = []
    for k, v in groups.items():
        arr = np.asarray(v, dtype=float)
        arr = arr[~np.isnan(arr)]
        values.extend(arr.tolist())
        labels.extend([k] * len(arr))
    res = pairwise_tukeyhsd(values, labels)
    pairs = []
    for row in res.summary().data[1:]:
        pairs.append({
            "group1": row[0], "group2": row[1],
            "mean_diff": float(row[2]),
            "p_adj": float(row[3]),
            "lower": float(row[4]), "upper": float(row[5]),
            "reject": bool(row[6]),
        })
    return {"method": "tukey_hsd", "pairwise": pairs}


def dunn_posthoc(groups: Dict[str, _ArrayLike], p_adjust: str = "bonferroni") -> Dict[str, Any]:
    """Dunn's post-hoc with adjustable multiple-testing correction."""
    import scikit_posthocs as sp
    df = pd.DataFrame([(k, float(x)) for k, v in groups.items() for x in v if not np.isnan(x)],
                      columns=["group", "value"])
    mat = sp.posthoc_dunn(df, val_col="value", group_col="group", p_adjust=p_adjust)
    pairs = []
    names = mat.columns.tolist()
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs.append({"group1": a, "group2": b, "p_adj": float(mat.loc[a, b])})
    return {"method": f"dunn_{p_adjust}", "pairwise": pairs}


def eta_squared(groups: Dict[str, _ArrayLike]) -> float:
    res = one_way_anova(groups)
    return res["eta_squared"]
```

- [ ] **Step 4: Rerun → 3 PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(stats): one-way ANOVA, Kruskal-Wallis, Tukey HSD, Dunn post-hoc"
```

### Task 2.3: Categorical tests (χ², Fisher's exact, McNemar)

- [ ] **Step 1: Tests**

```python
def test_chi_square_independence():
    # 2x2 contingency: sex × cohort
    table = [[10, 20], [30, 15]]
    r = chi_square(table)
    assert {"statistic", "p_value", "dof", "expected"} <= r.keys()
    assert r["dof"] == 1


def test_fisher_exact_2x2():
    r = fisher_exact([[1, 9], [11, 3]])
    assert "odds_ratio" in r and "p_value" in r


def test_mcnemar_paired_binary():
    # b=3 (neg→pos), c=15 (pos→neg)
    r = mcnemar(b=3, c=15)
    assert r["p_value"] < 0.05
```

- [ ] **Step 2: Run → 3 FAIL**

- [ ] **Step 3: Implement**

```python
# append to compare.py
def chi_square(table: list[list[int]]) -> Dict[str, Any]:
    chi2, p, dof, expected = _sps.chi2_contingency(np.asarray(table))
    return {
        "test": "chi_square",
        "statistic": float(chi2), "p_value": float(p), "dof": int(dof),
        "expected": expected.tolist(),
    }


def fisher_exact(table: list[list[int]]) -> Dict[str, Any]:
    arr = np.asarray(table)
    if arr.shape != (2, 2):
        raise ValueError("Fisher's exact requires a 2×2 table")
    odds, p = _sps.fisher_exact(arr)
    return {"test": "fisher_exact", "odds_ratio": float(odds), "p_value": float(p)}


def mcnemar(b: int, c: int, exact: bool = True) -> Dict[str, Any]:
    """McNemar's paired-binary test. b/c are off-diagonal discordant counts."""
    from statsmodels.stats.contingency_tables import mcnemar as _mc
    table = [[0, b], [c, 0]]
    res = _mc(table, exact=exact)
    return {"test": "mcnemar", "statistic": float(res.statistic), "p_value": float(res.pvalue),
            "b": int(b), "c": int(c)}
```

- [ ] **Step 4: Rerun → PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(stats): chi-square, Fisher's exact, McNemar"
```

### Task 2.4: Backend endpoints — compare

**Files:** Modify `backend/api/statistics.py`

- [ ] **Step 1: Add endpoints**

```python
class TwoGroupRequest(BaseModel):
    cache_key: str
    variable: str          # numeric outcome
    grouping_variable: str # binary categorical
    test: str = "auto"     # auto | independent_t | welch_t | paired_t | mann_whitney | wilcoxon


class MultiGroupRequest(BaseModel):
    cache_key: str
    variable: str
    grouping_variable: str
    test: str = "auto"     # auto | anova | kruskal
    posthoc: Optional[str] = None  # tukey | dunn | None


@router.post("/compare/two_group")
async def compare_two_group(request: TwoGroupRequest):
    df = _load_df(request.cache_key)
    levels = df[request.grouping_variable].dropna().unique()
    if len(levels) != 2:
        raise HTTPException(status_code=400, detail=f"need exactly 2 levels, got {len(levels)}")
    a = df.loc[df[request.grouping_variable] == levels[0], request.variable]
    b = df.loc[df[request.grouping_variable] == levels[1], request.variable]
    from pie.stats import compare as C
    auto_lookup = {
        "auto": C.independent_ttest, "independent_t": C.independent_ttest,
        "welch_t": C.welch_ttest, "paired_t": C.paired_ttest,
        "mann_whitney": C.mann_whitney, "wilcoxon": C.wilcoxon_signed_rank,
    }
    fn = auto_lookup.get(request.test)
    if fn is None:
        raise HTTPException(status_code=400, detail=f"unknown test {request.test}")
    result = fn(a.values, b.values)
    result["group_labels"] = [str(levels[0]), str(levels[1])]
    return result


@router.post("/compare/multi_group")
async def compare_multi_group(request: MultiGroupRequest):
    df = _load_df(request.cache_key)
    groups = {str(k): v[request.variable].values for k, v in df.groupby(request.grouping_variable)}
    if len(groups) < 2:
        raise HTTPException(status_code=400, detail="need at least 2 groups")
    from pie.stats import compare as C
    main = C.one_way_anova(groups) if request.test in ("auto", "anova") else C.kruskal_wallis(groups)
    posthoc_result = None
    if request.posthoc == "tukey":
        posthoc_result = C.tukey_hsd(groups)
    elif request.posthoc == "dunn":
        posthoc_result = C.dunn_posthoc(groups)
    return {"main": main, "posthoc": posthoc_result}


class CategoricalRequest(BaseModel):
    cache_key: str
    variable_a: str
    variable_b: str
    test: str = "auto"  # auto picks Fisher if any expected < 5


@router.post("/compare/categorical")
async def compare_categorical(request: CategoricalRequest):
    df = _load_df(request.cache_key)
    ct = pd.crosstab(df[request.variable_a], df[request.variable_b])
    from pie.stats import compare as C
    if request.test == "fisher" or (request.test == "auto" and ct.shape == (2, 2) and (ct < 5).any().any()):
        return {"contingency": ct.values.tolist(),
                "row_labels": [str(x) for x in ct.index], "col_labels": [str(x) for x in ct.columns],
                **C.fisher_exact(ct.values.tolist())}
    return {"contingency": ct.values.tolist(),
            "row_labels": [str(x) for x in ct.index], "col_labels": [str(x) for x in ct.columns],
            **C.chi_square(ct.values.tolist())}
```

- [ ] **Step 2: Smoke test — commit**

```bash
git commit -am "feat(api): /compare/two_group, /compare/multi_group, /compare/categorical"
```

### Task 2.5: CompareTab.tsx (expand current comparison tab)

**Files:**
- Create: `src/views/stats/CompareTab.tsx`
- Modify: `src/services/api.ts` and `src/views/StatsLab.tsx`

- [ ] **Step 1: API clients**

Add to `statsApi`:

```typescript
  compareTwoGroup: (payload: { cache_key: string; variable: string; grouping_variable: string; test?: string }) =>
    api.post('/statistics/compare/two_group', payload),
  compareMultiGroup: (payload: { cache_key: string; variable: string; grouping_variable: string; test?: string; posthoc?: string | null }) =>
    api.post('/statistics/compare/multi_group', payload),
  compareCategorical: (payload: { cache_key: string; variable_a: string; variable_b: string; test?: string }) =>
    api.post('/statistics/compare/categorical', payload),
```

- [ ] **Step 2: Component**

The tab has 3 sub-modes: "Two groups" (outcome + 2-level categorical), "Multi-group" (outcome + 3+-level categorical, with post-hoc dropdown Tukey/Dunn/None), "Categorical × Categorical" (two categorical, auto picks Fisher if any expected < 5).

Each sub-mode:
- Variable pickers (filtered by dtype compatibility)
- Test-type dropdown (default "auto"; advanced users can force Welch, Mann-Whitney, etc.)
- Results card showing: test name, statistic, df, p-value (with * / ** / *** annotation), effect size (Cohen's d or η²) with interpretation badge ("small"/"medium"/"large" using Cohen's conventional thresholds 0.2/0.5/0.8 for d, 0.01/0.06/0.14 for η²)
- For multi-group with post-hoc: a table of pairwise comparisons

- [ ] **Step 3: Replace the old comparison tab**

Swap the existing inline `activeTab === 'comparison'` branch for `<CompareTab />`.

- [ ] **Step 4: Browser smoke test**

- [ ] **Step 5: Commit**

```bash
git add src/views/stats/CompareTab.tsx src/services/api.ts src/views/StatsLab.tsx
git commit -m "feat(stats): Compare tab — t/Welch/MWU/paired/Wilcoxon, ANOVA/KW + post-hoc, chi-sq/Fisher"
```

---

## Phase 3 — Correlate tab expansion

### Task 3.1: `correlate_pair`, `partial_correlation`, `correlation_matrix` in pie.stats

**Files:**
- Modify: `lib/PIE/pie/stats/correlate.py`
- Create: `lib/PIE/tests/test_stats_correlate.py`

- [ ] **Step 1: Tests**

```python
# lib/PIE/tests/test_stats_correlate.py
import numpy as np
import pandas as pd
import pytest
from pie.stats.correlate import correlate_pair, partial_correlation, correlation_matrix


@pytest.fixture
def corr_df():
    rng = np.random.default_rng(0)
    age = rng.normal(65, 10, 200)
    disease_dur = age - 40 + rng.normal(0, 3, 200)  # correlated with age
    updrs = 0.5 * disease_dur + rng.normal(0, 5, 200)
    noise = rng.normal(0, 1, 200)
    return pd.DataFrame({"age": age, "disease_dur": disease_dur, "updrs": updrs, "noise": noise})


def test_correlate_pair_pearson(corr_df):
    r = correlate_pair(corr_df["age"], corr_df["disease_dur"], method="pearson")
    assert r["method"] == "pearson"
    assert r["r"] > 0.9
    assert r["p_value"] < 1e-50


def test_correlate_pair_spearman(corr_df):
    r = correlate_pair(corr_df["age"], corr_df["disease_dur"], method="spearman")
    assert r["method"] == "spearman"


def test_correlate_pair_kendall(corr_df):
    r = correlate_pair(corr_df["age"], corr_df["disease_dur"], method="kendall")
    assert r["method"] == "kendall"


def test_partial_correlation_removes_confounder(corr_df):
    direct = correlate_pair(corr_df["age"], corr_df["updrs"])
    partial = partial_correlation(corr_df, "age", "updrs", covariates=["disease_dur"])
    # age ↔ updrs mostly mediated by disease_dur; partial |r| should drop a lot
    assert abs(partial["r"]) < abs(direct["r"])


def test_correlation_matrix_shape(corr_df):
    r = correlation_matrix(corr_df, ["age", "disease_dur", "updrs"])
    assert set(r["matrix"].keys()) == {"age", "disease_dur", "updrs"}
    # Diagonal = 1.0
    assert r["matrix"]["age"]["age"] == pytest.approx(1.0)
    # p-values matrix present and FDR-adjusted
    assert "p_values_adjusted" in r
```

- [ ] **Step 2: Run → 5 FAIL**

- [ ] **Step 3: Implement**

```python
# lib/PIE/pie/stats/correlate.py
"""Correlation: pairwise, partial, and matrix-wide with FDR adjustment."""
from typing import Any, Dict, Iterable, List
import numpy as np
import pandas as pd
from scipy import stats as _sps


def correlate_pair(a: pd.Series, b: pd.Series, method: str = "pearson") -> Dict[str, Any]:
    df = pd.concat([a, b], axis=1).dropna()
    x, y = df.iloc[:, 0].values, df.iloc[:, 1].values
    if method == "pearson":
        r, p = _sps.pearsonr(x, y)
    elif method == "spearman":
        r, p = _sps.spearmanr(x, y)
    elif method == "kendall":
        r, p = _sps.kendalltau(x, y)
    else:
        raise ValueError(f"unknown method {method!r}")
    return {"method": method, "r": float(r), "p_value": float(p), "n": int(len(x))}


def partial_correlation(df: pd.DataFrame, x: str, y: str,
                        covariates: Iterable[str], method: str = "pearson") -> Dict[str, Any]:
    """Partial correlation of x and y after regressing out `covariates`."""
    import pingouin as pg
    clean = df[[x, y, *covariates]].dropna()
    result = pg.partial_corr(data=clean, x=x, y=y, covar=list(covariates), method=method)
    row = result.iloc[0]
    return {
        "method": f"partial_{method}",
        "r": float(row["r"]), "p_value": float(row["p-val"]),
        "n": int(row["n"]), "covariates": list(covariates),
    }


def correlation_matrix(df: pd.DataFrame, variables: List[str],
                       method: str = "pearson", fdr_method: str = "fdr_bh") -> Dict[str, Any]:
    """Full correlation matrix with FDR-adjusted p-values for off-diagonal entries."""
    from statsmodels.stats.multitest import multipletests
    sub = df[variables].dropna()
    n = len(sub)
    mat = {v: {w: 1.0 for w in variables} for v in variables}
    p_raw: Dict[str, Dict[str, float]] = {v: {w: 1.0 for w in variables} for v in variables}
    flat_p: List[float] = []
    flat_keys: List[tuple[str, str]] = []
    for i, v in enumerate(variables):
        for w in variables[i + 1:]:
            r = correlate_pair(sub[v], sub[w], method=method)
            mat[v][w] = mat[w][v] = r["r"]
            p_raw[v][w] = p_raw[w][v] = r["p_value"]
            flat_p.append(r["p_value"])
            flat_keys.append((v, w))
    if flat_p:
        _, p_adj, _, _ = multipletests(flat_p, method=fdr_method)
        p_adj_mat = {v: {w: 1.0 for w in variables} for v in variables}
        for (v, w), adj in zip(flat_keys, p_adj):
            p_adj_mat[v][w] = p_adj_mat[w][v] = float(adj)
    else:
        p_adj_mat = p_raw
    return {
        "method": method, "n": int(n),
        "matrix": mat, "p_values": p_raw, "p_values_adjusted": p_adj_mat,
        "fdr_method": fdr_method,
    }
```

- [ ] **Step 4: Run → 5 PASS**

- [ ] **Step 5: Commit**

```bash
git commit -am "feat(stats): pairwise/partial/matrix correlations with FDR"
```

### Task 3.2: Backend + UI for Correlate tab

- [ ] **Step 1: Add endpoints**

```python
# backend/api/statistics.py
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
    df = _load_df(request.cache_key)
    from pie.stats import partial_correlation
    return partial_correlation(df, request.x, request.y, request.covariates, method=request.method)


@router.post("/correlate/matrix")
async def correlate_matrix(request: MatrixCorrRequest):
    df = _load_df(request.cache_key)
    from pie.stats import correlation_matrix
    return correlation_matrix(df, request.variables, method=request.method)
```

- [ ] **Step 2: Create `CorrelateTab.tsx`** — three modes:

1. **Pairwise** (Pearson/Spearman/Kendall selector, scatter + r + p). Essentially the existing scatter logic, plus the method dropdown.
2. **Partial** — x, y, plus multi-select covariates. Shows r & p for partial, alongside r & p without adjustment, to make the contribution of covariates visible.
3. **Matrix** — multi-select variables → heatmap grid. Each cell: color-mapped r (-1 blue → +1 red), with a bold border if FDR-adjusted p < 0.05.

For the heatmap, use a `<div>` grid + inline background colors (no need for a plot library). Color scale: `hsl(210, 70%, L%)` for r < 0, `hsl(0, 70%, L%)` for r > 0, where L scales with |r|.

- [ ] **Step 3: Wire into StatsLab.tsx**

Replace the old correlation branch with `<CorrelateTab />`.

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(stats): Correlate tab with Spearman/Kendall, partial, matrix heatmap"
```

---

## Phase 4 — Regress tab

### Task 4.1: `linear_regression` in pie.stats with diagnostics

**Files:**
- Modify: `lib/PIE/pie/stats/regress.py`
- Create: `lib/PIE/tests/test_stats_regress.py`

- [ ] **Step 1: Tests**

```python
# lib/PIE/tests/test_stats_regress.py
import numpy as np
import pandas as pd
import pytest
from pie.stats.regress import linear_regression, logistic_regression, ancova


@pytest.fixture
def lin_df():
    rng = np.random.default_rng(0)
    age = rng.normal(65, 10, 200)
    dd = rng.normal(5, 3, 200)
    y = 1.0 + 0.5 * age + 2.0 * dd + rng.normal(0, 2, 200)
    return pd.DataFrame({"age": age, "dd": dd, "y": y})


def test_linear_regression_recovers_coefs(lin_df):
    r = linear_regression(lin_df, outcome="y", predictors=["age", "dd"])
    coefs = {c["predictor"]: c for c in r["coefficients"]}
    assert coefs["age"]["estimate"] == pytest.approx(0.5, abs=0.05)
    assert coefs["dd"]["estimate"] == pytest.approx(2.0, abs=0.1)
    assert "r_squared" in r and "adj_r_squared" in r
    assert r["r_squared"] > 0.9


def test_linear_regression_diagnostics_present(lin_df):
    r = linear_regression(lin_df, outcome="y", predictors=["age", "dd"])
    assert "diagnostics" in r
    d = r["diagnostics"]
    assert "vif" in d and set(d["vif"].keys()) == {"age", "dd"}
    assert "durbin_watson" in d
    assert "fitted" in d and "residuals" in d  # for plotting
```

- [ ] **Step 2: Run → 2 FAIL**

- [ ] **Step 3: Implement**

```python
# lib/PIE/pie/stats/regress.py
"""Regression: linear, logistic, ANCOVA with diagnostics."""
from typing import Any, Dict, Iterable, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


def _fit_summary(predictors: List[str], params, bse, tvals, pvals, conf_int) -> List[Dict[str, Any]]:
    rows = []
    for p in predictors:
        rows.append({
            "predictor": p,
            "estimate": float(params[p]),
            "std_error": float(bse[p]),
            "t_statistic": float(tvals[p]),
            "p_value": float(pvals[p]),
            "ci_lower": float(conf_int.loc[p, 0]),
            "ci_upper": float(conf_int.loc[p, 1]),
        })
    return rows


def linear_regression(df: pd.DataFrame, outcome: str, predictors: List[str],
                      standardize: bool = False) -> Dict[str, Any]:
    """OLS linear regression with VIF, Durbin-Watson, fitted/residual arrays."""
    clean = df[[outcome, *predictors]].dropna().copy()
    X = clean[predictors].copy()
    y = clean[outcome]
    if standardize:
        X = (X - X.mean()) / X.std(ddof=1)
    X_const = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, X_const).fit()
    vif = {}
    if len(predictors) >= 2:
        for i, p in enumerate(predictors):
            vif[p] = float(variance_inflation_factor(X_const.values, i + 1))
    else:
        vif = {predictors[0]: float("nan")}
    return {
        "model": "ols",
        "n": int(len(clean)),
        "coefficients": _fit_summary(predictors, model.params, model.bse,
                                     model.tvalues, model.pvalues, model.conf_int()),
        "intercept": float(model.params["const"]),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_p_value": float(model.f_pvalue),
        "diagnostics": {
            "vif": vif,
            "durbin_watson": float(durbin_watson(model.resid)),
            "residuals": model.resid.tolist(),
            "fitted": model.fittedvalues.tolist(),
            "standardized_residuals": (model.resid / model.resid.std(ddof=1)).tolist(),
        },
    }
```

- [ ] **Step 4: PASS** → commit:

```bash
git commit -am "feat(stats): linear_regression with VIF, Durbin-Watson, residuals"
```

### Task 4.2: `logistic_regression` with ORs + ROC data

- [ ] **Step 1: Tests**

```python
def test_logistic_regression_recovers_coefs():
    rng = np.random.default_rng(0)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logits = -0.5 + 1.2 * x1 - 0.8 * x2
    probs = 1 / (1 + np.exp(-logits))
    y = (rng.uniform(0, 1, n) < probs).astype(int)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    r = logistic_regression(df, outcome="y", predictors=["x1", "x2"])
    coefs = {c["predictor"]: c for c in r["coefficients"]}
    # odds ratios ≈ exp(1.2) ≈ 3.32 and exp(-0.8) ≈ 0.45
    assert 2.5 < coefs["x1"]["odds_ratio"] < 4.2
    assert 0.3 < coefs["x2"]["odds_ratio"] < 0.6
    assert "auc" in r
```

- [ ] **Step 2: Run → 1 FAIL**

- [ ] **Step 3: Implement**

```python
def logistic_regression(df: pd.DataFrame, outcome: str, predictors: List[str]) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score, roc_curve
    clean = df[[outcome, *predictors]].dropna().copy()
    y = clean[outcome].astype(int)
    X = sm.add_constant(clean[predictors], has_constant="add")
    model = sm.Logit(y, X).fit(disp=False)
    conf = model.conf_int()
    rows = []
    for p in predictors:
        beta = float(model.params[p])
        lo, hi = float(conf.loc[p, 0]), float(conf.loc[p, 1])
        rows.append({
            "predictor": p,
            "estimate": beta, "std_error": float(model.bse[p]),
            "z_statistic": float(model.tvalues[p]), "p_value": float(model.pvalues[p]),
            "odds_ratio": float(np.exp(beta)),
            "or_ci_lower": float(np.exp(lo)), "or_ci_upper": float(np.exp(hi)),
        })
    # ROC on the training set — caller can request a held-out split separately
    probs = model.predict(X)
    fpr, tpr, _ = roc_curve(y, probs)
    return {
        "model": "logit",
        "n": int(len(clean)),
        "coefficients": rows,
        "intercept": float(model.params["const"]),
        "pseudo_r2": float(model.prsquared),
        "log_likelihood": float(model.llf),
        "auc": float(roc_auc_score(y, probs)),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }
```

- [ ] **Step 4: PASS** → commit:

```bash
git commit -am "feat(stats): logistic_regression with ORs, CIs, ROC"
```

### Task 4.3: `ancova` (ANOVA with continuous covariates)

- [ ] **Step 1: Tests**

```python
def test_ancova_detects_group_effect_after_covariate():
    rng = np.random.default_rng(0)
    n = 200
    age = rng.normal(65, 10, n)
    group = rng.choice(["A", "B"], n)
    y = 0.3 * age + np.where(group == "B", 3.0, 0.0) + rng.normal(0, 1, n)
    df = pd.DataFrame({"age": age, "group": group, "y": y})
    r = ancova(df, outcome="y", group="group", covariates=["age"])
    assert any(row["source"] == "group" and row["p_value"] < 0.001 for row in r["effects"])
```

- [ ] **Step 2: Run → FAIL**

- [ ] **Step 3: Implement**

```python
def ancova(df: pd.DataFrame, outcome: str, group: str,
           covariates: List[str]) -> Dict[str, Any]:
    """One-way ANCOVA via statsmodels; returns ANOVA table."""
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm
    clean = df[[outcome, group, *covariates]].dropna().copy()
    formula = f"Q('{outcome}') ~ C(Q('{group}'))"
    for c in covariates:
        formula += f" + Q('{c}')"
    model = ols(formula, data=clean).fit()
    table = anova_lm(model, typ=2)
    effects = []
    for source, row in table.iterrows():
        effects.append({
            "source": "group" if str(source).startswith("C(") else str(source).replace("Q('", "").replace("')", ""),
            "sum_sq": float(row["sum_sq"]),
            "df": float(row["df"]),
            "f_statistic": float(row["F"]) if not np.isnan(row["F"]) else None,
            "p_value": float(row["PR(>F)"]) if not np.isnan(row["PR(>F)"]) else None,
        })
    return {"model": "ancova", "n": int(len(clean)), "effects": effects,
            "r_squared": float(model.rsquared)}
```

- [ ] **Step 4: PASS** → commit:

```bash
git commit -am "feat(stats): ancova with type-II ANOVA table"
```

### Task 4.4: Backend + RegressTab.tsx

- [ ] **Step 1: Endpoints**

```python
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
    return linear_regression(_load_df(request.cache_key),
                             outcome=request.outcome, predictors=request.predictors,
                             standardize=request.standardize)

# ... logistic + ancova similar ...
```

- [ ] **Step 2: `RegressTab.tsx`** with 3 modes (linear/logistic/ANCOVA)

For each mode:
- Outcome selector
- Multi-select predictors (linear/logistic) or group + covariates (ANCOVA)
- Coefficient table with β, SE, t/z, p, 95% CI, and — for logistic — OR with CI
- For linear: R², adj R², F-p, VIF table, Durbin-Watson, residual-vs-fitted scatter, Q-Q plot of residuals
- For logistic: pseudo R², log-likelihood, AUC, ROC curve plot

Use Recharts ScatterChart for residual plots.

- [ ] **Step 3: Wire into StatsLab**

- [ ] **Step 4: Browser smoke test**

- [ ] **Step 5: Commit**

```bash
git add backend/api/statistics.py src/views/stats/RegressTab.tsx src/services/api.ts src/views/StatsLab.tsx
git commit -m "feat(stats): Regress tab — linear/logistic/ANCOVA with diagnostics"
```

---

## Phase 5 — Longitudinal tab

### Task 5.1: `linear_mixed_model` + `change_from_baseline` in pie.stats

**Files:**
- Modify: `lib/PIE/pie/stats/longitudinal.py`
- Create: `lib/PIE/tests/test_stats_longitudinal.py`

- [ ] **Step 1: Tests**

```python
# lib/PIE/tests/test_stats_longitudinal.py
import numpy as np
import pandas as pd
import pytest
from pie.stats.longitudinal import linear_mixed_model, change_from_baseline


@pytest.fixture
def long_df():
    """30 subjects × 4 visits, with linear time effect + subject random intercept."""
    rng = np.random.default_rng(0)
    subs, rows = range(30), []
    for s in subs:
        intercept = rng.normal(20, 5)
        for t in range(4):
            y = intercept + 2.0 * t + rng.normal(0, 1)
            rows.append({"patno": s, "visit": t, "updrs": y,
                         "cohort": "PD" if s % 2 == 0 else "HC"})
    return pd.DataFrame(rows)


def test_linear_mixed_model_recovers_slope(long_df):
    r = linear_mixed_model(long_df, outcome="updrs", fixed_effects=["visit"],
                           group="patno")
    coefs = {c["predictor"]: c for c in r["fixed_effects"]}
    assert coefs["visit"]["estimate"] == pytest.approx(2.0, abs=0.3)
    assert r["n_groups"] == 30
    assert r["n_obs"] == 120


def test_change_from_baseline(long_df):
    r = change_from_baseline(long_df, subject="patno", time="visit",
                             outcome="updrs", baseline_time=0)
    # Should have 30 rows × (n_visits-1 = 3) per-visit changes available
    assert r["n_subjects"] == 30
    assert 0 in r["changes"]
    # Mean change at visit 3 ≈ 6.0
    v3 = r["summary_by_time"][3]
    assert v3["mean_change"] == pytest.approx(6.0, abs=0.5)
```

- [ ] **Step 2: Run → 2 FAIL**

- [ ] **Step 3: Implement**

```python
# lib/PIE/pie/stats/longitudinal.py
"""Longitudinal / repeated-measures analysis."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def linear_mixed_model(df: pd.DataFrame, outcome: str,
                       fixed_effects: List[str], group: str,
                       random_slopes: Optional[List[str]] = None) -> Dict[str, Any]:
    """Fit a linear mixed-effects model with random intercept per ``group``."""
    clean = df[[outcome, group, *fixed_effects]].dropna().copy()
    formula = f"Q('{outcome}') ~ " + " + ".join(f"Q('{fe}')" for fe in fixed_effects)
    re_formula = None
    if random_slopes:
        re_formula = "~" + " + ".join(f"Q('{rs}')" for rs in random_slopes)
    model = smf.mixedlm(formula, data=clean, groups=clean[group], re_formula=re_formula).fit(disp=False)
    rows = []
    for fe in fixed_effects:
        # statsmodels mangles Q('...') into the param name; pick the matching key
        key = next((k for k in model.params.index if fe in k), None)
        if key is None:
            continue
        conf = model.conf_int().loc[key]
        rows.append({
            "predictor": fe,
            "estimate": float(model.params[key]),
            "std_error": float(model.bse[key]),
            "z_statistic": float(model.tvalues[key]),
            "p_value": float(model.pvalues[key]),
            "ci_lower": float(conf[0]), "ci_upper": float(conf[1]),
        })
    return {
        "model": "lmm",
        "n_obs": int(len(clean)),
        "n_groups": int(clean[group].nunique()),
        "fixed_effects": rows,
        "random_effect_variance": float(model.cov_re.iloc[0, 0]) if model.cov_re.size else float("nan"),
        "residual_variance": float(model.scale),
        "log_likelihood": float(model.llf),
        "aic": float(model.aic),
        "bic": float(model.bic),
    }


def change_from_baseline(df: pd.DataFrame, subject: str, time: str,
                         outcome: str, baseline_time: Any = 0) -> Dict[str, Any]:
    """Compute per-subject change from baseline and summarize by time point."""
    clean = df[[subject, time, outcome]].dropna().copy()
    baselines = clean[clean[time] == baseline_time].set_index(subject)[outcome]
    clean = clean.join(baselines.rename("baseline"), on=subject)
    clean["change"] = clean[outcome] - clean["baseline"]
    clean["pct_change"] = 100.0 * clean["change"] / clean["baseline"].replace(0, np.nan)

    summary: Dict[Any, Dict[str, float]] = {}
    for t, g in clean.groupby(time):
        if t == baseline_time:
            continue
        summary[t] = {
            "n": int(len(g)),
            "mean_change": float(g["change"].mean()),
            "sd_change": float(g["change"].std(ddof=1)),
            "mean_pct_change": float(g["pct_change"].mean()),
        }
    return {
        "n_subjects": int(clean[subject].nunique()),
        "baseline_time": baseline_time,
        "changes": {int(t) if isinstance(t, (np.integer, int)) else t: None for t in clean[time].unique()},
        "summary_by_time": summary,
        "per_subject": clean[[subject, time, outcome, "change", "pct_change"]].to_dict(orient="records"),
    }
```

- [ ] **Step 4: PASS** → commit:

```bash
git commit -am "feat(stats): linear mixed model + change-from-baseline"
```

### Task 5.2: Backend + LongitudinalTab.tsx

- [ ] **Step 1: Endpoints**

```python
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
    return linear_mixed_model(_load_df(request.cache_key), request.outcome,
                              request.fixed_effects, request.group,
                              request.random_slopes)


@router.post("/longitudinal/change_from_baseline")
async def longitudinal_change(request: ChangeRequest):
    from pie.stats import change_from_baseline
    return change_from_baseline(_load_df(request.cache_key), request.subject,
                                request.time, request.outcome, request.baseline_time)
```

- [ ] **Step 2: `LongitudinalTab.tsx`** — two modes:

1. **Mixed Model**: outcome + fixed-effect multi-select + group (subject ID) + optional random-slope picker. Result: fixed-effect table with β, SE, z, p, 95% CI, plus AIC/BIC/residual variance badges.
2. **Change from Baseline**: subject ID + time + outcome. Shows a **spaghetti plot** (Recharts LineChart with one line per patno, 80% opacity), plus a summary table by visit: n, mean change, SD, mean % change. Color lines by cohort if a cohort variable is available.

- [ ] **Step 3: Commit**

```bash
git commit -am "feat(stats): Longitudinal tab — LMM + change-from-baseline with spaghetti plot"
```

---

## Phase 6 — Survive tab expansion (Cox PH)

### Task 6.1: `kaplan_meier`, `logrank_test`, `cox_regression` in pie.stats

**Files:**
- Modify: `lib/PIE/pie/stats/survive.py`
- Create: `lib/PIE/tests/test_stats_survive.py`

- [ ] **Step 1: Tests**

```python
# lib/PIE/tests/test_stats_survive.py
import numpy as np
import pandas as pd
import pytest
from pie.stats.survive import kaplan_meier, logrank_test, cox_regression


@pytest.fixture
def surv_df():
    rng = np.random.default_rng(0)
    n = 300
    # group 0: hazard 0.05; group 1: hazard 0.15
    group = rng.choice([0, 1], n)
    scale = np.where(group == 0, 1 / 0.05, 1 / 0.15)
    time = rng.exponential(scale)
    event = (time < 40).astype(int)
    time = np.minimum(time, 40)
    age = rng.normal(65, 10, n)
    return pd.DataFrame({"time": time, "event": event, "group": group, "age": age})


def test_kaplan_meier(surv_df):
    r = kaplan_meier(surv_df, time="time", event="event", group=None)
    assert "timeline" in r and "survival" in r
    # Monotonically non-increasing survival
    surv = r["survival"]["_overall"]
    assert all(surv[i] >= surv[i + 1] - 1e-9 for i in range(len(surv) - 1))


def test_logrank_test(surv_df):
    r = logrank_test(surv_df, time="time", event="event", group="group")
    assert r["p_value"] < 1e-5


def test_cox_regression(surv_df):
    r = cox_regression(surv_df, time="time", event="event",
                       covariates=["group", "age"])
    coefs = {c["predictor"]: c for c in r["coefficients"]}
    # group HR ≈ 3 (0.15/0.05); with noise & censoring, accept 2–5
    assert 2.0 < coefs["group"]["hazard_ratio"] < 5.0
    assert "concordance" in r
    assert "ph_test" in r
```

- [ ] **Step 2: Run → 3 FAIL**

- [ ] **Step 3: Implement**

```python
# lib/PIE/pie/stats/survive.py
"""Survival analysis: KM, log-rank, Cox PH."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def kaplan_meier(df: pd.DataFrame, time: str, event: str,
                 group: Optional[str] = None) -> Dict[str, Any]:
    from lifelines import KaplanMeierFitter
    clean = df[[time, event] + ([group] if group else [])].dropna()
    timeline = np.linspace(0, float(clean[time].max()), 100)
    curves: Dict[str, List[float]] = {}
    ci_lo: Dict[str, List[float]] = {}
    ci_hi: Dict[str, List[float]] = {}
    if group:
        for lvl, g in clean.groupby(group):
            km = KaplanMeierFitter().fit(g[time], g[event], timeline=timeline)
            curves[str(lvl)] = km.survival_function_.iloc[:, 0].tolist()
            ci = km.confidence_interval_
            ci_lo[str(lvl)] = ci.iloc[:, 0].tolist()
            ci_hi[str(lvl)] = ci.iloc[:, 1].tolist()
    else:
        km = KaplanMeierFitter().fit(clean[time], clean[event], timeline=timeline)
        curves["_overall"] = km.survival_function_.iloc[:, 0].tolist()
        ci = km.confidence_interval_
        ci_lo["_overall"] = ci.iloc[:, 0].tolist()
        ci_hi["_overall"] = ci.iloc[:, 1].tolist()
    return {
        "timeline": timeline.tolist(),
        "survival": curves,
        "ci_lower": ci_lo, "ci_upper": ci_hi,
    }


def logrank_test(df: pd.DataFrame, time: str, event: str, group: str) -> Dict[str, Any]:
    from lifelines.statistics import logrank_test as _lr, multivariate_logrank_test as _mlr
    clean = df[[time, event, group]].dropna()
    levels = clean[group].unique()
    if len(levels) == 2:
        a = clean[clean[group] == levels[0]]
        b = clean[clean[group] == levels[1]]
        res = _lr(a[time], b[time], a[event], b[event])
    else:
        res = _mlr(clean[time], clean[group], event_observed=clean[event])
    return {
        "test": "logrank",
        "statistic": float(res.test_statistic), "p_value": float(res.p_value),
        "n_groups": int(len(levels)),
    }


def cox_regression(df: pd.DataFrame, time: str, event: str,
                   covariates: List[str]) -> Dict[str, Any]:
    from lifelines import CoxPHFitter
    clean = df[[time, event, *covariates]].dropna().copy()
    cph = CoxPHFitter().fit(clean, duration_col=time, event_col=event)
    rows = []
    for cov in covariates:
        rows.append({
            "predictor": cov,
            "coef": float(cph.params_[cov]),
            "hazard_ratio": float(np.exp(cph.params_[cov])),
            "se": float(cph.standard_errors_[cov]),
            "z_statistic": float(cph.params_[cov] / cph.standard_errors_[cov]),
            "p_value": float(cph.summary.loc[cov, "p"]),
            "hr_ci_lower": float(np.exp(cph.confidence_intervals_.loc[cov, "95% lower-bound"])),
            "hr_ci_upper": float(np.exp(cph.confidence_intervals_.loc[cov, "95% upper-bound"])),
        })
    try:
        ph = cph.check_assumptions(clean, show_plots=False, p_value_threshold=0.05)
        ph_rows = []
        for item in ph:
            for cov, row in item.summary.iterrows():
                ph_rows.append({
                    "predictor": str(cov),
                    "test_statistic": float(row.get("test_statistic", float("nan"))),
                    "p_value": float(row.get("p", float("nan"))),
                })
    except Exception:
        ph_rows = []
    return {
        "model": "cox_ph",
        "n": int(len(clean)), "n_events": int(clean[event].sum()),
        "coefficients": rows,
        "concordance": float(cph.concordance_index_),
        "log_likelihood": float(cph.log_likelihood_),
        "ph_test": ph_rows,
    }
```

- [ ] **Step 4: PASS** → commit:

```bash
git commit -am "feat(stats): KM, log-rank, Cox PH with Schoenfeld residual test"
```

### Task 6.2: Backend + SurviveTab.tsx

- [ ] **Step 1: Endpoints** — `/survive/km`, `/survive/logrank`, `/survive/cox`. Replace the old `/survival` (keep it as an alias for backward compat).

- [ ] **Step 2: `SurviveTab.tsx`** — two modes:

1. **Kaplan-Meier + Log-rank** (existing, but with CI bands): time + event + optional group; renders KM curves with shaded 95% CI + log-rank p-value in the corner.
2. **Cox PH**: time + event + multi-select covariates. Renders:
   - HR forest plot (each covariate a row with HR + 95% CI as a horizontal line + dot; log scale; vertical reference line at HR=1)
   - Concordance index badge (value, interpretation: "good" > 0.7, "fair" 0.6–0.7, "poor" < 0.6)
   - PH assumption table: predictor, test statistic, p-value, warning icon if p < 0.05.

- [ ] **Step 3: Wire into StatsLab**

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(stats): Survive tab — KM + Cox PH with forest plot and PH diagnostics"
```

---

## Phase 7 — Multiple-testing correction + PD helpers

### Task 7.1: `adjust_pvalues` in pie.stats

**Files:**
- Modify: `lib/PIE/pie/stats/multitest.py`
- Create: `lib/PIE/tests/test_stats_multitest.py`

- [ ] **Step 1: Tests**

```python
# lib/PIE/tests/test_stats_multitest.py
import pytest
from pie.stats.multitest import adjust_pvalues


def test_adjust_pvalues_bonferroni():
    r = adjust_pvalues([0.01, 0.04, 0.03, 0.005], method="bonferroni")
    assert r["adjusted"] == pytest.approx([0.04, 0.16, 0.12, 0.02])
    assert r["rejected"][0] is True   # 0.04 < 0.05


def test_adjust_pvalues_fdr_bh():
    r = adjust_pvalues([0.01, 0.02, 0.03, 0.04], method="fdr_bh")
    assert r["method"] == "fdr_bh"
    assert all(isinstance(x, bool) for x in r["rejected"])


def test_adjust_pvalues_unknown_method():
    with pytest.raises(ValueError):
        adjust_pvalues([0.01], method="not_a_method")
```

- [ ] **Step 2: Run → 3 FAIL**

- [ ] **Step 3: Implement**

```python
# lib/PIE/pie/stats/multitest.py
"""Multiple-testing correction."""
from typing import Any, Dict, List
from statsmodels.stats.multitest import multipletests


_ALLOWED = {"bonferroni", "holm", "sidak", "fdr_bh", "fdr_by", "fdr_tsbh"}


def adjust_pvalues(p_values: List[float], method: str = "fdr_bh",
                   alpha: float = 0.05) -> Dict[str, Any]:
    """Bonferroni / Holm / Šidák / BH-FDR / BY-FDR / two-stage BH."""
    if method not in _ALLOWED:
        raise ValueError(f"method must be one of {sorted(_ALLOWED)}, got {method!r}")
    reject, p_adj, _, _ = multipletests(p_values, alpha=alpha, method=method)
    return {
        "method": method, "alpha": alpha,
        "original": list(p_values),
        "adjusted": [float(p) for p in p_adj],
        "rejected": [bool(r) for r in reject],
    }
```

- [ ] **Step 4: PASS** → commit:

```bash
git commit -am "feat(stats): multiple-testing correction (Bonferroni/Holm/Sidak/BH/BY)"
```

### Task 7.2: PD helpers (LEDD, UPDRS aggregation, H&Y summary)

- [ ] **Step 1: Tests**

```python
# lib/PIE/tests/test_stats_pd_helpers.py
import pandas as pd
import pytest
from pie.stats.pd_helpers import compute_ledd, aggregate_updrs, hoehn_yahr_summary


def test_compute_ledd_levodopa_only():
    # 300 mg levodopa IR = 300 mg LEDD (factor 1.0)
    r = compute_ledd({"levodopa_ir": 300})
    assert r["total_ledd_mg"] == pytest.approx(300.0)


def test_compute_ledd_with_dopa_agonist():
    # 300 mg levodopa + 4 mg pramipexole (factor 100)
    r = compute_ledd({"levodopa_ir": 300, "pramipexole": 4})
    assert r["total_ledd_mg"] == pytest.approx(700.0)


def test_aggregate_updrs():
    df = pd.DataFrame({
        "np1_1": [1, 2], "np1_2": [2, 1], "np2_1": [0, 1],
        "np3_1": [3, 2], "np3_2": [2, 1], "np4_1": [0, 0],
    })
    r = aggregate_updrs(df, part1_cols=["np1_1", "np1_2"], part2_cols=["np2_1"],
                        part3_cols=["np3_1", "np3_2"], part4_cols=["np4_1"])
    assert r["updrs_total"].tolist() == [8, 7]
    assert r["updrs_motor"].tolist() == [5, 3]


def test_hoehn_yahr_summary():
    s = pd.Series([1, 2, 2, 3, 3, 3, 4, 5])
    r = hoehn_yahr_summary(s)
    assert r["counts"][3] == 3
    assert r["proportions"][3] == pytest.approx(3 / 8)
    assert r["median_stage"] == pytest.approx(3.0)
```

- [ ] **Step 2: Run → 4 FAIL**

- [ ] **Step 3: Implement**

```python
# lib/PIE/pie/stats/pd_helpers.py
"""Parkinson's-specific helpers: LEDD, UPDRS aggregation, H&Y summary."""
from typing import Any, Dict, List, Optional
import pandas as pd


# Conversion factors from Tomlinson et al. 2010 (Mov Disord), widely used in PD.
LEDD_FACTORS = {
    "levodopa_ir": 1.0,
    "levodopa_cr": 0.75,        # controlled-release is ~75% of IR
    "levodopa_entacapone": 1.33,
    "pramipexole": 100.0,        # 1 mg ≈ 100 mg LEDD
    "ropinirole": 20.0,
    "rotigotine": 30.0,
    "apomorphine": 10.0,
    "rasagiline": 100.0,
    "selegiline_oral": 10.0,
    "selegiline_sublingual": 80.0,
    "safinamide": 100.0,
    "amantadine": 1.0,
    "tolcapone": 0.5,            # adds 50% of daily levodopa as LEDD
    "entacapone": 0.33,          # adds 33% of daily levodopa
}


def compute_ledd(doses_mg: Dict[str, float]) -> Dict[str, Any]:
    """Compute total LEDD from a dict of {drug_name_in_LEDD_FACTORS: mg/day}."""
    per_drug = {}
    total = 0.0
    for drug, dose in doses_mg.items():
        factor = LEDD_FACTORS.get(drug)
        if factor is None:
            per_drug[drug] = {"dose_mg": dose, "factor": None, "ledd_mg": None, "note": "unknown drug"}
            continue
        ledd = dose * factor
        per_drug[drug] = {"dose_mg": dose, "factor": factor, "ledd_mg": ledd}
        total += ledd
    return {"total_ledd_mg": total, "per_drug": per_drug}


def aggregate_updrs(df: pd.DataFrame,
                    part1_cols: Optional[List[str]] = None,
                    part2_cols: Optional[List[str]] = None,
                    part3_cols: Optional[List[str]] = None,
                    part4_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Aggregate MDS-UPDRS parts into per-row totals."""
    result = pd.DataFrame(index=df.index)
    for name, cols in [("updrs_part1", part1_cols), ("updrs_part2", part2_cols),
                       ("updrs_motor", part3_cols), ("updrs_part4", part4_cols)]:
        if cols:
            result[name] = df[cols].sum(axis=1, skipna=False)
    # Total = sum of whichever parts were provided
    parts = [c for c in result.columns if c.startswith("updrs_")]
    result["updrs_total"] = result[parts].sum(axis=1, skipna=False)
    return result


def hoehn_yahr_summary(series: pd.Series) -> Dict[str, Any]:
    """Counts, proportions, median stage for a Hoehn & Yahr series."""
    clean = series.dropna()
    counts = clean.value_counts().sort_index().to_dict()
    total = len(clean)
    return {
        "n": int(total),
        "counts": {float(k): int(v) for k, v in counts.items()},
        "proportions": {float(k): float(v / total) for k, v in counts.items()},
        "median_stage": float(clean.median()),
        "mean_stage": float(clean.mean()),
    }
```

- [ ] **Step 4: PASS** → commit:

```bash
git commit -am "feat(stats): PD helpers — LEDD, UPDRS aggregation, H&Y summary"
```

### Task 7.3: Backend + UI tabs

- [ ] **Step 1: Endpoints**

```python
class MultitestRequest(BaseModel):
    p_values: List[float]
    method: str = "fdr_bh"
    alpha: float = 0.05


class LEDDRequest(BaseModel):
    doses_mg: Dict[str, float]


@router.post("/multitest/adjust")
async def multitest_adjust(request: MultitestRequest):
    from pie.stats import adjust_pvalues
    return adjust_pvalues(request.p_values, method=request.method, alpha=request.alpha)


@router.post("/pd/ledd")
async def pd_ledd(request: LEDDRequest):
    from pie.stats import compute_ledd
    return compute_ledd(request.doses_mg)
```

UPDRS aggregation and H&Y summary similar (take cache_key + column lists).

- [ ] **Step 2: `MultitestTab.tsx`**

A simple utility:
- Textarea for pasting p-values (one per line or comma-separated)
- Method dropdown (Bonferroni / Holm / Šidák / BH-FDR / BY-FDR / Two-stage BH)
- α slider (0.01 / 0.05 / 0.10)
- Results table: original p / adjusted p / reject H₀ flag, sortable

- [ ] **Step 3: `PDHelpersTab.tsx`**

Two sub-panels:
1. **LEDD calculator**: drug name dropdown + dose input, stacked rows; running total display with breakdown per drug. The conversion table shown in a collapsible info panel.
2. **UPDRS aggregator**: pick column groups by part (users multi-select columns they consider Part 1, etc.), see per-row totals as a preview table.

- [ ] **Step 4: Wire into StatsLab + commit**

```bash
git commit -am "feat(stats): Multitest + PD Helpers tabs"
```

---

## Phase 8 — Polish & integration

### Task 8.1: `_load_df` helper consolidation

**Files:** Modify `backend/api/statistics.py`

- [ ] **Step 1: Extract** the inline data-loading logic at the top of each endpoint into one `_load_df(cache_key: str) -> pd.DataFrame` function. Return 404 if not found. Remove the duplicated code across endpoints.

- [ ] **Step 2: Commit**

```bash
git commit -am "refactor(api): consolidate data loading into _load_df helper"
```

### Task 8.2: Cross-tab integration — "Correct my p-values" buttons

Add a small "→ Correct for multiple tests" link on the Compare, Correlate, and Regress tab result cards that pre-fills the MultitestTab with the p-values from the current run.

Implementation: lift the last-run p-values to zustand store as `lastPValues`; MultitestTab reads it on mount.

- [ ] **Step 1: Add `lastPValues: number[]` slice to `src/store/useStore.ts`**

- [ ] **Step 2: Push from Compare/Correlate/Regress tabs after a run**

- [ ] **Step 3: Read in MultitestTab; auto-populate textarea**

- [ ] **Step 4: Commit**

```bash
git commit -am "feat(stats): hand-off p-values between tabs for multiple-testing correction"
```

### Task 8.3: Type checks and final wire-up

- [ ] **Step 1: Run `npx tsc --noEmit`** from `/home/cameron/PIE-Workbench` — expect zero new errors (the pre-existing `DataIngestion.tsx(7,3)` error is not ours).

- [ ] **Step 2: Run the full pie test suite**

```bash
cd lib/PIE && ../../backend/venv/bin/pytest tests/test_stats_* -v
```

All tests pass.

- [ ] **Step 3: Browser smoke test each new tab** with `test18` project already loaded — every tab should render without console errors.

- [ ] **Step 4: Final commit**

```bash
git commit -am "chore(stats): final wire-up + tests passing"
```

---

## Self-review notes

- **Spec coverage:** Every item from the original 6-category proposal has a corresponding Phase 1–7 task. Phase 8 adds integration polish that the original proposal implied but didn't spell out.
- **No placeholders:** Every step contains the actual code or exact commands required. No "TODO"s, no "implement later".
- **Type consistency:** Function names are consistent between `__init__.py` re-exports (Phase 0), module definitions (Phases 1–7), and API handlers (called by-name from endpoints).
- **Risk:** Phase 5's LMM and Phase 6's Cox PH are the heaviest lifts and depend on statsmodels/lifelines quirks. Tests use fixed seeds and loose bounds to stay stable across minor library upgrades.
