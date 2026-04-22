"""Smoke tests for the new /api/statistics endpoints.

Uses FastAPI's TestClient so the cache lives in the same process and we can
register synthetic fixtures via the cache module directly.
"""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from main import app  # type: ignore
from api import cache as C


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def fixtures():
    rng = np.random.default_rng(0)
    n = 500
    age = rng.normal(65, 10, n)
    dd = rng.normal(5, 3, n)
    cohort = rng.choice(["PD", "HC", "Prodromal"], n, p=[0.5, 0.3, 0.2])
    cohort_bin = np.where(cohort == "PD", 1, 0)
    updrs = 10 + 0.4 * age + 2 * dd + np.where(cohort == "PD", 5, 0) + rng.normal(0, 5, n)
    moca = 30 - 0.2 * dd - np.where(cohort == "PD", 3, 0) + rng.normal(0, 1.5, n)
    missing_col = rng.normal(0, 1, n).astype(float)
    missing_col[:100] = np.nan

    df = pd.DataFrame({
        "age": age, "disease_dur": dd, "cohort": cohort, "cohort_bin": cohort_bin,
        "updrs": updrs, "moca": moca, "missing_col": missing_col,
    })
    ck = "test_stats_main"
    C.store(ck, df)

    # Longitudinal
    rows = []
    for s in range(50):
        b = rng.normal(20, 5)
        for t in range(4):
            rows.append({"patno": s, "visit": t,
                         "updrs_long": b + 2.5 * t + rng.normal(0, 1)})
    long_df = pd.DataFrame(rows)
    ck_long = "test_stats_long"
    C.store(ck_long, long_df)

    # Survival
    rng2 = np.random.default_rng(1)
    n_s = 300
    grp = rng2.choice([0, 1], n_s)
    scale = np.where(grp == 0, 1 / 0.05, 1 / 0.15)
    t = rng2.exponential(scale)
    ev = (t < 40).astype(int)
    t = np.minimum(t, 40)
    s_df = pd.DataFrame({"time": t, "event": ev, "group": grp, "age2": rng2.normal(65, 10, n_s)})
    ck_surv = "test_stats_surv"
    C.store(ck_surv, s_df)

    yield {"main": ck, "long": ck_long, "surv": ck_surv}

    for k in (ck, ck_long, ck_surv):
        C.delete(k)


# -------- Describe --------------------------------------------------------

def test_describe_summary(client, fixtures):
    r = client.post("/api/statistics/describe/summary",
                    json={"cache_key": fixtures["main"], "variables": ["age", "updrs"]})
    assert r.status_code == 200
    body = r.json()
    assert body["age"]["n"] == 500
    assert "mean" in body["age"]


def test_describe_normality(client, fixtures):
    r = client.post("/api/statistics/describe/normality",
                    json={"cache_key": fixtures["main"], "variable": "age", "test": "shapiro"})
    assert r.status_code == 200
    body = r.json()
    assert "statistic" in body and "is_normal" in body


def test_describe_missingness(client, fixtures):
    r = client.post("/api/statistics/describe/missingness",
                    json={"cache_key": fixtures["main"], "variables": ["age", "missing_col"]})
    assert r.status_code == 200
    body = r.json()
    assert body["per_column"]["missing_col"]["n_missing"] == 100


# -------- Compare ---------------------------------------------------------

def test_compare_two_group(client, fixtures):
    r = client.post("/api/statistics/compare/two_group",
                    json={"cache_key": fixtures["main"], "variable": "updrs",
                          "grouping_variable": "cohort_bin", "test": "independent_t"})
    assert r.status_code == 200
    body = r.json()
    assert "p_value" in body and "cohens_d" in body


def test_compare_multi_group(client, fixtures):
    r = client.post("/api/statistics/compare/multi_group",
                    json={"cache_key": fixtures["main"], "variable": "updrs",
                          "grouping_variable": "cohort", "posthoc": "tukey"})
    assert r.status_code == 200
    body = r.json()
    assert "main" in body
    assert len(body["posthoc"]["pairwise"]) == 3


def test_compare_categorical(client, fixtures):
    r = client.post("/api/statistics/compare/categorical",
                    json={"cache_key": fixtures["main"], "variable_a": "cohort",
                          "variable_b": "cohort_bin"})
    assert r.status_code == 200
    body = r.json()
    assert "contingency" in body and "p_value" in body


# -------- Correlate -------------------------------------------------------

def test_correlate_partial(client, fixtures):
    r = client.post("/api/statistics/correlate/partial",
                    json={"cache_key": fixtures["main"], "x": "age", "y": "updrs",
                          "covariates": ["disease_dur"]})
    assert r.status_code == 200
    assert "r" in r.json()


def test_correlate_matrix(client, fixtures):
    r = client.post("/api/statistics/correlate/matrix",
                    json={"cache_key": fixtures["main"],
                          "variables": ["age", "disease_dur", "updrs", "moca"]})
    assert r.status_code == 200
    body = r.json()
    assert "matrix" in body and "p_values_adjusted" in body


# -------- Regress ---------------------------------------------------------

def test_regress_linear(client, fixtures):
    r = client.post("/api/statistics/regress/linear",
                    json={"cache_key": fixtures["main"], "outcome": "updrs",
                          "predictors": ["age", "disease_dur"]})
    assert r.status_code == 200
    body = r.json()
    assert body["r_squared"] > 0.5
    assert "diagnostics" in body


def test_regress_logistic(client, fixtures):
    r = client.post("/api/statistics/regress/logistic",
                    json={"cache_key": fixtures["main"], "outcome": "cohort_bin",
                          "predictors": ["age", "updrs"]})
    assert r.status_code == 200
    body = r.json()
    assert "auc" in body


def test_regress_ancova(client, fixtures):
    r = client.post("/api/statistics/regress/ancova",
                    json={"cache_key": fixtures["main"], "outcome": "updrs",
                          "group": "cohort", "covariates": ["age", "disease_dur"]})
    assert r.status_code == 200
    body = r.json()
    assert any(e["source"] == "group" for e in body["effects"])


# -------- Longitudinal ----------------------------------------------------

def test_longitudinal_lmm(client, fixtures):
    r = client.post("/api/statistics/longitudinal/mixed_model",
                    json={"cache_key": fixtures["long"], "outcome": "updrs_long",
                          "fixed_effects": ["visit"], "group": "patno"})
    assert r.status_code == 200
    body = r.json()
    assert body["n_groups"] == 50
    visit = next(f for f in body["fixed_effects"] if f["predictor"] == "visit")
    assert abs(visit["estimate"] - 2.5) < 0.5


def test_longitudinal_change(client, fixtures):
    r = client.post("/api/statistics/longitudinal/change_from_baseline",
                    json={"cache_key": fixtures["long"], "subject": "patno",
                          "time": "visit", "outcome": "updrs_long", "baseline_time": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["n_subjects"] == 50


# -------- Survive ---------------------------------------------------------

def test_survive_km(client, fixtures):
    r = client.post("/api/statistics/survive/km",
                    json={"cache_key": fixtures["surv"], "time": "time",
                          "event": "event", "group": "group"})
    assert r.status_code == 200
    body = r.json()
    assert "timeline" in body


def test_survive_logrank(client, fixtures):
    r = client.post("/api/statistics/survive/logrank",
                    json={"cache_key": fixtures["surv"], "time": "time",
                          "event": "event", "group": "group"})
    assert r.status_code == 200
    body = r.json()
    assert body["p_value"] < 0.01


def test_survive_cox(client, fixtures):
    r = client.post("/api/statistics/survive/cox",
                    json={"cache_key": fixtures["surv"], "time": "time",
                          "event": "event", "covariates": ["group", "age2"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["coefficients"]) == 2


# -------- Multitest + PD helpers -----------------------------------------

def test_multitest_adjust(client):
    r = client.post("/api/statistics/multitest/adjust",
                    json={"p_values": [0.01, 0.04, 0.005, 0.3], "method": "fdr_bh"})
    assert r.status_code == 200
    assert len(r.json()["rejected"]) == 4


def test_pd_ledd(client):
    r = client.post("/api/statistics/pd/ledd",
                    json={"doses_mg": {"levodopa_ir": 300, "pramipexole": 4}})
    assert r.status_code == 200
    assert abs(r.json()["total_ledd_mg"] - 700) < 1e-6


def test_pd_ledd_factors(client):
    r = client.get("/api/statistics/pd/ledd_factors")
    assert r.status_code == 200
    assert "levodopa_ir" in r.json()["factors"]


def test_pd_hoehn_yahr(client, fixtures):
    r = client.post("/api/statistics/pd/hoehn_yahr",
                    json={"cache_key": fixtures["surv"], "variable": "group"})
    assert r.status_code == 200
    assert "counts" in r.json()
