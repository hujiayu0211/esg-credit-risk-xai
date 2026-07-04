"""Smoke tests that run without licensed data (using the synthetic sample)."""
import numpy as np
import pandas as pd

from src.data_generator import generate_panel
from src.data_prep import load_dataset, split, winsorize
from src.ingest import schema as S
from src.model import build_pipeline, evaluate
from src.xai_analyzer import XGBoostXAIAnalyzer


def test_feature_count():
    assert len(S.ALL_FEATURES) == 18
    assert len(set(S.ALL_FEATURES)) == 18


def test_synthetic_panel_has_all_columns(tmp_path):
    df = generate_panel(n_firms=50, n_years=3, seed=0)
    for col in S.ALL_FEATURES + ["credit_risk_score"]:
        assert col in df.columns


def test_winsorize_clips(tmp_path):
    df = generate_panel(n_firms=80, n_years=3, seed=1)
    w = winsorize(df, ["leverage"])
    assert w["leverage"].max() <= df["leverage"].quantile(0.99) + 1e-9


def test_pipeline_fit_predict(tmp_path):
    p = tmp_path / "s.csv"
    generate_panel(n_firms=120, n_years=3, seed=2).to_csv(p, index=False)
    X, y, _ = load_dataset(str(p))
    Xtr, Xte, ytr, yte = split(X, y)
    model = build_pipeline(42)
    model.set_params(xgb__n_estimators=60, xgb__max_depth=3)
    model.fit(Xtr, ytr)
    m = evaluate(model, Xte, yte)
    assert set(m) == {"rmse", "mae", "r2"}
    assert np.isfinite(m["rmse"])


def test_shap_dimensions(tmp_path):
    p = tmp_path / "s.csv"
    generate_panel(n_firms=120, n_years=3, seed=3).to_csv(p, index=False)
    X, y, _ = load_dataset(str(p))
    Xtr, Xte, ytr, yte = split(X, y)
    model = build_pipeline(42)
    model.set_params(xgb__n_estimators=60, xgb__max_depth=3)
    model.fit(Xtr, ytr)
    az = XGBoostXAIAnalyzer(model, Xte, output_dir=str(tmp_path / "fig"),
                            interaction_sample=50, seed=42)
    assert az.shap_values.shape == (len(Xte), 18)
    # explainer baseline should be close to the training target mean scale
    assert np.isfinite(az.explainer.expected_value)
