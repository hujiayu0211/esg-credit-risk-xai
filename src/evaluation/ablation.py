"""ESG ablation study.

Answers the project's core question: do ESG scores carry *incremental*
predictive information about credit risk, beyond financial and market signals?

Design:
  - baseline model : financial + market features only (14 features)
  - full model     : baseline + 4 ESG features (18 features)
  - both tuned identically, evaluated out-of-sample
  - firm-grouped CV so the same firm never spans train/test (avoids the
    panel leakage that inflates naive random-split scores)
  - paired test across CV folds on per-fold RMSE to judge whether the ESG
    improvement is statistically meaningful

Run:  python -m src.evaluation.ablation
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

from ..data_prep import load_dataset
from ..model import build_pipeline
from ..ingest import schema as S

SEED = 42
BEST_PARAMS_FILE = "search_state.json"


def _best_params():
    try:
        st = json.load(open(BEST_PARAMS_FILE))
        return min(st["results"], key=lambda r: r["cv_rmse"])["params"]
    except FileNotFoundError:
        return {"xgb__n_estimators": 500, "xgb__max_depth": 5,
                "xgb__learning_rate": 0.05}


def run_ablation(path="data/panel_real.csv", n_splits=5):
    df = pd.read_csv(path)
    from ..data_prep import winsorize
    df = winsorize(df, S.ALL_FEATURES + ["credit_risk_score"])

    y = df["credit_risk_score"].values
    groups = df["stock_code"].values
    base_cols = S.FEATURE_GROUPS["financial"] + S.FEATURE_GROUPS["market"]
    full_cols = base_cols + S.FEATURE_GROUPS["esg"]
    params = _best_params()

    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for name, cols in [("baseline", base_cols), ("full", full_cols)]:
        X = df[cols].values
        fold_rmse, fold_r2 = [], []
        for tr, te in gkf.split(X, y, groups):
            m = build_pipeline(SEED); m.set_params(**params)
            m.fit(X[tr], y[tr])
            pred = m.predict(X[te])
            fold_rmse.append(np.sqrt(mean_squared_error(y[te], pred)))
            fold_r2.append(r2_score(y[te], pred))
        rows.append({"model": name, "n_features": len(cols),
                     "rmse_mean": np.mean(fold_rmse), "rmse_std": np.std(fold_rmse),
                     "r2_mean": np.mean(fold_r2), "_rmse_folds": fold_rmse})

    base, full = rows[0], rows[1]
    # paired t-test on per-fold RMSE (baseline - full); positive means ESG helps
    diff = np.array(base["_rmse_folds"]) - np.array(full["_rmse_folds"])
    t_stat, p_val = stats.ttest_rel(base["_rmse_folds"], full["_rmse_folds"])

    result = {
        "baseline_rmse": round(base["rmse_mean"], 4),
        "baseline_r2": round(base["r2_mean"], 4),
        "full_rmse": round(full["rmse_mean"], 4),
        "full_r2": round(full["r2_mean"], 4),
        "rmse_reduction": round(base["rmse_mean"] - full["rmse_mean"], 4),
        "rmse_reduction_pct": round(100 * (base["rmse_mean"] - full["rmse_mean"]) / base["rmse_mean"], 3),
        "paired_t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_val), 4),
        "n_splits": n_splits,
        "evaluation": "firm-grouped GroupKFold (no firm spans train/test)",
    }
    return result


if __name__ == "__main__":
    res = run_ablation()
    import os
    os.makedirs("outputs", exist_ok=True)
    json.dump(res, open("outputs/ablation.json", "w"), indent=2)
    print("ESG ablation (firm-grouped CV):")
    print(f"  baseline (14 feat): RMSE={res['baseline_rmse']}  R2={res['baseline_r2']}")
    print(f"  full     (18 feat): RMSE={res['full_rmse']}  R2={res['full_r2']}")
    print(f"  RMSE reduction from ESG: {res['rmse_reduction']} "
          f"({res['rmse_reduction_pct']}%)")
    print(f"  paired t-test: t={res['paired_t_stat']}, p={res['p_value']}")
