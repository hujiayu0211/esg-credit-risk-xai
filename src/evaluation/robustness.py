"""Robustness of the ESG effect across distance-to-default definitions.

Re-runs the baseline-vs-full comparison using each of the three DD measures
(KMV, Merton, Bharath-Shumway) as the regression target, to show the
incremental value of ESG is not an artefact of one particular DD algorithm.

Run:  python -m src.evaluation.robustness
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score

from ..data_prep import winsorize
from ..model import build_pipeline
from ..ingest import schema as S

SEED = 42


def _risk_from_dd(df, dd_col):
    dd = df[dd_col]
    return 50 + 10 * (-(dd - dd.mean()) / dd.std())


def run(path="data/panel_real.csv", n_splits=5):
    raw = pd.read_csv(path)
    base_cols = S.FEATURE_GROUPS["financial"] + S.FEATURE_GROUPS["market"]
    full_cols = base_cols + S.FEATURE_GROUPS["esg"]

    params = {}
    try:
        st = json.load(open("search_state.json"))
        params = min(st["results"], key=lambda r: r["cv_rmse"])["params"]
    except FileNotFoundError:
        params = {"xgb__n_estimators": 500, "xgb__max_depth": 5}

    results = {}
    for dd_col in ["dd_kmv", "dd_merton", "dd_bhsh"]:
        df = raw[raw[dd_col].notna()].copy()
        df["_risk"] = _risk_from_dd(df, dd_col)
        df = winsorize(df, S.ALL_FEATURES + ["_risk"])
        y = df["_risk"].values
        groups = df["stock_code"].values
        gkf = GroupKFold(n_splits=n_splits)

        scores = {}
        for name, cols in [("baseline", base_cols), ("full", full_cols)]:
            X = df[cols].values
            r2s = []
            for tr, te in gkf.split(X, y, groups):
                m = build_pipeline(SEED); m.set_params(**params)
                m.fit(X[tr], y[tr])
                r2s.append(r2_score(y[te], m.predict(X[te])))
            scores[name] = float(np.mean(r2s))
        results[dd_col] = {
            "baseline_r2": round(scores["baseline"], 4),
            "full_r2": round(scores["full"], 4),
            "esg_r2_gain": round(scores["full"] - scores["baseline"], 4),
        }
    return results


if __name__ == "__main__":
    import os
    res = run()
    os.makedirs("outputs", exist_ok=True)
    json.dump(res, open("outputs/robustness.json", "w"), indent=2)
    print("ESG incremental value across DD definitions (firm-grouped CV R2):")
    print(f"  {'target':10s} {'baseline':>10s} {'full':>8s} {'ESG gain':>10s}")
    for k, v in res.items():
        print(f"  {k:10s} {v['baseline_r2']:>10.4f} {v['full_r2']:>8.4f} {v['esg_r2_gain']:>+10.4f}")
