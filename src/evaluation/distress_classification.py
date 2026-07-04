"""Financial-distress classification (auxiliary task).

Predicts whether a firm enters ST / *ST status in year t+1 from year-t
features. This is a genuine forward-looking prediction (target is a future
event), so it complements the distance-to-default regression and reports the
metrics credit practitioners actually use: ROC-AUC and PR-AUC.

Also runs the ESG ablation on this task, so the incremental value of ESG is
tested against a second, independent target.

Run:  python -m src.evaluation.distress_classification
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from ..data_prep import winsorize
from ..ingest import schema as S

SEED = 42


def _clf(scale_pos_weight):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBClassifier(
            objective="binary:logistic", eval_metric="auc",
            tree_method="hist", n_estimators=400, max_depth=4,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight, random_state=SEED, n_jobs=1)),
    ])


def run(path="data/panel_real.csv", n_splits=5):
    df = pd.read_csv(path)
    df = df[df["distress_next"].notna()].copy()
    df = winsorize(df, S.ALL_FEATURES)
    y = df["distress_next"].astype(int).values
    groups = df["stock_code"].values

    base_cols = S.FEATURE_GROUPS["financial"] + S.FEATURE_GROUPS["market"]
    full_cols = base_cols + S.FEATURE_GROUPS["esg"]
    spw = (y == 0).sum() / max((y == 1).sum(), 1)

    gkf = GroupKFold(n_splits=n_splits)
    out = {}
    for name, cols in [("baseline", base_cols), ("full", full_cols)]:
        X = df[cols].values
        aucs, aps = [], []
        for tr, te in gkf.split(X, y, groups):
            m = _clf(spw); m.fit(X[tr], y[tr])
            proba = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], proba))
            aps.append(average_precision_score(y[te], proba))
        out[name] = {"roc_auc": round(float(np.mean(aucs)), 4),
                     "roc_auc_std": round(float(np.std(aucs)), 4),
                     "pr_auc": round(float(np.mean(aps)), 4),
                     "n_features": len(cols)}

    out["esg_auc_gain"] = round(out["full"]["roc_auc"] - out["baseline"]["roc_auc"], 4)
    out["positive_rate"] = round(float(y.mean()), 4)
    out["n_obs"] = int(len(y))
    return out


if __name__ == "__main__":
    import os
    res = run()
    os.makedirs("outputs", exist_ok=True)
    json.dump(res, open("outputs/distress_classification.json", "w"), indent=2)
    print("Distress classification (predict ST/*ST next year, firm-grouped CV):")
    print(f"  positive rate: {res['positive_rate']*100:.2f}%  (n={res['n_obs']:,})")
    print(f"  baseline (14 feat): ROC-AUC={res['baseline']['roc_auc']}  PR-AUC={res['baseline']['pr_auc']}")
    print(f"  full     (18 feat): ROC-AUC={res['full']['roc_auc']}  PR-AUC={res['full']['pr_auc']}")
    print(f"  ESG ROC-AUC gain: {res['esg_auc_gain']:+.4f}")
