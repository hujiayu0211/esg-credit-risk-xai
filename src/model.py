"""XGBoost regression pipeline with randomized hyperparameter search.

The full preprocessing + model workflow is wrapped in a single sklearn
Pipeline so that imputation is fit only on training folds during CV,
avoiding leakage.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def build_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("xgb", XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,  # search parallelizes across folds; keep the model single-threaded
        )),
    ])


PARAM_DISTRIBUTIONS = {
    "xgb__n_estimators": randint(200, 900),
    "xgb__max_depth": randint(3, 8),
    "xgb__learning_rate": uniform(0.01, 0.19),
    "xgb__subsample": uniform(0.6, 0.4),
    "xgb__colsample_bytree": uniform(0.6, 0.4),
    "xgb__min_child_weight": randint(1, 10),
    "xgb__reg_alpha": uniform(0.0, 1.0),
    "xgb__reg_lambda": uniform(0.5, 2.5),
}


def tune(X_train, y_train, n_iter: int = 28, cv: int = 5, seed: int = 42,
         n_jobs: int = 1) -> RandomizedSearchCV:
    """RandomizedSearchCV over the XGBoost pipeline (default: 28 candidate
    configurations x 5-fold CV = 140 fits).

    n_jobs defaults to 1 (serial). On a low-memory host, nested parallelism
    between the search and XGBoost oversubscribes cores and can exhaust RAM;
    serial is slower but robust. Raise n_jobs if you have headroom.
    """
    search = RandomizedSearchCV(
        estimator=build_pipeline(seed),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=seed,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }
