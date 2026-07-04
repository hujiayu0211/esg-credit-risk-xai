"""Data loading and preparation.

Works identically on the synthetic sample and on a real CSMAR extract, as
long as the input CSV follows the column schema documented in data/README.md.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .ingest.schema import ALL_FEATURES

TARGET = "credit_risk_score"


def winsorize(df: pd.DataFrame, cols: list[str], lower: float = 0.01,
              upper: float = 0.99) -> pd.DataFrame:
    """Winsorize columns at the given quantiles (standard in the
    empirical accounting/finance literature to limit outlier influence)."""
    out = df.copy()
    for col in cols:
        lo, hi = out[col].quantile([lower, upper])
        out[col] = out[col].clip(lo, hi)
    return out


def load_dataset(path: str, winsor: bool = True):
    """Load a firm-year panel CSV and return (X, y, meta)."""
    df = pd.read_csv(path)
    missing = [c for c in ALL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    if winsor:
        df = winsorize(df, ALL_FEATURES + [TARGET])

    X = df[ALL_FEATURES]
    y = df[TARGET]
    meta = df[[c for c in ("firm_id", "year") if c in df.columns]]
    return X, y, meta


def split(X, y, test_size: float = 0.2, seed: int = 42):
    """Random train/test split.

    Note: for a strict out-of-time evaluation on real panel data, split by
    `year` instead (train on early years, test on the latest year).
    """
    return train_test_split(X, y, test_size=test_size, random_state=seed)
