"""Synthetic data generator for the ESG-augmented credit risk pipeline.

Generates a firm-year panel that mimics the structure of CSMAR/Wind data for
Chinese A-share listed firms. The data-generating process embeds realistic
economic relationships (leverage raises risk, profitability lowers it, ESG
effects are stronger for highly levered firms, etc.) so that the downstream
SHAP analysis recovers interpretable patterns.

The synthetic sample exists solely so the pipeline is runnable end-to-end
without access to licensed data. See data/README.md for instructions on
reconstructing the real dataset from CSMAR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = {
    "financial": [
        "leverage",            # total debt / total assets
        "current_ratio",       # current assets / current liabilities
        "quick_ratio",         # (current assets - inventory) / current liabilities
        "roa",                 # net income / total assets
        "roe",                 # net income / shareholders' equity
        "gross_margin",        # gross profit / revenue
        "interest_coverage",   # EBIT / interest expense
        "asset_turnover",      # revenue / total assets
        "cash_ratio",          # cash / current liabilities
        "revenue_growth",      # YoY revenue growth
    ],
    "market": [
        "stock_volatility",    # annualised daily return volatility
        "ln_market_cap",       # log market capitalisation
        "momentum_12m",        # trailing 12-month return
        "turnover_rate",       # annual share turnover
    ],
    "esg": [
        "esg_score",           # composite ESG score (0-100)
        "env_score",           # environmental pillar (0-100)
        "soc_score",           # social pillar (0-100)
        "gov_score",           # governance pillar (0-100)
    ],
}

ALL_FEATURES = FEATURES["financial"] + FEATURES["market"] + FEATURES["esg"]
TARGET = "credit_risk_score"


def generate_panel(n_firms: int = 600, n_years: int = 4, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic firm-year panel with n_firms * n_years rows."""
    rng = np.random.default_rng(seed)
    n = n_firms * n_years

    firm_id = np.repeat(np.arange(1, n_firms + 1), n_years)
    year = np.tile(np.arange(2020, 2020 + n_years), n_firms)

    # Latent firm quality drives cross-feature correlation (persistent within firm)
    quality = np.repeat(rng.normal(0, 1, n_firms), n_years) + rng.normal(0, 0.3, n)

    # --- Market indicators -------------------------------------------------
    ln_market_cap = 22.5 + 1.1 * quality + rng.normal(0, 0.8, n)
    stock_volatility = np.clip(0.42 - 0.05 * quality + rng.normal(0, 0.08, n), 0.12, 0.9)
    momentum_12m = np.clip(0.05 + 0.08 * quality + rng.normal(0, 0.25, n), -0.7, 1.5)
    turnover_rate = np.clip(rng.lognormal(1.0, 0.5, n), 0.3, 15)

    # --- Financial ratios ---------------------------------------------------
    leverage = np.clip(0.45 - 0.06 * quality + rng.normal(0, 0.13, n), 0.03, 0.95)
    roa = np.clip(0.045 + 0.03 * quality + rng.normal(0, 0.03, n), -0.15, 0.25)
    roe = np.clip(roa / np.maximum(1 - leverage, 0.1) + rng.normal(0, 0.02, n), -0.4, 0.6)
    gross_margin = np.clip(0.28 + 0.05 * quality + rng.normal(0, 0.09, n), 0.02, 0.75)
    interest_coverage = np.clip(
        np.exp(1.3 + 0.5 * quality + rng.normal(0, 0.6, n)), 0.1, 80
    )
    current_ratio = np.clip(1.8 + 0.3 * quality - 1.2 * (leverage - 0.45) + rng.normal(0, 0.4, n), 0.3, 8)
    quick_ratio = np.clip(current_ratio * rng.uniform(0.55, 0.85, n), 0.15, 7)
    cash_ratio = np.clip(quick_ratio * rng.uniform(0.25, 0.6, n), 0.02, 4)
    asset_turnover = np.clip(rng.lognormal(-0.6, 0.4, n), 0.08, 3.5)
    revenue_growth = np.clip(0.08 + 0.05 * quality + rng.normal(0, 0.18, n), -0.5, 1.2)

    # --- ESG scores (larger, higher-quality firms disclose/score better) ----
    esg_base = 50 + 8 * quality + 2.5 * (ln_market_cap - 22.5) + rng.normal(0, 8, n)
    env_score = np.clip(esg_base + rng.normal(0, 7, n), 5, 98)
    soc_score = np.clip(esg_base + rng.normal(0, 7, n), 5, 98)
    gov_score = np.clip(esg_base + rng.normal(0, 6, n), 5, 98)
    esg_score = np.clip(0.35 * env_score + 0.3 * soc_score + 0.35 * gov_score
                        + rng.normal(0, 2, n), 5, 98)

    # --- Data-generating process for credit risk ----------------------------
    # Higher = riskier. Embeds nonlinearity (saturating interest-coverage
    # effect) and an ESG x leverage interaction: ESG matters more for
    # highly levered firms.
    esg_c = (esg_score - 50) / 25          # centred/scaled ESG
    lev_c = (leverage - 0.45) / 0.15       # centred/scaled leverage

    risk = (
        50
        + 8.0 * lev_c
        - 90.0 * roa
        + 22.0 * (stock_volatility - 0.42)
        - 6.0 * np.tanh(interest_coverage / 6)      # saturating benefit
        - 2.2 * (ln_market_cap - 22.5)              # size effect
        - 2.8 * esg_c                               # ESG main effect
        - 2.5 * esg_c * lev_c                       # ESG x leverage interaction
        - 3.0 * np.clip(revenue_growth, -0.2, 0.4)
        - 0.8 * (current_ratio - 1.8)
        + rng.normal(0, 4.5, n)                     # idiosyncratic noise
    )
    credit_risk_score = np.clip(risk, 0, 100)

    df = pd.DataFrame({
        "firm_id": firm_id,
        "year": year,
        "leverage": leverage,
        "current_ratio": current_ratio,
        "quick_ratio": quick_ratio,
        "roa": roa,
        "roe": roe,
        "gross_margin": gross_margin,
        "interest_coverage": interest_coverage,
        "asset_turnover": asset_turnover,
        "cash_ratio": cash_ratio,
        "revenue_growth": revenue_growth,
        "stock_volatility": stock_volatility,
        "ln_market_cap": ln_market_cap,
        "momentum_12m": momentum_12m,
        "turnover_rate": turnover_rate,
        "esg_score": esg_score,
        "env_score": env_score,
        "soc_score": soc_score,
        "gov_score": gov_score,
        TARGET: credit_risk_score,
    })

    # Inject realistic missingness (ESG coverage is incomplete in practice)
    for col in FEATURES["esg"]:
        mask = rng.random(n) < 0.06
        df.loc[mask, col] = np.nan
    return df.round(4)


if __name__ == "__main__":
    panel = generate_panel()
    panel.to_csv("data/sample_synthetic.csv", index=False)
    print(f"Generated {len(panel)} firm-year observations -> data/sample_synthetic.csv")
