"""Build the analysis panel from raw CSMAR / Huazheng tables.

Pipeline:
  1. Balance sheet + income statement  -> 10 financial ratios + EBIT
  2. Monthly market cap                -> ln_market_cap
  3. Monthly return                    -> momentum_12m
  4. Monthly turnover                  -> turnover_rate
  5. Daily return                      -> annualised volatility
  6. Merton DD table                   -> regression target + ST distress flag
  7. Huazheng ESG                      -> 4 ESG features + industry
  -> merge on (stock_code, year), apply sample filters, write panel CSV.

All raw field codes are translated to English names via src.ingest.schema.
Run:  python -m src.ingest.build_panel --raw data/raw --out data/panel_real.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from . import schema as S

TRADING_DAYS = 252


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _code(s: pd.Series) -> pd.Series:
    """Normalise stock codes to 6-digit zero-padded strings."""
    return s.astype(str).str.extract(r"(\d+)")[0].str.zfill(6)


def _read_csv(path: str, usecols=None) -> pd.DataFrame:
    for enc in ("utf-8-sig", "gbk", "utf-8"):
        try:
            return pd.read_csv(path, usecols=usecols, encoding=enc,
                               low_memory=False)
        except (UnicodeDecodeError, LookupError):
            continue
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _find(raw_dir: str, filename: str) -> list[str]:
    """Recursively locate all copies/shards of a raw table by base name."""
    hits = glob.glob(os.path.join(raw_dir, "**", filename), recursive=True)
    return sorted(h for h in hits if "[DES]" not in h)


# ----------------------------------------------------------------------
# 1. financial ratios
# ----------------------------------------------------------------------
def build_financials(raw_dir: str) -> pd.DataFrame:
    bs = _read_csv(_find(raw_dir, "FS_Combas.csv")[0]).rename(columns=S.BALANCE_SHEET)
    inc = _read_csv(_find(raw_dir, "FS_Comins.csv")[0]).rename(columns=S.INCOME_STATEMENT)

    # Consolidated statements only (Typrep == 'A'), annual only (period ends 12-31)
    for df in (bs, inc):
        df.query("report_type == 'A'", inplace=True)
    bs = bs[bs["period"].str.endswith("12-31")].copy()
    inc = inc[inc["period"].str.endswith("12-31")].copy()

    bs["stock_code"] = _code(bs["stock_code"])
    inc["stock_code"] = _code(inc["stock_code"])
    bs["year"] = bs["period"].str[:4].astype(int)
    inc["year"] = inc["period"].str[:4].astype(int)

    bs = bs.drop_duplicates(["stock_code", "year"])
    inc = inc.drop_duplicates(["stock_code", "year"])

    fin = bs.merge(inc, on=["stock_code", "year"], how="inner",
                   suffixes=("", "_inc"))

    # EBIT = pretax profit + interest expense
    ebit = fin["pretax_profit"] + fin["interest_expense"].fillna(0)

    eps = 1e-6
    fin["leverage"] = fin["total_liabilities"] / fin["total_assets"]
    fin["current_ratio"] = fin["current_assets"] / fin["current_liabilities"].replace(0, np.nan)
    fin["quick_ratio"] = ((fin["current_assets"] - fin["inventory"])
                          / fin["current_liabilities"].replace(0, np.nan))
    fin["roa"] = fin["net_income"] / fin["total_assets"]
    fin["roe"] = fin["net_income"] / fin["total_equity"].replace(0, np.nan)
    fin["gross_margin"] = (fin["revenue"] - fin["operating_cost"]) / fin["revenue"].replace(0, np.nan)
    fin["interest_coverage"] = ebit / fin["interest_expense"].replace(0, np.nan)
    fin["asset_turnover"] = fin["revenue"] / fin["total_assets"]
    fin["cash_ratio"] = fin["cash"] / fin["current_liabilities"].replace(0, np.nan)

    # revenue growth needs prior-year revenue (2014 acts as the lag base)
    fin = fin.sort_values(["stock_code", "year"])
    fin["revenue_lag"] = fin.groupby("stock_code")["revenue"].shift(1)
    fin["revenue_growth"] = (fin["revenue"] - fin["revenue_lag"]) / fin["revenue_lag"].abs().replace(0, np.nan)

    cols = ["stock_code", "year"] + S.FEATURE_GROUPS["financial"]
    return fin[cols]


# ----------------------------------------------------------------------
# 2-4. monthly-derived features
# ----------------------------------------------------------------------
def build_market_monthly(raw_dir: str) -> pd.DataFrame:
    # market cap (year-end month = December)
    mv = _read_csv(_find(raw_dir, "TRD_Mnth.csv")[0])
    # Some TRD_Mnth exports carry Msmvttl, others carry Mretwd; detect columns.
    frames_cap, frames_ret = [], []
    for path in _find(raw_dir, "TRD_Mnth.csv"):
        d = _read_csv(path)
        if "Msmvttl" in d.columns:
            frames_cap.append(d.rename(columns=S.MONTHLY_MKTCAP)
                              [["stock_code", "month", "market_cap_k"]])
        if "Mretwd" in d.columns:
            frames_ret.append(d.rename(columns=S.MONTHLY_RETURN)
                              [["stock_code", "month", "monthly_return"]])

    cap = pd.concat(frames_cap).drop_duplicates(["stock_code", "month"])
    cap["stock_code"] = _code(cap["stock_code"])
    cap["year"] = cap["month"].str[:4].astype(int)
    dec = cap[cap["month"].str.endswith("-12")].copy()
    # market_cap_k is in thousands RMB
    dec["ln_market_cap"] = np.log(dec["market_cap_k"].replace(0, np.nan) * 1_000)
    ln_mktcap = dec[["stock_code", "year", "ln_market_cap"]]

    # 12-month momentum = compounded return over the calendar year
    ret = pd.concat(frames_ret).drop_duplicates(["stock_code", "month"])
    ret["stock_code"] = _code(ret["stock_code"])
    ret["year"] = ret["month"].str[:4].astype(int)
    mom = (ret.groupby(["stock_code", "year"])["monthly_return"]
           .apply(lambda r: np.prod(1 + r.values) - 1)
           .reset_index(name="momentum_12m"))

    # turnover: average monthly turnover across the year
    tov = _read_csv(_find(raw_dir, "LIQ_TOVER_M.csv")[0]).rename(columns=S.MONTHLY_TURNOVER)
    tov["stock_code"] = _code(tov["stock_code"])
    tov["year"] = tov["month"].str[:4].astype(int)
    turnover = (tov.groupby(["stock_code", "year"])["turnover_pct"]
                .mean().reset_index(name="turnover_rate"))

    out = ln_mktcap.merge(mom, on=["stock_code", "year"], how="outer") \
                   .merge(turnover, on=["stock_code", "year"], how="outer")
    return out


# ----------------------------------------------------------------------
# 5. daily volatility
# ----------------------------------------------------------------------
def build_volatility(raw_dir: str) -> pd.DataFrame:
    frames = []
    for path in _find(raw_dir, "TRD_Dalyr.csv") + _find(raw_dir, "TRD_Dalyr1.csv") \
            + _find(raw_dir, "TRD_Dalyr2.csv") + _find(raw_dir, "TRD_Dalyr3.csv") \
            + _find(raw_dir, "TRD_Dalyr4.csv") + _find(raw_dir, "TRD_Dalyr5.csv"):
        d = _read_csv(path, usecols=["Stkcd", "Trddt", "Dretwd"]).rename(columns=S.DAILY_RETURN)
        d["stock_code"] = _code(d["stock_code"])
        d["year"] = d["date"].str[:4].astype(int)
        frames.append(d[["stock_code", "year", "daily_return"]])
    daily = pd.concat(frames, ignore_index=True)
    vol = (daily.groupby(["stock_code", "year"])["daily_return"]
           .std().reset_index(name="daily_std"))
    vol["stock_volatility"] = vol["daily_std"] * np.sqrt(TRADING_DAYS)
    return vol[["stock_code", "year", "stock_volatility"]]


# ----------------------------------------------------------------------
# 6. targets from Merton DD table
# ----------------------------------------------------------------------
def build_targets(raw_dir: str) -> pd.DataFrame:
    dd = _read_csv(_find(raw_dir, "BDT_FinDistMertonDD.csv")[0]).rename(columns=S.MERTON_DD)
    dd["stock_code"] = _code(dd["stock_code"])
    dd["year"] = dd["period"].str[:4].astype(int)
    dd = dd.drop_duplicates(["stock_code", "year"])

    # regression target: credit risk = negative distance-to-default
    # (higher DD = safer, so risk = -DD; rescaled to a positive 0-100-ish scale)
    dd = dd.sort_values(["stock_code", "year"])
    # classification target: does the firm enter ST/*ST next year?
    dd["distress_next"] = dd.groupby("stock_code")["is_st"].shift(-1)

    keep = ["stock_code", "year", "dd_kmv", "dd_merton", "dd_bhsh",
            "is_st", "distress_next"]
    return dd[keep]


# ----------------------------------------------------------------------
# 7. ESG
# ----------------------------------------------------------------------
def build_esg(raw_dir: str) -> pd.DataFrame:
    xls = glob.glob(os.path.join(raw_dir, "**", "*esg*.xlsx"), recursive=True) \
        + glob.glob(os.path.join(raw_dir, "**", "*ESG*.xlsx"), recursive=True) \
        + glob.glob(os.path.join(raw_dir, "**", "huazheng*.xlsx"), recursive=True)
    esg = pd.read_excel(xls[0]).rename(columns=S.ESG)
    esg["stock_code"] = _code(esg["stock_code"])
    esg["year"] = esg["year"].astype(int)
    esg = esg.drop_duplicates(["stock_code", "year"])
    keep = ["stock_code", "year", "esg_score", "env_score", "soc_score",
            "gov_score", "csrc_industry"]
    return esg[[c for c in keep if c in esg.columns]]


# ----------------------------------------------------------------------
# assemble
# ----------------------------------------------------------------------
FINANCIAL_INDUSTRY_KEYWORDS = ("金融", "货币金融", "资本市场", "保险", "银行",
                               "其他金融")


def build_panel(raw_dir: str, start_year: int = 2015, end_year: int = 2025):
    fin = build_financials(raw_dir)
    mkt = build_market_monthly(raw_dir)
    vol = build_volatility(raw_dir)
    tgt = build_targets(raw_dir)
    esg = build_esg(raw_dir)

    panel = (fin
             .merge(mkt, on=["stock_code", "year"], how="inner")
             .merge(vol, on=["stock_code", "year"], how="inner")
             .merge(tgt, on=["stock_code", "year"], how="inner")
             .merge(esg, on=["stock_code", "year"], how="inner"))

    n0 = len(panel)
    report = {"after_merge": n0}

    # --- sample filters -------------------------------------------------
    # (a) drop financial industry
    if "csrc_industry" in panel.columns:
        is_fin = panel["csrc_industry"].fillna("").str.contains(
            "|".join(FINANCIAL_INDUSTRY_KEYWORDS))
        panel = panel[~is_fin]
        report["after_drop_financial"] = len(panel)

    # (b) drop ST / *ST firm-years (contemporaneous distress)
    panel = panel[panel["is_st"] == 0]
    report["after_drop_st"] = len(panel)

    # (c) restrict to analysis window
    panel = panel[(panel["year"] >= start_year) & (panel["year"] <= end_year)]
    report["after_window"] = len(panel)

    # (d) require the regression target
    panel = panel[panel["dd_kmv"].notna()]
    report["after_require_target"] = len(panel)

    # regression target: invert DD to a risk score, standardised to mean 50
    dd = panel["dd_kmv"]
    panel["credit_risk_score"] = 50 + 10 * (-(dd - dd.mean()) / dd.std())

    panel = panel.sort_values(["stock_code", "year"]).reset_index(drop=True)
    return panel, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/panel_real.csv")
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()

    panel, report = build_panel(args.raw, args.start, args.end)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    panel.to_csv(args.out, index=False)

    print("Sample construction:")
    for k, v in report.items():
        print(f"  {k:28s} {v:>8,}")
    print(f"\nFinal panel: {len(panel):,} firm-year obs, "
          f"{panel['stock_code'].nunique():,} firms, "
          f"years {panel['year'].min()}-{panel['year'].max()}")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
