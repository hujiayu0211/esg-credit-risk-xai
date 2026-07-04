"""Schema definitions mapping raw CSMAR/Huazheng field codes to clean,
English feature names.

This module is the single source of truth for how the raw vendor tables map
onto the modelling variables. Keeping the mapping here (rather than scattered
through the ingestion code) makes the field lineage auditable and lets the
pipeline run on any CSMAR extract that follows the standard table schemas.

Raw source tables (standard CSMAR / Huazheng exports):
  FS_Combas          balance sheet
  FS_Comins          income statement
  TRD_Mnth           monthly stock return / market value
  LIQ_TOVER_M        monthly turnover
  TRD_Dalyr          daily stock return
  BDT_FinDistMertonDD  Merton distance-to-default + ST/*ST flag
  Huazheng ESG       annual ESG ratings and scores
"""

from __future__ import annotations

# --- Balance sheet: FS_Combas -------------------------------------------
BALANCE_SHEET = {
    "Stkcd": "stock_code",
    "Accper": "period",
    "Typrep": "report_type",
    "A001101000": "cash",                # 货币资金
    "A001123000": "inventory",           # 存货净额
    "A001100000": "current_assets",      # 流动资产合计
    "A001000000": "total_assets",        # 资产总计
    "A002100000": "current_liabilities", # 流动负债合计
    "A002000000": "total_liabilities",   # 负债合计
    "A003000000": "total_equity",        # 所有者权益合计
}

# --- Income statement: FS_Comins ----------------------------------------
INCOME_STATEMENT = {
    "Stkcd": "stock_code",
    "Accper": "period",
    "Typrep": "report_type",
    "B001101000": "revenue",             # 营业收入
    "B001201000": "operating_cost",      # 营业成本
    "B001211101": "interest_expense",    # 其中:利息费用(财务费用)
    "B001000000": "pretax_profit",       # 利润总额
    "B002000000": "net_income",          # 净利润
}

# --- Monthly market value: TRD_Mnth (with Msmvttl) ----------------------
MONTHLY_MKTCAP = {
    "Stkcd": "stock_code",
    "Trdmnt": "month",
    "Msmvttl": "market_cap_k",           # 月个股总市值 (unit: thousands RMB)
}

# --- Monthly return: TRD_Mnth (return-only export) ----------------------
MONTHLY_RETURN = {
    "Stkcd": "stock_code",
    "Trdmnt": "month",
    "Mretwd": "monthly_return",          # 考虑现金红利再投资的月个股回报率
}

# --- Monthly turnover: LIQ_TOVER_M --------------------------------------
MONTHLY_TURNOVER = {
    "Stkcd": "stock_code",
    "Trdmnt": "month",
    "ToverTlM": "turnover_pct",          # 月换手率(总股数)(%)
    "Days": "trading_days",              # 交易天数
}

# --- Daily return: TRD_Dalyr --------------------------------------------
DAILY_RETURN = {
    "Stkcd": "stock_code",
    "Trddt": "date",
    "Dretwd": "daily_return",            # 考虑现金红利再投资的日个股回报率
}

# --- Merton DD + distress flag: BDT_FinDistMertonDD ---------------------
MERTON_DD = {
    "Symbol": "stock_code",
    "Enddate": "period",
    "STPT": "is_st",                     # 当年是否 ST/*ST/PT
    "IsSuspend": "is_suspended",
    "DDBhsh": "dd_bhsh",                 # Bharath-Shumway DD
    "DDmerton": "dd_merton",             # Merton DD
    "DDKMV": "dd_kmv",                   # KMV DD
}

# --- Huazheng ESG (Chinese headers) -------------------------------------
ESG = {
    "证券代码": "stock_code",
    "年份": "year",
    "综合评级": "esg_rating",
    "综合得分": "esg_score",
    "E评级": "env_rating",
    "E得分": "env_score",
    "S评级": "soc_rating",
    "S得分": "soc_score",
    "G评级": "gov_rating",
    "G得分": "gov_score",
    "证监会行业新": "csrc_industry",
}

# The 18 modelling features, grouped.
FEATURE_GROUPS = {
    "financial": [
        "leverage", "current_ratio", "quick_ratio", "roa", "roe",
        "gross_margin", "interest_coverage", "asset_turnover",
        "cash_ratio", "revenue_growth",
    ],
    "market": [
        "stock_volatility", "ln_market_cap", "momentum_12m", "turnover_rate",
    ],
    "esg": [
        "esg_score", "env_score", "soc_score", "gov_score",
    ],
}
ALL_FEATURES = (FEATURE_GROUPS["financial"]
                + FEATURE_GROUPS["market"]
                + FEATURE_GROUPS["esg"])

# Regression target (primary) and classification target (auxiliary).
TARGET_REG = "dd_kmv"        # higher DD = safer; we invert to a risk score
TARGET_REG_ALT = ["dd_merton", "dd_bhsh"]   # robustness
TARGET_CLS = "distress_next"  # firm becomes ST/*ST in t+1
