# Data

## What ships with this repository

Nothing licensed. The modelling panel is built from **CSMAR** and **Huazheng ESG** data, which cannot be redistributed. This directory contains only:

- `sample_synthetic.csv` (generated on demand by `python -m src.data_generator`), a schema-compatible synthetic panel for smoke-testing the code.

Real raw exports go under `data/raw/` (git-ignored) and are never committed.

## Rebuilding the real panel

With CSMAR + Huazheng access, export the tables below, drop them anywhere under `data/raw/`, and run:

```bash
python -m src.ingest.build_panel --raw data/raw --out data/panel_real.csv
```

The builder locates each table by filename (recursively), translates raw field codes to English names via `src/ingest/schema.py`, computes all 18 features and the targets, merges on (stock_code, year), applies the sample filters, and writes the analysis panel.

### Time window

Export **2014-2025** for every table; the analysis window is **2015-2025**. The extra 2014 year is a lag base for revenue growth and 12-month momentum and is dropped from the final panel. Monthly and daily tables must start at **January 2014**, not year-end 2014.

### Required tables and fields

**1. Balance sheet: `FS_Combas.csv`** (annual, consolidated)
Stkcd, Accper, Typrep, A001101000 (cash), A001123000 (inventory, net), A001100000 (current assets), A001000000 (total assets), A002100000 (current liabilities), A002000000 (total liabilities), A003000000 (total equity).

**2. Income statement: `FS_Comins.csv`** (annual, consolidated)
Stkcd, Accper, Typrep, B001101000 (operating revenue), B001201000 (operating cost), B001211101 (interest expense within finance costs), B001000000 (total profit), B002000000 (net income).

**3. Monthly market value: `TRD_Mnth.csv`**
Stkcd, Trdmnt, Msmvttl (total market value, thousands RMB = total shares x close). Export monthly return (Mretwd) in the same or a second `TRD_Mnth` file; the builder auto-detects which columns each copy carries.

**4. Monthly turnover: `LIQ_TOVER_M.csv`**
Stkcd, Trdmnt, ToverTlM (monthly turnover on total shares, percent), Days (trading days).

**5. Daily return: `TRD_Dalyr.csv`** (may be sharded as TRD_Dalyr1..N.csv)
Stkcd, Trddt, Dretwd (return incl. cash-dividend reinvestment). Used to compute annualised volatility (daily std x sqrt(252)).

**6. Distance-to-default: `BDT_FinDistMertonDD.csv`** (annual)
Symbol, Enddate, STPT (ST/*ST/PT flag), IsSuspend, DDBhsh, DDmerton, DDKMV. Supplies the regression target (inverted DD) and the ST/*ST distress flag used to build the t+1 classification target.

**7. Huazheng ESG (xlsx)**
Columns: stock code, year, composite rating + score, E/S/G rating + score, CSRC industry. Any file matching `*esg*.xlsx` or `huazheng*.xlsx` under `data/raw/` is picked up. Use a single rating provider for all four ESG features so pillar and composite scores are on one scale.

### Feature construction

| Feature | Definition |
|---|---|
| leverage | total liabilities / total assets |
| current_ratio | current assets / current liabilities |
| quick_ratio | (current assets - inventory) / current liabilities |
| roa | net income / total assets |
| roe | net income / total equity |
| gross_margin | (revenue - operating cost) / revenue |
| interest_coverage | EBIT / interest expense, where EBIT = total profit + interest expense |
| asset_turnover | revenue / total assets |
| cash_ratio | cash / current liabilities |
| revenue_growth | (revenue_t - revenue_{t-1}) / \|revenue_{t-1}\| |
| stock_volatility | std(daily return within year) x sqrt(252) |
| ln_market_cap | ln(total market value in RMB); Msmvttl is in thousands |
| momentum_12m | product(1 + monthly return) - 1 over the year |
| turnover_rate | mean monthly turnover (total shares) over the year |
| esg_score, env_score, soc_score, gov_score | Huazheng composite and pillar scores |

### Targets

- `credit_risk_score` (regression): the KMV distance-to-default, standardised and inverted so higher = riskier.
- `distress_next` (classification): 1 if the firm is ST/*ST in year t+1, else 0.

### Standard filters applied by the builder

Consolidated statements only (Typrep = A); annual periods only (Accper ends 12-31); drop financial industry (CSRC industry text); drop contemporaneous ST/*ST firm-years; restrict to the analysis window; require a valid DD. Continuous variables are winsorized at 1%/99% in `src/data_prep.py`.

### Known coverage notes

- Interest expense (B001211101) is only populated from 2018 onward in CSMAR, so `interest_coverage` has meaningful missingness in earlier years; median imputation inside the pipeline handles this.
- ESG and DD coverage is near-complete over 2015-2025 in the sources used here; verify for your own extract.
