# src/analysis/correlation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Literal
import numpy as np
import pandas as pd
from src.data.price import fetch_daily_close_yf

@dataclass
class CorrConfig:
    days: int = 30
    min_overlap: int = 10
    method: Literal["pearson", "spearman"] = "pearson"
    topn: int = 10
    auto_adjust: bool = True

def _daily_returns(close_df: pd.DataFrame) -> pd.DataFrame:
    return close_df.pct_change().dropna(how="all")

def top_correlated(base: str, universe: Iterable[str], cfg: CorrConfig = CorrConfig()) -> pd.DataFrame:
    base = base.upper()
    tickers = {t.upper() for t in universe if t}
    tickers.add(base)

    close = fetch_daily_close_yf(tickers, days=cfg.days, auto_adjust=cfg.auto_adjust)
    if close.empty or base not in close.columns:
        return pd.DataFrame(columns=["ticker","corr","overlap_n","beta"])

    rets = _daily_returns(close)
    base_ret = rets[base].dropna()

    rows = []
    for t in rets.columns:
        if t == base:
            continue
        pair = pd.concat([base_ret, rets[t]], axis=1, join="inner").dropna()
        if len(pair) < cfg.min_overlap:
            continue

        x, y = pair[base].values, pair[t].values
        if cfg.method == "spearman":
            corr = pd.Series(x).rank().corr(pd.Series(y).rank())
        else:
            corr = float(np.corrcoef(x, y)[0, 1])

        vx = float(np.var(x, ddof=1))
        beta = float(np.cov(x, y, ddof=1)[0, 1] / vx) if vx > 0 else np.nan
        rows.append((t, corr, len(pair), beta))

    if not rows:
        return pd.DataFrame(columns=["ticker","corr","overlap_n","beta"])

    return (pd.DataFrame(rows, columns=["ticker","corr","overlap_n","beta"])
              .sort_values("corr", ascending=False)
              .head(cfg.topn)
              .reset_index(drop=True))

