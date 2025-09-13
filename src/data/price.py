# src/data/price.py
from __future__ import annotations
import datetime as dt
from typing import Iterable, List
import pandas as pd
import yfinance as yf

def _chunk(lst: List[str], n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def fetch_daily_close_yf(
    tickers: Iterable[str],
    days: int = 30,
    auto_adjust: bool = True,
    batch_size: int = 200,   # 대량 티커 대비 배치 다운로드
) -> pd.DataFrame:
    tickers = sorted({t.strip().upper() for t in tickers if t})
    if not tickers:
        return pd.DataFrame()

    end = dt.date.today()
    start = end - dt.timedelta(days=days)

    closes = []
    for chunk in _chunk(tickers, batch_size):
        df = yf.download(
            chunk,
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=auto_adjust,
            progress=False,
            group_by="ticker",
        )
        if df.empty:
            continue

        if "Close" in df.columns:  # 단일 티커
            closes.append(df[["Close"]].rename(columns={"Close": chunk[0]}))
        else:                      # 멀티 티커
            sub = []
            for t in chunk:
                try:
                    sub.append(df[(t, "Close")].rename(t))
                except KeyError:
                    continue
            if sub:
                closes.append(pd.concat(sub, axis=1))

    if not closes:
        return pd.DataFrame()

    out = pd.concat(closes, axis=1)
    out.index.name = "date"
    out = out.sort_index()
    # 관측치 너무 적은 종목 제거(옵션)
    out = out.loc[:, out.count() >= 3]
    return out

