#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable
from typing import List, Tuple

import numpy as np
import pandas as pd

# 너의 모듈: 캐시된 parquet에서 wide DF 만들어옴
from src.data.price_polygon import fetch_close_common_wide


def _returns(df_wide: pd.DataFrame, use_log: bool = True) -> pd.DataFrame:
    df = df_wide.sort_index().astype(float)
    rets = np.log(df).diff() if use_log else df.pct_change()
    return rets.dropna(how="all")


def _top_for_targets(
    df_wide: pd.DataFrame,
    targets: Iterable[str],
    topk: int,
    min_overlap: int,
    method: str = "pearson",
) -> pd.DataFrame:
    """Targets 각 종목에 대해 상관 상위 topk 행 반환"""
    targets = [t.strip().upper() for t in targets if t]
    if df_wide.empty:
        return pd.DataFrame(columns=["base", "mate", "corr", "overlap", "rank"])

    rets = _returns(df_wide)
    # pairwise min_periods 사용
    corr = rets.corr(method=method, min_periods=min_overlap)

    rows: List[Tuple[str, str, float, int, int]] = []
    for base in targets:
        if base not in corr.columns:
            print(f"[warn] {base} not in data columns (skipped).")
            continue
        s = corr[base].drop(labels=[base], errors="ignore").dropna()
        if s.empty:
            continue
        top = s.sort_values(ascending=False).head(topk)
        for rank, (mate, cval) in enumerate(top.items(), start=1):
            ov = int(rets[[base, mate]].dropna().shape[0])
            if ov < min_overlap:
                continue
            rows.append((base, mate, float(cval), ov, rank))

    res = pd.DataFrame(rows, columns=["base", "mate", "corr", "overlap", "rank"])
    return res.sort_values(["base", "rank"]).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(
        description="Find top correlated tickers using cached Parquet (cache/polygon_grouped).",
    )
    p.add_argument("tickers", nargs="+", help="Target tickers (e.g., NVDA AMD TSLA)")
    p.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of recent business days (default: 90)",
    )
    p.add_argument(
        "--topk", type=int, default=10, help="Top-K mates per ticker (default: 10)",
    )
    p.add_argument(
        "--min-overlap",
        type=int,
        default=30,
        help="Minimum overlapping days (default: 30)",
    )
    p.add_argument(
        "--pct",
        action="store_true",
        help="Use simple percent returns instead of log returns",
    )
    p.add_argument("--method", choices=["pearson", "spearman"], default="pearson")
    p.add_argument("--no-adjusted", action="store_true", help="Use unadjusted prices")
    p.add_argument("--csv", type=str, default="", help="Path to save result as CSV")
    args = p.parse_args()

    adjusted = not args.no_adjusted
    # 캐시만 읽는 wide DF (네 함수가 캐시를 사용)
    wide = fetch_close_common_wide(days=args.days, adjusted=adjusted)
    if wide.empty:
        print("[error] No data loaded. Check your cache or days range.")
        return

    # 상관 계산
    res = _top_for_targets(
        df_wide=wide,
        targets=args.tickers,
        topk=args.topk,
        min_overlap=args.min_overlap,
        method=args.method,
    )

    if res.empty:
        print("[info] No pairs met the criteria.")
        return

    # 콘솔 출력 정돈
    for base in res["base"].unique():
        sub = res[res["base"] == base].copy()
        print(
            f"\n=== {base} | top {len(sub)} (method={args.method}, days={args.days}, adjusted={adjusted}) ===",
        )
        # 보기 좋게 반올림
        sub["corr"] = sub["corr"].round(4)
        print(sub[["rank", "mate", "corr", "overlap"]].to_string(index=False))

    if args.csv:
        res.to_csv(args.csv, index=False)
        print(f"\n[saved] {args.csv}")


if __name__ == "__main__":
    main()
