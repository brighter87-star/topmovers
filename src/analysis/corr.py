# src/analysis/corr.py
from __future__ import annotations

from collections.abc import Iterable
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# df_wide: index=날짜, columns=티커, values=종가 (fetch_close_common_wide 결과)
def _returns(df_wide: pd.DataFrame, use_log: bool = True) -> pd.DataFrame:
    df = df_wide.sort_index().astype(float)
    ret = np.log(df).diff() if use_log else df.pct_change()
    return ret.dropna(how="all")


def top_correlated_for(
    df_wide: pd.DataFrame,
    targets: Iterable[str],
    candidates: Optional[Iterable[str]] = None,
    topk: int = 10,
    method: str = "pearson",
    min_overlap: int = 30,  # 두 종목 간 겹치는 일수 최소
) -> pd.DataFrame:
    """Targets 목록의 각 종목에 대해, candidates 중 상관계수 상위 topk를 반환.
    반환 컬럼: [base, mate, corr, overlap, rank]
    """
    targets = [t.strip().upper() for t in targets if t]
    if not len(df_wide.index):
        return pd.DataFrame(columns=["base", "mate", "corr", "overlap", "rank"])

    rets = _returns(df_wide)
    if candidates is None:
        candidates = list(rets.columns)
    else:
        candidates = [c.strip().upper() for c in candidates if c]

    # 상관 행렬(피어슨, pairwise)
    corr = rets[candidates].corr(method=method, min_periods=min_overlap)

    out_rows: List[Tuple[str, str, float, int, int]] = []
    for base in targets:
        if base not in corr.index:
            continue
        # base와의 상관계수 시리즈
        s = corr[base].drop(labels=[base], errors="ignore").dropna()
        if s.empty:
            continue
        # 실제 겹치는 샘플 수 계산(정렬 후 topk만 계산 비용 아끼기)
        top = s.sort_values(ascending=False).head(topk)
        for rank, (mate, cval) in enumerate(top.items(), start=1):
            # pairwise overlap 계산
            ov = int(rets[[base, mate]].dropna().shape[0])
            if ov < min_overlap:
                continue
            out_rows.append((base, mate, float(cval), ov, rank))

    res = pd.DataFrame(out_rows, columns=["base", "mate", "corr", "overlap", "rank"])
    return res.sort_values(["base", "rank"]).reset_index(drop=True)
