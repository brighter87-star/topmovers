from __future__ import annotations
import os, time, datetime as dt
from typing import Iterable, Set
import pandas as pd
import requests

from src.utils.config import KEY
from src.data.getcommonstocklist import get_common_tickers  # ← 보통주 유니버스

# ──────────────────────────────────────────────
# 캐시 경로
RAW_DIR   = os.getenv("POLY_CACHE_DIR", "cache/polygon_grouped")          # 전 종목(원천) 캐시
COMMON_DIR= os.getenv("POLY_COMMON_DIR", "cache/polygon_commonstocks")    # (선택) 보통주만 별도 저장
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(COMMON_DIR, exist_ok=True)

# ──────────────────────────────────────────────
def _api_key() -> str:
    key = KEY
    if not key:
        raise RuntimeError("POLYGON_API_KEY 가 필요합니다.")
    return key

def _last_business_days(n: int) -> list[dt.date]:
    days, d = [], dt.date.today() - dt.timedelta(days=1)  # 어제부터 거꾸로
    while len(days) < n:
        if d.weekday() < 5:  # 월(0)~금(4)
            days.append(d)
        d -= dt.timedelta(days=1)
    return list(reversed(days))

def _raw_path(day: dt.date, adjusted: bool) -> str:
    tag = "adj1" if adjusted else "adj0"
    return os.path.join(RAW_DIR, f"{day.isoformat()}_{tag}.parquet")

def _common_path(day: dt.date, adjusted: bool) -> str:
    tag = "adj1" if adjusted else "adj0"
    # Hive 파티션처럼 날짜별 디렉터리 구성(읽을 때 빠름)
    ddir = os.path.join(COMMON_DIR, f"date={day.isoformat()}")
    os.makedirs(ddir, exist_ok=True)
    return os.path.join(ddir, f"common_{tag}.parquet")

# ──────────────────────────────────────────────
def _download_grouped_one_day(day: dt.date, adjusted: bool, key: str) -> pd.DataFrame:
    url = (
        f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{day:%Y-%m-%d}"
        f"?adjusted={'true' if adjusted else 'false'}&apiKey={key}"
    )
    for attempt in range(5):
        r = requests.get(url, timeout=30)
        if r.status_code == 429:              # 레이트리밋 → 백오프
            time.sleep(2 * (attempt + 1)); continue
        if r.status_code in (403, 404, 204):  # 휴일/권한 등 → 빈 DF
            return pd.DataFrame(columns=["ticker", "close"])
        if r.ok:
            rows = r.json().get("results", []) or []
            if not rows:
                return pd.DataFrame(columns=["ticker", "close"])
            df = pd.DataFrame(rows)
            return df.rename(columns={"T": "ticker", "c": "close"})[["ticker", "close"]]
        time.sleep(1.0 * (attempt + 1))
    return pd.DataFrame(columns=["ticker", "close"])

def _ensure_raw_cached(day: dt.date, adjusted: bool, key: str, call_count: int) -> tuple[str, int]:
    """하루치 '전 종목' 원천 데이터를 RAW_DIR에 저장. 무료 5콜/분 준수."""
    path = _raw_path(day, adjusted)
    if os.path.exists(path):
        return path, call_count
    if call_count > 0 and call_count % 5 == 0:      # 무료 플랜: 5콜마다 60s 대기
        print(f"[rate-limit] {call_count} calls so far → sleeping 60s...")
        time.sleep(60)
    df = _download_grouped_one_day(day, adjusted, key)
    if not df.empty:
        df.to_parquet(path, index=False)
    return path, call_count + 1

# ──────────────────────────────────────────────
# 1) 보통주 유니버스 로드
def load_common_universe() -> list[str]:
    # get_common_tickers()는 'ticker' 컬럼을 갖는 DF를 반환한다고 가정
    df = get_common_tickers()
    return df["ticker"].dropna().astype(str).str.upper().tolist()

# 2) (옵션) 하루치 '보통주만' 별도 parquet로 머티리얼라이즈
def materialize_common_for_day(day: dt.date, adjusted: bool = True) -> str | None:
    key = _api_key()
    raw_path, _ = _ensure_raw_cached(day, adjusted, key, call_count=0)
    if not os.path.exists(raw_path):
        return None
    all_df = pd.read_parquet(raw_path)  # (ticker, close)
    if all_df.empty:
        return None
    uni = set(load_common_universe())
    df = all_df[all_df["ticker"].isin(uni)].copy()
    if df.empty:
        return None
    df["date"] = day
    out = _common_path(day, adjusted)
    if not os.path.exists(out):         # 이미 있으면 덮어쓰기 피함
        df.to_parquet(out, index=False)
    return out

# 3) 최근 N영업일 보통주 종가 wide DF (원천 캐시에서 필터)
def fetch_close_common_wide(days: int = 30, adjusted: bool = True) -> pd.DataFrame:
    key = _api_key()
    uni: Set[str] = set(load_common_universe())
    dates = _last_business_days(days)
    frames: list[pd.DataFrame] = []
    call_count = 0

    for d in dates:
        raw_path, call_count = _ensure_raw_cached(d, adjusted, key, call_count)
        if not os.path.exists(raw_path):
            continue
        df = pd.read_parquet(raw_path)
        if df.empty:
            continue
        df = df[df["ticker"].isin(uni)].copy()
        if df.empty:
            continue
        df["date"] = d
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    wide = (
        pd.concat(frames, ignore_index=True)
          .pivot(index="date", columns="ticker", values="close")
          .sort_index()
    )
    wide.index.name = "date"
    return wide.loc[:, wide.count() >= 3]

# 4) 특정 티커 서브셋만 뽑고 싶을 때
def fetch_close_subset_wide(tickers: Iterable[str], days: int = 30, adjusted: bool = True) -> pd.DataFrame:
    """보통주 전체 캐시에서 필요한 티커만 wide로."""
    tickers = {t.strip().upper() for t in tickers if t}
    if not tickers:
        return pd.DataFrame()
    full = fetch_close_common_wide(days=days, adjusted=adjusted)
    cols = [c for c in tickers if c in full.columns]
    return full[cols].copy() if cols else pd.DataFrame()

# ──────────────────────────────────────────────
if __name__ == "__main__":
    # ① 최근 5영업일, 보통주 전체 wide DF (상관분석 등에서 바로 사용)
    df = fetch_close_common_wide(days=30, adjusted=True)
    print("wide shape:", df.shape)
    print(df.tail())

    # ② (선택) 하루치씩 보통주만 별도 parquet로 머티리얼라이즈 하고싶을 때:
    # for d in _last_business_days(5):
    #     print("saved:", materialize_common_for_day(d, adjusted=True))
