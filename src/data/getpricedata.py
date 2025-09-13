import os, requests, pandas as pd
from src.data.getcommonstocklist import get_common_tickers
from src.utils import config

def getpriceall(date_str: str):
    url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date_str}?adjusted=true&apiKey={config.KEY}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json().get("results", []))

df_today  = getpriceall("2025-09-11")[["T","c","v","vw"]].rename(columns={"T":"ticker","c":"close"})
df_prev   = getpriceall("2025-09-10")[["T","c"]].rename(columns={"T":"ticker","c":"prev_close"})
df = df_today.merge(df_prev, on="ticker", how="left")
df["pct"] = (df["close"]/df["prev_close"] - 1)*100

if __name__ == "__main__":
    common = get_common_tickers()
    df.merge(common, on="ticker")
    print(df[df['pct'] > 55])
