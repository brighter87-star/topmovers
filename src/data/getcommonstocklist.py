from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
NAS_PATH = HERE / "nasdaqlisted.txt"
OTH_PATH = HERE / "otherlisted.txt"

def read_body_lines(p: Path):
    body = []
    for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not ln or ln.startswith("Symbol|") or ln.startswith("ACT Symbol|"):
            continue
        if line.startswith("File Creation Time"):
            continue
        body.append(ln)
    return body

def load_symbols(path: str):
    df = pd.read_csv(path, sep="|", dtype=str)
    # 공통 컬럼 이름 맞추기
    if "Symbol" in df.columns:
        df = df.rename(columns={"Symbol":"ticker"})
    elif "ACT Symbol" in df.columns:
        df = df.rename(columns={"ACT Symbol":"ticker"})
    # 필터링
    df = df[(df["ETF"]=="N") & (df["Test Issue"]=="N")]
    return df[["ticker","Security Name"]]

def get_common_tickers() -> pd.DataFrame:
    nasdaq = load_symbols(NAS_PATH)
    other  = load_symbols(OTH_PATH)
    tickers = pd.concat([nasdaq, other], ignore_index=True)
    block_words = ["Warrant","Right","Unit","Preferred","Depositary","ETN","Trust","SPAC","Bond"]
    mask = ~tickers["Security Name"].str.contains("|".join(block_words), na=False)
    common = tickers[mask]

    return common

if __name__ == "__main__":
    print(get_common_tickers())
