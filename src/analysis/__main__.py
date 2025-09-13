# src/analysis/__main__.py
import argparse
from src.analysis.correlation import top_correlated, CorrConfig
from src.data.getcommonstocklist import get_common_tickers  # 기존 파일 재사용

def main():
    p = argparse.ArgumentParser(description="최근 1개월 상관계수 TopN (유니버스=나스닥+others 보통주)")
    p.add_argument("--base", required=True, help="기준 티커 (예: AAPL)")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--topn", type=int, default=10)
    p.add_argument("--min-overlap", type=int, default=10)
    p.add_argument("--spearman", action="store_true", help="Spearman 사용(순위상관)")
    args = p.parse_args()

    # 당신이 이미 구현해둔 필터(ETF 제외, 워런트/우선주 등 제외)가 적용된 전체 보통주
    common_df = get_common_tickers()           # 반드시 'ticker' 컬럼 포함
    universe = common_df["ticker"].dropna().astype(str).str.upper().tolist()

    method = "spearman" if args.spearman else "pearson"
    cfg = CorrConfig(days=args.days, topn=args.topn, min_overlap=args.min_overlap, method=method)

    res = top_correlated(args.base, universe, cfg)
    if res.empty:
        print("결과가 비어 있습니다. 티커/기간을 확인하세요.")
    else:
        print(res.to_string(index=False))

if __name__ == "__main__":
    main()

