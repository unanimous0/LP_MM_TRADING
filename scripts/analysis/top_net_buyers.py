"""
최근 1개월 순매수금액 상위 30개 종목 분석

사용법:
    python scripts/analysis/top_net_buyers.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_connection


def get_top_net_buyers(days=30, limit=30):
    """
    최근 N일 동안 외국인과 기관투자자의 순매수금액 상위 종목 조회

    Args:
        days: 조회 기간 (거래일 기준)
        limit: 상위 N개 종목

    Returns:
        DataFrame: 순매수금액 상위 종목 정보
    """
    conn = get_connection()

    # 최신 날짜 조회
    latest_date_query = "SELECT MAX(trade_date) as max_date FROM investor_flows"
    latest_date = pd.read_sql(latest_date_query, conn).iloc[0]['max_date']

    print(f"\n{'='*80}")
    print(f"최근 1개월 순매수금액 상위 {limit}개 종목 분석")
    print(f"{'='*80}")
    print(f"[INFO] 최신 데이터 날짜: {latest_date}")
    print(f"[INFO] 조회 기간: 최근 {days} 거래일")
    print(f"{'='*80}\n")

    # 쿼리 실행
    query = f"""
    WITH ranked_data AS (
        SELECT
            s.stock_name,
            f.stock_code,
            m.market_name,
            SUM(f.foreign_net_amount) as foreign_net_total,
            SUM(f.institution_net_amount) as institution_net_total,
            SUM(f.foreign_net_amount + f.institution_net_amount) as combined_net_total,
            COUNT(DISTINCT f.trade_date) as trading_days,
            AVG(f.market_cap) as avg_market_cap
        FROM investor_flows f
        JOIN stocks s ON f.stock_code = s.stock_code
        JOIN markets m ON s.market_id = m.market_id
        WHERE f.trade_date >= (
            SELECT trade_date
            FROM (
                SELECT DISTINCT trade_date
                FROM investor_flows
                ORDER BY trade_date DESC
                LIMIT {days}
            )
            ORDER BY trade_date ASC
            LIMIT 1
        )
        GROUP BY f.stock_code, s.stock_name, m.market_name
    )
    SELECT
        stock_name as '종목명',
        stock_code as '종목코드',
        market_name as '시장',
        foreign_net_total as '외국인_순매수금액',
        institution_net_total as '기관_순매수금액',
        combined_net_total as '외국인+기관_순매수금액',
        avg_market_cap as '평균_시가총액',
        trading_days as '거래일수'
    FROM ranked_data
    ORDER BY combined_net_total DESC
    LIMIT {limit}
    """

    df = pd.read_sql(query, conn)
    conn.close()

    # 금액을 억원 단위로 변환
    # 주의: 금액은 원 단위로 DB에 저장되어 있음 (2026-02-09 마이그레이션 완료)
    df['외국인_순매수금액_억원'] = (df['외국인_순매수금액'] / 100_000_000).round(2)
    df['기관_순매수금액_억원'] = (df['기관_순매수금액'] / 100_000_000).round(2)
    df['외국인+기관_순매수금액_억원'] = (df['외국인+기관_순매수금액'] / 100_000_000).round(2)
    df['평균_시가총액_억원'] = (df['평균_시가총액'] / 100_000_000).round(2)

    return df


def print_results(df):
    """결과를 보기 좋게 출력"""

    print("=" * 120)
    print(f"{'순위':>4} {'종목명':<20} {'종목코드':<10} {'시장':<10} {'외국인(억)':<15} {'기관(억)':<15} {'합계(억)':<15} {'시가총액(억)':<15}")
    print("=" * 120)

    for idx, row in df.iterrows():
        rank = idx + 1
        print(f"{rank:>4} {row['종목명']:<20} {row['종목코드']:<10} {row['시장']:<10} "
              f"{row['외국인_순매수금액_억원']:>14,.0f} {row['기관_순매수금액_억원']:>14,.0f} "
              f"{row['외국인+기관_순매수금액_억원']:>14,.0f} {row['평균_시가총액_억원']:>14,.0f}")

    print("=" * 120)

    # 요약 통계
    print(f"\n[요약 통계]")
    print(f"  - 총 순매수금액 (외국인+기관): {df['외국인+기관_순매수금액_억원'].sum():,.0f} 억원")
    print(f"  - 평균 순매수금액: {df['외국인+기관_순매수금액_억원'].mean():,.0f} 억원")
    print(f"  - 외국인 순매수 총액: {df['외국인_순매수금액_억원'].sum():,.0f} 억원")
    print(f"  - 기관 순매수 총액: {df['기관_순매수금액_억원'].sum():,.0f} 억원")

    # 외국인 vs 기관 매수 패턴
    foreign_positive = (df['외국인_순매수금액_억원'] > 0).sum()
    institution_positive = (df['기관_순매수금액_억원'] > 0).sum()

    print(f"\n[매수 패턴]")
    print(f"  - 외국인 순매수 종목 수: {foreign_positive}개 / {len(df)}개")
    print(f"  - 기관 순매수 종목 수: {institution_positive}개 / {len(df)}개")


def main():
    # 최근 1개월 (약 20-22 거래일) 데이터 분석
    df = get_top_net_buyers(days=22, limit=30)

    # 결과 출력
    print_results(df)

    # CSV 저장 옵션
    output_file = project_root / 'data' / 'analysis_results' / 'top_net_buyers.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] 결과가 저장되었습니다: {output_file}")


if __name__ == '__main__':
    main()
