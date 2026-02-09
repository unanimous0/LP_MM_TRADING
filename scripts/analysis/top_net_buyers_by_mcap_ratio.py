"""
최근 1개월 시가총액 대비 순매수 비중 상위 30개 종목 분석

시가총액 비중 = (순매수금액 / 평균 시가총액) × 100

사용법:
    python scripts/analysis/top_net_buyers_by_mcap_ratio.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_connection


def get_top_net_buyers_by_mcap_ratio(days=30, limit=30):
    """
    최근 N일 동안 시가총액 대비 순매수 비중 상위 종목 조회

    Args:
        days: 조회 기간 (거래일 기준)
        limit: 상위 N개 종목

    Returns:
        DataFrame: 시가총액 대비 순매수 비중 상위 종목 정보
    """
    conn = get_connection()

    # 최신 날짜 조회
    latest_date_query = "SELECT MAX(trade_date) as max_date FROM investor_flows"
    latest_date = pd.read_sql(latest_date_query, conn).iloc[0]['max_date']

    print(f"\n{'='*80}")
    print(f"최근 1개월 시가총액 대비 순매수 비중 상위 {limit}개 종목 분석")
    print(f"{'='*80}")
    print(f"[INFO] 최신 데이터 날짜: {latest_date}")
    print(f"[INFO] 조회 기간: 최근 {days} 거래일")
    print(f"[INFO] 비중 계산식: (순매수금액 / 평균시가총액) × 100")
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
            AVG(f.market_cap) as avg_market_cap,
            COUNT(DISTINCT f.trade_date) as trading_days
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
        AND f.market_cap IS NOT NULL
        AND f.market_cap > 0
        GROUP BY f.stock_code, s.stock_name, m.market_name
        HAVING avg_market_cap > 0
    )
    SELECT
        stock_name as '종목명',
        stock_code as '종목코드',
        market_name as '시장',
        foreign_net_total as '외국인_순매수금액',
        institution_net_total as '기관_순매수금액',
        combined_net_total as '외국인+기관_순매수금액',
        avg_market_cap as '평균_시가총액',
        (CAST(foreign_net_total AS REAL) / avg_market_cap * 100) as '외국인_순매수비중',
        (CAST(institution_net_total AS REAL) / avg_market_cap * 100) as '기관_순매수비중',
        (CAST(combined_net_total AS REAL) / avg_market_cap * 100) as '순매수비중',
        trading_days as '거래일수'
    FROM ranked_data
    ORDER BY 순매수비중 DESC
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

    # 비중을 소수점 4자리로 반올림
    df['외국인_순매수비중'] = df['외국인_순매수비중'].round(4)
    df['기관_순매수비중'] = df['기관_순매수비중'].round(4)
    df['순매수비중'] = df['순매수비중'].round(4)

    return df


def print_results(df):
    """결과를 보기 좋게 출력"""

    print("=" * 140)
    print(f"{'순위':>4} {'종목명':<20} {'종목코드':<10} {'시장':<10} {'순매수금액(억)':<15} {'시가총액(억)':<15} {'순매수비중(%)':<15}")
    print("=" * 140)

    for idx, row in df.iterrows():
        rank = idx + 1
        print(f"{rank:>4} {row['종목명']:<20} {row['종목코드']:<10} {row['시장']:<10} "
              f"{row['외국인+기관_순매수금액_억원']:>14,.0f} {row['평균_시가총액_억원']:>14,.0f} "
              f"{row['순매수비중']:>14.4f}")

    print("=" * 140)

    # 세부 정보 출력
    print(f"\n{'='*140}")
    print(f"{'순위':>4} {'종목명':<20} {'외국인(억)':<15} {'기관(억)':<15} {'외국인비중(%)':<15} {'기관비중(%)':<15}")
    print("=" * 140)

    for idx, row in df.iterrows():
        rank = idx + 1
        print(f"{rank:>4} {row['종목명']:<20} "
              f"{row['외국인_순매수금액_억원']:>14,.0f} {row['기관_순매수금액_억원']:>14,.0f} "
              f"{row['외국인_순매수비중']:>14.4f} {row['기관_순매수비중']:>14.4f}")

    print("=" * 140)

    # 요약 통계
    print(f"\n[요약 통계]")
    print(f"  - 평균 순매수 비중: {df['순매수비중'].mean():.4f}%")
    print(f"  - 최대 순매수 비중: {df['순매수비중'].max():.4f}% ({df.loc[df['순매수비중'].idxmax(), '종목명']})")
    print(f"  - 최소 순매수 비중: {df['순매수비중'].min():.4f}% ({df.loc[df['순매수비중'].idxmin(), '종목명']})")

    # 비중 분포
    high_ratio = (df['순매수비중'] >= 1.0).sum()
    mid_ratio = ((df['순매수비중'] >= 0.5) & (df['순매수비중'] < 1.0)).sum()
    low_ratio = (df['순매수비중'] < 0.5).sum()

    print(f"\n[비중 분포]")
    print(f"  - 1% 이상: {high_ratio}개")
    print(f"  - 0.5% ~ 1%: {mid_ratio}개")
    print(f"  - 0.5% 미만: {low_ratio}개")

    # 외국인 vs 기관 비교
    foreign_dominant = (df['외국인_순매수비중'] > df['기관_순매수비중']).sum()
    institution_dominant = (df['기관_순매수비중'] > df['외국인_순매수비중']).sum()

    print(f"\n[매수 주도 세력]")
    print(f"  - 외국인 주도: {foreign_dominant}개")
    print(f"  - 기관 주도: {institution_dominant}개")


def main():
    # 최근 1개월 (약 20-22 거래일) 데이터 분석
    df = get_top_net_buyers_by_mcap_ratio(days=22, limit=30)

    # 결과 출력
    print_results(df)

    # CSV 저장 옵션
    output_file = project_root / 'data' / 'analysis_results' / 'top_net_buyers_by_mcap_ratio.csv'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n[INFO] 결과가 저장되었습니다: {output_file}")


if __name__ == '__main__':
    main()
