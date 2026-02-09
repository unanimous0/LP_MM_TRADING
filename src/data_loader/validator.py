"""
데이터 검증 모듈

로드된 데이터의 무결성을 확인합니다.
- 레코드 수 확인
- 중복 검사
- NULL 값 검사
- 날짜 범위 확인
"""

import pandas as pd
import sqlite3


def validate_data(conn: sqlite3.Connection):
    """
    로드된 데이터 검증

    Args:
        conn: SQLite 데이터베이스 연결 객체
    """
    print("=" * 60)
    print("Data Validation Started")
    print("=" * 60)

    # 1. 레코드 수 확인
    count = pd.read_sql("SELECT COUNT(*) as cnt FROM investor_flows", conn)
    print(f"\n[INFO] Total records: {count['cnt'].iloc[0]:,}")

    # 2. 중복 체크
    duplicates = pd.read_sql("""
        SELECT trade_date, stock_code, COUNT(*) as cnt
        FROM investor_flows
        GROUP BY trade_date, stock_code
        HAVING COUNT(*) > 1
    """, conn)

    if len(duplicates) > 0:
        print(f"\n[WARN] Found {len(duplicates)} duplicate records")
        print(duplicates.head())
    else:
        print("\n[OK] No duplicates found")

    # 3. NULL 체크 (필수 컬럼)
    nulls = pd.read_sql("""
        SELECT COUNT(*) as cnt
        FROM investor_flows
        WHERE stock_code IS NULL OR trade_date IS NULL
    """, conn)

    if nulls['cnt'].iloc[0] > 0:
        print(f"\n[WARN] Found {nulls['cnt'].iloc[0]} records with NULL values in critical columns")
    else:
        print("[OK] No NULL values in critical columns")

    # 4. 날짜 범위 확인
    date_range = pd.read_sql("""
        SELECT MIN(trade_date) as min_date, MAX(trade_date) as max_date
        FROM investor_flows
    """, conn)
    print(f"\n[INFO] Date range: {date_range['min_date'].iloc[0]} ~ {date_range['max_date'].iloc[0]}")

    # 5. 종목 수 확인
    stock_count = pd.read_sql("""
        SELECT COUNT(DISTINCT stock_code) as cnt
        FROM investor_flows
    """, conn)
    print(f"[INFO] Unique stocks: {stock_count['cnt'].iloc[0]}")

    # 6. 샘플 데이터 확인
    sample = pd.read_sql("""
        SELECT trade_date, stock_code, foreign_net_volume, foreign_net_amount,
               institution_net_volume, institution_net_amount, market_cap
        FROM investor_flows
        ORDER BY trade_date DESC
        LIMIT 5
    """, conn)
    print("\n[INFO] Sample data (recent 5 records):")
    print(sample.to_string(index=False))

    print("\n" + "=" * 60)
    print("Validation Complete")
    print("=" * 60)


if __name__ == '__main__':
    from src.database.connection import get_connection

    conn = get_connection()
    validate_data(conn)
    conn.close()
