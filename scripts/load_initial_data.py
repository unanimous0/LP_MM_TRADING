"""
초기 데이터 로드 스크립트

엑셀 파일에서 데이터를 읽어 SQLite 데이터베이스에 로드합니다.
1. 종목 마스터 데이터 로드 (stocks 테이블)
2. 투자자 수급 및 시가총액 데이터 로드 (investor_flows 테이블)
"""

import sqlite3
import pandas as pd
from pathlib import Path
import sys

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collector.excel_collector import ExcelCollector
from src.database.connection import get_connection


def load_stocks(conn: sqlite3.Connection):
    """
    종목 마스터 데이터 로드

    Args:
        conn: SQLite 데이터베이스 연결 객체
    """
    print("\n" + "=" * 60)
    print("1. Loading Stock Master Data")
    print("=" * 60)

    collector = ExcelCollector()

    # 종목 매핑 파일 로드
    mapping_path = project_root / 'data' / '종목코드_이름_맵핑.xlsx'
    print(f"[INFO] Loading: {mapping_path}")

    df_stocks = collector.load_stock_mapping(str(mapping_path))

    # KOSPI200/KOSDAQ150 구분은 향후 추가 (현재는 모두 로드)
    # 초기에는 market_id를 1로 설정 (KOSPI200 기본값)
    df_stocks['market_id'] = 1
    df_stocks['is_active'] = 1
    df_stocks['sector'] = None

    # 기존 데이터 확인
    existing_count = pd.read_sql("SELECT COUNT(*) as cnt FROM stocks", conn).iloc[0]['cnt']

    if existing_count > 0:
        print(f"[WARN] stocks table already has {existing_count} records.")
        print("       Keeping existing data and adding only new records.")

    # 중복 방지: INSERT OR IGNORE 사용
    df_stocks.to_sql('stocks', conn, if_exists='append', index=False)

    # 로드 후 확인
    total_count = pd.read_sql("SELECT COUNT(*) as cnt FROM stocks", conn).iloc[0]['cnt']
    print(f"[OK] Stocks loaded: {total_count} stocks")


def load_investor_flows(conn: sqlite3.Connection):
    """
    투자자 수급 및 시가총액 데이터 로드

    Args:
        conn: SQLite 데이터베이스 연결 객체
    """
    print("\n" + "=" * 60)
    print("2. Loading Investor Flow Data")
    print("=" * 60)

    collector = ExcelCollector()

    # 파일 경로 설정
    flows_path = project_root / 'data' / '투자자수급_200_150.xlsx'
    mcap_path = project_root / 'data' / '시가총액_200_150.xlsx'

    # KOSPI200 로드
    print("\n[INFO] Loading KOSPI200...")
    df_kospi_flows = collector.load_investor_flows(str(flows_path), 'KOSPI200')
    print(f"       Investor flows: {len(df_kospi_flows):,} records")

    df_kospi_mcap = collector.load_market_caps(str(mcap_path), 'KOSPI200')
    print(f"       Market cap: {len(df_kospi_mcap):,} records")

    df_kospi = df_kospi_flows.merge(df_kospi_mcap, on=['trade_date', 'stock_name'], how='left')
    print(f"       After merge: {len(df_kospi):,} records")

    # KOSDAQ150 로드
    print("\n[INFO] Loading KOSDAQ150...")
    df_kosdaq_flows = collector.load_investor_flows(str(flows_path), 'KOSDAQ150')
    print(f"       Investor flows: {len(df_kosdaq_flows):,} records")

    df_kosdaq_mcap = collector.load_market_caps(str(mcap_path), 'KOSDAQ150')
    print(f"       Market cap: {len(df_kosdaq_mcap):,} records")

    df_kosdaq = df_kosdaq_flows.merge(df_kosdaq_mcap, on=['trade_date', 'stock_name'], how='left')
    print(f"       After merge: {len(df_kosdaq):,} records")

    # 결합
    df_all = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
    print(f"\n[INFO] Total records before mapping: {len(df_all):,}")

    # 종목명 → 종목코드 매핑
    print("\n[INFO] Mapping stock names to stock codes...")
    df_stocks = pd.read_sql('SELECT stock_code, stock_name FROM stocks', conn)
    df_all = df_all.merge(df_stocks, on='stock_name', how='left')

    # NULL 체크
    null_count = df_all['stock_code'].isna().sum()
    if null_count > 0:
        print(f"\n[WARN] {null_count} records could not be mapped to stock_code.")
        unmapped_stocks = df_all[df_all['stock_code'].isna()]['stock_name'].unique()
        print(f"       Unmapped stocks (max 10): {unmapped_stocks[:10].tolist()}")

    # NULL 제거
    df_all = df_all.dropna(subset=['stock_code'])
    print(f"[OK] Mapping complete: {len(df_all):,} records")

    # stock_name 컬럼 제거 (DB에는 stock_code만 저장)
    df_all = df_all.drop(columns=['stock_name'])

    # 컬럼 순서 조정 (created_at 제외)
    df_all = df_all[['trade_date', 'stock_code', 'foreign_net_volume', 'foreign_net_amount',
                     'institution_net_volume', 'institution_net_amount', 'market_cap']]

    # 기존 데이터 확인
    existing_count = pd.read_sql("SELECT COUNT(*) as cnt FROM investor_flows", conn).iloc[0]['cnt']

    if existing_count > 0:
        print(f"\n[WARN] investor_flows table already has {existing_count:,} records.")
        print("       Keeping existing data and adding only new records.")

    # DB 저장 (Bulk Insert)
    print("\n[INFO] Saving to database...")
    df_all.to_sql('investor_flows', conn, if_exists='append', index=False)

    # 로드 후 확인
    total_count = pd.read_sql("SELECT COUNT(*) as cnt FROM investor_flows", conn).iloc[0]['cnt']
    print(f"[OK] Investor flows loaded: {total_count:,} records")


def main():
    """메인 실행 함수"""
    print("\n" + "=" * 60)
    print("Initial Data Load Started")
    print("=" * 60)

    try:
        conn = get_connection()

        # 1. 종목 마스터 로드
        load_stocks(conn)

        # 2. 투자자 수급 로드
        load_investor_flows(conn)

        conn.commit()
        conn.close()

        print("\n" + "=" * 60)
        print("[SUCCESS] Initial data load completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
