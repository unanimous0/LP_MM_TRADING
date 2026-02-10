"""
일별 증분 데이터 업데이트 스크립트

사용법:
    python scripts/load_daily_data.py <투자자수급_파일.xlsx> <시가총액_파일.xlsx>
    python scripts/load_daily_data.py <투자자수급_파일.xlsx> <시가총액_파일.xlsx> --dry-run
    python scripts/load_daily_data.py <투자자수급_파일.xlsx> <시가총액_파일.xlsx> --price-file <주가_파일.xlsx>
    python scripts/load_daily_data.py <투자자수급_파일.xlsx> <시가총액_파일.xlsx> --price-file <주가_파일.xlsx> --ff-file <유통주식_파일.xlsx>

예시:
    # 기본 (외국인/기관 수급 + 시가총액)
    python scripts/load_daily_data.py data/투자자수급_20260209.xlsx data/시가총액_20260209.xlsx

    # 주가/거래량 데이터 포함
    python scripts/load_daily_data.py data/투자자수급_20260209.xlsx data/시가총액_20260209.xlsx --price-file data/주가_20260209.xlsx

    # 전체 데이터 (수급 + 시총 + 주가 + 유통주식)
    python scripts/load_daily_data.py data/투자자수급_20260209.xlsx data/시가총액_20260209.xlsx --price-file data/주가_20260209.xlsx --ff-file data/유통주식_20260209.xlsx
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collector.excel_collector import ExcelCollector
from src.database.connection import get_connection


def validate_files(flow_file: str, mcap_file: str, price_file: str = None, ff_file: str = None) -> bool:
    """입력 파일 검증"""
    if not Path(flow_file).exists():
        print(f"[ERROR] File not found: {flow_file}")
        return False

    if not Path(mcap_file).exists():
        print(f"[ERROR] File not found: {mcap_file}")
        return False

    if price_file and not Path(price_file).exists():
        print(f"[ERROR] File not found: {price_file}")
        return False

    if ff_file and not Path(ff_file).exists():
        print(f"[ERROR] File not found: {ff_file}")
        return False

    return True


def load_and_merge_data(flow_file: str, mcap_file: str) -> pd.DataFrame:
    """엑셀 데이터 로드 및 병합"""
    collector = ExcelCollector()

    # KOSPI200
    df_kospi_flows = collector.load_investor_flows(flow_file, 'KOSPI200')
    df_kospi_mcap = collector.load_market_caps(mcap_file, 'KOSPI200')
    df_kospi = df_kospi_flows.merge(df_kospi_mcap, on=['trade_date', 'stock_name'], how='left')

    # KOSDAQ150
    df_kosdaq_flows = collector.load_investor_flows(flow_file, 'KOSDAQ150')
    df_kosdaq_mcap = collector.load_market_caps(mcap_file, 'KOSDAQ150')
    df_kosdaq = df_kosdaq_flows.merge(df_kosdaq_mcap, on=['trade_date', 'stock_name'], how='left')

    # 결합
    df_all = pd.concat([df_kospi, df_kosdaq], ignore_index=True)

    return df_all


def map_stock_codes(df: pd.DataFrame, conn) -> tuple:
    """종목명 → 종목코드 매핑 (대소문자 무시)

    Returns:
        (mapped_df, unmapped_count)
    """
    df_stocks = pd.read_sql('SELECT stock_code, stock_name FROM stocks', conn)

    # 정규화 (대소문자, 공백 처리)
    df['stock_name_upper'] = df['stock_name'].str.strip().str.upper()
    df_stocks['stock_name_upper'] = df_stocks['stock_name'].str.strip().str.upper()

    # 매핑
    df = df.merge(
        df_stocks[['stock_code', 'stock_name_upper']],
        on='stock_name_upper',
        how='left'
    )

    # 매핑 실패 확인
    unmapped = df[df['stock_code'].isna()]
    if len(unmapped) > 0:
        print(f"\n[WARN] {len(unmapped)} records could not be mapped to stock_code")
        unmapped_stocks = unmapped['stock_name'].unique()
        print(f"       Unmapped stocks (up to 10): {unmapped_stocks[:10].tolist()}")

    # 매핑 실패 레코드 제거
    df = df.dropna(subset=['stock_code'])

    # 정규화 컬럼 제거
    df = df.drop(columns=['stock_name', 'stock_name_upper'])

    return df, len(unmapped)


def insert_data(df: pd.DataFrame, conn) -> tuple:
    """데이터 삽입 (INSERT OR IGNORE)

    Returns:
        (inserted_count, ignored_count, error_count)
    """
    # 필수 컬럼
    base_cols = ['trade_date', 'stock_code', 'foreign_net_volume', 'foreign_net_amount',
                 'institution_net_volume', 'institution_net_amount', 'market_cap']

    # 선택 컬럼 (있으면 추가)
    optional_cols = ['close_price', 'trading_volume', 'trading_value',
                     'free_float_shares', 'free_float_ratio']

    # 실제 존재하는 컬럼만 선택
    cols = base_cols + [col for col in optional_cols if col in df.columns]
    df = df[cols]

    # SQL 생성 (동적)
    col_names = ', '.join(cols)
    placeholders = ', '.join(['?'] * len(cols))
    sql = f"""
        INSERT OR IGNORE INTO investor_flows
        ({col_names})
        VALUES ({placeholders})
    """

    cursor = conn.cursor()
    inserted_count = 0
    ignored_count = 0
    error_count = 0

    for _, row in df.iterrows():
        try:
            cursor.execute(sql, tuple(row))

            if cursor.rowcount > 0:
                inserted_count += 1
            else:
                ignored_count += 1

        except Exception as e:
            error_count += 1
            print(f"[ERROR] Failed to insert: {e}")
            print(f"        Record: {row.to_dict()}")

    conn.commit()

    return inserted_count, ignored_count, error_count


def print_report(flow_file: str, df: pd.DataFrame, inserted: int, ignored: int,
                 errors: int, unmapped: int, conn):
    """처리 결과 리포트 출력"""
    print("\n" + "=" * 70)
    print("Daily Data Load Report")
    print("=" * 70)
    print(f"[INFO] Source file: {Path(flow_file).name}")
    print(f"[INFO] Date range: {df['trade_date'].min()} ~ {df['trade_date'].max()}")
    print(f"[INFO] Unique stocks: {df['stock_code'].nunique()}")
    print(f"[INFO] Total records in file: {len(df) + unmapped}")
    print(f"")
    print(f"[OK]   Records inserted: {inserted}")
    print(f"[INFO] Records skipped (duplicates): {ignored}")
    if unmapped > 0:
        print(f"[WARN] Records unmapped: {unmapped}")
    if errors > 0:
        print(f"[ERROR] Records failed: {errors}")

    # DB 현황
    total = pd.read_sql("SELECT COUNT(*) as cnt FROM investor_flows", conn).iloc[0]['cnt']
    latest = pd.read_sql("SELECT MAX(trade_date) as max_date FROM investor_flows", conn).iloc[0]['max_date']

    print(f"")
    print(f"[INFO] Total records in database: {total:,}")
    print(f"[INFO] Latest date in database: {latest}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Load daily investor flow data into database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/load_daily_data.py data/투자자수급_20260209.xlsx data/시가총액_20260209.xlsx
  python scripts/load_daily_data.py data/투자자수급_20260209.xlsx data/시가총액_20260209.xlsx --dry-run
        """
    )

    parser.add_argument('flow_file', help='Path to investor flow Excel file')
    parser.add_argument('mcap_file', help='Path to market cap Excel file')
    parser.add_argument('--price-file', help='Path to price/volume Excel file (optional)')
    parser.add_argument('--ff-file', help='Path to free float Excel file (optional, updates monthly)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview data without inserting into database')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("Daily Data Load Started")
    print("=" * 70)

    # 1. 파일 검증
    if not validate_files(args.flow_file, args.mcap_file, args.price_file, args.ff_file):
        sys.exit(1)

    # 2. 데이터 로드
    print("\n[INFO] Loading investor flow and market cap data...")
    try:
        df_all = load_and_merge_data(args.flow_file, args.mcap_file)
        print(f"[OK]   Loaded {len(df_all)} records")
    except Exception as e:
        print(f"[ERROR] Failed to load data: {e}")
        sys.exit(1)

    # 2-1. 주가/거래량 데이터 로드 (선택)
    if args.price_file:
        print("\n[INFO] Loading price/volume data...")
        try:
            collector = ExcelCollector()
            df_kospi_price = collector.load_stock_prices(args.price_file, 'KOSPI200')
            df_kosdaq_price = collector.load_stock_prices(args.price_file, 'KOSDAQ150')
            df_prices = pd.concat([df_kospi_price, df_kosdaq_price], ignore_index=True)

            # Merge with main data
            df_all = df_all.merge(df_prices, on=['trade_date', 'stock_name'], how='left')
            print(f"[OK]   Merged price data ({len(df_prices)} records)")
        except Exception as e:
            print(f"[ERROR] Failed to load price data: {e}")
            sys.exit(1)

    # 2-2. 유통주식 데이터 로드 (선택)
    if args.ff_file:
        print("\n[INFO] Loading free float data...")
        try:
            collector = ExcelCollector()
            df_kospi_ff = collector.load_free_float(args.ff_file, 'KOSPI200')
            df_kosdaq_ff = collector.load_free_float(args.ff_file, 'KOSDAQ150')
            df_ff = pd.concat([df_kospi_ff, df_kosdaq_ff], ignore_index=True)

            # Merge with main data
            df_all = df_all.merge(df_ff, on='stock_name', how='left')
            print(f"[OK]   Merged free float data ({len(df_ff)} stocks)")
        except Exception as e:
            print(f"[ERROR] Failed to load free float data: {e}")
            sys.exit(1)

    # 3. DB 연결
    conn = get_connection()

    # 4. 종목 매핑
    print("\n[INFO] Mapping stock names to stock codes...")
    df_all, unmapped_count = map_stock_codes(df_all, conn)
    print(f"[OK]   Mapped {len(df_all)} records")

    # 5. Dry-run 모드
    if args.dry_run:
        print("\n[DRY-RUN] Preview mode - no data will be inserted")
        print("\nSample data (first 10 records):")
        print(df_all.head(10).to_string(index=False))
        print(f"\nWould insert {len(df_all)} records")
        conn.close()
        return

    # 6. 데이터 삽입
    print("\n[INFO] Inserting data into database...")
    inserted, ignored, errors = insert_data(df_all, conn)

    # 7. 리포트 출력
    print_report(args.flow_file, df_all, inserted, ignored, errors, unmapped_count, conn)

    conn.close()

    if errors > 0:
        print("\n[WARN] Completed with errors")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Daily data load completed successfully")


if __name__ == '__main__':
    main()
