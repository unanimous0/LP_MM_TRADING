"""
주가/거래량 데이터 크롤러 (FinanceDataReader → DB 직접 업데이트)

FinanceDataReader를 사용하여 주가/거래량/거래대금 데이터를 수집하고
investor_flows 테이블에 직접 업데이트합니다.

사전 준비:
    pip install finance-datareader

사용법:
    # 전체 종목, 2024-01-01부터 현재까지
    python scripts/crawl_stock_prices.py --start 2024-01-01

    # 특정 기간
    python scripts/crawl_stock_prices.py --start 2024-01-01 --end 2026-02-10

    # 특정 종목만 (삼성전자, SK하이닉스)
    python scripts/crawl_stock_prices.py --codes 005930,000660 --start 2024-01-01

    # KOSPI200만
    python scripts/crawl_stock_prices.py --market KOSPI200 --start 2024-01-01
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, date
import pandas as pd
from tqdm import tqdm

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_connection

# FinanceDataReader import (설치 필요)
try:
    import FinanceDataReader as fdr
except ImportError:
    print("[ERROR] FinanceDataReader not installed")
    print("       Please install: pip install finance-datareader")
    sys.exit(1)


def load_stock_list(conn, market: str = None, codes: list = None) -> pd.DataFrame:
    """
    DB에서 종목 리스트 로드

    Args:
        conn: DB 연결
        market: 시장 구분 (None=전체)
        codes: 특정 종목 코드 리스트

    Returns:
        DataFrame (stock_code, stock_name, market_name)
    """
    if codes:
        placeholders = ','.join(['?'] * len(codes))
        query = f"""
        SELECT s.stock_code, s.stock_name, m.market_name
        FROM stocks s
        JOIN markets m ON s.market_id = m.market_id
        WHERE s.stock_code IN ({placeholders})
        ORDER BY s.stock_code
        """
        df = pd.read_sql(query, conn, params=codes)
    elif market:
        query = """
        SELECT s.stock_code, s.stock_name, m.market_name
        FROM stocks s
        JOIN markets m ON s.market_id = m.market_id
        WHERE m.market_name = ?
        ORDER BY s.stock_code
        """
        df = pd.read_sql(query, conn, params=[market])
    else:
        query = """
        SELECT s.stock_code, s.stock_name, m.market_name
        FROM stocks s
        JOIN markets m ON s.market_id = m.market_id
        ORDER BY m.market_name, s.stock_code
        """
        df = pd.read_sql(query, conn)

    return df


def fetch_stock_prices(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    FinanceDataReader로 주가 데이터 가져오기

    Args:
        stock_code: 종목 코드 (6자리)
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)

    Returns:
        DataFrame (Date, Close, Volume, Value)
    """
    try:
        # fdr.DataReader는 Date 인덱스로 반환
        df = fdr.DataReader(stock_code, start=start_date, end=end_date)

        if df.empty:
            return pd.DataFrame()

        # 필요한 컬럼만 추출
        df = df.reset_index()
        df = df.rename(columns={'Date': 'trade_date'})

        # Close, Volume은 기본 제공
        # Value = Close × Volume으로 계산
        result = pd.DataFrame({
            'trade_date': df['trade_date'],
            'close_price': df['Close'],
            'trading_volume': df['Volume'],
            'trading_value': df['Close'] * df['Volume']
        })

        # 날짜를 문자열로 변환 (YYYY-MM-DD)
        result['trade_date'] = result['trade_date'].dt.strftime('%Y-%m-%d')

        return result

    except Exception as e:
        print(f"[ERROR] Failed to fetch {stock_code}: {e}")
        return pd.DataFrame()


def update_stock_prices(conn, stock_code: str, df_prices: pd.DataFrame) -> int:
    """
    investor_flows 테이블에 주가 데이터 업데이트

    Args:
        conn: DB 연결
        stock_code: 종목 코드
        df_prices: 주가 데이터 (trade_date, close_price, trading_volume, trading_value)

    Returns:
        업데이트된 레코드 수
    """
    cursor = conn.cursor()
    updated_count = 0

    for _, row in df_prices.iterrows():
        cursor.execute("""
            UPDATE investor_flows
            SET close_price = ?,
                trading_volume = ?,
                trading_value = ?
            WHERE trade_date = ? AND stock_code = ?
        """, (
            float(row['close_price']),
            int(row['trading_volume']),
            float(row['trading_value']),
            str(row['trade_date']),
            stock_code
        ))

        updated_count += cursor.rowcount

    conn.commit()
    return updated_count


def main():
    parser = argparse.ArgumentParser(
        description='Crawl stock prices using FinanceDataReader and update DB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 종목, 2024년부터
  python scripts/crawl_stock_prices.py --start 2024-01-01

  # 특정 기간
  python scripts/crawl_stock_prices.py --start 2024-01-01 --end 2026-02-10

  # 특정 종목만
  python scripts/crawl_stock_prices.py --codes 005930,000660 --start 2024-01-01

  # KOSPI200만
  python scripts/crawl_stock_prices.py --market KOSPI200 --start 2024-01-01
        """
    )

    parser.add_argument('--start', required=True,
                       help='시작일 (YYYY-MM-DD)')
    parser.add_argument('--end',
                       help='종료일 (YYYY-MM-DD, 기본: 오늘)')
    parser.add_argument('--market', choices=['KOSPI200', 'KOSDAQ150'],
                       help='시장 구분 (미지정 시 전체)')
    parser.add_argument('--codes',
                       help='특정 종목 코드 (쉼표 구분, 예: 005930,000660)')

    args = parser.parse_args()

    # 날짜 검증
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    except ValueError:
        print(f"[ERROR] Invalid start date: {args.start}")
        print("        Expected format: YYYY-MM-DD")
        sys.exit(1)

    if args.end:
        try:
            end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
        except ValueError:
            print(f"[ERROR] Invalid end date: {args.end}")
            print("        Expected format: YYYY-MM-DD")
            sys.exit(1)
    else:
        end_date = date.today()

    # 종목 코드 파싱
    codes = None
    if args.codes:
        codes = [code.strip() for code in args.codes.split(',')]

    print("=" * 70)
    print("📈 주가/거래량 크롤러 (FinanceDataReader)")
    print("=" * 70)
    print(f"파라미터:")
    print(f"  - 기간: {start_date} ~ {end_date}")
    print(f"  - 시장: {args.market or '전체'}")
    if codes:
        print(f"  - 종목: {len(codes)}개 ({', '.join(codes[:5])}{'...' if len(codes) > 5 else ''})")
    print("=" * 70)

    # DB 연결
    conn = get_connection()

    try:
        # 종목 리스트 로드
        print(f"\n[INFO] Loading stock list from database...")
        df_stocks = load_stock_list(conn, args.market, codes)
        print(f"[OK]   Found {len(df_stocks)} stocks")

        if df_stocks.empty:
            print("[WARN] No stocks to process")
            return

        # 크롤링 시작
        print(f"\n[INFO] Starting crawl...\n")

        results = []
        failed_stocks = []

        for idx, row in tqdm(df_stocks.iterrows(), total=len(df_stocks), desc="Progress"):
            stock_code = row['stock_code']
            stock_name = row['stock_name']
            market_name = row['market_name']

            try:
                # 주가 데이터 가져오기
                df_prices = fetch_stock_prices(stock_code, str(start_date), str(end_date))

                if df_prices.empty:
                    failed_stocks.append((stock_code, stock_name, 'no_data'))
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'market': market_name,
                        'records_fetched': 0,
                        'records_updated': 0,
                        'status': 'no_data'
                    })
                    continue

                # DB 업데이트
                updated_count = update_stock_prices(conn, stock_code, df_prices)

                results.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'market': market_name,
                    'records_fetched': len(df_prices),
                    'records_updated': updated_count,
                    'status': 'success'
                })

            except Exception as e:
                failed_stocks.append((stock_code, stock_name, str(e)))
                results.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'market': market_name,
                    'records_fetched': 0,
                    'records_updated': 0,
                    'status': f'error: {str(e)}'
                })

        # 결과 요약
        df_results = pd.DataFrame(results)

        print("\n" + "=" * 70)
        print("크롤링 결과 요약")
        print("=" * 70)

        success_count = (df_results['status'] == 'success').sum()
        no_data_count = (df_results['status'] == 'no_data').sum()
        error_count = len(df_results) - success_count - no_data_count

        total_fetched = df_results['records_fetched'].sum()
        total_updated = df_results['records_updated'].sum()

        print(f"총 처리 종목: {len(df_results)}")
        print(f"  ✓ 성공: {success_count}")
        print(f"  ⚠ 데이터 없음: {no_data_count}")
        print(f"  ✗ 오류: {error_count}")
        print(f"\n총 가져온 레코드: {total_fetched:,}")
        print(f"총 업데이트 레코드: {total_updated:,}")

        # 실패 종목 출력
        if failed_stocks:
            print(f"\n[WARN] {len(failed_stocks)} stocks failed:")
            for code, name, reason in failed_stocks[:10]:
                print(f"  - [{name}] ({code}): {reason}")
            if len(failed_stocks) > 10:
                print(f"  ... and {len(failed_stocks) - 10} more")

        # 샘플 결과 출력
        if success_count > 0:
            print(f"\n[INFO] Sample results (first 5 successful):")
            df_success = df_results[df_results['status'] == 'success'].head(5)
            for _, row in df_success.iterrows():
                print(f"  [{row['stock_name']}] ({row['stock_code']})")
                print(f"    가져온 데이터: {row['records_fetched']} 거래일")
                print(f"    업데이트: {row['records_updated']} records")

        # DB 검증
        print(f"\n[INFO] Verifying database...")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM investor_flows WHERE close_price IS NOT NULL")
        price_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM investor_flows")
        total_count = cursor.fetchone()[0]

        print(f"  - Total records: {total_count:,}")
        print(f"  - Records with price: {price_count:,} ({price_count/total_count*100:.1f}%)")

        print("=" * 70)

        if success_count > 0:
            print("\n[SUCCESS] 크롤링 완료!")
            print("\n다음 단계:")
            print("  1. python scripts/crawl_free_float.py")
            print("  2. python scripts/analysis/abnormal_supply_detector.py")

    except Exception as e:
        print(f"\n[ERROR] Crawling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
