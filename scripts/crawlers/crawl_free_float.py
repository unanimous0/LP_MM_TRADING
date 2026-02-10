"""
유통주식 + 업종 데이터 크롤러 (FnGuide → DB 직접 업데이트)

기존 free_float_crawler.py를 확장하여 유통주식 정보와 함께
FICS 업종 정보도 크롤링합니다.

사용법:
    # 유통주식 + 업종 동시 수집 (권장)
    python scripts/crawlers/crawl_free_float.py

    # 업종만 수집 (유통주식 스킵)
    python scripts/crawlers/crawl_free_float.py --sector-only

    # 유통주식만 수집 (업종 스킵)
    python scripts/crawlers/crawl_free_float.py --skip-sector

    # 특정 시장만
    python scripts/crawlers/crawl_free_float.py --market KOSPI200

    # 실패 종목만 재시도
    python scripts/crawlers/crawl_free_float.py --retry-failed

    # 요청 간격 조정
    python scripts/crawlers/crawl_free_float.py --delay 1.0
"""

import sys
import argparse
from pathlib import Path
import time
from tqdm import tqdm
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from typing import Optional, Dict

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_connection


def clean_number(text: str) -> Optional[str]:
    """숫자 문자열에서 쉼표와 불필요한 문자 제거"""
    if not text:
        return None
    cleaned = re.sub(r'[^\d,.]', '', str(text))
    return cleaned.replace(',', '') if cleaned else None


def parse_float_data(text: str):
    """
    "숫자 / 숫자" 형식의 텍스트에서 유동주식수와 유동비율 추출
    예: "14,435,577 / 61.54" -> ("14435577", "61.54")
    """
    if not text or '/' not in text:
        return None, None

    parts = [p.strip() for p in text.split('/', 1)]
    if len(parts) != 2:
        return None, None

    part1_clean = clean_number(parts[0])
    part2_clean = clean_number(parts[1])

    if not part1_clean or not part2_clean:
        return None, None

    try:
        part1_num = float(part1_clean)
        part2_num = float(part2_clean)

        if part1_num > 100000 and 0 < part2_num <= 100:
            return part1_clean, part2_clean
    except (ValueError, TypeError):
        pass

    return None, None


def extract_sector(soup) -> Optional[str]:
    """
    FnGuide HTML에서 FICS 업종 추출

    Args:
        soup: BeautifulSoup 객체

    Returns:
        업종명 (없으면 None)
    """
    try:
        text = soup.get_text()
        # 정규식으로 "FICS" 다음 업종명 추출
        # 예: "FICS 반도체 및 관련장비"
        match = re.search(r'FICS\s+([^|\n]+)', text)
        if match:
            sector = match.group(1).strip()
            # 공백 정규화 및 길이 검증
            sector = re.sub(r'\s+', ' ', sector)
            if 0 < len(sector) < 100:
                return sector
    except Exception:
        pass
    return None


def get_fnguide_data(code: str, retry: int = 3) -> Dict[str, Optional[str]]:
    """
    FnGuide에서 발행주식수, 유동주식수, 유동비율, 업종을 크롤링합니다.

    Args:
        code: 6자리 종목코드
        retry: 재시도 횟수

    Returns:
        발행주식수, 유동주식수, 유동비율, 업종을 담은 딕셔너리
    """
    url = f"https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A{code}&cID=&MenuYn=Y&ReportGB=&NewMenuID=101&stkGb=701"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        'Referer': 'https://comp.fnguide.com/'
    }

    for attempt in range(retry):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            response.encoding = 'utf-8'
            soup = BeautifulSoup(response.content, 'html.parser')

            issued_shares = None
            float_shares = None
            float_ratio = None
            sector = None

            # 발행주식수 찾기
            main_grid = soup.find('div', {'id': 'svdMainGrid1'})
            if main_grid:
                rows = main_grid.find_all('tr')
                for row in rows:
                    th = row.find('th')
                    if th and '발행주식수' in th.get_text(strip=True):
                        td = row.find('td')
                        if td:
                            td_text = td.get_text(strip=True)
                            if '/' in td_text:
                                issued_shares = clean_number(td_text.split('/')[0])
                            else:
                                issued_shares = clean_number(td_text)
                            break

            # 유동주식수 및 유동비율 찾기
            tables = soup.find_all('table', {'class': 'us_table_ty1'})
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all(['th', 'td'])
                    if len(cols) >= 2:
                        row_text = ' '.join([col.get_text(strip=True) for col in cols])

                        if '유동' in row_text:
                            for col in cols:
                                col_text = col.get_text(strip=True)
                                if '/' in col_text and '유동' not in col_text:
                                    shares, ratio = parse_float_data(col_text)
                                    if shares and ratio:
                                        float_shares = shares
                                        float_ratio = ratio
                                        break
                    if float_shares:
                        break
                if float_shares:
                    break

            # 업종 추출
            sector = extract_sector(soup)

            return {
                '발행주식수': issued_shares,
                '유동주식수': float_shares,
                '유동비율': float_ratio,
                '업종': sector
            }

        except requests.exceptions.RequestException as e:
            if attempt < retry - 1:
                time.sleep(1 * (attempt + 1))
                continue
            return {'발행주식수': None, '유동주식수': None, '유동비율': None, '업종': None}
        except Exception as e:
            return {'발행주식수': None, '유동주식수': None, '유동비율': None, '업종': None}

    return {'발행주식수': None, '유동주식수': None, '유동비율': None, '업종': None}


def load_stock_list(conn, market: str = None) -> pd.DataFrame:
    """
    DB에서 종목 리스트 로드

    Args:
        conn: DB 연결
        market: 시장 구분 (None=전체, 'KOSPI200', 'KOSDAQ150')

    Returns:
        DataFrame (stock_code, stock_name, market_name)
    """
    if market:
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


def update_free_float(conn, stock_code: str, ff_shares: str, ff_ratio: str) -> int:
    """
    investor_flows 테이블의 모든 레코드 업데이트

    Args:
        conn: DB 연결
        stock_code: 종목 코드
        ff_shares: 유통주식수
        ff_ratio: 유통비율

    Returns:
        업데이트된 레코드 수
    """
    cursor = conn.cursor()

    # 유통주식수와 비율을 숫자로 변환
    try:
        shares = int(float(ff_shares)) if ff_shares else None
        ratio = float(ff_ratio) if ff_ratio else None
    except (ValueError, TypeError):
        return 0

    # 해당 종목의 모든 레코드 업데이트
    cursor.execute("""
        UPDATE investor_flows
        SET free_float_shares = ?,
            free_float_ratio = ?
        WHERE stock_code = ?
    """, (shares, ratio, stock_code))

    conn.commit()
    return cursor.rowcount


def update_stock_sector(conn, stock_code: str, sector: str) -> int:
    """
    stocks 테이블의 sector 컬럼 업데이트

    Args:
        conn: DB 연결
        stock_code: 종목 코드
        sector: 업종명

    Returns:
        업데이트된 레코드 수
    """
    if not sector:
        return 0

    cursor = conn.cursor()
    cursor.execute("""
        UPDATE stocks
        SET sector = ?
        WHERE stock_code = ?
    """, (sector, stock_code))
    conn.commit()
    return cursor.rowcount


def load_failed_stocks() -> list:
    """이전 실패 종목 로드"""
    failed_file = Path(__file__).parent / 'failed_stocks.txt'

    if not failed_file.exists():
        return []

    with open(failed_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def save_failed_stocks(failed_codes: list):
    """실패 종목 저장"""
    failed_file = Path(__file__).parent / 'failed_stocks.txt'

    with open(failed_file, 'w', encoding='utf-8') as f:
        for code in failed_codes:
            f.write(f"{code}\n")

    print(f"\n[INFO] Failed stocks saved to: {failed_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Crawl free float data from FnGuide and update DB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 유통주식 + 업종 동시 수집 (권장)
  python scripts/crawlers/crawl_free_float.py

  # 업종만 수집
  python scripts/crawlers/crawl_free_float.py --sector-only

  # 유통주식만 수집
  python scripts/crawlers/crawl_free_float.py --skip-sector

  # KOSPI200만
  python scripts/crawlers/crawl_free_float.py --market KOSPI200

  # 실패 종목 재시도
  python scripts/crawlers/crawl_free_float.py --retry-failed

  # 요청 간격 1초
  python scripts/crawlers/crawl_free_float.py --delay 1.0
        """
    )

    parser.add_argument('--market', choices=['KOSPI200', 'KOSDAQ150'],
                       help='시장 구분 (미지정 시 전체)')
    parser.add_argument('--retry-failed', action='store_true',
                       help='이전 실패 종목만 재시도')
    parser.add_argument('--delay', type=float, default=0.3,
                       help='요청 간 대기 시간 (초, 기본: 0.3)')
    parser.add_argument('--sector-only', action='store_true',
                       help='업종 정보만 수집 (유통주식 스킵)')
    parser.add_argument('--skip-sector', action='store_true',
                       help='업종 정보 수집 스킵 (유통주식만)')

    args = parser.parse_args()

    print("=" * 70)
    print("🔄 FnGuide 유통주식 + 업종 크롤러")
    print("=" * 70)
    print(f"파라미터:")
    print(f"  - 시장: {args.market or '전체'}")
    print(f"  - 요청 간격: {args.delay}초")
    if args.retry_failed:
        print(f"  - 모드: 실패 종목 재시도")
    if args.sector_only:
        print(f"  - 수집: 업종만 (유통주식 스킵)")
    elif args.skip_sector:
        print(f"  - 수집: 유통주식만 (업종 스킵)")
    else:
        print(f"  - 수집: 유통주식 + 업종")
    print("=" * 70)

    # DB 연결
    conn = get_connection()

    try:
        # 종목 리스트 로드
        if args.retry_failed:
            failed_codes = load_failed_stocks()

            if not failed_codes:
                print("\n[INFO] No failed stocks found")
                return

            print(f"\n[INFO] Loading {len(failed_codes)} failed stocks...")

            # DB에서 해당 종목들 정보 가져오기
            placeholders = ','.join(['?'] * len(failed_codes))
            query = f"""
            SELECT s.stock_code, s.stock_name, m.market_name
            FROM stocks s
            JOIN markets m ON s.market_id = m.market_id
            WHERE s.stock_code IN ({placeholders})
            """
            df_stocks = pd.read_sql(query, conn, params=failed_codes)
        else:
            print(f"\n[INFO] Loading stock list from database...")
            df_stocks = load_stock_list(conn, args.market)

        print(f"[OK]   Found {len(df_stocks)} stocks")

        if df_stocks.empty:
            print("[WARN] No stocks to process")
            return

        # 크롤링 시작
        print(f"\n[INFO] Starting crawl (delay={args.delay}s)...\n")

        results = []
        failed_stocks = []

        for idx, row in tqdm(df_stocks.iterrows(), total=len(df_stocks), desc="Progress"):
            stock_code = row['stock_code']
            stock_name = row['stock_name']
            market_name = row['market_name']

            # 크롤링
            try:
                data = get_fnguide_data(stock_code)

                ff_shares = data.get('유동주식수')
                ff_ratio = data.get('유동비율')
                sector = data.get('업종')

                updated_ff = 0
                updated_sector = 0

                # 유통주식 업데이트 (--sector-only가 아닌 경우)
                if not args.sector_only and ff_shares and ff_ratio:
                    updated_ff = update_free_float(conn, stock_code, ff_shares, ff_ratio)

                # 섹터 업데이트 (--skip-sector가 아닌 경우)
                if not args.skip_sector and sector:
                    updated_sector = update_stock_sector(conn, stock_code, sector)

                # 성공 판정
                success = False
                if args.sector_only:
                    success = sector is not None
                elif args.skip_sector:
                    success = ff_shares is not None and ff_ratio is not None
                else:
                    success = (ff_shares is not None and ff_ratio is not None) or sector is not None

                if success:
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'market': market_name,
                        'ff_shares': ff_shares,
                        'ff_ratio': ff_ratio,
                        'sector': sector,
                        'updated_records': updated_ff,
                        'updated_sector': updated_sector,
                        'status': 'success'
                    })
                else:
                    failed_stocks.append(stock_code)
                    results.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'market': market_name,
                        'ff_shares': None,
                        'ff_ratio': None,
                        'sector': None,
                        'updated_records': 0,
                        'updated_sector': 0,
                        'status': 'no_data'
                    })

            except Exception as e:
                failed_stocks.append(stock_code)
                results.append({
                    'stock_code': stock_code,
                    'stock_name': stock_name,
                    'market': market_name,
                    'ff_shares': None,
                    'ff_ratio': None,
                    'sector': None,
                    'updated_records': 0,
                    'updated_sector': 0,
                    'status': f'error: {str(e)}'
                })

            # 대기
            time.sleep(args.delay)

        # 결과 요약
        df_results = pd.DataFrame(results)

        print("\n" + "=" * 70)
        print("크롤링 결과 요약")
        print("=" * 70)

        success_count = (df_results['status'] == 'success').sum()
        no_data_count = (df_results['status'] == 'no_data').sum()
        error_count = len(df_results) - success_count - no_data_count

        total_updated_ff = df_results['updated_records'].sum()
        total_updated_sector = df_results['updated_sector'].sum()

        print(f"총 처리 종목: {len(df_results)}")
        print(f"  ✓ 성공: {success_count}")
        print(f"  ⚠ 데이터 없음: {no_data_count}")
        print(f"  ✗ 오류: {error_count}")
        print(f"\n업데이트 통계:")
        print(f"  - 유통주식 레코드: {total_updated_ff:,}건")
        print(f"  - 섹터 정보: {total_updated_sector:,}건")

        # 실패 종목 저장
        if failed_stocks:
            print(f"\n[WARN] {len(failed_stocks)} stocks failed")
            print(f"       Failed codes: {failed_stocks[:10]}{'...' if len(failed_stocks) > 10 else ''}")
            save_failed_stocks(failed_stocks)
            print(f"       Use --retry-failed to retry these stocks")
        else:
            # 성공 시 failed_stocks.txt 삭제
            failed_file = Path(__file__).parent / 'failed_stocks.txt'
            if failed_file.exists():
                failed_file.unlink()
                print(f"\n[OK] All stocks processed successfully, failed_stocks.txt removed")

        # 샘플 결과 출력
        if success_count > 0:
            print(f"\n[INFO] Sample results (first 5 successful):")
            df_success = df_results[df_results['status'] == 'success'].head(5)
            for _, row in df_success.iterrows():
                print(f"  [{row['stock_name']}] ({row['stock_code']})")
                if not args.sector_only and row['ff_shares']:
                    print(f"    유통주식수: {int(float(row['ff_shares'])):,}")
                    print(f"    유통비율: {float(row['ff_ratio']):.2f}%")
                    print(f"    업데이트: {row['updated_records']:,} records")
                if not args.skip_sector and row['sector']:
                    print(f"    업종: {row['sector']}")

        print("=" * 70)

        if success_count > 0:
            print("\n[SUCCESS] 크롤링 완료!")
            print("\n다음 단계:")
            print("  1. python scripts/analysis/abnormal_supply_detector.py")
            print("  2. Sff, Z-Score 분석 실행 (섹터 정보 포함)")

    except Exception as e:
        print(f"\n[ERROR] Crawling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
