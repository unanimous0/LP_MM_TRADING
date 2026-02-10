"""
통합 데이터 크롤러 (주가 + 유통주식)

주가/거래량과 유통주식 데이터를 한 번에 수집합니다.

사전 준비:
    pip install finance-datareader

사용법:
    # 기본 (전체 종목, 2024-01-01부터)
    python scripts/crawl_all_data.py --start 2024-01-01

    # 특정 기간
    python scripts/crawl_all_data.py --start 2024-01-01 --end 2026-02-10

    # KOSPI200만
    python scripts/crawl_all_data.py --market KOSPI200 --start 2024-01-01

    # 주가만 (유통주식 스킵)
    python scripts/crawl_all_data.py --start 2024-01-01 --skip-ff

    # 유통주식만 (주가 스킵)
    python scripts/crawl_all_data.py --skip-prices
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, date
import subprocess

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd: list, description: str) -> bool:
    """
    외부 명령 실행

    Args:
        cmd: 명령어 리스트
        description: 작업 설명

    Returns:
        성공 여부
    """
    print("\n" + "=" * 70)
    print(f"🚀 {description}")
    print("=" * 70)
    print(f"[CMD] {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True, cwd=str(project_root))
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Command failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Crawl all data (prices + free float)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 데이터 (주가 + 유통주식)
  python scripts/crawl_all_data.py --start 2024-01-01

  # 특정 기간
  python scripts/crawl_all_data.py --start 2024-01-01 --end 2026-02-10

  # KOSPI200만
  python scripts/crawl_all_data.py --market KOSPI200 --start 2024-01-01

  # 주가만
  python scripts/crawl_all_data.py --start 2024-01-01 --skip-ff

  # 유통주식만
  python scripts/crawl_all_data.py --skip-prices
        """
    )

    parser.add_argument('--start',
                       help='시작일 (YYYY-MM-DD, 주가 크롤링에 필수)')
    parser.add_argument('--end',
                       help='종료일 (YYYY-MM-DD, 기본: 오늘)')
    parser.add_argument('--market', choices=['KOSPI200', 'KOSDAQ150'],
                       help='시장 구분 (미지정 시 전체)')
    parser.add_argument('--skip-prices', action='store_true',
                       help='주가 크롤링 건너뛰기')
    parser.add_argument('--skip-ff', action='store_true',
                       help='유통주식 크롤링 건너뛰기')
    parser.add_argument('--ff-delay', type=float, default=0.3,
                       help='유통주식 크롤링 요청 간격 (초, 기본: 0.3)')

    args = parser.parse_args()

    # 검증
    if not args.skip_prices and not args.start:
        print("[ERROR] --start is required for price crawling")
        print("        Use --skip-prices to skip price data")
        sys.exit(1)

    if args.skip_prices and args.skip_ff:
        print("[ERROR] Cannot skip both prices and free float")
        sys.exit(1)

    print("=" * 70)
    print("🔄 통합 데이터 크롤러")
    print("=" * 70)
    print(f"파라미터:")
    if not args.skip_prices:
        print(f"  - 주가 기간: {args.start} ~ {args.end or '오늘'}")
    if not args.skip_ff:
        print(f"  - 유통주식 요청 간격: {args.ff_delay}초")
    print(f"  - 시장: {args.market or '전체'}")
    print("=" * 70)

    results = []

    # Step 1: 주가/거래량 크롤링
    if not args.skip_prices:
        cmd_prices = [
            sys.executable,
            'scripts/crawl_stock_prices.py',
            '--start', args.start
        ]

        if args.end:
            cmd_prices.extend(['--end', args.end])
        if args.market:
            cmd_prices.extend(['--market', args.market])

        success = run_command(cmd_prices, "Step 1: 주가/거래량 크롤링")
        results.append(('주가/거래량', success))

        if not success:
            print("\n[WARN] Price crawling failed, but continuing with free float...")

    # Step 2: 유통주식 크롤링
    if not args.skip_ff:
        cmd_ff = [
            sys.executable,
            'scripts/crawl_free_float.py',
            '--delay', str(args.ff_delay)
        ]

        if args.market:
            cmd_ff.extend(['--market', args.market])

        success = run_command(cmd_ff, "Step 2: 유통주식 크롤링")
        results.append(('유통주식', success))

    # 최종 요약
    print("\n" + "=" * 70)
    print("📊 통합 크롤링 결과")
    print("=" * 70)

    all_success = True
    for task, success in results:
        status = "✓ 성공" if success else "✗ 실패"
        print(f"{status}: {task}")
        if not success:
            all_success = False

    print("=" * 70)

    if all_success:
        print("\n[SUCCESS] 모든 크롤링 완료!")
        print("\n다음 단계:")
        print("  1. python scripts/analysis/abnormal_supply_detector.py")
        print("  2. Sff, Z-Score 분석 실행")
    else:
        print("\n[WARN] 일부 크롤링 실패 - 로그를 확인하세요")
        sys.exit(1)


if __name__ == '__main__':
    main()
