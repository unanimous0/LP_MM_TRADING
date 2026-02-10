"""
이상 수급 이벤트 탐지기 (Abnormal Supply Event Detector)

Z-Score 분석을 통해 통계적으로 유의미한 수급 변화를 탐지합니다.

Z > 2.0: 평균 대비 표준편차 2배 이상 강한 매수세
Z < -2.0: 평균 대비 표준편차 2배 이상 강한 매도세

사용법:
    # 기본 (임계값 2.0, 상위 20개 종목)
    python scripts/analysis/abnormal_supply_detector.py

    # 임계값 2.5, 상위 30개
    python scripts/analysis/abnormal_supply_detector.py --threshold 2.5 --top 30

    # 매수 시그널만 표시
    python scripts/analysis/abnormal_supply_detector.py --direction buy

    # 특정 날짜 기준
    python scripts/analysis/abnormal_supply_detector.py --date 2026-02-09
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analyzer.normalizer import SupplyNormalizer
from src.database.connection import get_connection


def print_header(args):
    """헤더 출력"""
    print("=" * 80)
    print("🔍 이상 수급 이벤트 탐지기 (Abnormal Supply Event Detector)")
    print("=" * 80)
    print(f"파라미터:")
    print(f"  - Z-Score 임계값: {args.threshold}")
    print(f"  - 상위 N개 종목: {args.top}")
    print(f"  - 방향: {args.direction} (buy=매수, sell=매도, both=양방향)")
    if args.date:
        print(f"  - 기준일: {args.date}")
    else:
        print(f"  - 기준일: 최신 거래일")
    print("=" * 80)


def format_signal(z_score: float, threshold: float) -> str:
    """시그널 포맷팅"""
    if abs(z_score) < threshold:
        return "⚪ NORMAL"
    elif z_score > 0:
        if z_score > threshold * 1.5:
            return "🟢 STRONG BUY"
        else:
            return "🔵 BUY"
    else:
        if z_score < -threshold * 1.5:
            return "🔴 STRONG SELL"
        else:
            return "🟠 SELL"


def format_sff(sff: float) -> str:
    """Sff 값 포맷팅 (백분율)"""
    if abs(sff) < 0.01:
        return f"{sff:+.4f}%"
    else:
        return f"{sff:+.2f}%"


def print_results(df, args):
    """결과 출력"""
    if df.empty:
        print("\n[INFO] 임계값을 초과하는 이상 수급 이벤트가 발견되지 않았습니다.")
        print(f"       임계값을 낮추거나 (--threshold 1.5) 기간을 조정해보세요.")
        return

    print(f"\n🎯 발견된 이상 수급 이벤트: {len(df)}건\n")

    for idx, row in df.iterrows():
        stock_name = row['stock_name']
        stock_code = row['stock_code']
        date = row['trade_date']
        foreign_z = row['foreign_zscore']
        inst_z = row['institution_zscore']
        combined_z = row['combined_zscore']
        foreign_sff = row['foreign_sff']
        inst_sff = row['institution_sff']
        combined_sff = row['combined_sff']

        # 시그널 타입
        signal = format_signal(combined_z, args.threshold)

        print(f"{signal} [{stock_name}] ({stock_code})")
        print(f"    📅 날짜: {date}")
        print(f"    📊 Z-Score:")
        print(f"       • 외국인: {foreign_z:+.2f} σ")
        print(f"       • 기관:   {inst_z:+.2f} σ")
        print(f"       • 합계:   {combined_z:+.2f} σ")
        print(f"    💰 Sff (유통시총 대비 순매수 비율):")
        print(f"       • 외국인: {format_sff(foreign_sff)}")
        print(f"       • 기관:   {format_sff(inst_sff)}")
        print(f"       • 합계:   {format_sff(combined_sff)}")
        print()

    print("=" * 80)
    print("📌 해석 가이드:")
    print("  - Z-Score > 2.0: 최근 60일 평균 대비 통계적으로 유의미한 강한 매수")
    print("  - Z-Score < -2.0: 최근 60일 평균 대비 통계적으로 유의미한 강한 매도")
    print("  - Sff: 유통시가총액 대비 순매수 비율 (시총 크기 정규화)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='이상 수급 이벤트 탐지 (Z-Score 기반)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (임계값 2.0, 상위 20개)
  python scripts/analysis/abnormal_supply_detector.py

  # 임계값 조정
  python scripts/analysis/abnormal_supply_detector.py --threshold 2.5

  # 매수 시그널만
  python scripts/analysis/abnormal_supply_detector.py --direction buy --top 30

  # 매도 시그널만
  python scripts/analysis/abnormal_supply_detector.py --direction sell --top 30

  # 특정 날짜 기준
  python scripts/analysis/abnormal_supply_detector.py --date 2026-02-09
        """
    )

    parser.add_argument('--threshold', type=float, default=2.0,
                       help='Z-Score 임계값 (기본: 2.0 = 표준편차 2배)')
    parser.add_argument('--top', type=int, default=20,
                       help='상위 N개 종목 표시 (기본: 20)')
    parser.add_argument('--direction', choices=['buy', 'sell', 'both'], default='both',
                       help='탐지 방향: buy(매수), sell(매도), both(양방향, 기본)')
    parser.add_argument('--date', help='기준일 (YYYY-MM-DD, 기본: 최신 거래일)')

    args = parser.parse_args()

    # 날짜 형식 검증
    if args.date:
        try:
            datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            print(f"[ERROR] Invalid date format: {args.date}")
            print("        Expected format: YYYY-MM-DD (e.g., 2026-02-09)")
            sys.exit(1)

    # 헤더 출력
    print_header(args)

    # 데이터베이스 연결
    conn = get_connection()

    try:
        # Normalizer 초기화
        normalizer = SupplyNormalizer(conn)

        # 이상 수급 탐지
        print("\n[INFO] Z-Score 계산 중...")
        df_abnormal = normalizer.get_abnormal_supply(
            threshold=args.threshold,
            end_date=args.date,
            top_n=args.top,
            direction=args.direction
        )

        # 결과 출력
        print_results(df_abnormal, args)

    except Exception as e:
        print(f"\n[ERROR] 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
