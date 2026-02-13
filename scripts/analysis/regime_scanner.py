"""
Stage 3 수급 레짐 스캐너 (CLI)

Stage 1~3 전체 파이프라인 통합 실행:
- Stage 1: 이상 수급 탐지 (Sff/Z-Score)
- Stage 2: 시공간 히트맵 (6개 기간)
- Stage 3: 패턴 분류 (3개 바구니) + 시그널 통합

Usage:
    # 기본 실행 (전체 종목, 모든 패턴)
    python scripts/analysis/regime_scanner.py

    # 전환돌파형 종목만, 점수 70점 이상
    python scripts/analysis/regime_scanner.py --pattern 전환돌파형 --min-score 70

    # 시그널 2개 이상, 상위 10개
    python scripts/analysis/regime_scanner.py --min-signals 2 --top 10

    # 섹터 필터링 + CSV 저장
    python scripts/analysis/regime_scanner.py --sector "반도체 및 관련장비" --save-csv
"""

import argparse
import sys
import time
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import DEFAULT_CONFIG
from src.database.connection import get_connection
from src.analyzer.normalizer import SupplyNormalizer
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
from src.analyzer.pattern_classifier import PatternClassifier
from src.analyzer.signal_detector import SignalDetector
from src.analyzer.integrated_report import IntegratedReport


def main():
    parser = argparse.ArgumentParser(
        description='수급 레짐 스캐너 (Stage 3 통합)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (전체 종목, 모든 패턴)
  python scripts/analysis/regime_scanner.py

  # 모멘텀형 종목만, 점수 70점 이상
  python scripts/analysis/regime_scanner.py --pattern 모멘텀형 --min-score 70

  # 지속형 + 시그널 2개 이상, 상위 10개
  python scripts/analysis/regime_scanner.py --pattern 지속형 --min-signals 2 --top 10

  # 섹터 필터링 (반도체)
  python scripts/analysis/regime_scanner.py --sector "반도체 및 관련장비"

  # CSV 저장
  python scripts/analysis/regime_scanner.py --save-csv output/regime_report.csv

  # 콘솔 요약 카드 출력
  python scripts/analysis/regime_scanner.py --print-cards --top 5
        """
    )

    # ============================================================================
    # 패턴 필터링
    # ============================================================================
    parser.add_argument(
        '--pattern',
        choices=['모멘텀형', '지속형', '전환형', '기타'],
        help='특정 패턴만 (기본: 전체)'
    )

    parser.add_argument(
        '--min-score', type=float,
        default=0,
        help='최소 패턴 점수 (0~100, 기본: 0)'
    )

    parser.add_argument(
        '--min-signals', type=int,
        default=0,
        help='최소 시그널 개수 (0~3, 기본: 0)'
    )

    # ============================================================================
    # 섹터/종목 필터링
    # ============================================================================
    parser.add_argument(
        '--sector', type=str,
        help='특정 섹터만 (예: "반도체 및 관련장비")'
    )

    parser.add_argument(
        '--top', type=int,
        help='상위 N개 종목만 (점수 기준, 기본: 전체)'
    )

    # ============================================================================
    # 출력 옵션
    # ============================================================================
    parser.add_argument(
        '--save-csv', type=str,
        nargs='?',
        const='output/regime_report.csv',
        help='CSV 저장 경로 (기본: output/regime_report.csv)'
    )

    parser.add_argument(
        '--print-cards',
        action='store_true',
        help='종목별 요약 카드 출력'
    )

    parser.add_argument(
        '--print-summary',
        action='store_true',
        help='패턴별 요약 통계 출력'
    )

    parser.add_argument(
        '--watchlist',
        action='store_true',
        help='관심 종목 리스트 출력 (점수 70+, 시그널 2+)'
    )

    # ============================================================================
    # 디버깅
    # ============================================================================
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='상세 로그 출력'
    )

    args = parser.parse_args()

    # ============================================================================
    # 1. 데이터베이스 연결
    # ============================================================================
    start_time = time.time()

    print("\n" + "="*80)
    print("🔍 수급 레짐 스캐너 (Stage 3 통합)")
    print("="*80 + "\n")

    conn = get_connection()

    try:
        # ============================================================================
        # 2. Stage 1: 데이터 정규화
        # ============================================================================
        if args.verbose:
            print("[Stage 1] 데이터 정규화 (Sff/Z-Score)...")

        normalizer = SupplyNormalizer(conn)

        # ============================================================================
        # 3. Stage 2: 시공간 히트맵
        # ============================================================================
        if args.verbose:
            print("[Stage 2] 시공간 히트맵 계산 (6개 기간)...")

        optimizer = OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)
        zscore_matrix = optimizer.calculate_multi_period_zscores(DEFAULT_CONFIG['periods'])

        # stock_code를 인덱스에서 컬럼으로 변환
        zscore_matrix = zscore_matrix.reset_index()

        if zscore_matrix.empty:
            print("[ERROR] No data found. Exiting.")
            return

        if args.verbose:
            print(f"  → {len(zscore_matrix)} 종목 로드됨")

        # ============================================================================
        # 4. Stage 3: 패턴 분류
        # ============================================================================
        if args.verbose:
            print("[Stage 3] 패턴 분류 (3개 바구니)...")

        classifier = PatternClassifier()
        classified_df = classifier.classify_all(zscore_matrix)

        if args.verbose:
            print(f"  → 패턴 분류 완료: {len(classified_df)} 종목")

        # ============================================================================
        # 5. 시그널 탐지
        # ============================================================================
        if args.verbose:
            print("[Stage 3] 시그널 탐지 (MA/가속도/동조율)...")

        detector = SignalDetector(conn)
        signals_df = detector.detect_all_signals()

        if args.verbose:
            print(f"  → 시그널 탐지 완료: {len(signals_df)} 종목")

        # ============================================================================
        # 6. 통합 리포트 생성
        # ============================================================================
        if args.verbose:
            print("[Stage 3] 통합 리포트 생성...")

        report_gen = IntegratedReport(conn)
        report_df = report_gen.generate_report(classified_df, signals_df)

        if args.verbose:
            print(f"  → 통합 리포트 생성 완료: {len(report_df)} 종목")

        # ============================================================================
        # 7. 필터링
        # ============================================================================
        df_filtered = report_gen.filter_report(
            report_df,
            pattern=args.pattern,
            sector=args.sector,
            min_score=args.min_score,
            min_signal_count=args.min_signals,
            top_n=args.top
        )

        elapsed = time.time() - start_time

        # ============================================================================
        # 8. 출력
        # ============================================================================
        print(f"✅ 분석 완료! ({elapsed:.1f}초 소요)\n")

        # 8-1. 기본 출력 (테이블)
        if len(df_filtered) > 0:
            print(f"📊 필터링 결과: {len(df_filtered)} 종목")
            print("-"*80)

            # 핵심 컬럼만 출력
            display_cols = ['stock_code', 'stock_name', 'sector', 'pattern', 'score', 'signal_count']
            display_cols = [col for col in display_cols if col in df_filtered.columns]

            # 상위 20개만 미리보기
            preview_df = df_filtered[display_cols].head(20)
            print(preview_df.to_string(index=False))

            if len(df_filtered) > 20:
                print(f"\n... 외 {len(df_filtered) - 20}개 종목 (--save-csv로 전체 확인)")

        else:
            print("⚠️  필터링 조건에 맞는 종목이 없습니다.")

        print()

        # 8-2. 요약 카드 출력
        if args.print_cards and len(df_filtered) > 0:
            top_n = min(args.top if args.top else 10, len(df_filtered))
            report_gen.print_summary_card(df_filtered, top_n=top_n)

        # 8-3. 패턴별 요약 통계
        if args.print_summary:
            print("\n" + "="*80)
            print("📈 패턴별 요약 통계")
            print("="*80 + "\n")

            summary_df = report_gen.get_pattern_summary_report(report_df)
            print(summary_df.to_string(index=False))
            print()

        # 8-4. 관심 종목 리스트
        if args.watchlist:
            print("\n" + "="*80)
            print("⭐ 관심 종목 리스트 (점수 70+, 시그널 2+)")
            print("="*80 + "\n")

            watchlist = report_gen.get_watchlist(report_df, min_score=70, min_signal_count=2)

            for pattern, df_watch in watchlist.items():
                if len(df_watch) > 0:
                    print(f"\n[{pattern}] ({len(df_watch)} 종목)")
                    print("-"*40)
                    watch_cols = ['stock_code', 'stock_name', 'score', 'signal_count']
                    watch_cols = [col for col in watch_cols if col in df_watch.columns]
                    print(df_watch[watch_cols].head(10).to_string(index=False))

            print()

        # 8-5. CSV 저장
        if args.save_csv:
            # 출력 디렉토리 생성
            output_path = Path(args.save_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            report_gen.export_to_csv(df_filtered, str(output_path), include_all_columns=True)
            print(f"💾 CSV 저장 완료: {output_path}\n")

        # ============================================================================
        # 9. 요약 정보
        # ============================================================================
        print("="*80)
        print("📌 실행 요약")
        print("="*80)
        print(f"전체 종목: {len(report_df)}")
        print(f"필터링 결과: {len(df_filtered)}")

        if args.pattern:
            print(f"패턴 필터: {args.pattern}")
        if args.sector:
            print(f"섹터 필터: {args.sector}")
        if args.min_score > 0:
            print(f"최소 점수: {args.min_score}")
        if args.min_signals > 0:
            print(f"최소 시그널: {args.min_signals}")

        print(f"소요 시간: {elapsed:.1f}초")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    finally:
        conn.close()


if __name__ == '__main__':
    main()
