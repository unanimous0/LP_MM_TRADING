"""
Stage 2 히트맵 생성기 (CLI)

8개 기간 (1D~2Y) 시공간 히트맵 생성
- 350종목 × 8기간 매트릭스
- Y축 강도순 정렬 (상단 = 강한 매수)
- 파라미터 조정 가능 (기간, 색상, 필터)

Usage:
    # 기본 실행 (전체 8개 기간)
    python scripts/analysis/heatmap_generator.py

    # 단기 3개 기간만
    python scripts/analysis/heatmap_generator.py --periods 1D 1W 1M

    # 섹터 필터링
    python scripts/analysis/heatmap_generator.py --sector 반도체

    # 색상 임계값 조정
    python scripts/analysis/heatmap_generator.py --threshold 2.5
"""

import argparse
import sys
import time
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, DEFAULT_CONFIG
from src.database.connection import get_connection
from src.analyzer.normalizer import SupplyNormalizer
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
from src.visualizer.heatmap_renderer import HeatmapRenderer
from src.utils import sanitize_sector_name


def main():
    parser = argparse.ArgumentParser(
        description='수급 시공간 히트맵 생성 (Stage 2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (전체 8개 기간)
  python scripts/analysis/heatmap_generator.py

  # 단기 3개 기간만
  python scripts/analysis/heatmap_generator.py --periods 1D 1W 1M

  # 섹터 필터링
  python scripts/analysis/heatmap_generator.py --sector 반도체

  # 색상 임계값 조정 (±2.5σ)
  python scripts/analysis/heatmap_generator.py --threshold 2.5

  # 상위 50개 종목만
  python scripts/analysis/heatmap_generator.py --top 50

  # 고해상도 출력
  python scripts/analysis/heatmap_generator.py --dpi 300
        """
    )

    # ============================================================================
    # 기간 선택
    # ============================================================================
    parser.add_argument(
        '--periods', nargs='+',
        choices=['1D', '1W', '1M', '3M', '6M', '1Y', '2Y'],
        help='분석 기간 (기본: 전체 8개)'
    )

    # ============================================================================
    # 필터링
    # ============================================================================
    parser.add_argument(
        '--sector', type=str,
        help='특정 섹터만 (예: 반도체, 의료, 제약)'
    )

    parser.add_argument(
        '--top', type=int,
        help='상위 N개 종목만 (Z-Score 강도 기준)'
    )

    parser.add_argument(
        '--direction', choices=['buy', 'sell', 'both'],
        default='both',
        help='매수/매도 방향 (buy: 매수 상위, sell: 매도 상위, both: 전체)'
    )

    parser.add_argument(
        '--min-cap', type=float,
        help='최소 시가총액 (억원 단위)'
    )

    parser.add_argument(
        '--market', choices=['KOSPI200', 'KOSDAQ150'],
        help='특정 시장만'
    )

    # ============================================================================
    # 시각화
    # ============================================================================
    parser.add_argument(
        '--threshold', type=float, default=2.0,
        help='Z-Score 색상 강조 임계값 (기본: 2.0σ)'
    )

    parser.add_argument(
        '--colormap', default='RdYlGn',
        help='색상 스킴 (기본: RdYlGn, 옵션: coolwarm, seismic)'
    )

    parser.add_argument(
        '--figsize', nargs=2, type=int,
        help='차트 크기 (가로 세로, 예: 30 18)'
    )

    parser.add_argument(
        '--dpi', type=int,
        help='해상도 (기본: 150, 고해상도: 300)'
    )

    parser.add_argument(
        '--sort-by',
        choices=['recent', 'momentum', 'weighted', 'average'],
        default='recent',
        help='''Y축 정렬 기준:
  recent: 최근 기간(1W+1M) 우선 (기본, 추천!)
  momentum: 수급 모멘텀(1W-2Y) - 전환점 포착
  weighted: 가중 평균 (최근 높은 가중치)
  average: 단순 평균 (deprecated)'''
    )

    # ============================================================================
    # 성능
    # ============================================================================
    parser.add_argument(
        '--no-cache', action='store_true',
        help='Sff 캐싱 비활성화 (디버깅용)'
    )

    parser.add_argument(
        '--parallel', action='store_true',
        help='병렬 처리 활성화 (4 CPU 코어 활용, 73%% 빠름)'
    )

    parser.add_argument(
        '--workers', type=int, default=4,
        help='병렬 워커 스레드 수 (기본: 4)'
    )

    # ============================================================================
    # 출력
    # ============================================================================
    parser.add_argument(
        '--output', default='output/heatmap.png',
        help='저장 경로 (기본: output/heatmap.png)'
    )

    parser.add_argument(
        '--save-csv', action='store_true',
        help='CSV 동시 저장 (Z-Score 매트릭스)'
    )

    args = parser.parse_args()

    # ============================================================================
    # Config 로드 (기본값 + CLI 오버라이드)
    # ============================================================================
    config = load_config(cli_overrides=vars(args))

    # 기간 선택
    if args.periods:
        selected_periods = {p: DEFAULT_CONFIG['periods'][p] for p in args.periods}
    else:
        selected_periods = config['periods']

    # ============================================================================
    # 헤더 출력
    # ============================================================================
    print("=" * 80)
    print("🔥 Stage 2 시공간 히트맵 생성기")
    print("=" * 80)
    print(f"파라미터:")
    print(f"  - 분석 기간: {list(selected_periods.keys())}")
    print(f"  - 색상 스케일: {config['visualization']['colormap']}")
    print(f"  - 색상 임계값: ±{args.threshold}σ")
    print(f"  - Y축 정렬: Z-Score 강도순 (상단 = 강한 매수)")

    if args.sector:
        print(f"  - 섹터 필터: {args.sector}")
    if args.top:
        direction_map = {'buy': '매수', 'sell': '매도', 'both': '전체'}
        print(f"  - 상위 종목: {args.top}개 ({direction_map[args.direction]})")
    if args.min_cap:
        print(f"  - 최소 시총: {args.min_cap}억원")
    if args.market:
        print(f"  - 시장 필터: {args.market}")

    print(f"  - 출력 경로: {args.output}")
    print(f"  - Sff 캐싱: {'활성' if not args.no_cache else '비활성'}")
    print("=" * 80)

    # ============================================================================
    # DB 연결
    # ============================================================================
    conn = get_connection()

    try:
        start_time = time.time()

        # Step 1: 종목 필터링 (옵션)
        stock_codes = None
        if args.sector or args.top or args.market:
            stock_codes = _get_filtered_stocks(conn, config)
            print(f"[INFO] Filtered stocks: {len(stock_codes)}")

        # Step 2: Normalizer 초기화
        normalizer = SupplyNormalizer(conn)

        # Step 3: 최적화된 계산기
        optimizer = OptimizedMultiPeriodCalculator(
            normalizer,
            enable_caching=not args.no_cache,
            enable_parallel=args.parallel,
            max_workers=args.workers
        )

        # Step 4: 8개 기간 Z-Score 계산
        print("\n[INFO] Calculating Z-Scores for all periods...")
        zscore_matrix = optimizer.calculate_multi_period_zscores(
            selected_periods,
            stock_codes=stock_codes
        )

        print(f"[OK] Calculated {len(zscore_matrix)} stocks × {len(selected_periods)} periods")

        # Step 5: 매수/매도 방향 및 상위 N개 필터링 (평균 Z-Score 기준)
        # 평균 Z-Score 계산 (모든 기간의 평균)
        zscore_matrix['_avg_zscore'] = zscore_matrix.mean(axis=1)

        # 매수/매도 필터링
        if args.direction == 'buy':
            # 매수 상위: 평균 Z-Score 높은 순
            if args.top:
                zscore_matrix = zscore_matrix.nlargest(args.top, '_avg_zscore')
                print(f"[INFO] Filtered to top {args.top} BUY stocks (highest avg Z-Score)")
        elif args.direction == 'sell':
            # 매도 상위: 평균 Z-Score 낮은 순
            if args.top:
                zscore_matrix = zscore_matrix.nsmallest(args.top, '_avg_zscore')
                print(f"[INFO] Filtered to top {args.top} SELL stocks (lowest avg Z-Score)")
        else:  # both
            # 전체: 평균 절대값 기준 상위
            if args.top and len(zscore_matrix) > args.top:
                zscore_matrix['_abs_avg'] = zscore_matrix['_avg_zscore'].abs()
                zscore_matrix = zscore_matrix.nlargest(args.top, '_abs_avg')
                zscore_matrix = zscore_matrix.drop(columns=['_abs_avg'])
                print(f"[INFO] Filtered to top {args.top} stocks (by avg Z-Score)")

        # 평균 컬럼 제거 (히트맵에 표시하지 않음)
        if '_avg_zscore' in zscore_matrix.columns:
            zscore_matrix = zscore_matrix.drop(columns=['_avg_zscore'])

        # Step 6: 히트맵 렌더링
        print("\n[INFO] Rendering heatmap...")
        renderer = HeatmapRenderer(config)
        renderer.render_multi_period_heatmap(zscore_matrix, args.output)

        # Step 7: CSV 저장 (옵션)
        if args.save_csv:
            csv_path = args.output.replace('.png', '.csv')
            zscore_matrix.to_csv(csv_path)
            print(f"[OK] CSV saved to: {csv_path}")

        elapsed_time = time.time() - start_time

        # ============================================================================
        # 결과 요약
        # ============================================================================
        print("=" * 80)
        print("✅ 히트맵 생성 완료!")
        print(f"  - 처리 종목: {len(zscore_matrix)}개")
        print(f"  - 처리 기간: {len(selected_periods)}개")
        print(f"  - 소요 시간: {elapsed_time:.1f}초")
        print(f"  - 저장 경로: {args.output}")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


def _get_filtered_stocks(conn, config) -> list:
    """
    섹터/시장/시총 필터링된 종목 코드 반환

    Args:
        conn: DB 연결
        config: 설정 딕셔너리

    Returns:
        list: 종목 코드 리스트

    Raises:
        ValueError: 유효하지 않은 섹터명 또는 시장명
    """
    import pandas as pd

    where_clauses = []

    # 섹터 필터 (보안: 입력 검증)
    if config['filtering'].get('sectors'):
        sectors = config['filtering']['sectors']
        # 각 섹터명 검증
        validated_sectors = [sanitize_sector_name(s) for s in sectors]
        sectors_str = "','".join(validated_sectors)
        where_clauses.append(f"sector IN ('{sectors_str}')")

    # 시장 필터 (보안: 화이트리스트 검증)
    if config['filtering'].get('market'):
        market = config['filtering']['market']
        # 허용된 시장명만 허용
        allowed_markets = ['KOSPI200', 'KOSDAQ150']
        if market not in allowed_markets:
            raise ValueError(f"Invalid market: {market}. Allowed: {allowed_markets}")
        where_clauses.append(f"market_id = (SELECT market_id FROM markets WHERE market_name = '{market}')")

    where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

    query = f"""
    SELECT stock_code
    FROM stocks
    {where_sql}
    """

    df = pd.read_sql(query, conn)
    return df['stock_code'].tolist()


if __name__ == '__main__':
    main()
