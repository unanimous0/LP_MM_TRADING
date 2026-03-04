"""
Stage 4 백테스팅 CLI 도구 (Week 5 버전)

백테스트 실행, 결과 출력, 시각화, 파라미터 최적화, Walk-Forward Analysis

Usage:
    # 기본 실행 (3개월)
    python scripts/analysis/backtest_runner.py

    # 전체 기간
    python scripts/analysis/backtest_runner.py --start 2022-01-03 --end 2026-03-03

    # 특정 패턴만
    python scripts/analysis/backtest_runner.py --pattern 급등형

    # 차트 생성 (화면 표시)
    python scripts/analysis/backtest_runner.py --plot

    # PNG 저장
    python scripts/analysis/backtest_runner.py --save-dir output/charts

    # PDF 리포트 생성
    python scripts/analysis/backtest_runner.py --save-pdf output/backtest_report.pdf

    # CSV + 차트 모두 저장
    python scripts/analysis/backtest_runner.py --save-csv output/trades.csv --save-dir output/charts

    # Optuna 파라미터 최적화 (기본 50 trials)
    python scripts/analysis/backtest_runner.py --optimize

    # Optuna 최적화 (100 trials, total_return 기준)
    python scripts/analysis/backtest_runner.py --optimize --n-trials 100 --metric total_return

    # 최적화 결과 CSV 저장
    python scripts/analysis/backtest_runner.py --optimize --opt-save-csv output/optimization.csv

    # Walk-Forward Analysis (기본: 6개월 학습, 1개월 검증, Optuna 50 trials)
    python scripts/analysis/backtest_runner.py --walk-forward --start 2023-01-01 --end 2025-12-31

    # Walk-Forward (Optuna trials 수 지정)
    python scripts/analysis/backtest_runner.py --walk-forward --n-trials 100

    # Walk-Forward 병렬 실행 (기간 단위)
    python scripts/analysis/backtest_runner.py --walk-forward --workers 4 --n-trials 30

    # Walk-Forward 결과 CSV 저장
    python scripts/analysis/backtest_runner.py --walk-forward --wf-save-csv output/walk_forward.csv
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_pg_engine
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.metrics import PerformanceMetrics
from src.backtesting.visualizer import BacktestVisualizer
from src.backtesting.optimizer import OptunaOptimizer
import pandas as pd


def print_results(result: dict):
    """백테스트 결과 출력 (PerformanceMetrics 사용)"""
    config = result['config']
    portfolio = result['portfolio']
    trades = result['trades']
    daily_values = result['daily_values']

    # PerformanceMetrics 생성
    metrics = PerformanceMetrics(
        trades=trades,
        daily_values=daily_values,
        initial_capital=config.initial_capital
    )

    print(f"\n{'='*80}")
    print(f"📊 백테스트 결과 요약")
    print(f"{'='*80}\n")

    # 기본 정보
    print(f"[기본 정보]")
    print(f"초기 자본금: {config.initial_capital:,.0f}원")
    final_value = daily_values.iloc[-1]['value'] if not daily_values.empty else config.initial_capital
    print(f"최종 자본금: {final_value:,.0f}원")
    print(f"총 수익률: {metrics.total_return():+.2f}%\n")

    # 리스크 지표
    mdd_info = metrics.max_drawdown()
    print(f"[리스크 지표]")
    print(f"최대 낙폭(MDD): {mdd_info['mdd']:.2f}%")
    if mdd_info['start_date']:
        print(f"  └─ {mdd_info['start_date']} ~ {mdd_info['end_date']}")
    print(f"샤프 비율: {metrics.sharpe_ratio():.2f}")
    print(f"칼마 비율: {metrics.calmar_ratio():.2f}\n")

    # 거래 통계
    if trades:
        wins = [t for t in trades if t.return_pct > 0]
        losses = [t for t in trades if t.return_pct <= 0]
        duration_stats = metrics.trade_duration_stats()

        print(f"[거래 통계]")
        print(f"총 거래 횟수: {len(trades)}건")
        print(f"승리: {len(wins)}건 ({metrics.win_rate():.1f}%)")
        print(f"패배: {len(losses)}건 ({100-metrics.win_rate():.1f}%)")
        print(f"평균 수익률: {metrics.avg_return():+.2f}%")
        print(f"평균 승리: {metrics.avg_win():+.2f}%")
        print(f"평균 손실: {metrics.avg_loss():+.2f}%")
        print(f"Profit Factor: {metrics.profit_factor():.2f}")
        print(f"평균 보유 기간: {duration_stats['avg']:.1f}일 (중앙값: {duration_stats['median']:.0f}일)")
        print(f"최대 연속 손실: {metrics.max_consecutive_losses()}회\n")

        # 패턴별 통계
        pattern_df = metrics.performance_by_pattern()
        if not pattern_df.empty:
            print(f"[패턴별 성과]")
            for _, row in pattern_df.iterrows():
                print(f"{row['pattern']}: {row['trades']:.0f}건 | "
                      f"승률 {row['win_rate']:.1f}% | "
                      f"평균 {row['avg_return']:+.2f}% | "
                      f"보유 {row['avg_hold_days']:.1f}일")
            print()

        # 시그널별 통계
        signal_df = metrics.performance_by_signal_count()
        if not signal_df.empty:
            print(f"[시그널별 성과]")
            for _, row in signal_df.iterrows():
                star = " ⭐" if row['signal_count'] >= 2 else ""
                print(f"시그널 {row['signal_count']:.0f}개: {row['trades']:.0f}건 | "
                      f"승률 {row['win_rate']:.1f}% | "
                      f"평균 {row['avg_return']:+.2f}%{star}")

    else:
        print("[경고] 거래가 없습니다!")

    print(f"\n{'='*80}\n")


def save_trades_to_csv(trades, filepath: str):
    """거래 내역 CSV 저장"""
    if not trades:
        print("[WARN] 저장할 거래가 없습니다.")
        return

    df = pd.DataFrame([t.to_dict() for t in trades])
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"\n✅ 거래 내역 저장: {filepath}")


def run_walk_forward(args):
    """Walk-Forward Analysis 실행 (Optuna Bayesian Optimization)"""
    from src.backtesting.walk_forward import WalkForwardAnalyzer, WalkForwardConfig

    wf_config = WalkForwardConfig(
        train_months=args.train_months,
        val_months=args.val_months,
        step_months=args.step_months,
        metric=args.metric,
        workers=args.workers,
        n_trials=args.n_trials,
    )
    base_config = BacktestConfig(
        initial_capital=args.capital,
        max_positions=args.max_positions,
        strategy=args.strategy,
        reverse_signal_threshold=args.reverse_threshold,
        max_hold_days=args.max_days,
        force_exit_on_end=False,
        use_tc=not args.no_tc,
        use_divergence=not args.no_divergence,
    )
    analyzer = WalkForwardAnalyzer(
        start_date=args.start,
        end_date=args.end,
        wf_config=wf_config,
        base_config=base_config,
    )
    analyzer.run(verbose=True)
    analyzer.print_results()

    if args.wf_save_csv:
        Path(args.wf_save_csv).parent.mkdir(parents=True, exist_ok=True)
        analyzer.summary().to_csv(args.wf_save_csv, index=False, encoding='utf-8-sig')
        print(f"✅ Walk-Forward 결과 저장: {args.wf_save_csv}")


def run_optimization(args):
    """Optuna Bayesian Optimization 최적화 실행"""
    base_config = BacktestConfig(
        initial_capital=args.capital,
        max_positions=args.max_positions,
        strategy=args.strategy,
        reverse_signal_threshold=args.reverse_threshold,
        max_hold_days=args.max_days,
        force_exit_on_end=False,
        use_tc=not args.no_tc,
        use_divergence=not args.no_divergence,
    )

    optimizer = OptunaOptimizer(
        start_date=args.start,
        end_date=args.end,
        base_config=base_config,
    )

    result = optimizer.optimize(
        n_trials=args.n_trials,
        metric=args.metric,
        verbose=True,
    )

    if result is not None and args.opt_save_csv:
        # dict → 1행 DataFrame 변환
        row = {**result['params'], args.metric: result[args.metric],
               'total_complete': result['total_complete'],
               'total_pruned': result['total_pruned']}
        df = pd.DataFrame([row])
        Path(args.opt_save_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.opt_save_csv, index=False, encoding='utf-8-sig')
        print(f"✅ 최적화 결과 저장: {args.opt_save_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='백테스팅 CLI 도구 (Week 5 버전 - Walk-Forward Analysis 추가)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (3개월)
  python scripts/analysis/backtest_runner.py

  # 전체 기간
  python scripts/analysis/backtest_runner.py --start 2022-01-03 --end 2026-03-03

  # 급등형 종목만
  python scripts/analysis/backtest_runner.py --pattern 급등형

  # 숏 전략 (순매도)
  python scripts/analysis/backtest_runner.py --strategy short

  # 롱+숏 병행
  python scripts/analysis/backtest_runner.py --strategy both

  # 차트 생성 및 화면 표시
  python scripts/analysis/backtest_runner.py --plot

  # PNG 차트 저장
  python scripts/analysis/backtest_runner.py --save-dir output/charts

  # PDF 리포트 생성
  python scripts/analysis/backtest_runner.py --save-pdf output/report.pdf

  # CSV + 차트 모두 저장
  python scripts/analysis/backtest_runner.py --save-csv output/trades.csv --save-dir output/charts

  # Optuna 파라미터 최적화
  python scripts/analysis/backtest_runner.py --optimize

  # Optuna 최적화 (100 trials, total_return 기준)
  python scripts/analysis/backtest_runner.py --optimize --n-trials 100 --metric total_return

  # 최적화 결과 CSV 저장
  python scripts/analysis/backtest_runner.py --optimize --opt-save-csv output/optimization.csv

  # Walk-Forward Analysis (Optuna, 기본 50 trials)
  python scripts/analysis/backtest_runner.py --walk-forward --start 2023-01-01 --end 2025-12-31

  # Walk-Forward (100 trials, 4 workers 병렬)
  python scripts/analysis/backtest_runner.py --walk-forward --n-trials 100 --workers 4
        """
    )

    # 기간 설정
    parser.add_argument('--start', default='2022-01-03', help='시작일 (YYYY-MM-DD)')
    parser.add_argument('--end', default='2024-03-31', help='종료일 (YYYY-MM-DD)')

    # 진입 조건
    parser.add_argument('--min-score', type=float, default=60, help='최소 패턴 점수 (0~100)')
    parser.add_argument('--min-signals', type=int, default=1, help='최소 시그널 개수 (0~3)')

    # 청산 조건
    parser.add_argument('--target', type=float, default=0.15, help='목표 수익률 (예: 0.15 = 15%%, 순수 가격 변화율)')
    parser.add_argument('--stop', type=float, default=-0.075, help='손절 비율 (예: -0.075 = -7.5%%, 순수 가격 변화율)')
    parser.add_argument('--max-days', type=int, default=999, help='최대 보유 기간 (일, 999 = 무제한)')
    parser.add_argument('--reverse-threshold', type=float, default=60, help='반대 수급 손절 점수 (예: 60 = 60점 이상)')

    # 포트폴리오 설정
    parser.add_argument('--capital', type=float, default=10_000_000, help='초기 자본금 (원)')
    parser.add_argument('--max-positions', type=int, default=5, help='최대 동시 보유 종목 수')

    # 패턴 필터링
    parser.add_argument('--pattern', choices=['급등형', '지속형', '전환형', '기타'],
                        help='특정 패턴만 (기본: 전체)')

    # 전략 방향
    parser.add_argument('--strategy', choices=['long', 'short', 'both'], default='long',
                        help='전략 방향 (long: 순매수, short: 순매도, both: 롱+숏, 기본: long)')

    # 스코어링 버전 (2026-02-25 개선 항목 개별 토글)
    parser.add_argument('--no-tc', action='store_true',
                        help='Temporal Consistency 비활성화 (use_tc=False)')
    parser.add_argument('--no-divergence', action='store_true',
                        help='Divergence 비활성화 (use_divergence=False)')

    # 출력 설정
    parser.add_argument('--save-csv', help='거래 내역 CSV 저장 경로')
    parser.add_argument('--quiet', action='store_true', help='진행 상황 출력 안함')

    # 시각화 옵션 (Week 3 - matplotlib)
    parser.add_argument('--plot', action='store_true', help='차트 생성 및 화면 표시')
    parser.add_argument('--save-dir', help='차트 PNG 저장 디렉토리')
    parser.add_argument('--save-pdf', help='차트 PDF 리포트 저장 경로')
    parser.add_argument('--save-daily-values', help='일별 포트폴리오 가치 CSV 저장 경로')

    # 인터랙티브 시각화 옵션 (Option 2 - Plotly)
    parser.add_argument('--save-html', help='HTML 인터랙티브 리포트 저장 경로')
    parser.add_argument('--html-cdn', action='store_true',
                        help='HTML에 CDN 방식으로 Plotly.js 로드 (파일 경량, 인터넷 필요)')

    # 최적화 옵션 (Week 4)
    parser.add_argument('--optimize', action='store_true', help='Optuna 파라미터 최적화 실행')
    parser.add_argument('--workers', type=int, default=1, help='병렬 처리 worker 수 (기본: 1)')
    parser.add_argument('--metric', default='sharpe_ratio',
                        choices=['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor'],
                        help='최적화 평가 지표 (기본: sharpe_ratio)')
    parser.add_argument('--opt-save-csv', help='최적화 결과 CSV 저장 경로')

    # Walk-Forward 옵션 (Week 5)
    parser.add_argument('--walk-forward', action='store_true',
                        help='Walk-Forward Analysis 실행')
    parser.add_argument('--train-months', type=int, default=6,
                        help='학습 기간 (개월, 기본: 6)')
    parser.add_argument('--val-months', type=int, default=1,
                        help='검증 기간 (개월, 기본: 1)')
    parser.add_argument('--step-months', type=int, default=1,
                        help='롤링 스텝 (개월, 기본: 1)')
    parser.add_argument('--wf-save-csv', help='Walk-Forward 결과 CSV 저장 경로')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Optuna Trial 수 (--optimize, --walk-forward 공용, 기본: 50)')

    args = parser.parse_args()

    # Walk-Forward 모드
    if args.walk_forward:
        run_walk_forward(args)
        return

    # 최적화 모드
    if args.optimize:
        run_optimization(args)
        return

    # 설정 생성
    allowed_patterns = [args.pattern] if args.pattern else None

    config = BacktestConfig(
        initial_capital=args.capital,
        max_positions=args.max_positions,
        min_score=args.min_score,
        min_signals=args.min_signals,
        target_return=args.target,
        stop_loss=args.stop,
        max_hold_days=args.max_days,
        reverse_signal_threshold=args.reverse_threshold,
        allowed_patterns=allowed_patterns,
        strategy=args.strategy,
        force_exit_on_end=False,
        use_tc=not args.no_tc,
        use_divergence=not args.no_divergence,
    )

    # 데이터베이스 연결
    pg_engine = get_pg_engine()

    # 백테스트 실행
    engine = BacktestEngine(pg_engine, config)

    result = engine.run(
        start_date=args.start,
        end_date=args.end,
        verbose=not args.quiet
    )

    # 결과 출력
    print_results(result)

    # CSV 저장
    if args.save_csv:
        save_trades_to_csv(result['trades'], args.save_csv)

    # 일별 포트폴리오 가치 CSV 저장
    if args.save_daily_values:
        result['daily_values'].to_csv(args.save_daily_values, index=False, encoding='utf-8-sig')
        print(f"✅ 일별 포트폴리오 가치 저장: {args.save_daily_values}")

    # 인터랙티브 HTML 리포트 (Option 2 - Plotly)
    if args.save_html:
        from src.backtesting.plotly_visualizer import PlotlyVisualizer
        print("\n" + "="*80)
        print("📊 Plotly HTML 리포트 생성 중...")
        print("="*80)
        pv = PlotlyVisualizer(
            trades=result['trades'],
            daily_values=result['daily_values'],
            initial_capital=config.initial_capital,
        )
        pv.create_dashboard(
            save_html=args.save_html,
            show=False,
            cdn=args.html_cdn,
        )

    # 시각화 (Week 3 - matplotlib)
    if args.plot or args.save_dir or args.save_pdf:
        if not result['trades']:
            print("\n⚠️  거래가 없어서 차트를 생성할 수 없습니다.")
        else:
            print("\n" + "="*80)
            print("📊 차트 생성 중...")
            print("="*80)

            visualizer = BacktestVisualizer(
                trades=result['trades'],
                daily_values=result['daily_values'],
                initial_capital=config.initial_capital
            )

            # 모든 차트 생성
            visualizer.plot_all(
                save_dir=args.save_dir,
                save_pdf=args.save_pdf,
                show=args.plot
            )

    # PostgreSQL engine은 싱글턴 — close 불필요


if __name__ == '__main__':
    main()
