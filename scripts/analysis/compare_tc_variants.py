#!/usr/bin/env python3
"""tc_bonus 변형 백테스트 비교 스크립트

4가지 변형을 비교:
  A. use_tc=False (tc_bonus 완전 제거)
  B. center=0.5, scale=10 (범위 ±5점으로 축소)
  C. center=0.3, scale=20 (대형주 패널티 제거)
  D. center=0.5, scale=20 (현재 기본값)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import time
from src.database.connection import get_pg_engine
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.metrics import PerformanceMetrics


def run_variant(label, pg_engine, start, end, **config_overrides):
    """단일 변형 백테스트 실행"""
    config = BacktestConfig(
        initial_capital=100_000_000,
        min_score=70,
        min_signals=1,
        target_return=0.10,
        stop_loss=-0.05,
        max_hold_days=20,
        strategy='long',
        institution_weight=0.3,
        force_exit_on_end=True,
        **config_overrides,
    )
    engine = BacktestEngine(pg_engine, config)
    t0 = time.time()
    result = engine.run(start, end, verbose=False)
    elapsed = time.time() - t0

    trades = result.get('trades', [])
    daily_values = result.get('daily_values', {})

    if not trades:
        return {
            'label': label,
            'trades': 0,
            'total_return': 0,
            'win_rate': 0,
            'mdd': 0,
            'sharpe': 0,
            'calmar': 0,
            'profit_factor': 0,
            'avg_return': 0,
            'elapsed': elapsed,
        }

    metrics = PerformanceMetrics(trades, daily_values, config.initial_capital)
    mdd_info = metrics.max_drawdown()

    return {
        'label': label,
        'trades': len(trades),
        'total_return': metrics.total_return(),
        'win_rate': metrics.win_rate(),
        'mdd': mdd_info.get('max_drawdown_pct', 0),
        'sharpe': metrics.sharpe_ratio(),
        'calmar': metrics.calmar_ratio(),
        'profit_factor': metrics.profit_factor(),
        'avg_return': metrics.avg_return(),
        'elapsed': elapsed,
    }


def main():
    pg_engine = get_pg_engine()

    # 2개 기간 테스트 (상승장 + 하락장)
    periods = [
        ('2023-01-01', '2024-06-30', '23.01~24.06 (1.5년)'),
        ('2024-07-01', '2025-12-31', '24.07~25.12 (1.5년)'),
        ('2023-01-01', '2025-12-31', '23.01~25.12 (전체 3년)'),
    ]

    market_cap_top_n = 300  # 시총 상위 300 필터

    variants = [
        ('D. 현재 (c=0.5, s=20)', dict(use_tc=True, tc_center=0.5, tc_scale=20.0, market_cap_top_n=market_cap_top_n)),
        ('A. tc_bonus 제거',       dict(use_tc=False, market_cap_top_n=market_cap_top_n)),
        ('B. 범위 축소 (c=0.5, s=10)', dict(use_tc=True, tc_center=0.5, tc_scale=10.0, market_cap_top_n=market_cap_top_n)),
        ('C. 기준 하향 (c=0.3, s=20)', dict(use_tc=True, tc_center=0.3, tc_scale=20.0, market_cap_top_n=market_cap_top_n)),
    ]

    for start, end, period_label in periods:
        print(f"\n{'='*80}")
        print(f"  기간: {period_label}")
        print(f"{'='*80}")

        results = []
        for label, overrides in variants:
            print(f"  실행 중: {label}...", end='', flush=True)
            r = run_variant(label, pg_engine, start, end, **overrides)
            results.append(r)
            print(f" {r['elapsed']:.1f}초")

        # 결과 테이블
        print(f"\n  {'변형':<28} {'거래수':>6} {'총수익률':>8} {'승률':>6} {'MDD':>7} {'샤프':>6} {'칼마':>6} {'PF':>6} {'평균수익':>8}")
        print(f"  {'-'*28} {'-'*6} {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
        for r in results:
            print(f"  {r['label']:<28} {r['trades']:>6} {r['total_return']:>+7.1f}% {r['win_rate']:>5.1f}% {r['mdd']:>6.1f}% {r['sharpe']:>6.2f} {r['calmar']:>6.2f} {r['profit_factor']:>6.2f} {r['avg_return']:>+7.2f}%")


if __name__ == '__main__':
    main()
