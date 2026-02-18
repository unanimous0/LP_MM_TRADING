"""주성엔지니어링(036930) 2025년 1분기 백테스트 (조건부 Z-Score 적용 후)"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_connection
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.metrics import PerformanceMetrics
import pandas as pd

conn = get_connection()

config = BacktestConfig(
    initial_capital=10_000_000,
    max_positions=5,
    min_score=70,
    min_signals=1,
    target_return=0.15,        # +15%
    stop_loss=-0.08,           # -8%
    max_hold_days=999,         # 보유기간 제한 없음
    reverse_signal_threshold=60,  # 반대수급 60점
    strategy='long',
    force_exit_on_end=False
)

engine = BacktestEngine(conn, config)

# 주성엔지니어링만 필터링하기 위해 _scan_signals_on_date를 래핑
original_scan = engine._scan_signals_on_date

def filtered_scan(trade_date, direction='long'):
    signals = original_scan(trade_date, direction)
    if not signals.empty:
        signals = signals[signals['stock_code'] == '036930']
    return signals

engine._scan_signals_on_date = filtered_scan

result = engine.run(
    start_date='2025-01-02',
    end_date='2025-03-31',
    verbose=True
)

trades = result['trades']
daily_values = result['daily_values']

print(f"\n{'='*80}")
print(f"📊 주성엔지니어링(036930) 2025년 1분기 백테스트 결과")
print(f"{'='*80}")
print(f"조건: 익절 +15%, 손절 -8%, 반대수급 60점, 보유기간 무제한")
print(f"Z-Score: 조건부 공식 적용 (부호 전환 시 과잉 반응 방지)")
print(f"{'='*80}\n")

if trades:
    print(f"[거래 내역] ({len(trades)}건)")
    print(f"{'진입일':<12} {'청산일':<12} {'진입가':>8} {'청산가':>8} {'수량':>4} {'수익률':>8} {'보유일':>4} {'청산사유':<16} {'패턴':<8} {'점수':>6}")
    print("-" * 100)
    for t in trades:
        print(f"{t.entry_date:<12} {t.exit_date:<12} {t.entry_price:>8,.0f} {t.exit_price:>8,.0f} {t.shares:>4} {t.return_pct:>+7.2f}% {t.hold_days:>4}일 {t.exit_reason:<16} {t.pattern:<8} {t.score:>6.1f}")

    print(f"\n[요약]")
    metrics = PerformanceMetrics(
        trades=trades,
        daily_values=daily_values,
        initial_capital=config.initial_capital
    )
    print(f"총 거래: {len(trades)}건")
    print(f"승률: {metrics.win_rate():.1f}%")
    print(f"총 수익률: {metrics.total_return():+.2f}%")
    print(f"평균 수익률: {metrics.avg_return():+.2f}%")
    print(f"평균 보유일: {sum(t.hold_days for t in trades)/len(trades):.1f}일")

    mdd = metrics.max_drawdown()
    print(f"MDD: {mdd['mdd']:.2f}%")
else:
    print("❌ 거래 없음")

# 미보유 상태에서 종료했는지 확인
if engine.portfolio.positions:
    print(f"\n[보유 중인 포지션]")
    for code, pos in engine.portfolio.positions.items():
        print(f"  {code}: 진입 {pos.entry_date} @ {pos.entry_price:,.0f}원, {pos.shares}주")

conn.close()
print(f"\n{'='*80}")
