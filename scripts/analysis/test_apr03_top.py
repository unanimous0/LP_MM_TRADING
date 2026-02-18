"""2025-04-03 기준 매수종합점수 80점 이상 종목 백테스트"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_connection
from src.analyzer.normalizer import SupplyNormalizer
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
from src.analyzer.pattern_classifier import PatternClassifier
from src.analyzer.signal_detector import SignalDetector
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.backtesting.metrics import PerformanceMetrics
import pandas as pd

conn = get_connection()
periods = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252, '2Y': 504}

# 1. 2025-04-03이 영업일인지 확인
query = """
SELECT DISTINCT trade_date FROM investor_flows
WHERE trade_date >= '2025-04-03'
ORDER BY trade_date LIMIT 1
"""
scan_date = pd.read_sql(query, conn)['trade_date'].iloc[0]
print(f"스캔 기준일: {scan_date}")

# 다음 영업일 (진입일)
query2 = """
SELECT DISTINCT trade_date FROM investor_flows
WHERE trade_date > ?
ORDER BY trade_date LIMIT 1
"""
entry_date = pd.read_sql(query2, conn, params=[scan_date])['trade_date'].iloc[0]
print(f"진입일: {entry_date}")

# DB 마지막 날짜
query3 = "SELECT MAX(trade_date) as last_date FROM investor_flows"
last_date = pd.read_sql(query3, conn)['last_date'].iloc[0]
print(f"DB 마지막 날짜: {last_date}")

# 2. Stage 1-3 실행: 매수 종합점수 80점 이상 종목 추출
print(f"\n{'='*90}")
print(f"📊 {scan_date} 기준 Long 방향 스캔")
print(f"{'='*90}")

normalizer = SupplyNormalizer(conn)
calculator = OptimizedMultiPeriodCalculator(normalizer, enable_caching=False)

zscore_matrix = calculator.calculate_multi_period_zscores(
    periods_dict=periods,
    end_date=scan_date
)

if zscore_matrix.empty:
    print("데이터 없음")
    conn.close()
    sys.exit()

zscore_matrix = zscore_matrix.reset_index()

# Long 필터: 1W > 0
long_candidates = zscore_matrix[zscore_matrix['1W'] > 0].copy()
print(f"1W > 0 필터 통과: {len(long_candidates)}종목")

# 패턴 분류
classifier = PatternClassifier()
pattern_result = classifier.classify_all(long_candidates, direction='long')

# 시그널 탐지
detector = SignalDetector(conn)
if not pattern_result.empty:
    signal_result = detector.detect_all_signals(
        stock_codes=pattern_result['stock_code'].tolist(),
        end_date=scan_date
    )
    result = pd.merge(pattern_result, signal_result, on='stock_code', how='left')
    result['signal_count'] = result['signal_count'].fillna(0).astype(int)
    result['final_score'] = result['score'] + (result['signal_count'] * 5)

    # 종목명 조회
    names = {}
    for code in result['stock_code']:
        q = "SELECT stock_name FROM stocks WHERE stock_code = ?"
        r = pd.read_sql(q, conn, params=[code])
        names[code] = r['stock_name'].iloc[0] if not r.empty else code
    result['stock_name'] = result['stock_code'].map(names)

    # 80점 이상 필터
    top = result[result['final_score'] >= 80].sort_values('final_score', ascending=False)

    print(f"패턴 분류 완료: {len(result)}종목")
    print(f"종합점수 80점 이상: {len(top)}종목\n")

    if top.empty:
        print("80점 이상 종목 없음")
        conn.close()
        sys.exit()

    print(f"{'종목코드':<8} {'종목명':<14} {'패턴':<8} {'패턴점수':>8} {'시그널':>4} {'종합점수':>8}")
    print("-" * 60)
    for _, r in top.iterrows():
        print(f"{r['stock_code']:<8} {r['stock_name']:<14} {r['pattern']:<8} {r['score']:>8.1f} {r['signal_count']:>4} {r['final_score']:>8.1f}")

    target_stocks = top['stock_code'].tolist()
else:
    print("패턴 분류 결과 없음")
    conn.close()
    sys.exit()

# 3. 해당 종목들로 백테스트 실행
print(f"\n{'='*90}")
print(f"📈 백테스트 실행 ({entry_date} ~ {last_date})")
print(f"조건: 익절 +15%, 손절 -8%, 반대수급 60점, 보유기간 무제한")
print(f"{'='*90}")

config = BacktestConfig(
    initial_capital=10_000_000,
    max_positions=len(target_stocks),
    min_score=80,
    min_signals=0,
    target_return=0.15,
    stop_loss=-0.08,
    max_hold_days=999,
    reverse_signal_threshold=60,
    strategy='long',
    force_exit_on_end=False
)

engine = BacktestEngine(conn, config)

# 선정된 종목만 필터링
original_scan = engine._scan_signals_on_date
def filtered_scan(trade_date, direction='long'):
    signals = original_scan(trade_date, direction)
    if not signals.empty:
        signals = signals[signals['stock_code'].isin(target_stocks)]
    return signals
engine._scan_signals_on_date = filtered_scan

result = engine.run(
    start_date=scan_date,
    end_date=last_date,
    verbose=False
)

trades = result['trades']
daily_values = result['daily_values']

# 4. 결과 출력
print(f"\n[거래 내역] ({len(trades)}건)")
if trades:
    print(f"{'종목명':<14} {'진입일':<12} {'청산일':<12} {'진입가':>8} {'청산가':>8} {'수익률':>8} {'보유일':>4} {'청산사유':<16} {'점수':>6}")
    print("-" * 110)
    for t in trades:
        print(f"{names.get(t.stock_code, t.stock_code):<14} {t.entry_date:<12} {t.exit_date:<12} {t.entry_price:>8,.0f} {t.exit_price:>8,.0f} {t.return_pct:>+7.2f}% {t.hold_days:>4}일 {t.exit_reason:<16} {t.score:>6.1f}")

    print(f"\n[요약]")
    metrics = PerformanceMetrics(trades=trades, daily_values=daily_values, initial_capital=config.initial_capital)
    print(f"총 거래: {len(trades)}건")
    wins = [t for t in trades if t.return_pct > 0]
    losses = [t for t in trades if t.return_pct <= 0]
    print(f"승리: {len(wins)}건, 패배: {len(losses)}건 (승률 {metrics.win_rate():.1f}%)")
    print(f"총 수익률: {metrics.total_return():+.2f}%")
    print(f"평균 수익률: {metrics.avg_return():+.2f}%")
    if wins:
        print(f"평균 승리: {metrics.avg_win():+.2f}%")
    if losses:
        print(f"평균 손실: {metrics.avg_loss():+.2f}%")

# 보유 중 포지션
if engine.portfolio.positions:
    print(f"\n[보유 중 포지션] ({last_date} 기준)")
    for code, pos in engine.portfolio.positions.items():
        current_price_q = f"SELECT close_price FROM investor_flows WHERE stock_code='{code}' AND trade_date='{last_date}'"
        cp = pd.read_sql(current_price_q, conn)
        current_price = cp['close_price'].iloc[0] if not cp.empty else 0
        unrealized = ((current_price - pos.entry_price) / pos.entry_price) * 100
        print(f"  {names.get(code, code)}: 진입 {pos.entry_date} @ {pos.entry_price:,.0f}원 → 현재 {current_price:,.0f}원 ({unrealized:+.2f}%)")

conn.close()
print(f"\n{'='*90}")
