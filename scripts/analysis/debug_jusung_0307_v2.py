"""주성엔지니어링 3/7 reverse_signal 청산 원인 분석 (v2: 캐시 초기화)"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_connection
from src.analyzer.normalizer import SupplyNormalizer
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
from src.analyzer.pattern_classifier import PatternClassifier
from src.analyzer.signal_detector import SignalDetector
import pandas as pd

conn = get_connection()
periods = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252, '2Y': 504}

# 2/19~3/7 각 날짜에 대해 Short 방향 스캔
print("=" * 100)
print("주성엔지니어링(036930) Short 방향 일별 스캔 (2025-02-20 ~ 2025-03-07)")
print("조건부 Z-Score 적용 후")
print("=" * 100)

print(f"\n{'날짜':<12} {'1W_Z':>8} {'1M_Z':>8} {'Short?':>6} {'패턴':<8} {'점수':>6} {'시그널':>4} {'종합':>6} {'청산?'}")
print("-" * 80)

scan_dates = []
query = """
SELECT DISTINCT trade_date FROM investor_flows
WHERE trade_date BETWEEN '2025-02-20' AND '2025-03-07'
ORDER BY trade_date
"""
scan_dates = pd.read_sql(query, conn)['trade_date'].tolist()

for check_date in scan_dates:
    # 매번 새로운 calculator 생성 (캐시 초기화)
    normalizer = SupplyNormalizer(conn)
    calculator = OptimizedMultiPeriodCalculator(normalizer, enable_caching=False)

    zscore_matrix = calculator.calculate_multi_period_zscores(
        periods_dict=periods,
        stock_codes=['036930'],
        end_date=check_date
    )

    if zscore_matrix.empty:
        print(f"{check_date:<12} {'N/A':>8}")
        continue

    zscore_matrix = zscore_matrix.reset_index()
    row = zscore_matrix.iloc[0]

    z1w = row['1W']
    z1m = row['1M']

    if z1w >= 0:
        print(f"{check_date:<12} {z1w:>+8.3f} {z1m:>+8.3f} {'NO':>6}")
        continue

    # Short 필터 통과 → 패턴 분류
    classifier = PatternClassifier()
    pattern_result = classifier.classify_all(zscore_matrix, direction='short')

    if pattern_result.empty:
        print(f"{check_date:<12} {z1w:>+8.3f} {z1m:>+8.3f} {'YES':>6} {'N/A':<8}")
        continue

    p = pattern_result.iloc[0]

    # 시그널 탐지
    detector = SignalDetector(conn)
    signal_result = detector.detect_all_signals(
        stock_codes=['036930'],
        end_date=check_date
    )

    signal_count = 0
    if not signal_result.empty:
        signal_count = signal_result.iloc[0]['signal_count']

    final_score = p['score'] + (signal_count * 5)
    triggered = "⚠️ YES" if final_score >= 60 else ""

    print(f"{check_date:<12} {z1w:>+8.3f} {z1m:>+8.3f} {'YES':>6} {p['pattern']:<8} {p['score']:>6.1f} {signal_count:>4} {final_score:>6.1f} {triggered}")

conn.close()
print(f"\n{'=' * 100}")
