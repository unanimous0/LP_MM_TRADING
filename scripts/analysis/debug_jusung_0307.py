"""주성엔지니어링 3/7 reverse_signal 청산 원인 분석"""

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
import numpy as np

conn = get_connection()

# 1. 주성엔지니어링 일별 Sff 확인 (2/17 ~ 3/10)
print("=" * 80)
print("1. 주성엔지니어링(036930) 일별 수급 데이터 (2025-02-10 ~ 2025-03-10)")
print("=" * 80)

query = """
SELECT trade_date, foreign_net_amount, institution_net_amount,
       close_price, free_float_shares
FROM investor_flows
WHERE stock_code = '036930'
AND trade_date BETWEEN '2025-02-10' AND '2025-03-10'
ORDER BY trade_date
"""
df = pd.read_sql(query, conn)
df['net_amount'] = df['foreign_net_amount'] + df['institution_net_amount']
df['free_float_mcap'] = df['free_float_shares'] * df['close_price']
df['sff'] = (df['net_amount'] / df['free_float_mcap']) * 100

print(f"{'날짜':<12} {'외국인(억)':>10} {'기관(억)':>10} {'합계(억)':>10} {'종가':>8} {'Sff':>8}")
print("-" * 70)
for _, row in df.iterrows():
    foreign = row['foreign_net_amount'] / 1e8
    inst = row['institution_net_amount'] / 1e8
    net = row['net_amount'] / 1e8
    print(f"{row['trade_date']:<12} {foreign:>+10.1f} {inst:>+10.1f} {net:>+10.1f} {row['close_price']:>8,.0f} {row['sff']:>+8.3f}")

# 2. 3/6 기준 Z-Score 확인 (reverse signal은 3/6 스캔 → 3/7 청산)
print(f"\n{'=' * 80}")
print("2. 3/6 기준 Short 방향 Z-Score 분석")
print("=" * 80)

normalizer = SupplyNormalizer(conn)
calculator = OptimizedMultiPeriodCalculator(normalizer)

# 3/5, 3/6, 3/7 각각 확인
for check_date in ['2025-03-05', '2025-03-06', '2025-03-07']:
    print(f"\n--- {check_date} ---")

    periods = {'1W': 5, '1M': 21, '3M': 63, '6M': 126, '1Y': 252, '2Y': 504}
    zscore_matrix = calculator.calculate_multi_period_zscores(
        periods_dict=periods,
        stock_codes=['036930'],
        end_date=check_date
    )

    if zscore_matrix.empty:
        print(f"  데이터 없음")
        continue

    zscore_matrix = zscore_matrix.reset_index()
    row = zscore_matrix.iloc[0]

    print(f"  Z-Score: 1W={row['1W']:+.3f}, 1M={row['1M']:+.3f}, 3M={row['3M']:+.3f}")
    print(f"           6M={row.get('6M', 0):+.3f}, 1Y={row.get('1Y', 0):+.3f}, 2Y={row.get('2Y', 0):+.3f}")

    # Short 필터: 1W < 0?
    if row['1W'] < 0:
        print(f"  → 1W < 0 → Short 필터 통과")

        # 패턴 분류 (Short 방향)
        classifier = PatternClassifier()
        pattern_result = classifier.classify_all(zscore_matrix, direction='short')

        if not pattern_result.empty:
            p = pattern_result.iloc[0]
            print(f"  Short 부호 반전 후: 1W={p['1W']:+.3f}, 1M={p['1M']:+.3f}")
            print(f"  정렬키: recent={p['recent']:+.3f}, momentum={p['momentum']:+.3f}, weighted={p['weighted']:+.3f}")
            print(f"  패턴: {p['pattern']}, 점수: {p['score']:.1f}")

            # 시그널 탐지
            detector = SignalDetector(conn)
            signal_result = detector.detect_all_signals(
                stock_codes=['036930'],
                end_date=check_date
            )

            if not signal_result.empty:
                s = signal_result.iloc[0]
                signal_count = s['signal_count']
                final_score = p['score'] + (signal_count * 5)
                print(f"  시그널: {signal_count}개, 종합점수: {final_score:.1f}")

                if final_score >= 60:
                    print(f"  ⚠️  종합점수 {final_score:.1f} >= 60 → 반대수급 청산 트리거!")
                else:
                    print(f"  ✅ 종합점수 {final_score:.1f} < 60 → 청산 안됨")
    else:
        print(f"  → 1W > 0 → Short 필터 미통과 (청산 안됨)")

conn.close()
