# 수급 분석 시스템 구현 가이드

> **작성일**: 2026-02-12
> **상태**: Stage 1, 2, 3 완료
> **버전**: v2.0

---

## 목차

1. [시스템 개요](#시스템-개요)
2. [Stage 1: 데이터 정규화](#stage-1-데이터-정규화)
3. [Stage 2: 시공간 히트맵](#stage-2-시공간-히트맵)
4. [Stage 3: 패턴 분류 & 시그널 통합](#stage-3-패턴-분류--시그널-통합)
5. [데이터 플로우](#데이터-플로우)

---

## 시스템 개요

### 목표
**유통물량 기반 수급 분석**을 통해 시총 왜곡을 제거하고, 다차원 패턴 분류로 투자 의사결정 지원

### 3단계 아키텍처

```
[Stage 1] 데이터 정규화
    ↓
[Stage 2] 시공간 히트맵 (6개 기간 × 4가지 정렬)
    ↓
[Stage 3] 패턴 분류 (3개 바구니 자동 분류)
```

---

## Stage 1: 데이터 정규화

### 목표
단순 금액 대신 **유통시총 대비 비율(Sff)**과 **변동성 보정(Z-Score)**으로 "진짜 힘" 측정

### 핵심 공식

#### 1) Sff (Supply Float Factor)
```
Sff = (순매수 금액 / 유통시총) × 100
```

**효과**:
- 시총 5조 종목과 1000억 종목을 동일 선상에서 비교
- 유통물량이 적은 종목의 수급 왜곡 포착

#### 2) Z-Score
```
Z-Score = (현재값 - 60일 평균) / 60일 표준편차
```

**해석**:
- Z > 2.0: 이상 매수 (강한 매수세)
- -2.0 < Z < 2.0: 중립
- Z < -2.0: 이상 매도 (강한 매도세)

### 구현

**파일**: `src/analyzer/normalizer.py`

**핵심 클래스**: `SupplyNormalizer`

```python
from src.analyzer.normalizer import SupplyNormalizer
from src.database.connection import get_connection

conn = get_connection()
normalizer = SupplyNormalizer(conn)

# Sff 계산
df_sff = normalizer.calculate_sff(stock_codes=['005930'])

# 이상 수급 탐지
df_abnormal = normalizer.get_abnormal_supply(threshold=2.0, top_n=20)
print(df_abnormal[['stock_code', 'stock_name', 'sector', 'combined_zscore']])

conn.close()
```

### 출력 인터페이스

**메서드**: `normalizer.get_abnormal_supply(threshold, top_n)`

**반환**: `pd.DataFrame`

| 컬럼 | 타입 | 설명 |
|------|------|------|
| stock_code | str | 종목코드 (예: '005930') |
| stock_name | str | 종목명 (예: '삼성전자') |
| sector | str | 섹터 (예: '반도체 및 관련장비') |
| trade_date | str | 거래일 (예: '2026-01-20') |
| combined_sff | float | 외국인+기관 Sff (%) |
| foreign_sff | float | 외국인 Sff (%) |
| institution_sff | float | 기관 Sff (%) |
| combined_zscore | float | 외국인+기관 Z-Score |
| foreign_zscore | float | 외국인 Z-Score |
| institution_zscore | float | 기관 Z-Score |

**샘플 데이터**:
```
stock_code  stock_name  sector  combined_zscore  combined_sff
036460      한국가스공사  가스    4.299177         0.921185
014680      한솔케미칼   화학    3.844196         0.434423
```

### 성능
- 345종목 × 171,227 레코드: **~15초**
- 벡터화 최적화 완료

---

## Stage 2: 시공간 히트맵

### 목표
6개 기간(1W~2Y)의 수급 흐름을 한눈에 시각화하고, **4가지 정렬 모드**로 투자 스타일별 종목 필터링

### 핵심 개념

#### 1) 6개 기간 정의
```python
periods = {
    '1W': 5,      # 1주일 (5 영업일)
    '1M': 21,     # 1개월 (21 영업일)
    '3M': 63,     # 3개월
    '6M': 126,    # 6개월
    '1Y': 252,    # 1년
    '2Y': 504     # 2년
}
```

**Note**: 1D(1일)는 표준편차 계산 불가로 제외

#### 2) 4가지 정렬 모드

| 모드 | 공식 | 의미 | 용도 |
|------|------|------|------|
| **Recent** | (1W+1M)/2 | 현재 강도 | 지금 매수세 강한 종목 |
| **Momentum** | 1W-2Y | 수급 개선도 | 과거→현재 전환점 포착 |
| **Weighted** | 가중 평균<br>(3, 2.5, 2, 1.5, 1, 0.5) | 중장기 트렌드 | 일관된 매집 종목 |
| **Average** | 단순 평균 | 전체 일관성 | 장기 안정성 |

### 구현

**파일**:
- `src/visualizer/performance_optimizer.py` (벡터화 계산)
- `src/visualizer/heatmap_renderer.py` (렌더링)
- `scripts/analysis/heatmap_generator.py` (CLI)

**핵심 클래스**:
- `OptimizedMultiPeriodCalculator`: 6개 기간 Z-Score 계산
- `HeatmapRenderer`: 히트맵 시각화

```python
from src.config import DEFAULT_CONFIG
from src.database.connection import get_connection
from src.analyzer.normalizer import SupplyNormalizer
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
from src.visualizer.heatmap_renderer import HeatmapRenderer

conn = get_connection()
normalizer = SupplyNormalizer(conn)

# 6개 기간 Z-Score 계산 (벡터화 최적화)
optimizer = OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)
zscore_matrix = optimizer.calculate_multi_period_zscores(DEFAULT_CONFIG['periods'])

# 히트맵 렌더링 (4가지 정렬 모드 중 선택)
renderer = HeatmapRenderer(DEFAULT_CONFIG)
renderer.render_multi_period_heatmap(zscore_matrix, 'output/heatmap.png')

# CSV 저장
zscore_matrix.to_csv('output/heatmap.csv')

conn.close()
```

### 출력 인터페이스

**메서드**: `optimizer.calculate_multi_period_zscores(periods)`

**반환**: `pd.DataFrame`

| 컬럼 | 타입 | 설명 |
|------|------|------|
| stock_code | int | 종목코드 (예: 5930) |
| 1W | float | 1주일 Z-Score |
| 1M | float | 1개월 Z-Score |
| 3M | float | 3개월 Z-Score |
| 6M | float | 6개월 Z-Score |
| 1Y | float | 1년 Z-Score |
| 2Y | float | 2년 Z-Score |
| _sort_key | float | 정렬 키 (모드별 상이) |

**샘플 데이터 (Recent 모드)**:
```
stock_code    1W     1M     3M     6M     1Y     2Y  _sort_key
348210      1.17   0.84   0.13   0.55   0.95   1.17   1.004
131290      0.71   1.11   1.08   0.86   1.23   1.21   0.910
232140      1.73   0.08   0.00  -0.03   0.07   0.04   0.906
```

**4개 CSV 파일**:
```
output/heatmap_semi_recent.csv      # Recent 모드
output/heatmap_semi_momentum.csv    # Momentum 모드
output/heatmap_semi_weighted.csv    # Weighted 모드
output/heatmap_semi_average.csv     # Average 모드
```

### 4가지 정렬 모드 분석

#### Pattern 1: 모멘텀 돌파형 (Momentum 높음)
```
종목코드  recent  momentum  weighted  average
232140   0.906   1.696      0.519     0.315   ← 과거 약함, 최근 급등
101490  -0.764   1.541     -1.266    -1.525   ← 극강 전환 (매도→매수)
253590   0.601   1.226      0.374     0.248   ← 수급 개선 중
```

**특징**: 1W-2Y 값이 크다 = 과거엔 약했지만 최근 강해짐

**활용**: "전환점 포착", 단기 추격 매수 타겟

---

#### Pattern 2: 지속 매집형 (Weighted/Average 높음)
```
종목코드  recent  momentum  weighted  average
357780   0.880  -1.103      1.113     1.288   ← 장기 일관 매집
31980    0.529  -1.273      1.034     1.231   ← 전 기간 강세
131290   0.910  -0.500      0.972     1.036   ← 안정적 매집
```

**특징**: Weighted/Average 높음 + Momentum 낮음 = 장기간 일관된 매수세

**활용**: "저가 매수", 조정 후 재진입 타겟

---

#### Pattern 3: 최근 강도형 (Recent 높음)
```
종목코드  recent  momentum  weighted  average
348210   1.004   0.000      0.784     0.802   ← 최근 1W~1M 강함
131290   0.910  -0.500      0.972     1.036   ← 현재 매수세 강함
232140   0.906   1.696      0.519     0.315   ← 단기 급등
```

**특징**: (1W+1M)/2 높음 = 현재 매수세 강함

**활용**: "지금 강한 종목", 현재 수급 모니터링

---

### 성능 최적화 과정

#### Before: 순진한 구현 (120초)
```python
# 8개 기간마다 DB 쿼리 (8번)
for period in periods:
    df = pd.read_sql(f"SELECT ... WHERE lookback={period}", conn)
    # 종목별 루프 (O(n²))
    for stock in stock_codes:
        mean = df[df['stock_code'] == stock]['sff'].mean()
        std = df[df['stock_code'] == stock]['sff'].std()
```

**문제점**:
- DB 쿼리 8번 (중복 로드)
- 종목별 루프 (O(n²))

---

#### After: 최적화 (1.5초)

**최적화 1**: Sff 캐싱 (DB 쿼리 8회 → 1회)
```python
# 1번만 로드, 메모리 캐싱
self._sff_cache = normalizer._get_sff_data(stock_codes)

# 8개 기간 재사용
for period in periods:
    result = self._calculate_zscore_vectorized(period)
```

**최적화 2**: groupby.transform 벡터화 (O(n²) → O(n))
```python
# Before: 종목별 루프 (느림)
for stock in stock_codes:
    mean = df[df['stock_code'] == stock]['sff'].mean()

# After: 벡터화 (빠름)
df['rolling_mean'] = df.groupby('stock_code')['combined_sff'].transform(
    lambda x: x.rolling(window=lookback_days, min_periods=20).mean()
)
```

**결과**:
- 목표: 23초
- **실제: 1.5초** (93% 초과 달성!)

---

### CLI 사용법

```bash
# 기본 실행 (전체 6개 기간)
python scripts/analysis/heatmap_generator.py

# 정렬 모드 선택
python scripts/analysis/heatmap_generator.py --sort-by recent
python scripts/analysis/heatmap_generator.py --sort-by momentum

# 섹터 필터링
python scripts/analysis/heatmap_generator.py --sector "반도체 및 관련장비"

# 상위 50개만
python scripts/analysis/heatmap_generator.py --top 50

# CSV 동시 저장
python scripts/analysis/heatmap_generator.py --save-csv
```

---

## Stage 3: 패턴 분류 & 시그널 통합

### 목표
Stage 1~2 결과를 통합하여 **3개 바구니 자동 분류** + **시그널 탐지** + **통합 리포트 생성**

### 핵심 모듈

#### 1. PatternClassifier (패턴 분류)
**파일**: `src/analyzer/pattern_classifier.py`

**기능**:
- 4가지 정렬 키 계산 (Recent, Momentum, Weighted, Average)
- 추가 특성 추출 (변동성, 지속성, 가속도)
- 3개 바구니 자동 분류
- 패턴 강도 점수 (0~100)

**사용 예시**:
```python
from src.analyzer.pattern_classifier import PatternClassifier

classifier = PatternClassifier()
classified_df = classifier.classify_all(zscore_matrix)

# 결과: stock_code, pattern, score, recent, momentum, weighted, average
print(classified_df[['stock_code', 'pattern', 'score']].head())
```

**패턴 분류 규칙**:
1. **모멘텀형**: Momentum > 1.0 AND Recent > 0.5
   - 과거 약함 → 최근 급등 (단기 추격 매수)
2. **지속형**: Weighted > 0.8 AND Persistence > 0.7
   - 장기간 일관된 매집 (조정 후 재진입)
3. **전환형**: Weighted > 0.5 AND Momentum < 0
   - 장기 강함 + 최근 약화 (저가 매수)

---

#### 2. SignalDetector (시그널 탐지)
**파일**: `src/analyzer/signal_detector.py`

**기능**:
- MA 골든크로스 탐지 (외국인 5일MA > 20일MA)
- 수급 가속도 계산 (최근 5일 vs 직전 5일)
- 외인-기관 동조율 계산 (함께 매수한 비율)

**사용 예시**:
```python
from src.analyzer.signal_detector import SignalDetector
from src.database.connection import get_connection

conn = get_connection()
detector = SignalDetector(conn)
signals_df = detector.detect_all_signals()

# 결과: stock_code, ma_cross, acceleration, sync_rate, signal_count, signal_list
print(signals_df[['stock_code', 'signal_count', 'signal_list']].head())

conn.close()
```

**시그널 판단 기준**:
1. **MA 골든크로스**: 외국인 5일MA > 20일MA (크로스 발생일)
2. **수급 가속도**: 가속도 > 1.5배 (매수세 강화)
3. **동조율**: 동조율 > 70% (확신도 높음)

---

#### 3. IntegratedReport (통합 리포트)
**파일**: `src/analyzer/integrated_report.py`

**기능**:
- Stage 1~3 결과 통합
- 종목별 1줄 요약 카드 생성
- 진입/청산 포인트 제시
- CSV 저장 + 콘솔 출력

**사용 예시**:
```python
from src.analyzer.integrated_report import IntegratedReport

report_gen = IntegratedReport(conn)
report_df = report_gen.generate_report(classified_df, signals_df)

# 필터링
filtered = report_gen.filter_report(
    report_df,
    pattern='모멘텀형',
    min_score=70,
    min_signal_count=2,
    top_n=10
)

# 요약 카드 출력
report_gen.print_summary_card(filtered, top_n=10)

# CSV 저장
report_gen.export_to_csv(filtered, 'output/regime_report.csv')
```

**출력 형식**:
```
========================================
[1] 005930 삼성전자 (모멘텀형, 점수: 85)
========================================
섹터: 반도체 및 관련장비
정렬 키: Recent=0.91, Momentum=1.70, Weighted=0.52, Average=0.32
시그널: MA크로스, 가속도 1.8배 (2개)
진입: 현재가 진입 가능 (단기 추격 매수, 모멘텀 확인 후 진입)
손절: -5% 손절
```

---

### CLI 사용법

**RegimeScanner**: Stage 1~3 통합 실행

```bash
# 기본 실행 (전체 종목, 모든 패턴)
python scripts/analysis/regime_scanner.py

# 모멘텀형 종목만, 점수 70점 이상
python scripts/analysis/regime_scanner.py --pattern 모멘텀형 --min-score 70

# 지속형 + 시그널 2개 이상, 상위 10개
python scripts/analysis/regime_scanner.py --pattern 지속형 --min-signals 2 --top 10

# 섹터 필터링 (반도체)
python scripts/analysis/regime_scanner.py --sector "반도체 및 관련장비"

# CSV 저장 + 요약 카드 출력
python scripts/analysis/regime_scanner.py --save-csv output/report.csv --print-cards --top 10

# 관심 종목 리스트 출력 (점수 70+, 시그널 2+)
python scripts/analysis/regime_scanner.py --watchlist
```

---

### 성능

**전체 파이프라인 (Stage 1~3)**:
- 345종목 처리: 약 **3초**
- Stage 1 (정규화): ~15초 → 1.5초 (캐싱)
- Stage 2 (히트맵): ~1.5초
- Stage 3 (분류+시그널): ~1.5초

---

## Stage 3 준비사항 (Deprecated - 구현 완료)

### 입력 데이터

#### 옵션 1: CSV 파일 직접 로드 (권장)
```python
import pandas as pd

# 4개 CSV 로드
df_recent = pd.read_csv('output/heatmap_semi_recent.csv')
df_momentum = pd.read_csv('output/heatmap_semi_momentum.csv')
df_weighted = pd.read_csv('output/heatmap_semi_weighted.csv')
df_average = pd.read_csv('output/heatmap_semi_average.csv')

# 통합 DataFrame
data = {
    'recent': df_recent.set_index('stock_code')['_sort_key'],
    'momentum': df_momentum.set_index('stock_code')['_sort_key'],
    'weighted': df_weighted.set_index('stock_code')['_sort_key'],
    'average': df_average.set_index('stock_code')['_sort_key']
}
df_all = pd.DataFrame(data)
```

**장점**: 빠름 (재계산 불필요)
**단점**: 최신 데이터 반영 안 됨

---

#### 옵션 2: OptimizedMultiPeriodCalculator 재사용
```python
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
from src.analyzer.normalizer import SupplyNormalizer
from src.database.connection import get_connection

conn = get_connection()
normalizer = SupplyNormalizer(conn)
optimizer = OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)

zscore_matrix = optimizer.calculate_multi_period_zscores(DEFAULT_CONFIG['periods'])
```

**장점**: 최신 데이터 자동 반영
**단점**: 1.5초 추가 소요

---

### 추출 가능한 특성 (Features)

#### A. Stage 2 출력에서 직접 사용
1. **Recent**: (1W+1M)/2 - 현재 강도
2. **Momentum**: 1W-2Y - 수급 개선도
3. **Weighted**: 가중 평균 - 중장기 트렌드
4. **Average**: 단순 평균 - 전체 일관성

#### B. 추가 계산 필요
```python
import numpy as np

# 1. 변동성 (Volatility): 6개 기간 표준편차
df_all['volatility'] = df_all[['1W', '1M', '3M', '6M', '1Y', '2Y']].std(axis=1)

# 2. 지속성 (Persistence): 양수 기간 비율
df_all['persistence'] = (df_all[['1W', '1M', '3M', '6M', '1Y', '2Y']] > 0).sum(axis=1) / 6

# 3. 단기/장기 비율 (Short/Long Ratio): 최근 가속도
df_all['sl_ratio'] = (df_all['1W'] + df_all['1M']) / (df_all['1Y'] + df_all['2Y'] + 1e-6)
```

---

### 패턴 분류 규칙 (예시)

```python
def classify_pattern(row):
    """
    3개 바구니 자동 분류

    Returns:
        str: '지속형', '모멘텀형', '전환형'
    """
    # Pattern 1: 모멘텀 돌파형
    if row['momentum'] > 1.0 and row['recent'] > 0.5:
        return '모멘텀형'

    # Pattern 2: 지속 매집형
    if row['weighted'] > 0.8 and row['persistence'] > 0.7:
        return '지속형'

    # Pattern 3: 조정 반등형
    if row['weighted'] > 0.5 and row['momentum'] < 0:
        return '전환형'

    return '기타'

df_all['pattern'] = df_all.apply(classify_pattern, axis=1)
```

---

### 코드 의존성

#### 필수 import
```python
from src.config import DEFAULT_CONFIG
from src.database.connection import get_connection
from src.analyzer.normalizer import SupplyNormalizer
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
import pandas as pd
import numpy as np
```

#### 사용 가능한 메서드
```python
# Stage 1: 이상 수급 탐지
normalizer.calculate_sff(stock_codes)
normalizer.get_abnormal_supply(threshold, top_n)

# Stage 2: 다기간 Z-Score
optimizer.calculate_multi_period_zscores(periods)
```

---

## 데이터 플로우

### 전체 흐름

```
[DB] investor_flows 테이블 (171,227 레코드)
    ↓
[Stage 1] SupplyNormalizer
    ├── Sff 계산 (유통시총 대비 %)
    ├── Z-Score 계산 (변동성 보정)
    └── 출력: DataFrame (10개 컬럼)
        └── combined_zscore, foreign_zscore, institution_zscore
    ↓
[Stage 2] OptimizedMultiPeriodCalculator
    ├── 6개 기간 확장 (1W~2Y)
    ├── 벡터화 Z-Score 계산 (1.5초)
    ├── 4가지 정렬 키 생성 (Recent, Momentum, Weighted, Average)
    └── 출력: DataFrame (8개 컬럼) + 4개 CSV
        └── stock_code, 1W, 1M, 3M, 6M, 1Y, 2Y, _sort_key
    ↓
[Stage 3] PatternClassifier (예정)
    ├── 입력 1: Stage 2의 4개 CSV (4가지 정렬 키)
    ├── 입력 2: Stage 1의 이상 수급 결과 (최신 Z-Score)
    ├── 처리:
    │   ├── 4가지 정렬 키 통합
    │   ├── 추가 특성 계산 (변동성, 지속성, 가속도)
    │   ├── 패턴 분류 규칙 적용
    │   └── MA 골든크로스, 동조율 추가
    └── 출력: DataFrame
        ├── pattern: '지속형', '모멘텀형', '전환형'
        ├── score: 0~100 (패턴 강도)
        └── signals: ['MA크로스', '가속도', '동조율'] (리스트)
```

---

### Stage별 I/O 스펙

| Stage | 입력 | 출력 | 소요시간 |
|-------|------|------|----------|
| **1** | investor_flows (DB) | DataFrame (10 cols) | ~15초 |
| **2** | Stage 1 로직 재사용 | DataFrame (8 cols) + 4 CSV | ~1.5초 |
| **3** | Stage 2 Z-Score 매트릭스 | DataFrame (pattern, score, signals) | ~1.5초 |
| **전체** | DB → 통합 리포트 | CSV + 콘솔 출력 | **~3초** |

---

## 실전 활용 예시

### 케이스 1: 모멘텀 돌파형 발견

**종목**: 232140 (와이씨)

**4가지 정렬 키**:
```
recent: 0.906 (3위)   ← 현재 강도 높음
momentum: 1.696 (1위) ← 수급 개선도 최고!
weighted: 0.519 (중간) ← 장기 트렌드 보통
average: 0.315 (중간)  ← 전체 일관성 보통
```

**6개 기간 Z-Score**:
```
1W: 1.73  ← 최근 급등
1M: 0.08
3M: 0.00
6M: -0.03
1Y: 0.07
2Y: 0.04  ← 과거 약함
```

**해석**: "과거 2년간 약했지만 최근 1주일 급등" = **전환점 포착!**

**전략**: 단기 추격 매수, 손절 설정 필수

---

### 케이스 2: 지속 매집형 발견

**종목**: 357780 (솔브레인)

**4가지 정렬 키**:
```
recent: 0.880 (6위)    ← 현재 강도 중간
momentum: -1.103 (하위) ← 수급 개선도 낮음 (최근 약화)
weighted: 1.113 (1위)  ← 장기 트렌드 최고!
average: 1.288 (1위)   ← 전체 일관성 최고!
```

**6개 기간 Z-Score**:
```
1W: 0.68
1M: 1.08
3M: 1.22
6M: 1.32
1Y: 1.65
2Y: 1.78  ← 꾸준한 상승
```

**해석**: "2년간 일관된 매집, 최근 약간 약화" = **조정 후 재진입 타이밍!**

**전략**: 조정 구간 저가 매수, 장기 보유

---

## 다음 단계 (Stage 3)

### 구현 파일
1. `src/analyzer/pattern_classifier.py` - 패턴 분류 로직
2. `src/analyzer/integrated_report.py` - Stage 1~3 통합 리포트
3. `scripts/analysis/regime_scanner.py` - CLI 도구

### 추가 기능
1. **MA 골든크로스**: 외국인 5일 MA > 20일 MA 탐지
2. **수급 가속도**: 최근 5일 vs 직전 5일 비교
3. **외인-기관 동조율**: 함께 매수한 비율 점수화

### 최종 출력
```python
# 종목별 1줄 요약 카드
{
    'stock_code': '232140',
    'stock_name': '와이씨',
    'pattern': '모멘텀형',
    'score': 85,
    'signals': ['MA크로스', '가속도 1.8배', '동조율 72%'],
    'entry_point': '현재가 진입 가능',
    'stop_loss': '-5% 손절'
}
```

---

**문서 버전**: v2.0
**최종 업데이트**: 2026-02-12
**작성자**: Claude Code + User

---

## 다음 단계 (Stage 4+)

Stage 3 완료로 **핵심 기능 구현 완료**. 선택적 고도화:
1. **백테스팅 엔진**: 과거 데이터로 전략 검증
2. **알림 시스템**: 새로운 시그널 발생 시 알림
3. **웹 대시보드**: 실시간 모니터링 UI
4. **머신러닝**: 패턴 자동 학습 및 최적화
