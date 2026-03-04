# 수급 분석 시스템 구현 가이드

> **작성일**: 2026-02-12
> **마지막 업데이트**: 2026-03-05
> **상태**: Stage 1~3 완료, Stage 4(백테스트) 완료, Stage 5-1(Streamlit) 완료
> **버전**: v3.0

---

## 목차

1. [시스템 개요](#시스템-개요)
2. [Stage 1: 데이터 정규화](#stage-1-데이터-정규화)
3. [Stage 2: 시공간 히트맵](#stage-2-시공간-히트맵)
4. [Stage 3: 패턴 분류 & 시그널 통합](#stage-3-패턴-분류--시그널-통합)
5. [데이터 플로우](#데이터-플로우)
6. [성능 최적화](#성능-최적화)

---

## 시스템 개요

### 목표
**유통물량 기반 수급 분석**을 통해 시총 왜곡을 제거하고, 다차원 패턴 분류로 투자 의사결정 지원

### 3단계 아키텍처

```
[Stage 1] 데이터 정규화 (Sff + 조건부 Z-Score)
    ↓
[Stage 2] 시공간 히트맵 (7개 기간 × 4가지 정렬)
    ↓
[Stage 3] 패턴 분류 (3개 바구니 + sub_type 7종 + 시그널 3종)
```

### 데이터 소스
- **PostgreSQL** (`korea_stock_data`): 시장 데이터 (~10M 레코드, 2,721 종목)
- **Materialized View** (`mv_daily_sff`): Sff 사전 계산, SQL 함수로 Z-Score/시그널 고속 계산

---

## Stage 1: 데이터 정규화

### 구현

**파일**: `src/analyzer/normalizer.py`
**핵심 클래스**: `SupplyNormalizer`

```python
from src.analyzer.normalizer import SupplyNormalizer
from src.database.connection import get_pg_engine

engine = get_pg_engine()
normalizer = SupplyNormalizer(engine)

# Sff 계산
df_sff = normalizer.calculate_sff(stock_codes=['005930'])

# Z-Score 계산 (벡터화)
df_zscore = normalizer.calculate_zscore(window=60)

# 이상 수급 탐지
df_abnormal = normalizer.get_abnormal_supply(threshold=2.0, top_n=20)
```

> **이론 설명**: ANALYSIS_GUIDE.md의 "3. 지표 및 수식 정의" 참조

### 출력 인터페이스

**메서드**: `normalizer.get_abnormal_supply(threshold, top_n)`

| 컬럼 | 타입 | 설명 |
|------|------|------|
| stock_code | str | 종목코드 (예: '005930') |
| stock_name | str | 종목명 |
| sector | str | 섹터 |
| trade_date | str | 거래일 |
| combined_sff | float | 외국인+기관 Sff (%) |
| combined_zscore | float | 외국인+기관 Z-Score |
| foreign_zscore | float | 외국인 Z-Score |
| institution_zscore | float | 기관 Z-Score |

### MV 자동 분기

```python
from src.database.connection import is_mv_available

# MV 있으면 고속 경로, 없으면 3-table JOIN fallback
if is_mv_available():
    # mv_daily_sff 사용 (Sff 사전 계산됨)
else:
    # investor_trading + stock_prices + stock_master JOIN
```

---

## Stage 2: 시공간 히트맵

### 구현

**파일**:
- `src/visualizer/performance_optimizer.py` (벡터화 계산 + SQL 분기)
- `src/visualizer/heatmap_renderer.py` (렌더링)

**핵심 클래스**: `OptimizedMultiPeriodCalculator`

```python
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
from src.analyzer.normalizer import SupplyNormalizer
from src.database.connection import get_pg_engine

engine = get_pg_engine()
normalizer = SupplyNormalizer(engine)

# 7개 기간 Z-Score 계산
periods = {'5D': 5, '10D': 10, '20D': 20, '50D': 50, '100D': 100, '200D': 200, '500D': 500}
optimizer = OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)
zscore_matrix = optimizer.calculate_multi_period_zscores(periods)
```

### SQL 고속 경로

MV 사용 가능 + 전체 종목 조회 시 SQL 함수로 자동 전환:
```sql
-- fn_zscore_latest(weight, target_date): 7개 기간 Z-Score 일괄 계산 (0.38초)
-- fn_signals_latest(weight, target_date): MA크로스+가속도+동조율 (0.04초)
```

### 출력 인터페이스

| 컬럼 | 타입 | 설명 |
|------|------|------|
| stock_code | str | 종목코드 |
| 5D ~ 500D | float | 7개 기간 Z-Score |
| _sort_key | float | 정렬 키 (모드별 상이) |
| _today_sff | float | 방향 확신도 계산용 메타데이터 |

---

## Stage 3: 패턴 분류 & 시그널 통합

### 1. PatternClassifier (패턴 분류)
**파일**: `src/analyzer/pattern_classifier.py`

```python
from src.analyzer.pattern_classifier import PatternClassifier

classifier = PatternClassifier()
classified_df = classifier.classify_all(zscore_matrix, direction='long')

# 출력: stock_code, pattern, sub_type, pattern_label, score, final_score, ...
```

**출력 컬럼**: `pattern`, `sub_type`, `pattern_label`, `score`, `tc`, `short_divergence`, `mid_divergence`

### 2. SignalDetector (시그널 탐지)
**파일**: `src/analyzer/signal_detector.py`

```python
from src.analyzer.signal_detector import SignalDetector

detector = SignalDetector(engine)
signals_df = detector.detect_all_signals()

# 출력: stock_code, ma_cross, acceleration, sync_rate, signal_count, signal_list
```

MV 사용 가능 시 SQL 함수(`fn_signals_latest`)로 자동 전환.

### 3. IntegratedReport (통합 리포트)
**파일**: `src/analyzer/integrated_report.py`

```python
from src.analyzer.integrated_report import IntegratedReport

report_gen = IntegratedReport(engine)
report_df = report_gen.generate_report(classified_df, signals_df)

# 필터링
filtered = report_gen.filter_report(report_df, pattern='급등형', min_score=70, top_n=10)
report_gen.print_summary_card(filtered, top_n=10)
```

### CLI 사용법

```bash
# 기본 실행
python scripts/analysis/regime_scanner.py

# 필터링 + 저장
python scripts/analysis/regime_scanner.py --pattern 급등형 --min-score 70 --save-csv output/report.csv --print-cards
```

---

## 데이터 플로우

```
[PostgreSQL] investor_trading + stock_prices + stock_master (~10M 레코드)
    ↓ (MV 사용 시: mv_daily_sff → Sff 사전 계산됨)
[Stage 1] SupplyNormalizer
    ├── Sff 계산 (유통시총 대비 %)
    ├── 조건부 Z-Score (부호 전환 시 과잉 반응 방지)
    └── 출력: DataFrame (combined_zscore, foreign_zscore, institution_zscore)
    ↓
[Stage 2] OptimizedMultiPeriodCalculator
    ├── 7개 기간 Z-Score (5D~500D)
    ├── 4가지 정렬 키 (Recent, Momentum, Weighted, Average)
    ├── 방향 확신도 (tanh 기반)
    └── SQL 고속 경로 (fn_zscore_latest: 1.2초)
    ↓
[Stage 3] PatternClassifier + SignalDetector + IntegratedReport
    ├── 3개 바구니 분류 (급등형/지속형/전환형)
    ├── sub_type 7종 (장기기반/단기돌파/V자반등/전면수급/수급약화/감속/단기반등)
    ├── tc(시간 일관성) + divergence(이격도) 보정
    ├── 시그널 3종 (MA크로스/가속도/동조율)
    └── 출력: final_score = score + signal_count × 5
```

### Stage별 성능

| Stage | 처리 | SQL 경로 | Python 경로 |
|-------|------|----------|-------------|
| **1+2** | 2,721종목 Z-Score | **~1.2초** | ~17초 |
| **3a** | 패턴 분류 | ~0.1초 | ~0.1초 |
| **3b** | 시그널 탐지 | **~0.3초** | ~4.6초 |
| **전체** | Stage 1~3 | **~2.2초** | ~22초 |
| **캐시 히트** | 파일 캐시 로드 | **<0.01초** | — |

---

## 성능 최적화

### 최적화 이력

1. **MV (Materialized View)**: 3-table JOIN → 사전 계산 뷰 (I/O 2배 절감)
2. **groupby.transform 벡터화**: per-stock 루프 → 벡터 연산 (33~75배)
3. **SQL 함수**: Python Z-Score/시그널 → PostgreSQL LATERAL+FILTER (14~15배)
4. **파일 캐시**: 같은 날 재방문 시 DB 쿼리 건너뜀 (3000배+)
5. **lazy import**: 비백테스트 페이지 임포트 비용 ~450ms 절감

### 백테스트 성능

- **BacktestPrecomputer**: 벡터화 사전 계산으로 165~262배 향상
  - 38일: 177초 → 1.1초
  - 63일: 393초 → 1.5초
  - 1년: ~4초

---

**문서 버전**: v3.0
**최종 업데이트**: 2026-03-05
**작성자**: Claude + unanimous0
