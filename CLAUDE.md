# 한국 주식 외국인/기관 투자자 수급 분석 프로그램

## [Status]
- **현재 작업**: Stage 3 완료! ✅ (105개 테스트 100% 통과)
- **마지막 업데이트**: 2026-02-13
- **다음 시작점**: Stage 4 - 백테스팅 시스템 (선택)
- **현재 브랜치**: main

### 주요 성과

**Stage 1: 데이터 정규화** ✅
- Sff/Z-Score 정규화 완료
- 이상 수급 탐지 (20건 탐지, 15초 소요)
- 섹터 정보 통합 (97.9% 커버리지)

**Stage 2: 시공간 히트맵** ✅
- 6개 기간(1W~2Y) 히트맵 시각화
- 4가지 정렬 모드 (Recent/Momentum/Weighted/Average)
- 성능 최적화 (목표 23초 → 실제 1.5초, 93% 초과 달성)
- 61개 테스트 (100% 통과)

**Stage 3: 패턴 기반 분류 시스템** ✅
- 3개 바구니 자동 분류 (모멘텀형/지속형/전환형)
- 추가 시그널 탐지 (MA 골든크로스, 가속도, 동조율)
- 통합 리포트 생성 (종목별 1줄 요약 카드)
- 진입/청산 포인트 제시
- 105개 테스트 (Stage 1~3 통합, 100% 통과)

**핵심 인사이트**:
- 3개 패턴은 투자 스타일별 최적 종목 필터링 (단기=돌파형, 중기=매집형, 저가=반등형)
- 시그널 2개 이상 종목 = 확신도 높은 진입 타이밍
- 전체 파이프라인 실행 시간: 약 3초 (Stage 1~3 통합)

---

## [Quick Start]

### 환경 설정
```bash
git clone <repository>
cd LP_MM_TRADING
pip install -r requirements.txt
```

### 데이터베이스 초기화
```bash
# 1. DB 생성
python -c "from src.database.schema import create_database; create_database()"

# 2. 데이터 로드
python scripts/loaders/load_initial_data.py

# 3. 주가/유통주식 크롤링
python scripts/crawlers/crawl_all_data.py --start 2024-01-01
```

### 분석 실행

**상세 사용법**: `README.md` 및 `ANALYSIS_GUIDE.md` 참조

---

## [Next Steps]

### Stage 4: 백테스팅 시스템 (선택)

**목표**: 과거 데이터로 패턴 분류 전략의 수익률 검증 및 최적화

**구현 로드맵**:

#### 4-1. 백테스트 엔진 구축
**파일**: `src/backtesting/engine.py`

**핵심 기능**:
- 시계열 시뮬레이션 (롤링 윈도우)
  - 각 거래일마다 과거 N일 데이터로 패턴 분류 실행
  - 진입 조건: 패턴 점수 X점 이상, 시그널 Y개 이상
  - 청산 조건: 목표 수익률(+15%) 또는 손절(-7%) 달성, 또는 N일 보유
- 멀티 포지션 관리
  - 동시 보유 종목 수 제한 (예: 최대 10개)
  - 종목당 투자 비중 (동일 가중 or 점수 비례)
- 거래 비용 반영
  - 거래세 0.23% (매도 시)
  - 수수료 0.015% (매수/매도)
  - 슬리피지 0.1% (체결 가격 불리)

**데이터 요구사항**:
- 이미 구축된 DB 활용 (2024-01-02 ~ 2026-01-20)
- 학습 기간: 2024-01-02 ~ 2025-06-30 (18개월)
- 검증 기간: 2025-07-01 ~ 2026-01-20 (7개월)

**예상 출력**:
```python
{
    'total_trades': 152,
    'win_rate': 0.58,
    'avg_return': 0.089,  # 평균 8.9% 수익
    'total_return': 0.342,  # 누적 34.2% 수익
    'max_drawdown': -0.18,  # 최대 낙폭 -18%
    'sharpe_ratio': 1.24
}
```

#### 4-2. 성과 분석 모듈
**파일**: `src/backtesting/metrics.py`

**주요 메트릭**:
1. **패턴별 성과**
   - 모멘텀형: 승률 62%, 평균 수익 +12.3%, 평균 보유일 8일
   - 지속형: 승률 55%, 평균 수익 +18.7%, 평균 보유일 21일
   - 전환형: 승률 48%, 평균 수익 +9.1%, 평균 보유일 14일

2. **시그널 개수별 성과**
   - 시그널 0개: 승률 45%, 평균 수익 +5.2%
   - 시그널 1개: 승률 52%, 평균 수익 +8.9%
   - 시그널 2개: 승률 61%, 평균 수익 +13.4%
   - 시그널 3개: 승률 72%, 평균 수익 +18.7%

3. **리스크 지표**
   - MDD (Maximum Drawdown): 최대 낙폭
   - 샤프 비율: 위험 대비 수익률
   - 승률 × 평균 수익률: 기대값 계산
   - 최악의 연속 손실 횟수

4. **시각화**
   - 누적 수익률 곡선 (vs 코스피 벤치마크)
   - 월별 수익률 분포 (히트맵)
   - 보유 기간별 수익률 산점도

#### 4-3. 최적화 & 파라미터 튜닝
**파일**: `src/backtesting/optimizer.py`

**최적화 대상**:
- 패턴 점수 임계값 (50~90점)
- 시그널 개수 임계값 (0~3개)
- 목표 수익률 (10~25%)
- 손절 비율 (-5% ~ -10%)
- 최대 보유 기간 (5~30일)

**방법론**:
- Grid Search: 주요 파라미터 조합 전수 탐색
- Walk-Forward Analysis: 학습/검증 기간 롤링 재검증
- Out-of-Sample 검증: 최종 파라미터로 미래 기간 테스트

#### 4-4. CLI 도구
**파일**: `scripts/analysis/backtest_runner.py`

**사용 예시**:
```bash
# 기본 백테스트 (전체 패턴, 전체 기간)
python scripts/analysis/backtest_runner.py

# 특정 패턴만 테스트
python scripts/analysis/backtest_runner.py --pattern 모멘텀형

# 파라미터 최적화 모드
python scripts/analysis/backtest_runner.py --optimize --iterations 100

# 결과 시각화 + CSV 저장
python scripts/analysis/backtest_runner.py --plot --save-results output/backtest_results.csv

# Walk-Forward 검증
python scripts/analysis/backtest_runner.py --walk-forward --window 90 --step 30
```

**출력 예시**:
```
================================================================================
📈 백테스팅 결과 (2024-01-02 ~ 2026-01-20)
================================================================================

[전체 성과]
총 거래 횟수: 152
승률: 58.6%
평균 수익률: +8.9%
누적 수익률: +34.2%
코스피 수익률: +12.5% (알파: +21.7%)

최대 낙폭(MDD): -18.3% (2025-08-12 ~ 2025-09-20)
샤프 비율: 1.24
평균 보유 기간: 14.3일

[패턴별 성과]
모멘텀형: 승률 62.1% | 평균 +12.3% | 거래 45건
지속형: 승률 54.8% | 평균 +18.7% | 거래 72건
전환형: 승률 47.6% | 평균 +9.1% | 거래 35건

[시그널별 성과]
시그널 3개: 승률 72.0% | 평균 +18.7% | 거래 25건 ⭐
시그널 2개: 승률 61.3% | 평균 +13.4% | 거래 48건
시그널 1개: 승률 52.1% | 평균 +8.9% | 거래 54건
시그널 0개: 승률 44.8% | 평균 +5.2% | 거래 25건

[월별 수익률]
2024-Q1: +8.2%  | 2024-Q2: +12.5% | 2024-Q3: -3.1%  | 2024-Q4: +9.8%
2025-Q1: +15.3% | 2025-Q2: +7.6%  | 2025-Q3: -8.2%  | 2025-Q4: +11.4%
2026-Q1: +6.7%

[권장 전략]
✅ 시그널 2개 이상 종목만 진입 (승률 61%+, 평균 수익 13%+)
✅ 지속형 패턴 우선 (장기 상승 확률 높음)
⚠️  전환형 신중 (승률 낮음, 손절 엄격 필요)
================================================================================
```

#### 구현 순서
1. **Week 1**: BacktestEngine 클래스 구현 (진입/청산 로직)
2. **Week 2**: PerformanceMetrics 구현 (승률, MDD, 샤프 비율)
3. **Week 3**: 시각화 + CSV 저장
4. **Week 4**: Optimizer 구현 (Grid Search)
5. **Week 5**: Walk-Forward 검증 + 최종 리포트

**참고**:
- Stage 3 완료로 핵심 기능 구현 완료
- Stage 4는 전략 검증 및 실전 투자 준비를 위한 선택적 고도화 단계
- 구현 시 약 2~3주 소요 예상 (주말 작업 기준)

---

## [Workflow]

### 작업 시작 시
```bash
git pull origin main
```
→ CLAUDE.md [Status] 확인
→ Claude에게: "CLAUDE.md 읽고 작업 이어서 해줘"

### 작업 종료 시
```bash
git add .
git commit -m "[집/회사] 작업 내용"
git push origin main
```

**중요**: 커밋 후 반드시 **푸시**까지 완료! (집/회사 컴퓨터 동기화)

---

## [Project Structure]

```
LP_MM_TRADING/
├── CLAUDE.md                      # 프로젝트 상태 (이 파일)
├── IMPLEMENTATION_GUIDE.md        # Stage 1~2 구현 가이드 (상세)
├── DATABASE_README.md             # 데이터베이스 사용 가이드
├── README.md                      # 프로젝트 소개
├── requirements.txt               # 의존성
├── data/
│   └── processed/
│       └── investor_data.db       # SQLite DB (171,227 레코드)
├── src/
│   ├── database/                  # DB 모듈
│   │   ├── schema.py
│   │   └── connection.py
│   ├── analyzer/                  # 분석 모듈
│   │   ├── normalizer.py          # Sff/Z-Score 계산
│   │   ├── pattern_classifier.py  # 패턴 분류 (Stage 3) ✨
│   │   ├── signal_detector.py     # 시그널 탐지 (Stage 3) ✨
│   │   └── integrated_report.py   # 통합 리포트 (Stage 3) ✨
│   ├── visualizer/                # 시각화 모듈
│   │   ├── performance_optimizer.py
│   │   └── heatmap_renderer.py
│   ├── config.py                  # 전역 설정
│   └── utils.py                   # 입력 검증 (보안)
├── scripts/
│   ├── analysis/
│   │   ├── abnormal_supply_detector.py  # Stage 1 CLI
│   │   ├── heatmap_generator.py         # Stage 2 CLI
│   │   └── regime_scanner.py            # Stage 3 CLI (통합) ✨
│   ├── crawlers/
│   │   ├── crawl_all_data.py
│   │   ├── crawl_stock_prices.py
│   │   └── crawl_free_float.py
│   └── loaders/
│       ├── load_initial_data.py
│       └── load_daily_data.py
└── tests/                         # 테스트 (105개, Stage 1~3, 100% 통과)
    ├── test_config.py
    ├── test_normalizer.py
    ├── test_performance_optimizer.py
    ├── test_utils.py
    ├── test_pattern_classifier.py       # Stage 3 테스트 ✨
    ├── test_signal_detector.py          # Stage 3 테스트 ✨
    └── test_integrated_report.py        # Stage 3 테스트 ✨
```

---

## [Tech Stack]
- Python 3.10+
- SQLite (DB)
- pandas, numpy (분석)
- matplotlib, seaborn (시각화)
- FinanceDataReader, BeautifulSoup (크롤링)
- pytest (테스트)

---

## [Data]
- **171,227 레코드** (2024-01-02 ~ 2026-01-20)
- **345개 핵심 종목** (KOSPI200 + KOSDAQ150)
- **1,609개 종목 마스터** (섹터 정보 97.9% 커버리지)
- **13개 컬럼** (수급 + 주가 + 유통주식)

---

## [Key Concepts]

자세한 내용은 **ANALYSIS_GUIDE.md** 참조

---

## [Performance]

| Stage | 처리 | 소요시간 | 최적화 |
|-------|------|----------|--------|
| Stage 1 | 345종목 이상 수급 탐지 | ~15초 | 벡터화 완료 |
| Stage 2 | 345종목×6기간 히트맵 | ~1.5초 | Sff 캐싱 + groupby.transform |
| Stage 3 | 패턴 분류 + 시그널 통합 | ~1.5초 | 벡터화 + 병렬 처리 |
| **전체** | **Stage 1~3 통합 실행** | **~3초** | **최적화 완료** |

---

## [방법론: 수급 레짐 스캐너]

### ① Stage 1: 데이터 정규화
- Sff: 유통물량 대비 순매수 강도
- Z-Score: 변동성 보정 수급

### ② Stage 2: 시공간 매트릭스
- 6개 기간(1W~2Y) 히트맵
- 4가지 정렬 모드 (투자 스타일별)

### ③ Stage 3: 패턴 분류 & 시그널 통합
- **패턴 분류**: 3개 바구니 (모멘텀형/지속형/전환형)
- **시그널 탐지**: MA 골든크로스, 수급 가속도, 외인-기관 동조율
- **통합 리포트**: 종목별 1줄 요약 카드 + 진입/청산 포인트

### ④ Stage 4: 백테스팅 (선택)
- 과거 데이터 검증
- 패턴별 수익률 분석
- 리스크 관리 최적화

---

## [Progress History]

### 2026-02-13 (Stage 3 버그 수정)
- ✅ 엣지 케이스 처리 (빈 DataFrame, 잘못된 입력)
- ✅ 테스트 완성도 향상 (105개 테스트 100% 통과)
- ✅ Stage 4 상세 계획 작성 (백테스팅 시스템)
- ✅ CLAUDE.md 업데이트 (v3.1)

### 2026-02-12 (Stage 3 완료)
- ✅ PatternClassifier 구현 (3개 바구니 자동 분류)
- ✅ SignalDetector 구현 (MA/가속도/동조율)
- ✅ IntegratedReport 구현 (종목별 요약 카드)
- ✅ RegimeScanner CLI 도구 (통합 파이프라인)
- ✅ 105개 테스트 작성 (Stage 1~3 통합)
- ✅ CLAUDE.md 업데이트 (v3.0)

### 2026-02-11 (Stage 2 완료)
- ✅ 히트맵 시각화 구현 (6개 기간)
- ✅ 4가지 정렬 모드 추가 (Recent/Momentum/Weighted/Average)
- ✅ 성능 최적화 (1.5초)
- ✅ 61개 테스트 (100% 통과)
- ✅ IMPLEMENTATION_GUIDE.md 작성

### 2026-02-10 (Stage 1 완료)
- ✅ Sff/Z-Score 정규화 모듈
- ✅ 이상 수급 탐지기 CLI
- ✅ 섹터 정보 크롤링 (97.9% 커버리지)
- ✅ 주가/유통주식 크롤링

### 2026-02-09 (프로젝트 초기화)
- ✅ GitHub 저장소 연결
- ✅ SQLite DB 스키마 설계
- ✅ 초기 데이터 로드 (172,155 레코드)
- ✅ 엑셀 파싱 모듈

---

## [Reference]
- **IMPLEMENTATION_GUIDE.md**: Stage 1~2 상세 구현 내용
- **DATABASE_README.md**: DB 스키마 및 사용법
- **README.md**: 프로젝트 소개 및 설치 방법

---

**프로젝트 버전**: v3.1 (Stage 3 완료 + 버그 수정)
**마지막 업데이트**: 2026-02-13
