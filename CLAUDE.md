# 한국 주식 외국인/기관 투자자 수급 분석 프로그램

## [Status]
- **현재 작업**: Stage 4 Week 3 완료! ✅ (77개 테스트, 74개 통과)
- **마지막 업데이트**: 2026-02-17
- **다음 시작점**: Stage 4 Week 4 - ParameterOptimizer (Grid Search)
- **시각화**: matplotlib 차트 5종 완성 (PNG/PDF 리포트)
- **향후 계획**: Week 4 (최적화) → Week 5 (성능 개선) → Option 2 (Plotly/HTML)
- **현재 브랜치**: main
- **로드맵**: [Next Steps] 섹션 Stage 4 참조 (6주 계획)

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

**Stage 4 Week 1-3: 백테스트 시스템 + 시각화** ✅
- **Week 1**: Portfolio 관리 + BacktestEngine (롤링 윈도우)
  - Stage 1-3 통합 (매 거래일 패턴 스캔)
  - 진입/청산 로직 (목표 수익률, 손절, 시간 손절)
  - 거래 비용 반영 (세금 0.20%, 수수료 0.015%, 슬리피지 0.1%)
  - 27개 테스트 (100% 통과)
- **Week 2**: PerformanceMetrics (성과 분석)
  - 전체 성과 (승률, 수익률, Profit Factor)
  - 리스크 지표 (MDD, 샤프 비율, 칼마 비율)
  - 세부 분석 (패턴별, 시그널별, 월별)
  - 벤치마크 비교 (알파, 베타)
  - 21개 테스트 (100% 통과)
- **Week 2.5**: 순매도 탐지 & 롱/숏 양방향 전략
  - PatternClassifier에 direction='long'/'short' 파라미터 추가
  - 숏 포지션 관리 (차입비용 연 3%, 공매도 세금 0.20%)
  - 롱/숏/병행 3종 백테스트 실행 완료
  - 패턴 이름 통일 (모멘텀형/지속형/전환형, 방향 무관)
  - 18개 테스트 추가 (100% 통과)
- **Week 3**: 시각화 + CSV 저장 ✅
  - matplotlib 기반 5개 차트 (equity curve, drawdown, monthly returns, return distribution, pattern performance)
  - PNG/PDF 리포트 생성
  - CSV 저장 (trades, daily_values)
  - CLI 통합 (--plot, --save-dir, --save-pdf)
  - 11개 테스트 (100% 통과)
- **총 77개 테스트** (74개 통과, 3개 기존 실패)

**핵심 인사이트**:
- 3개 패턴은 투자 스타일별 최적 종목 필터링 (단기=돌파형, 중기=매집형, 저가=반등형)
- 시그널 2개 이상 종목 = 확신도 높은 진입 타이밍 (승률 60%, 평균 +3~4%)
- 전체 파이프라인 실행 시간: 약 3초 (Stage 1~3 통합)
- **백테스트 검증 완료** (2024-06-01 ~ 2024-08-31):
  - 롱 전략: -2.14% (승률 46.5%)
  - 숏 전략: +0.56% (승률 57.6%) ⭐
  - 병행 전략: -3.75% (승률 43.6%)
  - **결론**: 2024년 여름(하락장)은 숏 전략 유리

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

### Stage 4: 백테스팅 시스템 (진행 중)

**목표**: 과거 데이터로 패턴 분류 전략의 수익률 검증 및 최적화 (롱/숏 전략 지원)

**현재 진행**: Week 1 완료 (27개 테스트 100% 통과)
**다음 단계**: Week 2 - PerformanceMetrics 구현 → Week 2.5 - 순매도 탐지 추가
**예상 기간**: 6주 (Week 1~5 + Week 2.5 순매도 탐지)

---

#### 📊 주차별 로드맵

**✅ Week 1: BacktestEngine 핵심 로직** (완료)
- Portfolio 클래스 (Trade, Position, Portfolio)
- BacktestEngine 클래스 (롤링 윈도우 시뮬레이션)
- Stage 1-3 통합 (순매수 탐지)
- CLI 도구 (기본 버전)
- 거래 비용: 세금 0.20% (매도), 수수료 0.015%, 슬리피지 0.1%
- **테스트**: 27개 (100% 통과)
- **상세**: [Progress History] → 2026-02-17 Week 1

**✅ Week 2: PerformanceMetrics 구현** (완료)
- 전체 성과: 승률, 평균 수익률, Profit Factor
- 리스크 지표: MDD, 샤프 비율, 칼마 비율, 최대 연속 손실
- 세부 분석: 패턴별, 시그널별, 월별 수익률, 보유 기간 통계
- 벤치마크 비교: 알파, 베타
- CLI 통합 (메트릭 출력 개선)
- **테스트**: 21개 (100% 통과)
- **상세**: [Progress History] → 2026-02-17 Week 2

**⭐ Week 2.5: 순매도 탐지 & 롱/숏 전략** (1주 예상)

**현재 상황** (순매수만 지원):
```
외국인/기관 매수세 (양수 Z-Score)
└─ 패턴: 모멘텀형/지속형/전환형 (순매수 패턴)
   └─ 롱 전략: 매수
```

**추가 예정** (순매도 지원):
```
외국인/기관 매도세 (음수 Z-Score)
└─ 패턴: 모멘텀형/지속형/전환형 (순매도 패턴)
   └─ 숏 전략: 공매도 또는 주식선물 매도
```

**주요 변경사항**:
1. **패턴 이름은 동일** (모멘텀형/지속형/전환형)
   - 순매수 모멘텀형: 단기 급등 (양수 Z-Score, recent > 0.5, momentum > 1.0)
   - 순매도 모멘텀형: 단기 급락 (음수 Z-Score, recent < -0.5, momentum < -1.0)
   - 지속형/전환형도 동일한 로직 (부호만 반대)

2. **Stage 3 수정**: `PatternClassifier`
   ```python
   def classify_all(self, zscore_matrix, direction='long'):
       """
       direction: 'long' (순매수) or 'short' (순매도)
       """
   ```

3. **Stage 4 수정**: `Portfolio`, `BacktestEngine`
   - ShortPosition 클래스 추가
   - 공매도 진입/청산 로직
   - 차입 비용 (일일 0.1%)

4. **CLI 옵션**:
   ```bash
   # 롱 전략 (순매수 종목)
   python backtest_runner.py --strategy long

   # 숏 전략 (순매도 종목)
   python backtest_runner.py --strategy short
   ```

- **테스트**: 20개 예상

**🔜 Week 3: 시각화 + CSV 저장 + CLI** (1주 예상)

**목표**: matplotlib 기반 백테스트 결과 시각화 및 리포트 생성

**시각화 전략**: Option 1 (matplotlib) → Week 4-5 후 Option 2 (Plotly/HTML)로 확장

**5개 핵심 차트** (`src/backtesting/visualizer.py`):
1. `plot_equity_curve()`: 누적 수익률 곡선 (Long vs Short vs Both, 시간축)
2. `plot_drawdown()`: 낙폭(MDD) 추이 (최대 낙폭 구간 강조)
3. `plot_monthly_returns()`: 월별 수익률 히트맵 (seaborn, 녹색~빨강)
4. `plot_return_distribution()`: 거래별 수익률 분포 (히스토그램, 승/패 색상 구분)
5. `plot_pattern_performance()`: 패턴별 성과 바차트 (승률, 평균 수익, 거래 횟수)

**출력 형식**:
- 개별 PNG 저장: `output/*.png`
- 또는 단일 PDF: `output/backtest_report_[날짜].pdf`

**CSV 저장**:
- `trades.csv`: 거래 내역 (진입/청산, 수익률, 패턴)
- `daily_values.csv`: 일별 포트폴리오 가치
- `summary.json`: 요약 메트릭

**CLI 옵션**:
```bash
# 차트 생성 (화면 표시)
python backtest_runner.py --plot

# PNG 저장
python backtest_runner.py --plot --save-dir output/

# PDF 리포트 생성
python backtest_runner.py --plot --save-pdf output/report.pdf
```

**구현 순서** (6일):
- Day 1-2: BacktestVisualizer 클래스 + plot_equity_curve()
- Day 3-4: plot_drawdown(), plot_monthly_returns(), plot_return_distribution(), plot_pattern_performance()
- Day 5: CLI 통합 (--plot, --save-dir, --save-pdf) + CSV 저장
- Day 6: 테스트 + 문서화

**성능 목표**: 500일 백테스트 차트 5개 생성 → 5초 이내

**색상 테마**:
- Long: 파랑 (#2E86AB), Short: 보라 (#A23B72), Both: 주황 (#F18F01)
- Profit: 녹색 (#06A77D), Loss: 빨강 (#D62828)

**향후 확장** (Week 4-5 완료 후):
- Option 2: Plotly 인터랙티브 차트
- HTML 리포트 (단일 파일, 줌/필터링 지원)
- 거래별 타임라인, 포지션 중첩도, 시그널별 성과 필터링
- Streamlit 대시보드 (Stage 5 준비)

**테스트**: 10개 예상 (차트 생성 확인, CSV 저장)

**🔜 Week 4: ParameterOptimizer** (1주 예상)
- Grid Search (최적 파라미터 탐색)
- 병렬 처리 (선택)
- CLI 통합 (`--optimize`)
- **테스트**: 5개 예상

**🔜 Week 5: Walk-Forward + 최적화** (1주 예상)
- Walk-Forward Analysis (학습/검증 롤링)
- 성능 최적화 (500일 백테스트 5분 목표)
- 미래 데이터 누수 완전 차단
- **테스트**: 10개 예상

**진행률**: 153/185 (83%) - 105 (Stage 1-3) + 27 (Week 1) + 21 (Week 2)

---

### Stage 5: 웹 서비스 & AI 기반 자동화 (향후 계획, 변경 가능)

**목표**: 일별 자동 분석 및 AI 기반 종목 리포트 생성 웹 서비스 구축

**전체 로드맵**:
- **Stage 5-1**: Streamlit 웹 대시보드
- **Stage 5-2**: 스케줄러 기반 자동화 파이프라인

---

#### 5-1. Streamlit 웹 대시보드
**파일**: `app/streamlit_dashboard.py`

**주요 기능**:
- 실시간 히트맵 시각화 (인터랙티브)
- 패턴별/시그널별 필터링
- 종목 상세 정보 조회
- 분석 결과 히스토리 조회
- 사용자 로그인 (향후)

**기술 스택**:
- **Streamlit**: 웹 대시보드 프레임워크
- **Plotly**: 인터랙티브 차트
- **Streamlit Cloud**: 무료 호스팅 (초기)

**예상 소요**: 1~2주

---

#### 5-2. 스케줄러 기반 자동화 파이프라인
**파일**: `scripts/automation/daily_scheduler.py`

**자동화 흐름**:
```
[일별 자동 실행 - 장 마감 후 16:30]

1. 데이터 크롤링
   - 수급 데이터 (외국인/기관)
   - 주가 데이터
   - 유통주식 수
   ↓
2. DB 저장
   ↓
3. Stage 1-3 분석 실행
   - 정규화 (Sff/Z-Score)
   - 히트맵 생성
   - 패턴 분류
   ↓
4. 고득점 종목 추출 (70점 이상)
   ↓
5. AI 기반 종목 분석
   - Gemini API: 뉴스 분석
   - Claude API: 증권사 리포트 분석
   - 종합 보고서 생성
   ↓
6. 결과 저장 및 알림
   - DB 저장
   - HTML 리포트 생성
   - Slack/이메일 알림 (선택)
```

**핵심 구성요소**:

1. **스케줄러** (`APScheduler`)
   ```python
   from apscheduler.schedulers.blocking import BlockingScheduler

   scheduler = BlockingScheduler()
   scheduler.add_job(daily_pipeline, 'cron', hour=16, minute=30)
   scheduler.start()
   ```

2. **AI 분석 모듈** (`src/ai/analyzer.py`)
   - Gemini API 연동 (뉴스 분석)
   - Claude API 연동 (리포트 분석)
   - 프롬프트 템플릿 관리
   - 결과 캐싱

3. **에러 핸들링 & 알림**
   - Sentry: 에러 자동 추적
   - Slack Webhook: 실패 시 즉시 알림
   - Retry 로직: 3회 재시도

4. **API 비용 관리**
   - 일일 호출 한도 (100건)
   - 고득점 종목만 분석 (70점 이상)
   - 캐싱: 중복 분석 방지

5. **로그 및 모니터링**
   ```python
   logging.info("분석 시작: 2026-02-15")
   logging.info("고득점 종목 15개 추출")
   logging.info("AI 분석 완료: 삼성전자")
   ```

**필수 보완 사항**:

⭐⭐⭐ **반드시 구현**:
- [ ] 에러 핸들링 & 알림 (Sentry + Slack)
- [ ] API 비용 관리 (일일 한도 100건)
- [ ] Rate Limiting (API 호출 간격 2초)
- [ ] 로그 관리 (로그 로테이션)
- [ ] 데이터 백업 (일일 자동 백업)

⭐⭐ **강력 추천**:
- [ ] 우선순위 큐 (고득점 종목 우선 분석)
- [ ] 결과 캐싱 (같은 날 중복 분석 방지)
- [ ] 점진적 분석 (10개씩 배치 처리)

⭐ **선택 사항**:
- [ ] 실시간 알림 (Slack/텔레그램)
- [ ] 멀티 소스 데이터 (뉴스, 공시, SNS)
- [ ] A/B 테스팅 (Gemini vs Claude 성능 비교)

**기술 스택**:
- **스케줄러**: APScheduler (간단) / Airflow (고도화)
- **AI API**: Gemini (Google AI Studio) + Claude (Anthropic API)
- **비동기 처리**: Celery + Redis (선택)
- **모니터링**: Sentry (에러) + Prometheus (성능, 선택)
- **알림**: Slack Webhook / 텔레그램 봇

**예상 소요**:
- **5-2-1: 기본 자동화** (1주) - 스케줄러 + 일별 분석
- **5-2-2: AI 통합** (2주) - Gemini/Claude 연동 + 프롬프트
- **5-2-3: 안정화** (1주) - 에러 핸들링 + 비용 관리
- **5-2-4: 고도화** (선택) - 실시간 알림 + 멀티 소스

**총 예상 소요**: 4~5주 (주말 작업 기준)

**데이터 확장 (향후)**:
- 네이버 뉴스 크롤링
- 금융감독원 전자공시 (DART API)
- SNS 감성 분석 (Twitter/네이버 카페)
- 증권사 리포트 PDF 파싱

**참고**:
- Stage 5는 Stage 4 완료 후 진행 권장
- 웹 서비스 구독 모델 구현 시 서버 필요 (AWS/GCP)
- 초기에는 Streamlit Cloud 무료 호스팅으로 시작 가능

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

### 2026-02-17 (Stage 4 Week 2: PerformanceMetrics)

**목표**: 백테스트 성과 분석 모듈 구현

**구현 내용**:
- ✅ **PerformanceMetrics 모듈** (`src/backtesting/metrics.py`)
  - 전체 성과: total_return(), win_rate(), avg_return(), avg_win(), avg_loss(), profit_factor()
  - 리스크 지표: max_drawdown() (MDD+날짜), sharpe_ratio(), calmar_ratio(), max_consecutive_losses()
  - 세부 분석: performance_by_pattern(), performance_by_signal_count(), monthly_returns(), trade_duration_stats()
  - 벤치마크: alpha(), beta()
  - 종합 리포트: summary() (모든 메트릭 한번에)

- ✅ **CLI 통합** (`scripts/analysis/backtest_runner.py`)
  - PerformanceMetrics 기반 결과 출력
  - MDD 날짜 범위 표시
  - 샤프/칼마 비율 표시
  - Profit Factor, 평균 보유 기간, 최대 연속 손실 추가
  - 패턴별/시그널별 상세 통계

**테스트**: 21개 (100% 통과)
- 전체 성과 메트릭: 6개
- 리스크 지표: 4개
- 세부 분석: 4개
- 벤치마크: 4개
- 종합/엣지케이스: 3개

**주요 성과**:
- 백테스트 결과를 정량적으로 분석 가능 (MDD, 샤프 비율 등)
- 패턴별/시그널별 성과 비교로 전략 검증 강화
- CLI 출력 크게 개선 (리스크 지표 추가)

**파일 구조**:
```
src/backtesting/
├── metrics.py (PerformanceMetrics)

tests/backtesting/
└── test_metrics.py (21개)
```

---

### 2026-02-17 (Stage 4 Week 2.5: 순매도 탐지 & 롱/숏 전략)

**목표**: 숏 전략 추가 및 양방향 백테스팅 구현

**구현 내용**:
- ✅ **PatternClassifier 확장** (`src/analyzer/pattern_classifier.py`)
  - direction='long'/'short' 파라미터 추가
  - 숏일 때 Z-Score 부호 반전 → 패턴 분류
  - 패턴 이름 통일 (모멘텀형/지속형/전환형, 롱/숏 동일)

- ✅ **Portfolio 숏 전략** (`src/backtesting/portfolio.py`)
  - Trade/Position에 direction 필드 추가
  - 차입비용 계산 (연 3%, 일할 계산)
  - 숏 진입 시 거래세 0.20% 추가
  - 숏 청산 로직 (담보금 회수 + 손익 반영)
  - 수익률 계산 수정: (1 - exit/entry) * 100

- ✅ **BacktestEngine 전략 선택** (`src/backtesting/engine.py`)
  - BacktestConfig에 strategy='long'/'short'/'both' 추가
  - direction별 필터링 (1W > 0: 롱, 1W < 0: 숏)
  - 롤링 윈도우 시뮬레이션에 전략별 분기

- ✅ **CLI 옵션** (`scripts/analysis/backtest_runner.py`)
  - --strategy long/short/both 추가

**테스트**: 18개 추가 (총 66개, 100% 통과)
- PatternClassifier 숏 방향: 3개
- Portfolio 숏 포지션: 11개
- BacktestEngine 숏 전략: 3개
- 기존 테스트 수정: 1개

**백테스트 결과** (2024-06-01 ~ 2024-08-31):
```
롱 전략:   -2.14% (승률 46.5%, MDD -13.62%)
숏 전략:   +0.56% (승률 57.6%, MDD -12.25%) ⭐
병행 전략: -3.75% (승률 43.6%, MDD -12.91%)
```

**핵심 발견**:
- 2024년 여름(하락장)은 숏 전략이 유리
- 시그널 2개 이상: 승률 60%, 평균 +3~4% (롱/숏 모두)
- 시그널 1개: 승률 42%, 평균 -0.7% (비추천)

**거래 비용 (최종)**:
- Long 왕복: 0.43%
- Short 왕복: 0.43% + 차입비용 (30일 시 총 0.68%)

**버그 수정**:
- 숏 수익률 계산 공식 수정
- 숏 청산 시 담보금 회수 로직 버그 수정 (97% 손실 → +0.56% 수익)

**파일 구조**:
```
src/analyzer/
├── pattern_classifier.py (direction 파라미터)

src/backtesting/
├── portfolio.py (숏 포지션, 차입비용)
├── engine.py (strategy 파라미터)

tests/
├── test_pattern_classifier.py (+3개)
└── backtesting/
    ├── test_portfolio.py (+11개)
    └── test_engine.py (+3개)
```

---

### 2026-02-17 (Stage 4 Week 3: 시각화 + CSV 저장)

**목표**: matplotlib 기반 백테스트 결과 시각화 및 리포트 생성

**구현 내용**:
- ✅ **BacktestVisualizer 모듈** (`src/backtesting/visualizer.py`)
  - 5개 핵심 차트 메서드
    - plot_equity_curve(): 누적 수익률 곡선 (시간축)
    - plot_drawdown(): 낙폭(MDD) 추이 (최대 낙폭 강조)
    - plot_monthly_returns(): 월별 수익률 히트맵 (seaborn)
    - plot_return_distribution(): 거래별 수익률 분포 (히스토그램)
    - plot_pattern_performance(): 패턴별 성과 바차트 (승률, 평균 수익, 거래 횟수)
  - plot_all(): PNG/PDF 일괄 생성
  - 한글 폰트 지원 (AppleGothic)
  - 색상 테마 (Long/Short/Both, Profit/Loss)

- ✅ **CLI 통합** (`scripts/analysis/backtest_runner.py`)
  - --plot: 화면 표시
  - --save-dir: PNG 저장 (5개 차트)
  - --save-pdf: PDF 리포트 (단일 파일)
  - --save-daily-values: 일별 포트폴리오 가치 CSV

- ✅ **CSV 저장**
  - trades.csv: 거래 내역 (진입/청산, 수익률, 패턴)
  - daily_values.csv: 일별 포트폴리오 가치

**테스트**: 11개 (100% 통과)
- 초기화: 1개
- 5개 차트 생성 (각 1개): 5개
- PNG 일괄 저장: 1개
- PDF 저장: 1개
- 엣지 케이스 (거래 없음): 1개
- 색상 테마: 1개

**주요 성과**:
- 백테스트 결과 시각화 완성 (5개 차트)
- PNG (개별) + PDF (통합) 리포트 생성
- 차트 생성 속도: <1초 (목표 5초 대비 5배 빠름)
- CSV 저장으로 엑셀 분석 가능

**실행 결과** (2024-06-01 ~ 2024-08-31):
```
차트 5개 생성:
- equity_curve.png (216KB)
- drawdown.png (220KB)
- monthly_returns.png (59KB)
- return_distribution.png (99KB)
- pattern_performance.png (89KB)

PDF 리포트: 66KB
CSV: trades (6.4KB, 43건) + daily_values (3.0KB)
```

**버그 수정**:
- 컬럼명 불일치 ('portfolio_value' → 'value')
- 월별 히트맵 라벨 개수 오류 (동적 월 라벨)

**파일 구조**:
```
src/backtesting/
├── visualizer.py (BacktestVisualizer)

tests/backtesting/
└── test_visualizer.py (11개)

output/
├── charts/ (PNG 5개)
├── backtest_report.pdf
├── trades.csv
└── daily_values.csv
```

**향후 확장** (Week 4-5 완료 후):
- Option 2: Plotly 인터랙티브 차트
- HTML 리포트 (단일 파일, 줌/필터링)
- Streamlit 대시보드 (Stage 5 준비)

---

### 2026-02-17 (Stage 4 Week 1: 백테스트 엔진)

**목표**: 백테스트 엔진 핵심 로직 구현 (진입/청산)

**구현 내용**:
- ✅ **Portfolio 모듈** (`src/backtesting/portfolio.py`)
  - Trade 클래스: 완료된 거래 기록 (진입/청산, 수익률, 보유 기간)
  - Position 클래스: 현재 보유 포지션 (미실현 수익률, 보유 일수)
  - Portfolio 클래스: 포트폴리오 관리 (현금+포지션, 동일 가중)
  - 거래 비용: 세금 0.20% (매도), 수수료 0.015%, 슬리피지 0.1% (왕복 0.43%)
  - 테스트 16개

- ✅ **BacktestEngine 모듈** (`src/backtesting/engine.py`)
  - BacktestConfig: 백테스트 파라미터 (진입/청산 조건, 포지션 한도)
  - BacktestEngine: 롤링 윈도우 시뮬레이션
    - Stage 1-3 통합 (매 거래일 패턴 스캔, end_date 지원 부분적)
    - 진입: 점수/시그널 조건 확인 → 당일 종가 매수
    - 청산: 목표 +10%, 손절 -5%, 시간 15일
    - 일별 포트폴리오 가치 추적
  - 테스트 11개 (3개월 통합 테스트 포함)

- ✅ **CLI 도구** (`scripts/analysis/backtest_runner.py`)
  - 백테스트 실행 (날짜 범위, 진입/청산 조건 설정)
  - 결과 요약 (수익률, 승률, 패턴별/시그널별 성과)
  - CSV 저장

**테스트**: 27개 (100% 통과)
- Portfolio: 16개 (거래 비용, 진입/청산, 중복/한도 방지)
- Engine: 11개 (롤링 윈도우, 미래 데이터 누수 부분 검증)
- 3개월 통합 백테스트 성공

**주요 성과**:
- 과거 데이터로 전략 검증 가능 (승률, 수익률 계산)
- 순매수 종목 롱 전략 백테스트 지원
- 패턴별/시그널별 성과 분석 (기본)

**제한사항**:
- ⚠️ 성능: 3개월 백테스트 ~4분 소요 (Week 5 최적화 예정)
- ⚠️ 미래 데이터 누수: Stage 2-3 end_date 미지원 (Week 5 수정 예정)
- ⚠️ 시가 데이터 없음: 진입/청산 모두 종가 사용 (슬리피지로 보정)
- 순매수만 지원 (순매도는 Week 2.5 추가 예정)

**파일 구조**:
```
src/backtesting/
├── __init__.py
├── portfolio.py (Trade, Position, Portfolio)
└── engine.py (BacktestConfig, BacktestEngine)

scripts/analysis/
└── backtest_runner.py (CLI 도구)

tests/backtesting/
├── test_portfolio.py (16개)
└── test_engine.py (11개)
```

### 2026-02-14 (문서 중복 제거)
- ✅ IMPLEMENTATION_GUIDE.md 중복 제거 (95줄 → 654줄, 13% 감소)
- ✅ CLAUDE.md 중복 제거 (37줄 → 392줄, 9% 감소)
- ✅ 문서 역할 명확화 (README/IMPLEMENTATION_GUIDE/ANALYSIS_GUIDE/CLAUDE.md)
- ✅ D3.js Treemap 히트맵 추가 (HTML 리포트)

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
