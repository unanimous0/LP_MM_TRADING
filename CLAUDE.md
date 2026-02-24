# 한국 주식 외국인/기관 투자자 수급 분석 프로그램

## [Status]
- **현재 작업**: Stage 5-1 Streamlit 웹 대시보드 진행 중
- **마지막 업데이트**: 2026-02-25
- **백테스트 권장 시작일**: 2025-01-01 이후 (DB가 2024-01-02 시작이므로 1Y 데이터 확보)
- **다음 시작점**: 스코어링 시스템 개선 검토 (모멘텀 방향성/시간 순서 반영)
- **시각화**: matplotlib 5종 (PNG/PDF) + Plotly 5종 (Streamlit 인터랙티브)
- **Streamlit**: `venv/bin/streamlit run app/streamlit_app.py` → http://localhost:8501
- **현재 브랜치**: main
- **로드맵**: [Next Steps] 섹션 Stage 5 참조

### TODO (Optuna 파라미터 추가 시)
- **현재 최적화 대상 7개 파라미터, 기본 Trial 수 100** (2026-02-22 업데이트)
  - Trial 수는 파라미터 공간 탐색용 (데이터 기간/양과 무관)
  - 랜덤 시드 사용 중 (seed 고정 X) → trial 수를 충분히 늘려 편차를 줄이는 방식
  - 파라미터 수에 따른 권장 Trial 수:
    - 4개 (이전): 50 trial
    - 7개 (현재): 100 trial
    - 10개+: 200 trial 이상
  - 변경 위치: `src/backtesting/optimizer.py` (`DEFAULT_PARAM_SPACE` + `optimize()` 기본값)
               `app/utils/data_loader.py` (`run_optuna_optimization()` 기본값)
               `app/pages/3_📈_백테스트.py` (슬라이더 기본값)

### TODO (수급 DB 교체 시 — 22년 데이터 전환)
> **⚠️ 반드시 수행**: 수급 DB가 바뀌면 Optuna study 결과도 무효화됨

**배경**: Optuna study 이름은 최적화 기간만 보고 생성됨 (`opt__long__20250101__20250930__sharpe_ratio`)
- 기저 데이터(수급 DB)가 바뀌어도 이름이 동일 → 이전 DB 기반 trial 위에 누적됨
- 22년 데이터 추가 시 Z-Score 계산 기준 자체가 달라지므로 기존 trial은 무효

**DB 교체 시 수행 순서**:
```bash
# 1. 기존 Optuna study 삭제
rm data/optuna_studies.db

# 2. git에 반영 (양쪽 컴퓨터 동기화)
git add data/optuna_studies.db
git commit -m "Optuna study 초기화 (22년 DB 전환)"
git push
```

**Optuna study DB 관리 정책**:
- `data/optuna_studies.db` — git으로 관리 (집↔회사 컴퓨터 간 공유)
- study 이름 = `opt__{strategy}__{시작일}__{종료일}__{metric}` (기간+전략+지표 조합)
- 같은 컴퓨터에서 재시작/새로고침해도 누적 trial 유지됨
- **DB 교체 시에만** 위 삭제 절차 수행

### TODO (데이터 개선 시)
- [ ] **공유 한국주식 DB 연동** → 크롤링 제거 + 다중 프로젝트 공유
  - 여러 프로젝트에서 공동으로 사용하는 한국주식 DB 구축 중
  - 완성 시: `src/database/connection.py` 접속 설정만 변경하면 이 프로젝트에 즉시 적용
  - 효과: 크롤링 유지보수 부담 제거, 데이터 일관성 확보
- [ ] **23년 데이터 추가** → 백테스트 기간 확장 (현재 2024-01-02부터 시작)
  - 23년 데이터 추가 시 2024년 백테스트도 2Y Z-Score 완전 활용 가능
- [ ] **시가 데이터 추가** → 진입/청산 타이밍 개선
  - 현재: 진입(다음 날 종가), 청산(반대 수급 = 다음 날 종가, 가격 기준 = 당일 종가)
  - 변경 후: 진입(다음 날 시가), 청산(반대 수급 = 다음 날 시가, 가격 기준 = 도달 시)
  - 영향: 더 현실적인 백테스트 결과 (시가가 종가보다 유리한 경우 많음)

### 주요 성과

**Stage 1: 데이터 정규화** ✅
- Sff/Z-Score 정규화 완료
- **외국인 중심 조건부 Sff**: 같은 방향 → 외국인+기관×0.3, 반대 방향 → 외국인만
- **조건부 Z-Score 적용**: 부호 전환 시 과잉 반응 방지 (작은 매도 → 큰 매도 오인 해결)
- 이상 수급 탐지 (20건 탐지, 15초 소요)
- 섹터 정보 통합 (97.9% 커버리지)

**Stage 2: 시공간 히트맵** ✅
- 7개 기간(5D~500D) 히트맵 시각화 (영업일 기준)
- 4가지 정렬 모드 (Recent/Momentum/Weighted/Average)
- **방향 확신도**: `tanh(today_sff/rolling_std)` — 정렬/분류 시 매도 종목 양수 Z-Score 오분류 방지
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
- **총 77개 테스트** (77개 통과)

**Stage 5-1: Streamlit 웹 대시보드** (진행 중)
- 멀티페이지 앱 (홈/분석/백테스트)
- 백테스트 페이지: Plotly 인터랙티브 차트 5종 + KPI 카드 + 거래 내역
- **BacktestPrecomputer**: 백테스트 속도 165~262배 향상 (177초→1.1초)
- **Optuna 최적화 UI**: 사이드바에서 파라미터 자동 최적화 → 결과 즉시 반영
- **[수정] institution_weight 설계 개선**: 버그 수정 + 최적화 파라미터 분리 + Precomputer 공유 캐싱
- **Persistent Optuna Study**: SQLite 누적 저장 → Trial 수 늘릴수록 최적값 단조 증가 보장
- **백테스트 UI 개선**: 검증/최적화 결과 시각 구분 (색상 테두리), 사이드바 3섹션, 누적 Trial 표시
- **다크 테마 전면 적용**: `.streamlit/config.toml` + `plotly_visualizer.py` + `charts.py` + `theme=None`
- **슬라이더 + 직접 입력 동기화**: `_synced_slider()` 헬퍼 (마우스 드래그 OR 숫자 직접 입력)
- **분석 진행률 표시**: 전 페이지 `st.progress()` % 표시 (스테이지별 캐시 분리로 CacheReplayClosureError 해결)
- **2026년 날짜 선택 버그 수정**: 모든 date_input max_value 연장
- **백테스트 사이드바 구조 개선**: 🧪 최적화 대상 / 🔒 고정 조건 섹션 분리
- **Optuna 최적화 대상 확장**: 4개 → 7개 (max_positions/max_hold_days/reverse_threshold 추가)
- **초기 자본금 쉼표 포맷**: text_input + on_change 콜백으로 입력창 자체에 쉼표 표시
- **[버그수정] institution_weight 최적화 불일치**: 최적화 시 항상 0.3 사용 → 사이드바 값 전달로 수정
- **institution_weight 글로벌 사이드바**: 전 페이지 공유 슬라이더 (`key="w_institution_weight"`) + 분석 파이프라인 파라미터화
- **거래 비용 파라미터화**: Portfolio 클래스 속성 → `__init__` 파라미터 전환, BacktestConfig → Engine → Portfolio 전달 체인
- **백테스트 거래비용 UI**: 🔒 고정 조건에 expander 추가 (세금/수수료/슬리피지/차입비용 조정 가능)
- **[버그수정] 최적화/워크포워드 거래비용 누락**: optimizer/walk_forward의 params dict에 비용 필드 추가
- **메인 페이지 재설계**: KPI 5개 + 이상수급/수급순위 탭 + 패턴 요약 + 관심종목 테이블
- **이상 수급 섹션**: Z>2σ 매수/매도 바차트 + 테이블 (호버에 순매수금액 억/조 포맷)
- **당일 수급 순위 탭**: 외국인/기관 순매수·순매도 Top 50 (차트 10 + 테이블 50, 쉼표 포맷)
- **산출 방식 설명**: Sff→합산→Z-Score 3단계 수식 + 외국인Z≠종합Z 이유 expander
- **메인 페이지 브랜딩**: 타이틀 "Whale Supply", `pages.toml`로 사이드바 메뉴명 커스터마이즈
- **이상 수급 날짜 선택**: 사이드바 date_input으로 과거 시점 이상 수급 조회 (이상 수급만 영향)
- **Z-Score 기준 기간 조정**: 사이드바 슬라이더(20~240일, 기본 60) — 이상 수급 전용, 다른 페이지 무관
- **산출 방식 설명 동적화**: 기관 가중치·기준 기간 현재값 반영 + "사이드바에서 조정 가능" 안내 추가
- **히트맵 인터랙티브 고도화**: A(클릭→미니상세) + B(호버 패턴/점수) + C(필터 사이드바) + D(섹터 평균 탭)
- **Z-Score 기간 라벨 통일**: 1W/1M/3M/6M/1Y/2Y → 5D/10D/20D/50D/100D/200D/500D (영업일 기준)
- **방향 확신도(Direction Confidence)**: `tanh(today_sff/rolling_std)` — Z-Score가 방향과 괴리되는 문제 해결 (매도 중 종목이 모멘텀 상위 랭크 방지)
- **[버그수정] Precomputer 방향 확신도 누락**: `_today_sff`/`_std_*D` 메타데이터 미전파 → 백테스트에서 방향 확신도 미적용 문제 수정
- **[통일] normalizer Z-Score 조건부 공식 적용**: `normalizer.calculate_zscore()`에도 조건부 Z-Score 적용 → 종목 상세/이상 수급 등 모든 경로에서 동일한 Z-Score 공식 사용
- **종목 상세 Z-Score 기준 기간 기본값**: 60 → 50 (50D 히트맵과 직접 비교 용이)
- **패턴분석 페이지 고도화**: 사이드바 정렬/방향 필터 + 7개 탭 (종목리스트/패턴통계/시그널/섹터크로스/섹터Z히트맵/수급집중도/Treemap)
- **섹터 크로스 분석**: 섹터×패턴 스택 바차트 + 섹터별 평균 점수 + 교차 테이블 + 시그널 통계
- **수급 집중도**: 섹터점수 = 평균점수 × (1 + 고득점/전체) + TOP 10 수평 바차트
- **D3.js Treemap**: 섹터별 종목 박스 (크기=점수, 색=RdYlGn, 호버 툴팁, 섹터 라벨+점수)
- 258개 테스트 (100% 통과)

**핵심 인사이트**:
- 3개 패턴은 투자 스타일별 최적 종목 필터링 (단기=돌파형, 중기=매집형, 저가=반등형)
- 시그널 2개 이상 종목 = 확신도 높은 진입 타이밍 (승률 60%, 평균 +3~4%)
- 전체 파이프라인 실행 시간: 약 3초 (Stage 1~3 통합)
- **백테스트 Precomputer**: 38일 177초→1.1초 (165배), 63일 393초→1.5초 (262배)
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

### Stage 4: 백테스팅 시스템 (완료 ✅)

**목표**: 과거 데이터로 패턴 분류 전략의 수익률 검증 및 최적화 (롱/숏 전략 지원)

**완료**: Week 1~5 + Week 2.5 + Optuna 통합 + Precomputer 속도 최적화 + institution_weight 설계 개선 (258개 테스트, 258개 통과) ✅

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

**✅ Week 2.5: 순매도 탐지 & 롱/숏 전략** (완료)

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

**✅ Week 3: 시각화 + CSV 저장 + CLI** (완료)

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

**✅ Week 4: 파라미터 최적화** (완료)
- Optuna Bayesian Optimization (--optimize, --walk-forward 공용)
  - 백테스트 파라미터: min_score, min_signals, target_return, stop_loss
  - ~~기관 가중치 최적화~~ → institution_weight는 분석 철학 파라미터로 분리 (하단 참고)
- CLI 통합 (`--optimize`, `--n-trials`, `--workers`, `--metric`, `--opt-save-csv`)
- **테스트**: 8개 (100% 통과)

**✅ Week 5: Walk-Forward + 성능 최적화** (완료)
- Walk-Forward Analysis (학습/검증 롤링)
- 성능 최적화: normalizer preload() - DB 1회 로드 후 메모리 필터링
- **테스트**: 15개 (100% 통과)
- **상세**: [Progress History] → 2026-02-19 Week 5

**✅ Optuna 전면 통합** (완료)
- Grid Search → Optuna Bayesian Optimization 전면 교체 (--optimize + --walk-forward)
- **MedianPruner**: 절반 기간 중간 평가 → 나쁜 Trial 조기 중단
- **2단계 탐색**: Phase 1 (넓은 범위) → Phase 2 (상위 25% 기준 좁혀서 집중)
- **기간 단위 병렬 실행**: multiprocessing.Pool (--workers N)
- **CLI**: `--n-trials N` 옵션 (--optimize, --walk-forward 공용, 기본: 50)
- ParameterOptimizer(Grid Search) 완전 제거, OptunaOptimizer로 통일
- **테스트**: 16개 (walk_forward) + 8개 (optimizer) (100% 통과)
- **상세**: [Progress History] → 2026-02-19

**진행률**: 256/256 (100%) - Stage 4 완료 ✅

---

### Stage 5: 웹 서비스 & AI 기반 자동화

**목표**: 일별 자동 분석 및 AI 기반 종목 리포트 생성 웹 서비스 구축

**전체 로드맵**:
- **Stage 5-1**: Streamlit 웹 대시보드 (진행 중)
- **Stage 5-2**: 스케줄러 기반 자동화 파이프라인

---

#### 5-1. Streamlit 웹 대시보드 (진행 중)

**구조**: `app/` 멀티페이지 앱

**완료된 기능**:
- ✅ 멀티페이지 앱 구조 (`app/streamlit_app.py` + `app/pages/`)
- ✅ 공유 데이터 로더 (`app/utils/data_loader.py`) - DB 연결, 분석/백테스트 캐싱
- ✅ 홈 페이지 (`1_🏠_홈.py`) - DB 통계, 최근 업데이트
- ✅ 분석 페이지 (`2_🔍_수급_분석.py`) - Stage 1-3 파이프라인, 패턴/시그널 필터링
- ✅ 백테스트 페이지 (`3_📈_백테스트.py`) - 파라미터 설정 + 실행 + 결과 시각화
  - Plotly 인터랙티브 차트 5종 (수익률 곡선, 낙폭, 월별 수익률, 수익률 분포, 패턴별 성과)
  - KPI 카드 5개 (총 수익률, 승률, MDD, 샤프 비율, 총 거래)
  - 거래 내역 테이블 + CSV 다운로드
  - **Optuna 파라미터 최적화** 버튼 (Trial 수/평가 지표 설정 → 최적 파라미터 자동 반영)
- ✅ BacktestPrecomputer 속도 최적화 (165~262배 향상)
- ✅ PlotlyVisualizer (`src/backtesting/plotly_visualizer.py`)

**남은 작업**:
- ✅ 종목 상세 페이지 (`app/pages/5_📋_종목상세.py`) — 4탭 (Z-Score추이/수급금액/시그널MA/패턴현황)
- ✅ 종목 상세 UI 개선 — 수급금액 테이블/차트/사이드바 대폭 개선
- ✅ 히트맵 페이지 — 인터랙티브 히트맵 (클릭→미니상세, 호버 패턴/점수, 필터, 섹터 평균)
- ✅ 분석 페이지 고도화 — 정렬 7종 + 수급 방향 필터 + 7개 탭 (섹터 크로스/Z히트맵/집중도/Treemap)

**실행 방법**:
```bash
venv/bin/streamlit run app/streamlit_app.py
# → http://localhost:8501
```

**기술 스택**:
- **Streamlit**: 웹 대시보드 프레임워크
- **Plotly**: 인터랙티브 차트
- **Streamlit Cloud**: 무료 호스팅 (초기)

---

#### 5-2. 스케줄러 기반 자동화 파이프라인
**파일**: `scripts/automation/daily_scheduler.py`

> ⚠️ **데이터 소스 전환 계획**
> 현재는 이 프로젝트 내에서 직접 크롤링하여 `investor_data.db`에 저장하는 구조이지만,
> **별도로 구축 중인 한국주식 공유 DB**(여러 프로젝트에서 공동 사용)가 완성되면
> 크롤링 단계를 제거하고 **공유 DB에서 데이터를 조회**하는 방식으로 전환할 예정이다.
> 이 경우 `src/database/connection.py`의 연결 대상만 공유 DB로 교체하면 된다.

**자동화 흐름 (현재)**:
```
[일별 자동 실행 - 장 마감 후 16:30]

1. 데이터 크롤링 (→ 공유 DB 전환 시 이 단계 제거)
   - 수급 데이터 (외국인/기관)
   - 주가 데이터
   - 유통주식 수
   ↓
2. DB 저장 (→ 공유 DB 전환 시 이 단계 제거)
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

**자동화 흐름 (공유 DB 전환 후)**:
```
[일별 자동 실행 - 장 마감 후 16:30]

1. 공유 한국주식 DB에서 최신 데이터 조회
   - 수급 데이터, 주가, 유통주식 수
   (크롤링/저장 불필요 - 공유 DB가 자동 갱신)
   ↓
2. Stage 1-3 분석 실행
   - 정규화 (Sff/Z-Score)
   - 히트맵 생성
   - 패턴 분류
   ↓
3. 고득점 종목 추출 (70점 이상)
   ↓
4. AI 기반 종목 분석
   - Gemini API: 뉴스 분석
   - Claude API: 증권사 리포트 분석
   - 종합 보고서 생성
   ↓
5. 결과 저장 및 알림
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
- **공유 DB 전환 시**: `src/database/connection.py`의 DB 경로/접속 설정만 변경하면 됨
  - 현재: `data/processed/investor_data.db` (로컬 SQLite)
  - 전환 후: 공유 한국주식 DB (PostgreSQL 등) 접속 정보로 교체

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
├── app/                           # Streamlit 웹 대시보드 (Stage 5-1) ✨
│   ├── streamlit_app.py           # 메인 엔트리포인트
│   ├── utils/
│   │   └── data_loader.py         # 캐시 데이터 로더 (DB/분석/백테스트/최적화)
│   └── pages/
│       ├── 1_🏠_홈.py              # DB 통계, 최근 업데이트
│       ├── 2_🔍_수급_분석.py       # Stage 1-3 분석 파이프라인
│       └── 3_📈_백테스트.py        # 백테스트 실행 + Optuna 최적화 + Plotly 차트
├── src/
│   ├── database/                  # DB 모듈
│   │   ├── schema.py
│   │   └── connection.py
│   ├── analyzer/                  # 분석 모듈
│   │   ├── normalizer.py          # Sff/Z-Score 계산 (외국인 중심 조건부)
│   │   ├── pattern_classifier.py  # 패턴 분류 (Stage 3)
│   │   ├── signal_detector.py     # 시그널 탐지 (Stage 3)
│   │   └── integrated_report.py   # 통합 리포트 (Stage 3)
│   ├── backtesting/               # 백테스트 모듈 (Stage 4)
│   │   ├── engine.py              # BacktestEngine (Precomputer 기반 고속 실행)
│   │   ├── precomputer.py         # BacktestPrecomputer (벡터화 사전 계산) ✨
│   │   ├── plotly_visualizer.py   # PlotlyVisualizer (Streamlit용 인터랙티브 차트) ✨
│   │   ├── portfolio.py           # Trade, Position, Portfolio
│   │   ├── metrics.py             # PerformanceMetrics
│   │   ├── optimizer.py           # OptunaOptimizer (Bayesian Optimization)
│   │   ├── walk_forward.py        # WalkForwardAnalyzer
│   │   └── visualizer.py          # 차트 5종 (matplotlib)
│   ├── visualizer/                # 시각화 모듈
│   │   ├── performance_optimizer.py
│   │   └── heatmap_renderer.py
│   ├── config.py                  # 전역 설정
│   └── utils.py                   # 입력 검증 (보안)
├── scripts/
│   ├── analysis/
│   │   ├── abnormal_supply_detector.py  # Stage 1 CLI
│   │   ├── heatmap_generator.py         # Stage 2 CLI
│   │   └── regime_scanner.py            # Stage 3 CLI (통합)
│   ├── crawlers/
│   │   ├── crawl_all_data.py
│   │   ├── crawl_stock_prices.py
│   │   └── crawl_free_float.py
│   └── loaders/
│       ├── load_initial_data.py
│       └── load_daily_data.py
└── tests/                         # 테스트 (258개 통과)
    ├── test_config.py
    ├── test_normalizer.py
    ├── test_performance_optimizer.py
    ├── test_utils.py
    ├── test_pattern_classifier.py       # Stage 3 테스트
    ├── test_signal_detector.py          # Stage 3 테스트
    ├── test_integrated_report.py        # Stage 3 테스트
    └── backtesting/                     # Stage 4 테스트
        ├── test_engine.py
        ├── test_portfolio.py
        ├── test_metrics.py
        ├── test_visualizer.py
        ├── test_precomputer.py          # BacktestPrecomputer (23개) ✨
        ├── test_optimizer.py            # OptunaOptimizer
        └── test_walk_forward.py         # Walk-Forward (16개)
```

---

## [Tech Stack]
- Python 3.10+
- SQLite (DB)
- pandas, numpy (분석)
- matplotlib, seaborn (시각화 - CLI/PDF)
- plotly (인터랙티브 차트 - Streamlit)
- streamlit (웹 대시보드)
- optuna (Bayesian Optimization - --optimize, --walk-forward, Streamlit 공용)
- FinanceDataReader, BeautifulSoup (크롤링)
- pytest (테스트)

---

## [UI 테마 / 색상]

**Streamlit 앱 테마** (`.streamlit/config.toml`):
- `base = "dark"` — 다크 모드 기반
- `primaryColor = "#38bdf8"` — sky-400 (강조색, 버튼/링크)
- `backgroundColor = "#0f172a"` — slate-900 (페이지 배경)
- `secondaryBackgroundColor = "#1e293b"` — slate-800 (사이드바/카드 배경)
- `textColor = "#e2e8f0"` — slate-200 (기본 텍스트)
- `font = "sans serif"`

**Plotly 차트 색상** (`src/backtesting/plotly_visualizer.py`):

| 역할 | 색상코드 | Tailwind 이름 | 설명 |
|------|---------|--------------|------|
| 차트 내부 배경 | `#0f172a` | slate-900 | `plot_bgcolor` |
| 차트 외곽 배경 | `#1e293b` | slate-800 | `paper_bgcolor` |
| 그리드 선 | `#334155` | slate-700 | 축/그리드 |
| 일반 텍스트 | `#e2e8f0` | slate-200 | 레이블/제목 |
| 보조 텍스트 | `#94a3b8` | slate-400 | 부제목/벤치마크 |
| Long 전략 | `#38bdf8` | sky-400 | 수익률 곡선 |
| Short 전략 | `#f472b6` | pink-400 | 수익률 곡선 |
| Both 전략 | `#fb923c` | orange-400 | 수익률 곡선 |
| 수익 (양수) | `#4ade80` | green-400 | 거래 수익/양봉 |
| 손실 (음수) | `#f87171` | red-400 | 거래 손실/음봉 |
| 평균선 | `#60a5fa` | blue-400 | 분포 차트 평균 |
| 중앙값선 | `#4ade80` | green-400 | 분포 차트 중앙값 |
| 무관심 구간 | `#64748b` | slate-500 | 중립 색상 |

**테마 팔레트**: Tailwind CSS Slate 계열 (어두운 남색 계통) + 파스텔 강조색

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
| **백테스트 38일** | **Precomputer 적용** | **~1.1초** | **165배 향상 (177초→1.1초)** |
| **백테스트 63일** | **Precomputer 적용** | **~1.5초** | **262배 향상 (393초→1.5초)** |
| **백테스트 1년** | **Streamlit 실행** | **~4초** | **Precomputer 자동 적용** |

---

## [방법론: 수급 레짐 스캐너]

### ① Stage 1: 데이터 정규화

#### Sff (Supply Flow Force) — 유통물량 대비 순매수 강도
```
foreign_sff     = 외국인 순매수금액 / (종가 × 유통주식수) × 100
institution_sff = 기관 순매수금액   / (종가 × 유통주식수) × 100
```
예: 삼성전자 유통시총 100조, 외국인 +1000억 매수 → `sff = 0.001 × 100 = 0.1`

**combined_sff** (외국인 중심 조건부 합산):
```
외국인·기관 같은 방향 → foreign_sff + institution_sff × 기관가중치(기본 0.3)
외국인·기관 반대 방향 → foreign_sff만 (기관이 외국인 신호를 상쇄하지 않음)
```
- 기관가중치 0 → combined = foreign (외국인만)
- 기관가중치 1.0 → combined = foreign + institution (동등 합산)

#### Z-Score — 변동성 보정 수급 (조건부 공식)
```
평균 = 최근 N일 Sff 이동평균
std  = 최근 N일 Sff 표준편차
```
**조건부 공식** (모든 경로에서 동일 적용):
```
같은 방향 (오늘·평균 부호 동일): Z = (오늘 - 평균) / std  ← 평균 대비 폭발 감지
방향 전환 (오늘·평균 부호 반대): Z = 오늘 / std           ← 크기만 평가
```
| 상황 | 오늘 Sff | 평균 | 공식 | Z | 해석 |
|------|---------|------|------|---|------|
| 매수 폭발 | +2.0 | +0.5 | (2-0.5)/0.4 | **+3.75** | 평소보다 훨씬 강한 매수 |
| 평소 매수 | +0.5 | +0.5 | (0.5-0.5)/0.4 | **0.0** | 평균 수준 |
| 살짝 매도 전환 | -0.04 | +0.5 | -0.04/0.4 | **-0.1** | 무시 (노이즈) |
| 강한 매도 전환 | -2.0 | +0.5 | -2.0/0.4 | **-5.0** | 강한 매도 신호 |

방향 전환 시 `(오늘-평균)/std`를 쓰면 살짝 매도(-0.04)도 Z=-1.35로 과대평가 → 조건부 공식으로 해결.

**적용 위치** (2개 독립 구현, 동일 공식):
- `normalizer.calculate_zscore()` — 종목 상세 KPI, 이상 수급, Z-Score 추이 차트
- `performance_optimizer._calculate_zscore_vectorized()` — 히트맵, 패턴 분류, 백테스트

#### Z-Score 해석 시 주의 — "누적순매수"와 "Z-Score"는 다른 지표

Z-Score는 **오늘 하루**의 Sff가 최근 N일 평균 대비 얼마나 이례적인가를 측정한다.
누적순매수(장기 축적량)와는 별개이므로, 누적 감소 중이어도 오늘 Z가 양수일 수 있다.

**실제 사례 — 하이트진로 (000080, 2026-01-20 기준, window=50, 기관가중치=0)**:
```
누적순매수: -775억 (2024년부터 장기 매도 누적, 7월 -285억 → 현재 -775억)
최근 50일:  매수 23일 / 매도 27일 → 평균 Sff = -0.043 (전반적 매도세)
오늘:       +8.9억 매수 → Sff = +0.160 (방향 전환!)
```
- `오늘(+0.160) × 평균(-0.043) < 0` → **방향 전환** 케이스
- `Z = 오늘 / std = 0.160 / 0.138 = +1.15`
- **의미**: 누적은 여전히 -766억이지만, "오늘은 최근 50일 매도 추세 대비 의미있는 매수 전환"

| 지표 | 보는 것 | 하이트진로 |
|------|---------|-----------|
| 누적순매수 | 장기 축적량 (과거 전체) | -775억 (장기 매도) |
| Sff | 오늘 하루의 유통물량 대비 강도 | +0.160 (오늘은 매수) |
| Z-Score | 오늘 Sff가 최근 N일 대비 얼마나 이례적인가 | +1.15 (매도세 속 매수 전환) |

**기관 Z = +0.14인 이유**: 기관도 방향 전환이지만 매수 크기(+0.6억)가 변동성 대비 미미 → 노이즈 수준.

### ② Stage 2: 시공간 매트릭스
- 7개 기간(5D~500D) 히트맵 (영업일 기준)
- 4가지 정렬 모드 (투자 스타일별)

#### 방향 확신도 (Direction Confidence) — Z-Score 위의 별도 레이어

**문제**: Z-Score는 "평균 대비 편차"를 측정 → 매도세 완화 종목도 양수 Z가 됨
- 예: 현대모비스, 장기 매도세(-0.5) 지속 중 매도세 소폭 완화(-0.2) → Z = +0.7 (양수!)
- 모멘텀 정렬 시 매도 종목이 매수 상위 #1에 노출

**해결**: tanh 기반 방향 확신도로 정렬/분류 시에만 가중치 적용
```
confidence = tanh(today_sff / rolling_std)
```
- tanh(0) = 0 → 중립 (노이즈 감쇠)
- tanh(1) ≈ 0.76 → 중간 신호
- tanh(2) ≈ 0.96 → 강한 신호 보존
- 하드코딩 상수 없음 — 각 종목 자신의 변동성(std)으로 정규화

**적용 방식**:
```
long 정렬/분류: adjusted_z = z × max(confidence, 0)    ← 매수 방향만 반영
short 정렬/분류: adjusted_z = z × max(-confidence, 0)   ← 매도 방향만 반영
```
- 매도 중(sff<0) 종목 → confidence<0 → max(confidence,0)=0 → adjusted_z=0 → 매수 순위에서 탈락
- **히트맵 셀 값은 원본 Z-Score 유지** (편차 정보 보존), 정렬/패턴분류만 보정값 사용

**⚠️ tanh는 Z-Score 공식이 아님**:
- Z-Score 공식: 조건부 `(X-μ)/σ` or `X/σ` → 수급 편차 측정 (값 자체)
- 방향 확신도: `tanh(sff/std)` → 정렬/분류 시 가중치 (값을 바꾸지 않음)
- 적용 위치: `pattern_classifier.py`, `charts.py` (히트맵 정렬) — 2곳에서 독립 계산

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

### 2026-02-25 (패턴분석 페이지 고도화 — 정렬/방향 필터 + 섹터 분석 4개 탭 추가)

**목표**: 패턴분석 페이지를 히트맵 수준으로 고도화 — 정렬/필터 + 섹터 심층 분석

**구현 내용**:

- ✅ **사이드바 정렬/방향 필터 추가** (`2_🔍_패턴분석.py`)
  - 정렬 기준 7종: 패턴 점수 / 최종 점수 / 최근 수급(5D) / 모멘텀 / 가중 평균 / 단순 평균 / 시그널 수
  - 수급 방향 라디오: 전체 / 매수 상위 / 매도 상위 (5D Z-Score 기준)
  - `final_score` 계산 (패턴점수 + 시그널수×5) + 정렬 적용

- ✅ **탭 3개 → 7개 확장** (`2_🔍_패턴분석.py`)
  - 기존: 종목 리스트 / 패턴별 통계 / 시그널 분석
  - 추가: 섹터 크로스 분석 / 섹터 Z-Score 히트맵 / 수급 집중도 / Treemap
  - 종목 리스트 탭: 정렬 키 컬럼 토글 체크박스 (recent/momentum/weighted/average 수치 표시)

- ✅ **섹터 크로스 분석 탭** (charts.py + 페이지)
  - `create_sector_pattern_crosstab_chart()`: 섹터×패턴 스택 바차트
  - `create_sector_avg_score_chart()`: 섹터별 평균 점수 수평 바차트
  - 교차 테이블 (pd.crosstab) + 섹터별 시그널 통계 테이블

- ✅ **섹터 Z-Score 히트맵 탭** (기존 `create_sector_zscore_heatmap()` 재사용)
  - 현재 필터 조건 적용된 종목의 섹터별 평균 Z-Score
  - 정렬 기준 자동 매핑 (score→weighted, recent→recent 등)

- ✅ **수급 집중도 탭** (charts.py + 페이지)
  - `create_sector_concentration_chart()`: 섹터점수 = 평균점수 × (1 + 고득점/전체), 5개 이상 섹터만
  - TOP 10 수평 바차트 + 집중도 테이블

- ✅ **D3.js Treemap 탭** (charts.py + 페이지)
  - `create_sector_treemap_html()`: D3.js v7 기반 인터랙티브 Treemap HTML 반환
  - 박스 크기: 종합점수 비례, 색상: RdYlGn 스케일 (빨강→초록)
  - 호버 툴팁: 종목명/코드/섹터/패턴/점수/시그널
  - 섹터 라벨 배지: 섹터명 + 평균점수 + (섹터점수)
  - clipPath로 텍스트 오버플로 방지, 동적 글씨 크기
  - `st.components.v1.html()` + viewBox 기반 반응형 (1400×800)

**파일** (2개):
```
app/pages/2_🔍_패턴분석.py   (사이드바 정렬/방향 + 7탭 + 필터 로직)
app/utils/charts.py          (섹터 차트 4개 + Treemap HTML 신규 함수)
```

**테스트**: 258개 (100% 통과, 변경 없음)

---

### 2026-02-24 (normalizer 조건부 Z-Score 통일 + 종목상세 기본값 변경)

**목표**: 종목 상세 페이지의 Z-Score가 히트맵/패턴 Z-Score와 다른 공식으로 계산되는 문제 해결

**배경 — 두 가지 Z-Score 계산 경로**:
- `normalizer.calculate_zscore()` — 종목 상세 KPI, 이상 수급, Z-Score 추이 차트
- `performance_optimizer._calculate_zscore_vectorized()` — 히트맵, 패턴 분류, 백테스트
- 후자는 이미 조건부 Z-Score 적용 완료, 전자는 표준 `(X-μ)/σ` 사용 중이었음
- 결과: 동일 종목의 Z-Score가 페이지마다 다를 수 있음

**구현 내용**:

- ✅ **`normalizer.py` — `calculate_zscore()` 조건부 Z-Score 적용**
  - 기존: `df_stock[zscore_col] = (df_stock[col] - rolling_mean) / rolling_std`
  - 변경: `same_sign = (today * mean) > 0` 판단 후 조건부 분기
  - `performance_optimizer`와 동일한 공식 → 모든 경로에서 Z-Score 통일
  - 자동 영향: 종목 상세 KPI, Z-Score 추이 차트, 수급 금액 테이블 Z 컬럼, 이상 수급 탐지

- ✅ **`5_📋_종목상세.py` — Z-Score 기준 기간 기본값 60→50**
  - 히트맵의 50D 기간과 직접 비교 용이하도록 변경

**검증 결과** (345개 전종목, institution_weight=0, window=50):
- KPI `foreign_zscore` vs 패턴현황 `50D`: **345개 전종목 차이 0** (완벽 일치)
- `institution_weight=0`이면 `combined_sff = foreign_sff` → 종합Z = 외국인Z (동일)
- 기관가중치 ≠ 0이면 종합Z ≠ 외국인Z (의도된 차이 — 비교 대상이 다름)

**Z-Score 공식 적용 현황 (최종)**:

| 모듈 | 함수 | 조건부 Z | 용도 |
|------|------|---------|------|
| `normalizer.py` | `calculate_zscore()` | ✅ 적용 | 종목 상세 KPI, 이상 수급 |
| `performance_optimizer.py` | `_calculate_zscore_vectorized()` | ✅ 기존 | 히트맵, 패턴 분류 |
| `precomputer.py` | `_compute_multi_period_zscores_all_dates()` | ✅ 기존 | 백테스트 |

**tanh 방향 확신도는 변경 불필요** — Z-Score 공식(값 자체)과 방향 확신도(정렬/분류 가중치)는 별개 레이어:
- Z-Score: `(X-μ)/σ` or `X/σ` → 편차 측정 (이번에 통일)
- tanh: `tanh(sff/std)` → 정렬/분류 시 adjusted_z 가중치 (기존 그대로)

**파일** (2개):
```
src/analyzer/normalizer.py          (calculate_zscore 조건부 Z-Score 적용)
app/pages/5_📋_종목상세.py          (z_score_window 기본값 60→50)
```

**테스트**: 258개 (100% 통과)

---

### 2026-02-24 (Z-Score 기간 라벨 통일 + 방향 확신도 구현)

**목표**: Z-Score 기간명을 영업일 기반으로 통일 + 매도 종목 양수 Z-Score 오분류 방지

**배경 — 현대모비스 문제**:
- 현대모비스(012330): 장기 매도세 지속 중, 최근 매도세 소폭 완화
- Z-Score는 "편차"를 측정 → 매도 완화 = 양의 편차 → 양수 Z-Score 부여
- 결과: 모멘텀 정렬 시 매수 상위 #1에 매도 종목 노출 (전체 top 10이 모두 매도 종목)
- 조건부 Z-Score의 `(today * mean) > 0` 이진 판단도 zero 근처 불연속성 문제

**구현 내용**:

- ✅ **Z-Score 기간 라벨 통일** (전 모듈)
  - 기존: `1W, 1M, 3M, 6M, 1Y, 2Y` (일반적 기간명)
  - 변경: `5D, 10D, 20D, 50D, 100D, 200D, 500D` (영업일 기준 명확화)
  - 7개 기간으로 확장 (1D 제거, 500D 추가)

- ✅ **방향 확신도(Direction Confidence)** — 자기 정규화 솔루션
  - **공식**: `confidence = tanh(today_sff / rolling_std)`
  - **특성**: tanh(0)=0(노이즈 감쇠), tanh(1)≈0.76(중간), tanh(2)≈0.96(강한 신호 보존)
  - **하드코딩 상수 없음** — 각 종목 자신의 변동성(rolling_std)으로 정규화
  - **smooth transition** — 이진 판단 대신 연속적 가중치 (zero 근처 불연속성 해결)

- ✅ **performance_optimizer.py**: sff 메타데이터 반환
  - `_calculate_zscore_vectorized(return_metadata=True)`: zscore + combined_sff + rolling_std 반환
  - `calculate_multi_period_zscores()`: `_today_sff`, `_std_5D`, `_std_10D` 등 메타컬럼 추가
  - 기존 API 하위 호환 유지 (return_metadata=False 시 Series 반환)

- ✅ **pattern_classifier.py**: 패턴 분류에 방향 확신도 적용
  - `_apply_direction_confidence(df, direction)` static 메서드 추가
  - long: `adjusted_z = z × max(confidence, 0)` — 매수 방향 확신도만 반영
  - short: `adjusted_z = z × max(-confidence, 0)` — 매도 방향 확신도만 반영
  - 정렬키/특성치/패턴/점수는 보정값으로 계산, 출력 Z-Score는 원본 복원

- ✅ **charts.py**: 히트맵 정렬에 방향 확신도 적용
  - `_adj_z(col, dir_mode)`: 방향별 보정 Z-Score 계산
  - `_compute_sort_key(dir_mode)`: 4가지 정렬 모드에 보정값 적용
  - 'both' 모드: buy/sell 독립 정렬 후 병합
  - 메타데이터 컬럼(`_` prefix) 히트맵 렌더링 전 제거

- ✅ **히트맵.py**: period_cols 필터에 `_` prefix 제외

**검증 결과** (338개 종목 분석):
- 현대모비스: momentum +2.03 → 0.00, 패턴 모멘텀형 → 기타, 순위 #1 → #221
- 매도 종목 76개(22.5%)의 양수 Z-Score 오분류 모두 해결
- 모멘텀 정렬 상위: 매도 종목 → 진정한 매수 종목 (KB금융, 넥슨게임즈 등)
- 진정한 전환 종목: 57개 중 44개(77%) 보존 (confidence > 0.3)
- 노이즈 레벨 종목: |sff/std| < 0.1 → 0.034x 감쇠 (효과적 필터링)

**파일** (5개):
```
src/visualizer/performance_optimizer.py  (sff 메타데이터 반환)
src/analyzer/pattern_classifier.py       (_apply_direction_confidence 추가)
app/utils/charts.py                      (히트맵 정렬 보정)
app/pages/1_📊_히트맵.py                 (period_cols 필터)
tests/test_performance_optimizer.py      (메타데이터 컬럼 테스트 업데이트)
```

**테스트**: 258개 (100% 통과)

---

### 2026-02-24 (코드 리뷰 — Precomputer 방향 확신도 누락 수정)

**목표**: 방향 확신도 구현 후 전체 코드 리뷰 → 4가지 이슈 발견 및 수정

**발견 이슈 및 수정**:

- ✅ **[버그] Precomputer 방향 확신도 미적용** (`precomputer.py`)
  - **원인**: `_compute_multi_period_zscores_all_dates()`에 `_today_sff`/`_std_*D` 메타데이터 미포함
  - `classify_all()` → `_apply_direction_confidence()`에서 `'_today_sff' not in df.columns` → 조기 반환 → 방향 확신도 미적용
  - **수정**: `_today_sff` (= `combined_sff`) + 각 기간별 `_std_{period}` (= `rolling_std`) 컬럼을 결과 DataFrame에 추가
  - `_compute_patterns_all_dates()`에 메타컬럼 전파 주석 추가

- ✅ **[확인] charts.py tanh 중복 계산** — 수정 불필요
  - `charts.py`(히트맵 정렬)와 `pattern_classifier.py`(패턴 분류)에서 동일한 tanh 계산
  - 서로 다른 컨텍스트(정렬 vs 분류)에서 독립 사용 → 공유 함수화 시 커플링 증가로 오히려 불리

- ✅ **[주석] heatmap_generator.py "8개 기간" → "7개 기간"** 수정

- ✅ **[일관성] performance_optimizer.py 병렬 실패 반환 타입 통일**
  - 기존: 성공 시 `pd.DataFrame`, 실패 시 `pd.Series(dtype=float)` → 타입 불일치
  - 수정: 실패 시에도 `pd.DataFrame(columns=['zscore', 'combined_sff', 'rolling_std'])` 반환

**파일** (3개):
```
src/backtesting/precomputer.py          (메타데이터 컬럼 추가 — 방향 확신도 전파)
src/visualizer/performance_optimizer.py (병렬 실패 반환 타입 통일)
scripts/analysis/heatmap_generator.py   (주석 수정 8개→7개)
```

**테스트**: 258개 (100% 통과)

---

### 2026-02-24 (히트맵 인터랙티브 고도화 A/B/C/D)

**목표**: 히트맵 페이지 4가지 고도화 기능 구현

**구현 내용**:

- ✅ **A. 히트맵 클릭 → 하단 미니 상세 패널** (`1_📊_히트맵.py`)
  - `st.plotly_chart(on_select="rerun", selection_mode="points")` 활용
  - 클릭 시 y_label에서 종목코드 추출 → KPI 4개 (1W Z-Score, 패턴, 점수, 시그널)
  - 멀티기간 Z-Score 바차트 (`create_multiperiod_zscore_bar`)
  - "종목 상세 보기 →" 버튼 (`st.switch_page` + `session_state['heatmap_selected_code']`)

- ✅ **B. 호버에 패턴/점수/시그널 정보 표시** (`charts.py`)
  - `create_zscore_heatmap()`에 `report_df` 파라미터 추가
  - 3D numpy customdata 배열: `np.empty((n_stocks, n_periods, 3), dtype=object)`
  - 각 셀별 패턴/점수/시그널 수 표시 (hovertemplate 확장)

- ✅ **C. 필터 사이드바** (`1_📊_히트맵.py`)
  - 패턴 필터 (전체/모멘텀형/지속형/전환형/기타)
  - 최소 점수 슬라이더 (0~100)
  - 최소 시그널 수 슬라이더 (0~3)
  - 필터 적용: zscore_matrix + report_df 동시 필터링

- ✅ **D. 섹터 평균 히트맵 탭** (`charts.py`)
  - `create_sector_zscore_heatmap(zscore_matrix, stock_list, sort_by)` 신규 함수
  - 섹터별 종목들의 평균 Z-Score 히트맵 (현재 필터 반영)
  - 탭 구조: "📈 종목별 히트맵" | "🏭 섹터 평균 히트맵"

- ✅ **종목 상세 페이지 히트맵 연동** (`5_📋_종목상세.py`)
  - `session_state['heatmap_selected_code']` → 종목 선택 드롭다운 자동 매칭

**파일** (3개):
```
app/pages/1_📊_히트맵.py     (A/C: 클릭 미니상세 + 필터 사이드바 + 탭 구조)
app/utils/charts.py          (B/D: 호버 customdata + 섹터 평균 히트맵)
app/pages/5_📋_종목상세.py   (히트맵→상세 네비게이션 연동)
```

**테스트**: 258개 (100% 통과, 변경 없음)

---

### 2026-02-23 (전 페이지 기관가중치 설명 + MA 멀티셀렉트 + 패턴분석 포맷)

**목표**: 기관가중치 도움말 전면 개선, MA 차트 멀티셀렉트 + 크로스 정확도 개선, 패턴분석 점수 소수점 포맷

**구현 내용**:

- ✅ **기관 가중치 help 텍스트 전 페이지 업데이트** (5개 페이지)
  - 기존: 한 줄 요약 (`"기관 수급 반영 비율 (0=외국인만, 0.3=기본, 1.0=동등)"`)
  - 변경: `[로직]` / `[반대방향 무시 이유]` / `[값별 의미]` 3섹션 상세 설명
  - 반대방향 무시 이유: 외국인 강매수 신호가 기관 역매매로 희석/반전되는 사례 명시
  - 적용: `streamlit_app.py`, `1_📊_히트맵.py`, `2_🔍_패턴분석.py`, `3_📈_백테스트.py`, `5_📋_종목상세.py`

- ✅ **MA 차트 멀티셀렉트 + 동조율 제거** (`시그널 & MA 탭`, `charts.py`)
  - `create_signal_ma_chart(df, start_date, ma_periods)`: `ma_periods` 파라미터 추가
  - `_MA_COLORS` 팔레트: `{5:sky, 10:cyan, 20:slate, 60:violet, 120:amber, 240:orange}`
  - 사용자가 MA 기간 자유롭게 선택/해제 (`st.multiselect`, 기본 [5, 20])
  - 2개 선택 시에만 골든/데드크로스 표시, 그 외엔 안내 캡션
  - 동조율(`sync_rate`) 차트 및 메트릭에서 완전 제거 (4열 → 3열: 골든/데드/가속도)
  - `make_subplots` 제거 → `go.Figure()` 단순화 (secondary y축 불필요)

- ✅ **골든/데드크로스 위치 선형 보간 적용** (`charts.py`)
  - 기존: 크로스 발생일 MA 값 → 마커 y축이 교차점과 어긋남
  - 변경: 전일/당일 두 점 사이 선형 보간 → 정확한 교차점 좌표 계산
    ```python
    t = diff_p / (diff_p - diff_c)
    x_cross = p_date + Timedelta(t * (c_date - p_date))
    y_cross = (p_short + t * (c_short - p_short)) / 1e8
    ```
  - 마커 크기: 13 → 9 (시각적 정돈)

- ✅ **수급 금액 차트 범례/제목 간격 조정** (`charts.py`)
  - `margin(t=90→120)`, `legend y=1.05→1.07` — 범례와 차트 제목 겹침 해소

- ✅ **패턴분석 종목 리스트 점수 소수점 표시** (`2_🔍_패턴분석.py`)
  - 패턴 점수 / 최종 점수: `"%.0f"` → `"%.1f"` (소수점 첫째자리 표시)

- ✅ **Stage 3 HTML 리포트 생성** (`output/regime_report_full.html`)
  - `scripts/analysis/generate_html_report.py` 활용
  - 345개 전체 종목 기준 인터랙티브 HTML (58KB)
  - 패턴 도넛 차트 + 섹터 바차트 + TOP 20 테이블 + D3.js Treemap

**파일** (6개):
```
app/streamlit_app.py               (기관가중치 help 업데이트)
app/pages/1_📊_히트맵.py           (기관가중치 help 업데이트)
app/pages/2_🔍_패턴분석.py         (기관가중치 help 업데이트 + 점수 %.1f)
app/pages/3_📈_백테스트.py         (기관가중치 help 업데이트)
app/pages/5_📋_종목상세.py         (기관가중치 help + MA 멀티셀렉트 + 동조율 제거 + Z-Score 기간 캡션)
app/utils/charts.py                (MA 멀티셀렉트 + 선형보간 크로스 + 마커9 + 범례간격)
```

---

### 2026-02-23 (종목 상세 페이지 UI 개선)

**목표**: 수급 금액 탭 테이블/차트 심층 개선, MA 크로스 개선, 사이드바 재정렬

**구현 내용**:

- ✅ **로딩 progress bar** — 다른 페이지와 동일하게 단계별 % 표시 추가

- ✅ **MA 크로스 개선** (`시그널 & MA 탭`)
  - 이벤트 기반(발생일만 활성) → **상태 기반** (MA5 > MA20 유지 중 = 골든 활성)
  - 데드크로스 추가: 차트에 ▽ 빨간 마커 + 하단 메트릭 "골든크로스 활성/비활성" & "데드크로스 활성/비활성" 분리 표시

- ✅ **수급 금액 차트 개선** (`create_supply_amount_chart`)
  - 누적순매수 라인 추가 (보조 y축, 표시 기간 시작일 기준)
  - 개인 순매수 3번째 서브플롯 추가 (개인 = -(외국인+기관) 근사)
  - 외국인/기관 x축 날짜 표시 추가 (기존: 기관만)
  - 범례-차트 제목 겹침 수정: `margin(t=90)`, `legend y=1.05`

- ✅ **수급 금액 테이블 신규** (차트 하단 메트릭 카드 → HTML 테이블)
  - 컬럼: 날짜 / 외국인 순매수·Z·누적 / 기관 순매수·Z·누적 / 개인 순매수·누적
  - 종합 Z 제외 (외국인 Z, 기관 Z만)
  - 금액: 억원 단위, 1의 자리, 쉼표 포맷 (양수 🟢, 음수 🔴)
  - Z-Score 색상: **황금(#fbbf24) ≥+2σ** / **하늘(#7dd3fc) ≤-2σ** / 회색 중립 (순매수 초록/빨강과 구분)
  - 그룹 2행 헤더: 외국인(sky) / 기관(pink) / 개인(orange) — 불투명 배경 + 컬러 하단선 + 좌측 구분선
  - Sticky 헤더 수정: 반투명 rgba → 불투명 단색, `z-index:10`

- ✅ **Z-Score 기준 기간 슬라이더 추가** (사이드바)
  - 범위: 20~1,300 거래일 (최대 5년)
  - 실제 데이터 수 초과 시 자동 캡 + 사이드바 캡션 표시

- ✅ **사이드바 순서 재정렬**
  - 변경 전: 기관가중치 → 기준날짜 → 표시기간 → Z-Score기간 → 종목선택
  - 변경 후: **종목선택** → 기준날짜 → 표시기간 → (구분선) → Z-Score기간 → **기관가중치**

**파일** (3개):
```
app/pages/5_📋_종목상세.py   (progress bar, MA 개선, 테이블 신규, 슬라이더, 사이드바 재정렬)
app/utils/charts.py          (create_supply_amount_chart: 개인 추가, 누적라인, 범례 간격)
app/utils/data_loader.py     (get_stock_zscore_history에 z_score_window 파라미터 추가)
```

---

### 2026-02-23 (종목 상세 페이지 구현)

**목표**: Stage 5-1 종목 상세 페이지 신규 구현 — 단일 종목 수급 심층 분석

**구현 내용**:

- ✅ **`app/pages/5_📋_종목상세.py` 신규 생성**
  - 사이드바: 기관 가중치(글로벌), 기준 날짜, 표시 기간(3M/6M/1Y/전체), 종목 선택
  - 종목 선택: 전체 종목 리스트 (1,609개) — 입력 검색 지원
  - KPI 5개: Z-Score 종합 / Z-Score 외국인 / Z-Score 기관 / 현재가 / 활성 시그널
  - 패턴 배너: 현재 패턴/점수/시그널 컬러 표시
  - **4탭 구조**:
    - 📈 Z-Score 추이: foreign/institution/combined 3선 + ±2σ 기준선
    - 💰 수급 금액: 외국인/기관 순매수금액 바차트 (2행 서브플롯, 억 단위) + 누적 요약
    - 🔔 시그널 & MA: 외국인 MA5/MA20 + 동조율(보조축) + 골든크로스 마커 + 현재 시그널 메트릭 3개
    - 📊 패턴 현황: 6기간 Z-Score 바차트 + 진입/손절 + 시그널 목록 + Z-Score 수치 테이블

- ✅ **`app/utils/data_loader.py` 함수 2개 추가**
  - `get_stock_zscore_history(stock_code, end_date, institution_weight)`: 단일 종목 Z-Score 시계열
  - `get_stock_raw_history(stock_code, end_date)`: 원시 수급+가격 이력 + ma5/ma20/sync_rate 파생 지표

- ✅ **`app/utils/charts.py` 함수 4개 추가** (`from plotly.subplots import make_subplots` 추가)
  - `create_zscore_history_chart(df, start_date)`: 3선 라인 차트 + ±2σ
  - `create_supply_amount_chart(df, start_date)`: 2행 서브플롯 바차트
  - `create_signal_ma_chart(df, start_date)`: MA + 보조축 동조율 + 골든크로스 마커
  - `create_multiperiod_zscore_bar(zscore_row)`: 6기간 Z-Score 바차트

- ✅ **`.streamlit/pages.toml` 메뉴 추가**
  - "종목 상세" `:material/bar_chart:`

**파일** (4개):
```
app/pages/5_📋_종목상세.py        (신규 — 종목 상세 페이지)
app/utils/data_loader.py          (get_stock_zscore_history, get_stock_raw_history 추가)
app/utils/charts.py               (차트 4종 + make_subplots import 추가)
.streamlit/pages.toml             (종목 상세 메뉴 추가)
```

**테스트**: 258개 (100% 통과, 변경 없음 — 새 함수 기본값이 기존과 동일)

---

### 2026-02-23 (메인 페이지 브랜딩 + 이상 수급 날짜/Z-Score 기간 조정)

**목표**: 메인 페이지 타이틀 변경, 이상 수급 탭에서 과거 날짜 조회 + Z-Score 이동평균 기간 사용자 조정

**구현 내용**:

- ✅ **메인 페이지 브랜딩** (`streamlit_app.py`)
  - 타이틀/탭 제목: "수급 분석 대시보드" → "Whale Supply"
  - `.streamlit/pages.toml` 신규: 사이드바 메뉴명 커스터마이즈 (Material icons 적용)

- ✅ **이상 수급 날짜 선택기** (`streamlit_app.py`)
  - 사이드바에 "이상 수급 기준일" date_input 추가
  - `get_abnormal_supply_data(end_date=end_date_str)` 전달
  - 이상 수급 탭에만 영향, 다른 페이지/섹션 무관

- ✅ **Z-Score 기준 기간 조정** (`streamlit_app.py`, `data_loader.py`)
  - 사이드바에 슬라이더 추가 (20~240거래일, 기본 60, step 10)
  - `get_abnormal_supply_data(z_score_window=z_score_window)` 전달
  - `data_loader.py`: `get_abnormal_supply_data()`에 `z_score_window=60` 파라미터 추가

- ✅ **산출 방식 설명 동적화** (`streamlit_app.py`)
  - 하드코딩된 "60거래일", "30%", "0.3" → 사이드바 현재값으로 동적 표시
  - "사이드바에서 조정 가능한 파라미터" 안내 추가 (기관 가중치, Z-Score 기준 기간)

- ✅ **사이드바 위젯 순서 변경**: 이상 수급 기준일 → 기관 가중치 → Z-Score 기준 기간

**파일** (3개):
```
app/streamlit_app.py        (브랜딩 + 날짜 선택 + Z-Score 기간 슬라이더 + 산출 방식 동적화)
app/utils/data_loader.py    (get_abnormal_supply_data에 z_score_window 파라미터 추가)
.streamlit/pages.toml       (신규 — 사이드바 메뉴명/아이콘 커스터마이즈)
```

---

### 2026-02-23 (institution_weight 글로벌 사이드바 + 거래 비용 파라미터화)

**목표**: institution_weight를 전 페이지 글로벌 사이드바로 이동 + 백테스트 거래 비용(세금/수수료/슬리피지/차입비용) 사용자 조정 가능하도록 파라미터화

**구현 내용**:

- ✅ **`portfolio.py` 거래비용 파라미터화**
  - 클래스 속성 4개(`TAX_RATE`, `COMMISSION_RATE`, `SLIPPAGE_RATE`, `BORROWING_RATE`) → `__init__` 파라미터로 전환
  - 기본값 동일 유지 → 기존 테스트 하위 호환
  - 6곳 참조 업데이트: `self.TAX_RATE` → `self.tax_rate` 등

- ✅ **`engine.py` BacktestConfig → Portfolio 전달 체인**
  - `BacktestConfig`에 `tax_rate`, `commission_rate`, `slippage_rate`, `borrowing_rate` 4개 필드 추가
  - `BacktestEngine.__init__`에서 Portfolio 생성 시 4개 비용 파라미터 전달

- ✅ **`data_loader.py` 분석 파이프라인 institution_weight 파라미터 추가**
  - 7개 함수에 `institution_weight=0.3` 파라미터 추가:
    `_stage_zscore`, `_stage_classify`, `_stage_signals`, `_stage_report`,
    `run_analysis_pipeline`, `run_analysis_pipeline_with_progress`, `get_abnormal_supply_data`
  - `SupplyNormalizer` 생성 시 완전한 config dict 전달 (`z_score_window`, `min_data_points`, `institution_weight`)
  - 백테스트 3개 함수에 거래비용 4개 파라미터 추가:
    `run_backtest`, `run_backtest_with_progress`, `run_optuna_optimization`

- ✅ **전 페이지 글로벌 사이드바 위젯**
  - 4개 페이지에 동일한 슬라이더 추가 (`key="w_institution_weight"` → session_state 공유)
  - 파이프라인 호출에 `institution_weight=institution_weight` 전달
  - 적용: `streamlit_app.py`, `1_📊_히트맵.py`, `2_🔍_패턴분석.py`, `3_📈_백테스트.py`

- ✅ **백테스트 거래비용 UI** (`3_📈_백테스트.py`)
  - 🔒 고정 조건 섹션에 "거래 비용" expander 추가
  - 4개 `number_input`: 증권거래세(%), 수수료(%), 슬리피지(%), 차입비용(%/연)
  - 3개 호출부에 비용 파라미터 전달: `run_backtest`, `run_optuna_optimization`, `run_backtest_with_progress`

- ✅ **[버그수정] optimizer/walk_forward 거래비용 누락**
  - `optimizer.py` `_build_base_params()`: 거래비용 4개 필드 누락 → 추가
  - `walk_forward.py` `_extract_params_from_row()`: 동일 누락 → 추가
  - `walk_forward.py` `_build_base_config_dict()`: 동일 누락 → 추가
  - **영향**: 수정 전에는 UI에서 비용 변경해도 최적화/워크포워드 시 항상 기본값 사용

**버그 수정 이력**:
- `KeyError: 'z_score_window'`: `SupplyNormalizer`에 부분 config dict 전달 → 완전한 config dict로 수정
- optimizer/walk_forward params dict에 거래비용 필드 누락 → 3곳 추가

**파일** (10개):
```
src/backtesting/portfolio.py       (클래스 속성 → __init__ 파라미터)
src/backtesting/engine.py          (BacktestConfig 비용 필드 + Portfolio 전달)
src/backtesting/optimizer.py       (_build_base_params 비용 필드 추가)
src/backtesting/walk_forward.py    (_extract_params_from_row, _build_base_config_dict 비용 필드 추가)
app/utils/data_loader.py           (institution_weight 전 파이프라인 + 백테스트 비용 파라미터)
app/streamlit_app.py               (글로벌 사이드바 위젯 + 파이프라인 전달)
app/pages/1_📊_히트맵.py           (글로벌 사이드바 위젯 + 파이프라인 전달)
app/pages/2_🔍_패턴분석.py         (글로벌 사이드바 위젯 + 파이프라인 전달)
app/pages/3_📈_백테스트.py         (거래비용 UI expander + 3개 호출부 전달)
app/utils/charts.py                (미사용 import 정리)
```

**테스트**: 258개 (100% 통과, 변경 없음 — 모든 새 파라미터 기본값이 기존과 동일)

---

### 2026-02-22 (메인 페이지 재설계 + 이상 수급 + 당일 수급 순위)

**목표**: 메인 페이지 정보량 확대, 이상 수급(Z>2σ) 표시, 당일 순매수/순매도 금액 순위 추가

**구현 내용**:

- ✅ **메인 페이지 전면 재설계** (`app/streamlit_app.py`)
  - KPI 카드 4개→5개: 분석 종목 / 관심 종목 / 강한 매수 / 강한 매도 / 시그널 2+
  - 탭 구조 도입: "이상 수급 (Z-Score > 2σ)" | "당일 수급 순위"
  - 패턴 분석 요약 (파이차트 + 히스토그램) 유지
  - 관심 종목 테이블 `column_config` 포맷 개선 (ProgressColumn, NumberColumn 등)
  - CSS `:has()` 선택자로 매수(green)/매도(red) 컨테이너 테두리 구분

- ✅ **탭 1: 이상 수급 (Z-Score > 2σ)**
  - 매수/매도 2열 레이아웃: 바차트 top 10 + 테이블 30개
  - 바차트 호버에 외국인/기관/종합 Z-Score + 순매수금액(억/조 포맷) 표시
  - "산출 방식 보기" expander: Sff→합산→Z-Score 3단계 수식 + 외국인Z≠종합Z 설명

- ✅ **탭 2: 당일 수급 순위** (NEW)
  - 외국인/기관 순매수 상위 + 순매도 상위 (4섹션)
  - 각 섹션: 바차트 top 10 (억/조 포맷) + 테이블 top 50 (쉼표 포맷 원시금액)
  - 캡션: "원시 금액 기준, 정규화 미적용"

- ✅ **`data_loader.py` 함수 2개 추가**
  - `get_abnormal_supply_data()`: 이상 수급 캐시 래퍼 (순매수금액 조인 포함)
  - `get_today_supply_ranking()`: 당일 전 종목 외국인/기관 순매수금액 조회

- ✅ **`charts.py` 함수 2개 + 헬퍼 추가**
  - `_fmt_amount()`: 원→억/조 변환 (부호+쉼표, 1조 이상 시 조 표기)
  - `create_abnormal_supply_chart()`: 이상 수급 수평 바차트 (Z-Score 기준)
  - `create_supply_ranking_chart()`: 수급 순위 수평 바차트 (금액 기준)

**파일**:
```
app/streamlit_app.py     (메인 페이지 전면 재설계)
app/utils/data_loader.py (get_abnormal_supply_data, get_today_supply_ranking 추가)
app/utils/charts.py      (_fmt_amount, create_abnormal_supply_chart, create_supply_ranking_chart 추가)
```

---

### 2026-02-20 (Streamlit 다크 테마 + UI 개선 + 차트 개선 + 거래내역 성과 컬럼)

**목표**: 차트/앱 전반 다크 테마 적용, 슬라이더+직접입력 동기화, 진행률 표시 개선, 월별수익률 재설계, 거래내역 기간 내 성과 추가

**구현 내용**:

- ✅ **2026년 날짜 선택 버그 수정**
  - 모든 페이지 `date_input`의 `max_value`를 `_max_dt.replace(month=12, day=31)`로 연장
  - 영향 파일: `3_📈_백테스트.py` (5곳), `1_📊_히트맵.py`, `2_🔍_패턴분석.py`

- ✅ **슬라이더 + 직접 입력 동기화** (`3_📈_백테스트.py`)
  - `_synced_slider()` 헬퍼 함수 추가: 슬라이더(3열) + number_input(1열) 나란히 배치
  - `on_change` 콜백으로 양방향 동기화 (슬라이더 조작 시 입력창 갱신, 입력창 변경 시 슬라이더 갱신)
  - Optuna 최적 파라미터 반영 시 `_ni` 키도 함께 업데이트
  - 슬라이더 범위 확장: 목표수익률 1~200%, 손절 -100~-1%

- ✅ **사이드바 레이아웃 개선** (`3_📈_백테스트.py`)
  - CSS: 사이드바 너비 340px, 슬라이더-입력창 `align-items: flex-end` 수직 정렬

- ✅ **다크 테마 전면 적용**
  - **`.streamlit/config.toml` 신규 생성**: 앱 전체 다크 테마 (Tailwind Slate 팔레트)
    ```toml
    [theme]
    base = "dark"
    primaryColor = "#38bdf8"        # sky-400
    backgroundColor = "#0f172a"     # slate-900
    secondaryBackgroundColor = "#1e293b"  # slate-800
    textColor = "#e2e8f0"           # slate-200
    ```
  - **`src/backtesting/plotly_visualizer.py`**: `_apply_theme()` static 메서드 추가
    - 다크 상수: `_BG_PLOT='#0f172a'`, `_BG_PAPER='#1e293b'`, `_GRID='#334155'`
    - `template='plotly_dark'` + 명시적 배경색 + 그리드/축 색상 통일
    - 커스텀 컬러스케일 (히트맵): red-500 → slate-700 → green-500
    - 히트맵 셀 텍스트: `color='#0f172a'` (검은색)
  - **`app/utils/charts.py`**: `_apply_dark()` 함수 추가 (동일 다크 상수 체계)
  - **모든 `st.plotly_chart()` 호출에 `theme=None` 추가** (10곳)
    - **근본 원인**: `st.plotly_chart`의 기본값 `theme="streamlit"`이 명시적 `plot_bgcolor`/`paper_bgcolor`를 덮어쓰는 것이 문제였음
    - 영향 파일: `3_📈_백테스트.py`(5곳), `1_📊_히트맵.py`, `2_🔍_패턴분석.py`, `4_🔄_워크포워드.py`, `streamlit_app.py`(2곳)
  - **`fileWatcherType = "poll"` 추가** (`.streamlit/config.toml`)
    - WSL 환경에서 inotify가 `src/` 하위 파일 변경을 감지 못하는 문제 해결
    - 폴링 방식으로 전환 → 코드 저장 시 자동 리로드 정상 동작

- ✅ **분석 진행률 표시 (% 포함)** — CacheReplayClosureError 해결
  - **원인**: `@st.cache_data` 함수 내부에서 `st.progress` 업데이트 시도 → Streamlit이 캐시 재생 시 외부 위젯 호출 실패
  - **해결**: 단일 캐시 함수를 4개 스테이지별 캐시 함수로 분리 (`_stage_zscore`, `_stage_classify`, `_stage_signals`, `_stage_report`)
  - `run_analysis_pipeline_with_progress()`: 비캐시 래퍼에서 스테이지 사이에 `st.progress(pct, text=...)` 업데이트
  - 전 페이지 적용: `streamlit_app.py`, `1_📊_히트맵.py`, `2_🔍_패턴분석.py`

- ✅ **`use_container_width` → `width='stretch'` 전환** (Streamlit 1.54.0 deprecation 대응)
  - `st.plotly_chart` 호출 전부 교체 (10곳)
  - `st.dataframe` / `st.button`의 `use_container_width`는 해당 없음 (chart 전용 deprecation)

- ✅ **월별 수익률 차트 완전 재설계** (`src/backtesting/plotly_visualizer.py`)
  - **기존**: 연도×월 히트맵 (x축=월1~12, y축=연도) → 정렬 오류 + 가독성 불편
  - **변경**: 타임라인 바차트 (x축=연도·월 순서 라벨, y축=월별 수익률%)
  - 첫 달 수익률: `initial_capital` 기준으로 계산 (`pct_change()` NaN 문제 해결)
  - y축 범위 자동 패딩 (`pad = max(|max_r|, |min_r|) × 0.35`) → 바 위 텍스트 클리핑 방지
  - 바 텍스트 (소수점 2자리 `%{customdata}`) + 호버 텍스트 동기화
  - 글씨 크기 13px로 확대
  - groupby 방식: `dt.to_period('M')` → `year*100 + month` 정수 (WSL pandas 호환)

- ✅ **수익률 분포 차트 개선** (`src/backtesting/plotly_visualizer.py`)
  - 평균/중앙값 표시: `add_annotation()` → 범례(legend) 방식으로 변경
    - 차트 본문 텍스트가 바에 가려지는 문제 해결
  - 범례 위치: 차트 우측 → 차트 상단 가로 배치 (`orientation='h', y=1.02, x=0`)
  - dummy trace (`x=[None], y=[None]`) 활용 → vline 색상을 legend에 표시

- ✅ **히트맵 통계 위치 이동** (`app/pages/1_📊_히트맵.py`)
  - 기존: 히트맵 아래에 "표시 종목 수 / 평균 1W Z-Score / 강한 매수" 표시
  - 변경: 히트맵 **위**로 이동 → 차트 보기 전에 맥락 파악 가능

- ✅ **거래내역 기간 내 성과 컬럼 추가** (`app/pages/3_📈_백테스트.py`)
  - DB `close_price` 일괄 조회 (1회 쿼리, 전 종목 기간 포함)
  - 진입가 기준 일별 수익률 계산 → max_ret / min_ret / MDD
  - **컬럼 추가**: `max_ret (%)`, `min_ret (%)`, `MDD (%)` (소수점 2자리)
  - `st.column_config.NumberColumn` 포맷 적용

**버그 수정 이력**:
- `import numpy as np as _np` → `import numpy as _np` (문법 오류)
- 월별 수익률 차트 첫 달 누락: `pct_change()` 첫 행 NaN → `initial_capital`로 기준값 설정
- 월별 수익률 차트 전체 사라짐: `dt.to_period('M')` WSL 이슈 → `year*100+month` 정수 groupby로 교체

**파일 구조**:
```
.streamlit/config.toml                 (신규 — 앱 전체 다크 테마 + fileWatcherType=poll)
src/backtesting/plotly_visualizer.py   (_apply_theme, 다크 상수, 월별수익률 재설계, 수익률분포 개선)
app/utils/charts.py                    (_apply_dark, 다크 상수, 커스텀 컬러스케일)
app/utils/data_loader.py               (스테이지별 캐시 분리 + run_analysis_pipeline_with_progress)
app/streamlit_app.py                   (progress bar + theme=None + width='stretch')
app/pages/1_📊_히트맵.py               (progress bar + theme=None + 통계 위치 차트 위로)
app/pages/2_🔍_패턴분석.py             (progress bar + theme=None + width='stretch')
app/pages/3_📈_백테스트.py             (_synced_slider + 날짜 버그 수정 + theme=None × 5 + 거래내역 성과 컬럼)
app/pages/4_🔄_워크포워드.py           (theme=None + width='stretch')
```

---

### 2026-02-21 (백테스트 기간 종료 시 미청산 포지션 강제 청산 + 표시)

**목표**: 백테스트 종료일에 아직 청산되지 않은 포지션을 마지막 날 종가로 강제 청산 + UI 표시

**구현 내용**:
- ✅ **`data_loader.py`**: `force_exit_on_end=True` 적용
  - `run_backtest()`: 종료일 도달 시 미청산 포지션 마지막 날 종가로 강제 청산
  - `run_optuna_optimization()`: 최적화 시에도 동일 적용 (일관성 유지)
- ✅ **`3_📈_백테스트.py`**: 기간 종료 청산 종목 표시 섹션 추가
  - `exit_reason == 'end'` 거래 필터링
  - 접이식(expander)으로 종목명/코드/진입일/보유일/수익률/진입가/청산가 표시
  - "보유 기간이 짧아 전략 효과가 반영되지 않았을 수 있습니다" 안내

**파일**:
```
app/utils/data_loader.py (force_exit_on_end=True)
app/pages/3_📈_백테스트.py (기간 종료 청산 종목 expander)
```

---

### 2026-02-21 (Streamlit 백테스트 - 최적화/검증 기간 분리 UI 추가)

**목표**: 과적합 없는 신뢰도 높은 백테스트 결과를 위한 기간 분리 기능

**문제**: 기존 "최적 파라미터 찾기"는 같은 기간으로 최적화+백테스트 → 과적합 (수익률 과대평가)

**구현 내용**:
- ✅ **`3_📈_백테스트.py`**: 기간 분리 체크박스 추가
  - "최적화 / 검증 기간 분리" 체크박스 (기본: 비활성)
  - 활성화 시: 최적화 기간(파라미터 탐색) + 검증 기간(백테스트) 별도 입력
  - "최적 파라미터 찾기": Optuna를 최적화 기간으로 실행 → 검증 기간으로 백테스트
  - "백테스트 실행": 검증 기간으로 실행 (분리 여부 관계없이 val 기간 사용)
  - 결과 화면에 기간 배너 표시 (분리 시: 최적화 기간 / 검증 기간 명시)
  - Optuna 섹션 caption: 분리 시 두 기간 명시, 미분리 시 기존 텍스트

**사용 흐름**:
```
[기간 미분리 - 탐색용]
시작일~종료일 → "최적 파라미터 찾기" → 같은 기간 최적화+검증 (과적합, 탐색 목적)

[기간 분리 - 신뢰도 높은 검증]
최적화: 2025-01~2025-09 → Optuna 파라미터 탐색
검증:   2025-10~2026-01 → 위 파라미터로 백테스트 (미래 데이터, 신뢰 가능)
```

**파일**:
```
app/pages/3_📈_백테스트.py (기간 분리 UI + 기간 배너)
```

---

### 2026-02-21 (institution_weight 설계 개선: 버그 수정 + 파라미터 분리 + Precomputer 캐싱)

**배경**: institution_weight의 역할과 사용 방식에 대한 설계 리뷰 결과 3가지 문제 발견

**문제 1 - 버그**: 가속도 시그널의 institution_weight 하드코딩
- `signal_detector.py`의 `calculate_acceleration()`에서 0.3 고정 사용
- `precomputer.py`의 `_compute_signals_all_dates()`에서 0.3 고정 사용
- `normalizer.py`만 config 파라미터 사용 → 세 모듈 간 불일치

**문제 2 - 설계**: institution_weight를 Optuna 최적화 파라미터로 잘못 분류
- institution_weight는 "기관 동조를 어떻게 해석할지"를 결정하는 **분석 철학 파라미터**
- min_score, stop_loss 같은 **전략 파라미터**(언제 사고 팔지)와 성격이 다름
- 같은 Optuna 탐색 공간에 묶이는 것은 개념적으로 부적절

**문제 3 - 비효율**: institution_weight가 Optuna 파라미터이므로 Trial마다 Precomputer 재실행
- institution_weight가 고정되면 모든 Trial이 동일한 Precomputed 데이터 공유 가능

**구현 내용**:

- ✅ **단계 1: 버그 수정** (3개 파일)
  - `signal_detector.py`: `SignalDetector.__init__`에 `institution_weight=0.3` 파라미터 추가
    - `calculate_acceleration()`: `0.3` → `self.institution_weight`
  - `precomputer.py`: `_compute_signals_all_dates()`: `0.3` → `self.institution_weight`
  - `engine.py`: `SignalDetector(conn)` → `SignalDetector(conn, institution_weight=self.config.institution_weight)`
  - 이제 normalizer / precomputer / signal_detector 세 곳 모두 동일한 institution_weight 사용

- ✅ **단계 2: institution_weight 최적화 파라미터 분리**
  - `optimizer.py`: `DEFAULT_PARAM_SPACE`에서 `institution_weight` 제거
  - `BacktestConfig`의 고정 파라미터로만 유지 (기본값 0.3)
  - 검증 필요 시: 별도 독립 백테스트로 0.0/0.1/0.2/0.3/0.5 수동 비교

- ✅ **단계 3: Precomputer 공유 캐싱**
  - `engine.run()`: `precomputed=` 파라미터 추가 (외부 주입 지원)
  - `optimizer.py`: Phase 1/2 시작 전 `BacktestPrecomputer` 1회 실행 → 모든 Trial에 주입
  - `optimizer.py`: `BacktestPrecomputer` import 추가
  - 효과: 50 Trial × ~1초 → 1초(사전계산) + 50 Trial × ~0초 ≈ 1~2초

**테스트**: 2개 신규 (총 258개, 100% 통과)
- `test_institution_weight_passed_to_signal_detector`: SignalDetector에 weight 전달 확인
- `test_precomputer_called_once_per_optimize`: optimize() 1회당 Precomputer 1회 호출 확인
- 기존 `test_default_param_space_keys`: institution_weight 제외 반영
- 기존 optimizer mock 테스트: BacktestPrecomputer mock 추가

- ✅ **단계 4: Streamlit 페이지 institution_weight 참조 오류 수정**
  - `3_📈_백테스트.py`:
    - `pending_opt_params` 핸들러에서 `w_institution_weight` 세션 업데이트 라인 제거
    - 최적화 후 백테스트 실행 시 `institution_weight=params['institution_weight']` (KeyError) → `institution_weight=institution_weight` (사이드바 위젯 값) 수정
    - 최적화 결과 표시: institution_weight 항목 제거, 컬럼 5개 → 4개
    - Optuna 최적화 대상 caption: "기관 가중치" 제거
  - `4_🔄_워크포워드.py`:
    - `known_params`에서 `'institution_weight'` 제거

**파일 구조**:
```
src/analyzer/signal_detector.py      (institution_weight 파라미터 추가)
src/backtesting/precomputer.py       (0.3 하드코딩 → self.institution_weight)
src/backtesting/engine.py            (SignalDetector에 weight 전달, run()에 precomputed= 추가)
src/backtesting/optimizer.py         (institution_weight 제거, Precomputer 1회 공유 캐싱)
tests/backtesting/test_optimizer.py  (테스트 2개 추가, mock 업데이트)
app/pages/3_📈_백테스트.py           (institution_weight 참조 오류 4곳 수정)
app/pages/4_🔄_워크포워드.py         (known_params에서 institution_weight 제거)
```

---

### 2026-02-22 (최적화 파이프라인 전체 검증 + institution_weight 불일치 수정)

**목표**: Optuna 최적화 파이프라인 전체 재검증 및 발견된 버그 수정

**검증 결과** (9개 항목 통과):
1. ✅ 파라미터 샘플링 흐름: `_build_base_params()` → `trial.suggest_*()` 덮어쓰기 → `BacktestConfig(**params)`
2. ✅ 7개 파라미터 engine.py 사용 위치 확인 (진입 필터/익절/손절/시간/포지션한도/반대수급)
3. ✅ Precomputer 1회 생성 → 전 Trial 공유 (상태 누수 없음)
4. ✅ MedianPruner 절반 기간 평가 순서 정상
5. ✅ 최적화 후 검증 백테스트: 7개 파라미터 모두 `params[]`에서 사용
6. ✅ "백테스트 실행" 버튼: 사이드바 위젯 값 직접 사용
7. ✅ pending_opt_params 위젯 동기화 정상
8. ✅ 엣지 케이스: max_positions=1, max_hold_days=1, reverse_threshold=0/115 모두 유효
9. ✅ Persistent Study 누적 및 최고 Trial 선택 정상

**발견된 버그: institution_weight 최적화-검증 불일치**:
- **원인**: `run_optuna_optimization()`에 `institution_weight` 파라미터 없음
  → 최적화 시 항상 BacktestConfig 기본값 0.3 사용
  → 검증 백테스트는 사이드바 값(0.3 아닐 수 있음) 사용
- **영향**: 사용자가 기관 가중치를 0.3이 아닌 값으로 변경 시, 최적화와 검증의 Z-Score/시그널이 불일치
- **수정**: `data_loader.py`에 `institution_weight` 파라미터 추가 + `3_📈_백테스트.py`에서 사이드바 값 전달

**파일**:
```
app/utils/data_loader.py    (run_optuna_optimization에 institution_weight 파라미터 추가 + BacktestConfig 전달)
app/pages/3_📈_백테스트.py  (최적화 호출 시 institution_weight=institution_weight 전달)
```

---

### 2026-02-22 (백테스트 사이드바 구조 개선 + Optuna 파라미터 확장)

**목표**: 사용자가 어떤 파라미터가 Optuna 탐색 대상인지 직관적으로 파악할 수 있도록 사이드바 재구성 + 최적화 대상 파라미터 확장

**구현 내용**:

- ✅ **사이드바 섹션 재구성** (`3_📈_백테스트.py`)
  - 기존: 진입 조건 / 청산 조건 / 포트폴리오 (역할 기준 분류)
  - 변경: **🧪 최적화 대상 파라미터** / **🔒 고정 조건** (누가 결정하나 기준 분류)
  - 🧪 섹션: min_score, min_signals, target_return, stop_loss, max_positions, max_hold_days, reverse_threshold
    - 캡션: "'최적 파라미터 찾기' 실행 시 Optuna가 자동 결정. 수동 설정 후 '백테스트 실행'도 가능"
  - 🔒 섹션: initial_capital, institution_weight
    - 캡션: "최적화·백테스트 모두 이 값으로 고정"
  - institution_weight: 고급 설정 expander → 🔒 섹션으로 이동

- ✅ **Optuna 최적화 대상 파라미터 3개 추가** (`optimizer.py`)
  - `max_positions`: int, 1~50
  - `max_hold_days`: int, 1~500 (2년 거래일)
  - `reverse_signal_threshold`: float, 0~115 (최대 점수까지)
  - 총 4개 → 7개로 확장 (institution_weight는 Precomputer 공유 불가로 제외 유지)

- ✅ **[버그수정] 최적화 후 검증 백테스트에 최적 파라미터 미반영** (`3_📈_백테스트.py`)
  - **원인**: max_hold_days, max_positions, reverse_threshold를 최적화 대상에 추가했으나,
    "최적 파라미터 찾기" 후 자동 실행되는 검증 백테스트에서 사이드바 위젯 값을 그대로 사용
  - **수정**: `params['max_hold_days']`, `params['max_positions']`,
    `params['reverse_signal_threshold']`에서 Optuna 최적값을 사용하도록 변경
  - 영향: 이전에는 최적화 결과를 표시만 하고 실제 검증 백테스트에는 미적용

- ✅ **Trial 기본값 상향** (파라미터 7개 기준)
  - `data_loader.py`: 50 → 100
  - `3_📈_백테스트.py` 슬라이더 기본값: 50 → 100

- ✅ **pending_opt_params 핸들러 업데이트** (`3_📈_백테스트.py`)
  - 최적화 후 새 3개 파라미터도 슬라이더/입력창에 자동 반영

- ✅ **최적화 결과 표시 확장** (`3_📈_백테스트.py`)
  - 4개 메트릭 → 7개 (최대 포지션, 최대 보유일, 반대수급 청산 추가, 4+3 레이아웃)

- ✅ **초기 자본금 쉼표 포맷** (`3_📈_백테스트.py`)
  - `number_input` → `text_input` + `on_change` 콜백
  - 입력창 자체에 `10,000,000` 형식 표시 (Enter/포커스아웃 시 자동 포맷)

- ✅ **기존 실패 테스트 수정** (`test_plotly_visualizer.py`)
  - `test_heatmap_trace_type` → `test_bar_trace_type`
  - 월별 수익률 차트 히트맵→바차트 재설계 시 테스트 미갱신 건 수정

**파라미터 탐색 범위 (최종)**:

| 파라미터 | 범위 | 비고 |
|---------|------|------|
| min_score | 50~90 | 진입 최소 점수 |
| min_signals | 1~3 | 진입 최소 시그널 수 |
| target_return | 5~25% | 목표 수익률 |
| stop_loss | -15~-3% | 손절 비율 |
| max_positions | 1~50 | 최대 동시 포지션 |
| max_hold_days | 1~500 | 최대 보유 기간 (2년 거래일) |
| reverse_signal_threshold | 0~115 | 반대 수급 청산 점수 (최대 점수=115) |
| institution_weight | 고정 (0.3) | Precomputer 공유 불가로 제외 |

**파일**:
```
src/backtesting/optimizer.py          (DEFAULT_PARAM_SPACE 7개로 확장, 범위 설정)
app/utils/data_loader.py              (n_trials 기본값 50→100)
app/pages/3_📈_백테스트.py            (사이드바 재구성, pending_opt_params, 결과표시, 자본금 쉼표)
tests/backtesting/test_optimizer.py   (required_keys 7개로 업데이트)
tests/backtesting/test_plotly_visualizer.py (test_bar_trace_type으로 수정)
```

---

### 2026-02-22 (대시보드 버그 수정 & UI 개선)

**목표**: 날짜 선택 오류, 점수 표시 개선, 히트맵/패턴분석 날짜 필터 등 전반 버그 수정

**구현 내용**:

- ✅ **2026년 연도 선택 불가 수정** (3개 페이지)
  - 원인: `max_value=datetime(2026, 1, 20)` → Streamlit 캘린더가 2026년 나머지 달을 비활성화
  - 수정: `max_value=_max_dt.replace(month=12, day=31)` (해당 연도 말일로 확장)
  - 적용 파일: `1_📊_히트맵.py`, `2_🔍_패턴분석.py`, `3_📈_백테스트.py` (5개 date_input)

- ✅ **날짜 선택기 추가** (히트맵·패턴분석)
  - 사이드바 상단에 "기준 날짜" date_input 추가
  - `run_analysis_pipeline(end_date=end_date_str)` 연결 → 과거 시점 조회 가능

- ✅ **점수 3단 분리** (패턴분석·백테스트 거래내역)
  - `패턴점수 (score)` + `시그널 수 (signal_count)` + `최종점수 = 패턴점수 + 시그널수×5`
  - 최종점수만 ProgressColumn 적용 (최대 115점), 패턴점수는 숫자만 표시
  - 백테스트 거래내역: `trade.score`(final) 역산 → `pattern_score = score - signal_count * 5`

- ✅ **누적 Trial 캡션 DB 직접 조회**
  - 기존: `session_state`의 opt_result에서 읽음 → 평가 지표 변경 시 구 값 잔류
  - 수정: `get_optuna_trial_count(strategy, metric, ...)` DB 직접 읽기 → 항상 최신값

- ✅ **기타 버그 수정**
  - `/div` 텍스트 노출: HTML f-string을 단일 라인 문자열 연결로 재작성
  - `NameError: strategy is not defined`: `strategy` selectbox를 Optuna expander 위로 이동
  - 기본 평가지표 변경: `sharpe_ratio` → `total_return`
  - "기간 종료 시 청산" expander: 0개일 때도 항상 표시 (안내 캡션 포함)

**파일**:
```
app/pages/1_📊_히트맵.py         (날짜 선택기 추가, max_value 연말로 확장)
app/pages/2_🔍_패턴분석.py       (날짜 선택기 추가, 점수 3단 분리, max_value 확장)
app/pages/3_📈_백테스트.py       (점수 3단 분리, 각종 버그 수정, max_value 확장)
app/utils/data_loader.py         (get_optuna_trial_count 추가)
```

---

### 2026-02-20 (Streamlit 백테스트 UI 개선 + Persistent Optuna Study)

**목표**: 백테스트 결과 시각 구분 개선 + Optuna Trial 누적 저장으로 재현성 확보

**구현 내용**:

- ✅ **Persistent Optuna Study** (`src/backtesting/optimizer.py`)
  - `study_storage` 파라미터 추가: SQLite 파일에 Study 누적 저장
  - `_make_study_name(metric)`: 전략/기간/메트릭 기반 고유 Study 이름 (`opt__{strategy}__{sd}__{ed}__{metric}`)
  - `load_if_exists=True`: 동일 이름 Study 재실행 시 기존 Trial 위에 누적 추가
  - `reset=False` 기본 → 재실행할수록 최적값 단조 증가 (≥) 보장
  - 반환값에 `existing_before` 추가 (이번 실행 전 누적 Trial 수)

- ✅ **data_loader.py 수정**: `_OPTUNA_STORAGE` 상수 + `study_storage` 주입
  - `_OPTUNA_STORAGE = f"sqlite:///{_PROJECT_ROOT / 'data' / 'optuna_studies.db'}"`
  - `run_optuna_optimization()`: `reset_study` 파라미터 + `study_storage` 전달

- ✅ **백테스트 UI 개선** (`app/pages/3_📈_백테스트.py`)
  - **위젯 키 충돌 수정**: `_defaults` dict로 session_state 1회 초기화 → 슬라이더에서 `value=` 제거
  - **최적화 결과 스타일**: `st.container(border=True)` + 주황색 좌측 테두리 헤더 (`#ff9800`)
  - **검증 결과 스타일**: `st.container(border=True)` + 녹색 좌측 테두리 헤더 (`#00c853`)
  - **CSS `:has()` 선택자**: `stVerticalBlockBorderWrapper:has([style*="ff9800"])` 등으로 테두리 색상 자동 구분
  - **사이드바 3섹션 구조** (divider로 구분):
    - ① 기간 분리 설정 (최상단) - 언체크 시 과적합 경고 항상 표시
    - ② 파라미터 최적화 expander
    - ③ 백테스트 설정
  - **최적화 결과 레이아웃**: Sharpe / 완료 / 중단 (좌 3열) | 최적 파라미터 (우)
  - **누적 Trial 표시** 3종:
    - ① 사이드바 caption: 실행 전 예상 누적 수 ("이전 누적 N회 → 실행 후 약 M회")
    - ② metric delta: 이전 best 대비 개선량 (`+0.0123` 형식)
    - ③ collapsible expander: "💾 누적 study 정보" (study 이름/DB 경로/누적 수 상세)

**파일**:
```
src/backtesting/optimizer.py       (Persistent Study: study_storage, _make_study_name, reset 파라미터)
app/utils/data_loader.py           (_OPTUNA_STORAGE 상수, reset_study 파라미터)
app/pages/3_📈_백테스트.py         (UI 개선 전반, 누적 Trial 표시 3종, 사이드바 구조 개선)
data/optuna_studies.db             (신규 - SQLite Optuna 저장소)
```

---

### 2026-02-20 (BacktestPrecomputer 속도 최적화 + Streamlit 백테스트 페이지)

**목표**: 백테스트 속도 극적 향상 (사전 계산) + Streamlit 백테스트 페이지에 Optuna 최적화 UI 추가

**구현 내용**:

- ✅ **precomputer.py 신규 생성** (`src/backtesting/precomputer.py`)
  - `PrecomputeResult` dataclass: MultiIndex(trade_date, stock_code) Z-Score/시그널 + price_lookup dict
  - `BacktestPrecomputer` 클래스: DB 1회 로드 → 전 날짜 벡터화 계산
    - `_compute_sff_all_dates()`: 외국인 중심 조건부 Sff 벡터화
    - `_compute_multi_period_zscores_all_dates()`: 6기간 Z-Score 벡터화 (조건부 Z-Score 적용)
    - `_compute_signals_all_dates()`: MA 크로스/가속도/동조율 벡터화
    - `_build_price_lookup()`: (stock_code, trade_date) → float O(1) 조회
  - **효과**: 매 거래일 Stage 1-3 재계산 → 1회 사전 계산 후 O(1) 참조

- ✅ **engine.py 수정**: Precomputer 기반 고속 실행
  - `_scan_signals_on_date()`: precomputed 데이터 있으면 fast path 라우팅
  - `_scan_signals_on_date_fast()`: O(1) Z-Score/시그널 lookup + 패턴 분류
  - `get_price()` / `get_stock_name()`: precomputed 데이터 우선 참조
  - `run()`: normalizer.preload() → BacktestPrecomputer.precompute() 교체

- ✅ **plotly_visualizer.py 신규 생성** (`src/backtesting/plotly_visualizer.py`)
  - `PlotlyVisualizer` 클래스: Streamlit용 인터랙티브 차트 5종
  - `fig_equity_curve()`, `fig_drawdown()`, `fig_monthly_returns()`
  - `fig_return_distribution()`, `fig_pattern_performance()`

- ✅ **Streamlit 백테스트 페이지** (`app/pages/3_📈_백테스트.py`)
  - 사이드바: 백테스트 파라미터 설정 + 실행 버튼
  - **Optuna 최적화 UI** (사이드바 하단 접기 섹션):
    - Trial 수 슬라이더 (10~200, 기본 30)
    - 평가 지표 선택 (Sharpe Ratio / 총 수익률 / 승률 / Profit Factor)
    - "최적 파라미터 찾기" 버튼 → 최적 파라미터 사이드바 위젯 자동 반영 + 백테스트 자동 실행
    - 최적화 결과 표시 (메트릭값, Trial 수, 5개 최적 파라미터)
  - KPI 카드 5개 + Plotly 차트 5탭 + 거래 내역 테이블 + CSV 다운로드
  - session_state 기반 파라미터 전달 (`pending_opt_params` → widget key 자동 반영)

- ✅ **data_loader.py 수정** (`app/utils/data_loader.py`)
  - `run_optuna_optimization()` 함수 추가: OptunaOptimizer 래핑

**속도 벤치마크**:
| 기간 | 기존 (초) | Precomputer (초) | 향상 |
|------|----------|-----------------|------|
| 38일 (2025-01~02) | 177.6 | 1.1 | 165배 |
| 63일 (2025-06~08) | 392.8 | 1.5 | 262배 |
| 1년 (Streamlit) | N/A | 4.0 | - |

**테스트**: 23개 신규 (test_precomputer.py)
- PrecomputeResult 구조: 6개
- Z-Score 사전 계산: 4개 (slow path와 수치 일치 검증 포함)
- 시그널 사전 계산: 5개
- 가격/종목명 lookup: 5개
- start_date 필터/기관 가중치: 3개

**전체 테스트**: 256개 (100% 통과)

**파일 구조**:
```
src/backtesting/precomputer.py (BacktestPrecomputer - 신규)
src/backtesting/plotly_visualizer.py (PlotlyVisualizer - 신규)
src/backtesting/engine.py (Precomputer 기반 fast path)
src/backtesting/__init__.py (BacktestPrecomputer 추가)
app/pages/3_📈_백테스트.py (Optuna 최적화 UI + Plotly 차트)
app/utils/data_loader.py (run_optuna_optimization 추가)
tests/backtesting/test_precomputer.py (23개 - 신규)
```

---

### 2026-02-19 (--optimize Grid Search → Optuna 교체)

**목표**: `--optimize`의 Grid Search(ParameterOptimizer)를 Optuna로 교체하여 코드 통일

**구현 내용**:
- ✅ **optimizer.py 정리**: `ParameterOptimizer` 클래스 + `_run_backtest_worker()` 함수 제거
  - `OptunaOptimizer`만 남김 (--optimize, --walk-forward 공용)
  - 미사용 import 제거 (`itertools`, `Pool`, `List`)
  - 모듈 docstring 업데이트

- ✅ **backtest_runner.py 수정**: `ParameterOptimizer` → `OptunaOptimizer`
  - `run_optimization()`: `optimizer.optimize(n_trials, metric)` 호출
  - 결과(dict) → 1행 DataFrame 변환 후 CSV 저장
  - `--optimize` help: "Grid Search" → "Optuna 파라미터 최적화"
  - `--top-n` 옵션 제거 (Optuna는 최적 1개 반환)
  - `--n-trials` help: --optimize, --walk-forward 공용으로 변경

- ✅ **walk_forward.py 정리**: 미사용 `ParameterOptimizer` import 제거

- ✅ **test_optimizer.py 재작성**:
  - `TestParameterOptimizer` (5개) 제거
  - `TestOptunaOptimizer` (5개) 추가:
    - `test_default_param_space_keys`, `test_optimize_returns_expected_structure`
    - `test_optimize_returns_none_on_failure`, `test_optimize_best_metric_value`
    - `test_optimize_n_trials`
  - `TestInstitutionWeightConfig` (3개) 유지

**테스트**: 233개 (100% 통과, 테스트 수 변동 없음)

**파일 구조**:
```
src/backtesting/optimizer.py (ParameterOptimizer 제거, OptunaOptimizer만 유지)
src/backtesting/walk_forward.py (미사용 import 제거)
scripts/analysis/backtest_runner.py (OptunaOptimizer 사용, --top-n 제거)
tests/backtesting/test_optimizer.py (TestOptunaOptimizer로 교체)
```

**CLI 사용 예시**:
```bash
# Optuna 파라미터 최적화 (기본: 50 trials)
python scripts/analysis/backtest_runner.py --optimize

# 100 trials, total_return 기준
python scripts/analysis/backtest_runner.py --optimize --n-trials 100 --metric total_return

# Walk-Forward도 동일한 OptunaOptimizer 사용
python scripts/analysis/backtest_runner.py --walk-forward --n-trials 100 --workers 4
```

---

### 2026-02-19 (Walk-Forward Optuna 업그레이드)

**목표**: Walk-Forward의 Grid Search를 Optuna Bayesian Optimization으로 교체

**구현 내용**:
- ✅ **optimizer.py 수정**: `OptunaOptimizer` 클래스 추가 (ParameterOptimizer는 유지)
  - `DEFAULT_PARAM_SPACE`: 연속 범위 파라미터 공간 (float/int)
  - `_build_objective()`: Optuna objective 클로저
    - Step 0: 학습 기간 **절반** 평가 → `trial.report()` → MedianPruner 판단
    - 통과 시: **전체 기간** 평가 → 최종 메트릭 반환
  - `_narrow_param_space()`: Phase 1 상위 25% Trial 기준 탐색 범위 좁히기
  - `optimize(n_trials=50)`: 2단계 Bayesian 최적화
    - Phase 1 (`n//2` trials): 넓은 범위, MedianPruner 활성화
    - Phase 2 (`n - n//2` trials): 좁힌 범위 + Phase 1 최고값 seed 삽입
    - 두 Phase 통합 후 최고 Trial 선택

- ✅ **walk_forward.py 수정**: Optuna + 기간 단위 병렬 실행
  - `_run_wf_period_optuna_worker()` 모듈 레벨 worker 추가 (pickle 호환)
    - 1개 기간: OptunaOptimizer로 학습 기간 최적화 → 검증 기간 백테스트
    - multiprocessing.Pool 병렬 지원
  - `WalkForwardConfig`: `n_trials=50` 파라미터 추가
  - `WalkForwardAnalyzer.run()`: Grid Search → OptunaOptimizer로 교체
    - `workers > 1`: Pool로 기간 단위 병렬 실행
    - `workers == 1`: 순차 실행 (verbose 상세 출력)

- ✅ **backtest_runner.py 수정**: `--n-trials` CLI 옵션 추가
  - `--n-trials N`: Optuna Trial 수 (기본: 50, Phase 1: n//2, Phase 2: 나머지)
  - `--workers N`: 기간 단위 병렬 실행 수

- ✅ **requirements.txt**: `optuna>=3.0.0` 추가

**테스트**: 16개 (100% 통과, 기존 15개 + n_trials 테스트 1개 추가)
- 기존 run() 테스트 3개: `ParameterOptimizer` mock → `_run_wf_period_optuna_worker` mock으로 교체
- `test_wf_config_n_trials_custom` 신규 추가

**전체 테스트**: 206개 (100% 통과)

**파일 구조**:
```
requirements.txt (optuna>=3.0.0 추가)
src/backtesting/optimizer.py (OptunaOptimizer 클래스 추가)
src/backtesting/walk_forward.py (_run_wf_period_optuna_worker + WalkForwardConfig n_trials)
scripts/analysis/backtest_runner.py (--n-trials 옵션)
tests/backtesting/test_walk_forward.py (mock 교체 + n_trials 테스트)
```

**CLI 사용 예시**:
```bash
# Walk-Forward (기본: 50 trials, 순차)
python scripts/analysis/backtest_runner.py --walk-forward \
  --start 2024-01-01 --end 2024-12-31

# 100 trials, 4 workers 병렬 (기간 단위)
python scripts/analysis/backtest_runner.py --walk-forward \
  --n-trials 100 --workers 4

# Grid Search는 그대로
python scripts/analysis/backtest_runner.py --optimize --workers 4
```

---

### 2026-02-19 (Stage 4 Week 5: Walk-Forward Analysis + 성능 최적화)

**목표**: Walk-Forward Analysis 구현 및 백테스트 성능 최적화

**구현 내용**:
- ✅ **normalizer.py 수정**: preload/clear_preload 메커니즘 추가
  - `preload(end_date)`: DB 1회 로드 → `self._preload_raw` 캐시
  - `clear_preload()`: 캐시 해제 (메모리 반환)
  - `_apply_sff_formula()`: 원본 데이터 → Sff 계산 공통 메서드
  - `calculate_sff()` / `_get_sff_data()`: preload 활성화 시 메모리 필터링 사용
  - **효과**: 매 거래일 DB 쿼리 → 백테스트 시작 시 1회 쿼리 (대폭 속도 향상)

- ✅ **engine.py 수정**: `run(preload_data=True)` 파라미터 추가
  - `preload_data=True`: run() 시작 시 preload(), 종료 시 clear_preload() 자동 호출
  - `preload_data=False`: 기존 방식 유지 (메모리 절약 필요 시)

- ✅ **walk_forward.py 신규 생성** (`src/backtesting/walk_forward.py`)
  - `_add_months()`: stdlib calendar만 사용 (dateutil 불필요)
  - `WalkForwardConfig`: train_months/val_months/step_months/metric/workers
  - `WalkForwardAnalyzer`:
    - `split_periods()`: 학습/검증 기간 롤링 분할
    - `run()`: 각 기간 Grid Search → 최적 파라미터로 검증 백테스트
    - `summary()`: 기간별 결과 DataFrame 반환
    - `print_results()`: 통합 통계 출력

- ✅ **backtest_runner.py 수정**: --walk-forward 옵션 추가
  - `--walk-forward`: Walk-Forward Analysis 실행
  - `--train-months N`: 학습 기간 (기본: 6)
  - `--val-months N`: 검증 기간 (기본: 1)
  - `--step-months N`: 롤링 스텝 (기본: 1)
  - `--wf-save-csv PATH`: 결과 CSV 저장
  - `run_walk_forward()` 함수 분리

**테스트**: 15개 (100% 통과)
- WalkForwardConfig 기본값/커스텀: 2개
- split_periods 기간 수/중복/연속성/데이터부족: 5개
- run() 반환키/메트릭포함/summary/빈결과: 4개
- normalizer preload 활성화/해제/end_date필터/결과일치: 4개

**전체 테스트**: 205개 (100% 통과) - Stage 4 완료

**파일 구조**:
```
src/analyzer/normalizer.py (preload/clear_preload/_apply_sff_formula 추가)
src/backtesting/engine.py (run()에 preload_data 파라미터)
src/backtesting/walk_forward.py (WalkForwardAnalyzer - 신규)
scripts/analysis/backtest_runner.py (--walk-forward 옵션)
tests/backtesting/test_walk_forward.py (15개 테스트 - 신규)
```

---

### 2026-02-19 (Stage 4 Week 4: ParameterOptimizer)

**목표**: Grid Search 기반 파라미터 최적화 시스템 구현

**구현 내용**:
- ✅ **normalizer.py 수정**: institution_weight 파라미터화
  - 0.3 하드코딩 → `self.config.get('institution_weight', 0.3)` 참조
  - `calculate_sff()`, `_get_sff_data()` 두 곳 모두 수정
  - `__init__` config에 기본값 포함

- ✅ **engine.py 수정**: BacktestConfig에 institution_weight 추가
  - `institution_weight: float = 0.3` 파라미터 추가
  - `BacktestEngine.__init__` 에서 SupplyNormalizer에 institution_weight 전달

- ✅ **optimizer.py 신규 생성** (`src/backtesting/optimizer.py`)
  - `_run_backtest_worker()`: 모듈 레벨 worker (multiprocessing pickle 호환)
  - `ParameterOptimizer` 클래스
    - `DEFAULT_PARAM_GRID`: 5개 파라미터 × 기본 값 세트
    - `grid_search()`: 모든 조합 실행 → top_n DataFrame 반환
    - `_build_param_combinations()`: itertools.product 조합 생성
    - `print_results()`: 결과 테이블 출력
  - `workers=1`: 순차 / `workers>1`: multiprocessing.Pool 병렬

- ✅ **backtest_runner.py 수정**: --optimize 옵션 추가
  - `--optimize`: Grid Search 실행
  - `--workers N`: 병렬 worker 수
  - `--metric`: 평가 지표 (sharpe_ratio/total_return/win_rate/profit_factor)
  - `--top-n N`: 상위 N개 출력
  - `--opt-save-csv PATH`: 결과 CSV 저장
  - `run_optimization()` 함수 분리

**테스트**: 8개 (100% 통과)
- institution_weight Config 기본값/커스텀: 2개
- BacktestEngine normalizer 전달: 1개
- 파라미터 조합 수/값 검증: 2개
- grid_search 반환 타입/정렬/top_n: 3개

**전체 테스트**: 190개 (100% 통과) - 기존 3개 실패도 복구됨

**파일 구조**:
```
src/analyzer/normalizer.py (institution_weight 파라미터화)
src/backtesting/engine.py (institution_weight in BacktestConfig)
src/backtesting/optimizer.py (ParameterOptimizer - 신규)
scripts/analysis/backtest_runner.py (--optimize 옵션)
tests/backtesting/test_optimizer.py (8개 테스트 - 신규)
```

---

### 2026-02-19 (외국인 중심 조건부 Sff 적용)

**목표**: 외국인 수급 중심 분석으로 전환 (기관은 동반 여부만 반영)

**문제**:
- 기존: `combined_sff = foreign_sff + institution_sff` (1:1 동등 합산)
- 외국인 +1000억, 기관 -1050억 → combined = -50 (매도로 판단) ❌
- 실제: 외국인 강한 매수 → 유의미한 신호
- 기관만 강한 매수(외국인 미미)도 강한 신호로 잡히는 문제

**해결: 외국인 중심 조건부 합산**:
```python
same_direction = (foreign_sff * institution_sff) > 0
combined_sff = where(
    same_direction,
    foreign_sff + institution_sff * 0.3,  # 동반: 기관 30%만 반영
    foreign_sff                           # 반대: 외국인만
)
```

**변경 파일**:
- `src/analyzer/normalizer.py`: calculate_sff(), _get_sff_data() 2곳
- `src/analyzer/signal_detector.py`: calculate_acceleration() 1곳

**영향 범위**: Stage 1 근원 데이터 변경 → Stage 2~4 자동 반영 (코드 변경 없음)
- MA크로스: 이미 외국인만 사용 (변경 불필요)
- 동조율: 이미 외국인 AND 기관 동반 측정 (변경 불필요)
- 가속도: 동일 조건부 로직 적용

**테스트**: 179개 통과 (기존 3개 실패 유지, 새로운 실패 없음)

---

### 2026-02-18 (Z-Score 조건부 공식 적용)

**목표**: 부호 전환 시 Z-Score 과잉 반응 문제 해결

**문제**:
- 기존 Z-Score: `Z = (today - mean) / std`
- 매수세(mean=+0.488) 중 작은 매도(today=-0.038) 발생 시:
  - Z = (-0.038 - 0.488) / 0.436 = **-1.21** (큰 매도 신호로 오인)
- 원인: today와 mean의 부호가 다르면 빼기 연산이 차이를 증폭
- 영향: 반대 수급 청산(reverse_signal)에서 잘못된 청산 발생
  - 예: 주성엔지니어링 3/18 매수 → 3/25 잘못된 청산 (패턴점수 75→60점 초과)

**해결: 조건부 Z-Score**:
```python
same_sign = (today * mean) > 0
Z = (today - mean) / std   # 같은 방향: 기존 (폭발 감지 유지)
Z = today / std             # 방향 전환: 크기만 평가 (과잉 반응 방지)
```

**검증 케이스**:
- 폭발 매수(today=20, mean=12): 같은 방향 → Z=1.79 ✓ (폭발 감지)
- 작은 매도(today=-0.038, mean=+0.488): 방향 전환 → Z=-0.087 ✓ (무시)
- 큰 매도 시작(today=-2.0, mean=+0.488): 방향 전환 → Z=-4.59 ✓ (강한 매도)
- 연속 매도 중(today=-0.7, mean=-0.58): 같은 방향 → Z=-0.64 ✓ (기존 동일)

**변경 파일**:
- `src/visualizer/performance_optimizer.py:139-145` (Z-Score 계산 조건부 변경)

**영향 범위**:
- Stage 2 Z-Score 계산만 변경, 나머지 파이프라인(Stage 3~4) 변경 없음
- 기존 테스트 165개 통과 (2개 기존 실패는 Z-Score 변경과 무관)

---

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

**프로젝트 버전**: v5.0 (Stage 5-1 Streamlit 대시보드 + Precomputer 속도 최적화)
**마지막 업데이트**: 2026-02-20
