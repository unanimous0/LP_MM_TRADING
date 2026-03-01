# 한국 주식 외국인/기관 투자자 수급 분석 프로그램

## [Status]
- **현재 작업**: Stage 5-1 Streamlit 웹 대시보드 진행 중
- **마지막 업데이트**: 2026-03-01
- **백테스트 권장 시작일**: 2025-01-01 이후 (DB가 2024-01-02 시작이므로 1Y 데이터 확보)
- **다음 시작점**: 페이지 재편 완료 (수급 메인 + 이상수급 분리) — 다음 기능 기획 또는 실제 데이터 검증
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
- **Plotly Treemap**: D3.js → `go.Treemap` 교체 (섹터 라벨에 점수 직접 표시, 다크테마 통합, CDN 의존성 제거)
  - 섹터 노드 라벨: `'섹터명  평균 XX.X점  (종합점수)'` — 호버 없이 박스에 즉시 표시
  - per-node texttemplate 배열: 섹터/종목 노드별 다른 표시 형식
  - 색상: 딥레드→앰버→옐로→sky-400(앱 primary)→에메랄드
- **스코어링 개선**: Temporal Consistency(tc) + Short Trend → 인접 기간 Z-Score 순서 + 단기 모멘텀 방향 반영
  - tc 미달 시 모멘텀형 → 기타 (tc≥0.5), 지속형은 tc 조건 없음 (장기매집 특성상 tc=0.0 정상)
  - 점수 보너스 ±10점: `tc_bonus = (tc - 0.5) × 20` (지속형 제외)
  - short_trend = 5D - 20D (가중치 0.15, tanh 이후 계산 — sort key 스케일 일치)
  - 지속형은 short_trend 가중치=0 (5D<20D가 이상적 패턴이므로 패널티 방지)
  - 방향 확신도 기준: `_sff_5d_avg` (5일 평균) 사용 — 하루 소폭 매도로 제외되는 엣지 케이스 방지
  - 3개 경로 모두 일치: `pattern_classifier.py`, `precomputer.py`, `charts.py`
- **프로젝트 보고서**: `WHALE_SUPPLY_REPORT.html` — 배경·수식·파이프라인·사용법·해석 가이드 포함
- **Phase 1 스코어링 검증**: `use_tc`/`use_short_trend` 토글 전 파이프라인 적용 + CLI `--no-tc`/`--no-short-trend`
  - min_score=70 기준: 현재 스코어링 +28.15% (레거시 +25.89%), 칼마 2.18 (레거시 1.86)
- **Phase 2 대시보드 개선 4종**: 관심종목 저장 / 종목 비교 페이지 / 고득점 변동 알림 / 백테스트 히스토리
- **[버그수정] 코드 리뷰 5건**: MDD 공식 / snapshot 중복·날짜 / Z-Score 슬라이더 최댓값 통일 / Tab2 캡션
- **복합 패턴 분류(sub_type)**: 기본 패턴(4종) 위에 세부 한정자(7종) 추가 → `pattern_label` 합성 라벨
  - 모멘텀형: ①장기기반(+5) / ⑥감속(-5) / ⑦단기반등(-8)
  - 지속형: ②단기돌파(+5) / ④전면수급(+3) / ⑤모멘텀약화(-5)
  - 전환형: ③V자반등(+3)
  - UI 전면 반영: 패턴분석 sub_type 필터 + 종목리스트/히트맵/Treemap/관심종목 pattern_label 표시
- **UI 단순화 "점수 하나로 줄 세우기"**: 패턴분석 사이드바 12→4개, 탭 7→2개(+패턴가이드), 히트맵 필터 3개 제거
  - `final_score = score + signal_count × 5`로 항상 정렬, 나머지 지표는 설명용으로만 사용
  - 커스텀 HTML 테이블: 5D Z / 종합점수 셀 호버 시 점수 산출 근거 툴팁 (Z-Score 그리드 + 지표별 공식 + 2열 레이아웃)
  - 테이블 `max-height:680px` 스크롤 + sticky header
- **[변경] 모멘텀 계산 기준 500D→200D**: 500D는 참고용, 장기 기준 = 200D/100D
  - `pattern_classifier.py` 폴백: 200D→100D→50D→20D (500D 제거)
  - 히트맵 정렬(`charts.py`, `heatmap_renderer.py`)도 200D 기준으로 통일
  - 스크립트/문서 10곳 일괄 수정
- 290개 테스트 (100% 통과)

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

> 전체 이력은 **CHANGELOG.md** 참조. 아래는 최근 항목만 기록.

### 2026-03-01 (페이지 재편 + UI 개선)

**목표**: 메인 페이지를 final_score 단일 랭킹 중심으로 재설계 + 이상수급/수급순위를 별도 페이지로 분리

**구현 내용**:

- ✅ **메인 페이지 → "수급 TOP N" 랭킹 + 드릴다운** (`streamlit_app.py` 전면 개편)
  - KPI 5개 + `final_score` 내림차순 TOP N 테이블 + 드릴다운 패널
  - 드릴다운: 패턴 배너 + 점수 산출 근거(6개 컴포넌트) + 멀티기간 Z-Score 바차트 + 이상수급 여부 + 종목상세 링크
  - 테이블 클릭 → 드릴다운 자동 연동 (`on_select="rerun"` + `session_state` 동기화)
  - 하단 패턴 분석 요약 (파이차트 + 히스토그램)

- ✅ **이상 수급 페이지 신규** (`7_⚡_이상수급.py`)
  - 메인에서 분리된 참고 데이터 4탭: 이상수급 / 당일수급순위 / 수급금액 / 고득점변동알림
  - 수급 금액 탭: 종목상세와 동일한 차트+HTML 테이블 (이상수급 종목 우선 표시)
  - 탭 이모지 제거, 관심종목 탭 제거 (다른 페이지에 존재)

- ✅ **UI 개선 4건**
  - "수급 왕" → "수급" 명칭 변경
  - 랭킹 테이블 15행 스크롤 (`height=600`)
  - 테이블 클릭→드릴다운 동기화 (`session_state['drill_select']` 직접 업데이트)
  - 드릴다운 섹션 `st.subheader` 적용 (다른 섹션과 글씨 크기 통일)

**pages.toml 업데이트**: 수급 왕(:material/trophy:) + 이상 수급(:material/bolt:) 메뉴 추가

**파일** (3개):
```
app/streamlit_app.py           (메인 페이지 전면 재설계)
app/pages/7_⚡_이상수급.py      (신규 — 이상수급/수급순위/수급금액/변동알림)
.streamlit/pages.toml          (메뉴 구조 업데이트)
```

**테스트**: 300개 (100% 통과)

---

### 2026-02-27 (복합 패턴 분류 시스템 — sub_type 7종 구현)

**목표**: 기본 패턴(4종) 위에 세부 한정자(7종)를 추가하여 동일 패턴 내 품질 차이 표현

**배경**:
- "모멘텀형" 안에 장기매집 위 단기폭발(★★★★★)과 단기반등 함정(★)이 공존
- 지속형→모멘텀형 전환(단기돌파) 시 500D가 이미 높아 momentum이 작음 → 모멘텀형 탈락
- 복수 조건 충족 종목도 첫 매칭 패턴만 표시 → 정보 손실

**3컬럼 시스템**: `pattern` (기존) + `sub_type` (한정자) + `pattern_label` (합성 표시)

**복합 패턴 7종**:

| # | 기본 패턴 | sub_type | 조건 | 점수 보정 |
|---|----------|----------|------|----------|
| ① | 모멘텀형 | 장기기반 | 200D>0.3 AND 100D>0.3 | +5 |
| ② | 지속형 | 단기돌파 | 5D>1.0 AND short_trend>0.5 | +5 |
| ③ | 전환형 | V자반등 | 5D>1.0 AND recent>0.5 | +3 |
| ④ | 지속형 | 전면수급 | 5D~200D 모두 Z>0 AND std<0.5 | +3 |
| ⑤ | 지속형 | 모멘텀약화 | short_trend<-0.3 AND 5D<20D | -5 |
| ⑥ | 모멘텀형 | 감속 | short_trend<-0.3 | -5 |
| ⑦ | 모멘텀형 | 단기반등 | 200D<-0.3 OR 100D<-0.3 | -8 |

**판정 우선순위** (위험 먼저):
- 모멘텀형: ⑦단기반등 → ⑥감속 → ①장기기반 → None
- 지속형: ②단기돌파 → ④전면수급 → ⑤모멘텀약화 → None
- 전환형: ③V자반등 → None

**구현 내용**:

- ✅ **Phase 1: 핵심 로직** (`pattern_classifier.py`)
  - `_get_default_config()`: `sub_type_thresholds` + `sub_type_score_bonus` 딕셔너리 추가
  - `_classify_sub_type(pattern, row, config)`: 신규 static 메서드 — 기본 패턴별 조건 분기
  - `classify_all()`: sub_type/pattern_label 컬럼 추가 + sub_type 점수 보정 (tc_bonus와 별개)
  - 출력 컬럼: `result_cols`에 `sub_type`, `pattern_label` 추가

- ✅ **Phase 2: UI 업데이트** (4개 파일)
  - `app/pages/2_🔍_패턴분석.py`: 사이드바 sub_type 멀티셀렉트 필터 + 종목리스트 pattern_label 표시 + 교차테이블 pattern_label 기준
  - `app/utils/charts.py`: 히트맵 호버 pattern_label 사용 + Treemap pattern_label 사용
  - `app/pages/1_📊_히트맵.py`: 미니 상세 패널 pattern_label 표시
  - `app/streamlit_app.py`: 관심종목 테이블 pattern_label 표시 + 저장 관심종목 조인에 pattern_label 포함

- ✅ **Phase 3: 테스트** (11개 신규)
  - `TestSubType` 클래스: 7개 sub_type 각각 단위 테스트 + None 케이스 + pattern_label 포맷 + 점수 보정 + classify_all 출력 컬럼

**하위 호환**: `pattern` 컬럼 기존과 동일 유지, `sub_type`이 None이면 `pattern_label == pattern`

**파일** (7개):
```
src/analyzer/pattern_classifier.py    (sub_type 핵심 로직)
app/utils/charts.py                   (호버/Treemap pattern_label)
app/pages/2_🔍_패턴분석.py            (sub_type 필터 + pattern_label 표시)
app/pages/1_📊_히트맵.py              (미니 상세 pattern_label)
app/streamlit_app.py                  (관심종목 pattern_label)
tests/test_pattern_classifier.py      (TestSubType 11개)
CLAUDE.md                             (진행 기록)
```

**테스트**: 287개 (100% 통과) — 기존 276 + 신규 11

---

### 2026-02-26 (코드 리뷰 + 버그수정 5건)

**목표**: Phase 1+2 구현 완료 후 전체 코드베이스 정밀 검토 — 로직·설계·파이프라인·누락·중복 관점

**검토 방법**: 5개 병렬 에이전트 실행 후 수동 검증으로 False Positive 필터링

| 검토 영역 | 대상 파일 |
|----------|---------|
| 핵심 분석 파이프라인 | normalizer, pattern_classifier, signal_detector, performance_optimizer |
| 백테스팅 시스템 | engine, precomputer, portfolio, optimizer, walk_forward, metrics |
| Streamlit 앱 + data_loader | streamlit_app, data_loader |
| Streamlit 페이지 | backtest, stock_detail, stock_compare, heatmap, pattern_analysis |
| 차트 + 시각화 | charts, plotly_visualizer |

**확인된 실제 버그 3건 + UI 불일치 2건**:

- ✅ **[버그] MDD 공식 오류** (`plotly_visualizer.py:563`)
  - **문제**: `_build_kpi_html()`의 KPI 카드 MDD 값이 실제보다 작게 표시됨
  - **원인**: `(v - np.maximum.accumulate(v)).min() / np.maximum.accumulate(v).max()` — 분모에 전체 구간 최대값(글로벌 max) 사용
  - **예시**: `[100→150→100→200]` 시 trough=-50, 분모=200 → -25% (오류) vs 분모=150 → -33.3% (정상)
  - **수정**: 원소별 분모로 교체 `((v - _running_max) / _running_max).min() * 100`
  - **영향**: KPI 카드의 MDD 표시값만 영향 (백테스트 내부 계산 경로와 별개)

- ✅ **[버그] snapshot_scores() 중복 레코드** (`data_loader.py`)
  - **문제**: 같은 날 다른 브라우저 세션에서 홈 페이지 방문 시 동일 `analysis_date`로 중복 이벤트 삽입
  - **원인**: INSERT 전 존재 여부 확인 없음 → 직전 스냅샷 기준 재비교 → 동일 이벤트 재삽입
  - **수정**: 함수 시작 시 `COUNT(*) WHERE analysis_date = ?` 체크 → 이미 있으면 조기 반환

- ✅ **[버그] snapshot_scores() 날짜 레이블 오류** (`streamlit_app.py`)
  - **문제**: `snapshot_scores(report_df, end_date_str)` 호출 — `end_date_str`은 "이상 수급 기준일" (과거 날짜일 수 있음)
  - **원인**: `report_df`는 항상 최신 분석 결과인데, 날짜는 이상 수급 사이드바 기준일로 레이블링
  - **수정**: `_, _latest_date = get_date_range()` → `snapshot_scores(report_df, _latest_date)`

- ✅ **[UI] Z-Score 슬라이더 최댓값 불일치** (`6_🔀_종목비교.py:61`)
  - **문제**: 종목 비교 페이지 Z-Score 기준 기간 슬라이더 최댓값 500 (다른 페이지는 1300)
  - **수정**: `max=500` → `max=1300`

- ✅ **[UI] 패턴분석 Tab2 캡션 누락** (`2_🔍_패턴분析.py`)
  - **문제**: 패턴별 통계 탭이 사이드바 필터를 반영하지 않는데 사용자가 필터 적용된 것으로 오해 가능
  - **수정**: `st.caption("※ 사이드바 필터와 무관하게 전체 분석 종목 기준으로 집계됩니다.")` 추가

**False Positive 목록** (에이전트 보고 후 수동 검증으로 정상 확인):
- `max_hold_days _ni sync 없음`: `number_input` 단독 위젯 (슬라이더 쌍 없음) → `_ni` 키 불필요 ✓
- `fig_monthly_returns 첫 달 누락`: `shift(1)` 후 신규 Series에 `iloc[0] = initial_capital` 할당 → 뷰 아님 ✓
- `DataFrame.get()` 미지원: pandas 공식 API (`DataFrame.get(key, default)`) → 정상 ✓
- Plotly 3D customdata: `%{customdata[i]}` 문법 지원됨 → 정상 ✓
- `individual_net_amount` 컬럼 없음: DB 스키마 확인 결과 `investor_flows` 테이블에 없음 → `-(foreign+institution)` 근사값 사용은 의도된 설계 ✓

**파일** (5개):
```
src/backtesting/plotly_visualizer.py   (MDD 공식 수정: 원소별 running_max 분모)
app/utils/data_loader.py               (snapshot_scores: analysis_date 중복 가드)
app/streamlit_app.py                   (snapshot_scores 호출 시 max_date 사용)
app/pages/6_🔀_종목비교.py             (Z-Score 슬라이더 max 500→1300)
app/pages/2_🔍_패턴분析.py             (Tab2 캡션 추가)
```

**테스트**: 276개 (100% 통과, 버그 수정은 UI/날짜 레이어 — 기존 분석 로직 무변경)

---

## [Reference]
- **IMPLEMENTATION_GUIDE.md**: Stage 1~2 상세 구현 내용
- **DATABASE_README.md**: DB 스키마 및 사용법
- **README.md**: 프로젝트 소개 및 설치 방법

---

**프로젝트 버전**: v5.0 (Stage 5-1 Streamlit 대시보드 + Precomputer 속도 최적화)
**마지막 업데이트**: 2026-02-20
