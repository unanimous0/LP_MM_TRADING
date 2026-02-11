# 한국 주식 외국인/기관 투자자 수급 분석 프로그램

## [Status]
- **현재 작업**: Stage 2 완료! ✅
- **마지막 업데이트**: 2026-02-11
- **다음 시작점**: Stage 3 - 패턴 기반 분류 시스템
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

**핵심 인사이트**:
- 4가지 정렬 키는 각각 다른 차원의 정보 제공 (Recent=현재 강도, Momentum=전환점, Weighted=중장기 트렌드, Average=일관성)
- 단일 정렬로는 베스트 픽 찾기 불가 → Stage 3에서 다차원 통합 판단 필요

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

#### Stage 1: 이상 수급 탐지
```bash
# 기본 분석 (임계값 2.0, 상위 20개)
python scripts/analysis/abnormal_supply_detector.py

# 매수 시그널만
python scripts/analysis/abnormal_supply_detector.py --direction buy --threshold 2.5
```

#### Stage 2: 히트맵 생성
```bash
# 기본 실행 (6개 기간, Recent 모드)
python scripts/analysis/heatmap_generator.py

# 정렬 모드 선택
python scripts/analysis/heatmap_generator.py --sort-by momentum

# 섹터 필터링 + CSV 저장
python scripts/analysis/heatmap_generator.py --sector "반도체 및 관련장비" --save-csv
```

**자세한 내용**: `IMPLEMENTATION_GUIDE.md` 참조

---

## [Next Steps]

### Stage 3: 패턴 기반 분류 시스템 (예정)

**목표**: Stage 1~2 결과를 통합하여 3개 바구니 자동 분류

**구현 계획**:
1. **다차원 특성 추출**
   - 4가지 정렬 키 (Recent, Momentum, Weighted, Average)
   - 추가 특성 (변동성, 지속성, 단기/장기 비율)
   - 이벤트 센서 (MA 골든크로스, 가속도, 동조율)

2. **패턴 분류 규칙**
   - 지속 매집형: Weighted/Average 높음 + 일관성
   - 전환 돌파형: Momentum 높음 + Recent 높음
   - 조정 반등형: Weighted 높음 + Momentum 낮음

3. **통합 리포트**
   - 종목별 1줄 요약 카드
   - 패턴, 점수, 시그널 통합 제공
   - 진입/청산 포인트 제시

**파일 생성 예정**:
- `src/analyzer/pattern_classifier.py`
- `src/analyzer/integrated_report.py`
- `scripts/analysis/regime_scanner.py`

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
│   │   └── normalizer.py          # Sff/Z-Score 계산
│   ├── visualizer/                # 시각화 모듈
│   │   ├── performance_optimizer.py
│   │   └── heatmap_renderer.py
│   ├── config.py                  # 전역 설정
│   └── utils.py                   # 입력 검증 (보안)
├── scripts/
│   ├── analysis/
│   │   ├── abnormal_supply_detector.py  # Stage 1 CLI
│   │   └── heatmap_generator.py         # Stage 2 CLI
│   ├── crawlers/
│   │   ├── crawl_all_data.py
│   │   ├── crawl_stock_prices.py
│   │   └── crawl_free_float.py
│   └── loaders/
│       ├── load_initial_data.py
│       └── load_daily_data.py
└── tests/                         # 테스트 (61개, 100% 통과)
    ├── test_config.py
    ├── test_normalizer.py
    ├── test_performance_optimizer.py
    └── test_utils.py
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

### Sff (Supply Float Factor)
```
Sff = (순매수 금액 / 유통시총) × 100
```
→ 시총 왜곡 제거, 유통물량 대비 비율로 정규화

### Z-Score
```
Z-Score = (현재값 - 60일 평균) / 60일 표준편차
```
→ 변동성 보정, 이상 수급 탐지 (|Z| > 2.0)

### 4가지 정렬 모드
- **Recent**: (1W+1M)/2 - 현재 강도
- **Momentum**: 1W-2Y - 수급 개선도 (전환점 포착)
- **Weighted**: 가중 평균 - 중장기 트렌드
- **Average**: 단순 평균 - 전체 일관성

---

## [Performance]

| Stage | 처리 | 소요시간 | 최적화 |
|-------|------|----------|--------|
| Stage 1 | 345종목 이상 수급 탐지 | 15초 | 벡터화 완료 |
| Stage 2 | 345종목×6기간 히트맵 | 1.5초 | Sff 캐싱 + groupby.transform |

---

## [방법론: 수급 레짐 스캐너]

### ① Stage 1: 데이터 정규화
- Sff: 유통물량 대비 순매수 강도
- Z-Score: 변동성 보정 수급

### ② Stage 2: 시공간 매트릭스
- 6개 기간(1W~2Y) 히트맵
- 4가지 정렬 모드 (투자 스타일별)

### ③ Stage 3: 이벤트 센서 (예정)
- 수급 MA 골든크로스
- 수급 가속도 (2차 미분)
- 외인-기관 동조율

### ④ Stage 4: 통합 스코어링 (예정)
- 3개 바구니 분류 (지속/돌파/반등)
- 종목별 1줄 요약 카드

---

## [Progress History]

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

**프로젝트 버전**: v2.0 (Stage 2 완료)
**마지막 업데이트**: 2026-02-11
