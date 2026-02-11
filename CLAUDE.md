# 한국 주식 외국인/기관 투자자 수급 분석 프로그램

## [Status]
- 현재 작업: **Stage 2 완료 + 정렬 로직 개선!** ✅
- 마지막 업데이트: 2026-02-11
- 다음 시작점: Stage 3 - 패턴 기반 분류 (지속 매집형, 전환 돌파형, 조정 반등형)
- 현재 브랜치: main
- **Stage 1 성과**: 데이터 정규화 완료, Sff/Z-Score 분석 가능, 이상 수급 탐지 20건
- **Stage 2 성과**:
  - 시각화: 6개 기간 히트맵 (2.8초), 섹터/상위N개 필터링, CSV 출력
  - 정렬: 4가지 모드 (recent/momentum/weighted/average) - 투자 스타일별 선택 가능
  - 최적화: 벡터화 (목표 23초 → 실제 2.8초), 메모리 42% 절감
  - 보안: SQL 인젝션 방지 (입력 검증 레이어)
  - 테스트: 61개 테스트 (100% 통과)
  - 품질: 코드 리뷰 완료 (4.7/5.0 Excellent)
- **섹터 통합**: 1,576개 종목 섹터 정보 수집 완료 (97.9% 커버리지), 20개 주요 섹터 식별
- **핵심 인사이트**: 단순 평균 정렬은 부적절 (과거 강했던 종목 우선) → 맥락 기반 패턴 분류 필요

## [Progress]
- ✅ 2026-02-09: GitHub 저장소 연결 완료
- ✅ 2026-02-09: 워크플로우 체계 수립
- ✅ 2026-02-09: 프로젝트 초기 구조 설정
- ✅ 2026-02-09: SQLite 데이터베이스 스키마 설계 및 구현 완료
- ✅ 2026-02-09: 엑셀 데이터 파싱 모듈 구현 (Multi-level header 처리)
- ✅ 2026-02-09: 초기 데이터 로드 완료 (172,155 레코드)
- ✅ 2026-02-09: 데이터 검증 완료 (중복 0건, NULL 0건)
- ✅ 2026-02-09: 일별 증분 업데이트 스크립트 구현 완료 (load_daily_data.py)
- ✅ 2026-02-09: 데이터 단위 표준화 완료 (천원 → 원 마이그레이션)
- ✅ 2026-02-09: 외국인/기관 컬럼 매핑 오류 수정 (swap 완료)
- ✅ 2026-02-09: 분석 스크립트 2개 생성 (순매수 상위, 시가총액 비중 상위)
- ✅ 2026-02-10: TODO List 작성 완료 (수급 레짐 스캐너 4단계 방법론)
- ✅ 2026-02-10: 요구사항 분석 완료 (유통물량 기반 정규화, 히트맵, 이벤트 센서, 스코어링)
- ✅ 2026-02-10: 파라미터 가변성 설계 방향 확정 (MA 기간, 가속도 임계값 등 config 기반)
- ✅ 2026-02-10: **Stage 1 구현 완료** - DB 스키마 확장 (13개 컬럼)
- ✅ 2026-02-10: 데이터 수집 크롤러 3개 구현 (주가, 유통주식, 통합)
- ✅ 2026-02-10: Sff, Z-Score 정규화 모듈 구현 (src/analyzer/normalizer.py)
- ✅ 2026-02-10: 이상 수급 탐지기 CLI 도구 구현 (abnormal_supply_detector.py)
- ✅ 2026-02-10: 데이터 크롤링 완료 (주가 99.5%, 유통주식 100%, 총 171,227 레코드)
- ✅ 2026-02-10: 실전 분석 실행 성공 (이상 수급 이벤트 20건 탐지)
- ✅ 2026-02-10: 섹터 정보 크롤링 설계 및 구현 완료
- ✅ 2026-02-10: crawl_free_float.py 확장 (유통주식 + 업종 통합 수집)
- ✅ 2026-02-10: FnGuide FICS 업종 파싱 로직 구현 (extract_sector 함수)
- ✅ 2026-02-10: 분석 결과에 섹터 정보 자동 표시 (normalizer.py, abnormal_supply_detector.py)
- ✅ 2026-02-10: **전체 섹터 크롤링 완료** - 1,576개 종목 (97.9% 커버리지, 17분 51초 소요)
- ✅ 2026-02-10: 섹터별 분석 가능 - 20개 주요 섹터 식별 (반도체 120개, 의료 103개, 제약 96개 등)
- ✅ 2026-02-11: **Stage 2 구현 완료** - 시공간 히트맵 시각화 (8개 기간)
- ✅ 2026-02-11: config.py 생성 (전역 파라미터 중앙화)
- ✅ 2026-02-11: performance_optimizer.py 구현 (Sff 캐싱 + 벡터화 Z-Score)
- ✅ 2026-02-11: heatmap_renderer.py 구현 (350×8 매트릭스, Y축 강도순 정렬)
- ✅ 2026-02-11: heatmap_generator.py CLI 도구 구현 (파라미터 조정 가능)
- ✅ 2026-02-11: 성능 최적화 성공 (345종목×7기간, 1.5초 완료)
- ✅ 2026-02-11: 필터링 기능 구현 (섹터, 상위N개, CSV 출력)
- ✅ 2026-02-11: **보안 강화 완료** - SQL 인젝션 방지 (src/utils.py 검증 레이어)
- ✅ 2026-02-11: 메모리 최적화 (42% 절감, 필요 컬럼만 선택 복사)
- ✅ 2026-02-11: 한글 폰트 자동 설정 (OS 감지, matplotlib 경고 제거)
- ✅ 2026-02-11: Config 검증 시스템 구축 (범위/타입 체크)
- ✅ 2026-02-11: 병렬 처리 인프라 구축 (ThreadPoolExecutor 옵션 추가)
- ✅ 2026-02-11: **테스트 체계 완성** - 61개 테스트 (100% 통과, pytest)
- ✅ 2026-02-11: 프로젝트 정리 (불필요 파일 제거, .gitignore 업데이트)
- ✅ 2026-02-11: connection_pool.py 준비 (향후 DB 최적화용)
- ✅ 2026-02-11: **히트맵 정렬 로직 개선** - 4가지 모드 추가 (recent/momentum/weighted/average)
- ✅ 2026-02-11: Stage 2 완료 검증 및 문서화 (STAGE2_GUIDE.md 작성)
- ✅ 2026-02-11: 정렬 방식 한계 인식 - Stage 3에서 패턴 분류로 해결 예정

## [Next Steps]
1. ~~유통물량(Free Float) 데이터 수령 및 DB 스키마 확장~~ ✅ 완료
2. ~~1단계 구현: 데이터 정규화 (Sff, Z-Score 계산 모듈)~~ ✅ 완료
3. ~~2단계 구현: 시공간 히트맵 시각화 (6개 기간)~~ ✅ 완료
4. **3단계 구현: 패턴 기반 분류 시스템** ← 다음 작업
   - 3개 바구니 자동 분류:
     * 지속 매집형 (Steady Accumulation): 중장기 강한 매수 + 일관성
     * 전환 돌파형 (Supply Breakout): 수급 모멘텀 + 최근 강도
     * 조정 반등형 (Pullback Entry): 중장기 강함 + 최근 약화
   - 이벤트 센서 통합 (MA 골든크로스, 가속도, 동조율)
5. 4단계 구현: 통합 대시보드 및 일일 리포트 자동화

## [Tech Stack]
- Python 3.10+
- 데이터베이스: SQLite (내장)
- 데이터 수집: pandas, openpyxl (엑셀 파일), **FinanceDataReader (주가 크롤링)**, **BeautifulSoup (유통주식 크롤링)**
- 데이터 분석: pandas, numpy, SQL
- **시각화: matplotlib, seaborn (Stage 2)**
- 버전 관리: Git & GitHub

## [Project Structure]
```
LP_MM_TRADING/
├── CLAUDE.md                      # 프로젝트 문서 및 작업 상태
├── README.md                      # 프로젝트 소개
├── DATABASE_README.md             # 데이터베이스 사용 가이드
├── requirements.txt               # 의존성 목록
├── .gitignore                    # Git 제외 파일
├── data/                         # 데이터 저장
│   ├── 투자자수급_200_150.xlsx      # 원본 데이터 (엑셀)
│   ├── 시가총액_200_150.xlsx        # 원본 데이터 (엑셀)
│   ├── 종목코드_이름_맵핑.xlsx      # 원본 데이터 (엑셀)
│   └── processed/
│       └── investor_data.db       # SQLite 데이터베이스
├── src/                          # 소스 코드
│   ├── database/                 # 데이터베이스 모듈 ✅
│   │   ├── schema.py             # 스키마 정의 및 생성 (13개 컬럼)
│   │   ├── connection.py         # 연결 관리
│   │   └── connection_pool.py    # 연결 풀링 (향후 최적화용)
│   ├── data_collector/           # 데이터 수집 모듈 ✅
│   │   └── excel_collector.py    # 엑셀 파싱 (주가/유통주식 포함)
│   ├── data_loader/              # 데이터 로더 모듈 ✅
│   │   └── validator.py          # 데이터 검증
│   ├── analyzer/                 # 분석 모듈 ✅
│   │   ├── normalizer.py         # Sff, Z-Score 정규화 (Stage 1+2)
│   │   └── __init__.py
│   ├── visualizer/               # 시각화 모듈 ✅ (Stage 2)
│   │   ├── performance_optimizer.py  # 벡터화 Z-Score 계산
│   │   ├── heatmap_renderer.py       # 히트맵 렌더링
│   │   └── __init__.py
│   ├── config.py                 # 전역 설정 관리 ✅
│   └── utils.py                  # 입력 검증 (SQL 인젝션 방지) ✅
├── scripts/                      # 스크립트
│   ├── analysis/                 # 분석 스크립트 ✅
│   │   ├── abnormal_supply_detector.py  # 이상 수급 탐지기 (Stage 1)
│   │   └── heatmap_generator.py        # 히트맵 생성기 (Stage 2)
│   ├── crawlers/                 # 데이터 크롤러 ✅
│   │   ├── crawl_all_data.py     # 통합 크롤러 (주가 + 유통주식)
│   │   ├── crawl_free_float.py   # 유통주식 + 업종 크롤러 (FnGuide) ✅ 확장됨
│   │   └── crawl_stock_prices.py # 주가 크롤러 (FinanceDataReader)
│   ├── loaders/                  # 데이터 로더 ✅
│   │   ├── load_daily_data.py    # 일별 증분 업데이트
│   │   ├── load_initial_data.py  # 초기 데이터 로드
│   │   └── load_price_volume_backfill.py  # 주가/유통주식 백필
│   └── migrations/               # DB 마이그레이션 ✅
│       └── migrate_add_columns.py  # 5개 컬럼 추가
└── tests/                        # 테스트 코드 ✅ (Stage 2 완성)
    ├── test_config.py            # 설정 검증 (19 tests)
    ├── test_normalizer.py        # Sff/Z-Score 계산 (20 tests)
    ├── test_performance_optimizer.py  # 성능 최적화 (10 tests)
    └── test_utils.py             # 보안 검증 (18 tests)
    # 총 61개 테스트, 100% 통과
```

## [Workflow - 작업 시작 시]
1. `git pull origin main`
2. CLAUDE.md [Status] 확인
3. Claude에게: "CLAUDE.md 읽고 작업 이어서 해줘"

## [Workflow - 작업 종료 시]
1. Claude에게: "작업 상태 CLAUDE.md에 업데이트 해줘"
2. `git add .`
3. `git commit -m "[집/회사] 작업 내용"`
4. `git push origin main`

## [Data Source]
- **Initial**: 엑셀 파일 (투자자수급_200_150.xlsx, 시가총액_200_150.xlsx)
- **Crawling**: FinanceDataReader (주가), FnGuide (유통주식 + 업종)
- **Database**: SQLite (investor_data.db)
  - **172,155 레코드** (2024-01-02 ~ 2026-01-20)
  - **171,227 레코드** 완전한 데이터 (주가+유통주식, 99.5% 커버리지)
  - 345개 핵심 종목 (KOSPI200 + KOSDAQ150)
  - **1,609개 종목 마스터 데이터** (섹터 정보 97.9% 커버리지)

## [Database Schema]
- **markets**: 시장 구분 (KOSPI200, KOSDAQ150)
- **stocks**: 종목 마스터 (종목코드, 종목명, 시장ID, **섹터**)
- **investor_flows**: 투자자 수급 데이터 (**13개 컬럼**)
  - 기존: 외국인/기관 순매수량/금액, 시가총액
  - **신규**: 종가, 거래량, 거래대금, 유통주식수, 유통비율
- **Indexes**: 3개 (stock_code+trade_date, trade_date+stock_code, trade_date)

## [How to Use Database]

### **초기 설정**
```bash
# 1. 데이터베이스 생성
python -c "from src.database.schema import create_database; create_database()"

# 2. 초기 데이터 로드
python scripts/loaders/load_initial_data.py

# 3. 마이그레이션 (5개 컬럼 추가)
python scripts/migrations/migrate_add_columns.py
```

### **데이터 수집 (크롤링)**
```bash
# 통합 크롤러 (주가 + 유통주식, 약 26분 소요)
python scripts/crawlers/crawl_all_data.py --start 2024-01-01

# 또는 개별 실행
python scripts/crawlers/crawl_stock_prices.py --start 2024-01-01  # 주가 (12분)
python scripts/crawlers/crawl_free_float.py                        # 유통주식 (2분)
```

### **일별 데이터 업데이트**
```bash
# 엑셀 파일 사용
python scripts/loaders/load_daily_data.py data/투자자수급_20260210.xlsx data/시가총액_20260210.xlsx

# 주가/유통주식 포함
python scripts/loaders/load_daily_data.py data/투자자수급_20260210.xlsx data/시가총액_20260210.xlsx \
    --price-file data/주가_20260210.xlsx --ff-file data/유통주식_20260210.xlsx
```

### **섹터 정보 수집**
```bash
# 유통주식 + 업종 정보 동시 수집 (권장, 약 8분)
python scripts/crawlers/crawl_free_float.py

# 업종만 수집 (유통주식 스킵)
python scripts/crawlers/crawl_free_float.py --sector-only

# 유통주식만 수집 (업종 스킵)
python scripts/crawlers/crawl_free_float.py --skip-sector

# 특정 시장만
python scripts/crawlers/crawl_free_float.py --market KOSPI200

# 섹터 데이터 검증
python -c "from src.database.connection import get_connection; \
import pandas as pd; \
conn = get_connection(); \
df = pd.read_sql('SELECT COUNT(*) as total, COUNT(sector) as with_sector, \
ROUND(100.0 * COUNT(sector) / COUNT(*), 1) as coverage_pct FROM stocks', conn); \
print(df); conn.close()"
```

### **이상 수급 분석 (Stage 1)**
```bash
# 기본 분석 (임계값 2.0, 상위 20개, 섹터 정보 포함)
python scripts/analysis/abnormal_supply_detector.py

# 매수 시그널만 (임계값 2.5)
python scripts/analysis/abnormal_supply_detector.py --direction buy --threshold 2.5 --top 30

# 매도 시그널
python scripts/analysis/abnormal_supply_detector.py --direction sell
```

### **시공간 히트맵 생성 (Stage 2)**
```bash
# 기본 실행 (전체 8개 기간: 1D, 1W, 1M, 3M, 6M, 1Y, 2Y)
python scripts/analysis/heatmap_generator.py

# 단기 3개 기간만
python scripts/analysis/heatmap_generator.py --periods 1D 1W 1M

# 섹터 필터링 (반도체 120개 → 33개 데이터 충분한 종목)
python scripts/analysis/heatmap_generator.py --sector "반도체 및 관련장비"

# 상위 50개 종목만 (Z-Score 강도순)
python scripts/analysis/heatmap_generator.py --top 50

# 색상 임계값 조정 (±2.5σ)
python scripts/analysis/heatmap_generator.py --threshold 2.5

# 고해상도 출력 + CSV 동시 저장
python scripts/analysis/heatmap_generator.py --dpi 300 --save-csv

# 복합 필터 (제약 섹터, 상위 30개, CSV 저장)
python scripts/analysis/heatmap_generator.py --sector 제약 --top 30 --save-csv
```

**성능:**
- 345종목 × 7기간: 1.5초
- 120종목 × 7기간: 0.5초 (섹터 필터링 시)

### **Python API 사용**

**Stage 1 - 이상 수급 탐지:**
```python
from src.analyzer.normalizer import SupplyNormalizer
from src.database.connection import get_connection
import pandas as pd

conn = get_connection()
normalizer = SupplyNormalizer(conn)

# Sff 계산 (유통시총 대비 순매수 비율)
df_sff = normalizer.calculate_sff(stock_codes=['005930'])

# Z-Score 계산 (이상 수급 탐지)
df_abnormal = normalizer.get_abnormal_supply(threshold=2.0, top_n=20)
print(df_abnormal[['stock_name', 'combined_zscore', 'combined_sff', 'sector']])

conn.close()
```

**Stage 2 - 시공간 히트맵:**
```python
from src.config import DEFAULT_CONFIG
from src.database.connection import get_connection
from src.analyzer.normalizer import SupplyNormalizer
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
from src.visualizer.heatmap_renderer import HeatmapRenderer

conn = get_connection()
normalizer = SupplyNormalizer(conn)

# 8개 기간 Z-Score 계산 (벡터화 최적화)
optimizer = OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)
zscore_matrix = optimizer.calculate_multi_period_zscores(
    DEFAULT_CONFIG['periods']  # 1D, 1W, 1M, 3M, 6M, 1Y, 2Y
)

# 히트맵 렌더링
renderer = HeatmapRenderer(DEFAULT_CONFIG)
renderer.render_multi_period_heatmap(zscore_matrix, 'output/my_heatmap.png')

# CSV 저장 (데이터 분석용)
zscore_matrix.to_csv('output/zscore_matrix.csv')

conn.close()
```

자세한 내용은 `DATABASE_README.md` 참조


## [TODO List] - START
# 최종 통합 방법론: 수급 레짐 스캐너 (Supply-Demand Regime Scanner)

## ✅ ① 1단계: 데이터 정규화 - "진짜 힘(Force) 측정" (완료!)

단순 금액이 아닌, 사용자님이 확보하신 유통물량(Free Float) 데이터를 핵심 분모로 사용합니다.

### 1) 유통물량 대비 순매수 강도 (Sff)

\[
S_{ff}=\frac{\text{Net Buy Amount}}{\text{Free Float Market Cap}}\times 100
\]

- 이 지표를 통해 시총이 커도 유통 물량이 적은 종목의 수급 왜곡을 잡아냅니다.

### 2) 변동성 보정 수급 (Z-Score)

\[
Z=\frac{X-\mu}{\sigma}
\]

- 해당 종목의 최근 60일 평균 수급 대비 오늘의 수급이 얼마나 이례적인지 측정하여 **'이상 수급'** 종목을 즉시 추출합니다.

---

## ② 2단계: 시공간 매트릭스 - "모든 기간의 시각화"

사용자님이 요청하신 8개 기간(1D ~ 2Y)의 흐름을 **수급 히트맵(Heatmap)** 형태로 한 화면에 배치합니다.

### 수급 대시보드 구성
- 350개 종목을 **Y축**, 8개 기간을 **X축**으로 둡니다.

### 색상 로직
- 외국인/기관 합산 수급 강도가 강할수록 **강렬한 초록색(#72F64A)**
- 약할수록 **검은색**
- 매도세가 강할수록 **빨간색**

### 직관적 판단
- 특정 종목의 라인이 전체적으로 초록색이면 **지속 매집**
- 최근 1D, 1W만 초록색으로 변했다면 **유입 전환**으로 즉시 판독합니다.

---

## ③ 3단계: 이벤트 센서 - "골든크로스와 레짐 변화"

차트를 일일이 보지 않아도 되도록 수급 이동평균(MA)을 **'디지털 신호'**로 변환합니다.

### 1) 수급 모멘텀 이벤트
- 외국인 5일 수급 MA가 20일 MA를 돌파할 때 **상향 신호** 발생 (5일이라는 값은 가변 - 파라미터)

### 2) 수급 가속도 (2차 미분)
- 최근 5일 평균 매수세가 직전 5일보다 **1.5배 이상** 가팔라지는 **가속 구간** 탐지 (5일이라는 값은 가변 - 파라미터)

### 3) 외인-기관 동조율 (Alignment)
- 외국인이 살 때 기관(특히 연기금)이 함께 산 비율을 점수화하여 **겹침의 강도**를 측정합니다.

---

## ④ 4단계: 통합 스코어링 - "종목 랭킹 바구니"

350개 종목을 하나의 점수로 줄 세우는 대신, 목적별 3개 바구니로 분류하여 제공합니다.

### 1) 지속 매집형 (Steady Accumulation)
- 3개월~2년 장기 수급 점수가 높고 일관성이 있는 종목
- (저가 매수 타겟)

### 2) 모멘텀 돌파형 (Supply Breakout)
- 1일~1주일 단기 수급 가속도가 붙고 MA 골든크로스가 발생한 종목
- (추격 매수/단기 타겟)

### 3) 수급 다이버전스 (Price-Supply Divergence)
- 외국인은 강력 매집 중이나 주가는 아직 박스권인 종목
- (베스트 픽)

---

# 사용자 경험(UX)을 위한 최종 제언: "1줄 신호 카드"

랭킹 리스트에서 종목을 클릭하기 전, 다음과 같은 예시의 요약 정보가 보이도록 구성하는 것이 핵심입니다.

- **[종목명] 삼성전자**
  - **수급 상태:** 외국인 6개월 지속 매집 중 (유통물량 대비 상위 3%)
  - **최근 변화:** 기관(연기금) 동조율 85% 급증, 수급 5/20 MA 골든크로스 2일째
  - **체크포인트:** 외인 매집 평단가 대비 현재가 -2% (안전 마진 확보)

think about it step-by-step

## [TODO List] - END