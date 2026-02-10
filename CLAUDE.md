# 한국 주식 외국인/기관 투자자 수급 분석 프로그램

## [Status]
- 현재 작업: **Stage 1 완료! Stage 2 준비 중** ✅
- 마지막 업데이트: 2026-02-10
- 다음 시작점: Stage 2 - 시공간 히트맵 시각화 (8개 기간)
- 현재 브랜치: main
- **Stage 1 성과**: 데이터 정규화 완료, Sff/Z-Score 분석 가능, 이상 수급 탐지 20건

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

## [Next Steps]
1. ~~유통물량(Free Float) 데이터 수령 및 DB 스키마 확장~~ ✅ 완료
2. ~~1단계 구현: 데이터 정규화 (Sff, Z-Score 계산 모듈)~~ ✅ 완료
3. **2단계 구현: 시공간 히트맵 시각화 (8개 기간)** ← 다음 작업
4. 3단계 구현: 이벤트 센서 (MA 골든크로스, 가속도, 동조율)
5. 4단계 구현: 통합 스코어링 및 3개 바구니 분류

## [Tech Stack]
- Python 3.10+
- 데이터베이스: SQLite (내장)
- 데이터 수집: pandas, openpyxl (엑셀 파일), **FinanceDataReader (주가 크롤링)**, **BeautifulSoup (유통주식 크롤링)**
- 데이터 분석: pandas, numpy, SQL
- 시각화: matplotlib, seaborn (Stage 2에서 추가 예정)
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
│   │   └── connection.py         # 연결 관리
│   ├── data_collector/           # 데이터 수집 모듈 ✅
│   │   └── excel_collector.py    # 엑셀 파싱 (주가/유통주식 포함)
│   ├── data_loader/              # 데이터 로더 모듈 ✅
│   │   └── validator.py          # 데이터 검증
│   ├── analyzer/                 # 분석 모듈 ✅
│   │   ├── normalizer.py         # Sff, Z-Score 정규화
│   │   └── __init__.py
│   └── visualizer/               # 시각화 모듈 (Stage 2에서 개발)
├── scripts/                      # 스크립트
│   ├── load_initial_data.py      # 초기 데이터 로드 ✅
│   ├── load_daily_data.py        # 일별 증분 업데이트 ✅
│   ├── load_price_volume_backfill.py  # 주가/유통주식 백필 ✅
│   ├── crawl_stock_prices.py     # 주가 크롤러 (FinanceDataReader) ✅
│   ├── crawl_free_float.py       # 유통주식 크롤러 (FnGuide) ✅
│   ├── crawl_all_data.py         # 통합 크롤러 ✅
│   ├── analysis/                 # 분석 스크립트 ✅
│   │   ├── top_net_buyers.py    # 순매수 상위 종목
│   │   ├── top_net_buyers_by_mcap_ratio.py  # 시총 대비 순매수 비중 상위
│   │   └── abnormal_supply_detector.py  # 이상 수급 탐지기 ✅
│   ├── migrations/               # DB 마이그레이션 ✅
│   │   └── migrate_add_columns.py  # 5개 컬럼 추가
│   └── maintenance/              # 유지보수 스크립트 ✅
│       ├── migrate_to_won_unit.py  # 단위 변환 마이그레이션
│       └── swap_foreign_institution.py  # 컬럼 swap 마이그레이션
└── tests/                        # 테스트 코드 (향후 개발)
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
- **Crawling**: FinanceDataReader (주가), FnGuide (유통주식)
- **Database**: SQLite (investor_data.db)
  - **172,155 레코드** (2024-01-02 ~ 2026-01-20)
  - **171,227 레코드** 완전한 데이터 (주가+유통주식, 99.5% 커버리지)
  - 345개 핵심 종목 (KOSPI200 + KOSDAQ150)
  - 1,609개 종목 마스터 데이터

## [Database Schema]
- **markets**: 시장 구분 (KOSPI200, KOSDAQ150)
- **stocks**: 종목 마스터 (종목코드, 종목명, 시장ID)
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
python scripts/load_initial_data.py

# 3. 마이그레이션 (5개 컬럼 추가)
python scripts/migrations/migrate_add_columns.py
```

### **데이터 수집 (크롤링)**
```bash
# 통합 크롤러 (주가 + 유통주식, 약 26분 소요)
python scripts/crawl_all_data.py --start 2024-01-01

# 또는 개별 실행
python scripts/crawl_stock_prices.py --start 2024-01-01  # 주가 (12분)
python scripts/crawl_free_float.py                        # 유통주식 (2분)
```

### **일별 데이터 업데이트**
```bash
# 엑셀 파일 사용
python scripts/load_daily_data.py data/투자자수급_20260210.xlsx data/시가총액_20260210.xlsx

# 주가/유통주식 포함
python scripts/load_daily_data.py data/투자자수급_20260210.xlsx data/시가총액_20260210.xlsx \
    --price-file data/주가_20260210.xlsx --ff-file data/유통주식_20260210.xlsx
```

### **이상 수급 분석 (Stage 1)**
```bash
# 기본 분석 (임계값 2.0, 상위 20개)
python scripts/analysis/abnormal_supply_detector.py

# 매수 시그널만 (임계값 2.5)
python scripts/analysis/abnormal_supply_detector.py --direction buy --threshold 2.5 --top 30

# 매도 시그널
python scripts/analysis/abnormal_supply_detector.py --direction sell
```

### **Python API 사용**
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
print(df_abnormal[['stock_name', 'combined_zscore', 'combined_sff']])

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