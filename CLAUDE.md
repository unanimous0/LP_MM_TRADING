# 한국 주식 외국인/기관 투자자 수급 분석 프로그램

## [Status]
- 현재 작업: 데이터 분석 기반 구축 완료 ✅
- 마지막 업데이트: 2026-02-09
- 다음 시작점: 데이터 분석 심화 및 시각화
- 현재 브랜치: main

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

## [Next Steps]
1. 데이터 분석 심화 (추가 분석 지표 개발)
2. 시각화 모듈 개발 (matplotlib/seaborn)
3. 자동화 스케줄링 (일별 데이터 자동 수집)

## [Tech Stack]
- Python 3.10+
- 데이터베이스: SQLite (내장)
- 데이터 수집: pandas, openpyxl (엑셀 파일)
- 데이터 분석: pandas, numpy, SQL
- 시각화: matplotlib, seaborn (추후 추가 가능)
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
│   │   ├── schema.py             # 스키마 정의 및 생성
│   │   └── connection.py         # 연결 관리
│   ├── data_collector/           # 데이터 수집 모듈 ✅
│   │   └── excel_collector.py    # 엑셀 파싱
│   ├── data_loader/              # 데이터 로더 모듈 ✅
│   │   └── validator.py          # 데이터 검증
│   ├── analyzer/                 # 분석 모듈 (향후 개발)
│   └── visualizer/               # 시각화 모듈 (향후 개발)
├── scripts/                      # 스크립트
│   ├── load_initial_data.py      # 초기 데이터 로드 ✅
│   ├── load_daily_data.py        # 일별 증분 업데이트 ✅
│   ├── analysis/                 # 분석 스크립트 ✅
│   │   ├── top_net_buyers.py    # 순매수 상위 종목
│   │   └── top_net_buyers_by_mcap_ratio.py  # 시총 대비 순매수 비중 상위
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
- **Current**: 엑셀 파일 (투자자수급_200_150.xlsx, 시가총액_200_150.xlsx)
- **Database**: SQLite (investor_data.db)
  - 172,155 레코드 (2024-01-02 ~ 2026-01-20)
  - 345개 종목 (KOSPI200 + KOSDAQ150)
  - 1,609개 종목 마스터 데이터

## [Database Schema]
- **markets**: 시장 구분 (KOSPI200, KOSDAQ150)
- **stocks**: 종목 마스터 (종목코드, 종목명, 시장ID)
- **investor_flows**: 투자자 수급 데이터 (외국인/기관 순매수량/금액, 시가총액)
- **Indexes**: 3개 (stock_code+trade_date, trade_date+stock_code, trade_date)

## [How to Use Database]
```bash
# 1. 데이터베이스 생성
python -c "from src.database.schema import create_database; create_database()"

# 2. 초기 데이터 로드
python scripts/load_initial_data.py

# 3. 일별 데이터 추가 (신규!)
python scripts/load_daily_data.py data/투자자수급_20260209.xlsx data/시가총액_20260209.xlsx

# 3-1. 미리보기 모드 (삽입하지 않고 확인만)
python scripts/load_daily_data.py data/투자자수급_20260209.xlsx data/시가총액_20260209.xlsx --dry-run

# 4. 데이터 검증
python -c "from src.data_loader.validator import validate_data; from src.database.connection import get_connection; conn = get_connection(); validate_data(conn); conn.close()"

# 5. 데이터 조회 (Python)
from src.database.connection import get_connection
import pandas as pd

conn = get_connection()
df = pd.read_sql("SELECT * FROM investor_flows WHERE stock_code = '005930' ORDER BY trade_date DESC LIMIT 10", conn)
conn.close()
```

자세한 내용은 `DATABASE_README.md` 참조
