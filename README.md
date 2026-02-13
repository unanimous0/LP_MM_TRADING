# 한국 주식 외국인/기관 투자자 수급 분석 프로그램

한국 주식 시장의 외국인 및 기관 투자자 수급 데이터를 통계적으로 정규화하고 패턴 분류하여 투자 인사이트를 제공하는 Python 프로그램입니다.

## 프로젝트 개요

Z-Score 정규화를 통해 시총/변동성 왜곡을 제거하고, 6개 기간(1주~2년) 수급 트렌드를 분석하여 3가지 투자 패턴(전환돌파형/지속매집형/조정반등형)으로 종목을 자동 분류합니다.

## 주요 기능 (Stage 1~3 완료 ✅)

- ✅ **데이터 정규화**: Sff(Supply Float Factor) + Z-Score 변환
- ✅ **시공간 분석**: 6개 기간 히트맵 + 4가지 정렬 모드
- ✅ **패턴 분류**: 3개 바구니 자동 분류 (0~100점 점수화)
- ✅ **시그널 탐지**: MA 골든크로스, 수급 가속도, 동조율
- ✅ **통합 리포트**: CSV + Excel (6개 시트) 자동 생성
- ⏳ **백테스팅**: 과거 수익률 검증 (Stage 4, 개발 예정)

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/unanimous0/Investor_Analysis.git
cd Investor_Analysis
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 빠른 시작

### 1. 전체 파이프라인 실행
```bash
# CSV + Excel 리포트 자동 생성
bash scripts/analysis/run_all.sh
```

### 2. 개별 분석 실행
```bash
# Stage 1: 이상 수급 탐지
python scripts/analysis/abnormal_supply_detector.py

# Stage 2: 히트맵 생성
python scripts/analysis/heatmap_generator.py

# Stage 3: 통합 레짐 스캐너
python scripts/analysis/regime_scanner.py --save-csv --print-cards --top 20
```

자세한 사용법은 **IMPLEMENTATION_GUIDE.md** 참조

## 분석 결과 (2026-02-13 기준)

- **분석 종목**: 338개 (KOSPI200 + KOSDAQ150)
- **분석 기간**: 2024-01-02 ~ 2026-01-20 (약 2년)
- **핵심 발견**:
  - 고득점(70점+) 종목의 95.5%가 지속매집형 → 중장기 수급이 핵심
  - 시그널 2개 이상 종목 평균 점수 +16.4% (강한 진입 타이밍)
  - 실전 투자 대상: 44개 (13%, 70점+ 지속매집형)

자세한 분석 결과는 **ANALYSIS_REPORT.md** 참조

## 프로젝트 구조

```
LP_MM_TRADING/
├── CLAUDE.md                   # 프로젝트 상태 및 진행 현황
├── README.md                   # 프로젝트 소개 (이 파일)
├── ANALYSIS_REPORT.md          # 분석 결과 보고서 (통계, 인사이트)
├── IMPLEMENTATION_GUIDE.md     # 구현 가이드 (Stage 1~3 사용법)
├── DATABASE_README.md          # 데이터베이스 스키마
├── requirements.txt            # 의존성 목록
├── data/processed/            # SQLite DB (171,227 레코드)
├── src/                       # 소스 코드
│   ├── database/              # DB 연결 및 스키마
│   ├── analyzer/              # 분석 모듈 (정규화, 패턴 분류, 시그널)
│   └── visualizer/            # 시각화 모듈 (히트맵)
├── scripts/                   # 실행 스크립트
│   ├── analysis/              # 분석 CLI 도구
│   ├── crawlers/              # 데이터 크롤러
│   └── loaders/               # 데이터 로더
└── tests/                     # 테스트 (105개, 100% 통과)
```

## 기술 스택

- Python 3.10+
- SQLite (데이터베이스)
- pandas, numpy (데이터 분석)
- matplotlib, seaborn (시각화)
- openpyxl (Excel 리포트)
- pytest (테스트)

## 문서 가이드

| 문서 | 용도 |
|------|------|
| **CLAUDE.md** | 프로젝트 상태, 진행 현황, Quick Start, 다음 단계 |
| **ANALYSIS_REPORT.md** | 분석 결과, 통계, 인사이트, 실전 투자 전략 |
| **IMPLEMENTATION_GUIDE.md** | Stage 1~3 상세 구현, API 사용법, 코드 예시 |
| **DATABASE_README.md** | 데이터베이스 스키마, 테이블 구조, 쿼리 예시 |

## 라이선스

MIT License

## 작성자

unanimous0

---

**마지막 업데이트**: 2026-02-13 (Stage 3 완료 + Excel 리포트 추가)
