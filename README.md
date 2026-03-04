# 한국 주식 외국인/기관 투자자 수급 분석 프로그램

한국 주식 시장의 외국인 및 기관 투자자 수급 데이터를 통계적으로 정규화하고 패턴 분류하여 투자 인사이트를 제공하는 Python 프로그램입니다.

## 프로젝트 개요

Z-Score 정규화를 통해 시총/변동성 왜곡을 제거하고, 7개 기간(5D~500D) 수급 트렌드를 분석하여 3가지 투자 패턴(급등형/지속형/전환형)으로 종목을 자동 분류합니다.

## 주요 기능

- **데이터 정규화**: Sff(Supply Float Factor) + Z-Score 변환 (외국인 중심 조건부 합산)
- **시공간 분석**: 7개 기간 히트맵 + 4가지 정렬 모드
- **패턴 분류**: 3개 바구니 자동 분류 (0~100점 점수화) + 7종 복합 패턴(sub_type)
- **시그널 탐지**: MA 골든크로스, 수급 가속도, 동조율
- **백테스팅**: 롱/숏/병행 전략, Optuna 최적화, Walk-Forward 검증
- **Streamlit 대시보드**: 인터랙티브 차트, 히트맵, 종목 비교, 이상 수급, 백테스트 UI

## 설치 방법

```bash
git clone <repository>
cd LP_MM_TRADING
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 데이터베이스 연결

PostgreSQL 공유 DB 필요 (별도 프로젝트에서 관리):
```bash
# 기본값: korea_stock_reader@localhost:5432/korea_stock_data
# 환경변수로 변경:
export KOREA_STOCK_DB_HOST=your-server
export KOREA_STOCK_DB_PORT=5432
```

## 빠른 시작

### Streamlit 대시보드
```bash
venv/bin/streamlit run app/streamlit_app.py
# → http://localhost:8501
```

### CLI 분석
```bash
# 통합 레짐 스캐너
python scripts/analysis/regime_scanner.py --save-csv --print-cards --top 20

# 백테스트
python scripts/analysis/backtest_runner.py --start 2023-01-01 --end 2025-12-31

# Optuna 최적화
python scripts/analysis/backtest_runner.py --optimize --n-trials 100
```

자세한 사용법은 **ANALYSIS_GUIDE.md** 참조

## 데이터

- **~10,000,000 레코드** (PostgreSQL)
- **2,721개 종목** (KOSPI + KOSDAQ 전 종목)
- **기간**: 2022-01-03 ~ 2026-03-03

## 프로젝트 구조

```
LP_MM_TRADING/
├── CLAUDE.md                   # 프로젝트 상태 및 진행 현황
├── README.md                   # 프로젝트 소개 (이 파일)
├── DATABASE_README.md          # 데이터베이스 스키마
├── requirements.txt            # 의존성 목록
├── data/
│   └── app.db                  # 앱 전용 SQLite (관심종목/백테스트 히스토리)
├── app/                        # Streamlit 웹 대시보드
│   ├── streamlit_app.py        # 메인 엔트리포인트
│   ├── utils/data_loader.py    # 캐시 데이터 로더
│   └── pages/                  # 멀티페이지 (히트맵/패턴분석/백테스트/종목상세 등)
├── src/                        # 소스 코드
│   ├── database/               # DB 연결 (PostgreSQL + SQLite)
│   ├── analyzer/               # 분석 모듈 (정규화, 패턴 분류, 시그널)
│   ├── backtesting/            # 백테스트 (엔진, 최적화, 시각화)
│   └── visualizer/             # 히트맵 시각화
├── scripts/analysis/           # CLI 도구
└── tests/                      # 테스트 (294개, 100% 통과)
```

## 기술 스택

- Python 3.10+
- PostgreSQL (시장 데이터), SQLite (앱 전용 데이터)
- pandas, numpy (데이터 분석)
- Streamlit (웹 대시보드)
- Plotly (인터랙티브 차트)
- matplotlib, seaborn (CLI 시각화)
- Optuna (Bayesian Optimization)
- pytest (테스트)

## 문서 가이드

| 문서 | 용도 |
|------|------|
| **CLAUDE.md** | 프로젝트 상태, 진행 현황, Quick Start, 다음 단계 |
| **ANALYSIS_GUIDE.md** | 분석 사용법 가이드 |
| **DATABASE_README.md** | 데이터베이스 스키마, 테이블 구조, 쿼리 예시 |

## 라이선스

MIT License

## 작성자

unanimous0

---

**마지막 업데이트**: 2026-03-04 (Stage 5-1 완료 + DB 전환 정리)
