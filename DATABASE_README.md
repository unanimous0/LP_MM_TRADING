# Database Architecture Documentation

## Overview

This project uses **two databases**:
1. **PostgreSQL** (`korea_stock_data`): 시장 데이터 (수급, 주가, 종목 마스터) — 공유 DB, 별도 프로젝트에서 관리
2. **SQLite** (`data/app.db`): 앱 전용 데이터 (관심종목, 백테스트 히스토리, 점수 변동 로그)

---

## PostgreSQL — 시장 데이터

### 연결 설정
```python
from src.database.connection import get_pg_engine

engine = get_pg_engine()  # 싱글턴 엔진
```

환경변수:
- `KOREA_STOCK_DB_HOST` (기본: localhost)
- `KOREA_STOCK_DB_PORT` (기본: 5432)
- 사용자: `korea_stock_reader` (읽기 전용)

### 주요 테이블

#### investor_trading
투자자별 수급 데이터

| Column | Type | Description |
|--------|------|-------------|
| stock_code | TEXT | 종목코드 (6자리) |
| trade_date | DATE | 거래일 |
| investor_type | TEXT | 투자자 유형 (FOREIGN/INSTITUTION/PENSION/RETAIL) |
| net_buy_volume | BIGINT | 순매수 수량 |
| net_buy_amount | BIGINT | 순매수 금액 (원) |

> **참고**: PENSION(연기금)은 INSTITUTION에 이미 포함됨. RETAIL(개인)은 참고용.

#### stock_prices
일별 주가 데이터

| Column | Type | Description |
|--------|------|-------------|
| stock_code | TEXT | 종목코드 |
| trade_date | DATE | 거래일 |
| open_price | INTEGER | 시가 |
| high_price | INTEGER | 고가 |
| low_price | INTEGER | 저가 |
| close_price | INTEGER | 종가 |
| volume | BIGINT | 거래량 |

#### stock_master
종목 마스터 테이블

| Column | Type | Description |
|--------|------|-------------|
| stock_code | TEXT (PK) | 종목코드 |
| stock_name | TEXT | 종목명 |
| market | TEXT | 시장 (KOSPI/KOSDAQ) |
| sector | TEXT | 섹터 |
| shares_outstanding | BIGINT | 유통주식수 |

### Materialized View

#### mv_daily_sff
Sff(Supply Flow Force) 사전 계산 뷰 — 3-table JOIN + GROUP BY 결과를 캐싱

```sql
-- 리프레시 (장 마감 후)
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_sff;
```

리프레시 스크립트: `scripts/refresh_mv.sh`
생성 SQL: `scripts/setup_materialized_views.sql`

### SQL 함수

- `fn_zscore_latest(weight, target_date)`: 7개 기간 Z-Score 일괄 계산
- `fn_signals_latest(weight, target_date)`: MA 크로스오버 + 가속도 + 동조율

---

## SQLite — 앱 전용 데이터

**파일**: `data/app.db`

### 테이블

| 테이블 | 용도 |
|--------|------|
| `watchlist` | 관심종목 저장 |
| `backtest_history` | 백테스트 결과 히스토리 |
| `score_change_log` | 고득점 종목 변동 이벤트 |

---

## Current Data Status

- **Total Records:** ~10,000,000 (PostgreSQL)
- **Active Stocks:** 2,721 (KOSPI + KOSDAQ 전 종목)
- **Date Range:** 2022-01-03 ~ 2026-03-03
- **Stock Master:** 1,609+ (섹터 정보 97.9% 커버리지)

---

## Query Examples

### PostgreSQL 쿼리
```python
from src.database.connection import get_pg_engine
import pandas as pd

engine = get_pg_engine()

# 삼성전자 최근 30일 수급
df = pd.read_sql("""
    SELECT trade_date, investor_type, net_buy_amount
    FROM investor_trading
    WHERE stock_code = '005930'
    ORDER BY trade_date DESC
    LIMIT 30
""", engine)
```

### MV 사용 여부 확인
```python
from src.database.connection import is_mv_available

if is_mv_available():
    print("MV 사용 가능 — 고속 경로")
else:
    print("MV 없음 — 원본 테이블 JOIN 사용")
```

---

## Notes

1. **종목코드 패딩:** 6자리 문자열 (e.g., '000660')
2. **datetime.date 변환:** PostgreSQL은 DATE를 `datetime.date`로 반환 → 필요시 `.astype(str)` 변환
3. **MV 자동 분기:** `is_mv_available()` 결과에 따라 MV/원본 테이블 자동 선택
4. **크롤링 불필요:** 데이터는 공유 DB에서 자동 갱신됨
