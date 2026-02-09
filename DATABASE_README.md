# Database Architecture Documentation

## Overview

This project uses SQLite database to store and manage investor flow data for KOSPI200 and KOSDAQ150 stocks.

**Database File:** `data/processed/investor_data.db`

---

## Database Schema

### Tables

#### 1. markets
시장 구분 테이블 (KOSPI200, KOSDAQ150)

| Column | Type | Description |
|--------|------|-------------|
| market_id | INTEGER (PK) | 시장 ID (1=KOSPI200, 2=KOSDAQ150) |
| market_name | TEXT (UNIQUE) | 시장명 |
| description | TEXT | 설명 |
| created_at | TIMESTAMP | 생성일시 |

#### 2. stocks
종목 마스터 테이블

| Column | Type | Description |
|--------|------|-------------|
| stock_code | TEXT (PK) | 종목코드 (6자리) |
| stock_name | TEXT | 종목명 |
| market_id | INTEGER (FK) | 시장 ID |
| sector | TEXT | 섹터 (추후 추가) |
| is_active | BOOLEAN | 활성 여부 |
| created_at | TIMESTAMP | 생성일시 |

#### 3. investor_flows
투자자 수급 데이터 (메인 테이블)

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER (PK) | 자동 증가 ID |
| trade_date | DATE | 거래일 |
| stock_code | TEXT (FK) | 종목코드 |
| foreign_net_volume | BIGINT | 외국인 순매수 수량 |
| foreign_net_amount | BIGINT | 외국인 순매수 금액 |
| institution_net_volume | BIGINT | 기관 순매수 수량 |
| institution_net_amount | BIGINT | 기관 순매수 금액 |
| market_cap | BIGINT | 시가총액 |
| created_at | TIMESTAMP | 생성일시 |

**단위:**
- foreign_net_volume, institution_net_volume: 주 (원 단위 환산)
- foreign_net_amount, institution_net_amount: 원 (₩)
- market_cap: 원 (₩)

**주의:** 엑셀 원본 파일은 천원(1,000원) 단위이지만,
데이터베이스에는 원 단위로 변환되어 저장됩니다.
(2026-02-09 마이그레이션 완료)

**Constraints:**
- UNIQUE(trade_date, stock_code) - 중복 방지

**Indexes:**
- idx_flows_stock_date: (stock_code, trade_date)
- idx_flows_date_stock: (trade_date, stock_code)
- idx_flows_date: (trade_date)

---

## Usage

### 1. Create Database Schema
```python
from src.database.schema import create_database

create_database()
```

### 2. Load Initial Data
```bash
python scripts/load_initial_data.py
```

### 3. Load Daily Data (Incremental Update)
```bash
# Load new daily data
python scripts/load_daily_data.py data/투자자수급_20260209.xlsx data/시가총액_20260209.xlsx

# Preview without inserting (dry-run mode)
python scripts/load_daily_data.py data/투자자수급_20260209.xlsx data/시가총액_20260209.xlsx --dry-run
```

**Features:**
- Automatically skips duplicate records (based on trade_date + stock_code)
- Case-insensitive stock name mapping
- Detailed report showing inserted/skipped/failed records
- Safe to run multiple times (idempotent)

### 4. Query Data
```python
from src.database.connection import get_connection
import pandas as pd

conn = get_connection()

# Example: Get Samsung Electronics data for the last 30 days
df = pd.read_sql("""
    SELECT
        s.stock_name,
        f.trade_date,
        f.foreign_net_volume,
        f.foreign_net_amount,
        f.institution_net_volume,
        f.institution_net_amount,
        f.market_cap
    FROM investor_flows f
    JOIN stocks s ON f.stock_code = s.stock_code
    WHERE f.stock_code = '005930'
    ORDER BY f.trade_date DESC
    LIMIT 30
""", conn)

conn.close()
```

### 5. Validate Data
```python
from src.data_loader.validator import validate_data
from src.database.connection import get_connection

conn = get_connection()
validate_data(conn)
conn.close()
```

---

## Current Data Status

- **Total Stocks:** 1,609 (from 종목코드_이름_맵핑.xlsx)
- **Active Stocks in Data:** 345 (KOSPI200 + KOSDAQ150)
- **Total Records:** 172,155
- **Date Range:** 2024-01-02 ~ 2026-01-20 (approximately 500 trading days)
- **Database Size:** ~50MB

---

## Query Examples

### Get top 10 stocks by foreign net buying
```sql
SELECT
    s.stock_name,
    SUM(f.foreign_net_amount) as total_foreign_buying
FROM investor_flows f
JOIN stocks s ON f.stock_code = s.stock_code
WHERE f.trade_date >= '2026-01-01'
GROUP BY f.stock_code, s.stock_name
ORDER BY total_foreign_buying DESC
LIMIT 10;
```

### Get daily summary for a specific date
```sql
SELECT
    COUNT(DISTINCT stock_code) as stock_count,
    SUM(foreign_net_amount) as total_foreign,
    SUM(institution_net_amount) as total_institution
FROM investor_flows
WHERE trade_date = '2026-01-20';
```

### Calculate 5-day moving average
```sql
SELECT
    trade_date,
    stock_code,
    foreign_net_amount,
    AVG(foreign_net_amount) OVER (
        PARTITION BY stock_code
        ORDER BY trade_date
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) as ma_5day
FROM investor_flows
WHERE stock_code = '005930'
ORDER BY trade_date DESC
LIMIT 30;
```

---

## Notes

1. **종목코드 패딩:** All stock codes are 6 digits (e.g., '000660' not '660')
2. **NULL 처리:** Some records may have NULL values in volume/amount fields
3. **트랜잭션:** Use `conn.commit()` after bulk inserts
4. **인덱스 유지:** Run `ANALYZE` periodically for statistics update

---

## Future Enhancements

- [x] Add daily incremental update script
- [ ] Implement sector classification
- [ ] Add data archiving for old records
- [ ] Create query builder utility
- [ ] Add PostgreSQL migration option
- [ ] Add batch processing for multiple daily files
- [ ] Add automatic backup before updates
