# 일별 데이터 업데이트 가이드

## 개요

매일 새로운 엑셀 파일을 받으면 `load_daily_data.py` 스크립트를 사용하여 데이터베이스에 추가할 수 있습니다.

---

## 사용 방법

### 1. 기본 사용

```bash
python scripts/load_daily_data.py <투자자수급_파일.xlsx> <시가총액_파일.xlsx>
```

**예시:**
```bash
python scripts/load_daily_data.py data/투자자수급_20260209.xlsx data/시가총액_20260209.xlsx
```

### 2. 미리보기 모드 (Dry-run)

데이터를 삽입하지 않고 어떤 데이터가 로드될지 미리 확인:

```bash
python scripts/load_daily_data.py data/투자자수급_20260209.xlsx data/시가총액_20260209.xlsx --dry-run
```

---

## 출력 예시

### 정상 처리

```
======================================================================
Daily Data Load Started
======================================================================

[INFO] Loading data from Excel files...
[OK]   Loaded 350 records

[INFO] Mapping stock names to stock codes...
[OK]   Mapped 345 records

[INFO] Inserting data into database...

======================================================================
Daily Data Load Report
======================================================================
[INFO] Source file: 투자자수급_20260209.xlsx
[INFO] Date range: 2026-02-09 ~ 2026-02-09
[INFO] Unique stocks: 345
[INFO] Total records in file: 350

[OK]   Records inserted: 345
[INFO] Records skipped (duplicates): 0
[WARN] Records unmapped: 5

[INFO] Total records in database: 172,500
[INFO] Latest date in database: 2026-02-09
======================================================================

[SUCCESS] Daily data load completed successfully
```

### 중복 데이터 재실행

같은 파일을 다시 실행하면:

```
[OK]   Records inserted: 0
[INFO] Records skipped (duplicates): 345
```

---

## 주요 기능

### 1. 중복 방지

- 데이터베이스의 UNIQUE 제약 조건(trade_date + stock_code)을 활용
- INSERT OR IGNORE 방식으로 중복 데이터 자동 건너뜀
- 같은 파일을 여러 번 실행해도 안전

### 2. 대소문자 무시 매핑

- 종목명 매핑 시 대소문자 구분 없이 처리
- 예: 'S-OIL', 'S-Oil', 's-oil' 모두 동일하게 매핑
- 기존 1.4% 매핑 실패율 개선

### 3. 상세한 리포트

- 추가된 레코드 수
- 중복으로 건너뛴 레코드 수
- 매핑 실패한 레코드 수
- 데이터베이스 현황 (총 레코드 수, 최신 날짜)

### 4. 에러 처리

- 파일 존재 여부 검증
- 매핑 실패 시 경고 메시지
- 삽입 실패 시 상세 에러 메시지

---

## 문제 해결

### 매핑 실패 경고

```
[WARN] 5 records could not be mapped to stock_code
       Unmapped stocks: ['NewStock', 'AnotherStock']
```

**원인:** 종목코드 매핑 파일에 없는 종목명

**해결:**
1. `data/종목코드_이름_맵핑.xlsx` 파일에 해당 종목 추가
2. 데이터베이스 재생성 또는 수동으로 stocks 테이블에 추가

### 파일 경로 오류

```
[ERROR] File not found: data/투자자수급_20260209.xlsx
```

**해결:** 파일 경로를 정확히 확인 (상대 경로 또는 절대 경로)

---

## 권장 워크플로우

### 매일 업데이트

```bash
# 1. 새 엑셀 파일을 data/ 폴더에 저장
# 2. 미리보기로 확인 (선택사항)
python scripts/load_daily_data.py data/투자자수급_20260209.xlsx data/시가총액_20260209.xlsx --dry-run

# 3. 실제 데이터 삽입
python scripts/load_daily_data.py data/투자자수급_20260209.xlsx data/시가총액_20260209.xlsx

# 4. 데이터 검증 (선택사항)
python -c "from src.data_loader.validator import validate_data; from src.database.connection import get_connection; conn = get_connection(); validate_data(conn); conn.close()"
```

### Git 커밋 (선택사항)

```bash
git add data/processed/investor_data.db
git commit -m "데이터 업데이트: 2026-02-09"
git push origin main
```

---

## 파일 요구사항

### 필수 시트

엑셀 파일에 다음 시트가 반드시 존재해야 함:

1. **투자자수급 파일:**
   - 'KOSPI200' 시트
   - 'KOSDAQ150' 시트

2. **시가총액 파일:**
   - 'KOSPI200' 시트
   - 'KOSDAQ150' 시트

### 컬럼 형식

기존 엑셀 파일 형식과 동일해야 함 (multi-level header 포함)

---

## 성능

- **처리 시간:** 일반적으로 1초 이내 (350 종목 × 1일 = 350 레코드)
- **데이터베이스 크기 증가:** 일별 약 100KB
- **메모리 사용:** 최대 50MB

---

## 추가 정보

자세한 내용은 다음 문서를 참조하세요:

- `DATABASE_README.md` - 데이터베이스 아키텍처 및 사용법
- `CLAUDE.md` - 프로젝트 전체 상태 및 워크플로우
- `README.md` - 프로젝트 소개
