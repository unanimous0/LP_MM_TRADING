#!/bin/bash
# Materialized View 리프레시 스크립트
# 장 마감 후 데이터 업데이트 이후 1회 실행
#
# 사용법:
#   bash scripts/refresh_mv.sh
#   또는 crontab에 등록: 30 16 * * 1-5 /path/to/scripts/refresh_mv.sh

set -e

DB_HOST="${KOREA_STOCK_DB_HOST:-localhost}"
DB_PORT="${KOREA_STOCK_DB_PORT:-5432}"
DB_NAME="korea_stock_data"

echo "$(date '+%Y-%m-%d %H:%M:%S') MV 리프레시 시작..."
DB_USER="${KOREA_STOCK_DB_USER:-una0}"

psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" \
    -c "REFRESH MATERIALIZED VIEW mv_daily_sff;"
echo "$(date '+%Y-%m-%d %H:%M:%S') MV 리프레시 완료"
