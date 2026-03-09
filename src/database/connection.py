"""데이터베이스 연결 관리 모듈

PostgreSQL (한국주식 공유 DB) + SQLite (앱 전용 데이터) 분리 관리.

PostgreSQL: 수급/주가/종목 데이터 (읽기 전용)
SQLite:     watchlist, backtest_history, score_change_log (앱 전용)

환경변수:
    KOREA_STOCK_DB_HOST: PostgreSQL 호스트 (기본: localhost)
    KOREA_STOCK_DB_PORT: PostgreSQL 포트 (기본: 5432)
    KOREA_STOCK_DB_USER: PostgreSQL 유저 (기본: una0)
    KOREA_STOCK_DB_PASS: PostgreSQL 패스워드 (~/.bashrc에 설정)
"""

import os
import sqlite3
from contextlib import contextmanager

from sqlalchemy import create_engine, text


# ---------------------------------------------------------------------------
# PostgreSQL (한국주식 공유 DB) — 읽기 전용
# ---------------------------------------------------------------------------

PG_HOST = os.environ.get("KOREA_STOCK_DB_HOST", "localhost")
PG_PORT = os.environ.get("KOREA_STOCK_DB_PORT", "5432")
PG_USER = os.environ.get("KOREA_STOCK_DB_USER", "una0")
PG_PASS = os.environ.get("KOREA_STOCK_DB_PASS", "")
PG_DSN = f"postgresql+psycopg2://{PG_USER}:{PG_PASS}@{PG_HOST}:{PG_PORT}/korea_stock_data"

_pg_engine = None


def get_pg_engine():
    """PostgreSQL SQLAlchemy Engine (싱글턴, pool_pre_ping 활성화)"""
    global _pg_engine
    if _pg_engine is None:
        _pg_engine = create_engine(PG_DSN, pool_pre_ping=True)
    return _pg_engine


def get_connection():
    """
    Backward compatibility: PostgreSQL SQLAlchemy Connection 반환.
    기존 코드에서 conn.close() 호출 가능.

    Usage:
        conn = get_connection()
        df = pd.read_sql(text(query), conn)
        conn.close()
    """
    return get_pg_engine().connect()


# ---------------------------------------------------------------------------
# Materialized View 존재 확인
# ---------------------------------------------------------------------------

_mv_available = None


def is_mv_available() -> bool:
    """mv_daily_sff 존재 여부 확인 (1회만 체크, 캐싱)"""
    global _mv_available
    if _mv_available is None:
        try:
            with get_pg_engine().connect() as conn:
                conn.execute(text("SELECT 1 FROM mv_daily_sff LIMIT 1"))
                _mv_available = True
        except Exception:
            _mv_available = False
    return _mv_available


# ---------------------------------------------------------------------------
# SQLite (앱 전용 데이터) — watchlist, backtest_history, score_change_log
# ---------------------------------------------------------------------------

APP_DB_PATH = "data/app.db"


def get_app_conn(app_db_path: str = APP_DB_PATH) -> sqlite3.Connection:
    """SQLite 앱 DB 연결 생성"""
    conn = sqlite3.connect(app_db_path)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


@contextmanager
def get_app_db(app_db_path: str = APP_DB_PATH):
    """
    Context Manager로 앱 DB 연결 관리

    Usage:
        with get_app_db() as conn:
            conn.execute("INSERT INTO watchlist ...")
    """
    conn = get_app_conn(app_db_path)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
