"""
데이터베이스 연결 관리 모듈

SQLite 연결을 생성하고 Context Manager를 통해 안전한 트랜잭션 처리를 제공합니다.
"""

import sqlite3
from contextlib import contextmanager

DB_PATH = 'data/processed/investor_data.db'


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    SQLite 연결 생성

    Args:
        db_path: 데이터베이스 파일 경로

    Returns:
        sqlite3.Connection: 데이터베이스 연결 객체
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # 컬럼명으로 접근 가능
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


@contextmanager
def get_db(db_path: str = DB_PATH):
    """
    Context Manager로 안전한 연결 관리

    Usage:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM stocks")

    Args:
        db_path: 데이터베이스 파일 경로

    Yields:
        sqlite3.Connection: 데이터베이스 연결 객체
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
