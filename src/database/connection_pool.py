"""
데이터베이스 연결 풀링 (Connection Pooling)

기존 get_connection()을 대체하여 연결 재사용으로 성능 향상
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

# 싱글톤 패턴으로 연결 풀 관리
_connection = None
_lock = Lock()


def get_db_path() -> Path:
    """데이터베이스 파일 경로 반환"""
    return Path(__file__).parent.parent.parent / 'data' / 'processed' / 'investor_data.db'


def get_connection_pooled():
    """
    연결 풀링을 사용한 데이터베이스 연결

    장점:
    - 연결 재사용으로 오버헤드 감소
    - 스레드 안전 (Lock 사용)
    - 메모리 효율성 향상

    Returns:
        sqlite3.Connection: 데이터베이스 연결 객체

    Example:
        >>> conn = get_connection_pooled()
        >>> df = pd.read_sql("SELECT * FROM stocks", conn)
    """
    global _connection

    with _lock:
        if _connection is None:
            db_path = get_db_path()
            _connection = sqlite3.connect(
                str(db_path),
                check_same_thread=False,  # 멀티스레드 허용
                isolation_level=None      # 자동 커밋
            )
            # 읽기 전용 최적화
            _connection.execute("PRAGMA query_only = ON")
            _connection.execute("PRAGMA temp_store = MEMORY")

    return _connection


@contextmanager
def get_connection_context():
    """
    Context manager for safe connection handling

    Example:
        >>> with get_connection_context() as conn:
        ...     df = pd.read_sql(query, conn)
    """
    conn = get_connection_pooled()
    try:
        yield conn
    except Exception as e:
        print(f"[ERROR] Database error: {e}")
        raise
    # finally에서 close()하지 않음 (연결 재사용)


def close_connection():
    """연결 풀 종료 (프로그램 종료 시)"""
    global _connection

    with _lock:
        if _connection:
            _connection.close()
            _connection = None
