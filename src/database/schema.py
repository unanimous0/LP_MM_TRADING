"""
데이터베이스 스키마 생성 모듈

SQLite 데이터베이스 및 테이블 구조를 정의하고 생성합니다.
- markets: 시장 구분 (KOSPI200, KOSDAQ150)
- stocks: 종목 마스터
- investor_flows: 투자자 수급 데이터 (메인 테이블)
"""

import sqlite3
from pathlib import Path


def create_database(db_path: str = 'data/processed/investor_data.db'):
    """
    데이터베이스 및 스키마 생성

    Args:
        db_path: 데이터베이스 파일 경로
    """
    # 디렉토리 생성
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Foreign Key 활성화
    cursor.execute('PRAGMA foreign_keys = ON')

    # Markets 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS markets (
            market_id INTEGER PRIMARY KEY,
            market_name TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Stocks 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stocks (
            stock_code TEXT PRIMARY KEY,
            stock_name TEXT NOT NULL,
            market_id INTEGER NOT NULL,
            sector TEXT,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (market_id) REFERENCES markets(market_id),
            CHECK (length(stock_code) = 6)
        )
    ''')

    # Investor Flows 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS investor_flows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date DATE NOT NULL,
            stock_code TEXT NOT NULL,
            foreign_net_volume BIGINT,
            foreign_net_amount BIGINT,
            institution_net_volume BIGINT,
            institution_net_amount BIGINT,
            market_cap BIGINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (stock_code) REFERENCES stocks(stock_code),
            UNIQUE(trade_date, stock_code)
        )
    ''')

    # 인덱스 생성
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_stocks_market ON stocks(market_id, is_active)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_flows_stock_date ON investor_flows(stock_code, trade_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_flows_date_stock ON investor_flows(trade_date, stock_code)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_flows_date ON investor_flows(trade_date)')

    # 초기 데이터: Markets
    cursor.execute('''
        INSERT OR IGNORE INTO markets (market_id, market_name, description)
        VALUES
            (1, 'KOSPI200', 'KOSPI 대형주 200종목'),
            (2, 'KOSDAQ150', 'KOSDAQ 우량주 150종목')
    ''')

    conn.commit()
    conn.close()

    print(f'[OK] Database created: {db_path}')


if __name__ == '__main__':
    create_database()
