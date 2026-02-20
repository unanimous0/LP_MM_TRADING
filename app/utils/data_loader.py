"""
Streamlit 캐시 데이터 로더

DB 연결, Stage 1-3 분석 파이프라인, 백테스트 실행을 캐싱하여 성능 확보.
기존 모듈(normalizer, pattern_classifier 등)을 수정 없이 재사용.
"""

import sqlite3
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd
import streamlit as st

# 프로젝트 루트 경로 등록
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DEFAULT_CONFIG
from src.database.connection import DB_PATH
from src.analyzer.normalizer import SupplyNormalizer
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
from src.analyzer.pattern_classifier import PatternClassifier
from src.analyzer.signal_detector import SignalDetector
from src.analyzer.integrated_report import IntegratedReport
from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.metrics import PerformanceMetrics
from src.backtesting.portfolio import Trade


# ---------------------------------------------------------------------------
# DB 연결 (싱글턴)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db_connection() -> sqlite3.Connection:
    """Streamlit 스레드 안전 DB 연결 (check_same_thread=False)"""
    db_path = str(_PROJECT_ROOT / DB_PATH)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


# ---------------------------------------------------------------------------
# 정적 데이터 (종목/섹터/날짜 범위)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_stock_list() -> pd.DataFrame:
    """종목 리스트 (stock_code, stock_name, sector)"""
    conn = get_db_connection()
    df = pd.read_sql_query(
        "SELECT stock_code, stock_name, sector FROM stocks ORDER BY stock_code",
        conn,
    )
    return df


@st.cache_data(ttl=3600)
def get_sectors() -> List[str]:
    """고유 섹터 목록"""
    conn = get_db_connection()
    rows = conn.execute(
        "SELECT DISTINCT sector FROM stocks WHERE sector IS NOT NULL ORDER BY sector"
    ).fetchall()
    return [r[0] for r in rows]


@st.cache_data(ttl=3600)
def get_date_range() -> Tuple[str, str]:
    """DB 내 거래 날짜 범위 (min_date, max_date)"""
    conn = get_db_connection()
    row = conn.execute(
        "SELECT MIN(trade_date), MAX(trade_date) FROM investor_flows"
    ).fetchone()
    return row[0], row[1]


# ---------------------------------------------------------------------------
# Stage 1-3 분석 파이프라인
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner="분석 중...")
def run_analysis_pipeline(
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stage 1-3 전체 파이프라인 실행

    Returns:
        (zscore_matrix, classified_df, signals_df, report_df)
    """
    conn = get_db_connection()

    # Stage 1: 정규화
    normalizer = SupplyNormalizer(conn)

    # Stage 2: 멀티 기간 Z-Score
    calculator = OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)
    zscore_matrix = calculator.calculate_multi_period_zscores(
        DEFAULT_CONFIG['periods'], end_date=end_date
    )
    zscore_matrix = zscore_matrix.reset_index()

    if zscore_matrix.empty:
        empty = pd.DataFrame()
        return zscore_matrix, empty, empty, empty

    # Stage 3: 패턴 분류
    classifier = PatternClassifier()
    classified_df = classifier.classify_all(zscore_matrix)

    # Stage 3: 시그널 탐지
    detector = SignalDetector(conn)
    signals_df = detector.detect_all_signals(end_date=end_date)

    # Stage 3: 통합 리포트
    report_gen = IntegratedReport(conn)
    report_df = report_gen.generate_report(classified_df, signals_df)

    return zscore_matrix, classified_df, signals_df, report_df


# ---------------------------------------------------------------------------
# 백테스트 실행
# ---------------------------------------------------------------------------

def _serialize_trades(trades: List[Trade]) -> List[dict]:
    """Trade 객체 리스트 → dict 리스트 (캐싱 가능 형태)"""
    result = []
    for t in trades:
        d = t.to_dict()
        # to_dict()에 profit (property) 포함될 수 있으므로 제거
        d.pop('profit', None)
        result.append(d)
    return result


def _deserialize_trades(trade_dicts: List[dict]) -> List[Trade]:
    """dict 리스트 → Trade 객체 리스트"""
    trades = []
    for d in trade_dicts:
        d_clean = {k: v for k, v in d.items() if k != 'profit'}
        trades.append(Trade(**d_clean))
    return trades


@st.cache_data(ttl=300, show_spinner="백테스트 실행 중...")
def run_backtest(
    start_date: str,
    end_date: str,
    strategy: str = 'long',
    min_score: float = 60,
    min_signals: int = 1,
    target_return: float = 0.15,
    stop_loss: float = -0.075,
    max_hold_days: int = 999,
    initial_capital: float = 10_000_000,
    max_positions: int = 5,
    institution_weight: float = 0.3,
    reverse_threshold: float = 60,
    allowed_patterns: Optional[List[str]] = None,
) -> Dict:
    """
    백테스트 실행 (캐싱)

    Returns:
        {
            'trade_dicts': List[dict],
            'daily_values': DataFrame,
            'config': dict,
            'initial_capital': float,
        }
    """
    conn = get_db_connection()

    config = BacktestConfig(
        initial_capital=initial_capital,
        max_positions=max_positions,
        min_score=min_score,
        min_signals=min_signals,
        target_return=target_return,
        stop_loss=stop_loss,
        max_hold_days=max_hold_days,
        reverse_signal_threshold=reverse_threshold,
        allowed_patterns=allowed_patterns,
        strategy=strategy,
        institution_weight=institution_weight,
        force_exit_on_end=True,
    )

    engine = BacktestEngine(conn, config)
    result = engine.run(
        start_date=start_date,
        end_date=end_date,
        verbose=False,
    )

    return {
        'trade_dicts': _serialize_trades(result['trades']),
        'daily_values': result['daily_values'],
        'config': {
            'initial_capital': config.initial_capital,
            'max_positions': config.max_positions,
            'min_score': config.min_score,
            'min_signals': config.min_signals,
            'target_return': config.target_return,
            'stop_loss': config.stop_loss,
            'max_hold_days': config.max_hold_days,
            'strategy': config.strategy,
        },
        'initial_capital': config.initial_capital,
    }


def get_metrics_from_result(result: Dict) -> Optional[PerformanceMetrics]:
    """캐싱된 백테스트 결과에서 PerformanceMetrics 생성"""
    trades = _deserialize_trades(result['trade_dicts'])
    if not trades:
        return None
    return PerformanceMetrics(
        trades=trades,
        daily_values=result['daily_values'],
        initial_capital=result['initial_capital'],
    )


def get_trades_from_result(result: Dict) -> List[Trade]:
    """캐싱된 백테스트 결과에서 Trade 리스트 복원"""
    return _deserialize_trades(result['trade_dicts'])


# ---------------------------------------------------------------------------
# Optuna 최적화
# ---------------------------------------------------------------------------

def run_optuna_optimization(
    start_date: str,
    end_date: str,
    strategy: str = 'long',
    n_trials: int = 50,
    metric: str = 'sharpe_ratio',
    initial_capital: float = 10_000_000,
    max_positions: int = 5,
    max_hold_days: int = 999,
    reverse_threshold: float = 60,
) -> Optional[Dict]:
    """
    Optuna Bayesian Optimization 실행

    Returns:
        {'params': dict, metric: float, 'total_complete': int, 'total_pruned': int}
        또는 None (완료된 Trial이 없을 때)
    """
    from src.backtesting.optimizer import OptunaOptimizer

    db_path = str(_PROJECT_ROOT / DB_PATH)
    base_config = BacktestConfig(
        strategy=strategy,
        initial_capital=initial_capital,
        max_positions=max_positions,
        max_hold_days=max_hold_days,
        reverse_signal_threshold=reverse_threshold,
        force_exit_on_end=True,
    )

    optimizer = OptunaOptimizer(
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        base_config=base_config,
    )

    return optimizer.optimize(
        n_trials=n_trials,
        metric=metric,
        verbose=False,
    )
