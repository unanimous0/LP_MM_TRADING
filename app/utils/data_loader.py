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
# 이상 수급 탐지 (캐싱)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def get_today_supply_ranking(top_n: int = 50) -> pd.DataFrame:
    """당일 전 종목 외국인/기관 순매수금액 조회 (캐싱)"""
    conn = get_db_connection()
    max_date = conn.execute(
        "SELECT MAX(trade_date) FROM investor_flows"
    ).fetchone()[0]
    df = pd.read_sql_query(
        "SELECT f.stock_code, s.stock_name, s.sector, "
        "f.foreign_net_amount, f.institution_net_amount "
        "FROM investor_flows f "
        "JOIN stocks s ON f.stock_code = s.stock_code "
        "WHERE f.trade_date = ?",
        conn,
        params=[max_date],
    )
    return df


@st.cache_data(ttl=600, show_spinner=False)
def get_abnormal_supply_data(
    end_date: Optional[str] = None,
    threshold: float = 2.0,
    top_n: int = 10,
    direction: str = 'both',
    institution_weight: float = 0.3,
    z_score_window: int = 60,
) -> pd.DataFrame:
    """이상 수급 종목 조회 (캐싱) — 순매수금액 포함"""
    conn = get_db_connection()
    normalizer = SupplyNormalizer(conn, config={
        'z_score_window': z_score_window,
        'min_data_points': 30,
        'institution_weight': institution_weight,
    })
    df = normalizer.get_abnormal_supply(
        threshold=threshold,
        end_date=end_date,
        top_n=top_n,
        direction=direction,
    )
    if df.empty:
        return df

    # 순매수금액 조인
    trade_date = df['trade_date'].iloc[0]
    codes = df['stock_code'].tolist()
    placeholders = ','.join('?' for _ in codes)
    amounts = pd.read_sql_query(
        f"SELECT stock_code, foreign_net_amount, institution_net_amount "
        f"FROM investor_flows WHERE trade_date = ? AND stock_code IN ({placeholders})",
        conn,
        params=[trade_date] + codes,
    )
    df = df.merge(amounts, on='stock_code', how='left')
    return df


# ---------------------------------------------------------------------------
# Stage 1-3 분석 파이프라인 (단계별 캐시 분리)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def _stage_zscore(end_date: Optional[str] = None, institution_weight: float = 0.3) -> pd.DataFrame:
    """Stage 1+2: 수급 정규화 + 멀티 기간 Z-Score"""
    conn = get_db_connection()
    normalizer = SupplyNormalizer(conn, config={
        'z_score_window': 60,
        'min_data_points': 30,
        'institution_weight': institution_weight,
    })
    calculator = OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)
    zscore_matrix = calculator.calculate_multi_period_zscores(
        DEFAULT_CONFIG['periods'], end_date=end_date
    )
    return zscore_matrix.reset_index()


@st.cache_data(ttl=600, show_spinner=False)
def _stage_classify(end_date: Optional[str] = None, institution_weight: float = 0.3) -> pd.DataFrame:
    """Stage 3a: 패턴 분류"""
    zscore_matrix = _stage_zscore(end_date=end_date, institution_weight=institution_weight)
    if zscore_matrix.empty:
        return pd.DataFrame()
    classifier = PatternClassifier()
    return classifier.classify_all(zscore_matrix)


@st.cache_data(ttl=600, show_spinner=False)
def _stage_signals(end_date: Optional[str] = None, institution_weight: float = 0.3) -> pd.DataFrame:
    """Stage 3b: 시그널 탐지"""
    conn = get_db_connection()
    detector = SignalDetector(conn, institution_weight=institution_weight)
    return detector.detect_all_signals(end_date=end_date)


@st.cache_data(ttl=600, show_spinner=False)
def _stage_report(end_date: Optional[str] = None, institution_weight: float = 0.3) -> pd.DataFrame:
    """Stage 3c: 통합 리포트"""
    classified_df = _stage_classify(end_date=end_date, institution_weight=institution_weight)
    signals_df = _stage_signals(end_date=end_date, institution_weight=institution_weight)
    if classified_df.empty:
        return pd.DataFrame()
    conn = get_db_connection()
    report_gen = IntegratedReport(conn)
    return report_gen.generate_report(classified_df, signals_df)


def run_analysis_pipeline(
    end_date: Optional[str] = None,
    institution_weight: float = 0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stage 1-3 전체 파이프라인 (progress bar 없는 버전)"""
    return run_analysis_pipeline_with_progress(
        end_date=end_date, progress_bar=None,
        institution_weight=institution_weight,
    )


def run_analysis_pipeline_with_progress(
    end_date: Optional[str] = None,
    progress_bar=None,
    institution_weight: float = 0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stage 1-3 전체 파이프라인 (단계별 진행률 표시 지원)

    Args:
        end_date: 분석 기준 날짜
        progress_bar: st.progress 위젯 (None이면 진행률 표시 안 함)
        institution_weight: 기관 가중치 (0.0=외국인만, 0.3=기본, 1.0=동등)

    Returns:
        (zscore_matrix, classified_df, signals_df, report_df)
    """
    def _upd(pct: float, msg: str):
        if progress_bar is not None:
            progress_bar.progress(pct, text=msg)

    _upd(0.05, "📐 수급 데이터 정규화 중... 5%")
    zscore_matrix = _stage_zscore(end_date=end_date, institution_weight=institution_weight)

    if zscore_matrix.empty:
        _upd(1.0, "✅ 완료 100%")
        empty = pd.DataFrame()
        return zscore_matrix, empty, empty, empty

    _upd(0.40, "📊 Z-Score 계산 완료 → 패턴 분류 중... 40%")
    classified_df = _stage_classify(end_date=end_date, institution_weight=institution_weight)

    _upd(0.65, "🔍 패턴 분류 완료 → 시그널 탐지 중... 65%")
    signals_df = _stage_signals(end_date=end_date, institution_weight=institution_weight)

    _upd(0.75, "📡 시그널 탐지 완료 → 리포트 생성 중... 75%")
    report_df = _stage_report(end_date=end_date, institution_weight=institution_weight)

    _upd(0.85, "📋 리포트 생성 완료 85%")
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
    tax_rate: float = 0.0020,
    commission_rate: float = 0.00015,
    slippage_rate: float = 0.001,
    borrowing_rate: float = 0.03,
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
        tax_rate=tax_rate,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        borrowing_rate=borrowing_rate,
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


def run_backtest_with_progress(
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
    allowed_patterns=None,
    progress_callback=None,
    tax_rate: float = 0.0020,
    commission_rate: float = 0.00015,
    slippage_rate: float = 0.001,
    borrowing_rate: float = 0.03,
) -> Dict:
    """백테스트 실행 (캐시 없음, progress_callback 지원)"""
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
        tax_rate=tax_rate,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        borrowing_rate=borrowing_rate,
    )
    engine = BacktestEngine(conn, config)
    result = engine.run(
        start_date=start_date,
        end_date=end_date,
        verbose=False,
        progress_callback=progress_callback,
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


# ---------------------------------------------------------------------------
# Optuna 최적화
# ---------------------------------------------------------------------------

# Optuna study 누적 저장 경로 (메인 DB와 별개의 경량 SQLite)
_OPTUNA_STORAGE = f"sqlite:///{_PROJECT_ROOT / 'data' / 'optuna_studies.db'}"


def get_optuna_trial_count(
    start_date: str,
    end_date: str,
    strategy: str = 'long',
    metric: str = 'sharpe_ratio',
) -> int:
    """저장된 Optuna study의 누적 완료 Trial 수 반환 (study 없으면 0)"""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sd = start_date.replace('-', '')
        ed = end_date.replace('-', '')
        study_name = f"opt__{strategy}__{sd}__{ed}__{metric}"
        study = optuna.load_study(study_name=study_name, storage=_OPTUNA_STORAGE)
        return sum(1 for t in study.trials if t.state.name == 'COMPLETE')
    except Exception:
        return 0


def run_optuna_optimization(
    start_date: str,
    end_date: str,
    strategy: str = 'long',
    n_trials: int = 100,
    metric: str = 'sharpe_ratio',
    initial_capital: float = 10_000_000,
    max_positions: int = 5,
    max_hold_days: int = 999,
    reverse_threshold: float = 60,
    institution_weight: float = 0.3,
    progress_callback=None,
    reset_study: bool = False,
    tax_rate: float = 0.0020,
    commission_rate: float = 0.00015,
    slippage_rate: float = 0.001,
    borrowing_rate: float = 0.03,
) -> Optional[Dict]:
    """
    Optuna Persistent Bayesian Optimization 실행

    동일 기간+전략+메트릭으로 재실행 시 이전 Trial 위에 누적 탐색.
    실행 횟수가 많을수록 최고값이 단조 증가(≥)함을 보장.

    Returns:
        {
            'params': dict,
            metric: float,
            'total_complete': int,  # 누적 완료 Trial
            'total_pruned': int,
            'existing_before': int, # 이번 실행 전 누적 수
        }
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
        institution_weight=institution_weight,
        force_exit_on_end=True,
        tax_rate=tax_rate,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        borrowing_rate=borrowing_rate,
    )

    optimizer = OptunaOptimizer(
        db_path=db_path,
        start_date=start_date,
        end_date=end_date,
        base_config=base_config,
        study_storage=_OPTUNA_STORAGE,
    )

    return optimizer.optimize(
        n_trials=n_trials,
        metric=metric,
        verbose=False,
        progress_callback=progress_callback,
        reset=reset_study,
    )
