"""
Streamlit 캐시 데이터 로더

DB 연결, Stage 1-3 분석 파이프라인, 백테스트 실행을 캐싱하여 성능 확보.
기존 모듈(normalizer, pattern_classifier 등)을 수정 없이 재사용.

연결 분리:
    PostgreSQL (get_db_connection): 수급/주가/종목 데이터
    SQLite app.db (get_app_db_path): watchlist/backtest_history/score_change_log
"""

import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import text

# 프로젝트 루트 경로 등록
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DEFAULT_CONFIG
from src.database.connection import get_pg_engine, APP_DB_PATH
from src.analyzer.normalizer import SupplyNormalizer
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
from src.analyzer.pattern_classifier import PatternClassifier
from src.analyzer.signal_detector import SignalDetector
from src.analyzer.integrated_report import IntegratedReport

# 백테스트 모듈은 lazy import (비백테스트 페이지 ~450ms 절감)
# from src.backtesting.engine import BacktestConfig, BacktestEngine
# from src.backtesting.metrics import PerformanceMetrics
# from src.backtesting.portfolio import Trade


# ---------------------------------------------------------------------------
# 앱 DB 경로 (SQLite — watchlist/backtest_history/score_change_log)
# ---------------------------------------------------------------------------

def _get_app_db_path() -> str:
    return str(_PROJECT_ROOT / APP_DB_PATH)


# ---------------------------------------------------------------------------
# DB 연결 (싱글턴)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db_connection():
    """Streamlit 캐시: PostgreSQL SQLAlchemy Engine (싱글턴)"""
    return get_pg_engine()


# ---------------------------------------------------------------------------
# 정적 데이터 (종목/섹터/날짜 범위)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_stock_list() -> pd.DataFrame:
    """종목 리스트 (stock_code, stock_name, sector) — 활성 KOSPI/KOSDAQ 종목"""
    engine = get_db_connection()
    df = pd.read_sql(
        text(
            "SELECT s.stock_code, s.stock_name, ss.fics_sector AS sector "
            "FROM stocks s "
            "LEFT JOIN stock_sectors ss ON s.stock_code = ss.stock_code "
            "WHERE s.is_active = true AND s.market IN ('KOSPI', 'KOSDAQ') "
            "ORDER BY s.stock_code"
        ),
        engine,
    )
    return df


@st.cache_data(ttl=3600)
def get_sectors() -> List[str]:
    """고유 섹터 목록 (stock_sectors.fics_sector 기준)"""
    engine = get_db_connection()
    df = pd.read_sql(
        text(
            "SELECT DISTINCT ss.fics_sector "
            "FROM stock_sectors ss "
            "JOIN stocks s ON ss.stock_code = s.stock_code "
            "WHERE s.is_active = true AND ss.fics_sector IS NOT NULL "
            "ORDER BY ss.fics_sector"
        ),
        engine,
    )
    return df['fics_sector'].tolist()


@st.cache_data(ttl=3600)
def get_market_cap_latest() -> pd.DataFrame:
    """최신 날짜 기준 전체 종목 시총 + 시총순위 반환.

    Returns:
        DataFrame(stock_code, market_cap, market_cap_rank, market_cap_str)
    """
    engine = get_db_connection()
    df = pd.read_sql(
        text("""
        SELECT stock_code, market_cap,
               RANK() OVER (ORDER BY market_cap DESC) AS market_cap_rank
        FROM market_cap_daily
        WHERE time = (SELECT MAX(time) FROM market_cap_daily)
          AND market_cap IS NOT NULL
        """),
        engine,
    )
    if df.empty:
        return df

    # 억/조 포맷
    def _fmt(v):
        if v >= 1_000_000_000_000:
            return f"{v / 1_000_000_000_000:.1f}조"
        return f"{v / 100_000_000:,.0f}억"

    df['market_cap_str'] = df['market_cap'].apply(_fmt)
    df['market_cap_rank'] = df['market_cap_rank'].astype(int)
    return df


@st.cache_data(ttl=3600)
def get_date_range() -> Tuple[str, str]:
    """DB 내 거래 날짜 범위 (min_date, max_date) — MV 우선 사용"""
    engine = get_db_connection()
    try:
        df = pd.read_sql(
            text("SELECT MIN(trade_date) AS min_date, MAX(trade_date) AS max_date FROM mv_daily_sff"),
            engine,
        )
    except Exception:
        df = pd.read_sql(
            text("SELECT MIN(time) AS min_date, MAX(time) AS max_date FROM investor_trading"),
            engine,
        )
    row = df.iloc[0]
    return str(row['min_date']), str(row['max_date'])


# ---------------------------------------------------------------------------
# 이상 수급 탐지 (캐싱)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_today_supply_ranking(top_n: int = 50) -> pd.DataFrame:
    """당일 전 종목 외국인/기관 순매수금액 조회 (캐싱)"""
    engine = get_db_connection()
    df = pd.read_sql(
        text("""
        SELECT
            it.stock_code,
            s.stock_name,
            ss.fics_sector AS sector,
            SUM(CASE WHEN it.investor_type = 'FOREIGN'      THEN it.net_buy_value ELSE 0 END) AS foreign_net_amount,
            SUM(CASE WHEN it.investor_type = 'INSTITUTION'  THEN it.net_buy_value ELSE 0 END) AS institution_net_amount
        FROM investor_trading it
        JOIN stocks s ON it.stock_code = s.stock_code
        LEFT JOIN stock_sectors ss ON it.stock_code = ss.stock_code
        WHERE it.investor_type IN ('FOREIGN', 'INSTITUTION')
          AND it.time = (SELECT MAX(time) FROM investor_trading)
        GROUP BY it.stock_code, s.stock_name, ss.fics_sector
        """),
        engine,
    )
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_abnormal_supply_data(
    end_date: Optional[str] = None,
    threshold: float = 2.0,
    top_n: int = 10,
    direction: str = 'both',
    institution_weight: float = 0.3,
    z_score_window: int = 60,
) -> pd.DataFrame:
    """이상 수급 종목 조회 (캐싱) — 순매수금액 포함"""
    engine = get_db_connection()
    normalizer = SupplyNormalizer(engine, config={
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

    # 순매수금액 조인 (해당 날짜)
    trade_date = df['trade_date'].iloc[0]
    codes = df['stock_code'].tolist()
    placeholders = ", ".join([f":c{i}" for i in range(len(codes))])
    params = {f"c{i}": c for i, c in enumerate(codes)}
    params["td"] = trade_date
    amounts = pd.read_sql(
        text(f"""
        SELECT
            it.stock_code,
            SUM(CASE WHEN it.investor_type = 'FOREIGN'      THEN it.net_buy_value ELSE 0 END) AS foreign_net_amount,
            SUM(CASE WHEN it.investor_type = 'INSTITUTION'  THEN it.net_buy_value ELSE 0 END) AS institution_net_amount
        FROM investor_trading it
        WHERE it.investor_type IN ('FOREIGN', 'INSTITUTION')
          AND it.time = :td
          AND it.stock_code IN ({placeholders})
        GROUP BY it.stock_code
        """),
        engine,
        params=params,
    )
    df = df.merge(amounts, on='stock_code', how='left')
    return df


# ---------------------------------------------------------------------------
# 파이프라인 파일 캐시 (pkl)
# ---------------------------------------------------------------------------

_CACHE_DIR = str(_PROJECT_ROOT / "data" / "cache")


def _get_cache_path(end_date: str, weight: float, direction: str = 'long') -> str:
    w_int = int(round(weight * 100))
    suffix = f"_{direction}" if direction != 'long' else ""
    return os.path.join(_CACHE_DIR, f"pipeline_{end_date}_{w_int}{suffix}.pkl")


def _load_pipeline_cache(end_date: str, weight: float, direction: str = 'long'):
    """파일 캐시 로드 (없으면 None)"""
    path = _get_cache_path(end_date, weight, direction)
    if not os.path.exists(path):
        return None
    try:
        cache = pd.read_pickle(path)
        if cache.get('end_date') != end_date:
            return None
        # weight 검증: 캐시 키 반올림으로 인한 충돌 방지
        cached_weight = cache.get('institution_weight')
        if cached_weight is not None and round(cached_weight, 4) != round(weight, 4):
            return None
        return cache
    except Exception:
        return None


def _save_pipeline_cache(end_date, weight, zscore, classified, signals, report, direction='long'):
    """파이프라인 결과를 pkl 파일로 저장"""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _get_cache_path(end_date, weight, direction)
    pd.to_pickle({
        'end_date': end_date,
        'institution_weight': weight,
        'created_at': datetime.now().isoformat(),
        'zscore_matrix': zscore,
        'classified_df': classified,
        'signals_df': signals,
        'report_df': report,
    }, path)


def _cleanup_old_cache(max_age_days: int = 7):
    """오래된 캐시 파일 자동 정리"""
    if not os.path.exists(_CACHE_DIR):
        return
    cutoff = time.time() - max_age_days * 86400
    for f in os.listdir(_CACHE_DIR):
        fpath = os.path.join(_CACHE_DIR, f)
        if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
            try:
                os.remove(fpath)
            except OSError:
                pass


# 모듈 로드 시 1회 정리
_cleanup_old_cache()


# ---------------------------------------------------------------------------
# Stage 1-3 분석 파이프라인 (단계별 캐시 분리)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _stage_zscore(end_date: Optional[str] = None, institution_weight: float = 0.3) -> pd.DataFrame:
    """Stage 1+2: 수급 정규화 + 멀티 기간 Z-Score"""
    engine = get_db_connection()
    normalizer = SupplyNormalizer(engine, config={
        'z_score_window': 60,
        'min_data_points': 30,
        'institution_weight': institution_weight,
    })
    calculator = OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)
    zscore_matrix = calculator.calculate_multi_period_zscores(
        DEFAULT_CONFIG['periods'], end_date=end_date
    )
    return zscore_matrix.reset_index()


@st.cache_data(ttl=3600, show_spinner=False)
def _stage_supply_persistence(end_date: Optional[str] = None,
                               institution_weight: float = 0.3) -> pd.DataFrame:
    """수급 지속 강도 조회 (평균Sff × √연속일수, Long/Short 양방향)"""
    engine = get_db_connection()
    df = pd.read_sql(text('''
        WITH recent AS (
            SELECT stock_code, trade_date,
                CASE WHEN foreign_sff * institution_sff > 0
                     THEN foreign_sff + institution_sff * :weight
                     ELSE foreign_sff
                END AS combined_sff
            FROM mv_daily_sff
            WHERE trade_date <= :end_date
              AND trade_date >= CAST(:end_date AS date) - INTERVAL '365 days'
            ORDER BY stock_code, trade_date
        )
        SELECT stock_code, trade_date, combined_sff
        FROM recent
    '''), engine, params={'end_date': end_date, 'weight': institution_weight})

    if df.empty:
        return pd.DataFrame(columns=['stock_code', 'supply_persistence_long', 'supply_persistence_short'])

    df = df.sort_values(['stock_code', 'trade_date'])
    result = []
    for code, group in df.groupby('stock_code'):
        vals = group['combined_sff'].values
        # Long: 연속 양수 강도
        long_streak = []
        for v in vals:
            if v > 0:
                long_streak.append(v)
            else:
                long_streak = []
        sp_long = (sum(long_streak) / len(long_streak) * len(long_streak) ** 0.5) if long_streak else 0.0

        # Short: 연속 음수 강도
        short_streak = []
        for v in vals:
            if v < 0:
                short_streak.append(abs(v))
            else:
                short_streak = []
        sp_short = (sum(short_streak) / len(short_streak) * len(short_streak) ** 0.5) if short_streak else 0.0

        result.append({
            'stock_code': code,
            'supply_persistence_long': sp_long,
            'supply_persistence_short': sp_short,
        })

    return pd.DataFrame(result)


@st.cache_data(ttl=3600, show_spinner=False)
def _stage_classify(end_date: Optional[str] = None, institution_weight: float = 0.3,
                    direction: str = 'long') -> pd.DataFrame:
    """Stage 3a: 패턴 분류 (supply_persistence 포함)"""
    zscore_matrix = _stage_zscore(end_date=end_date, institution_weight=institution_weight)
    if zscore_matrix.empty:
        return pd.DataFrame()

    # 수급 지속 강도 병합
    sp = _stage_supply_persistence(end_date=end_date, institution_weight=institution_weight)
    if not sp.empty:
        zscore_matrix = zscore_matrix.merge(sp, on='stock_code', how='left')
        zscore_matrix['supply_persistence_long'] = zscore_matrix['supply_persistence_long'].fillna(0.0)
        zscore_matrix['supply_persistence_short'] = zscore_matrix['supply_persistence_short'].fillna(0.0)

    classifier = PatternClassifier(use_tc=True, use_divergence=True, tc_center=0.5, tc_scale=10.0)
    return classifier.classify_all(zscore_matrix, direction=direction)


@st.cache_data(ttl=3600, show_spinner=False)
def _stage_signals(end_date: Optional[str] = None, institution_weight: float = 0.3) -> pd.DataFrame:
    """Stage 3b: 시그널 탐지"""
    engine = get_db_connection()
    detector = SignalDetector(engine, institution_weight=institution_weight)
    return detector.detect_all_signals(end_date=end_date)


@st.cache_data(ttl=3600, show_spinner=False)
def _stage_report(end_date: Optional[str] = None, institution_weight: float = 0.3,
                  direction: str = 'long') -> pd.DataFrame:
    """Stage 3c: 통합 리포트"""
    classified_df = _stage_classify(end_date=end_date, institution_weight=institution_weight,
                                    direction=direction)
    signals_df = _stage_signals(end_date=end_date, institution_weight=institution_weight)
    if classified_df.empty:
        return pd.DataFrame()
    engine = get_db_connection()
    report_gen = IntegratedReport(engine)
    return report_gen.generate_report(classified_df, signals_df)


def run_analysis_pipeline(
    end_date: Optional[str] = None,
    institution_weight: float = 0.3,
    direction: str = 'long',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stage 1-3 전체 파이프라인 (progress bar 없는 버전)"""
    return run_analysis_pipeline_with_progress(
        end_date=end_date, progress_bar=None,
        institution_weight=institution_weight,
        direction=direction,
    )


def run_analysis_pipeline_with_progress(
    end_date: Optional[str] = None,
    progress_bar=None,
    institution_weight: float = 0.3,
    direction: str = 'long',
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stage 1-3 전체 파이프라인 (단계별 진행률 표시 지원)

    1. 파일 캐시 확인 (pkl) → 히트 시 <1초 반환
    2. 캐시 미스 → 기존 파이프라인 실행 + 결과 캐시 저장

    Args:
        end_date: 분석 기준 날짜
        progress_bar: st.progress 위젯 (None이면 진행률 표시 안 함)
        institution_weight: 기관 가중치 (0.0=외국인만, 0.3=기본, 1.0=동등)
        direction: 'long' (순매수) 또는 'short' (순매도)

    Returns:
        (zscore_matrix, classified_df, signals_df, report_df)
    """
    def _upd(pct: float, msg: str):
        if progress_bar is not None:
            progress_bar.progress(pct, text=msg)

    # 1. 파일 캐시 확인
    if end_date is not None:
        cache = _load_pipeline_cache(end_date, institution_weight, direction)
        if cache is not None:
            _upd(1.0, "✅ 캐시 로드 완료 100%")
            return (cache['zscore_matrix'], cache['classified_df'],
                    cache['signals_df'], cache['report_df'])

    # 2. 캐시 미스: 기존 파이프라인 실행
    _upd(0.05, "📐 수급 데이터 정규화 중... 5%")
    zscore_matrix = _stage_zscore(end_date=end_date, institution_weight=institution_weight)

    if zscore_matrix.empty:
        _upd(1.0, "✅ 완료 100%")
        empty = pd.DataFrame()
        return zscore_matrix, empty, empty, empty

    _upd(0.40, "📊 Z-Score 계산 완료 → 패턴 분류 중... 40%")
    classified_df = _stage_classify(end_date=end_date, institution_weight=institution_weight,
                                    direction=direction)

    _upd(0.65, "🔍 패턴 분류 완료 → 시그널 탐지 중... 65%")
    signals_df = _stage_signals(end_date=end_date, institution_weight=institution_weight)

    _upd(0.75, "📡 시그널 탐지 완료 → 리포트 생성 중... 75%")
    report_df = _stage_report(end_date=end_date, institution_weight=institution_weight,
                              direction=direction)

    # ── 필터 1: 거래대금 1억 미만 종목 제외 (거래정지 + 극소거래량 포함) ──
    # 거래정지(volume=0) + 유동성 부족 종목은 sff 노이즈 → 의미 없는 점수 산출
    if not classified_df.empty and end_date is not None:
        try:
            _engine = get_db_connection()
            _illiquid = pd.read_sql(text('''
                SELECT stock_code FROM ohlcv_daily
                WHERE time = :dt AND (trading_value < 100000000 OR trading_value IS NULL)
            '''), _engine, params={'dt': end_date})
            if not _illiquid.empty:
                _illiquid_codes = set(_illiquid['stock_code'])
                classified_df = classified_df[~classified_df['stock_code'].isin(_illiquid_codes)]
                if not report_df.empty:
                    report_df = report_df[~report_df['stock_code'].isin(_illiquid_codes)]
                if not signals_df.empty:
                    signals_df = signals_df[~signals_df['stock_code'].isin(_illiquid_codes)]
        except Exception:
            pass  # DB 오류 시 필터 스킵

    # ── 필터 2: 방향 확신도 — 실제 수급 방향이 반대인 종목 제외 ──
    # direction_confidence가 Z-Score를 0으로 만든 종목은 weighted_sum≈0 → score≈50
    # 임계값 0.05: tanh(x)>0.05 ↔ sff>0.05×std — 노이즈 수준 제외
    if not classified_df.empty:
        _active = classified_df[
            (classified_df['recent'].abs() > 0.05) |
            (classified_df['weighted'].abs() > 0.05)
        ]['stock_code']
        classified_df = classified_df[classified_df['stock_code'].isin(_active)]
        if not report_df.empty:
            report_df = report_df[report_df['stock_code'].isin(_active)]
        signals_df = signals_df[signals_df['stock_code'].isin(_active)] if not signals_df.empty else signals_df

    _upd(0.85, "📋 리포트 생성 완료 85%")

    # 3. 결과 캐시 저장
    if end_date is not None:
        _save_pipeline_cache(end_date, institution_weight,
                             zscore_matrix, classified_df, signals_df, report_df,
                             direction=direction)

    return zscore_matrix, classified_df, signals_df, report_df


# ---------------------------------------------------------------------------
# 종목 상세 데이터
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_zscore_history(
    stock_code: str,
    end_date: Optional[str] = None,
    institution_weight: float = 0.3,
    z_score_window: int = 60,
) -> pd.DataFrame:
    """단일 종목의 Z-Score 전체 시계열 이력 반환

    Returns:
        컬럼: trade_date, stock_code, foreign_zscore, institution_zscore, combined_zscore
    """
    engine = get_db_connection()
    normalizer = SupplyNormalizer(engine, config={
        'z_score_window': z_score_window,
        'min_data_points': min(30, z_score_window // 2),
        'institution_weight': institution_weight,
    })
    return normalizer.calculate_zscore(stock_codes=[stock_code], end_date=end_date)


@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_raw_history(
    stock_code: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """단일 종목의 원시 수급+가격 이력 + 파생 지표

    Returns:
        컬럼: trade_date, close_price, foreign_net_amount, institution_net_amount,
               trading_volume, ma5, ma20, sync_rate
    """
    engine = get_db_connection()
    date_filter = "AND it.time <= :end_date" if end_date else ""
    params = {"stock_code": stock_code}
    if end_date:
        params["end_date"] = end_date
    df = pd.read_sql(
        text(f"""
        SELECT
            it.time AS trade_date,
            MAX(o.close_price) AS close_price,
            SUM(CASE WHEN it.investor_type = 'FOREIGN'      THEN it.net_buy_value ELSE 0 END) AS foreign_net_amount,
            SUM(CASE WHEN it.investor_type = 'INSTITUTION'  THEN it.net_buy_value ELSE 0 END) AS institution_net_amount,
            MAX(o.volume)      AS trading_volume
        FROM investor_trading it
        JOIN ohlcv_daily o ON it.time = o.time AND it.stock_code = o.stock_code
        WHERE it.investor_type IN ('FOREIGN', 'INSTITUTION')
          AND it.stock_code = :stock_code
          {date_filter}
        GROUP BY it.time
        ORDER BY it.time
        """),
        engine,
        params=params,
    )

    if df.empty:
        return df

    # PostgreSQL DATE → 문자열 통일 (datetime.date 비교 오류 방지)
    if 'trade_date' in df.columns:
        df['trade_date'] = df['trade_date'].astype(str)

    # 개인 순매수 (외국인+기관+개인 합계 ≈ 0 원리)
    df['individual_net_amount'] = -(df['foreign_net_amount'] + df['institution_net_amount'])

    # 파생 지표 (외국인 순매수 기준 이동평균)
    df['ma5']  = df['foreign_net_amount'].rolling(5).mean()
    df['ma20'] = df['foreign_net_amount'].rolling(20).mean()

    both_buy = (
        (df['foreign_net_amount'] > 0) & (df['institution_net_amount'] > 0)
    ).astype(int)
    df['sync_rate'] = both_buy.rolling(20).mean() * 100

    return df


# ---------------------------------------------------------------------------
# OHLCV (주가 차트용)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_stock_ohlcv(stock_code: str, end_date: str, start_date: str = None) -> pd.DataFrame:
    """종목의 OHLCV 데이터 조회 (주가 차트용)

    Returns:
        컬럼: date, open, high, low, close, volume
    """
    engine = get_db_connection()
    query = """
        SELECT time AS date, open_price AS open, high_price AS high,
               low_price AS low, close_price AS close, volume
        FROM ohlcv_daily
        WHERE stock_code = :code AND time <= :end_date
    """
    params: dict = {"code": stock_code, "end_date": end_date}
    if start_date:
        query += " AND time >= :start_date"
        params["start_date"] = start_date
    query += " ORDER BY time"
    df = pd.read_sql(text(query), engine, params=params)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    return df


# ---------------------------------------------------------------------------
# 백테스트 실행
# ---------------------------------------------------------------------------

def _serialize_trades(trades) -> List[dict]:
    """Trade 객체 리스트 → dict 리스트 (캐싱 가능 형태)"""
    result = []
    for t in trades:
        d = t.to_dict()
        # to_dict()에 profit (property) 포함될 수 있으므로 제거
        d.pop('profit', None)
        result.append(d)
    return result


def _deserialize_trades(trade_dicts: List[dict]):
    """dict 리스트 → Trade 객체 리스트"""
    from src.backtesting.portfolio import Trade
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
    use_tc: bool = True,
    use_divergence: bool = True,
    market_cap_top_n: Optional[int] = None,
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
    from src.backtesting.engine import BacktestConfig, BacktestEngine
    pg_engine = get_db_connection()

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
        use_tc=use_tc,
        use_divergence=use_divergence,
        tax_rate=tax_rate,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        borrowing_rate=borrowing_rate,
        market_cap_top_n=market_cap_top_n,
    )

    engine = BacktestEngine(pg_engine, config)
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
            'reverse_signal_threshold': config.reverse_signal_threshold,
            'strategy': config.strategy,
            'institution_weight': config.institution_weight,
            'use_tc': config.use_tc,
            'use_divergence': config.use_divergence,
            'tax_rate': config.tax_rate,
            'commission_rate': config.commission_rate,
            'slippage_rate': config.slippage_rate,
            'borrowing_rate': config.borrowing_rate,
            'market_cap_top_n': config.market_cap_top_n,
        },
        'initial_capital': config.initial_capital,
    }


def get_metrics_from_result(result: Dict):
    """캐싱된 백테스트 결과에서 PerformanceMetrics 생성"""
    from src.backtesting.metrics import PerformanceMetrics
    trades = _deserialize_trades(result['trade_dicts'])
    if not trades:
        return None
    return PerformanceMetrics(
        trades=trades,
        daily_values=result['daily_values'],
        initial_capital=result['initial_capital'],
    )


def get_trades_from_result(result: Dict):
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
    use_tc: bool = True,
    use_divergence: bool = True,
    market_cap_top_n: Optional[int] = None,
) -> Dict:
    """백테스트 실행 (캐시 없음, progress_callback 지원)"""
    from src.backtesting.engine import BacktestConfig, BacktestEngine
    pg_engine = get_db_connection()
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
        use_tc=use_tc,
        use_divergence=use_divergence,
        tax_rate=tax_rate,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        borrowing_rate=borrowing_rate,
        market_cap_top_n=market_cap_top_n,
    )
    engine = BacktestEngine(pg_engine, config)
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
            'reverse_signal_threshold': config.reverse_signal_threshold,
            'strategy': config.strategy,
            'institution_weight': config.institution_weight,
            'use_tc': config.use_tc,
            'use_divergence': config.use_divergence,
            'tax_rate': config.tax_rate,
            'commission_rate': config.commission_rate,
            'slippage_rate': config.slippage_rate,
            'borrowing_rate': config.borrowing_rate,
            'market_cap_top_n': config.market_cap_top_n,
        },
        'initial_capital': config.initial_capital,
    }


# ---------------------------------------------------------------------------
# 워크포워드 실행
# ---------------------------------------------------------------------------

def run_walk_forward(
    start_date: str,
    end_date: str,
    train_months: int = 6,
    val_months: int = 1,
    step_months: int = 1,
    n_trials: int = 100,
    metric: str = 'sharpe_ratio',
    strategy: str = 'long',
    initial_capital: float = 10_000_000,
    max_positions: int = 5,
    max_hold_days: int = 999,
    reverse_threshold: float = 60,
    institution_weight: float = 0.3,
    use_tc: bool = True,
    use_divergence: bool = True,
    tax_rate: float = 0.0020,
    commission_rate: float = 0.00015,
    slippage_rate: float = 0.001,
    borrowing_rate: float = 0.03,
    optuna_param_space: Optional[Dict] = None,
    progress_callback=None,
    market_cap_top_n: Optional[int] = None,
) -> Dict:
    """
    Walk-Forward Analysis 실행 래퍼

    Returns:
        {
            'periods': List[dict],          # 기간별 메트릭 + best_params
            'combined_trade_dicts': List[dict],  # 직렬화된 통합 거래
            'combined_daily_values': DataFrame,
            'summary': DataFrame,           # summary() 결과
            'initial_capital': float,
        }
    """
    from src.backtesting.engine import BacktestConfig
    from src.backtesting.walk_forward import WalkForwardAnalyzer, WalkForwardConfig

    wf_config = WalkForwardConfig(
        train_months=train_months,
        val_months=val_months,
        step_months=step_months,
        metric=metric,
        workers=1,
        n_trials=n_trials,
    )

    base_config = BacktestConfig(
        strategy=strategy,
        initial_capital=initial_capital,
        max_positions=max_positions,
        max_hold_days=max_hold_days,
        reverse_signal_threshold=reverse_threshold,
        institution_weight=institution_weight,
        force_exit_on_end=True,
        use_tc=use_tc,
        use_divergence=use_divergence,
        tax_rate=tax_rate,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        borrowing_rate=borrowing_rate,
        market_cap_top_n=market_cap_top_n,
    )

    analyzer = WalkForwardAnalyzer(
        start_date=start_date,
        end_date=end_date,
        wf_config=wf_config,
        base_config=base_config,
        optuna_param_space=optuna_param_space,
    )

    result = analyzer.run(verbose=False, progress_callback=progress_callback)

    # Trade 객체 → dict 직렬화
    combined_trade_dicts = _serialize_trades(result['combined_trades'])

    return {
        'periods': result['periods'],
        'combined_trade_dicts': combined_trade_dicts,
        'combined_daily_values': result['combined_daily_values'],
        'summary': analyzer.summary(),
        'initial_capital': initial_capital,
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
    use_tc: bool = True,
    use_divergence: bool = True,
    market_cap_top_n: Optional[int] = None,
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
    from src.backtesting.engine import BacktestConfig

    base_config = BacktestConfig(
        strategy=strategy,
        initial_capital=initial_capital,
        max_positions=max_positions,
        max_hold_days=max_hold_days,
        reverse_signal_threshold=reverse_threshold,
        institution_weight=institution_weight,
        force_exit_on_end=True,
        use_tc=use_tc,
        use_divergence=use_divergence,
        tax_rate=tax_rate,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        borrowing_rate=borrowing_rate,
        market_cap_top_n=market_cap_top_n,
    )

    optimizer = OptunaOptimizer(
        db_path=None,  # PostgreSQL 사용 (db_path는 무시됨)
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


# ---------------------------------------------------------------------------
# 관심종목 (Watchlist) — SQLite app.db
# ---------------------------------------------------------------------------

def _ensure_watchlist_table() -> None:
    """watchlist 테이블이 없으면 생성 (앱 최초 실행 시 자동 호출)"""
    app_db_path = _get_app_db_path()
    conn = sqlite3.connect(app_db_path)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                stock_code TEXT PRIMARY KEY,
                stock_name TEXT NOT NULL,
                sector     TEXT,
                added_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                note       TEXT DEFAULT ''
            )
        ''')
        conn.commit()
    finally:
        conn.close()


# 모듈 임포트 시 테이블 자동 생성
_ensure_watchlist_table()


def get_watchlist() -> pd.DataFrame:
    """관심종목 목록 반환 (stock_code, stock_name, sector, added_at, note)"""
    app_db_path = _get_app_db_path()
    conn = sqlite3.connect(app_db_path)
    try:
        df = pd.read_sql_query(
            "SELECT stock_code, stock_name, sector, added_at, note FROM watchlist ORDER BY added_at DESC",
            conn,
        )
    finally:
        conn.close()
    return df


def is_in_watchlist(stock_code: str) -> bool:
    """해당 종목이 관심종목에 포함되어 있는지 확인"""
    app_db_path = _get_app_db_path()
    conn = sqlite3.connect(app_db_path)
    try:
        cursor = conn.execute(
            "SELECT 1 FROM watchlist WHERE stock_code = ?", (stock_code,)
        )
        result = cursor.fetchone() is not None
    finally:
        conn.close()
    return result


def add_to_watchlist(stock_code: str, stock_name: str, sector: str = '', note: str = '') -> None:
    """관심종목 추가 (이미 있으면 무시)"""
    app_db_path = _get_app_db_path()
    conn = sqlite3.connect(app_db_path)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (stock_code, stock_name, sector, note) VALUES (?, ?, ?, ?)",
            (stock_code, stock_name, sector or '', note),
        )
        conn.commit()
    finally:
        conn.close()


def remove_from_watchlist(stock_code: str) -> None:
    """관심종목 삭제"""
    app_db_path = _get_app_db_path()
    conn = sqlite3.connect(app_db_path)
    try:
        conn.execute("DELETE FROM watchlist WHERE stock_code = ?", (stock_code,))
        conn.commit()
    finally:
        conn.close()


def update_watchlist_note(stock_code: str, note: str) -> None:
    """관심종목 메모 수정"""
    app_db_path = _get_app_db_path()
    conn = sqlite3.connect(app_db_path)
    try:
        conn.execute(
            "UPDATE watchlist SET note = ? WHERE stock_code = ?", (note, stock_code)
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 백테스트 결과 히스토리 — SQLite app.db
# ---------------------------------------------------------------------------

def _ensure_backtest_history_table() -> None:
    """backtest_history 테이블이 없으면 생성"""
    app_db_path = _get_app_db_path()
    conn = sqlite3.connect(app_db_path)
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS backtest_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                run_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                strategy      TEXT,
                start_date    TEXT,
                end_date      TEXT,
                total_return  REAL,
                mdd           REAL,
                sharpe        REAL,
                calmar        REAL,
                win_rate      REAL,
                total_trades  INTEGER,
                profit_factor REAL,
                min_score     REAL,
                min_signals   INTEGER,
                target_return REAL,
                stop_loss     REAL,
                max_positions INTEGER,
                max_hold_days INTEGER,
                institution_weight REAL,
                note          TEXT DEFAULT '',
                label         TEXT DEFAULT ''
            )
        ''')
        conn.commit()
    finally:
        conn.close()


# 모듈 임포트 시 테이블 자동 생성
_ensure_backtest_history_table()


def save_backtest_history(
    result: dict,
    start_date: str,
    end_date: str,
    note: str = '',
    label: str = '',
) -> int:
    """
    백테스트 결과를 히스토리 DB에 저장하고 id 반환.

    Parameters
    ----------
    result     : dict   run_backtest() / run_backtest_with_progress() 반환값
    start_date : str    백테스트 시작일
    end_date   : str    백테스트 종료일
    note       : str    사용자 메모
    label      : str    결과 식별 레이블
    """
    metrics = get_metrics_from_result(result)
    cfg = result.get('config', {})
    mdd_info = metrics.max_drawdown() if metrics else {}
    total_trades = len(get_trades_from_result(result))

    app_db_path = _get_app_db_path()
    conn = sqlite3.connect(app_db_path)
    try:
        cursor = conn.execute('''
            INSERT INTO backtest_history (
                strategy, start_date, end_date,
                total_return, mdd, sharpe, calmar, win_rate,
                total_trades, profit_factor,
                min_score, min_signals, target_return, stop_loss,
                max_positions, max_hold_days, institution_weight,
                note, label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            cfg.get('strategy', 'long'),
            start_date,
            end_date,
            metrics.total_return()  if metrics else 0.0,
            mdd_info.get('mdd', 0.0),
            metrics.sharpe_ratio()  if metrics else 0.0,
            metrics.calmar_ratio()  if metrics else 0.0,
            metrics.win_rate()      if metrics else 0.0,
            total_trades,
            metrics.profit_factor() if metrics else 0.0,
            cfg.get('min_score', 60),
            cfg.get('min_signals', 1),
            cfg.get('target_return', 0.10),
            cfg.get('stop_loss', -0.05),
            cfg.get('max_positions', 5),
            cfg.get('max_hold_days', 999),
            cfg.get('institution_weight', 0.3),
            note,
            label,
        ))
        row_id = cursor.lastrowid
        conn.commit()
    finally:
        conn.close()
    return row_id


def get_backtest_history(limit: int = 50) -> pd.DataFrame:
    """저장된 백테스트 히스토리 조회 (최신순)"""
    app_db_path = _get_app_db_path()
    conn = sqlite3.connect(app_db_path)
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM backtest_history ORDER BY run_at DESC LIMIT {limit}",
            conn,
        )
    finally:
        conn.close()
    return df


def delete_backtest_history(row_id: int) -> None:
    """백테스트 히스토리 행 삭제"""
    app_db_path = _get_app_db_path()
    conn = sqlite3.connect(app_db_path)
    try:
        conn.execute("DELETE FROM backtest_history WHERE id = ?", (row_id,))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 고득점 변동 알림 (Score Change Log) — SQLite app.db
# ---------------------------------------------------------------------------

_SCORE_LOG_TABLE = "score_change_log"
_SCORE_HIGH_THRESHOLD = 70   # "고득점" 기준


def _ensure_score_change_log_table() -> None:
    """score_change_log 테이블이 없으면 생성"""
    app_db_path = _get_app_db_path()
    conn = sqlite3.connect(app_db_path)
    try:
        conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {_SCORE_LOG_TABLE} (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analysis_date TEXT NOT NULL,
                stock_code    TEXT NOT NULL,
                stock_name    TEXT,
                sector        TEXT,
                pattern       TEXT,
                score         REAL,
                signal_count  INTEGER,
                prev_score    REAL,
                change_type   TEXT  -- 'new_entry', 'score_up', 'score_down', 'exit'
            )
        ''')
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_scl_date ON {_SCORE_LOG_TABLE}(analysis_date DESC)"
        )
        conn.commit()
    finally:
        conn.close()


# 모듈 임포트 시 테이블 자동 생성
_ensure_score_change_log_table()



def snapshot_scores(report_df: pd.DataFrame, analysis_date: str) -> None:
    """
    현재 분석 결과를 score_change_log에 스냅샷 저장.
    직전 스냅샷과 비교하여 신규 진입 / 급등 / 이탈 이벤트를 기록한다.

    Parameters
    ----------
    report_df     : 현재 분석 결과 DataFrame (get_stage_report 결과)
    analysis_date : YYYY-MM-DD 기준일 문자열
    """
    app_db_path = _get_app_db_path()
    conn = sqlite3.connect(app_db_path)
    try:
        # 같은 analysis_date로 이미 기록된 이벤트가 있으면 중복 삽입 방지
        existing = conn.execute(
            f"SELECT COUNT(*) FROM {_SCORE_LOG_TABLE} WHERE analysis_date = ?",
            (analysis_date,),
        ).fetchone()[0]
        if existing > 0:
            return

        # 직전 스냅샷: DB에서 가장 최근 날짜의 고득점 종목
        prev_df = pd.read_sql_query(
            f"""
            SELECT stock_code, score, pattern
            FROM {_SCORE_LOG_TABLE}
            WHERE analysis_date = (
                SELECT MAX(analysis_date)
                FROM {_SCORE_LOG_TABLE}
                WHERE analysis_date < ?
            )
            """,
            conn,
            params=(analysis_date,),
        )
        prev_scores = dict(zip(prev_df['stock_code'], prev_df['score'])) if not prev_df.empty else {}

        # 현재 고득점 종목 (threshold 이상)
        high_df = report_df[report_df['score'] >= _SCORE_HIGH_THRESHOLD].copy()
        curr_codes = set(high_df['stock_code'].tolist())
        prev_codes = set(prev_scores.keys())

        rows = []
        for _, row in high_df.iterrows():
            code = row['stock_code']
            curr_s = float(row.get('score', 0))
            prev_s = prev_scores.get(code)

            if code not in prev_codes:
                change_type = 'new_entry'
            elif curr_s - (prev_s or 0) >= 5:
                change_type = 'score_up'
            elif (prev_s or 0) - curr_s >= 5:
                change_type = 'score_down'
            else:
                change_type = None  # 변동 없음 → 로그 불필요 (중복 방지)

            if change_type:
                rows.append((
                    analysis_date,
                    code,
                    str(row.get('stock_name', '')),
                    str(row.get('sector', '')),
                    str(row.get('pattern', '')),
                    curr_s,
                    int(row.get('signal_count', 0)),
                    prev_s,
                    change_type,
                ))

        # 이탈 종목 (직전 고득점이었으나 지금 없음)
        for code in prev_codes - curr_codes:
            rows.append((
                analysis_date, code, '', '', '', None, 0, prev_scores.get(code), 'exit',
            ))

        if rows:
            conn.executemany(
                f"""INSERT INTO {_SCORE_LOG_TABLE}
                (analysis_date, stock_code, stock_name, sector, pattern,
                 score, signal_count, prev_score, change_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 누적 수급 순위
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def _get_trading_dates(end_date: str, lookback_days: int) -> list:
    """end_date 이전 lookback_days개 거래일 목록 반환"""
    engine = get_db_connection()
    df = pd.read_sql(
        text("""
        SELECT DISTINCT trade_date FROM mv_daily_sff
        WHERE trade_date <= :end_date
        ORDER BY trade_date DESC LIMIT :n
        """),
        engine,
        params={'end_date': end_date, 'n': lookback_days},
    )
    return sorted(df['trade_date'].astype(str).tolist())


# ---------------------------------------------------------------------------
# 점수 보정 (Score Rescaling) — 상위 N개 × 최근 M일 분포 기반
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def get_score_reference_distributions(
    end_date: str,
    lookback_days: int = 120,
    institution_weight: float = 0.3,
    direction: str = 'long',
) -> Dict[str, List[float]]:
    """Precomputer 1회 실행(상위 500종목) → 체급별 정렬된 점수 분포 반환.

    Returns:
        {'all': [...], 'top_100': [...], 'top_200': [...], 'top_300': [...],
         'large': [...], 'mid': [...], 'small': [...]}
        빈 경우 빈 dict.
    """
    from src.backtesting.precomputer import BacktestPrecomputer

    trading_dates = _get_trading_dates(end_date, lookback_days)
    if not trading_dates:
        return {}

    start_date = trading_dates[0]
    pg_engine = get_db_connection()
    precomputer = BacktestPrecomputer(
        pg_engine,
        institution_weight=institution_weight,
        use_tc=True,
        use_divergence=True,
        market_cap_top_n=500,
    )
    result = precomputer.precompute(end_date=end_date, start_date=start_date, verbose=False)
    merged = result.merged_long if direction == 'long' else result.merged_short

    # 점수 + 종목코드 수집 (벡터화)
    all_dfs = []
    for date in result.trading_dates:
        df = merged.get(date)
        if df is not None and not df.empty and 'score' in df.columns:
            sub = df[['stock_code', 'score']].copy()
            sub['final_score'] = sub['score'] + df['signal_count'].fillna(0) * 2  # v2: 시그널 가산 축소
            all_dfs.append(sub[['stock_code', 'final_score']].dropna(subset=['final_score']))

    if not all_dfs:
        return {}

    scores_df = pd.concat(all_dfs, ignore_index=True)

    # 시총 순위 조인
    mcap = get_market_cap_latest()
    if not mcap.empty:
        scores_df = scores_df.merge(
            mcap[['stock_code', 'market_cap_rank']], on='stock_code', how='left',
        )
    else:
        scores_df['market_cap_rank'] = np.nan

    has_rank = scores_df['market_cap_rank'].notna()
    fs = scores_df['final_score']
    rank = scores_df['market_cap_rank']

    dists: Dict[str, List[float]] = {
        'all': sorted(fs.tolist()),
        'top_100': sorted(fs[has_rank & (rank <= 100)].tolist()),
        'top_200': sorted(fs[has_rank & (rank <= 200)].tolist()),
        'top_300': sorted(fs[has_rank & (rank <= 300)].tolist()),
        'large': sorted(fs[has_rank & (rank <= 100)].tolist()),
        'mid': sorted(fs[has_rank & (rank > 100) & (rank <= 300)].tolist()),
        'small': sorted(fs[has_rank & (rank > 300)].tolist()),
    }
    return dists


def rescale_scores(
    df: pd.DataFrame,
    ref_sorted: List[float],
) -> pd.DataFrame:
    """백분위 기반 점수 보정: 역사적 분포에서 해당 종합점수의 백분위 × 100.

    final_score(패턴점수+시그널보너스)를 기준 분포와 비교하여 백분위로 변환.
    - 같은 체급 안에서 0~100으로 자연 분포 (100 초과 없음)
    - 강한 날 1위는 99점, 약한 날 1위는 70점 (교차일 비교 보존)
    """
    df = df.copy()
    n = len(ref_sorted)
    if n == 0:
        return df

    ref_arr = np.array(ref_sorted)

    # final_score(raw) 기준으로 백분위 계산
    raw_final = df['final_score'].values
    percentiles = np.searchsorted(ref_arr, raw_final, side='right') / n * 100
    df['final_score'] = np.clip(percentiles, 0, 100)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def get_cumulative_score_ranking(
    end_date: str,
    lookback_days: int = 20,
    institution_weight: float = 0.3,
    direction: str = 'long',
    top_rank_n: int = 50,
    market_cap_top_n: Optional[int] = 500,
) -> pd.DataFrame:
    """N일간 일별 final_score 집계 (BacktestPrecomputer 사용).

    Returns:
        DataFrame(stock_code, avg_score, max_score, score_top_n_ratio, appearance_days, appearance_ratio)
    """
    from src.backtesting.precomputer import BacktestPrecomputer

    trading_dates = _get_trading_dates(end_date, lookback_days)
    if not trading_dates:
        return pd.DataFrame()

    start_date = trading_dates[0]
    pg_engine = get_db_connection()
    precomputer = BacktestPrecomputer(
        pg_engine,
        institution_weight=institution_weight,
        use_tc=True,
        use_divergence=True,
        market_cap_top_n=market_cap_top_n,
    )
    result = precomputer.precompute(end_date=end_date, start_date=start_date, verbose=False)
    merged = result.merged_long if direction == 'long' else result.merged_short

    # 벡터화: iterrows 대신 pd.concat로 한번에 결합
    frames = []
    for date in result.trading_dates:
        df = merged.get(date)
        if df is not None and not df.empty:
            frames.append(df[['stock_code', 'final_score']].assign(date=date))

    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    all_df['final_score'] = all_df['final_score'].fillna(0)
    all_df['daily_rank'] = all_df.groupby('date')['final_score'].rank(ascending=False)

    total_days = len(result.trading_dates)
    agg = all_df.groupby('stock_code').agg(
        avg_score=('final_score', 'mean'),
        max_score=('final_score', 'max'),
        score_top_n_count=('daily_rank', lambda x: (x <= top_rank_n).sum()),
        appearance_days=('final_score', 'count'),
    ).reset_index()

    # 분모 = appearance_days (등장한 날 중 비율, sff_top_n_ratio와 기준 통일)
    agg['score_top_n_ratio'] = agg['score_top_n_count'] / agg['appearance_days']
    agg['appearance_ratio'] = agg['appearance_days'] / total_days

    return agg


@st.cache_data(ttl=3600, show_spinner=False)
def get_cumulative_sff_ranking(
    end_date: str,
    lookback_days: int = 20,
    institution_weight: float = 0.3,
    direction: str = 'long',
    top_rank_n: int = 50,
) -> pd.DataFrame:
    """N일간 Sff/수급금액 누적 집계 (SQL).

    Returns:
        DataFrame(stock_code, cum_sff, avg_sff, positive_ratio, sff_top_n_ratio,
                  cum_net_amount, trading_days)
    """
    dir_sign = 1 if direction == 'long' else -1
    engine = get_db_connection()
    df = pd.read_sql(
        text("""
        WITH date_range AS (
            SELECT DISTINCT trade_date
            FROM mv_daily_sff
            WHERE trade_date <= :end_date
            ORDER BY trade_date DESC
            LIMIT :lookback
        ),
        daily_sff AS (
            SELECT m.stock_code, m.trade_date,
                CASE WHEN m.foreign_sff * m.institution_sff > 0
                     THEN m.foreign_sff + m.institution_sff * :weight
                     ELSE m.foreign_sff
                END * :dir_sign AS combined_sff,
                (m.foreign_net_amount + m.institution_net_amount) * :dir_sign AS net_amount
            FROM mv_daily_sff m
            WHERE m.trade_date IN (SELECT trade_date FROM date_range)
        ),
        ranked AS (
            SELECT *,
                RANK() OVER (PARTITION BY trade_date ORDER BY combined_sff DESC) AS daily_rank
            FROM daily_sff
        )
        SELECT stock_code,
            SUM(combined_sff) AS cum_sff,
            AVG(combined_sff) AS avg_sff,
            COUNT(*) FILTER (WHERE combined_sff > 0)::float
                / NULLIF(COUNT(*), 0) AS positive_ratio,
            COUNT(*) FILTER (WHERE daily_rank <= :top_rank_n)::float
                / NULLIF(COUNT(*), 0) AS sff_top_n_ratio,
            SUM(net_amount) AS cum_net_amount,
            COUNT(*) AS trading_days
        FROM ranked
        GROUP BY stock_code
        """),
        engine,
        params={
            'end_date': end_date,
            'lookback': lookback_days,
            'weight': institution_weight,
            'dir_sign': dir_sign,
            'top_rank_n': top_rank_n,
        },
    )
    return df


def get_score_change_alerts(limit: int = 100) -> pd.DataFrame:
    """
    최근 고득점 변동 알림 조회.
    Returns: DataFrame (analysis_date, change_type, stock_code, stock_name, score, prev_score, ...)
    """
    app_db_path = _get_app_db_path()
    conn = sqlite3.connect(app_db_path)
    try:
        df = pd.read_sql_query(
            f"""
            SELECT *
            FROM {_SCORE_LOG_TABLE}
            ORDER BY logged_at DESC
            LIMIT {limit}
            """,
            conn,
        )
    finally:
        conn.close()
    return df
