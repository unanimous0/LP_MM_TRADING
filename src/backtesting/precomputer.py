"""
백테스트 사전 계산 모듈

매 거래일 반복하던 Stage 1-3 계산을 벡터화하여 한 번에 수행.
Z-Score(6기간), 시그널(3종), 가격/종목명을 사전 계산하여
백테스트 루프에서 O(1) lookup으로 10배+ 속도 향상.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import sqlite3


PERIODS = {
    '5D': 5,
    '10D': 10,
    '20D': 20,
    '50D': 50,
    '100D': 100,
    '200D': 200,
    '500D': 500,
}


@dataclass
class PrecomputeResult:
    """사전 계산 결과 컨테이너

    Attributes:
        zscore_all_dates: MultiIndex(trade_date, stock_code) → [5D,10D,20D,50D,100D,200D,500D]
        signals_all_dates: MultiIndex(trade_date, stock_code) → [ma_cross,ma_diff,acceleration,sync_rate,signal_count]
        price_lookup: (stock_code, trade_date) → close_price
        stock_names: stock_code → stock_name
        trading_dates: 정렬된 거래일 목록
        patterns_long: trade_date → classify_all(direction='long') DataFrame
        patterns_short: trade_date → classify_all(direction='short') DataFrame
        merged_long: trade_date → patterns+signals+stock_name+final_score 통합 DataFrame (O(1) 조회용)
        merged_short: trade_date → patterns+signals+stock_name+final_score 통합 DataFrame (O(1) 조회용)
    """
    zscore_all_dates: pd.DataFrame
    signals_all_dates: pd.DataFrame
    price_lookup: dict = field(default_factory=dict)
    stock_names: dict = field(default_factory=dict)
    trading_dates: list = field(default_factory=list)
    patterns_long: dict = field(default_factory=dict)
    patterns_short: dict = field(default_factory=dict)
    merged_long: dict = field(default_factory=dict)
    merged_short: dict = field(default_factory=dict)


class BacktestPrecomputer:
    """백테스트 사전 계산기

    DB에서 원본 데이터를 1회 로드한 뒤,
    벡터화 연산으로 전체 날짜의 Z-Score/시그널을 한 번에 계산.
    """

    def __init__(self, conn: sqlite3.Connection, institution_weight: float = 0.3):
        """
        Args:
            conn: 데이터베이스 연결
            institution_weight: 기관 가중치 (Sff 계산용, 기본 0.3)
        """
        self.conn = conn
        self.institution_weight = institution_weight

    def precompute(self, end_date: str, start_date: Optional[str] = None,
                   verbose: bool = True) -> PrecomputeResult:
        """
        전체 기간 Z-Score/시그널 사전 계산

        Args:
            end_date: 데이터 로드 종료일 (YYYY-MM-DD)
            start_date: 거래일 필터 시작일 (None이면 전체)
            verbose: 진행 상황 출력
        """
        if verbose:
            print("사전 계산 시작...")

        raw_df = self._load_raw_data(end_date)
        if verbose:
            print(f"  원본 데이터: {len(raw_df):,}건")

        sff_df = self._compute_sff_all_dates(raw_df)
        if verbose:
            print("  Sff 계산 완료")

        zscore_df = self._compute_multi_period_zscores_all_dates(sff_df)
        if verbose:
            print(f"  Z-Score 계산 완료: {len(zscore_df):,}건")

        signals_df = self._compute_signals_all_dates(raw_df)
        if verbose:
            print(f"  시그널 계산 완료: {len(signals_df):,}건")

        price_lookup = self._build_price_lookup(raw_df)
        stock_names = self._build_stock_names()

        trading_dates = sorted(raw_df['trade_date'].unique())
        if start_date:
            trading_dates = [d for d in trading_dates if d >= start_date]

        patterns_long, patterns_short = self._compute_patterns_all_dates(
            zscore_df, trading_dates
        )
        if verbose:
            print(f"  패턴 사전 계산 완료: {len(patterns_long)}일(long), {len(patterns_short)}일(short)")

        merged_long, merged_short = self._merge_patterns_signals(
            patterns_long, patterns_short, signals_df, stock_names
        )
        if verbose:
            print(f"  패턴+시그널 통합 완료: {len(merged_long)}일(long), {len(merged_short)}일(short)")

        if verbose:
            print(f"  거래일: {len(trading_dates)}일, "
                  f"종목: {raw_df['stock_code'].nunique()}종목")
            print("사전 계산 완료!")

        return PrecomputeResult(
            zscore_all_dates=zscore_df,
            signals_all_dates=signals_df,
            price_lookup=price_lookup,
            stock_names=stock_names,
            trading_dates=trading_dates,
            patterns_long=patterns_long,
            patterns_short=patterns_short,
            merged_long=merged_long,
            merged_short=merged_short,
        )

    def _load_raw_data(self, end_date: str) -> pd.DataFrame:
        """DB에서 원본 데이터 1회 로드"""
        query = """
        SELECT trade_date, stock_code,
               foreign_net_amount, institution_net_amount,
               close_price, free_float_shares
        FROM investor_flows
        WHERE close_price IS NOT NULL
          AND free_float_shares IS NOT NULL
          AND trade_date <= ?
        ORDER BY stock_code, trade_date
        """
        return pd.read_sql(query, self.conn, params=[end_date])

    def _compute_sff_all_dates(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """전체 날짜 Sff 벡터화 계산 (normalizer._apply_sff_formula 동일 로직)"""
        df = raw_df.copy()
        free_float_mcap = df['close_price'] * df['free_float_shares']

        df['foreign_sff'] = (df['foreign_net_amount'] / free_float_mcap) * 100
        df['institution_sff'] = (df['institution_net_amount'] / free_float_mcap) * 100

        same_direction = (df['foreign_sff'] * df['institution_sff']) > 0
        df['combined_sff'] = np.where(
            same_direction,
            df['foreign_sff'] + df['institution_sff'] * self.institution_weight,
            df['foreign_sff']
        )

        return df.replace([np.inf, -np.inf], np.nan)

    def _compute_multi_period_zscores_all_dates(self, sff_df: pd.DataFrame) -> pd.DataFrame:
        """7기간 Z-Score 벡터화 계산 (모든 날짜)

        performance_optimizer._calculate_zscore_vectorized와 동일 로직을
        전체 날짜에 대해 한 번에 수행.
        """
        df = sff_df[['trade_date', 'stock_code', 'combined_sff']].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        std_cols = []
        for period_name, lookback_days in PERIODS.items():
            min_periods = max(1, lookback_days // 2)

            rolling_mean = df.groupby('stock_code')['combined_sff'].transform(
                lambda x: x.rolling(window=lookback_days, min_periods=min_periods).mean()
            )
            rolling_std = df.groupby('stock_code')['combined_sff'].transform(
                lambda x: x.rolling(window=lookback_days, min_periods=min_periods).std()
            )

            # 조건부 Z-Score (부호 전환 시 과잉 반응 방지)
            same_sign = (df['combined_sff'] * rolling_mean) > 0
            df[period_name] = np.where(
                same_sign,
                (df['combined_sff'] - rolling_mean) / rolling_std,
                df['combined_sff'] / rolling_std
            )

            # 방향 확신도 메타데이터: 각 기간별 rolling_std 저장
            std_col = f'_std_{period_name}'
            df[std_col] = rolling_std
            std_cols.append(std_col)

        period_cols = list(PERIODS.keys())
        df[period_cols] = df[period_cols].replace([np.inf, -np.inf], np.nan)

        # _today_sff + _std_*D 메타데이터도 함께 저장 (방향 확신도용)
        meta_cols = ['_today_sff'] + std_cols
        df['_today_sff'] = df['combined_sff']

        result = df[['trade_date', 'stock_code'] + period_cols + meta_cols].copy()
        return result.set_index(['trade_date', 'stock_code'])

    def _compute_signals_all_dates(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """MA크로스/가속도/동조율 벡터화 계산 (signal_detector 동일 로직)"""
        df = raw_df[['trade_date', 'stock_code', 'foreign_net_amount',
                     'institution_net_amount']].copy()
        df = df.sort_values(['stock_code', 'trade_date'])

        # 외국인 중심 조건부 합산 (normalizer/signal_detector와 동일 로직)
        same_dir = (df['foreign_net_amount'] * df['institution_net_amount']) > 0
        df['combined_net'] = np.where(
            same_dir,
            df['foreign_net_amount'] + df['institution_net_amount'] * self.institution_weight,
            df['foreign_net_amount']
        )

        # 1. MA Cross (signal_detector.detect_ma_crossover 동일)
        df['ma5'] = df.groupby('stock_code')['foreign_net_amount'].transform(
            lambda x: x.rolling(5).mean())
        df['ma20'] = df.groupby('stock_code')['foreign_net_amount'].transform(
            lambda x: x.rolling(20).mean())
        df['prev_ma5'] = df.groupby('stock_code')['ma5'].shift(1)
        df['prev_ma20'] = df.groupby('stock_code')['ma20'].shift(1)
        df['ma_cross'] = (
            (df['ma5'] > df['ma20']) & (df['prev_ma5'] <= df['prev_ma20'])
        )
        df['ma_diff'] = df['ma5'] - df['ma20']

        # 2. Acceleration (signal_detector.calculate_acceleration 동일)
        df['recent_avg'] = df.groupby('stock_code')['combined_net'].transform(
            lambda x: x.rolling(5).mean())
        df['prev_avg'] = df.groupby('stock_code')['combined_net'].transform(
            lambda x: x.shift(5).rolling(5).mean())
        df['acceleration'] = np.where(
            df['prev_avg'].abs() < 1e-6, np.nan,
            df['recent_avg'] / df['prev_avg']
        )
        df['acceleration'] = df['acceleration'].replace([np.inf, -np.inf], np.nan)

        # 3. Sync rate (signal_detector.calculate_sync_rate 동일)
        df['is_sync'] = (
            (df['foreign_net_amount'] > 0) & (df['institution_net_amount'] > 0)
        ).astype(float)
        df['sync_rate'] = df.groupby('stock_code')['is_sync'].transform(
            lambda x: x.rolling(20, min_periods=20).mean()
        ) * 100

        # 4. Signal count (detect_all_signals 동일 임계값)
        df['signal_count'] = (
            df['ma_cross'].fillna(False).astype(int) +
            (df['acceleration'].fillna(0) > 1.5).astype(int) +
            (df['sync_rate'].fillna(0) > 70).astype(int)
        )

        result = df[['trade_date', 'stock_code', 'ma_cross', 'ma_diff',
                     'acceleration', 'sync_rate', 'signal_count']].copy()
        return result.set_index(['trade_date', 'stock_code'])

    def _build_price_lookup(self, raw_df: pd.DataFrame) -> dict:
        """(stock_code, trade_date) → close_price dict 생성"""
        valid = raw_df.dropna(subset=['close_price'])
        return dict(zip(
            zip(valid['stock_code'], valid['trade_date']),
            valid['close_price'].astype(float)
        ))

    def _build_stock_names(self) -> dict:
        """stock_code → stock_name dict 생성"""
        df = pd.read_sql("SELECT stock_code, stock_name FROM stocks", self.conn)
        return dict(zip(df['stock_code'], df['stock_name']))

    def _compute_patterns_all_dates(self, zscore_df: pd.DataFrame,
                                     trading_dates: list) -> tuple:
        """모든 거래일의 패턴 분류 사전 계산

        classify_all()을 Precomputer 단계에서 1회만 실행.
        Trial 루프에서 매 거래일 classify_all() 호출 → dict O(1) 조회로 교체.

        Args:
            zscore_df: MultiIndex(trade_date, stock_code) Z-Score DataFrame
            trading_dates: 사전 계산할 거래일 목록

        Returns:
            (patterns_long, patterns_short): 각각 {trade_date: DataFrame} dict
        """
        from src.analyzer.pattern_classifier import PatternClassifier
        classifier = PatternClassifier()

        patterns_long = {}
        patterns_short = {}

        for date in trading_dates:
            try:
                zscore_on_date = zscore_df.loc[date].reset_index()
            except KeyError:
                continue

            # _today_sff, _std_*D 메타컬럼이 포함됨 → classify_all()에서 방향 확신도 적용
            long_stocks = zscore_on_date[zscore_on_date['5D'] > 0].copy()
            if not long_stocks.empty:
                patterns_long[date] = classifier.classify_all(long_stocks, direction='long')

            short_stocks = zscore_on_date[zscore_on_date['5D'] < 0].copy()
            if not short_stocks.empty:
                patterns_short[date] = classifier.classify_all(short_stocks, direction='short')

        return patterns_long, patterns_short

    def _merge_patterns_signals(self, patterns_long: dict, patterns_short: dict,
                                 signals_df: pd.DataFrame,
                                 stock_names: dict) -> tuple:
        """패턴 + 시그널 + 종목명 + final_score 사전 통합

        _scan_signals_on_date_fast()에서 매번 수행하던 merge/map/insert를
        Precomputer 단계에서 1회만 실행.
        Trial 루프에서는 dict O(1) 조회 + copy만 수행.

        Returns:
            (merged_long, merged_short): 각각 {trade_date: DataFrame} dict
            컬럼: stock_code, stock_name, pattern, score, direction,
                  5D~500D, ma_cross, ma_diff, acceleration, sync_rate, signal_count, final_score
        """
        merged_long = {}
        merged_short = {}

        all_dates = set(patterns_long.keys()) | set(patterns_short.keys())

        for date in all_dates:
            # 해당 날짜 시그널 (MultiIndex loc)
            try:
                signals_on_date = signals_df.loc[date].reset_index()
            except KeyError:
                signals_on_date = pd.DataFrame()

            for direction, patterns_dict, merged_dict in [
                ('long', patterns_long, merged_long),
                ('short', patterns_short, merged_short),
            ]:
                pat = patterns_dict.get(date)
                if pat is None or pat.empty:
                    continue

                # merge
                if not signals_on_date.empty:
                    merged = pd.merge(pat, signals_on_date, on='stock_code', how='left')
                else:
                    merged = pat.copy()
                    for col in ['ma_cross', 'ma_diff', 'acceleration', 'sync_rate']:
                        merged[col] = np.nan
                    merged['signal_count'] = 0

                # fill defaults
                merged['signal_count'] = merged['signal_count'].fillna(0).astype(int)
                merged['ma_cross'] = merged['ma_cross'].fillna(False)

                # stock_name 추가
                merged.insert(1, 'stock_name',
                              merged['stock_code'].map(lambda c: stock_names.get(c, c)))

                # final_score 사전 계산 (score + signal_count × 5)
                merged['final_score'] = merged['score'] + merged['signal_count'] * 5

                merged_dict[date] = merged

        return merged_long, merged_short
