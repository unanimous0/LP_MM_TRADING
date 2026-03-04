"""
Signal detection module

Implements Stage 3 additional signals:
- MA 골든크로스: 외국인 5일MA > 20일MA 탐지
- 수급 가속도: 최근 5일 vs 직전 5일 비교 (배율)
- 외인-기관 동조율: 함께 매수한 비율 계산

용도:
    패턴 분류 결과에 추가 시그널을 제공하여
    진입/청산 타이밍과 확신도를 높입니다.
    -> Find "When"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from sqlalchemy import text
from src.utils import validate_stock_codes
from src.database.connection import is_mv_available


class SignalDetector:
    """수급 시그널 탐지 클래스"""

    def __init__(self, conn, config: Optional[dict] = None,
                 institution_weight: float = 0.3):
        """
        Args:
            conn: 데이터베이스 연결
            config: 시그널 탐지 설정
                - ma_short: 단기 이동평균 (기본: 5일)
                - ma_long: 장기 이동평균 (기본: 20일)
                - acceleration_window: 가속도 계산 윈도우 (기본: 5일)
                - sync_threshold: 동조 판단 임계값 (기본: 0)
            institution_weight: 기관 가중치 (Sff와 동일 값 사용, 기본: 0.3)
                - normalizer/precomputer와 일관성 유지를 위해 BacktestEngine에서 전달받음
        """
        self.conn = conn
        self.config = config or self._get_default_config()
        self.institution_weight = institution_weight

    @staticmethod
    def _get_default_config() -> dict:
        """기본 설정값 반환"""
        return {
            # 이동평균 설정
            'ma_short': 5,     # 단기 MA (5일)
            'ma_long': 20,     # 장기 MA (20일)

            # 가속도 설정
            'acceleration_window': 5,  # 최근 5일 vs 직전 5일

            # 동조율 설정
            'sync_threshold': 0,  # 0원 이상 매수를 동조로 판단
            'sync_window': 20,    # 최근 20일간 동조율 계산
        }

    def _load_supply_data(self, stock_codes: Optional[List[str]] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        수급 데이터 로드 (내부 헬퍼 메서드)

        MV 존재 시 mv_daily_sff에서 로드 (더 빠름).
        날짜 필터 자동 적용: MA 크로스오버 최대 120일 → 시작일 = end_date - 300 달력일

        Args:
            stock_codes: 종목 코드 리스트 (None이면 전체)
            end_date: 종료일 (YYYY-MM-DD, None이면 최신까지)

        Returns:
            pd.DataFrame: (trade_date, stock_code, foreign_net_amount, institution_net_amount)
        """
        # 빈 리스트 처리 (명시적으로 빈 결과 요청)
        if stock_codes is not None and len(stock_codes) == 0:
            return pd.DataFrame(columns=['trade_date', 'stock_code', 'foreign_net_amount', 'institution_net_amount'])

        # 보안: 입력 검증
        if stock_codes:
            stock_codes = validate_stock_codes(stock_codes)

        # 날짜 필터 자동 계산 (시그널 계산에 필요한 최소 기간)
        # MA 크로스오버: 최대 ma_long=20 영업일 + warmup → 120영업일 기준
        start_date = None
        if end_date:
            ma_long = self.config.get('ma_long', 20)
            max_period = max(ma_long * 6, 120)  # 120 영업일 이상 확보
            calendar_days = int(max_period * 1.47)  # 영업일→달력일 변환
            ref = datetime.strptime(end_date, '%Y-%m-%d')
            start_date = (ref - timedelta(days=calendar_days)).strftime('%Y-%m-%d')

        if is_mv_available():
            # MV 경로: foreign_net_amount, institution_net_amount 컬럼 존재
            clauses = []
            if stock_codes:
                codes_str = "','".join(stock_codes)
                clauses.append(f"stock_code IN ('{codes_str}')")
            if start_date:
                clauses.append(f"trade_date >= '{start_date}'")
            if end_date:
                clauses.append(f"trade_date <= '{end_date}'")
            where = "WHERE " + " AND ".join(clauses) if clauses else ""

            query = f"""
            SELECT trade_date, stock_code, foreign_net_amount, institution_net_amount
            FROM mv_daily_sff
            {where}
            ORDER BY stock_code, trade_date
            """
        else:
            # Fallback: investor_trading 직접 쿼리
            extra_clauses = []
            if stock_codes:
                codes_str = "','".join(stock_codes)
                extra_clauses.append(f"AND it.stock_code IN ('{codes_str}')")
            if start_date:
                extra_clauses.append(f"AND it.time >= '{start_date}'")
            if end_date:
                extra_clauses.append(f"AND it.time <= '{end_date}'")
            extra = "\n          ".join(extra_clauses)

            query = f"""
            SELECT
                it.time AS trade_date,
                it.stock_code,
                SUM(CASE WHEN it.investor_type = 'FOREIGN'      THEN it.net_buy_value ELSE 0 END) AS foreign_net_amount,
                SUM(CASE WHEN it.investor_type = 'INSTITUTION'  THEN it.net_buy_value ELSE 0 END) AS institution_net_amount
            FROM investor_trading it
            WHERE it.investor_type IN ('FOREIGN', 'INSTITUTION')
              {extra}
            GROUP BY it.time, it.stock_code
            ORDER BY it.stock_code, it.time
            """

        df = pd.read_sql(text(query), self.conn)

        # PostgreSQL은 DATE를 datetime.date로 반환 → 문자열 통일
        if not df.empty and 'trade_date' in df.columns:
            df['trade_date'] = df['trade_date'].astype(str)

        if df.empty:
            print("[WARN] No supply data found")
            return pd.DataFrame()

        return df

    def _detect_ma_crossover_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """이미 로드된 데이터에서 MA 골든크로스 탐지 (벡터화)"""
        if df.empty:
            return pd.DataFrame(columns=[
                'stock_code', 'trade_date', 'ma_short', 'ma_long', 'ma_diff', 'is_golden_cross'
            ])

        ma_short = self.config['ma_short']
        ma_long = self.config['ma_long']

        d = df.sort_values(['stock_code', 'trade_date']).copy()
        d['ma_short'] = d.groupby('stock_code')['foreign_net_amount'].transform(
            lambda x: x.rolling(ma_short).mean())
        d['ma_long'] = d.groupby('stock_code')['foreign_net_amount'].transform(
            lambda x: x.rolling(ma_long).mean())
        d['prev_ma_short'] = d.groupby('stock_code')['ma_short'].shift(1)
        d['prev_ma_long'] = d.groupby('stock_code')['ma_long'].shift(1)

        # 최신 날짜만 추출
        latest = d.groupby('stock_code').tail(1).copy()

        # 골든크로스 조건
        cross = (
            (latest['ma_short'] > latest['ma_long']) &
            (latest['prev_ma_short'] <= latest['prev_ma_long'])
        )
        result = latest[cross][['stock_code', 'trade_date', 'ma_short', 'ma_long']].copy()

        if result.empty:
            return pd.DataFrame(columns=['stock_code', 'trade_date', 'ma_short',
                                        'ma_long', 'ma_diff', 'is_golden_cross'])

        result['ma_diff'] = result['ma_short'] - result['ma_long']
        result['is_golden_cross'] = True
        return result.reset_index(drop=True)

    def _calculate_acceleration_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """이미 로드된 데이터에서 수급 가속도 계산 (벡터화)"""
        if df.empty:
            return pd.DataFrame(columns=[
                'stock_code', 'trade_date', 'recent_avg', 'prev_avg', 'acceleration'
            ])

        window = self.config['acceleration_window']

        d = df.sort_values(['stock_code', 'trade_date']).copy()
        # combined_net: normalizer의 combined_sff와 동일 공식 (순매수금액 기반, Sff가 아닌 원금액)
        same_direction = (d['foreign_net_amount'] * d['institution_net_amount']) > 0
        d['combined_net'] = np.where(
            same_direction,
            d['foreign_net_amount'] + d['institution_net_amount'] * self.institution_weight,
            d['foreign_net_amount']
        )

        d['recent_avg'] = d.groupby('stock_code')['combined_net'].transform(
            lambda x: x.rolling(window).mean())
        d['prev_avg'] = d.groupby('stock_code')['combined_net'].transform(
            lambda x: x.shift(window).rolling(window).mean())

        # 최신 날짜만
        latest = d.groupby('stock_code').tail(1).copy()
        latest = latest.dropna(subset=['recent_avg', 'prev_avg'])

        latest['acceleration'] = np.where(
            latest['prev_avg'].abs() < 1e-6, np.nan,
            latest['recent_avg'] / latest['prev_avg']
        )
        latest['acceleration'] = latest['acceleration'].replace([np.inf, -np.inf], np.nan)

        result = latest[['stock_code', 'trade_date', 'recent_avg', 'prev_avg', 'acceleration']].copy()
        return result.reset_index(drop=True)

    def _calculate_sync_rate_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """이미 로드된 데이터에서 동조율 계산 (벡터화)"""
        if df.empty:
            return pd.DataFrame(columns=[
                'stock_code', 'trade_date', 'sync_days', 'total_days', 'sync_rate'
            ])

        window = self.config['sync_window']
        threshold = self.config['sync_threshold']

        d = df.sort_values(['stock_code', 'trade_date']).copy()
        d['is_sync'] = (
            (d['foreign_net_amount'] > threshold) &
            (d['institution_net_amount'] > threshold)
        ).astype(float)
        d['sync_rate'] = d.groupby('stock_code')['is_sync'].transform(
            lambda x: x.rolling(window, min_periods=window).mean()
        ) * 100

        # 최신 날짜만
        latest = d.groupby('stock_code').tail(1).copy()
        latest = latest.dropna(subset=['sync_rate'])
        latest['sync_days'] = (latest['sync_rate'] / 100 * window).round().astype(int)
        latest['total_days'] = window

        result = latest[['stock_code', 'trade_date', 'sync_days', 'total_days', 'sync_rate']].copy()
        return result.reset_index(drop=True)

    def _detect_all_signals_sql(self, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        SQL 함수 fn_signals_latest()로 전체 시그널을 한 번에 계산.

        Returns:
            기존 Python 경로와 동일 포맷의 DataFrame 또는 None (SQL 함수 없을 때)
        """
        try:
            weight = self.institution_weight
            if end_date:
                query = text("SELECT * FROM fn_signals_latest(:w, :d)")
                params = {'w': weight, 'd': end_date}
            else:
                query = text("SELECT * FROM fn_signals_latest(:w)")
                params = {'w': weight}

            df = pd.read_sql(query, self.conn, params=params)

            if df.empty:
                return None

            # ma_cross: NULL → False
            df['ma_cross'] = df['ma_cross'].fillna(False).astype(bool)
            # acceleration: inf → NaN
            df['acceleration'] = df['acceleration'].replace([np.inf, -np.inf], np.nan)

            # 시그널 카운트 및 리스트 생성
            def count_signals(row):
                signals = []
                count = 0
                if row['ma_cross']:
                    signals.append('MA크로스')
                    count += 1
                if pd.notna(row['acceleration']) and row['acceleration'] > 1.5:
                    signals.append(f"가속도 {row['acceleration']:.1f}배")
                    count += 1
                if pd.notna(row['sync_rate']) and row['sync_rate'] > 70:
                    signals.append(f"동조율 {row['sync_rate']:.0f}%")
                    count += 1
                return pd.Series({'signal_count': count, 'signal_list': signals})

            df[['signal_count', 'signal_list']] = df.apply(count_signals, axis=1)

            output_cols = ['stock_code', 'ma_cross', 'ma_diff', 'acceleration',
                          'sync_rate', 'signal_count', 'signal_list']
            print(f"[OK] SQL signals: {len(df)} stocks")
            return df[output_cols]

        except Exception as e:
            print(f"[WARN] SQL signals failed, falling back to Python: {e}")
            return None

    def detect_ma_crossover(self, stock_codes: Optional[List[str]] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """MA 골든크로스 탐지 (하위 호환 API)"""
        df = self._load_supply_data(stock_codes, end_date=end_date)
        return self._detect_ma_crossover_from_df(df)

    def calculate_acceleration(self, stock_codes: Optional[List[str]] = None,
                              end_date: Optional[str] = None) -> pd.DataFrame:
        """수급 가속도 계산 (하위 호환 API)"""
        df = self._load_supply_data(stock_codes, end_date=end_date)
        return self._calculate_acceleration_from_df(df)

    def calculate_sync_rate(self, stock_codes: Optional[List[str]] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """동조율 계산 (하위 호환 API)"""
        df = self._load_supply_data(stock_codes, end_date=end_date)
        return self._calculate_sync_rate_from_df(df)

    def detect_all_signals(self, stock_codes: Optional[List[str]] = None,
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        모든 시그널 통합 탐지 (메인 메서드)

        데이터를 1회만 로드하고 3개 시그널을 각각 계산.

        Args:
            stock_codes: 종목 코드 리스트 (None이면 전체)
            end_date: 종료일 (YYYY-MM-DD, None이면 최신까지)

        Returns:
            pd.DataFrame: 통합 시그널 결과
        """
        # SQL 고속 경로 (MV + fn_signals_latest 존재 시)
        if is_mv_available() and stock_codes is None:
            result = self._detect_all_signals_sql(end_date)
            if result is not None:
                return result

        # Fallback: 기존 Python 경로
        supply_data = self._load_supply_data(stock_codes, end_date=end_date)

        # 각 시그널 계산 (이미 로드된 데이터 사용)
        df_ma = self._detect_ma_crossover_from_df(supply_data)
        df_ma = df_ma[['stock_code', 'ma_diff', 'is_golden_cross']].rename(
            columns={'is_golden_cross': 'ma_cross'}
        )

        df_accel = self._calculate_acceleration_from_df(supply_data)
        df_accel = df_accel[['stock_code', 'acceleration']]

        df_sync = self._calculate_sync_rate_from_df(supply_data)
        df_sync = df_sync[['stock_code', 'sync_rate']]

        # 4. 통합 (outer join으로 전체 종목 유지)
        df_result = df_ma if not df_ma.empty else pd.DataFrame(columns=['stock_code', 'ma_diff', 'ma_cross'])

        if not df_accel.empty:
            df_result = df_result.merge(df_accel, on='stock_code', how='outer')

        if not df_sync.empty:
            df_result = df_result.merge(df_sync, on='stock_code', how='outer')

        # NaN 처리 (where로 FutureWarning 방지 — fillna의 silent downcasting 우회)
        if 'ma_cross' in df_result.columns:
            df_result['ma_cross'] = df_result['ma_cross'].where(
                df_result['ma_cross'].notna(), False).astype(bool)
        else:
            df_result['ma_cross'] = False

        if 'ma_diff' not in df_result.columns:
            df_result['ma_diff'] = np.nan

        if 'acceleration' not in df_result.columns:
            df_result['acceleration'] = np.nan

        if 'sync_rate' not in df_result.columns:
            df_result['sync_rate'] = np.nan

        # 빈 DataFrame 처리
        if df_result.empty:
            df_result['signal_count'] = []
            df_result['signal_list'] = []
            return df_result[['stock_code', 'ma_cross', 'ma_diff', 'acceleration',
                            'sync_rate', 'signal_count', 'signal_list']]

        # 5. 시그널 카운트 및 리스트 생성
        def count_signals(row):
            signals = []
            count = 0

            if row['ma_cross']:
                signals.append('MA크로스')
                count += 1

            if pd.notna(row['acceleration']) and row['acceleration'] > 1.5:
                signals.append(f"가속도 {row['acceleration']:.1f}배")
                count += 1

            if pd.notna(row['sync_rate']) and row['sync_rate'] > 70:
                signals.append(f"동조율 {row['sync_rate']:.0f}%")
                count += 1

            return pd.Series({
                'signal_count': count,
                'signal_list': signals
            })

        df_result[['signal_count', 'signal_list']] = df_result.apply(count_signals, axis=1)

        # 6. 컬럼 순서 정리
        output_cols = ['stock_code', 'ma_cross', 'ma_diff', 'acceleration',
                      'sync_rate', 'signal_count', 'signal_list']

        return df_result[output_cols]

    def get_strong_signals(self,
                          stock_codes: Optional[List[str]] = None,
                          min_signal_count: int = 2) -> pd.DataFrame:
        """
        강력한 시그널 종목 필터링

        Args:
            stock_codes: 종목 코드 리스트 (None이면 전체)
            min_signal_count: 최소 시그널 개수 (기본: 2개 이상)

        Returns:
            pd.DataFrame: 강력한 시그널 종목 (signal_count 내림차순 정렬)
        """
        df_signals = self.detect_all_signals(stock_codes)

        # 필터링
        df_filtered = df_signals[df_signals['signal_count'] >= min_signal_count].copy()

        # 정렬 (signal_count 내림차순)
        df_filtered = df_filtered.sort_values('signal_count', ascending=False)

        return df_filtered
