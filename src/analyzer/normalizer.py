"""
Supply-demand data normalization module

Implements Stage 1 calculations:
- Sff (Supply Force vs Free Float): 유통시가총액 대비 순매수 강도
- Z-Score: 이상 수급 탐지 (60일 평균 대비 표준편차)

공식:
    Sff = (순매수금액 / 유통시가총액) × 100
    Z = (X - μ) / σ

용도:
    시총이 크지만 유통물량이 적은 종목의 수급 왜곡을 정규화하고
    통계적으로 유의미한 이상 수급 이벤트를 탐지합니다.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy import text

# 보안: SQL 인젝션 방지를 위한 입력 검증
from src.utils import validate_stock_codes, validate_date_format
from src.database.connection import is_mv_available


class SupplyNormalizer:
    """수급 데이터 정규화 클래스"""

    def __init__(self, conn, config: Optional[dict] = None):
        """
        Args:
            conn: 데이터베이스 연결
            config: 파라미터 설정
                - z_score_window: Z-Score 계산 윈도우 (기본: 60일)
                - min_data_points: 최소 데이터 포인트 (기본: 30일)
        """
        self.conn = conn
        self.config = config or {
            'z_score_window': 60,     # 60 거래일
            'min_data_points': 30,    # 최소 30일 필요
            'institution_weight': 0.3,  # 기관 가중치 (0.0=외국인만, 0.3=기본, 0.5=기관 강조)
        }
        self._preload_raw = None  # 프리로드 원본 데이터 (None이면 비활성)

    # ------------------------------------------------------------------
    # 공통 데이터 로드 (MV / 원본 테이블 자동 분기)
    # ------------------------------------------------------------------

    def _query_raw_data(self, stock_codes=None, start_date=None, end_date=None):
        """MV 또는 원본 테이블에서 데이터 로드 (단일 쿼리 포인트)

        MV(mv_daily_sff)가 존재하면 사전 계산된 Sff 포함 데이터를 단순 SELECT.
        MV가 없으면 기존 3-table JOIN fallback.

        Returns:
            pd.DataFrame: trade_date, stock_code, foreign_net_amount,
                          institution_net_amount, close_price, free_float_shares,
                          foreign_sff, institution_sff (MV 경로만)
        """
        if is_mv_available():
            query = self._build_mv_query(stock_codes, start_date, end_date)
        else:
            query = self._build_raw_query(stock_codes, start_date, end_date)

        df = pd.read_sql(text(query), self.conn)
        if not df.empty and 'trade_date' in df.columns:
            df['trade_date'] = df['trade_date'].astype(str)
        return df

    @staticmethod
    def _build_mv_query(stock_codes=None, start_date=None, end_date=None):
        """mv_daily_sff에서 단순 SELECT (Sff 이미 계산됨)"""
        clauses = []
        if stock_codes:
            codes_str = "','".join(stock_codes)
            clauses.append(f"stock_code IN ('{codes_str}')")
        if start_date:
            clauses.append(f"trade_date >= '{start_date}'")
        if end_date:
            clauses.append(f"trade_date <= '{end_date}'")

        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return f"""
        SELECT trade_date, stock_code, foreign_net_amount, institution_net_amount,
               close_price, free_float_shares, foreign_sff, institution_sff
        FROM mv_daily_sff
        {where}
        ORDER BY stock_code, trade_date
        """

    @staticmethod
    def _build_raw_query(stock_codes=None, start_date=None, end_date=None):
        """기존 3-table JOIN fallback 쿼리"""
        extra_clauses = []
        if stock_codes:
            codes_str = "','".join(stock_codes)
            extra_clauses.append(f"AND it.stock_code IN ('{codes_str}')")
        if start_date:
            extra_clauses.append(f"AND it.time >= '{start_date}'")
        if end_date:
            extra_clauses.append(f"AND it.time <= '{end_date}'")
        extra = "\n          ".join(extra_clauses)

        return f"""
        SELECT
            it.time AS trade_date,
            it.stock_code,
            SUM(CASE WHEN it.investor_type = 'FOREIGN'      THEN it.net_buy_value ELSE 0 END) AS foreign_net_amount,
            SUM(CASE WHEN it.investor_type = 'INSTITUTION'  THEN it.net_buy_value ELSE 0 END) AS institution_net_amount,
            MAX(o.close_price)      AS close_price,
            MAX(ff.floating_shares) AS free_float_shares
        FROM investor_trading it
        JOIN ohlcv_daily o ON it.time = o.time AND it.stock_code = o.stock_code
        JOIN (
            SELECT DISTINCT ON (stock_code) stock_code, floating_shares
            FROM floating_shares
            ORDER BY stock_code, base_date DESC
        ) ff ON it.stock_code = ff.stock_code
        WHERE it.investor_type IN ('FOREIGN', 'INSTITUTION')
          AND o.close_price IS NOT NULL
          {extra}
        GROUP BY it.time, it.stock_code
        ORDER BY it.stock_code, it.time
        """

    def _query_combined_sff_only(self, stock_codes, start_date, end_date, institution_weight):
        """MV에서 combined_sff만 계산하여 로드 (3컬럼만 전송 → I/O 최소화)

        _get_sff_data() 전용 경로. Sff 개별값(foreign_sff, institution_sff)이 MV에 있으므로
        combined_sff를 SQL에서 직접 계산하여 최소 데이터만 전송.
        """
        clauses = []
        if stock_codes:
            codes_str = "','".join(stock_codes)
            clauses.append(f"stock_code IN ('{codes_str}')")
        if start_date:
            clauses.append(f"trade_date >= '{start_date}'")
        if end_date:
            clauses.append(f"trade_date <= '{end_date}'")
        where = "WHERE " + " AND ".join(clauses) if clauses else ""

        w = institution_weight
        query = f"""
        SELECT trade_date, stock_code,
               CASE WHEN (foreign_sff * institution_sff) > 0
                    THEN foreign_sff + institution_sff * {w}
                    ELSE foreign_sff
               END AS combined_sff
        FROM mv_daily_sff
        {where}
        ORDER BY stock_code, trade_date
        """
        df = pd.read_sql(text(query), self.conn)
        if not df.empty:
            df['trade_date'] = df['trade_date'].astype(str)
        return df

    @staticmethod
    def _calc_date_start(end_date, max_period=500, buffer_ratio=0.5):
        """Z-Score 계산에 필요한 최소 시작일 자동 계산

        Args:
            end_date: 종료일 (YYYY-MM-DD)
            max_period: 최장 Z-Score 기간 (영업일)
            buffer_ratio: warmup 버퍼 비율

        Returns:
            str: 시작일 (YYYY-MM-DD)
        """
        buffer = int(max_period * buffer_ratio)
        calendar_days = int((max_period + buffer) * 1.47)  # 영업일→달력일
        ref = datetime.strptime(end_date, '%Y-%m-%d')
        return (ref - timedelta(days=calendar_days)).strftime('%Y-%m-%d')

    # ------------------------------------------------------------------
    # 프리로드 (백테스트용)
    # ------------------------------------------------------------------

    def preload(self, end_date: Optional[str] = None):
        """
        백테스트 전 전체 원본 데이터를 메모리에 로드.
        이후 calculate_sff / _get_sff_data 호출 시 DB 쿼리 없이 메모리 필터링 사용.

        Args:
            end_date: 로드 종료일 (None이면 전체)
        """
        df = self._query_raw_data(end_date=end_date)
        # MV 경로에서 이미 Sff가 있을 수 있지만, preload는 raw 데이터로 저장
        # (_apply_sff_formula에서 weight를 적용하므로)
        self._preload_raw = df

    def clear_preload(self):
        """프리로드 데이터 삭제 (메모리 해제)"""
        self._preload_raw = None

    def _apply_sff_formula(self, df: pd.DataFrame) -> pd.DataFrame:
        """원본 데이터에서 Sff 계산 (내부 공통 메서드)"""
        df = df.copy()
        free_float_mcap = df['close_price'] * df['free_float_shares']
        df['foreign_sff'] = (df['foreign_net_amount'] / free_float_mcap) * 100
        df['institution_sff'] = (df['institution_net_amount'] / free_float_mcap) * 100
        institution_weight = self.config.get('institution_weight', 0.3)
        same_direction = (df['foreign_sff'] * df['institution_sff']) > 0
        df['combined_sff'] = np.where(
            same_direction,
            df['foreign_sff'] + df['institution_sff'] * institution_weight,
            df['foreign_sff']
        )
        return df.replace([np.inf, -np.inf], np.nan)

    def _apply_combined_sff(self, df: pd.DataFrame) -> pd.DataFrame:
        """MV 데이터(foreign_sff, institution_sff 이미 있음)에서 combined_sff만 계산"""
        df = df.copy()
        institution_weight = self.config.get('institution_weight', 0.3)
        same_direction = (df['foreign_sff'] * df['institution_sff']) > 0
        df['combined_sff'] = np.where(
            same_direction,
            df['foreign_sff'] + df['institution_sff'] * institution_weight,
            df['foreign_sff']
        )
        return df.replace([np.inf, -np.inf], np.nan)

    def _compute_sff(self, df: pd.DataFrame) -> pd.DataFrame:
        """raw 데이터 → Sff 계산 (MV 여부에 따라 자동 분기)"""
        if 'foreign_sff' in df.columns:
            # MV 경로: Sff 이미 있음, combined만 계산
            return self._apply_combined_sff(df)
        else:
            # fallback: 전체 Sff 계산
            return self._apply_sff_formula(df)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------

    def calculate_sff(self,
                     stock_codes: Optional[list] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Sff (Supply Force vs Free Float) 계산

        공식: Sff = (순매수금액 / 유통시가총액) × 100
        유통시가총액 = 종가 × 유통주식수

        Args:
            stock_codes: 종목 코드 리스트 (None이면 전체)
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)

        Returns:
            pd.DataFrame: (trade_date, stock_code, foreign_sff, institution_sff, combined_sff)

        Raises:
            ValueError: 유효하지 않은 종목 코드 또는 날짜 형식
        """

        # 보안: 입력 검증 (SQL 인젝션 방지)
        if stock_codes:
            stock_codes = validate_stock_codes(stock_codes)

        if start_date and not validate_date_format(start_date):
            raise ValueError(f"Invalid start_date format: {start_date}. Expected YYYY-MM-DD")

        if end_date and not validate_date_format(end_date):
            raise ValueError(f"Invalid end_date format: {end_date}. Expected YYYY-MM-DD")

        # 프리로드 데이터 우선 사용 (DB 쿼리 없음)
        if self._preload_raw is not None:
            df = self._preload_raw.copy()
            if stock_codes:
                df = df[df['stock_code'].isin(stock_codes)]
            if start_date:
                df = df[df['trade_date'] >= start_date]
            if end_date:
                df = df[df['trade_date'] <= end_date]
            if df.empty:
                return pd.DataFrame(columns=['trade_date', 'stock_code', 'foreign_sff',
                                            'institution_sff', 'combined_sff'])
            result = self._compute_sff(df)
            return result[['trade_date', 'stock_code', 'foreign_sff', 'institution_sff', 'combined_sff']]

        # DB 쿼리 (MV 또는 원본 테이블)
        # 날짜 필터 자동 계산 (end_date만 있고 start_date 없으면)
        effective_start = start_date
        if effective_start is None and end_date is not None:
            effective_start = self._calc_date_start(end_date)

        df = self._query_raw_data(stock_codes, effective_start, end_date)

        if df.empty:
            print("[WARN] No data found for Sff calculation")
            return pd.DataFrame(columns=['trade_date', 'stock_code', 'foreign_sff',
                                        'institution_sff', 'combined_sff'])

        result = self._compute_sff(df)
        return result[['trade_date', 'stock_code', 'foreign_sff', 'institution_sff', 'combined_sff']]

    def calculate_zscore(self,
                        stock_codes: Optional[list] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Z-Score 계산 (수급 이상 탐지) — 벡터화 구현

        공식: Z = (X - μ) / σ
        μ, σ = 최근 N일 이동평균 및 표준편차

        조건부 공식:
            같은 방향(today·mean > 0): (today - mean) / std (폭발 감지)
            방향 전환(today·mean ≤ 0): today / std (크기만 평가)

        Z > 2.0: 통계적으로 유의미한 강한 매수세
        Z < -2.0: 통계적으로 유의미한 강한 매도세

        Args:
            stock_codes: 종목 코드 리스트 (None이면 전체)
            end_date: 종료일 (YYYY-MM-DD, None이면 최신 데이터까지)

        Returns:
            pd.DataFrame: (trade_date, stock_code, foreign_zscore, institution_zscore, combined_zscore)
        """

        window = self.config['z_score_window']
        min_points = min(self.config['min_data_points'], max(1, window // 2))

        # 보안: 입력 검증
        if stock_codes:
            stock_codes = validate_stock_codes(stock_codes)

        # Sff 데이터 로드 (preload 또는 DB)
        if self._preload_raw is not None:
            df = self._preload_raw.copy()
            if stock_codes:
                df = df[df['stock_code'].isin(stock_codes)]
            if end_date:
                df = df[df['trade_date'] <= end_date]
            if df.empty:
                return pd.DataFrame()
            df = self._compute_sff(df)
        else:
            # 날짜 범위 최적화
            # - 전체 종목(stock_codes=None): window+buffer만 로드 (이상수급용, 속도 우선)
            # - 특정 종목(stock_codes 지정): 전체 이력 로드 (종목상세 차트, 데이터 소량)
            effective_end = end_date
            if effective_end is None:
                try:
                    table = 'mv_daily_sff' if is_mv_available() else 'investor_trading'
                    col = 'trade_date' if is_mv_available() else 'time'
                    r = pd.read_sql(text(f"SELECT MAX({col}) AS d FROM {table}"), self.conn)
                    effective_end = str(r.iloc[0]['d'])
                except Exception:
                    effective_end = None

            effective_start = None
            if effective_end and not stock_codes:
                # 전체 종목 조회 시에만 날짜 범위 최적화 (데이터 로드량 절감)
                # 단일/소수 종목은 전체 이력 로드 (추이 차트용)
                effective_start = self._calc_date_start(effective_end, max_period=window)

            df = self._query_raw_data(stock_codes, effective_start, effective_end)
            if df.empty:
                return pd.DataFrame()
            df = self._compute_sff(df)

        # 벡터화 Z-Score 계산 (groupby.transform — per-stock 루프 제거)
        df = df.sort_values(['stock_code', 'trade_date'])

        for col in ['foreign_sff', 'institution_sff', 'combined_sff']:
            zscore_col = col.replace('_sff', '_zscore')
            rolling_mean = df.groupby('stock_code')[col].transform(
                lambda x: x.rolling(window=window, min_periods=min_points).mean()
            )
            rolling_std = df.groupby('stock_code')[col].transform(
                lambda x: x.rolling(window=window, min_periods=min_points).std()
            )

            # 조건부 Z-Score: 부호 전환 시 과잉 반응 방지
            same_sign = (df[col] * rolling_mean) > 0
            df[zscore_col] = np.where(
                same_sign,
                (df[col] - rolling_mean) / rolling_std,
                df[col] / rolling_std
            )

        # inf/nan 처리 (std=0인 경우)
        df = df.replace([np.inf, -np.inf], np.nan)

        return df[['trade_date', 'stock_code', 'foreign_sff', 'institution_sff',
                    'combined_sff', 'foreign_zscore', 'institution_zscore', 'combined_zscore']]

    def get_abnormal_supply(self,
                           threshold: float = 2.0,
                           end_date: Optional[str] = None,
                           top_n: int = 20,
                           direction: str = 'both') -> pd.DataFrame:
        """
        이상 수급 종목 탐지

        Args:
            threshold: Z-Score 임계값 (기본: 2.0 = 표준편차 2배)
            end_date: 종료일 (YYYY-MM-DD)
            top_n: 상위 N개 종목 반환
            direction: 'buy' (매수), 'sell' (매도), 'both' (양방향)

        Returns:
            pd.DataFrame: 이상 수급 종목 리스트 (종목명 포함)
        """

        df = self.calculate_zscore(end_date=end_date)

        if df.empty:
            return pd.DataFrame()

        # 최근 날짜 데이터만 사용 (가장 최근 거래일)
        latest_date = df['trade_date'].max()
        df_latest = df[df['trade_date'] == latest_date].copy()

        # 방향 필터링
        if direction == 'buy':
            df_filtered = df_latest[
                (df_latest['foreign_zscore'] > threshold) |
                (df_latest['institution_zscore'] > threshold) |
                (df_latest['combined_zscore'] > threshold)
            ]
            sort_ascending = False
        elif direction == 'sell':
            df_filtered = df_latest[
                (df_latest['foreign_zscore'] < -threshold) |
                (df_latest['institution_zscore'] < -threshold) |
                (df_latest['combined_zscore'] < -threshold)
            ]
            sort_ascending = True
        else:  # both
            df_filtered = df_latest[
                (df_latest['foreign_zscore'].abs() > threshold) |
                (df_latest['institution_zscore'].abs() > threshold) |
                (df_latest['combined_zscore'].abs() > threshold)
            ]
            sort_ascending = False

        # 종목명 + 섹터 추가
        df_stocks = pd.read_sql(text(
            "SELECT s.stock_code, s.stock_name, ss.fics_sector AS sector "
            "FROM stocks s LEFT JOIN stock_sectors ss ON s.stock_code = ss.stock_code"
        ), self.conn)
        df_filtered = df_filtered.merge(df_stocks, on='stock_code', how='left')

        # combined_zscore로 정렬 후 상위 N개
        df_result = df_filtered.sort_values('combined_zscore', ascending=sort_ascending).head(top_n)

        return df_result[['stock_code', 'stock_name', 'sector', 'trade_date',
                         'foreign_sff', 'institution_sff', 'combined_sff',
                         'foreign_zscore', 'institution_zscore', 'combined_zscore']]

    def get_sff_summary(self, stock_code: str, days: int = 30) -> dict:
        """
        특정 종목의 Sff 요약 통계

        Args:
            stock_code: 종목 코드
            days: 최근 N일

        Returns:
            dict: 통계 요약 (평균, 최대, 최소, 표준편차)
        """

        df = self.calculate_sff(stock_codes=[stock_code])

        if df.empty:
            return {}

        # 최근 N일 데이터
        df_recent = df.sort_values('trade_date', ascending=False).head(days)

        summary = {
            'stock_code': stock_code,
            'period_days': len(df_recent),
            'foreign': {
                'mean': df_recent['foreign_sff'].mean(),
                'std': df_recent['foreign_sff'].std(),
                'max': df_recent['foreign_sff'].max(),
                'min': df_recent['foreign_sff'].min()
            },
            'institution': {
                'mean': df_recent['institution_sff'].mean(),
                'std': df_recent['institution_sff'].std(),
                'max': df_recent['institution_sff'].max(),
                'min': df_recent['institution_sff'].min()
            },
            'combined': {
                'mean': df_recent['combined_sff'].mean(),
                'std': df_recent['combined_sff'].std(),
                'max': df_recent['combined_sff'].max(),
                'min': df_recent['combined_sff'].min()
            }
        }

        return summary

    def _get_sff_data(self, stock_codes: Optional[list] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
        """
        [Stage 2] Sff 데이터 추출 메서드 (캐싱용)

        기존 calculate_sff() 로직을 재사용하되,
        OptimizedMultiPeriodCalculator에서 호출하기 쉽게 분리

        Args:
            stock_codes: 종목 코드 리스트 (None이면 전체)
            end_date: 종료일 (YYYY-MM-DD, None이면 최신까지)

        Returns:
            pd.DataFrame: (trade_date, stock_code, combined_sff)

        Raises:
            ValueError: 유효하지 않은 종목 코드
        """
        # 보안: 입력 검증 (SQL 인젝션 방지)
        if stock_codes:
            stock_codes = validate_stock_codes(stock_codes)

        # 프리로드 데이터 우선 사용 (DB 쿼리 없음)
        if self._preload_raw is not None:
            df = self._preload_raw.copy()
            if stock_codes:
                df = df[df['stock_code'].isin(stock_codes)]
            if end_date:
                df = df[df['trade_date'] <= end_date]
            if df.empty:
                return pd.DataFrame(columns=['trade_date', 'stock_code', 'combined_sff'])
            result = self._compute_sff(df)
            return result[['trade_date', 'stock_code', 'combined_sff']]

        # DB 쿼리 (MV 또는 원본 테이블)
        # 날짜 필터 자동 계산 (end_date만 있고 preload 없으면)
        effective_start = None
        if end_date is not None:
            effective_start = self._calc_date_start(end_date)

        # MV 경로: combined_sff를 SQL에서 직접 계산 (3컬럼만 전송 → I/O 절감)
        if is_mv_available():
            institution_weight = self.config.get('institution_weight', 0.3)
            df = self._query_combined_sff_only(
                stock_codes, effective_start, end_date, institution_weight)
            if df.empty:
                print("[WARN] No data found for Sff calculation")
                return pd.DataFrame(columns=['trade_date', 'stock_code', 'combined_sff'])
            return df

        df = self._query_raw_data(stock_codes, effective_start, end_date)

        if df.empty:
            print("[WARN] No data found for Sff calculation")
            return pd.DataFrame(columns=['trade_date', 'stock_code', 'combined_sff'])

        result = self._compute_sff(df)
        return result[['trade_date', 'stock_code', 'combined_sff']]
