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
from typing import Optional


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
            'min_data_points': 30      # 최소 30일 필요
        }

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
        """

        # WHERE 절 생성
        where_clauses = ["close_price IS NOT NULL", "free_float_shares IS NOT NULL"]

        if stock_codes:
            codes_str = "','".join(stock_codes)
            where_clauses.append(f"stock_code IN ('{codes_str}')")
        if start_date:
            where_clauses.append(f"trade_date >= '{start_date}'")
        if end_date:
            where_clauses.append(f"trade_date <= '{end_date}'")

        where_sql = "WHERE " + " AND ".join(where_clauses)

        # 쿼리 실행
        query = f"""
        SELECT
            trade_date,
            stock_code,
            foreign_net_amount,
            institution_net_amount,
            close_price,
            free_float_shares,
            (close_price * free_float_shares) as free_float_mcap
        FROM investor_flows
        {where_sql}
        ORDER BY stock_code, trade_date
        """

        df = pd.read_sql(query, self.conn)

        if df.empty:
            print("[WARN] No data found for Sff calculation")
            return pd.DataFrame(columns=['trade_date', 'stock_code', 'foreign_sff',
                                        'institution_sff', 'combined_sff'])

        # Sff 계산 (백분율)
        df['foreign_sff'] = (df['foreign_net_amount'] / df['free_float_mcap']) * 100
        df['institution_sff'] = (df['institution_net_amount'] / df['free_float_mcap']) * 100
        df['combined_sff'] = df['foreign_sff'] + df['institution_sff']

        # inf/nan 처리 (division by zero)
        df = df.replace([np.inf, -np.inf], np.nan)

        return df[['trade_date', 'stock_code', 'foreign_sff', 'institution_sff', 'combined_sff']]

    def calculate_zscore(self,
                        stock_codes: Optional[list] = None,
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Z-Score 계산 (수급 이상 탐지)

        공식: Z = (X - μ) / σ
        μ, σ = 최근 60일 이동평균 및 표준편차

        Z > 2.0: 통계적으로 유의미한 강한 매수세
        Z < -2.0: 통계적으로 유의미한 강한 매도세

        Args:
            stock_codes: 종목 코드 리스트 (None이면 전체)
            end_date: 종료일 (YYYY-MM-DD, None이면 최신 데이터까지)

        Returns:
            pd.DataFrame: (trade_date, stock_code, foreign_zscore, institution_zscore, combined_zscore)
        """

        window = self.config['z_score_window']
        min_points = self.config['min_data_points']

        # Sff 데이터 가져오기
        df = self.calculate_sff(stock_codes=stock_codes, end_date=end_date)

        if df.empty:
            return pd.DataFrame()

        # 종목별로 Z-Score 계산
        results = []

        for stock_code in df['stock_code'].unique():
            df_stock = df[df['stock_code'] == stock_code].sort_values('trade_date').copy()

            # 데이터 부족 시 스킵
            if len(df_stock) < min_points:
                continue

            # 각 유형별 Z-Score 계산
            for col in ['foreign_sff', 'institution_sff', 'combined_sff']:
                rolling_mean = df_stock[col].rolling(window=window, min_periods=min_points).mean()
                rolling_std = df_stock[col].rolling(window=window, min_periods=min_points).std()

                # Z-Score = (X - 평균) / 표준편차
                zscore_col = col.replace('_sff', '_zscore')
                df_stock[zscore_col] = (df_stock[col] - rolling_mean) / rolling_std

            results.append(df_stock)

        if not results:
            print("[WARN] Insufficient data for Z-score calculation")
            return pd.DataFrame()

        df_final = pd.concat(results, ignore_index=True)

        # inf/nan 처리 (std=0인 경우)
        df_final = df_final.replace([np.inf, -np.inf], np.nan)

        return df_final[['trade_date', 'stock_code', 'foreign_sff', 'institution_sff',
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

        # 종목명 추가
        df_stocks = pd.read_sql('SELECT stock_code, stock_name FROM stocks', self.conn)
        df_filtered = df_filtered.merge(df_stocks, on='stock_code', how='left')

        # combined_zscore로 정렬 후 상위 N개
        df_result = df_filtered.sort_values('combined_zscore', ascending=sort_ascending).head(top_n)

        return df_result[['stock_code', 'stock_name', 'trade_date',
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
