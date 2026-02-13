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
from typing import Optional, Dict, List
from src.utils import validate_stock_codes


class SignalDetector:
    """수급 시그널 탐지 클래스"""

    def __init__(self, conn, config: Optional[dict] = None):
        """
        Args:
            conn: 데이터베이스 연결
            config: 시그널 탐지 설정
                - ma_short: 단기 이동평균 (기본: 5일)
                - ma_long: 장기 이동평균 (기본: 20일)
                - acceleration_window: 가속도 계산 윈도우 (기본: 5일)
                - sync_threshold: 동조 판단 임계값 (기본: 0)
        """
        self.conn = conn
        self.config = config or self._get_default_config()

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

    def _load_supply_data(self, stock_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        수급 데이터 로드 (내부 헬퍼 메서드)

        Args:
            stock_codes: 종목 코드 리스트 (None이면 전체)

        Returns:
            pd.DataFrame: (trade_date, stock_code, foreign_net_amount, institution_net_amount)
        """
        # 빈 리스트 처리 (명시적으로 빈 결과 요청)
        if stock_codes is not None and len(stock_codes) == 0:
            return pd.DataFrame(columns=['trade_date', 'stock_code', 'foreign_net_amount', 'institution_net_amount'])

        # 보안: 입력 검증
        if stock_codes:
            stock_codes = validate_stock_codes(stock_codes)

        # WHERE 절 생성
        where_clauses = []
        if stock_codes:
            codes_str = "','".join(stock_codes)
            where_clauses.append(f"stock_code IN ('{codes_str}')")

        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        # 쿼리 실행
        query = f"""
        SELECT
            trade_date,
            stock_code,
            foreign_net_amount,
            institution_net_amount
        FROM investor_flows
        {where_sql}
        ORDER BY stock_code, trade_date
        """

        df = pd.read_sql(query, self.conn)

        if df.empty:
            print("[WARN] No supply data found")
            return pd.DataFrame()

        return df

    def detect_ma_crossover(self, stock_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        MA 골든크로스 탐지

        공식:
            - 외국인 순매수 5일MA > 20일MA
            - 직전일: 5일MA <= 20일MA (크로스 감지)

        Args:
            stock_codes: 종목 코드 리스트 (None이면 전체)

        Returns:
            pd.DataFrame: 골든크로스 종목
                - stock_code: 종목코드
                - trade_date: 크로스 발생일 (최신 거래일)
                - ma_short: 5일MA 값
                - ma_long: 20일MA 값
                - ma_diff: ma_short - ma_long (차이)
                - is_golden_cross: True (골든크로스 발생)

        Example:
            >>> detector = SignalDetector(conn)
            >>> df_cross = detector.detect_ma_crossover()
            >>> print(df_cross[['stock_code', 'ma_short', 'ma_long']])
        """
        df = self._load_supply_data(stock_codes)

        if df.empty:
            return pd.DataFrame(columns=[
                'stock_code', 'trade_date', 'ma_short', 'ma_long', 'ma_diff', 'is_golden_cross'
            ])

        ma_short = self.config['ma_short']
        ma_long = self.config['ma_long']

        results = []

        for stock_code in df['stock_code'].unique():
            df_stock = df[df['stock_code'] == stock_code].sort_values('trade_date').copy()

            # 데이터 부족 시 스킵
            if len(df_stock) < ma_long + 1:
                continue

            # 이동평균 계산
            df_stock['ma_short'] = df_stock['foreign_net_amount'].rolling(window=ma_short).mean()
            df_stock['ma_long'] = df_stock['foreign_net_amount'].rolling(window=ma_long).mean()

            # 최근 2일 데이터
            df_recent = df_stock.tail(2)

            if len(df_recent) < 2:
                continue

            prev_row = df_recent.iloc[0]
            curr_row = df_recent.iloc[1]

            # 골든크로스 조건:
            # - 현재: ma_short > ma_long
            # - 직전: ma_short <= ma_long
            if (curr_row['ma_short'] > curr_row['ma_long'] and
                prev_row['ma_short'] <= prev_row['ma_long']):

                results.append({
                    'stock_code': stock_code,
                    'trade_date': curr_row['trade_date'],
                    'ma_short': curr_row['ma_short'],
                    'ma_long': curr_row['ma_long'],
                    'ma_diff': curr_row['ma_short'] - curr_row['ma_long'],
                    'is_golden_cross': True
                })

        if not results:
            return pd.DataFrame(columns=['stock_code', 'trade_date', 'ma_short',
                                        'ma_long', 'ma_diff', 'is_golden_cross'])

        return pd.DataFrame(results)

    def calculate_acceleration(self, stock_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        수급 가속도 계산

        공식:
            Acceleration = (최근 5일 평균 순매수) / (직전 5일 평균 순매수)
            - 외국인 + 기관 합산 기준

        해석:
            - 가속도 > 1.5: 수급 가속 (매수세 강화)
            - 가속도 < 0.7: 수급 둔화 (매수세 약화)
            - 0.7 ~ 1.5: 중립

        Args:
            stock_codes: 종목 코드 리스트 (None이면 전체)

        Returns:
            pd.DataFrame: 가속도 결과
                - stock_code: 종목코드
                - trade_date: 거래일 (최신)
                - recent_avg: 최근 5일 평균
                - prev_avg: 직전 5일 평균
                - acceleration: 가속도 배율

        Example:
            >>> df_accel = detector.calculate_acceleration()
            >>> df_hot = df_accel[df_accel['acceleration'] > 1.5]
        """
        df = self._load_supply_data(stock_codes)

        if df.empty:
            return pd.DataFrame(columns=[
                'stock_code', 'trade_date', 'recent_avg', 'prev_avg', 'acceleration'
            ])

        window = self.config['acceleration_window']

        # 외국인 + 기관 합산
        df['combined_net'] = df['foreign_net_amount'] + df['institution_net_amount']

        results = []

        for stock_code in df['stock_code'].unique():
            df_stock = df[df['stock_code'] == stock_code].sort_values('trade_date').copy()

            # 데이터 부족 시 스킵 (최소 window*2일 필요)
            if len(df_stock) < window * 2:
                continue

            # 최근 window일 평균
            recent_avg = df_stock['combined_net'].tail(window).mean()

            # 직전 window일 평균
            prev_avg = df_stock['combined_net'].iloc[-(window*2):-window].mean()

            # 가속도 계산 (0으로 나누기 방지)
            if abs(prev_avg) < 1e-6:
                acceleration = np.nan
            else:
                acceleration = recent_avg / prev_avg

            latest_date = df_stock['trade_date'].iloc[-1]

            results.append({
                'stock_code': stock_code,
                'trade_date': latest_date,
                'recent_avg': recent_avg,
                'prev_avg': prev_avg,
                'acceleration': acceleration
            })

        if not results:
            return pd.DataFrame(columns=['stock_code', 'trade_date', 'recent_avg',
                                        'prev_avg', 'acceleration'])

        df_result = pd.DataFrame(results)

        # inf/nan 처리
        df_result = df_result.replace([np.inf, -np.inf], np.nan)

        return df_result

    def calculate_sync_rate(self, stock_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        외인-기관 동조율 계산

        공식:
            동조율 = (외인·기관 동시 매수일 수 / 전체 일수) × 100%
            - 최근 20일 기준
            - 동시 매수: 외인 순매수 > 0 AND 기관 순매수 > 0

        해석:
            - 동조율 > 70%: 강한 동조 (확신도 높음)
            - 동조율 < 30%: 약한 동조 (확신도 낮음)

        Args:
            stock_codes: 종목 코드 리스트 (None이면 전체)

        Returns:
            pd.DataFrame: 동조율 결과
                - stock_code: 종목코드
                - trade_date: 거래일 (최신)
                - sync_days: 동조 일수
                - total_days: 전체 일수
                - sync_rate: 동조율 (%)

        Example:
            >>> df_sync = detector.calculate_sync_rate()
            >>> df_strong = df_sync[df_sync['sync_rate'] > 70]
        """
        df = self._load_supply_data(stock_codes)

        if df.empty:
            return pd.DataFrame(columns=[
                'stock_code', 'trade_date', 'sync_days', 'total_days', 'sync_rate'
            ])

        window = self.config['sync_window']
        threshold = self.config['sync_threshold']

        results = []

        for stock_code in df['stock_code'].unique():
            df_stock = df[df['stock_code'] == stock_code].sort_values('trade_date').copy()

            # 데이터 부족 시 스킵
            if len(df_stock) < window:
                continue

            # 최근 window일 데이터
            df_recent = df_stock.tail(window)

            # 동시 매수일 수 계산
            sync_days = (
                (df_recent['foreign_net_amount'] > threshold) &
                (df_recent['institution_net_amount'] > threshold)
            ).sum()

            total_days = len(df_recent)
            sync_rate = (sync_days / total_days) * 100

            latest_date = df_stock['trade_date'].iloc[-1]

            results.append({
                'stock_code': stock_code,
                'trade_date': latest_date,
                'sync_days': sync_days,
                'total_days': total_days,
                'sync_rate': sync_rate
            })

        if not results:
            return pd.DataFrame(columns=['stock_code', 'trade_date', 'sync_days',
                                        'total_days', 'sync_rate'])

        return pd.DataFrame(results)

    def detect_all_signals(self, stock_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        모든 시그널 통합 탐지 (메인 메서드)

        Args:
            stock_codes: 종목 코드 리스트 (None이면 전체)

        Returns:
            pd.DataFrame: 통합 시그널 결과
                - stock_code: 종목코드
                - ma_cross: MA 골든크로스 여부 (True/False)
                - ma_diff: MA 차이 (5일MA - 20일MA)
                - acceleration: 수급 가속도 배율
                - sync_rate: 동조율 (%)
                - signal_count: 활성 시그널 개수 (0~3)
                - signal_list: 활성 시그널 리스트

        Example:
            >>> detector = SignalDetector(conn)
            >>> df_signals = detector.detect_all_signals()
            >>> print(df_signals[['stock_code', 'signal_count', 'signal_list']])
        """
        # 1. MA 골든크로스
        df_ma = self.detect_ma_crossover(stock_codes)
        df_ma = df_ma[['stock_code', 'ma_diff', 'is_golden_cross']].rename(
            columns={'is_golden_cross': 'ma_cross'}
        )

        # 2. 수급 가속도
        df_accel = self.calculate_acceleration(stock_codes)
        df_accel = df_accel[['stock_code', 'acceleration']]

        # 3. 동조율
        df_sync = self.calculate_sync_rate(stock_codes)
        df_sync = df_sync[['stock_code', 'sync_rate']]

        # 4. 통합 (outer join으로 전체 종목 유지)
        df_result = df_ma if not df_ma.empty else pd.DataFrame(columns=['stock_code', 'ma_diff', 'ma_cross'])

        if not df_accel.empty:
            df_result = df_result.merge(df_accel, on='stock_code', how='outer')

        if not df_sync.empty:
            df_result = df_result.merge(df_sync, on='stock_code', how='outer')

        # NaN 처리
        if 'ma_cross' in df_result.columns:
            df_result['ma_cross'] = df_result['ma_cross'].fillna(False)
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

        Example:
            >>> # 2개 이상 시그널 활성화된 종목
            >>> df_strong = detector.get_strong_signals(min_signal_count=2)
        """
        df_signals = self.detect_all_signals(stock_codes)

        # 필터링
        df_filtered = df_signals[df_signals['signal_count'] >= min_signal_count].copy()

        # 정렬 (signal_count 내림차순)
        df_filtered = df_filtered.sort_values('signal_count', ascending=False)

        return df_filtered
