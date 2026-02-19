"""
백테스트 엔진 모듈

롤링 윈도우 시뮬레이션 구현:
- 과거 데이터로 Stage 1-3 실행 (미래 데이터 차단)
- 진입/청산 조건 관리
- 포트폴리오 시뮬레이션
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import sqlite3

from .portfolio import Portfolio, Trade, Position
from .precomputer import BacktestPrecomputer
from src.analyzer.normalizer import SupplyNormalizer
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
from src.analyzer.pattern_classifier import PatternClassifier
from src.analyzer.signal_detector import SignalDetector


class BacktestConfig:
    """백테스트 설정"""

    def __init__(self,
                 initial_capital: float = 10_000_000,  # 초기 자본금 (천만원)
                 max_positions: int = 10,  # 최대 동시 보유 종목
                 min_score: float = 70,  # 최소 패턴 점수
                 min_signals: int = 2,  # 최소 시그널 개수
                 target_return: float = 0.15,  # 목표 수익률 (+15%)
                 stop_loss: float = -0.075,  # 손절 비율 (-7.5%)
                 max_hold_days: int = 999,  # 최대 보유 기간 (999 = 사실상 무제한)
                 reverse_signal_threshold: float = 60,  # 반대 수급 손절 점수 (60점 이상)
                 allowed_patterns: Optional[List[str]] = None,  # 허용 패턴 (None이면 전체)
                 strategy: str = 'long',  # 'long', 'short', 'both'
                 institution_weight: float = 0.3,  # 기관 가중치 (0.0=외국인만, 0.3=기본, 0.5=기관 강조)
                 force_exit_on_end: bool = False):  # 백테스트 종료 시 강제 청산 여부
        """
        백테스트 설정 초기화

        Args:
            initial_capital: 초기 자본금
            max_positions: 최대 동시 보유 종목 수
            min_score: 진입 최소 점수 (0~100)
            min_signals: 진입 최소 시그널 개수 (0~3)
            target_return: 목표 수익률 (예: 0.15 = +15%, 순수 가격 변화율)
            stop_loss: 손절 비율 (예: -0.075 = -7.5%, 순수 가격 변화율)
            max_hold_days: 시간 손절 (N일 보유 후 강제 청산, 999 = 무제한)
            reverse_signal_threshold: 반대 수급 손절 점수 (Long→매도 60점 이상, Short→매수 60점 이상)
            allowed_patterns: 허용 패턴 리스트 (예: ['모멘텀형', '지속형'])
            strategy: 전략 방향 ('long': 순매수, 'short': 순매도, 'both': 롱+숏)
            institution_weight: 기관 가중치 (0.0=외국인만, 0.3=기본, 0.5=기관 강조)
            force_exit_on_end: 백테스트 종료일에 강제 청산 여부 (기본: False)
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.min_score = min_score
        self.min_signals = min_signals
        self.target_return = target_return
        self.stop_loss = stop_loss
        self.max_hold_days = max_hold_days
        self.reverse_signal_threshold = reverse_signal_threshold
        self.allowed_patterns = allowed_patterns
        self.strategy = strategy
        self.institution_weight = institution_weight
        self.force_exit_on_end = force_exit_on_end

        if strategy not in ['long', 'short', 'both']:
            raise ValueError(f"strategy must be 'long', 'short', or 'both', got: {strategy}")


class BacktestEngine:
    """백테스트 엔진"""

    def __init__(self, conn: sqlite3.Connection, config: Optional[BacktestConfig] = None):
        """
        초기화

        Args:
            conn: 데이터베이스 연결
            config: 백테스트 설정 (None이면 기본값)
        """
        self.conn = conn
        self.config = config or BacktestConfig()

        # Stage 1-3 모듈 초기화
        self.normalizer = SupplyNormalizer(conn, config={
            'z_score_window': 60,
            'min_data_points': 30,
            'institution_weight': self.config.institution_weight,
        })
        self.calculator = OptimizedMultiPeriodCalculator(
            self.normalizer, enable_caching=False  # 백테스트는 end_date가 매번 바뀌므로 캐싱 비활성화
        )
        self.classifier = PatternClassifier()
        self.signal_detector = SignalDetector(conn)

        # 기간 설정 (Stage 2 히트맵용)
        self.periods = {
            '1W': 5,
            '1M': 21,
            '3M': 63,
            '6M': 126,
            '1Y': 252,
            '2Y': 504
        }

        # 포트폴리오
        self.portfolio = Portfolio(
            initial_capital=self.config.initial_capital,
            max_positions=self.config.max_positions
        )

        # 사전 계산 결과 (preload_data=True일 때 활성화)
        self._precomputed = None  # PrecomputeResult or None

        # 백테스트 결과
        self.daily_values: List[Dict] = []  # 일별 포트폴리오 가치

    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        거래일 목록 조회 (DB에 존재하는 날짜만)

        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)

        Returns:
            거래일 리스트 (정렬됨)
        """
        query = """
        SELECT DISTINCT trade_date
        FROM investor_flows
        WHERE trade_date BETWEEN ? AND ?
        ORDER BY trade_date
        """
        df = pd.read_sql(query, self.conn, params=[start_date, end_date])
        return df['trade_date'].tolist()

    def get_price(self, stock_code: str, trade_date: str) -> Optional[float]:
        """
        특정 종목의 종가 조회

        Args:
            stock_code: 종목 코드
            trade_date: 거래일 (YYYY-MM-DD)

        Returns:
            종가 (없으면 None)

        Note:
            DB에 시가 데이터가 없으므로 종가만 사용
            진입/청산 모두 당일 종가로 계산
        """
        # 사전 계산 데이터 우선 사용 (O(1) lookup)
        if self._precomputed is not None:
            price = self._precomputed.price_lookup.get((stock_code, trade_date))
            if price is not None:
                return float(price)
            return None

        query = """
        SELECT close_price
        FROM investor_flows
        WHERE stock_code = ? AND trade_date = ?
        LIMIT 1
        """
        df = pd.read_sql(query, self.conn, params=[stock_code, trade_date])

        if df.empty or pd.isna(df.iloc[0]['close_price']):
            return None

        return float(df.iloc[0]['close_price'])

    def get_stock_name(self, stock_code: str) -> str:
        """종목명 조회"""
        # 사전 계산 데이터 우선 사용 (O(1) lookup)
        if self._precomputed is not None:
            return self._precomputed.stock_names.get(stock_code, stock_code)

        query = """
        SELECT stock_name
        FROM stocks
        WHERE stock_code = ?
        LIMIT 1
        """
        df = pd.read_sql(query, self.conn, params=[stock_code])
        if df.empty:
            return stock_code
        return df.iloc[0]['stock_name']

    def _scan_signals_on_date(self, end_date: str, direction: str = 'long') -> pd.DataFrame:
        """
        특정 날짜 기준 Stage 1-3 실행 (미래 데이터 차단!)

        Args:
            end_date: 기준일 (YYYY-MM-DD)
            direction: 'long' (순매수) 또는 'short' (순매도)

        Returns:
            pd.DataFrame: 패턴 분류 + 시그널 통합 결과
                - stock_code, stock_name
                - 1W~2Y, recent, momentum, weighted, average
                - pattern, score, direction
                - ma_cross, acceleration, sync_rate, signal_count
        """
        # 사전 계산 데이터가 있으면 빠른 경로 사용
        if self._precomputed is not None:
            return self._scan_signals_on_date_fast(end_date, direction)

        # Stage 1: Z-Score 계산 (end_date까지만 사용!)
        zscore_latest = self.normalizer.calculate_zscore(end_date=end_date)

        if zscore_latest.empty:
            return pd.DataFrame()

        # Stage 2: 히트맵 계산 (end_date 적용 - 미래 데이터 차단!)
        zscore_matrix = self.calculator.calculate_multi_period_zscores(
            periods_dict=self.periods,
            stock_codes=zscore_latest['stock_code'].tolist(),
            end_date=end_date  # 미래 데이터 누수 방지
        )

        if zscore_matrix.empty:
            return pd.DataFrame()

        # stock_code를 인덱스에서 컬럼으로 변환
        zscore_matrix = zscore_matrix.reset_index()

        # direction별 필터링 (양수/음수 Z-Score)
        # Stage 2 출력에서 대표 기간(1W)으로 구분
        if direction == 'long':
            # Long: 양수 Z-Score만 (순매수)
            zscore_matrix = zscore_matrix[zscore_matrix['1W'] > 0].copy()
        else:
            # Short: 음수 Z-Score만 (순매도)
            zscore_matrix = zscore_matrix[zscore_matrix['1W'] < 0].copy()

        if zscore_matrix.empty:
            return pd.DataFrame()

        # Stage 3-1: 패턴 분류 (direction별)
        pattern_result = self.classifier.classify_all(zscore_matrix, direction=direction)

        # Stage 3-2: 시그널 탐지 (end_date 적용 - 미래 데이터 차단!)
        signal_result = self.signal_detector.detect_all_signals(
            stock_codes=pattern_result['stock_code'].tolist(),
            end_date=end_date
        )

        # 통합
        result = pd.merge(pattern_result, signal_result, on='stock_code', how='left')

        # 종목명 추가
        stock_names = []
        for code in result['stock_code']:
            stock_names.append(self.get_stock_name(code))
        result.insert(1, 'stock_name', stock_names)

        return result

    def _scan_signals_on_date_fast(self, end_date: str, direction: str = 'long') -> pd.DataFrame:
        """
        사전 계산 데이터를 사용한 빠른 시그널 스캔

        _scan_signals_on_date()의 빠른 경로. DB 쿼리 없이
        O(1) lookup + 패턴 분류(~0.01초)로 동일한 결과 반환.
        """
        pc = self._precomputed

        # 1. Z-Score lookup (O(1))
        try:
            zscore_on_date = pc.zscore_all_dates.loc[end_date].copy()
        except KeyError:
            return pd.DataFrame()

        zscore_matrix = zscore_on_date.reset_index()  # stock_code → column

        # 2. Direction filter (1W > 0: long, 1W < 0: short)
        if direction == 'long':
            zscore_matrix = zscore_matrix[zscore_matrix['1W'] > 0].copy()
        else:
            zscore_matrix = zscore_matrix[zscore_matrix['1W'] < 0].copy()

        if zscore_matrix.empty:
            return pd.DataFrame()

        # 3. Pattern classification (~0.01초, DB 접근 없음)
        pattern_result = self.classifier.classify_all(zscore_matrix, direction=direction)

        # 4. Signal lookup (O(1))
        try:
            signals_on_date = pc.signals_all_dates.loc[end_date].copy()
            signals_on_date = signals_on_date.reset_index()
        except KeyError:
            signals_on_date = pd.DataFrame()

        # 5. Merge
        if not signals_on_date.empty:
            result = pd.merge(pattern_result, signals_on_date, on='stock_code', how='left')
        else:
            result = pattern_result.copy()
            result['ma_cross'] = False
            result['ma_diff'] = np.nan
            result['acceleration'] = np.nan
            result['sync_rate'] = np.nan
            result['signal_count'] = 0

        # 6. Fill defaults
        result['signal_count'] = result['signal_count'].fillna(0).astype(int)
        result['ma_cross'] = result['ma_cross'].fillna(False)

        # 7. Stock names
        result.insert(1, 'stock_name', result['stock_code'].map(
            lambda c: pc.stock_names.get(c, c)))

        return result

    def _check_exit_conditions_price(self, current_date: str) -> List[Trade]:
        """
        가격 기준 청산 확인 (목표 수익률, 손절, 시간 손절)

        당일 종가로 즉시 청산

        Args:
            current_date: 현재 거래일

        Returns:
            청산된 Trade 리스트
        """
        trades = []
        positions_to_check = list(self.portfolio.positions.items())

        for stock_code, position in positions_to_check:
            current_price = self.get_price(stock_code, current_date)
            if current_price is None:
                continue

            # 수익률 계산 (순수 가격 변화율)
            if position.direction == 'long':
                return_pct = (current_price / position.entry_price - 1)
            else:
                return_pct = (position.entry_price / current_price - 1)

            hold_days = position.hold_days(current_date)
            exit_reason = None

            # 1. 목표 수익률 달성 (당일 청산)
            if return_pct >= self.config.target_return:
                exit_reason = 'target'

            # 2. 가격 손절 (당일 청산)
            elif return_pct <= self.config.stop_loss:
                exit_reason = 'stop_loss'

            # 3. 시간 손절 (당일 청산)
            elif hold_days >= self.config.max_hold_days:
                exit_reason = 'time'

            # 청산 실행 (당일 종가)
            if exit_reason:
                trade = self.portfolio.exit_position(
                    stock_code, current_date, current_price, exit_reason
                )
                if trade:
                    trades.append(trade)

        return trades

    def _check_exit_conditions_reverse(self, signal_date: str, exit_date: str) -> List[Trade]:
        """
        반대 수급 청산 확인

        signal_date에 반대 수급 감지 → exit_date 종가로 청산
        (진입과 동일한 타이밍: 시그널 다음 날 청산)

        Args:
            signal_date: 반대 수급 시그널 스캔일
            exit_date: 청산일 (signal_date 다음 거래일)

        Returns:
            청산된 Trade 리스트
        """
        trades = []

        if self.config.reverse_signal_threshold <= 0:
            return trades  # 반대 수급 손절 비활성화

        positions_to_check = list(self.portfolio.positions.items())
        reverse_signals_cache = {}

        for stock_code, position in positions_to_check:
            # signal_date에 반대 수급 확인
            reverse_direction = 'short' if position.direction == 'long' else 'long'

            if reverse_direction not in reverse_signals_cache:
                reverse_signals_cache[reverse_direction] = self._scan_signals_on_date(
                    signal_date, direction=reverse_direction
                )

            reverse_signals = reverse_signals_cache[reverse_direction]

            if not reverse_signals.empty:
                stock_signal = reverse_signals[reverse_signals['stock_code'] == stock_code]

                if not stock_signal.empty:
                    reverse_pattern_score = stock_signal.iloc[0]['score']
                    reverse_signal_count = stock_signal.iloc[0]['signal_count']
                    reverse_final_score = reverse_pattern_score + (reverse_signal_count * 5)

                    # 반대 수급 조건 충족 시 exit_date 종가로 청산
                    if reverse_final_score >= self.config.reverse_signal_threshold:
                        exit_price = self.get_price(stock_code, exit_date)
                        if exit_price:
                            trade = self.portfolio.exit_position(
                                stock_code, exit_date, exit_price, 'reverse_signal'
                            )
                            if trade:
                                trades.append(trade)

        return trades

    def _select_entry_candidates(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        진입 후보 종목 선택

        Args:
            signals: Stage 1-3 결과

        Returns:
            진입 조건 충족 종목 (종합점수 내림차순)
        """
        if signals.empty:
            return signals

        # 종합점수 계산: 패턴점수 + (시그널 × 5점)
        signals = signals.copy()
        signals['final_score'] = signals['score'] + (signals['signal_count'] * 5)

        # 필터링: 종합점수 & 시그널 개수
        candidates = signals[
            (signals['final_score'] >= self.config.min_score) &
            (signals['signal_count'] >= self.config.min_signals)
        ].copy()

        # 패턴 필터링
        if self.config.allowed_patterns:
            candidates = candidates[
                candidates['pattern'].isin(self.config.allowed_patterns)
            ]

        # 종합점수 내림차순 정렬
        candidates = candidates.sort_values('final_score', ascending=False)

        return candidates

    def _execute_entries(self, candidates: pd.DataFrame, entry_date: str) -> List[Position]:
        """
        진입 실행 (다음 날 시가)

        Args:
            candidates: 진입 후보 종목 (direction 포함)
            entry_date: 진입일 (시그널 발생 다음 날)

        Returns:
            생성된 Position 리스트
        """
        positions = []

        for _, row in candidates.iterrows():
            # 포지션이 꽉 찼으면 중단
            if self.portfolio.is_full:
                break

            stock_code = row['stock_code']
            stock_name = row['stock_name']
            direction = row.get('direction', 'long')  # 기본값 'long'

            # 이미 보유 중이면 skip
            if self.portfolio.has_position(stock_code):
                continue

            # 진입 가격: 당일 종가 (시가 데이터 없음)
            entry_price = self.get_price(stock_code, entry_date)
            if entry_price is None or entry_price <= 0:
                continue  # 가격 없으면 skip

            # 진입 실행 (direction 전달, 종합점수 사용)
            position = self.portfolio.enter_position(
                stock_code=stock_code,
                stock_name=stock_name,
                entry_date=entry_date,
                entry_price=entry_price,
                pattern=row['pattern'],
                score=row['final_score'],  # 종합점수 사용
                signal_count=row['signal_count'],
                direction=direction
            )

            if position:
                positions.append(position)

        return positions

    def run(self, start_date: str, end_date: str, verbose: bool = True,
            preload_data: bool = True) -> Dict:
        """
        백테스트 실행

        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            verbose: 진행 상황 출력 여부
            preload_data: True이면 시작 전 전체 Sff 데이터를 메모리 로드 (기본: True)
                         False이면 매 계산마다 DB 조회 (메모리 절약 필요 시)

        Returns:
            {
                'trades': List[Trade],
                'daily_values': pd.DataFrame,
                'portfolio': Portfolio,
                'config': BacktestConfig
            }
        """
        if preload_data:
            if verbose:
                print("데이터 프리로드 중...")
            pc = BacktestPrecomputer(self.conn, self.config.institution_weight)
            self._precomputed = pc.precompute(end_date, verbose=verbose)

        if verbose:
            print(f"\n{'='*80}")
            print(f"📈 백테스트 시작: {start_date} ~ {end_date}")
            print(f"{'='*80}\n")
            print(f"초기 자본금: {self.config.initial_capital:,.0f}원")
            print(f"진입 조건: 점수 {self.config.min_score}점 이상, 시그널 {self.config.min_signals}개 이상")
            print(f"청산 조건: 목표 +{self.config.target_return*100:.0f}%, 손절 {self.config.stop_loss*100:.0f}%, 시간 {self.config.max_hold_days}일")
            print(f"\n시뮬레이션 시작...\n")

        # 거래일 목록
        trading_dates = self.get_trading_dates(start_date, end_date)

        if not trading_dates:
            raise ValueError(f"거래일이 없습니다: {start_date} ~ {end_date}")

        # 롤링 윈도우 시뮬레이션
        for i, trade_date in enumerate(trading_dates):
            # 1-1. 가격 기준 청산 (목표가/손절가/시간 손절) - 당일 종가
            self._check_exit_conditions_price(trade_date)

            # 1-2. 반대 수급 청산 - 전날 시그널 감지 → 오늘 종가 청산
            if i > 0:
                prev_date = trading_dates[i - 1]
                self._check_exit_conditions_reverse(prev_date, trade_date)

            # 2. Stage 1-3 실행 (미래 데이터 차단!)
            # strategy별로 direction 설정
            directions_to_scan = []
            if self.config.strategy == 'long':
                directions_to_scan = ['long']
            elif self.config.strategy == 'short':
                directions_to_scan = ['short']
            else:  # 'both'
                directions_to_scan = ['long', 'short']

            all_candidates = pd.DataFrame()
            for direction in directions_to_scan:
                signals = self._scan_signals_on_date(trade_date, direction=direction)

                # 3. 진입 후보 선택
                if not signals.empty:
                    candidates = self._select_entry_candidates(signals)
                    all_candidates = pd.concat([all_candidates, candidates], ignore_index=True)

            # 4. 다음 날 진입 (시가)
            if not all_candidates.empty and i + 1 < len(trading_dates):
                next_date = trading_dates[i + 1]
                self._execute_entries(all_candidates, next_date)

            # 5. 일별 포트폴리오 가치 기록
            current_prices = {}
            for stock_code in self.portfolio.positions.keys():
                price = self.get_price(stock_code, trade_date)
                if price:
                    current_prices[stock_code] = price

            portfolio_value = self.portfolio.get_portfolio_value(current_prices)

            self.daily_values.append({
                'date': trade_date,
                'value': portfolio_value,
                'cash': self.portfolio.cash,
                'position_count': self.portfolio.position_count,
                'total_trades': len(self.portfolio.trades),
            })

            # 진행 상황 출력 (10일마다)
            if verbose and (i + 1) % 10 == 0:
                total_return = (portfolio_value / self.config.initial_capital - 1) * 100
                print(f"[{trade_date}] 포트폴리오: {portfolio_value:,.0f}원 ({total_return:+.1f}%) | "
                      f"포지션: {self.portfolio.position_count}/{self.config.max_positions} | "
                      f"거래: {len(self.portfolio.trades)}건")

        # 6. 마지막 날 모든 포지션 청산 (옵션)
        if self.config.force_exit_on_end:
            last_date = trading_dates[-1]
            for stock_code in list(self.portfolio.positions.keys()):
                exit_price = self.get_price(stock_code, last_date)
                if exit_price:
                    self.portfolio.exit_position(stock_code, last_date, exit_price, 'end')

        # 결과 반환
        daily_df = pd.DataFrame(self.daily_values)

        if verbose:
            final_value = daily_df.iloc[-1]['value'] if not daily_df.empty else self.config.initial_capital
            total_return = (final_value / self.config.initial_capital - 1) * 100
            print(f"\n{'='*80}")
            print(f"✅ 백테스트 완료!")
            print(f"{'='*80}\n")
            print(f"최종 자본금: {final_value:,.0f}원")
            print(f"총 수익률: {total_return:+.2f}%")
            print(f"총 거래 횟수: {len(self.portfolio.trades)}건\n")

        if preload_data:
            self._precomputed = None  # 메모리 해제

        return {
            'trades': self.portfolio.trades,
            'daily_values': daily_df,
            'portfolio': self.portfolio,
            'config': self.config,
        }
