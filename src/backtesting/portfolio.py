"""
포트폴리오 관리 모듈

Trade, Position, Portfolio 클래스 구현
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd


@dataclass
class Trade:
    """완료된 거래 기록"""
    stock_code: str
    stock_name: str
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    shares: int
    pattern: str
    score: float
    signal_count: int
    return_pct: float
    hold_days: int
    exit_reason: str  # 'target', 'stop_loss', 'time'
    costs: float  # 거래 비용 (세금 + 수수료 + 슬리피지 + 차입비용)
    direction: str = 'long'  # 'long' 또는 'short'

    @property
    def entry_value(self) -> float:
        """진입 시 투자 금액 (long) 또는 매도 대금 (short)"""
        return self.entry_price * self.shares

    @property
    def exit_value(self) -> float:
        """청산 시 회수 금액 (long) 또는 매수 대금 (short)"""
        return self.exit_price * self.shares

    @property
    def profit(self) -> float:
        """
        순이익 (거래 비용 차감)

        - Long: (exit_price - entry_price) * shares - costs
        - Short: (entry_price - exit_price) * shares - costs
        """
        if self.direction == 'long':
            return self.exit_value - self.entry_value - self.costs
        else:  # short
            return self.entry_value - self.exit_value - self.costs

    def to_dict(self) -> dict:
        """딕셔너리로 변환 (CSV 저장용)"""
        return {
            'stock_code': self.stock_code,
            'stock_name': self.stock_name,
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'exit_date': self.exit_date,
            'exit_price': self.exit_price,
            'shares': self.shares,
            'pattern': self.pattern,
            'score': self.score,
            'signal_count': self.signal_count,
            'return_pct': self.return_pct,
            'hold_days': self.hold_days,
            'exit_reason': self.exit_reason,
            'costs': self.costs,
            'profit': self.profit,
            'direction': self.direction,
        }


@dataclass
class Position:
    """현재 보유 중인 포지션"""
    stock_code: str
    stock_name: str
    entry_date: str
    entry_price: float
    shares: int
    pattern: str
    score: float
    signal_count: int
    entry_costs: float  # 진입 시 거래 비용
    direction: str = 'long'  # 'long' 또는 'short'

    @property
    def entry_value(self) -> float:
        """진입 시 투자 금액 (long) 또는 매도 대금 (short)"""
        return self.entry_price * self.shares

    def current_value(self, current_price: float) -> float:
        """현재 평가 금액"""
        return current_price * self.shares

    def unrealized_return(self, current_price: float) -> float:
        """
        미실현 수익률 (%)

        - Long: (current_price - entry_price) / entry_price * 100
        - Short: (entry_price - current_price) / entry_price * 100
        """
        if self.direction == 'long':
            return (current_price / self.entry_price - 1) * 100
        else:  # short
            return (1 - current_price / self.entry_price) * 100

    def hold_days(self, current_date: str) -> int:
        """보유 일수"""
        entry = datetime.strptime(self.entry_date, '%Y-%m-%d')
        current = datetime.strptime(current_date, '%Y-%m-%d')
        return (current - entry).days

    def to_trade(self, exit_date: str, exit_price: float, exit_reason: str,
                 exit_costs: float) -> Trade:
        """청산하여 Trade 객체 생성"""
        # 수익률 계산 (방향별)
        if self.direction == 'long':
            return_pct = (exit_price / self.entry_price - 1) * 100
        else:  # short
            return_pct = (1 - exit_price / self.entry_price) * 100

        hold_days_count = self.hold_days(exit_date)
        total_costs = self.entry_costs + exit_costs

        return Trade(
            stock_code=self.stock_code,
            stock_name=self.stock_name,
            entry_date=self.entry_date,
            entry_price=self.entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            shares=self.shares,
            pattern=self.pattern,
            score=self.score,
            signal_count=self.signal_count,
            return_pct=return_pct,
            hold_days=hold_days_count,
            exit_reason=exit_reason,
            costs=total_costs,
            direction=self.direction,
        )


class Portfolio:
    """포트폴리오 관리 클래스"""

    def __init__(self, initial_capital: float, max_positions: int = 10,
                 tax_rate: float = 0.0020, commission_rate: float = 0.00015,
                 slippage_rate: float = 0.001, borrowing_rate: float = 0.03):
        """
        초기화

        Args:
            initial_capital: 초기 자본금
            max_positions: 최대 동시 보유 종목 수
            tax_rate: 증권거래세 (매도 시, 기본 0.20%)
            commission_rate: 수수료 (매수/매도, 기본 0.015%)
            slippage_rate: 슬리피지 (매수/매도, 기본 0.1%)
            borrowing_rate: 공매도 차입비용 (연환산, 기본 3.0%)
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.tax_rate = tax_rate
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.borrowing_rate = borrowing_rate

        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}  # stock_code -> Position
        self.trades: List[Trade] = []

    @property
    def position_count(self) -> int:
        """현재 보유 포지션 수"""
        return len(self.positions)

    @property
    def is_full(self) -> bool:
        """포지션이 꽉 찼는지 확인"""
        return self.position_count >= self.max_positions

    def has_position(self, stock_code: str) -> bool:
        """특정 종목을 보유 중인지 확인"""
        return stock_code in self.positions

    def calculate_borrowing_fee(self, entry_price: float, shares: int, hold_days: int) -> float:
        """
        공매도 차입 비용 계산

        Args:
            entry_price: 진입 가격
            shares: 주식 수
            hold_days: 보유 기간 (일)

        Returns:
            차입 비용 (원)

        Formula:
            fee = entry_value × borrowing_rate × (hold_days / 365)
        """
        entry_value = entry_price * shares
        return entry_value * self.borrowing_rate * (hold_days / 365)

    def calculate_entry_costs(self, price: float, shares: int, direction: str = 'long') -> float:
        """
        진입 시 거래 비용 계산

        Args:
            price: 진입 가격
            shares: 주식 수
            direction: 'long' 또는 'short'

        Returns:
            거래 비용 (원)

        - Long 매수: 수수료 + 슬리피지
        - Short 매도: 세금 + 수수료 + 슬리피지 (공매도도 증권거래세 부과)
        """
        value = price * shares
        commission = value * self.commission_rate
        slippage = value * self.slippage_rate

        if direction == 'long':
            # Long 매수: 세금 없음
            return commission + slippage
        else:
            # Short 매도: 세금 포함
            tax = value * self.tax_rate
            return tax + commission + slippage

    def calculate_exit_costs(self, price: float, shares: int, direction: str = 'long',
                            hold_days: int = 0, entry_price: float = 0) -> float:
        """
        청산 시 거래 비용 계산

        Args:
            price: 청산 가격
            shares: 주식 수
            direction: 'long' 또는 'short'
            hold_days: 보유 기간 (short 차입비용 계산용)
            entry_price: 진입 가격 (short 차입비용 계산용)

        Returns:
            거래 비용 (원)

        - Long 매도: 세금 + 수수료 + 슬리피지
        - Short 매수: 수수료 + 슬리피지 + 차입비용 (세금 없음)
        """
        value = price * shares
        commission = value * self.commission_rate
        slippage = value * self.slippage_rate

        if direction == 'long':
            # Long 매도: 세금 포함
            tax = value * self.tax_rate
            return tax + commission + slippage
        else:
            # Short 매수(환매): 차입비용 포함 (세금 없음)
            borrowing_fee = self.calculate_borrowing_fee(entry_price, shares, hold_days)
            return commission + slippage + borrowing_fee

    def calculate_position_size(self, price: float) -> int:
        """
        포지션 크기 계산 (동일 가중)

        각 종목당 자본금 / max_positions 투자
        """
        available_per_position = self.initial_capital / self.max_positions
        shares = int(available_per_position / price)
        return shares

    def enter_position(self, stock_code: str, stock_name: str, entry_date: str,
                       entry_price: float, pattern: str, score: float,
                       signal_count: int, direction: str = 'long') -> Optional[Position]:
        """
        포지션 진입

        Args:
            direction: 'long' (매수) 또는 'short' (공매도)

        Returns:
            Position 객체 (진입 성공 시) 또는 None (실패 시)

        Note:
            - Long: 현금 차감 (매수)
            - Short: 담보금 개념으로 현금 차감 (실제로는 매도하여 현금 증가하나,
                     백테스팅에서는 동일 가중으로 처리)
        """
        # 이미 보유 중이거나 포지션이 꽉 찼으면 진입 불가
        if self.has_position(stock_code) or self.is_full:
            return None

        # 포지션 크기 계산
        shares = self.calculate_position_size(entry_price)
        if shares <= 0:
            return None

        # 거래 비용 계산
        entry_costs = self.calculate_entry_costs(entry_price, shares, direction)
        total_cost = entry_price * shares + entry_costs

        # 현금 부족 확인
        if total_cost > self.cash:
            return None

        # 현금 차감 (long: 매수, short: 담보금)
        self.cash -= total_cost

        # 포지션 생성
        position = Position(
            stock_code=stock_code,
            stock_name=stock_name,
            entry_date=entry_date,
            entry_price=entry_price,
            shares=shares,
            pattern=pattern,
            score=score,
            signal_count=signal_count,
            entry_costs=entry_costs,
            direction=direction,
        )

        self.positions[stock_code] = position
        return position

    def exit_position(self, stock_code: str, exit_date: str, exit_price: float,
                      exit_reason: str) -> Optional[Trade]:
        """
        포지션 청산

        Returns:
            Trade 객체 (청산 성공 시) 또는 None (실패 시)

        Note:
            - Long 청산: 매도 → 현금 증가
            - Short 청산: 매수 → 현금 회수 (손익 반영)
        """
        if not self.has_position(stock_code):
            return None

        position = self.positions[stock_code]

        # 보유 기간 계산 (차입비용용)
        hold_days_count = position.hold_days(exit_date)

        # 거래 비용 계산 (방향별)
        exit_costs = self.calculate_exit_costs(
            exit_price, position.shares, position.direction,
            hold_days_count, position.entry_price
        )

        # 현금 회수 계산
        if position.direction == 'long':
            # Long 청산: 매도 대금 - 비용
            proceeds = exit_price * position.shares - exit_costs
        else:
            # Short 청산: 담보금 회수 + 손익 - 청산 비용
            # 손익 = (entry_price - exit_price) * shares
            # 담보금 = entry_price * shares (진입 시 차감했던 금액)
            entry_value = position.entry_price * position.shares
            profit = (position.entry_price - exit_price) * position.shares
            proceeds = entry_value + profit - exit_costs

        self.cash += proceeds

        # Trade 생성
        trade = position.to_trade(exit_date, exit_price, exit_reason, exit_costs)
        self.trades.append(trade)

        # 포지션 제거
        del self.positions[stock_code]

        return trade

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        현재 포트폴리오 총 가치 계산

        Args:
            current_prices: {stock_code: current_price}

        Returns:
            현금 + 보유 포지션 평가액
        """
        total_value = self.cash

        for stock_code, position in self.positions.items():
            if stock_code in current_prices:
                total_value += position.current_value(current_prices[stock_code])

        return total_value

    def get_statistics(self) -> dict:
        """포트폴리오 통계"""
        return {
            'cash': self.cash,
            'position_count': self.position_count,
            'max_positions': self.max_positions,
            'total_trades': len(self.trades),
        }
