"""
Portfolio 모듈 테스트

Trade, Position, Portfolio 클래스 검증
"""

import pytest
from src.backtesting.portfolio import Trade, Position, Portfolio


class TestTrade:
    """Trade 데이터 클래스 테스트"""

    def test_trade_creation(self):
        """Trade 생성 테스트"""
        trade = Trade(
            stock_code='005930',
            stock_name='삼성전자',
            entry_date='2024-01-02',
            entry_price=70000,
            exit_date='2024-01-10',
            exit_price=77000,
            shares=100,
            pattern='모멘텀형',
            score=85.5,
            signal_count=2,
            return_pct=10.0,
            hold_days=8,
            exit_reason='target',
            costs=35000,
        )

        assert trade.stock_code == '005930'
        assert trade.entry_value == 7_000_000
        assert trade.exit_value == 7_700_000
        assert trade.profit == 7_700_000 - 7_000_000 - 35000  # 665,000원

    def test_trade_to_dict(self):
        """Trade.to_dict() 테스트"""
        trade = Trade(
            stock_code='005930',
            stock_name='삼성전자',
            entry_date='2024-01-02',
            entry_price=70000,
            exit_date='2024-01-10',
            exit_price=77000,
            shares=100,
            pattern='모멘텀형',
            score=85.5,
            signal_count=2,
            return_pct=10.0,
            hold_days=8,
            exit_reason='target',
            costs=35000,
        )

        d = trade.to_dict()
        assert d['stock_code'] == '005930'
        assert d['profit'] == 665000


class TestPosition:
    """Position 데이터 클래스 테스트"""

    def test_position_creation(self):
        """Position 생성 테스트"""
        position = Position(
            stock_code='005930',
            stock_name='삼성전자',
            entry_date='2024-01-02',
            entry_price=70000,
            shares=100,
            pattern='모멘텀형',
            score=85.5,
            signal_count=2,
            entry_costs=8000,
        )

        assert position.stock_code == '005930'
        assert position.entry_value == 7_000_000
        assert position.current_value(75000) == 7_500_000
        assert position.unrealized_return(77000) == pytest.approx(10.0, abs=0.01)

    def test_position_hold_days(self):
        """Position.hold_days() 테스트"""
        position = Position(
            stock_code='005930',
            stock_name='삼성전자',
            entry_date='2024-01-02',
            entry_price=70000,
            shares=100,
            pattern='모멘텀형',
            score=85.5,
            signal_count=2,
            entry_costs=8000,
        )

        assert position.hold_days('2024-01-02') == 0
        assert position.hold_days('2024-01-10') == 8
        assert position.hold_days('2024-02-01') == 30

    def test_position_to_trade(self):
        """Position.to_trade() 테스트"""
        position = Position(
            stock_code='005930',
            stock_name='삼성전자',
            entry_date='2024-01-02',
            entry_price=70000,
            shares=100,
            pattern='모멘텀형',
            score=85.5,
            signal_count=2,
            entry_costs=8000,
        )

        trade = position.to_trade(
            exit_date='2024-01-10',
            exit_price=77000,
            exit_reason='target',
            exit_costs=27000
        )

        assert isinstance(trade, Trade)
        assert trade.stock_code == '005930'
        assert trade.entry_price == 70000
        assert trade.exit_price == 77000
        assert trade.return_pct == pytest.approx(10.0, abs=0.01)
        assert trade.hold_days == 8
        assert trade.costs == 8000 + 27000  # entry + exit


class TestPortfolio:
    """Portfolio 클래스 테스트"""

    def test_portfolio_initialization(self):
        """Portfolio 초기화 테스트"""
        portfolio = Portfolio(initial_capital=10_000_000, max_positions=5)

        assert portfolio.initial_capital == 10_000_000
        assert portfolio.max_positions == 5
        assert portfolio.cash == 10_000_000
        assert portfolio.position_count == 0
        assert portfolio.is_full is False

    def test_calculate_entry_costs(self):
        """진입 거래 비용 계산 테스트"""
        portfolio = Portfolio(initial_capital=10_000_000)

        # 7,000,000원 투자
        costs = portfolio.calculate_entry_costs(price=70000, shares=100)

        # 수수료: 7,000,000 * 0.00015 = 1,050원
        # 슬리피지: 7,000,000 * 0.001 = 7,000원
        # 총: 8,050원
        expected = 7_000_000 * (0.00015 + 0.001)
        assert costs == pytest.approx(expected, abs=1)

    def test_calculate_exit_costs(self):
        """청산 거래 비용 계산 테스트"""
        portfolio = Portfolio(initial_capital=10_000_000)

        # 7,700,000원 청산 (Long)
        costs = portfolio.calculate_exit_costs(price=77000, shares=100, direction='long')

        # 세금: 7,700,000 * 0.0020 = 15,400원
        # 수수료: 7,700,000 * 0.00015 = 1,155원
        # 슬리피지: 7,700,000 * 0.001 = 7,700원
        # 총: 24,255원
        expected = 7_700_000 * (0.0020 + 0.00015 + 0.001)
        assert costs == pytest.approx(expected, abs=1)

    def test_calculate_position_size(self):
        """포지션 크기 계산 테스트 (동일 가중)"""
        portfolio = Portfolio(initial_capital=10_000_000, max_positions=10)

        # 각 포지션당 1,000,000원
        shares = portfolio.calculate_position_size(price=70000)

        # 1,000,000 / 70,000 = 14.28... → 14주
        assert shares == 14

    def test_enter_position_success(self):
        """포지션 진입 성공 테스트"""
        portfolio = Portfolio(initial_capital=10_000_000, max_positions=2)

        position = portfolio.enter_position(
            stock_code='005930',
            stock_name='삼성전자',
            entry_date='2024-01-02',
            entry_price=70000,
            pattern='모멘텀형',
            score=85.5,
            signal_count=2
        )

        assert position is not None
        assert portfolio.position_count == 1
        assert portfolio.has_position('005930')
        assert portfolio.cash < 10_000_000  # 현금 차감

    def test_enter_position_duplicate(self):
        """중복 진입 방지 테스트"""
        portfolio = Portfolio(initial_capital=10_000_000, max_positions=2)

        # 첫 번째 진입
        position1 = portfolio.enter_position(
            stock_code='005930',
            stock_name='삼성전자',
            entry_date='2024-01-02',
            entry_price=70000,
            pattern='모멘텀형',
            score=85.5,
            signal_count=2
        )
        assert position1 is not None

        # 중복 진입 시도 (실패)
        position2 = portfolio.enter_position(
            stock_code='005930',
            stock_name='삼성전자',
            entry_date='2024-01-03',
            entry_price=72000,
            pattern='모멘텀형',
            score=80.0,
            signal_count=1
        )
        assert position2 is None  # 진입 실패
        assert portfolio.position_count == 1  # 여전히 1개

    def test_enter_position_full(self):
        """포지션 한도 초과 방지 테스트"""
        portfolio = Portfolio(initial_capital=10_000_000, max_positions=2)

        # 2개 진입
        portfolio.enter_position('005930', '삼성전자', '2024-01-02', 70000, '모멘텀형', 85.5, 2)
        portfolio.enter_position('000660', 'SK하이닉스', '2024-01-02', 100000, '지속형', 80.0, 1)

        assert portfolio.is_full

        # 3번째 진입 시도 (실패)
        position3 = portfolio.enter_position('035420', 'NAVER', '2024-01-02', 200000, '전환형', 75.0, 0)
        assert position3 is None

    def test_exit_position_success(self):
        """포지션 청산 성공 테스트"""
        portfolio = Portfolio(initial_capital=10_000_000, max_positions=2)

        # 진입
        portfolio.enter_position('005930', '삼성전자', '2024-01-02', 70000, '모멘텀형', 85.5, 2)
        initial_cash = portfolio.cash

        # 청산
        trade = portfolio.exit_position('005930', '2024-01-10', 77000, 'target')

        assert trade is not None
        assert portfolio.position_count == 0
        assert not portfolio.has_position('005930')
        assert portfolio.cash > initial_cash  # 수익 발생
        assert len(portfolio.trades) == 1

    def test_exit_position_nonexistent(self):
        """없는 포지션 청산 방지 테스트"""
        portfolio = Portfolio(initial_capital=10_000_000)

        trade = portfolio.exit_position('005930', '2024-01-10', 77000, 'target')
        assert trade is None

    def test_get_portfolio_value(self):
        """포트폴리오 총 가치 계산 테스트"""
        portfolio = Portfolio(initial_capital=10_000_000, max_positions=2)

        # 2개 진입
        portfolio.enter_position('005930', '삼성전자', '2024-01-02', 70000, '모멘텀형', 85.5, 2)
        portfolio.enter_position('000660', 'SK하이닉스', '2024-01-02', 100000, '지속형', 80.0, 1)

        # 현재 가격
        current_prices = {
            '005930': 77000,  # +10%
            '000660': 110000,  # +10%
        }

        total_value = portfolio.get_portfolio_value(current_prices)

        # 현금 + 포지션 평가액
        assert total_value > portfolio.cash  # 수익 발생
        assert total_value < 11_000_000  # 거래 비용 차감

    def test_get_statistics(self):
        """포트폴리오 통계 테스트"""
        portfolio = Portfolio(initial_capital=10_000_000, max_positions=2)

        portfolio.enter_position('005930', '삼성전자', '2024-01-02', 70000, '모멘텀형', 85.5, 2)
        portfolio.exit_position('005930', '2024-01-10', 77000, 'target')

        stats = portfolio.get_statistics()

        assert stats['cash'] > 0
        assert stats['position_count'] == 0
        assert stats['max_positions'] == 2
        assert stats['total_trades'] == 1


class TestShortPosition:
    """Short Position 테스트 (Week 2.5)"""

    def test_short_position_creation(self):
        """Short Position 생성 테스트"""
        position = Position(
            stock_code='005930',
            stock_name='삼성전자',
            entry_date='2024-01-02',
            entry_price=80000,  # 고가에 공매도
            shares=100,
            pattern='모멘텀형',
            score=85.5,
            signal_count=2,
            entry_costs=10000,
            direction='short'
        )

        assert position.direction == 'short'
        assert position.entry_value == 8_000_000

    def test_short_position_unrealized_return(self):
        """Short Position 미실현 수익률 테스트"""
        position = Position(
            stock_code='005930',
            stock_name='삼성전자',
            entry_date='2024-01-02',
            entry_price=80000,  # 진입가
            shares=100,
            pattern='모멘텀형',
            score=85.5,
            signal_count=2,
            entry_costs=10000,
            direction='short'
        )

        # 주가 하락 → 수익
        unrealized_70k = position.unrealized_return(70000)  # -12.5% 하락
        assert unrealized_70k == pytest.approx(12.5, abs=0.1)  # +12.5% 수익

        # 주가 상승 → 손실
        unrealized_88k = position.unrealized_return(88000)  # +10% 상승
        assert unrealized_88k == pytest.approx(-10.0, abs=0.1)  # -10% 손실

    def test_short_position_to_trade(self):
        """Short Position → Trade 변환 테스트"""
        position = Position(
            stock_code='005930',
            stock_name='삼성전자',
            entry_date='2024-01-02',
            entry_price=80000,
            shares=100,
            pattern='모멘텀형',
            score=85.5,
            signal_count=2,
            entry_costs=10000,
            direction='short'
        )

        # 주가 하락하여 청산 (수익)
        trade = position.to_trade(
            exit_date='2024-01-10',
            exit_price=70000,
            exit_reason='target',
            exit_costs=8000
        )

        assert trade.direction == 'short'
        assert trade.entry_price == 80000
        assert trade.exit_price == 70000
        assert trade.return_pct == pytest.approx(12.5, abs=0.1)  # (1 - 70k/80k) * 100
        assert trade.costs == 10000 + 8000

    def test_short_trade_profit(self):
        """Short Trade 손익 계산 테스트"""
        trade = Trade(
            stock_code='005930',
            stock_name='삼성전자',
            entry_date='2024-01-02',
            entry_price=80000,
            exit_date='2024-01-10',
            exit_price=70000,
            shares=100,
            pattern='모멘텀형',
            score=85.5,
            signal_count=2,
            return_pct=12.5,
            hold_days=8,
            exit_reason='target',
            costs=18000,
            direction='short'
        )

        # Profit = (entry - exit) * shares - costs
        # = (80000 - 70000) * 100 - 18000 = 1,000,000 - 18,000 = 982,000
        expected_profit = (80000 - 70000) * 100 - 18000
        assert trade.profit == expected_profit

    def test_portfolio_borrowing_fee_calculation(self):
        """차입 비용 계산 테스트"""
        portfolio = Portfolio(initial_capital=10_000_000)

        # 8,000,000원 규모, 30일 보유
        fee = portfolio.calculate_borrowing_fee(
            entry_price=80000,
            shares=100,
            hold_days=30
        )

        # Fee = 8,000,000 * 0.03 * (30/365) = 약 19,726원
        expected = 8_000_000 * 0.03 * (30 / 365)
        assert fee == pytest.approx(expected, abs=1)

    def test_portfolio_enter_short_position(self):
        """Short Position 진입 테스트"""
        portfolio = Portfolio(initial_capital=10_000_000, max_positions=2)
        initial_cash = portfolio.cash

        position = portfolio.enter_position(
            stock_code='005930',
            stock_name='삼성전자',
            entry_date='2024-01-02',
            entry_price=80000,
            pattern='모멘텀형',
            score=85.5,
            signal_count=2,
            direction='short'
        )

        assert position is not None
        assert position.direction == 'short'
        assert portfolio.position_count == 1
        assert portfolio.cash < initial_cash  # 담보금 차감

    def test_portfolio_exit_short_position_profit(self):
        """Short Position 청산 테스트 (수익)"""
        portfolio = Portfolio(initial_capital=10_000_000, max_positions=2)

        # 진입 (80,000원에 공매도)
        portfolio.enter_position('005930', '삼성전자', '2024-01-02', 80000,
                                '모멘텀형', 85.5, 2, direction='short')
        cash_after_entry = portfolio.cash

        # 청산 (70,000원에 매수 → 수익)
        trade = portfolio.exit_position('005930', '2024-01-10', 70000, 'target')

        assert trade is not None
        assert trade.direction == 'short'
        assert portfolio.position_count == 0
        assert portfolio.cash > cash_after_entry  # 수익 발생

    def test_portfolio_exit_short_position_loss(self):
        """Short Position 청산 테스트 (손실)"""
        portfolio = Portfolio(initial_capital=10_000_000, max_positions=2)

        # 진입 (80,000원에 공매도)
        portfolio.enter_position('005930', '삼성전자', '2024-01-02', 80000,
                                '모멘텀형', 85.5, 2, direction='short')
        cash_after_entry = portfolio.cash

        # 청산 (90,000원에 매수 → 손실)
        trade = portfolio.exit_position('005930', '2024-01-10', 90000, 'stop_loss')

        assert trade is not None
        assert trade.direction == 'short'
        assert trade.return_pct < 0  # 손실
        assert portfolio.cash < cash_after_entry  # 손실 발생

    def test_short_exit_costs_include_borrowing_fee(self):
        """Short 청산 비용에 차입 비용 포함 확인"""
        portfolio = Portfolio(initial_capital=10_000_000)

        # 80,000원, 100주, 30일 보유
        exit_costs = portfolio.calculate_exit_costs(
            price=70000,
            shares=100,
            direction='short',
            hold_days=30,
            entry_price=80000
        )

        # 비용 = 수수료 + 슬리피지 + 차입비용
        # 차입비용 = 8,000,000 * 0.03 * (30/365)
        borrowing = 8_000_000 * 0.03 * (30 / 365)
        commission = 70000 * 100 * 0.00015
        slippage = 70000 * 100 * 0.001

        expected = commission + slippage + borrowing
        assert exit_costs == pytest.approx(expected, abs=1)

    def test_short_entry_costs_include_tax(self):
        """Short 진입 비용에 세금 포함 확인"""
        portfolio = Portfolio(initial_capital=10_000_000)

        # 80,000원, 100주 공매도 진입
        entry_costs = portfolio.calculate_entry_costs(
            price=80000,
            shares=100,
            direction='short'
        )

        # 비용 = 세금 + 수수료 + 슬리피지
        value = 80000 * 100
        tax = value * 0.0020
        commission = value * 0.00015
        slippage = value * 0.001

        expected = tax + commission + slippage
        assert entry_costs == pytest.approx(expected, abs=1)
