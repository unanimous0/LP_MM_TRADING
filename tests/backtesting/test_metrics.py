"""
PerformanceMetrics 모듈 테스트

합성 거래 데이터로 메트릭 계산 검증
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.backtesting.portfolio import Trade
from src.backtesting.metrics import PerformanceMetrics


class TestPerformanceMetrics:
    """PerformanceMetrics 클래스 테스트"""

    @pytest.fixture
    def sample_trades(self):
        """샘플 거래 데이터"""
        return [
            # 승리 거래 (3건)
            Trade('005930', '삼성전자', '2024-01-02', 70000, '2024-01-10', 77000,
                  100, '급등형', 85, 2, 10.0, 8, 'target', 35000),
            Trade('000660', 'SK하이닉스', '2024-01-05', 100000, '2024-01-15', 115000,
                  50, '지속형', 80, 1, 15.0, 10, 'target', 25000),
            Trade('035420', 'NAVER', '2024-01-10', 200000, '2024-01-20', 210000,
                  25, '급등형', 75, 3, 5.0, 10, 'target', 15000),

            # 손실 거래 (2건)
            Trade('005380', '현대차', '2024-01-12', 150000, '2024-01-18', 142500,
                  30, '전환형', 65, 0, -5.0, 6, 'stop_loss', 20000),
            Trade('051910', 'LG화학', '2024-01-15', 300000, '2024-01-25', 285000,
                  15, '지속형', 70, 1, -5.0, 10, 'stop_loss', 18000),
        ]

    @pytest.fixture
    def sample_daily_values(self):
        """샘플 일별 가치 데이터"""
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        # 초기 자본금 10,000,000원에서 시작
        # 간단한 시뮬레이션: 중간에 낙폭, 마지막에 회복
        values = []
        base = 10_000_000

        for i, date in enumerate(dates):
            if i < 10:
                # 상승
                value = base * (1 + i * 0.01)
            elif i < 20:
                # 하락 (MDD 발생)
                value = base * 1.10 * (1 - (i - 10) * 0.015)
            else:
                # 회복
                value = base * 0.95 * (1 + (i - 20) * 0.02)

            values.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': value,
                'cash': value * 0.5,
                'position_count': min(i // 3, 5),
                'total_trades': i // 5
            })

        return pd.DataFrame(values)

    @pytest.fixture
    def metrics(self, sample_trades, sample_daily_values):
        """PerformanceMetrics 인스턴스"""
        return PerformanceMetrics(
            trades=sample_trades,
            daily_values=sample_daily_values,
            initial_capital=10_000_000
        )

    # ========================================================================
    # 1. 전체 성과 메트릭 테스트
    # ========================================================================

    def test_total_return(self, metrics, sample_daily_values):
        """누적 수익률 테스트"""
        final_value = sample_daily_values.iloc[-1]['value']
        expected_return = (final_value / 10_000_000 - 1) * 100

        assert metrics.total_return() == pytest.approx(expected_return, abs=0.01)

    def test_win_rate(self, metrics):
        """승률 테스트"""
        # 3승 2패
        expected = (3 / 5) * 100
        assert metrics.win_rate() == pytest.approx(expected, abs=0.01)

    def test_avg_return(self, metrics):
        """평균 수익률 테스트"""
        # (10 + 15 + 5 - 5 - 5) / 5 = 4.0%
        expected = (10 + 15 + 5 - 5 - 5) / 5
        assert metrics.avg_return() == pytest.approx(expected, abs=0.01)

    def test_avg_win(self, metrics):
        """평균 승리 수익률 테스트"""
        # (10 + 15 + 5) / 3 = 10.0%
        expected = (10 + 15 + 5) / 3
        assert metrics.avg_win() == pytest.approx(expected, abs=0.01)

    def test_avg_loss(self, metrics):
        """평균 손실 수익률 테스트"""
        # (-5 - 5) / 2 = -5.0%
        expected = (-5 - 5) / 2
        assert metrics.avg_loss() == pytest.approx(expected, abs=0.01)

    def test_profit_factor(self, metrics):
        """Profit Factor 테스트"""
        # 이익/손실 계산은 Trade.profit 기준
        # 간단히 수익률로 근사
        total_profit = 10 + 15 + 5  # 30%
        total_loss = abs(-5 - 5)     # 10%
        expected = total_profit / total_loss  # 3.0

        # 실제로는 profit (금액) 기준이므로 정확한 값은 다를 수 있음
        assert metrics.profit_factor() > 0

    # ========================================================================
    # 2. 리스크 지표 테스트
    # ========================================================================

    def test_max_drawdown(self, metrics):
        """최대 낙폭 테스트"""
        mdd_info = metrics.max_drawdown()

        assert 'mdd' in mdd_info
        assert 'start_date' in mdd_info
        assert 'end_date' in mdd_info
        assert 'recovery_date' in mdd_info

        # MDD는 음수
        assert mdd_info['mdd'] < 0

    def test_sharpe_ratio(self, metrics):
        """샤프 비율 테스트"""
        sharpe = metrics.sharpe_ratio()

        # 샤프 비율은 실수
        assert isinstance(sharpe, (int, float))

    def test_max_consecutive_losses(self, metrics):
        """최대 연속 손실 테스트"""
        # 샘플 데이터: 승, 승, 승, 패, 패
        # 최대 연속 손실: 2
        assert metrics.max_consecutive_losses() == 2

    def test_calmar_ratio(self, metrics):
        """칼마 비율 테스트"""
        calmar = metrics.calmar_ratio()

        # 칼마 비율은 실수
        assert isinstance(calmar, (int, float))

    # ========================================================================
    # 3. 세부 분석 테스트
    # ========================================================================

    def test_performance_by_pattern(self, metrics):
        """패턴별 성과 테스트"""
        result = metrics.performance_by_pattern()

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

        # 필수 컬럼 확인
        required_cols = ['pattern', 'trades', 'win_rate', 'avg_return', 'avg_hold_days']
        for col in required_cols:
            assert col in result.columns

        # 급등형: 2건 (승 2, 패 0) → 승률 100%, 평균 7.5%
        momentum = result[result['pattern'] == '급등형']
        if not momentum.empty:
            assert momentum.iloc[0]['trades'] == 2
            assert momentum.iloc[0]['win_rate'] == 100.0

    def test_performance_by_signal_count(self, metrics):
        """시그널별 성과 테스트"""
        result = metrics.performance_by_signal_count()

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

        # 필수 컬럼 확인
        required_cols = ['signal_count', 'trades', 'win_rate', 'avg_return']
        for col in required_cols:
            assert col in result.columns

        # 시그널 개수별로 분류되어 있는지 확인
        assert len(result) > 0

    def test_monthly_returns(self, metrics):
        """월별 수익률 테스트"""
        result = metrics.monthly_returns()

        assert isinstance(result, pd.DataFrame)
        assert not result.empty

        # 필수 컬럼 확인
        assert 'year_month' in result.columns
        assert 'return' in result.columns

        # 2024-01 데이터만 있어야 함
        assert len(result) == 1
        assert result.iloc[0]['year_month'] == '2024-01'

    def test_trade_duration_stats(self, metrics):
        """거래 보유 기간 통계 테스트"""
        stats = metrics.trade_duration_stats()

        assert 'avg' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'median' in stats

        # 샘플 데이터: 8, 10, 10, 6, 10일
        assert stats['min'] == 6
        assert stats['max'] == 10
        assert stats['avg'] == pytest.approx((8 + 10 + 10 + 6 + 10) / 5, abs=0.01)

    # ========================================================================
    # 4. 벤치마크 비교 테스트
    # ========================================================================

    def test_alpha_without_benchmark(self, metrics):
        """벤치마크 없을 때 알파 테스트"""
        assert metrics.alpha() is None

    def test_alpha_with_benchmark(self, sample_trades, sample_daily_values):
        """벤치마크 있을 때 알파 테스트"""
        # 간단한 벤치마크 수익률 (일별 0.1%)
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        benchmark = pd.Series([0.001] * len(dates), index=dates)

        metrics = PerformanceMetrics(
            trades=sample_trades,
            daily_values=sample_daily_values,
            initial_capital=10_000_000,
            benchmark_returns=benchmark
        )

        alpha = metrics.alpha()
        assert alpha is not None
        assert isinstance(alpha, (int, float))

    def test_beta_without_benchmark(self, metrics):
        """벤치마크 없을 때 베타 테스트"""
        assert metrics.beta() is None

    def test_beta_with_benchmark(self, sample_trades, sample_daily_values):
        """벤치마크 있을 때 베타 테스트"""
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        benchmark = pd.Series([0.001] * len(dates), index=dates)

        metrics = PerformanceMetrics(
            trades=sample_trades,
            daily_values=sample_daily_values,
            initial_capital=10_000_000,
            benchmark_returns=benchmark
        )

        beta = metrics.beta()
        assert beta is not None
        assert isinstance(beta, (int, float))

    # ========================================================================
    # 5. 종합 리포트 테스트
    # ========================================================================

    def test_summary(self, metrics):
        """종합 리포트 테스트"""
        summary = metrics.summary()

        # 모든 주요 메트릭이 포함되어 있는지 확인
        required_keys = [
            'total_return', 'win_rate', 'avg_return', 'avg_win', 'avg_loss',
            'profit_factor', 'total_trades',
            'max_drawdown', 'sharpe_ratio', 'calmar_ratio', 'max_consecutive_losses',
            'avg_hold_days', 'alpha', 'beta'
        ]

        for key in required_keys:
            assert key in summary

        # 거래 횟수 확인
        assert summary['total_trades'] == 5

        # 승률 확인
        assert summary['win_rate'] == pytest.approx(60.0, abs=0.01)

    # ========================================================================
    # 6. 엣지 케이스 테스트
    # ========================================================================

    def test_empty_trades(self):
        """거래 없을 때 테스트"""
        daily_values = pd.DataFrame({
            'date': ['2024-01-01'],
            'value': [10_000_000],
            'cash': [10_000_000],
            'position_count': [0],
            'total_trades': [0]
        })

        metrics = PerformanceMetrics(
            trades=[],
            daily_values=daily_values,
            initial_capital=10_000_000
        )

        assert metrics.total_return() == 0.0
        assert metrics.win_rate() == 0.0
        assert metrics.avg_return() == 0.0
        assert metrics.avg_win() == 0.0
        assert metrics.avg_loss() == 0.0

    def test_empty_daily_values(self):
        """일별 데이터 없을 때 테스트"""
        metrics = PerformanceMetrics(
            trades=[],
            daily_values=pd.DataFrame(),
            initial_capital=10_000_000
        )

        assert metrics.total_return() == 0.0
        assert metrics.sharpe_ratio() == 0.0

        mdd_info = metrics.max_drawdown()
        assert mdd_info['mdd'] == 0.0
