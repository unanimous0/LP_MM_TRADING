"""
BacktestVisualizer 테스트
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # GUI 없이 테스트

from src.backtesting.visualizer import BacktestVisualizer
from src.backtesting.portfolio import Trade


class TestBacktestVisualizer:
    """BacktestVisualizer 테스트"""

    @pytest.fixture
    def sample_trades(self):
        """샘플 거래 내역"""
        trades = [
            Trade(
                stock_code='005930',
                stock_name='삼성전자',
                entry_date='2024-01-02',
                entry_price=70000,
                exit_date='2024-01-10',
                exit_price=75000,
                shares=14,
                pattern='모멘텀형',
                score=85.0,
                signal_count=2,
                return_pct=7.14,
                hold_days=8,
                exit_reason='target',
                costs=2000,
                direction='long'
            ),
            Trade(
                stock_code='000660',
                stock_name='SK하이닉스',
                entry_date='2024-01-05',
                entry_price=120000,
                exit_date='2024-01-15',
                exit_price=115000,
                shares=8,
                pattern='지속형',
                score=75.0,
                signal_count=1,
                return_pct=-4.17,
                hold_days=10,
                exit_reason='stop_loss',
                costs=3000,
                direction='long'
            ),
            Trade(
                stock_code='035720',
                stock_name='카카오',
                entry_date='2024-01-10',
                entry_price=50000,
                exit_date='2024-01-25',
                exit_price=55000,
                shares=20,
                pattern='모멘텀형',
                score=90.0,
                signal_count=3,
                return_pct=10.0,
                hold_days=15,
                exit_reason='target',
                costs=1500,
                direction='long'
            ),
        ]
        return trades

    @pytest.fixture
    def sample_daily_values(self):
        """샘플 일별 포트폴리오 가치"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        values = 10_000_000 + np.cumsum(np.random.randn(30) * 50000)

        df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'value': values,
            'cash': values * 0.2,
            'position_count': [3] * 30,
            'total_trades': range(30),
        })
        return df

    @pytest.fixture
    def visualizer(self, sample_trades, sample_daily_values):
        """Visualizer 인스턴스"""
        return BacktestVisualizer(
            trades=sample_trades,
            daily_values=sample_daily_values,
            initial_capital=10_000_000
        )

    def test_visualizer_initialization(self, visualizer):
        """초기화 테스트"""
        assert visualizer.initial_capital == 10_000_000
        assert len(visualizer.trades) == 3
        assert 'return_pct' in visualizer.daily_values.columns
        # datetime64 타입 확인 (ns 또는 us)
        assert pd.api.types.is_datetime64_any_dtype(visualizer.daily_values['date'])

    def test_plot_equity_curve_no_save(self, visualizer):
        """수익률 곡선 생성 (저장 안 함)"""
        fig = visualizer.plot_equity_curve(show=False)
        assert fig is not None

        # Figure 닫기 (메모리 절약)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_equity_curve_save(self, visualizer, tmp_path):
        """수익률 곡선 저장"""
        save_path = tmp_path / 'equity_curve.png'
        visualizer.plot_equity_curve(save_path=str(save_path), show=False)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_plot_drawdown(self, visualizer, tmp_path):
        """낙폭 차트 생성"""
        save_path = tmp_path / 'drawdown.png'
        visualizer.plot_drawdown(save_path=str(save_path), show=False)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_plot_monthly_returns(self, visualizer, tmp_path):
        """월별 수익률 히트맵"""
        save_path = tmp_path / 'monthly_returns.png'
        visualizer.plot_monthly_returns(save_path=str(save_path), show=False)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_plot_return_distribution(self, visualizer, tmp_path):
        """수익률 분포 히스토그램"""
        save_path = tmp_path / 'return_distribution.png'
        visualizer.plot_return_distribution(save_path=str(save_path), show=False)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_plot_pattern_performance(self, visualizer, tmp_path):
        """패턴별 성과 바차트"""
        save_path = tmp_path / 'pattern_performance.png'
        visualizer.plot_pattern_performance(save_path=str(save_path), show=False)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_plot_all_png(self, visualizer, tmp_path):
        """모든 차트 PNG 저장"""
        visualizer.plot_all(save_dir=str(tmp_path), show=False)

        # 5개 파일 모두 생성 확인
        assert (tmp_path / 'equity_curve.png').exists()
        assert (tmp_path / 'drawdown.png').exists()
        assert (tmp_path / 'monthly_returns.png').exists()
        assert (tmp_path / 'return_distribution.png').exists()
        assert (tmp_path / 'pattern_performance.png').exists()

    def test_plot_all_pdf(self, visualizer, tmp_path):
        """PDF 리포트 저장"""
        pdf_path = tmp_path / 'report.pdf'
        visualizer.plot_all(save_pdf=str(pdf_path), show=False)

        assert pdf_path.exists()
        assert pdf_path.stat().st_size > 0

    def test_empty_trades(self, sample_daily_values):
        """거래 없는 경우"""
        visualizer = BacktestVisualizer(
            trades=[],
            daily_values=sample_daily_values,
            initial_capital=10_000_000
        )

        # 수익률 곡선, 낙폭, 월별 수익률은 거래 없어도 생성 가능
        fig1 = visualizer.plot_equity_curve(show=False)
        assert fig1 is not None

        fig2 = visualizer.plot_drawdown(show=False)
        assert fig2 is not None

        fig3 = visualizer.plot_monthly_returns(show=False)
        assert fig3 is not None

        # 수익률 분포, 패턴별 성과는 None 반환
        fig4 = visualizer.plot_return_distribution(show=False)
        assert fig4 is None

        fig5 = visualizer.plot_pattern_performance(show=False)
        assert fig5 is None

        # Figure 닫기
        import matplotlib.pyplot as plt
        if fig1: plt.close(fig1)
        if fig2: plt.close(fig2)
        if fig3: plt.close(fig3)

    def test_color_theme(self, visualizer):
        """색상 테마 확인"""
        assert visualizer.COLORS['long'] == '#2E86AB'
        assert visualizer.COLORS['short'] == '#A23B72'
        assert visualizer.COLORS['both'] == '#F18F01'
        assert visualizer.COLORS['profit'] == '#06A77D'
        assert visualizer.COLORS['loss'] == '#D62828'
