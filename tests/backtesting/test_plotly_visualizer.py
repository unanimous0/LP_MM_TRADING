"""
PlotlyVisualizer 모듈 테스트 (Option 2)

테스트 항목:
1. 초기화 - daily_values 변환 및 return_pct 계산
2. fig_equity_curve - go.Figure 반환 + 거래 마커 포함
3. fig_drawdown - go.Figure 반환
4. fig_monthly_returns - go.Figure 반환 (히트맵)
5. fig_return_distribution - go.Figure 반환 / 거래 없을 시 None
6. fig_pattern_performance - go.Figure 반환 / 거래 없을 시 None
7. create_dashboard - HTML 문자열 반환 + 필수 내용 포함
8. create_dashboard HTML 파일 저장 확인
9. 거래 없을 때 예외 없이 동작
10. _build_kpi_html - KPI 요약 포함 확인
"""

import pytest
import tempfile
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.backtesting.plotly_visualizer import PlotlyVisualizer
from src.backtesting.portfolio import Trade


# ============================================================================
# 픽스처
# ============================================================================

def _make_trade(stock_code='005930', stock_name='삼성전자',
                entry_date='2024-01-02', exit_date='2024-01-15',
                entry_price=70000, exit_price=73500,
                return_pct=5.0, pattern='급등형',
                signal_count=2, direction='long',
                exit_reason='target'):
    return Trade(
        stock_code=stock_code,
        stock_name=stock_name,
        entry_date=entry_date,
        entry_price=entry_price,
        exit_date=exit_date,
        exit_price=exit_price,
        shares=10,
        pattern=pattern,
        score=75.0,
        signal_count=signal_count,
        return_pct=return_pct,
        hold_days=(
            pd.Timestamp(exit_date) - pd.Timestamp(entry_date)
        ).days,
        exit_reason=exit_reason,
        costs=entry_price * 10 * 0.0043,
        direction=direction,
    )


def _make_daily_values(start='2024-01-02', end='2024-01-31',
                       initial=10_000_000):
    """테스트용 일별 포트폴리오 가치 DataFrame"""
    dates = pd.date_range(start=start, end=end, freq='B')  # 거래일만
    n = len(dates)
    # 간단한 랜덤 워크
    np.random.seed(42)
    values = initial * np.cumprod(1 + np.random.normal(0.001, 0.01, n))
    return pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'value': values,
        'cash': values * 0.3,
        'position_count': np.random.randint(0, 5, n),
        'total_trades': np.arange(n),
    })


@pytest.fixture
def sample_trades():
    return [
        _make_trade(return_pct=5.0, pattern='급등형', exit_date='2024-01-15'),
        _make_trade(stock_code='000660', stock_name='SK하이닉스',
                    return_pct=-3.0, pattern='지속형',
                    entry_date='2024-01-05', exit_date='2024-01-20',
                    entry_price=150000, exit_price=145500, exit_reason='stop_loss'),
        _make_trade(stock_code='035420', stock_name='NAVER',
                    return_pct=8.0, pattern='전환형',
                    entry_date='2024-01-08', exit_date='2024-01-25',
                    entry_price=200000, exit_price=216000),
    ]


@pytest.fixture
def sample_daily_values():
    return _make_daily_values()


@pytest.fixture
def visualizer(sample_trades, sample_daily_values):
    return PlotlyVisualizer(
        trades=sample_trades,
        daily_values=sample_daily_values,
        initial_capital=10_000_000,
    )


@pytest.fixture
def empty_visualizer():
    """거래 없는 PlotlyVisualizer"""
    return PlotlyVisualizer(
        trades=[],
        daily_values=_make_daily_values(),
        initial_capital=10_000_000,
    )


# ============================================================================
# 테스트 클래스
# ============================================================================

class TestPlotlyVisualizerInit:
    """초기화 테스트"""

    def test_init_stores_trades(self, sample_trades, sample_daily_values):
        """거래 내역이 올바르게 저장됨"""
        pv = PlotlyVisualizer(sample_trades, sample_daily_values, 10_000_000)
        assert len(pv.trades) == 3

    def test_init_converts_date_to_datetime(self, visualizer):
        """date 컬럼이 datetime으로 변환됨"""
        assert pd.api.types.is_datetime64_any_dtype(visualizer.daily_values['date'])

    def test_init_calculates_return_pct(self, visualizer):
        """return_pct 컬럼이 계산됨"""
        assert 'return_pct' in visualizer.daily_values.columns
        # 초기 자본 대비 수익률이므로 NaN 없어야 함
        assert not visualizer.daily_values['return_pct'].isna().all()

    def test_init_empty_daily_values(self, sample_trades):
        """빈 daily_values도 예외 없이 초기화됨"""
        pv = PlotlyVisualizer(sample_trades, pd.DataFrame(), 10_000_000)
        assert pv.daily_values.empty


class TestFigEquityCurve:
    """fig_equity_curve() 테스트"""

    def test_returns_go_figure(self, visualizer):
        """go.Figure 반환 확인"""
        fig = visualizer.fig_equity_curve()
        assert isinstance(fig, go.Figure)

    def test_has_equity_line_trace(self, visualizer):
        """수익률 곡선 트레이스 존재"""
        fig = visualizer.fig_equity_curve()
        names = [t.name for t in fig.data]
        assert '전략 수익률' in names

    def test_has_trade_markers_when_trades_exist(self, visualizer):
        """거래가 있을 때 진입/청산 마커 트레이스 포함"""
        fig = visualizer.fig_equity_curve()
        names = [t.name for t in fig.data]
        assert '진입' in names
        assert '청산' in names

    def test_no_trade_markers_when_empty(self, empty_visualizer):
        """거래가 없을 때 마커 트레이스 없음"""
        fig = empty_visualizer.fig_equity_curve()
        names = [t.name for t in fig.data]
        assert '진입' not in names
        assert '청산' not in names

    def test_empty_daily_values_returns_figure(self, sample_trades):
        """daily_values 없을 때도 go.Figure 반환"""
        pv = PlotlyVisualizer(sample_trades, pd.DataFrame(), 10_000_000)
        fig = pv.fig_equity_curve()
        assert isinstance(fig, go.Figure)


class TestFigDrawdown:
    """fig_drawdown() 테스트"""

    def test_returns_go_figure(self, visualizer):
        fig = visualizer.fig_drawdown()
        assert isinstance(fig, go.Figure)

    def test_has_drawdown_trace(self, visualizer):
        """낙폭 fill 트레이스 존재"""
        fig = visualizer.fig_drawdown()
        assert len(fig.data) >= 1

    def test_has_mdd_marker(self, visualizer):
        """최대 낙폭 마커 포함"""
        fig = visualizer.fig_drawdown()
        names = [t.name for t in fig.data]
        assert any('낙폭' in (n or '') for n in names)


class TestFigMonthlyReturns:
    """fig_monthly_returns() 테스트"""

    def test_returns_go_figure(self, visualizer):
        fig = visualizer.fig_monthly_returns()
        assert isinstance(fig, go.Figure)

    def test_bar_trace_type(self, visualizer):
        """월별 수익률 바차트 트레이스 타입 확인 (히트맵→바차트 재설계)"""
        fig = visualizer.fig_monthly_returns()
        assert any(isinstance(t, go.Bar) for t in fig.data)


class TestFigReturnDistribution:
    """fig_return_distribution() 테스트"""

    def test_returns_go_figure_with_trades(self, visualizer):
        fig = visualizer.fig_return_distribution()
        assert isinstance(fig, go.Figure)

    def test_returns_none_when_no_trades(self, empty_visualizer):
        """거래 없을 때 None 반환"""
        result = empty_visualizer.fig_return_distribution()
        assert result is None

    def test_has_histogram_traces(self, visualizer):
        """히스토그램 트레이스 존재"""
        fig = visualizer.fig_return_distribution()
        assert any(isinstance(t, go.Histogram) for t in fig.data)


class TestFigPatternPerformance:
    """fig_pattern_performance() 테스트"""

    def test_returns_go_figure_with_trades(self, visualizer):
        fig = visualizer.fig_pattern_performance()
        assert isinstance(fig, go.Figure)

    def test_returns_none_when_no_trades(self, empty_visualizer):
        """거래 없을 때 None 반환"""
        result = empty_visualizer.fig_pattern_performance()
        assert result is None

    def test_has_bar_traces(self, visualizer):
        """Bar 트레이스 3개 (평균수익률/승률/거래수)"""
        fig = visualizer.fig_pattern_performance()
        bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
        assert len(bar_traces) == 3


class TestCreateDashboard:
    """create_dashboard() 테스트"""

    def test_returns_html_string(self, visualizer):
        """HTML 문자열 반환"""
        html = visualizer.create_dashboard(show=False)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_html_contains_doctype(self, visualizer):
        """유효한 HTML 구조"""
        html = visualizer.create_dashboard(show=False)
        assert '<!DOCTYPE html>' in html

    def test_html_contains_chart_titles(self, visualizer):
        """차트 제목이 HTML에 포함됨"""
        html = visualizer.create_dashboard(show=False)
        assert '누적 수익률 곡선' in html
        assert '낙폭' in html
        assert '월별 수익률' in html

    def test_save_html_creates_file(self, visualizer):
        """save_html 경로에 파일 생성"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'report.html')
            visualizer.create_dashboard(save_html=path, show=False)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 1000  # 최소 1KB 이상

    def test_save_html_creates_parent_dir(self, visualizer):
        """중간 디렉토리가 없어도 자동 생성"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'sub', 'dir', 'report.html')
            visualizer.create_dashboard(save_html=path, show=False)
            assert os.path.exists(path)

    def test_empty_trades_no_exception(self, empty_visualizer):
        """거래 없을 때 예외 없이 HTML 생성"""
        html = empty_visualizer.create_dashboard(show=False)
        assert isinstance(html, str)
        assert '<!DOCTYPE html>' in html

    def test_kpi_section_included_when_trades_exist(self, visualizer):
        """거래가 있을 때 KPI 요약 섹션 포함"""
        html = visualizer.create_dashboard(show=False)
        assert '요약' in html
        assert '총 수익률' in html
        assert '승률' in html
