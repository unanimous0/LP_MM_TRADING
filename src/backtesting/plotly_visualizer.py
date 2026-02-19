"""
백테스트 결과 인터랙티브 시각화 모듈 (Option 2)

Plotly 기반 차트 + 단일 HTML 리포트 생성:
- 줌/팬/호버 인터랙션
- 거래 진입/청산 마커 오버레이
- 단일 HTML 파일 (self-contained 또는 CDN)
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from .portfolio import Trade


# HTML 리포트 헤더/푸터 템플릿
_HTML_HEADER = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>백테스트 결과 리포트</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0;
           background: #f0f2f5; color: #333; }}
    .header {{ background: #2E86AB; color: white;
               padding: 24px 32px; }}
    .header h1 {{ margin: 0; font-size: 22px; }}
    .header p  {{ margin: 6px 0 0 0; opacity: 0.85; font-size: 13px; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px; }}
    .card {{ background: white; margin: 16px 0; padding: 20px;
             border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    .card h2 {{ margin: 0 0 12px 0; font-size: 15px; color: #444;
                border-left: 4px solid #2E86AB; padding-left: 10px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
                     gap: 12px; margin-bottom: 8px; }}
    .kpi {{ background: #f8f9fa; border-radius: 6px; padding: 14px 16px; text-align: center; }}
    .kpi .label {{ font-size: 12px; color: #888; margin-bottom: 4px; }}
    .kpi .value {{ font-size: 20px; font-weight: bold; }}
    .positive {{ color: #06A77D; }}
    .negative {{ color: #D62828; }}
    .neutral  {{ color: #555; }}
  </style>
</head>
<body>
<div class="header">
  <h1>📊 백테스트 결과 리포트</h1>
  <p>생성 시각: {timestamp}</p>
</div>
<div class="container">
"""

_HTML_FOOTER = """\
</div>
</body>
</html>
"""


class PlotlyVisualizer:
    """Plotly 기반 백테스트 결과 인터랙티브 시각화 클래스"""

    # 색상 테마 (matplotlib 버전과 동일)
    COLORS = {
        'long':      '#2E86AB',
        'short':     '#A23B72',
        'both':      '#F18F01',
        'profit':    '#06A77D',
        'loss':      '#D62828',
        'benchmark': '#6C757D',
        'neutral':   '#AAAAAA',
    }

    def __init__(self, trades: List[Trade], daily_values: pd.DataFrame,
                 initial_capital: float):
        """
        초기화

        Args:
            trades: 거래 내역 리스트
            daily_values: 일별 포트폴리오 가치 (컬럼: date, value, ...)
            initial_capital: 초기 자본금
        """
        self.trades = trades
        self.daily_values = daily_values.copy()
        self.initial_capital = initial_capital

        # 날짜 변환 및 정렬
        if not self.daily_values.empty and 'date' in self.daily_values.columns:
            self.daily_values['date'] = pd.to_datetime(self.daily_values['date'])
            self.daily_values = self.daily_values.sort_values('date').reset_index(drop=True)
            # 누적 수익률 (%)
            self.daily_values['return_pct'] = (
                self.daily_values['value'] / self.initial_capital - 1
            ) * 100

        # 거래 날짜 → return_pct 룩업 딕셔너리 (거래 마커용)
        if not self.daily_values.empty and 'date' in self.daily_values.columns:
            self._date_to_return = dict(zip(
                self.daily_values['date'].dt.strftime('%Y-%m-%d'),
                self.daily_values['return_pct']
            ))
        else:
            self._date_to_return = {}

    # ------------------------------------------------------------------ #
    # 개별 차트 메서드                                                       #
    # ------------------------------------------------------------------ #

    def fig_equity_curve(self) -> go.Figure:
        """
        누적 수익률 곡선 + 거래 진입/청산 마커

        Returns:
            go.Figure: 인터랙티브 수익률 곡선
        """
        fig = go.Figure()

        if self.daily_values.empty:
            fig.update_layout(title='누적 수익률 곡선 (데이터 없음)')
            return fig

        # 수익률 곡선
        fig.add_trace(go.Scatter(
            x=self.daily_values['date'],
            y=self.daily_values['return_pct'],
            mode='lines',
            name='전략 수익률',
            line=dict(color=self.COLORS['both'], width=2),
            customdata=self.daily_values['value'],
            hovertemplate=(
                '%{x|%Y-%m-%d}<br>'
                '수익률: %{y:.2f}%<br>'
                '가치: %{customdata:,.0f}원'
                '<extra></extra>'
            ),
        ))

        # 0% 기준선
        fig.add_hline(y=0, line_dash='dash', line_color='rgba(0,0,0,0.4)', line_width=1)

        # 거래 마커 (진입 ▲ / 청산 ▼)
        if self.trades:
            entry_x, entry_y, entry_text = [], [], []
            exit_x, exit_y, exit_text = [], [], []

            for t in self.trades:
                entry_ret = self._date_to_return.get(t.entry_date)
                exit_ret = self._date_to_return.get(t.exit_date)

                if entry_ret is not None:
                    entry_x.append(t.entry_date)
                    entry_y.append(entry_ret)
                    entry_text.append(
                        f'{t.stock_name}({t.stock_code})<br>'
                        f'진입: {t.entry_price:,.0f}원<br>'
                        f'패턴: {t.pattern} | 시그널: {t.signal_count}개'
                    )

                if exit_ret is not None:
                    exit_x.append(t.exit_date)
                    exit_y.append(exit_ret)
                    color = self.COLORS['profit'] if t.return_pct > 0 else self.COLORS['loss']
                    exit_text.append(
                        f'{t.stock_name}({t.stock_code})<br>'
                        f'청산: {t.exit_price:,.0f}원<br>'
                        f'수익률: {t.return_pct:+.2f}% | 이유: {t.exit_reason}'
                    )

            if entry_x:
                fig.add_trace(go.Scatter(
                    x=entry_x, y=entry_y, mode='markers',
                    marker=dict(symbol='triangle-up', size=9,
                                color=self.COLORS['long'], opacity=0.8,
                                line=dict(color='white', width=1)),
                    name='진입',
                    text=entry_text,
                    hovertemplate='%{text}<extra></extra>',
                ))

            if exit_x:
                fig.add_trace(go.Scatter(
                    x=exit_x, y=exit_y, mode='markers',
                    marker=dict(symbol='triangle-down', size=9,
                                color=self.COLORS['loss'], opacity=0.8,
                                line=dict(color='white', width=1)),
                    name='청산',
                    text=exit_text,
                    hovertemplate='%{text}<extra></extra>',
                ))

        fig.update_layout(
            title='누적 수익률 곡선',
            xaxis_title='날짜',
            yaxis_title='누적 수익률 (%)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=450,
        )
        fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

        return fig

    def fig_drawdown(self) -> go.Figure:
        """
        낙폭(Drawdown) 추이

        Returns:
            go.Figure: 낙폭 fill-area 차트
        """
        fig = go.Figure()

        if self.daily_values.empty:
            fig.update_layout(title='낙폭 추이 (데이터 없음)')
            return fig

        # 낙폭 계산
        values = self.daily_values['value'].values
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max * 100

        # 낙폭 fill
        fig.add_trace(go.Scatter(
            x=self.daily_values['date'],
            y=drawdown,
            fill='tozeroy',
            fillcolor=f'rgba(214,40,40,0.2)',
            mode='lines',
            name='낙폭',
            line=dict(color=self.COLORS['loss'], width=1.5),
            hovertemplate='%{x|%Y-%m-%d}<br>낙폭: %{y:.2f}%<extra></extra>',
        ))

        # 최대 낙폭 마커
        max_dd_idx = int(np.argmin(drawdown))
        max_dd_value = float(drawdown[max_dd_idx])
        max_dd_date = self.daily_values['date'].iloc[max_dd_idx]

        fig.add_trace(go.Scatter(
            x=[max_dd_date],
            y=[max_dd_value],
            mode='markers+text',
            marker=dict(symbol='star', size=14, color=self.COLORS['loss']),
            text=[f'MDD: {max_dd_value:.2f}%'],
            textposition='top right',
            name=f'최대 낙폭',
            hovertemplate=f'최대 낙폭: {max_dd_value:.2f}%<extra></extra>',
            showlegend=True,
        ))

        # 0% 기준선
        fig.add_hline(y=0, line_dash='dash', line_color='rgba(0,0,0,0.4)', line_width=1)

        fig.update_layout(
            title='포트폴리오 낙폭(Drawdown) 추이',
            xaxis_title='날짜',
            yaxis_title='낙폭 (%)',
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=380,
        )
        fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

        return fig

    def fig_monthly_returns(self) -> go.Figure:
        """
        월별 수익률 히트맵

        Returns:
            go.Figure: 연×월 히트맵
        """
        if self.daily_values.empty:
            fig = go.Figure()
            fig.update_layout(title='월별 수익률 (데이터 없음)')
            return fig

        df = self.daily_values.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        monthly = df.groupby(['year', 'month'])['value'].last().reset_index()
        monthly['monthly_return'] = monthly['value'].pct_change() * 100

        pivot = monthly.pivot(index='year', columns='month', values='monthly_return')

        MONTH_NAMES = ['1월', '2월', '3월', '4월', '5월', '6월',
                       '7월', '8월', '9월', '10월', '11월', '12월']
        x_labels = [MONTH_NAMES[int(m) - 1] for m in pivot.columns]
        y_labels = [str(y) for y in pivot.index]

        z_values = pivot.values.tolist()
        text_values = [
            [f'{v:.2f}%' if not np.isnan(v) else '' for v in row]
            for row in pivot.values
        ]

        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=x_labels,
            y=y_labels,
            colorscale='RdYlGn',
            zmid=0,
            text=text_values,
            texttemplate='%{text}',
            hovertemplate='%{y}년 %{x}<br>수익률: %{z:.2f}%<extra></extra>',
            colorbar=dict(title='수익률 (%)'),
        ))

        fig.update_layout(
            title='월별 수익률 히트맵',
            xaxis_title='월',
            yaxis_title='연도',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=max(280, len(pivot) * 60 + 120),
        )

        return fig

    def fig_return_distribution(self) -> Optional[go.Figure]:
        """
        거래별 수익률 분포 히스토그램

        Returns:
            go.Figure 또는 None (거래 없을 시)
        """
        if not self.trades:
            return None

        returns = [t.return_pct for t in self.trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        fig = go.Figure()

        # 손실 히스토그램 (빨강)
        if losses:
            fig.add_trace(go.Histogram(
                x=losses,
                name=f'패배 ({len(losses)}건)',
                marker_color=self.COLORS['loss'],
                opacity=0.75,
                nbinsx=20,
                hovertemplate='수익률: %{x:.2f}%<br>건수: %{y}<extra></extra>',
            ))

        # 수익 히스토그램 (녹색)
        if wins:
            fig.add_trace(go.Histogram(
                x=wins,
                name=f'승리 ({len(wins)}건)',
                marker_color=self.COLORS['profit'],
                opacity=0.75,
                nbinsx=20,
                hovertemplate='수익률: %{x:.2f}%<br>건수: %{y}<extra></extra>',
            ))

        # 평균/중앙값 수직선
        mean_r = float(np.mean(returns))
        median_r = float(np.median(returns))

        fig.add_vline(x=0, line_color='black', line_width=1, opacity=0.5)
        fig.add_vline(
            x=mean_r, line_dash='dash', line_color='navy', line_width=1.5,
            annotation_text=f'평균 {mean_r:+.2f}%',
            annotation_position='top right',
        )
        fig.add_vline(
            x=median_r, line_dash='dot', line_color='darkgreen', line_width=1.5,
            annotation_text=f'중앙값 {median_r:+.2f}%',
            annotation_position='top left',
        )

        fig.update_layout(
            title='거래별 수익률 분포',
            xaxis_title='수익률 (%)',
            yaxis_title='거래 횟수',
            barmode='overlay',
            hovermode='x',
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=380,
        )
        fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')

        return fig

    def fig_pattern_performance(self) -> Optional[go.Figure]:
        """
        패턴별 성과 (평균 수익률 / 승률 / 거래 횟수)

        Returns:
            go.Figure 또는 None (거래 없을 시)
        """
        if not self.trades:
            return None

        df = pd.DataFrame([t.to_dict() for t in self.trades])
        stats = df.groupby('pattern').agg(
            count=('return_pct', 'count'),
            avg_return=('return_pct', 'mean'),
            win_rate=('return_pct', lambda x: (x > 0).mean() * 100),
        ).reset_index().sort_values('avg_return', ascending=True)

        # 3개 서브플롯 (평균수익률 / 승률 / 거래 수)
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['평균 수익률 (%)', '승률 (%)', '거래 횟수'],
            shared_yaxes=True,
        )

        # 색상 (평균 수익률 기준)
        bar_colors = [
            self.COLORS['profit'] if v >= 0 else self.COLORS['loss']
            for v in stats['avg_return']
        ]

        fig.add_trace(go.Bar(
            y=stats['pattern'], x=stats['avg_return'],
            orientation='h', name='평균 수익률',
            marker_color=bar_colors,
            hovertemplate='%{y}<br>평균 수익률: %{x:.2f}%<extra></extra>',
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            y=stats['pattern'], x=stats['win_rate'],
            orientation='h', name='승률',
            marker_color=self.COLORS['both'],
            hovertemplate='%{y}<br>승률: %{x:.1f}%<extra></extra>',
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            y=stats['pattern'], x=stats['count'],
            orientation='h', name='거래 수',
            marker_color=self.COLORS['neutral'],
            hovertemplate='%{y}<br>거래: %{x}건<extra></extra>',
        ), row=1, col=3)

        # 50% 기준선 (승률 패널)
        fig.add_vline(x=50, line_dash='dash', line_color='rgba(0,0,0,0.3)',
                      line_width=1, row=1, col=2)
        # 0% 기준선 (평균 수익률 패널)
        fig.add_vline(x=0, line_color='rgba(0,0,0,0.3)',
                      line_width=1, row=1, col=1)

        fig.update_layout(
            title='패턴별 성과',
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=max(300, len(stats) * 80 + 100),
        )
        fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')

        return fig

    # ------------------------------------------------------------------ #
    # 대시보드 생성                                                          #
    # ------------------------------------------------------------------ #

    def create_dashboard(self, save_html: Optional[str] = None,
                         show: bool = True,
                         cdn: bool = False) -> str:
        """
        모든 차트를 하나의 HTML 파일로 결합

        Args:
            save_html: 저장 경로 (None이면 저장 안 함)
            show: 브라우저에서 즉시 열기 (기본: True)
            cdn: True이면 CDN에서 Plotly.js 로드 (인터넷 필요, 파일 경량),
                 False이면 Plotly.js를 HTML에 내장 (오프라인 가능, ~3MB)

        Returns:
            str: 생성된 HTML 문자열
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 요약 KPI 섹션
        kpi_html = self._build_kpi_html()

        # 차트 목록 (타이틀, figure_method)
        chart_defs = [
            ('1. 누적 수익률 곡선', self.fig_equity_curve),
            ('2. 낙폭(Drawdown) 추이', self.fig_drawdown),
            ('3. 월별 수익률 히트맵', self.fig_monthly_returns),
            ('4. 거래별 수익률 분포', self.fig_return_distribution),
            ('5. 패턴별 성과', self.fig_pattern_performance),
        ]

        html_parts = [_HTML_HEADER.format(timestamp=timestamp), kpi_html]
        plotlyjs_included = False

        for title, method in chart_defs:
            fig = method()
            if fig is None:
                continue

            if cdn:
                include_js = 'cdn' if not plotlyjs_included else False
            else:
                include_js = (not plotlyjs_included)  # True for first, False for rest

            div_html = pio.to_html(
                fig,
                include_plotlyjs=include_js,
                full_html=False,
            )
            plotlyjs_included = True
            html_parts.append(
                f'<div class="card"><h2>{title}</h2>{div_html}</div>\n'
            )

        html_parts.append(_HTML_FOOTER)
        full_html = ''.join(html_parts)

        if save_html:
            Path(save_html).parent.mkdir(parents=True, exist_ok=True)
            Path(save_html).write_text(full_html, encoding='utf-8')
            file_size_kb = Path(save_html).stat().st_size // 1024
            print(f"✅ HTML 리포트 저장: {save_html} ({file_size_kb}KB)")

        if show:
            import tempfile
            import webbrowser
            with tempfile.NamedTemporaryFile(
                mode='w', suffix='.html', delete=False, encoding='utf-8'
            ) as f:
                f.write(full_html)
                webbrowser.open(f'file://{f.name}')

        return full_html

    # ------------------------------------------------------------------ #
    # 내부 헬퍼                                                             #
    # ------------------------------------------------------------------ #

    def _build_kpi_html(self) -> str:
        """요약 KPI 카드 HTML 생성"""
        if not self.trades:
            return ''

        returns = [t.return_pct for t in self.trades]
        wins = [r for r in returns if r > 0]
        total_return = 0.0
        if not self.daily_values.empty:
            total_return = float(
                self.daily_values['value'].iloc[-1] / self.initial_capital - 1
            ) * 100

        win_rate = len(wins) / len(returns) * 100 if returns else 0
        avg_return = float(np.mean(returns)) if returns else 0

        # MDD
        if not self.daily_values.empty:
            v = self.daily_values['value'].values
            mdd = float((v - np.maximum.accumulate(v)).min() / np.maximum.accumulate(v).max() * 100)
        else:
            mdd = 0.0

        def _color(v):
            return 'positive' if v > 0 else ('negative' if v < 0 else 'neutral')

        kpis = [
            ('총 수익률', f'{total_return:+.2f}%', _color(total_return)),
            ('승률', f'{win_rate:.1f}%', 'neutral'),
            ('총 거래', f'{len(self.trades)}건', 'neutral'),
            ('평균 수익률', f'{avg_return:+.2f}%', _color(avg_return)),
            ('최대 낙폭', f'{mdd:.2f}%', _color(mdd)),
        ]

        items = '\n'.join(
            f'<div class="kpi">'
            f'<div class="label">{label}</div>'
            f'<div class="value {css}">{value}</div>'
            f'</div>'
            for label, value, css in kpis
        )
        return f'<div class="card"><h2>요약</h2><div class="summary-grid">{items}</div></div>\n'
