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


# ---------------------------------------------------------------------------
# 다크 테마 상수
# ---------------------------------------------------------------------------
_BG_PLOT  = '#0a0a0a'   # gray-950   (차트 내부)
_BG_PAPER = '#111111'   # gray-900   (차트 외곽)
_GRID     = '#1a1a1a'   # gray-800   (그리드 선)
_TEXT     = '#e2e8f0'   # slate-200  (일반 텍스트)
_MUTED    = '#94a3b8'   # slate-400  (보조 텍스트)


# HTML 리포트 헤더/푸터 템플릿
_HTML_HEADER = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>백테스트 결과 리포트</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0;
           background: #0a0a0a; color: #e2e8f0; }}
    .header {{ background: linear-gradient(135deg, #065f46 0%, #0a0a0a 100%);
               padding: 24px 32px; border-bottom: 1px solid #1a1a1a; }}
    .header h1 {{ margin: 0; font-size: 22px; color: #f1f5f9; }}
    .header p  {{ margin: 6px 0 0 0; opacity: 0.7; font-size: 13px; color: #94a3b8; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 16px; }}
    .card {{ background: #111111; margin: 16px 0; padding: 20px;
             border-radius: 12px; border: 1px solid #1a1a1a; }}
    .card h2 {{ margin: 0 0 12px 0; font-size: 15px; color: #e2e8f0;
                border-left: 4px solid #4ade80; padding-left: 10px; }}
    .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
                     gap: 12px; margin-bottom: 8px; }}
    .kpi {{ background: #0a0a0a; border-radius: 8px; padding: 14px 16px;
            text-align: center; border: 1px solid #1a1a1a; }}
    .kpi .label {{ font-size: 12px; color: #64748b; margin-bottom: 4px; }}
    .kpi .value {{ font-size: 20px; font-weight: bold; }}
    .positive {{ color: #4ade80; }}
    .negative {{ color: #f87171; }}
    .neutral  {{ color: #94a3b8; }}
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

    COLORS = {
        'long':      '#4ade80',  # green-400
        'short':     '#f472b6',  # pink-400
        'both':      '#fb923c',  # orange-400
        'profit':    '#4ade80',  # green-400
        'loss':      '#f87171',  # red-400
        'benchmark': '#94a3b8',  # slate-400
        'neutral':   '#64748b',  # slate-500
    }

    def __init__(self, trades: List[Trade], daily_values: pd.DataFrame,
                 initial_capital: float):
        self.trades = trades
        self.daily_values = daily_values.copy()
        self.initial_capital = initial_capital

        if not self.daily_values.empty and 'date' in self.daily_values.columns:
            self.daily_values['date'] = pd.to_datetime(self.daily_values['date'])
            self.daily_values = self.daily_values.sort_values('date').reset_index(drop=True)
            self.daily_values['return_pct'] = (
                self.daily_values['value'] / self.initial_capital - 1
            ) * 100

        if not self.daily_values.empty and 'date' in self.daily_values.columns:
            self._date_to_return = dict(zip(
                self.daily_values['date'].dt.strftime('%Y-%m-%d'),
                self.daily_values['return_pct']
            ))
        else:
            self._date_to_return = {}

    # ------------------------------------------------------------------ #
    # 다크 테마 공통 적용                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _apply_theme(fig: go.Figure) -> go.Figure:
        """모든 차트에 다크 테마 공통 스타일 적용"""
        fig.update_layout(
            template='plotly_dark',   # 다크 기본 템플릿 강제 적용
            plot_bgcolor=_BG_PLOT,
            paper_bgcolor=_BG_PAPER,
            font=dict(color=_TEXT, size=12),
            title_font=dict(color=_TEXT, size=15, family='Segoe UI, Arial, sans-serif'),
            legend=dict(
                bgcolor='rgba(10,10,10,0.85)',
                bordercolor=_GRID,
                borderwidth=1,
                font=dict(color=_TEXT),
            ),
        )
        fig.update_xaxes(
            showgrid=True, gridcolor=_GRID, gridwidth=1,
            linecolor=_GRID, linewidth=1,
            tickfont=dict(color=_MUTED),
            title_font=dict(color=_MUTED),
            zerolinecolor=_GRID, zerolinewidth=1,
        )
        fig.update_yaxes(
            showgrid=True, gridcolor=_GRID, gridwidth=1,
            linecolor=_GRID, linewidth=1,
            tickfont=dict(color=_MUTED),
            title_font=dict(color=_MUTED),
            zerolinecolor=_GRID, zerolinewidth=1,
        )
        return fig

    # ------------------------------------------------------------------ #
    # 개별 차트 메서드                                                       #
    # ------------------------------------------------------------------ #

    def fig_equity_curve(self) -> go.Figure:
        """누적 수익률 곡선 + 거래 진입/청산 마커"""
        fig = go.Figure()

        if self.daily_values.empty:
            fig.update_layout(title='누적 수익률 곡선 (데이터 없음)')
            return self._apply_theme(fig)

        fig.add_trace(go.Scatter(
            x=self.daily_values['date'],
            y=self.daily_values['return_pct'],
            mode='lines',
            name='전략 수익률',
            line=dict(color=self.COLORS['both'], width=2.5),
            fill='tozeroy',
            fillcolor='rgba(251,146,60,0.07)',
            customdata=self.daily_values['value'],
            hovertemplate=(
                '%{x|%Y-%m-%d}<br>'
                '수익률: %{y:.2f}%<br>'
                '가치: %{customdata:,.0f}원'
                '<extra></extra>'
            ),
        ))

        fig.add_hline(y=0, line_dash='dash', line_color='rgba(148,163,184,0.4)', line_width=1)

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
                    exit_text.append(
                        f'{t.stock_name}({t.stock_code})<br>'
                        f'청산: {t.exit_price:,.0f}원<br>'
                        f'수익률: {t.return_pct:+.2f}% | 이유: {t.exit_reason}'
                    )

            if entry_x:
                fig.add_trace(go.Scatter(
                    x=entry_x, y=entry_y, mode='markers',
                    marker=dict(symbol='triangle-up', size=9,
                                color=self.COLORS['long'], opacity=0.9,
                                line=dict(color=_BG_PAPER, width=1)),
                    name='진입',
                    text=entry_text,
                    hovertemplate='%{text}<extra></extra>',
                ))

            if exit_x:
                fig.add_trace(go.Scatter(
                    x=exit_x, y=exit_y, mode='markers',
                    marker=dict(symbol='triangle-down', size=9,
                                color=self.COLORS['loss'], opacity=0.9,
                                line=dict(color=_BG_PAPER, width=1)),
                    name='청산',
                    text=exit_text,
                    hovertemplate='%{text}<extra></extra>',
                ))

        fig.update_layout(
            title='누적 수익률 곡선',
            xaxis_title='날짜',
            yaxis_title='누적 수익률 (%)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=450,
        )
        return self._apply_theme(fig)

    def fig_drawdown(self) -> go.Figure:
        """낙폭(Drawdown) 추이"""
        fig = go.Figure()

        if self.daily_values.empty:
            fig.update_layout(title='낙폭 추이 (데이터 없음)')
            return self._apply_theme(fig)

        values = self.daily_values['value'].values
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max * 100

        fig.add_trace(go.Scatter(
            x=self.daily_values['date'],
            y=drawdown,
            fill='tozeroy',
            fillcolor='rgba(248,113,113,0.15)',
            mode='lines',
            name='낙폭',
            line=dict(color=self.COLORS['loss'], width=1.5),
            hovertemplate='%{x|%Y-%m-%d}<br>낙폭: %{y:.2f}%<extra></extra>',
        ))

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
            textfont=dict(color=self.COLORS['loss'], size=11),
            name='최대 낙폭',
            hovertemplate=f'최대 낙폭: {max_dd_value:.2f}%<extra></extra>',
            showlegend=True,
        ))

        fig.add_hline(y=0, line_dash='dash', line_color='rgba(148,163,184,0.4)', line_width=1)

        fig.update_layout(
            title='포트폴리오 낙폭(Drawdown) 추이',
            xaxis_title='날짜',
            yaxis_title='낙폭 (%)',
            hovermode='x unified',
            height=380,
        )
        return self._apply_theme(fig)

    def fig_monthly_returns(self) -> go.Figure:
        """월별 수익률 바차트 (x=기간 시간순, y=수익률)"""
        if self.daily_values.empty:
            fig = go.Figure()
            fig.update_layout(title='월별 수익률 (데이터 없음)')
            return self._apply_theme(fig)

        df = self.daily_values.copy()
        df['ym'] = df['date'].dt.year * 100 + df['date'].dt.month

        # 월별 마지막 값 기준, 시간순 정렬
        monthly = df.groupby('ym')['value'].last().reset_index()
        monthly = monthly.sort_values('ym').reset_index(drop=True)

        # 첫 달도 포함: initial_capital 대비 수익률 계산 (pct_change는 첫 달 NaN 제거해버림)
        prev_values = monthly['value'].shift(1)
        prev_values.iloc[0] = self.initial_capital
        monthly['monthly_return'] = (monthly['value'] / prev_values - 1) * 100

        if monthly.empty:
            fig = go.Figure()
            fig.update_layout(title='월별 수익률 (데이터 부족)')
            return self._apply_theme(fig)

        # x축 라벨: "2025년 6월" 형태
        monthly['label'] = monthly['ym'].apply(
            lambda ym: f"{ym // 100}년 {ym % 100}월"
        )

        colors = [
            self.COLORS['profit'] if r >= 0 else self.COLORS['loss']
            for r in monthly['monthly_return']
        ]

        hover_texts = [f'{r:+.2f}%' for r in monthly['monthly_return']]

        fig = go.Figure(data=go.Bar(
            x=monthly['label'],
            y=monthly['monthly_return'],
            customdata=hover_texts,
            marker_color=colors,
            marker_line=dict(color=_BG_PAPER, width=0.5),
            text=hover_texts,
            textposition='outside',
            textfont=dict(color=_TEXT, size=13),
            hovertemplate='%{x}<br>수익률: %{customdata}<extra></extra>',
        ))

        fig.add_hline(y=0, line_dash='dash', line_color='rgba(148,163,184,0.4)', line_width=1)

        # 텍스트 잘림 방지: y축 범위에 30% 여백 추가
        max_r = monthly['monthly_return'].max()
        min_r = monthly['monthly_return'].min()
        pad = max(abs(max_r), abs(min_r), 1.0) * 0.35
        fig.update_yaxes(range=[min(0, min_r) - pad, max(0, max_r) + pad])

        fig.update_layout(
            title='월별 수익률',
            xaxis_title='기간',
            yaxis_title='수익률 (%)',
            xaxis=dict(tickfont=dict(size=13)),
            height=420,
        )
        return self._apply_theme(fig)

    def fig_return_distribution(self) -> Optional[go.Figure]:
        """거래별 수익률 분포 히스토그램"""
        if not self.trades:
            return None

        returns = [t.return_pct for t in self.trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        fig = go.Figure()

        if losses:
            fig.add_trace(go.Histogram(
                x=losses,
                name=f'손실 ({len(losses)}건)',
                marker_color=self.COLORS['loss'],
                marker_line=dict(color=_BG_PAPER, width=0.5),
                opacity=0.85,
                nbinsx=20,
                hovertemplate='수익률: %{x:.2f}%<br>건수: %{y}<extra></extra>',
            ))

        if wins:
            fig.add_trace(go.Histogram(
                x=wins,
                name=f'수익 ({len(wins)}건)',
                marker_color=self.COLORS['profit'],
                marker_line=dict(color=_BG_PAPER, width=0.5),
                opacity=0.85,
                nbinsx=20,
                hovertemplate='수익률: %{x:.2f}%<br>건수: %{y}<extra></extra>',
            ))

        mean_r = float(np.mean(returns))
        median_r = float(np.median(returns))

        fig.add_vline(x=0, line_color='rgba(148,163,184,0.5)', line_width=1)
        # 선만 표시 (annotation은 차트 밖으로 분리)
        fig.add_vline(x=mean_r, line_dash='dash', line_color='#60a5fa', line_width=1.5)
        fig.add_vline(x=median_r, line_dash='dot', line_color='#4ade80', line_width=1.5)

        # 평균/중앙값 차트 밖에 별도 표시 (손실/수익 범례처럼)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='#60a5fa', width=2, dash='dash'),
            name=f'평균 {mean_r:+.2f}%',
        ))
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='#4ade80', width=2, dash='dot'),
            name=f'중앙값 {median_r:+.2f}%',
        ))

        fig.update_layout(
            title='거래별 수익률 분포',
            xaxis_title='수익률 (%)',
            yaxis_title='거래 횟수',
            barmode='overlay',
            hovermode='x',
            height=420,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='left',
                x=0,
            ),
            margin=dict(t=80),
        )
        return self._apply_theme(fig)

    def fig_pattern_performance(self) -> Optional[go.Figure]:
        """패턴별 성과 (평균 수익률 / 승률 / 거래 횟수)"""
        if not self.trades:
            return None

        df = pd.DataFrame([t.to_dict() for t in self.trades])
        stats = df.groupby('pattern').agg(
            count=('return_pct', 'count'),
            avg_return=('return_pct', 'mean'),
            win_rate=('return_pct', lambda x: (x > 0).mean() * 100),
        ).reset_index().sort_values('avg_return', ascending=True)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['평균 수익률 (%)', '승률 (%)', '거래 횟수'],
            shared_yaxes=True,
        )

        bar_colors = [
            self.COLORS['profit'] if v >= 0 else self.COLORS['loss']
            for v in stats['avg_return']
        ]

        fig.add_trace(go.Bar(
            y=stats['pattern'], x=stats['avg_return'],
            orientation='h', name='평균 수익률',
            marker_color=bar_colors,
            marker_line=dict(color=_BG_PAPER, width=0.5),
            hovertemplate='%{y}<br>평균 수익률: %{x:.2f}%<extra></extra>',
        ), row=1, col=1)

        fig.add_trace(go.Bar(
            y=stats['pattern'], x=stats['win_rate'],
            orientation='h', name='승률',
            marker_color=self.COLORS['both'],
            marker_line=dict(color=_BG_PAPER, width=0.5),
            hovertemplate='%{y}<br>승률: %{x:.1f}%<extra></extra>',
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            y=stats['pattern'], x=stats['count'],
            orientation='h', name='거래 수',
            marker_color=self.COLORS['neutral'],
            marker_line=dict(color=_BG_PAPER, width=0.5),
            hovertemplate='%{y}<br>거래: %{x}건<extra></extra>',
        ), row=1, col=3)

        fig.add_vline(x=50, line_dash='dash', line_color='rgba(148,163,184,0.4)',
                      line_width=1, row=1, col=2)
        fig.add_vline(x=0, line_color='rgba(148,163,184,0.3)',
                      line_width=1, row=1, col=1)

        # 서브플롯 타이틀 색상
        for ann in fig.layout.annotations:
            ann.font.color = _MUTED
            ann.font.size = 12

        fig.update_layout(
            title='패턴별 성과',
            showlegend=False,
            height=max(300, len(stats) * 80 + 100),
        )
        return self._apply_theme(fig)

    # ------------------------------------------------------------------ #
    # 대시보드 생성                                                          #
    # ------------------------------------------------------------------ #

    def create_dashboard(self, save_html: Optional[str] = None,
                         show: bool = True,
                         cdn: bool = False) -> str:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        kpi_html = self._build_kpi_html()

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
                include_js = (not plotlyjs_included)

            div_html = pio.to_html(fig, include_plotlyjs=include_js, full_html=False)
            plotlyjs_included = True
            html_parts.append(f'<div class="card"><h2>{title}</h2>{div_html}</div>\n')

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

        if not self.daily_values.empty:
            v = self.daily_values['value'].values
            _running_max = np.maximum.accumulate(v)
            mdd = float(((v - _running_max) / _running_max).min() * 100)
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
