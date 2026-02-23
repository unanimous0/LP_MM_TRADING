"""
Plotly 차트 헬퍼 모듈

Z-Score 히트맵, 패턴 파이차트, 점수 히스토그램 등
Streamlit 대시보드 전용 차트 생성.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# 다크 테마 상수
# ---------------------------------------------------------------------------
_BG_PLOT  = '#0f172a'   # slate-900
_BG_PAPER = '#1e293b'   # slate-800
_GRID     = '#334155'   # slate-700
_TEXT     = '#e2e8f0'   # slate-200
_MUTED    = '#94a3b8'   # slate-400


def _fmt_amount(val):
    """원 단위 → 억/조 포맷 (부호, 쉼표 포함)"""
    eok = val / 1e8
    sign = '+' if eok >= 0 else '-'
    abs_eok = abs(eok)
    if abs_eok >= 10000:
        jo = abs_eok / 10000
        if jo == int(jo):
            return f'{sign}{int(jo):,}조'
        return f'{sign}{jo:,.1f}조'
    return f'{sign}{int(abs_eok):,}억'


def _apply_dark(fig: go.Figure) -> go.Figure:
    """모든 차트에 다크 테마 공통 스타일 적용"""
    fig.update_layout(
        template='plotly_dark',   # 다크 기본 템플릿 강제 적용
        plot_bgcolor=_BG_PLOT,
        paper_bgcolor=_BG_PAPER,
        font=dict(color=_TEXT, size=12),
        title_font=dict(color=_TEXT, size=14),
        legend=dict(
            bgcolor='rgba(15,23,42,0.85)',
            bordercolor=_GRID,
            borderwidth=1,
            font=dict(color=_TEXT),
        ),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=_GRID, gridwidth=1,
        linecolor=_GRID, tickfont=dict(color=_MUTED),
        title_font=dict(color=_MUTED),
        zerolinecolor=_GRID,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=_GRID, gridwidth=1,
        linecolor=_GRID, tickfont=dict(color=_MUTED),
        title_font=dict(color=_MUTED),
        zerolinecolor=_GRID,
    )
    return fig


# ---------------------------------------------------------------------------
# Z-Score 히트맵 (Stage 2 시각화를 Plotly로 재구현)
# ---------------------------------------------------------------------------

def create_zscore_heatmap(
    zscore_matrix: pd.DataFrame,
    sort_by: str = 'recent',
    top_n: int = 50,
    stock_names: pd.DataFrame = None,
) -> go.Figure:
    """멀티 기간 Z-Score 인터랙티브 히트맵"""
    if zscore_matrix.empty:
        fig = go.Figure()
        fig.update_layout(title='Z-Score 히트맵 (데이터 없음)')
        return _apply_dark(fig)

    df = zscore_matrix.copy()
    period_cols = [c for c in df.columns if c not in ('stock_code',)]

    if sort_by == 'recent':
        sort_key = df['1W'] if '1W' in df.columns else df[period_cols[0]]
    elif sort_by == 'momentum':
        first, last = period_cols[0], period_cols[-1]
        sort_key = df[first] - df[last]
    elif sort_by == 'weighted':
        weights = list(range(len(period_cols), 0, -1))
        total_w = sum(weights)
        sort_key = sum(df[c] * w for c, w in zip(period_cols, weights)) / total_w
    else:
        sort_key = df[period_cols].mean(axis=1)

    df['_sort'] = sort_key
    df = df.sort_values('_sort', ascending=False).head(top_n).drop(columns=['_sort'])

    if stock_names is not None and 'stock_code' in stock_names.columns:
        name_map = dict(zip(stock_names['stock_code'], stock_names['stock_name']))
        y_labels = [f"{name_map.get(code, code)}({code})" for code in df['stock_code']]
    else:
        y_labels = df['stock_code'].tolist()

    z_values = df[period_cols].values
    text_values = [[f'{v:.2f}' for v in row] for row in z_values]

    # 다크 배경용 커스텀 컬러스케일
    colorscale = [
        [0.0,  '#ef4444'],   # 강한 매도 - red-500
        [0.35, '#fca5a5'],   # 약한 매도 - red-300
        [0.5,  '#334155'],   # 중립 - slate-700
        [0.65, '#86efac'],   # 약한 매수 - green-300
        [1.0,  '#22c55e'],   # 강한 매수 - green-500
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=period_cols,
        y=y_labels,
        colorscale=colorscale,
        zmid=0,
        zmin=-3,
        zmax=3,
        text=text_values,
        texttemplate='%{text}',
        textfont=dict(color='#0f172a', size=10),
        hovertemplate='%{y}<br>기간: %{x}<br>Z-Score: %{z:.2f}<extra></extra>',
        colorbar=dict(
            title=dict(text='Z-Score', font=dict(color=_TEXT)),
            tickfont=dict(color=_MUTED),
            bgcolor=_BG_PAPER,
            bordercolor=_GRID,
            borderwidth=1,
        ),
    ))

    fig.update_layout(
        title=f'수급 Z-Score 히트맵 (정렬: {sort_by}, 상위 {len(df)}개)',
        xaxis_title='기간',
        yaxis_title='종목',
        yaxis=dict(autorange='reversed'),
        height=max(400, len(df) * 22 + 100),
    )
    return _apply_dark(fig)


# ---------------------------------------------------------------------------
# 패턴 분석 차트
# ---------------------------------------------------------------------------

def create_pattern_pie_chart(report_df: pd.DataFrame) -> go.Figure:
    """패턴별 분포 파이차트"""
    if report_df.empty or 'pattern' not in report_df.columns:
        fig = go.Figure()
        fig.update_layout(title='패턴 분포 (데이터 없음)')
        return _apply_dark(fig)

    counts = report_df['pattern'].value_counts().reset_index()
    counts.columns = ['pattern', 'count']

    color_map = {
        '모멘텀형': '#f472b6',   # pink-400
        '지속형':   '#38bdf8',   # sky-400
        '전환형':   '#4ade80',   # green-400
        '기타':     '#64748b',   # slate-500
    }
    colors = [color_map.get(p, '#64748b') for p in counts['pattern']]

    fig = go.Figure(data=go.Pie(
        labels=counts['pattern'],
        values=counts['count'],
        marker=dict(colors=colors, line=dict(color=_BG_PAPER, width=2)),
        textinfo='label+value+percent',
        textfont=dict(color=_TEXT, size=12),
        hovertemplate='%{label}<br>종목 수: %{value}<br>비율: %{percent}<extra></extra>',
    ))

    fig.update_layout(
        title='패턴별 종목 분포',
        paper_bgcolor=_BG_PAPER,
        font=dict(color=_TEXT),
        title_font=dict(color=_TEXT),
        legend=dict(font=dict(color=_TEXT), bgcolor='rgba(15,23,42,0.85)'),
        height=350,
    )
    return fig


def create_score_histogram(report_df: pd.DataFrame) -> go.Figure:
    """패턴 점수 분포 히스토그램"""
    if report_df.empty or 'score' not in report_df.columns:
        fig = go.Figure()
        fig.update_layout(title='점수 분포 (데이터 없음)')
        return _apply_dark(fig)

    fig = go.Figure(data=go.Histogram(
        x=report_df['score'],
        nbinsx=20,
        marker_color='#38bdf8',
        marker_line=dict(color=_BG_PAPER, width=0.5),
        opacity=0.85,
        hovertemplate='점수: %{x:.0f}<br>종목 수: %{y}<extra></extra>',
    ))

    fig.add_vline(
        x=70, line_dash='dash', line_color='#fb923c', line_width=1.5,
        annotation_text='관심 기준 (70점)',
        annotation_font=dict(color='#fb923c', size=11),
        annotation_position='top right',
    )

    fig.update_layout(
        title='패턴 점수 분포',
        xaxis_title='점수',
        yaxis_title='종목 수',
        height=350,
    )
    return _apply_dark(fig)


def create_signal_distribution_chart(report_df: pd.DataFrame) -> go.Figure:
    """시그널 수 분포 바차트"""
    if report_df.empty or 'signal_count' not in report_df.columns:
        fig = go.Figure()
        fig.update_layout(title='시그널 분포 (데이터 없음)')
        return _apply_dark(fig)

    counts = report_df['signal_count'].value_counts().sort_index().reset_index()
    counts.columns = ['signal_count', 'count']

    colors = ['#64748b' if s < 2 else '#4ade80' for s in counts['signal_count']]

    fig = go.Figure(data=go.Bar(
        x=counts['signal_count'].astype(str) + '개',
        y=counts['count'],
        marker_color=colors,
        marker_line=dict(color=_BG_PAPER, width=0.5),
        hovertemplate='시그널 %{x}<br>종목 수: %{y}<extra></extra>',
    ))

    fig.update_layout(
        title='시그널 수별 종목 분포',
        xaxis_title='시그널 수',
        yaxis_title='종목 수',
        height=350,
    )
    return _apply_dark(fig)


# ---------------------------------------------------------------------------
# 이상 수급 바차트
# ---------------------------------------------------------------------------

def create_abnormal_supply_chart(
    df: pd.DataFrame,
    direction: str = 'buy',
) -> go.Figure:
    """이상 수급 종목 수평 바차트 (Z-Score 기준)

    Args:
        df: get_abnormal_supply() 결과 DataFrame
        direction: 'buy' 또는 'sell'
    """
    if df.empty:
        fig = go.Figure()
        label = '매수' if direction == 'buy' else '매도'
        fig.update_layout(title=f'이상 {label} 수급 (데이터 없음)', height=300)
        return _apply_dark(fig)

    plot_df = df.copy()
    # 매도는 절대값으로 변환 (바 길이 비교를 위해)
    if direction == 'sell':
        plot_df['abs_zscore'] = plot_df['combined_zscore'].abs()
        sort_col = 'abs_zscore'
    else:
        sort_col = 'combined_zscore'

    plot_df = plot_df.sort_values(sort_col, ascending=True)  # 수평 바는 아래→위

    y_labels = [
        f"{row['stock_name']}" for _, row in plot_df.iterrows()
    ]

    bar_color = '#4ade80' if direction == 'buy' else '#f87171'  # green-400 / red-400
    title = '강한 매수 수급 (Z > 2σ)' if direction == 'buy' else '강한 매도 수급 (Z < -2σ)'

    has_amounts = 'foreign_net_amount' in plot_df.columns

    customdata_cols = [
        plot_df['foreign_zscore'].values,
        plot_df['institution_zscore'].values,
        plot_df['combined_zscore'].values,
        plot_df['sector'].fillna('-').values,
    ]
    hover_lines = [
        '<b>%{y}</b><br>',
        '섹터: %{customdata[3]}<br>',
        '외국인 Z: %{customdata[0]:.2f}<br>',
        '기관 Z: %{customdata[1]:.2f}<br>',
        '종합 Z: %{customdata[2]:.2f}',
    ]
    if has_amounts:
        foreign_labels = [_fmt_amount(v) for v in plot_df['foreign_net_amount'].fillna(0)]
        institution_labels = [_fmt_amount(v) for v in plot_df['institution_net_amount'].fillna(0)]
        customdata_cols.append(np.array(foreign_labels, dtype=object))
        customdata_cols.append(np.array(institution_labels, dtype=object))
        hover_lines.insert(-1, '──────────<br>')
        hover_lines.append('<br>──────────<br>')
        hover_lines.append('외국인 순매수: %{customdata[4]}<br>')
        hover_lines.append('기관 순매수: %{customdata[5]}')

    fig = go.Figure(data=go.Bar(
        y=y_labels,
        x=plot_df['combined_zscore'].abs(),
        orientation='h',
        marker_color=bar_color,
        marker_line=dict(color=_BG_PAPER, width=0.5),
        customdata=np.column_stack(customdata_cols),
        hovertemplate=''.join(hover_lines) + '<extra></extra>',
        text=[f'{v:.1f}σ' for v in plot_df['combined_zscore'].abs()],
        textposition='inside',
        insidetextanchor='end',
        textfont=dict(color='#0f172a', size=12, family='sans-serif'),
        cliponaxis=False,
    ))

    fig.update_layout(
        title=title,
        xaxis_title='|Z-Score|',
        yaxis_title='',
        height=max(250, len(plot_df) * 32 + 80),
        margin=dict(l=10, r=40, t=40, b=30),
    )
    return _apply_dark(fig)


# ---------------------------------------------------------------------------
# 수급 순위 바차트
# ---------------------------------------------------------------------------

def create_supply_ranking_chart(
    df: pd.DataFrame,
    amount_col: str,
    title: str,
    top_n: int = 10,
) -> go.Figure:
    """순매수/순매도 금액 순위 수평 바차트

    Args:
        df: stock_name, sector, amount_col 컬럼 포함 DataFrame (이미 정렬됨)
        amount_col: 금액 컬럼명 (foreign_net_amount / institution_net_amount)
        title: 차트 제목
        top_n: 표시할 종목 수
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=f'{title} (데이터 없음)', height=300)
        return _apply_dark(fig)

    plot_df = df.head(top_n).copy()
    # 수평 바: 아래→위 순서
    plot_df = plot_df.iloc[::-1]

    amounts_eok = plot_df[amount_col].fillna(0) / 1e8
    y_labels = plot_df['stock_name'].tolist()
    colors = ['#4ade80' if v >= 0 else '#f87171' for v in amounts_eok]

    text_labels = [_fmt_amount(v) for v in plot_df[amount_col].fillna(0)]

    fig = go.Figure(data=go.Bar(
        y=y_labels,
        x=amounts_eok.abs(),
        orientation='h',
        marker_color=colors,
        marker_line=dict(color=_BG_PAPER, width=0.5),
        customdata=np.column_stack([
            plot_df['sector'].fillna('-').values,
            np.array(text_labels, dtype=object),
        ]),
        hovertemplate=(
            '<b>%{y}</b><br>'
            '섹터: %{customdata[0]}<br>'
            '순매수: %{customdata[1]}'
            '<extra></extra>'
        ),
        text=text_labels,
        textposition='inside',
        insidetextanchor='end',
        textfont=dict(color='#0f172a', size=12, family='sans-serif'),
        cliponaxis=False,
    ))

    fig.update_layout(
        title=title,
        xaxis_title='|금액| (억원)',
        yaxis_title='',
        height=max(250, len(plot_df) * 32 + 80),
        margin=dict(l=10, r=60, t=40, b=30),
    )
    return _apply_dark(fig)


# ---------------------------------------------------------------------------
# 종목 상세 차트
# ---------------------------------------------------------------------------

def create_zscore_history_chart(
    df: pd.DataFrame,
    start_date: str = None,
) -> go.Figure:
    """Z-Score 시계열 라인 차트 (외국인/기관/종합 3선 + ±2σ 기준선)"""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title='Z-Score 추이 (데이터 없음)', height=400)
        return _apply_dark(fig)

    plot_df = df.copy()
    if start_date:
        plot_df = plot_df[plot_df['trade_date'] >= start_date]
    if plot_df.empty:
        plot_df = df.copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df['trade_date'], y=plot_df['combined_zscore'],
        name='종합 Z-Score', mode='lines',
        line=dict(color='#fb923c', width=2),
        hovertemplate='%{x}<br>종합 Z: %{y:.2f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=plot_df['trade_date'], y=plot_df['foreign_zscore'],
        name='외국인 Z-Score', mode='lines',
        line=dict(color='#38bdf8', width=1.5),
        hovertemplate='%{x}<br>외국인 Z: %{y:.2f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=plot_df['trade_date'], y=plot_df['institution_zscore'],
        name='기관 Z-Score', mode='lines',
        line=dict(color='#f472b6', width=1.5, dash='dot'),
        hovertemplate='%{x}<br>기관 Z: %{y:.2f}<extra></extra>',
    ))

    fig.add_hline(y=2.0, line_dash='dash', line_color='#4ade80', line_width=1,
                  annotation_text='+2σ', annotation_font=dict(color='#4ade80', size=11),
                  annotation_position='right')
    fig.add_hline(y=-2.0, line_dash='dash', line_color='#f87171', line_width=1,
                  annotation_text='-2σ', annotation_font=dict(color='#f87171', size=11),
                  annotation_position='right')
    fig.add_hline(y=0, line_color='#64748b', line_width=0.8)

    fig.update_layout(
        title='Z-Score 시계열',
        xaxis_title='날짜',
        yaxis_title='Z-Score (σ)',
        height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    return _apply_dark(fig)


def create_supply_amount_chart(
    df: pd.DataFrame,
    start_date: str = None,
) -> go.Figure:
    """외국인/기관/개인 순매수금액 바차트 + 누적순매수 라인 (3행 서브플롯, 억 단위)

    - 바차트 (좌측 y축): 일별 순매수금액
    - 라인차트 (우측 y축): 표시 기간 시작일 기준 누적순매수
    """
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title='수급 금액 (데이터 없음)', height=600)
        return _apply_dark(fig)

    plot_df = df.copy()
    if start_date:
        plot_df = plot_df[plot_df['trade_date'] >= start_date]
    if plot_df.empty:
        plot_df = df.copy()

    plot_df = plot_df.reset_index(drop=True)

    foreign_eok = plot_df['foreign_net_amount'].fillna(0) / 1e8
    inst_eok    = plot_df['institution_net_amount'].fillna(0) / 1e8
    indiv_col   = plot_df.get('individual_net_amount',
                              -(plot_df['foreign_net_amount'] + plot_df['institution_net_amount']))
    indiv_eok   = indiv_col.fillna(0) / 1e8

    foreign_cumsum = foreign_eok.cumsum()
    inst_cumsum    = inst_eok.cumsum()
    indiv_cumsum   = indiv_eok.cumsum()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=['외국인 순매수 (억원)', '기관 순매수 (억원)', '개인 순매수 (억원)'],
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{"secondary_y": True}]],
    )

    def _add_investor(row, daily, cumsum, bar_pos, bar_neg, line_color, name):
        fig.add_trace(go.Bar(
            x=plot_df['trade_date'], y=daily,
            name=f'{name} 일별',
            marker_color=[bar_pos if v >= 0 else bar_neg for v in daily],
            marker_line_width=0, opacity=0.75,
            hovertemplate=f'%{{x}}<br>{name} 순매수: %{{y:.1f}}억<extra></extra>',
        ), row=row, col=1, secondary_y=False)
        fig.add_trace(go.Scatter(
            x=plot_df['trade_date'], y=cumsum,
            name=f'{name} 누적',
            mode='lines', line=dict(color=line_color, width=2),
            hovertemplate=f'%{{x}}<br>{name} 누적: %{{y:.1f}}억<extra></extra>',
        ), row=row, col=1, secondary_y=True)
        fig.update_yaxes(title_text='일별 (억원)', secondary_y=False, row=row,
                         tickfont=dict(color=_MUTED), title_font=dict(color=_MUTED),
                         showgrid=True, gridcolor=_GRID)
        fig.update_yaxes(title_text='누적 (억원)', secondary_y=True, row=row,
                         tickfont=dict(color=line_color), title_font=dict(color=line_color),
                         showgrid=False)

    _add_investor(1, foreign_eok, foreign_cumsum, '#4ade80', '#f87171', '#38bdf8', '외국인')
    _add_investor(2, inst_eok,    inst_cumsum,    '#a78bfa', '#f472b6', '#c084fc', '기관')
    _add_investor(3, indiv_eok,   indiv_cumsum,   '#fbbf24', '#f87171', '#fb923c', '개인')

    # 모든 행 x축 날짜 표시
    for r in [1, 2, 3]:
        fig.update_xaxes(showticklabels=True, row=r, col=1)

    fig.update_layout(
        height=820,
        margin=dict(t=90),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='left', x=0),
        barmode='relative',
    )
    return _apply_dark(fig)


def create_signal_ma_chart(
    df: pd.DataFrame,
    start_date: str = None,
) -> go.Figure:
    """외국인 MA5/MA20 라인 + 동조율 보조 y축 + 골든크로스 마커"""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title='시그널 & MA (데이터 없음)', height=420)
        return _apply_dark(fig)

    plot_df = df.copy()
    if start_date:
        plot_df = plot_df[plot_df['trade_date'] >= start_date]
    if plot_df.empty:
        plot_df = df.copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    ma5_eok  = plot_df['ma5'].fillna(float('nan')) / 1e8
    ma20_eok = plot_df['ma20'].fillna(float('nan')) / 1e8

    fig.add_trace(go.Scatter(
        x=plot_df['trade_date'], y=ma5_eok,
        name='외국인 MA5', mode='lines',
        line=dict(color='#38bdf8', width=2),
        hovertemplate='%{x}<br>MA5: %{y:.1f}억<extra></extra>',
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=plot_df['trade_date'], y=ma20_eok,
        name='외국인 MA20', mode='lines',
        line=dict(color='#94a3b8', width=1.5, dash='dash'),
        hovertemplate='%{x}<br>MA20: %{y:.1f}억<extra></extra>',
    ), secondary_y=False)

    # 동조율 (secondary y)
    sync_valid = plot_df.dropna(subset=['sync_rate'])
    if not sync_valid.empty:
        fig.add_trace(go.Scatter(
            x=sync_valid['trade_date'], y=sync_valid['sync_rate'],
            name='동조율(%)', mode='lines',
            line=dict(color='#fb923c', width=1.5),
            hovertemplate='%{x}<br>동조율: %{y:.1f}%<extra></extra>',
        ), secondary_y=True)

    # 골든크로스(▲) / 데드크로스(▽) 마커
    valid_ma = plot_df.dropna(subset=['ma5', 'ma20']).copy()
    if len(valid_ma) >= 2:
        valid_ma = valid_ma.reset_index(drop=True)
        prev_ma5  = valid_ma['ma5'].shift(1)
        prev_ma20 = valid_ma['ma20'].shift(1)

        golden = valid_ma[(prev_ma5 < prev_ma20) & (valid_ma['ma5'] >= valid_ma['ma20'])]
        dead   = valid_ma[(prev_ma5 > prev_ma20) & (valid_ma['ma5'] <= valid_ma['ma20'])]

        if not golden.empty:
            fig.add_trace(go.Scatter(
                x=golden['trade_date'], y=golden['ma5'] / 1e8,
                name='골든크로스', mode='markers',
                marker=dict(symbol='triangle-up', color='#4ade80', size=13,
                            line=dict(width=1, color='#0f172a')),
                hovertemplate='골든크로스 (MA5↑MA20)<br>%{x}<extra></extra>',
            ), secondary_y=False)

        if not dead.empty:
            fig.add_trace(go.Scatter(
                x=dead['trade_date'], y=dead['ma5'] / 1e8,
                name='데드크로스', mode='markers',
                marker=dict(symbol='triangle-down', color='#f87171', size=13,
                            line=dict(width=1, color='#0f172a')),
                hovertemplate='데드크로스 (MA5↓MA20)<br>%{x}<extra></extra>',
            ), secondary_y=False)

    fig.add_hline(y=0, line_color='#64748b', line_width=0.8, secondary_y=False)

    fig.update_yaxes(title_text='순매수 (억원)', secondary_y=False,
                     tickfont=dict(color=_MUTED), title_font=dict(color=_MUTED),
                     showgrid=True, gridcolor=_GRID)
    fig.update_yaxes(title_text='동조율 (%)', secondary_y=True, range=[0, 100],
                     tickfont=dict(color='#fb923c'), title_font=dict(color='#fb923c'),
                     showgrid=False)

    fig.update_layout(
        title='외국인 MA 시그널 & 동조율',
        xaxis_title='날짜',
        height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    return _apply_dark(fig)


def create_multiperiod_zscore_bar(zscore_row: "pd.Series") -> go.Figure:
    """6기간(1W/1M/3M/6M/1Y/2Y) Z-Score 바차트 + ±2σ 기준선"""
    periods = ['1W', '1M', '3M', '6M', '1Y', '2Y']
    available = [p for p in periods if p in zscore_row.index and pd.notna(zscore_row[p])]

    if not available:
        fig = go.Figure()
        fig.update_layout(title='기간별 Z-Score (데이터 없음)', height=350)
        return _apply_dark(fig)

    values = [float(zscore_row[p]) for p in available]
    colors = ['#4ade80' if v >= 0 else '#f87171' for v in values]
    text_labels = [f'{v:.2f}σ' for v in values]

    fig = go.Figure(data=go.Bar(
        x=available,
        y=values,
        marker_color=colors,
        marker_line=dict(color=_BG_PAPER, width=0.5),
        text=text_labels,
        textposition='outside',
        textfont=dict(color=_TEXT, size=12),
        hovertemplate='기간: %{x}<br>Z-Score: %{y:.2f}σ<extra></extra>',
        cliponaxis=False,
    ))

    fig.add_hline(y=2.0, line_dash='dash', line_color='#4ade80', line_width=1.2,
                  annotation_text='+2σ 이상 수급', annotation_font=dict(color='#4ade80', size=11),
                  annotation_position='right')
    fig.add_hline(y=-2.0, line_dash='dash', line_color='#f87171', line_width=1.2,
                  annotation_text='-2σ 이상 매도', annotation_font=dict(color='#f87171', size=11),
                  annotation_position='right')
    fig.add_hline(y=0, line_color='#64748b', line_width=0.8)

    # y축 범위: 값 범위 + 여백 (텍스트 클리핑 방지)
    pad = max(abs(max(values)), abs(min(values)), 2.5) * 0.3
    fig.update_layout(
        title='기간별 Z-Score (멀티 기간 패턴)',
        xaxis_title='기간',
        yaxis_title='Z-Score (σ)',
        yaxis_range=[min(values) - pad, max(values) + pad],
        height=380,
    )
    return _apply_dark(fig)


# ---------------------------------------------------------------------------
# 워크포워드 차트
# ---------------------------------------------------------------------------

def create_wf_period_returns_chart(wf_df: pd.DataFrame) -> go.Figure:
    """워크포워드 기간별 수익률 바차트"""
    if wf_df.empty:
        fig = go.Figure()
        fig.update_layout(title='기간별 수익률 (데이터 없음)')
        return _apply_dark(fig)

    return_col = None
    for col in ['val_return', 'total_return', 'return']:
        if col in wf_df.columns:
            return_col = col
            break

    if return_col is None:
        fig = go.Figure()
        fig.update_layout(title='수익률 컬럼을 찾을 수 없습니다')
        return _apply_dark(fig)

    period_col = None
    for col in ['period', 'val_start', 'start']:
        if col in wf_df.columns:
            period_col = col
            break

    labels = wf_df[period_col].astype(str) if period_col else [str(i + 1) for i in range(len(wf_df))]
    returns = wf_df[return_col]
    colors = ['#4ade80' if r >= 0 else '#f87171' for r in returns]

    fig = go.Figure(data=go.Bar(
        x=labels,
        y=returns,
        marker_color=colors,
        marker_line=dict(color=_BG_PAPER, width=0.5),
        hovertemplate='기간: %{x}<br>수익률: %{y:.2f}%<extra></extra>',
    ))

    fig.add_hline(y=0, line_dash='dash', line_color='rgba(148,163,184,0.4)', line_width=1)

    fig.update_layout(
        title='기간별 검증 수익률',
        xaxis_title='기간',
        yaxis_title='수익률 (%)',
        height=400,
    )
    return _apply_dark(fig)
