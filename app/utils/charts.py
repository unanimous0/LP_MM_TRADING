"""
Plotly 차트 헬퍼 모듈

Z-Score 히트맵, 패턴 파이차트, 점수 히스토그램 등
Streamlit 대시보드 전용 차트 생성.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# ---------------------------------------------------------------------------
# 다크 테마 상수
# ---------------------------------------------------------------------------
_BG_PLOT  = '#0f172a'   # slate-900
_BG_PAPER = '#1e293b'   # slate-800
_GRID     = '#334155'   # slate-700
_TEXT     = '#e2e8f0'   # slate-200
_MUTED    = '#94a3b8'   # slate-400


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
