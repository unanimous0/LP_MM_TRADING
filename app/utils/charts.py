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
# Z-Score 히트맵 (Stage 2 시각화를 Plotly로 재구현)
# ---------------------------------------------------------------------------

def create_zscore_heatmap(
    zscore_matrix: pd.DataFrame,
    sort_by: str = 'recent',
    top_n: int = 50,
    stock_names: pd.DataFrame = None,
) -> go.Figure:
    """
    멀티 기간 Z-Score 인터랙티브 히트맵

    Args:
        zscore_matrix: stock_code + 6개 기간 컬럼 (1W~2Y)
        sort_by: 정렬 기준 ('recent', 'momentum', 'weighted', 'average')
        top_n: 표시 종목 수
        stock_names: stock_code → stock_name 매핑 DataFrame

    Returns:
        go.Figure
    """
    if zscore_matrix.empty:
        fig = go.Figure()
        fig.update_layout(title='Z-Score 히트맵 (데이터 없음)')
        return fig

    df = zscore_matrix.copy()
    period_cols = [c for c in df.columns if c not in ('stock_code',)]

    # 정렬 키 계산
    if sort_by == 'recent':
        sort_key = df['1W'] if '1W' in df.columns else df[period_cols[0]]
    elif sort_by == 'momentum':
        first, last = period_cols[0], period_cols[-1]
        sort_key = df[first] - df[last]
    elif sort_by == 'weighted':
        weights = list(range(len(period_cols), 0, -1))
        total_w = sum(weights)
        sort_key = sum(df[c] * w for c, w in zip(period_cols, weights)) / total_w
    else:  # average
        sort_key = df[period_cols].mean(axis=1)

    df['_sort'] = sort_key
    df = df.sort_values('_sort', ascending=False).head(top_n)
    df = df.drop(columns=['_sort'])

    # 종목명 매핑
    if stock_names is not None and 'stock_code' in stock_names.columns:
        name_map = dict(zip(stock_names['stock_code'], stock_names['stock_name']))
        y_labels = [
            f"{name_map.get(code, code)}({code})"
            for code in df['stock_code']
        ]
    else:
        y_labels = df['stock_code'].tolist()

    z_values = df[period_cols].values
    text_values = [[f'{v:.2f}' for v in row] for row in z_values]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=period_cols,
        y=y_labels,
        colorscale='RdYlGn',
        zmid=0,
        zmin=-3,
        zmax=3,
        text=text_values,
        texttemplate='%{text}',
        hovertemplate='%{y}<br>기간: %{x}<br>Z-Score: %{z:.2f}<extra></extra>',
        colorbar=dict(title='Z-Score'),
    ))

    fig.update_layout(
        title=f'수급 Z-Score 히트맵 (정렬: {sort_by}, 상위 {len(df)}개)',
        xaxis_title='기간',
        yaxis_title='종목',
        yaxis=dict(autorange='reversed'),
        height=max(400, len(df) * 22 + 100),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    return fig


# ---------------------------------------------------------------------------
# 패턴 분석 차트
# ---------------------------------------------------------------------------

def create_pattern_pie_chart(report_df: pd.DataFrame) -> go.Figure:
    """패턴별 분포 파이차트"""
    if report_df.empty or 'pattern' not in report_df.columns:
        fig = go.Figure()
        fig.update_layout(title='패턴 분포 (데이터 없음)')
        return fig

    counts = report_df['pattern'].value_counts().reset_index()
    counts.columns = ['pattern', 'count']

    color_map = {
        '모멘텀형': '#FF6B6B',
        '지속형': '#4ECDC4',
        '전환형': '#45B7D1',
        '기타': '#95A5A6',
    }
    colors = [color_map.get(p, '#95A5A6') for p in counts['pattern']]

    fig = go.Figure(data=go.Pie(
        labels=counts['pattern'],
        values=counts['count'],
        marker=dict(colors=colors),
        textinfo='label+value+percent',
        hovertemplate='%{label}<br>종목 수: %{value}<br>비율: %{percent}<extra></extra>',
    ))

    fig.update_layout(
        title='패턴별 종목 분포',
        height=350,
    )

    return fig


def create_score_histogram(report_df: pd.DataFrame) -> go.Figure:
    """패턴 점수 분포 히스토그램"""
    if report_df.empty or 'score' not in report_df.columns:
        fig = go.Figure()
        fig.update_layout(title='점수 분포 (데이터 없음)')
        return fig

    fig = go.Figure(data=go.Histogram(
        x=report_df['score'],
        nbinsx=20,
        marker_color='#2E86AB',
        opacity=0.8,
        hovertemplate='점수: %{x:.0f}<br>종목 수: %{y}<extra></extra>',
    ))

    # 70점 기준선
    fig.add_vline(
        x=70, line_dash='dash', line_color='red', line_width=1.5,
        annotation_text='관심 기준 (70점)',
        annotation_position='top right',
    )

    fig.update_layout(
        title='패턴 점수 분포',
        xaxis_title='점수',
        yaxis_title='종목 수',
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    return fig


def create_signal_distribution_chart(report_df: pd.DataFrame) -> go.Figure:
    """시그널 수 분포 바차트"""
    if report_df.empty or 'signal_count' not in report_df.columns:
        fig = go.Figure()
        fig.update_layout(title='시그널 분포 (데이터 없음)')
        return fig

    counts = report_df['signal_count'].value_counts().sort_index().reset_index()
    counts.columns = ['signal_count', 'count']

    colors = ['#95A5A6' if s < 2 else '#06A77D' for s in counts['signal_count']]

    fig = go.Figure(data=go.Bar(
        x=counts['signal_count'].astype(str) + '개',
        y=counts['count'],
        marker_color=colors,
        hovertemplate='시그널 %{x}<br>종목 수: %{y}<extra></extra>',
    ))

    fig.update_layout(
        title='시그널 수별 종목 분포',
        xaxis_title='시그널 수',
        yaxis_title='종목 수',
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    return fig


# ---------------------------------------------------------------------------
# 워크포워드 차트
# ---------------------------------------------------------------------------

def create_wf_period_returns_chart(wf_df: pd.DataFrame) -> go.Figure:
    """워크포워드 기간별 수익률 바차트"""
    if wf_df.empty:
        fig = go.Figure()
        fig.update_layout(title='기간별 수익률 (데이터 없음)')
        return fig

    # 컬럼명 자동 감지
    return_col = None
    for col in ['val_return', 'total_return', 'return']:
        if col in wf_df.columns:
            return_col = col
            break

    if return_col is None:
        fig = go.Figure()
        fig.update_layout(title='수익률 컬럼을 찾을 수 없습니다')
        return fig

    period_col = None
    for col in ['period', 'val_start', 'start']:
        if col in wf_df.columns:
            period_col = col
            break

    labels = wf_df[period_col].astype(str) if period_col else [str(i+1) for i in range(len(wf_df))]
    returns = wf_df[return_col]
    colors = ['#06A77D' if r >= 0 else '#D62828' for r in returns]

    fig = go.Figure(data=go.Bar(
        x=labels,
        y=returns,
        marker_color=colors,
        hovertemplate='기간: %{x}<br>수익률: %{y:.2f}%<extra></extra>',
    ))

    fig.add_hline(y=0, line_dash='dash', line_color='black', line_width=1)

    fig.update_layout(
        title='기간별 검증 수익률',
        xaxis_title='기간',
        yaxis_title='수익률 (%)',
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    return fig
