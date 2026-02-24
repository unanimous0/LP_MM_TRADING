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
    direction: str = 'buy',
    report_df: pd.DataFrame = None,
) -> go.Figure:
    """멀티 기간 Z-Score 인터랙티브 히트맵 + 정렬 기준 바차트

    Args:
        direction: 'buy' (매수 상위) | 'sell' (매도 상위) | 'both' (양쪽 각 top_n//2)
        report_df: 패턴/점수/시그널 정보 (있으면 호버에 표시)
    """
    if zscore_matrix.empty:
        fig = go.Figure()
        fig.update_layout(title='Z-Score 히트맵 (데이터 없음)')
        return _apply_dark(fig)

    df = zscore_matrix.copy()
    period_cols = [c for c in df.columns if c not in ('stock_code',) and not c.startswith('_')]

    # 방향 확신도 기반 정렬 키 계산
    # Z-Score는 편차 측정이므로 매도 중 종목도 양수 Z가 가능 (매도 완화)
    # confidence = tanh(today_sff / rolling_std)로 실제 수급 방향을 반영하여 정렬
    has_sff_meta = '_today_sff' in df.columns
    adj = {}
    if has_sff_meta:
        for col in period_cols:
            std_col = f'_std_{col}'
            if std_col in df.columns:
                std_safe = df[std_col].replace(0, np.nan)
                conf = np.tanh(df['_today_sff'] / std_safe).fillna(0)
                adj[col] = conf
            else:
                adj[col] = pd.Series(1.0, index=df.index)

    def _adj_z(col, dir_mode):
        """방향별 confidence 적용된 Z-Score 반환"""
        if not has_sff_meta or col not in adj:
            return df[col]
        if dir_mode == 'buy':
            return df[col] * np.maximum(adj[col], 0)
        elif dir_mode == 'sell':
            return df[col] * np.maximum(-adj[col], 0)
        return df[col] * np.abs(adj[col])  # both: 절대값

    def _compute_sort_key(dir_mode):
        if sort_by == 'recent':
            return _adj_z('5D', dir_mode) if '5D' in df.columns else _adj_z(period_cols[0], dir_mode)
        elif sort_by == 'momentum':
            first, last = period_cols[0], period_cols[-1]
            return _adj_z(first, dir_mode) - _adj_z(last, dir_mode)
        elif sort_by == 'weighted':
            weights = list(range(len(period_cols), 0, -1))
            total_w = sum(weights)
            return sum(_adj_z(c, dir_mode) * w for c, w in zip(period_cols, weights)) / total_w
        else:
            return pd.concat([_adj_z(c, dir_mode) for c in period_cols], axis=1).mean(axis=1)

    if direction == 'buy':
        df['_sort'] = _compute_sort_key('buy')
        df = df.sort_values('_sort', ascending=False).head(top_n)
    elif direction == 'sell':
        df['_sort'] = _compute_sort_key('sell')
        df = df.sort_values('_sort', ascending=True).head(top_n)
    else:  # both: 매수 절반(위) + 매도 절반(아래)
        half = max(1, top_n // 2)
        df['_sort_buy'] = _compute_sort_key('buy')
        df['_sort_sell'] = _compute_sort_key('sell')
        buy_df  = df.sort_values('_sort_buy', ascending=False).head(half)
        sell_df = df.sort_values('_sort_sell', ascending=True).head(half)
        buy_df['_sort'] = buy_df['_sort_buy']
        sell_df['_sort'] = sell_df['_sort_sell']
        df = pd.concat([buy_df, sell_df]).drop_duplicates(subset='stock_code')
        df = df.drop(columns=['_sort_buy', '_sort_sell'], errors='ignore')

    # 정렬 기준 값 추출 (바차트용) — drop 전에 저장
    sort_values = df['_sort'].values
    # 메타데이터/내부 컬럼 제거 (히트맵 셀에 표시하지 않음)
    drop_cols = [c for c in df.columns if c.startswith('_')]
    df = df.drop(columns=drop_cols)

    # B: 패턴/점수/시그널 customdata (report_df 있을 때 hover에 표시)
    has_report = (
        report_df is not None and not report_df.empty
        and 'stock_code' in report_df.columns
        and 'pattern' in report_df.columns
    )
    heatmap_customdata = None
    if has_report:
        rmap = report_df.drop_duplicates(subset='stock_code').set_index('stock_code')
        n_s, n_p = len(df), len(period_cols)
        cd = np.empty((n_s, n_p, 3), dtype=object)
        for i, code in enumerate(df['stock_code'].tolist()):
            if code in rmap.index:
                pat = str(rmap.at[code, 'pattern'])
                sc  = float(rmap.at[code, 'score']) if 'score' in rmap.columns else 0.0
                sig = int(rmap.at[code, 'signal_count']) if 'signal_count' in rmap.columns else 0
            else:
                pat, sc, sig = '기타', 0.0, 0
            for j in range(n_p):
                cd[i, j, 0] = pat
                cd[i, j, 1] = f'{sc:.0f}'
                cd[i, j, 2] = sig
        heatmap_customdata = cd
    hover_tmpl = (
        '%{y}<br>기간: %{x}<br>Z-Score: %{z:.2f}σ<br>'
        '────────<br>패턴: %{customdata[0]}<br>'
        '점수: %{customdata[1]} · 시그널: %{customdata[2]}개'
        '<extra></extra>'
    ) if has_report else (
        '%{y}<br>기간: %{x}<br>Z-Score: %{z:.2f}<extra></extra>'
    )

    if stock_names is not None and 'stock_code' in stock_names.columns:
        name_map = dict(zip(stock_names['stock_code'], stock_names['stock_name']))
        y_labels = [f"{name_map.get(code, code)}({code})" for code in df['stock_code']]
    else:
        y_labels = df['stock_code'].tolist()

    z_values = df[period_cols].values
    text_values = [[f'{v:.2f}' for v in row] for row in z_values]

    colorscale = [
        [0.0,  '#ef4444'],
        [0.35, '#fca5a5'],
        [0.5,  '#334155'],
        [0.65, '#86efac'],
        [1.0,  '#22c55e'],
    ]

    sort_label = {
        'recent':   '5D Z',
        'momentum': '모멘텀(5D-500D)',
        'weighted': '가중평균',
        'average':  '단순평균',
    }.get(sort_by, sort_by)

    # 히트맵(좌 80%) + 정렬기준 바차트(우 20%) 서브플롯
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.80, 0.20],
        shared_yaxes=True,
        horizontal_spacing=0.01,
    )

    fig.add_trace(go.Heatmap(
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
        customdata=heatmap_customdata,
        hovertemplate=hover_tmpl,
        colorbar=dict(
            title=dict(text='Z-Score', font=dict(color=_TEXT)),
            tickfont=dict(color=_MUTED),
            bgcolor=_BG_PAPER,
            bordercolor=_GRID,
            borderwidth=1,
            x=1.02,
        ),
    ), row=1, col=1)

    bar_colors = ['#4ade80' if v >= 0 else '#f87171' for v in sort_values]
    fig.add_trace(go.Bar(
        x=sort_values,
        y=y_labels,
        orientation='h',
        marker_color=bar_colors,
        marker_line_width=0,
        opacity=0.85,
        text=[f'{v:.2f}' for v in sort_values],
        textposition='auto',
        textfont=dict(color='#0f172a', size=9),
        cliponaxis=False,
        hovertemplate='%{y}<br>' + sort_label + ': %{x:.2f}<extra></extra>',
        showlegend=False,
    ), row=1, col=2)

    fig.update_layout(
        title=f'수급 Z-Score 히트맵 (정렬: {sort_label}, 상위 {len(df)}개)',
        yaxis=dict(autorange='reversed', title=''),
        xaxis=dict(side='top', title=''),
        xaxis2=dict(
            title=sort_label,
            side='top',
            tickfont=dict(color=_MUTED, size=9),
            title_font=dict(color=_MUTED, size=10),
            zeroline=True,
            zerolinecolor=_GRID,
            zerolinewidth=1,
        ),
        height=max(400, len(df) * 22 + 100),
    )
    return _apply_dark(fig)


def create_sector_zscore_heatmap(
    zscore_matrix: pd.DataFrame,
    stock_list: pd.DataFrame,
    sort_by: str = 'recent',
) -> go.Figure:
    """섹터별 평균 Z-Score 히트맵

    Args:
        zscore_matrix: stock_code + 기간 컬럼 DataFrame
        stock_list: stock_code / sector 컬럼 포함 종목 마스터
        sort_by: 정렬 기준 (recent / momentum / weighted / average)
    """
    if zscore_matrix.empty or stock_list is None or stock_list.empty:
        fig = go.Figure()
        fig.update_layout(title='섹터 Z-Score (데이터 없음)', height=400)
        return _apply_dark(fig)

    period_cols = [c for c in zscore_matrix.columns if c != 'stock_code' and not c.startswith('_')]

    sl = stock_list[['stock_code', 'sector']].dropna(subset=['sector'])
    merged = zscore_matrix.merge(sl, on='stock_code', how='inner')
    if merged.empty:
        fig = go.Figure()
        fig.update_layout(title='섹터 Z-Score (섹터 정보 없음)', height=400)
        return _apply_dark(fig)

    sector_df = merged.groupby('sector')[period_cols].mean().reset_index()
    stock_counts = merged.groupby('sector')['stock_code'].count().to_dict()

    # 정렬 키
    if sort_by == 'recent':
        sort_key = sector_df['5D'] if '5D' in sector_df.columns else sector_df[period_cols[0]]
    elif sort_by == 'momentum':
        first, last = period_cols[0], period_cols[-1]
        sort_key = sector_df[first] - sector_df[last]
    elif sort_by == 'weighted':
        weights = list(range(len(period_cols), 0, -1))
        total_w = sum(weights)
        sort_key = sum(sector_df[c] * w for c, w in zip(period_cols, weights)) / total_w
    else:
        sort_key = sector_df[period_cols].mean(axis=1)

    sector_df['_sort'] = sort_key
    sector_df = sector_df.sort_values('_sort', ascending=False)

    sort_label = {
        'recent':   '5D Z',
        'momentum': '모멘텀(5D-500D)',
        'weighted': '가중평균',
        'average':  '단순평균',
    }.get(sort_by, sort_by)

    y_labels = [
        f"{row['sector']} ({stock_counts.get(row['sector'], 0)}개)"
        for _, row in sector_df.iterrows()
    ]
    z_values = sector_df[period_cols].values
    text_values = [[f'{v:.2f}' for v in row] for row in z_values]

    colorscale = [
        [0.0,  '#ef4444'],
        [0.35, '#fca5a5'],
        [0.5,  '#334155'],
        [0.65, '#86efac'],
        [1.0,  '#22c55e'],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=period_cols,
        y=y_labels,
        colorscale=colorscale,
        zmid=0, zmin=-3, zmax=3,
        text=text_values,
        texttemplate='%{text}',
        textfont=dict(color='#0f172a', size=10),
        hovertemplate='%{y}<br>기간: %{x}<br>평균 Z-Score: %{z:.2f}σ<extra></extra>',
        colorbar=dict(
            title=dict(text='Z-Score', font=dict(color=_TEXT)),
            tickfont=dict(color=_MUTED),
            bgcolor=_BG_PAPER,
            bordercolor=_GRID,
            borderwidth=1,
        ),
    ))

    fig.update_layout(
        title=f'섹터별 평균 Z-Score 히트맵 (정렬: {sort_label})',
        yaxis=dict(autorange='reversed', title=''),
        xaxis=dict(side='top', title=''),
        height=max(400, len(sector_df) * 32 + 120),
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
# 섹터 크로스 분석 차트
# ---------------------------------------------------------------------------

_PATTERN_COLORS = {
    '모멘텀형': '#f472b6',   # pink-400
    '지속형':   '#38bdf8',   # sky-400
    '전환형':   '#4ade80',   # green-400
    '기타':     '#64748b',   # slate-500
}


def create_sector_pattern_crosstab_chart(report_df: pd.DataFrame) -> go.Figure:
    """섹터별 패턴 분포 스택 바차트

    x축: 섹터 (종목수 내림차순), y축: 종목 수, 색상: 패턴별
    """
    if report_df.empty or 'sector' not in report_df.columns:
        fig = go.Figure()
        fig.update_layout(title='섹터별 패턴 분포 (데이터 없음)', height=400)
        return _apply_dark(fig)

    df = report_df.copy()
    df['sector'] = df['sector'].fillna('기타')
    df['pattern'] = df['pattern'].fillna('기타')

    # 섹터별 종목수 내림차순 정렬
    sector_order = df['sector'].value_counts().index.tolist()

    patterns = ['모멘텀형', '지속형', '전환형', '기타']
    ct = pd.crosstab(df['sector'], df['pattern'])

    fig = go.Figure()
    for pat in patterns:
        if pat not in ct.columns:
            continue
        vals = [int(ct.at[s, pat]) if s in ct.index else 0 for s in sector_order]
        fig.add_trace(go.Bar(
            x=sector_order,
            y=vals,
            name=pat,
            marker_color=_PATTERN_COLORS.get(pat, '#64748b'),
            hovertemplate=f'%{{x}}<br>{pat}: %{{y}}개<extra></extra>',
        ))

    fig.update_layout(
        barmode='stack',
        title='섹터별 패턴 분포',
        xaxis_title='',
        yaxis_title='종목 수',
        xaxis=dict(tickangle=-45),
        height=450,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    return _apply_dark(fig)


def create_sector_avg_score_chart(report_df: pd.DataFrame) -> go.Figure:
    """섹터별 평균 점수 수평 바차트

    y축: '섹터 (N개)', x축: 평균 점수, 호버: 평균 시그널 수
    """
    if report_df.empty or 'sector' not in report_df.columns or 'score' not in report_df.columns:
        fig = go.Figure()
        fig.update_layout(title='섹터별 평균 점수 (데이터 없음)', height=400)
        return _apply_dark(fig)

    df = report_df.copy()
    df['sector'] = df['sector'].fillna('기타')

    agg = df.groupby('sector').agg(
        avg_score=('score', 'mean'),
        count=('score', 'size'),
        avg_signals=('signal_count', 'mean'),
    ).reset_index()
    agg = agg.sort_values('avg_score', ascending=True)  # 수평바: 아래→위

    y_labels = [f"{row['sector']} ({row['count']}개)" for _, row in agg.iterrows()]

    fig = go.Figure(data=go.Bar(
        y=y_labels,
        x=agg['avg_score'],
        orientation='h',
        marker_color='#38bdf8',
        marker_line=dict(color=_BG_PAPER, width=0.5),
        customdata=np.column_stack([
            agg['avg_signals'].round(1).values,
            agg['count'].values,
        ]),
        text=[f'{v:.1f}' for v in agg['avg_score']],
        textposition='auto',
        textfont=dict(color='#0f172a', size=11),
        hovertemplate=(
            '<b>%{y}</b><br>'
            '평균 점수: %{x:.1f}<br>'
            '평균 시그널: %{customdata[0]}개<br>'
            '종목 수: %{customdata[1]}개'
            '<extra></extra>'
        ),
    ))

    fig.update_layout(
        title='섹터별 평균 패턴 점수',
        xaxis_title='평균 점수',
        yaxis_title='',
        height=max(350, len(agg) * 28 + 100),
    )
    return _apply_dark(fig)


def create_sector_concentration_chart(report_df: pd.DataFrame, min_stocks: int = 5) -> go.Figure:
    """섹터별 수급 집중도 수평 바차트

    섹터점수 = 평균점수 × (1 + 고득점종목수/전체종목수)
    고득점 = final_score ≥ 70
    """
    if report_df.empty or 'sector' not in report_df.columns:
        fig = go.Figure()
        fig.update_layout(title='섹터별 수급 집중도 (데이터 없음)', height=400)
        return _apply_dark(fig)

    df = report_df.copy()
    df['sector'] = df['sector'].fillna('기타')
    if 'final_score' not in df.columns:
        df['final_score'] = df['score'] + df.get('signal_count', 0) * 5

    agg = df.groupby('sector').agg(
        avg_score=('final_score', 'mean'),
        total_count=('stock_code', 'size'),
    ).reset_index()

    high = df[df['final_score'] >= 70].groupby('sector').size().reset_index(name='high_count')
    agg = agg.merge(high, on='sector', how='left')
    agg['high_count'] = agg['high_count'].fillna(0).astype(int)

    # 최소 종목 수 필터
    agg = agg[agg['total_count'] >= min_stocks]
    if agg.empty:
        fig = go.Figure()
        fig.update_layout(title=f'섹터별 수급 집중도 ({min_stocks}개 이상 섹터 없음)', height=400)
        return _apply_dark(fig)

    agg['sector_score'] = agg['avg_score'] * (1 + agg['high_count'] / agg['total_count'])
    agg = agg.nlargest(10, 'sector_score').sort_values('sector_score', ascending=True)

    y_labels = [f"{row['sector']} ({row['total_count']}개)" for _, row in agg.iterrows()]

    fig = go.Figure(data=go.Bar(
        y=y_labels,
        x=agg['sector_score'],
        orientation='h',
        marker_color='#fb923c',
        marker_line=dict(color=_BG_PAPER, width=0.5),
        customdata=np.column_stack([
            agg['avg_score'].round(1).values,
            agg['total_count'].values,
            agg['high_count'].values,
        ]),
        text=[f'{v:.1f}' for v in agg['sector_score']],
        textposition='auto',
        textfont=dict(color='#0f172a', size=11),
        hovertemplate=(
            '<b>%{y}</b><br>'
            '섹터 점수: %{x:.1f}<br>'
            '평균 점수: %{customdata[0]}<br>'
            '종목 수: %{customdata[1]}개<br>'
            '고득점(≥70): %{customdata[2]}개'
            '<extra></extra>'
        ),
    ))

    fig.update_layout(
        title='섹터별 수급 집중도 (TOP 10)',
        xaxis_title='섹터 점수',
        yaxis_title='',
        height=max(350, len(agg) * 32 + 100),
    )
    return _apply_dark(fig)


def create_sector_treemap_html(report_df: pd.DataFrame, top_per_sector: int = 10) -> str:
    """섹터별 종목 Treemap — D3.js 기반 HTML 반환 (st.components.v1.html용)

    박스 크기: 종합점수 비례, 색상: 빨강(낮음) → 초록(높음)
    다크 테마, 섹터 헤더 라벨, 동적 텍스트 크기, 호버 툴팁
    """
    import json as _json

    if report_df.empty or 'sector' not in report_df.columns:
        return '<div style="color:#94a3b8;padding:40px;text-align:center;">데이터 없음</div>'

    df = report_df.copy()
    df['sector'] = df['sector'].fillna('기타')
    if 'final_score' not in df.columns:
        df['final_score'] = df['score'] + df.get('signal_count', 0) * 5

    # 섹터별 상위 종목 → 계층 데이터
    sector_order = df.groupby('sector')['final_score'].sum().nlargest(20).index.tolist()
    treemap_data = {'name': 'root', 'children': []}
    for sector in sector_order:
        sec_df = df[df['sector'] == sector].nlargest(top_per_sector, 'final_score')
        if sec_df.empty:
            continue
        children = []
        for _, r in sec_df.iterrows():
            children.append({
                'name': str(r['stock_name']),
                'value': float(r['final_score']),
                'code': str(r['stock_code']),
                'pattern': str(r.get('pattern', '기타')),
                'score': round(float(r['final_score']), 1),
                'signals': int(r.get('signal_count', 0)),
            })
        sec_avg = float(sec_df['final_score'].mean())
        high_count = int((sec_df['final_score'] >= 70).sum())
        sec_score = sec_avg * (1 + high_count / len(sec_df))
        treemap_data['children'].append({
            'name': sector,
            'avg_score': round(sec_avg, 1),
            'sector_score': round(sec_score, 1),
            'children': children,
        })

    data_json = _json.dumps(treemap_data, ensure_ascii=False)

    return f'''
<div id="treemap-container" style="width:100%;background:#0f172a;border-radius:8px;position:relative;"></div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const iframe = window.frameElement;
if (iframe) {{ iframe.style.width = "100%"; }}

(function() {{
const data = {data_json};
const container = document.getElementById("treemap-container");
const W = 1400;
const H = 800;

const svg = d3.select(container).append("svg")
    .attr("viewBox", `0 0 ${{W}} ${{H}}`)
    .attr("width", "100%")
    .style("display", "block");

// 툴팁
const tip = d3.select(container).append("div")
    .style("position","absolute").style("pointer-events","none")
    .style("background","rgba(15,23,42,0.95)").style("color","#e2e8f0")
    .style("border","1px solid #334155").style("border-radius","8px")
    .style("padding","12px 16px").style("font-size","13px")
    .style("line-height","1.7").style("opacity",0)
    .style("box-shadow","0 8px 24px rgba(0,0,0,0.5)").style("z-index","100");

// 색상: HTML 리포트와 동일한 RdYlGn
const cScale = d3.scaleSequential()
    .domain([40, 100])
    .interpolator(d3.interpolateRdYlGn);

function textColor(bg) {{
    const c = d3.color(bg);
    return (0.299*c.r + 0.587*c.g + 0.114*c.b) > 140 ? "#1F2937" : "#FFFFFF";
}}

const root = d3.hierarchy(data).sum(d => d.value).sort((a,b) => b.value - a.value);

d3.treemap()
    .size([W, H])
    .padding(1)
    .paddingOuter(2)
    .paddingTop(24)
    .round(true)(root);

// 섹터 배경 (어두운 경계)
const sectorG = svg.selectAll("g.sector")
    .data(root.children || []).enter().append("g").attr("class","sector");

sectorG.append("rect")
    .attr("x", d => d.x0).attr("y", d => d.y0)
    .attr("width", d => d.x1 - d.x0).attr("height", d => d.y1 - d.y0)
    .attr("fill","#1e293b").attr("stroke","#334155").attr("stroke-width",1)
    .attr("rx",2);

// 종목 박스
const leaves = svg.selectAll("g.leaf")
    .data(root.leaves()).enter().append("g").attr("class","leaf")
    .attr("transform", d => `translate(${{d.x0}},${{d.y0}})`);

// clipPath (텍스트 오버플로 방지)
const defs = svg.append("defs");
leaves.each(function(d, i) {{
    defs.append("clipPath").attr("id", "clip-" + i)
        .append("rect")
        .attr("width", Math.max(0, d.x1 - d.x0 - 2))
        .attr("height", Math.max(0, d.y1 - d.y0 - 2))
        .attr("x", 1).attr("y", 1);
}});

leaves.append("rect")
    .attr("width", d => d.x1 - d.x0)
    .attr("height", d => d.y1 - d.y0)
    .attr("fill", d => cScale(d.data.score))
    .attr("stroke","#475569").attr("stroke-width",1).attr("rx",3)
    .style("cursor","pointer")
    .on("mouseover", function(e, d) {{
        d3.select(this).attr("stroke","#FFFFFF").attr("stroke-width",2.5)
            .style("filter","brightness(1.1)");
        const pat_c = {{"모멘텀형":"#f472b6","지속형":"#38bdf8","전환형":"#4ade80"}}[d.data.pattern] || "#94a3b8";
        tip.html(
            `<div style="border-bottom:1px solid #334155;padding-bottom:6px;margin-bottom:6px;">` +
            `<strong style="font-size:15px;color:#38bdf8;">${{d.data.name}}</strong>` +
            `<span style="color:#64748b;margin-left:8px;">${{d.data.code}}</span></div>` +
            `<span style="color:#94a3b8;">섹터:</span> ${{d.parent.data.name}}<br>` +
            `<span style="color:#94a3b8;">패턴:</span> <span style="color:${{pat_c}};font-weight:600;">${{d.data.pattern}}</span><br>` +
            `<span style="color:#94a3b8;">점수:</span> <strong style="color:#4ade80;font-size:15px;">${{d.data.score}}</strong><br>` +
            `<span style="color:#94a3b8;">시그널:</span> <strong style="color:#fbbf24;">${{d.data.signals}}개</strong>`
        ).style("opacity",1)
         .style("left", Math.min(e.offsetX + 15, W - 220) + "px")
         .style("top", Math.max(e.offsetY - 80, 10) + "px");
    }})
    .on("mouseout", function() {{
        d3.select(this).attr("stroke","#475569").attr("stroke-width",1)
            .style("filter","none");
        tip.style("opacity",0);
    }});

// 텍스트 그룹 (clipPath 적용)
const textG = leaves.append("g")
    .attr("clip-path", (d, i) => `url(#clip-${{i}})`)
    .style("pointer-events","none");

// 종목명
textG.append("text")
    .attr("x", d => (d.x1 - d.x0) / 2)
    .attr("y", d => {{
        const h = d.y1 - d.y0;
        return h > 50 ? h/2 - 5 : h/2;
    }})
    .attr("text-anchor","middle").attr("dominant-baseline","middle")
    .text(d => {{
        const w = d.x1 - d.x0, h = d.y1 - d.y0;
        if (w > 70 && h > 35) return d.data.name.length > 8 ? d.data.name.substring(0,7) + "…" : d.data.name;
        if (w > 45 && h > 25) return d.data.name.length > 5 ? d.data.name.substring(0,4) + "…" : d.data.name;
        return "";
    }})
    .attr("font-size", d => {{
        const w = d.x1 - d.x0;
        if (w > 120) return "14px";
        if (w > 80) return "12px";
        if (w > 50) return "10px";
        return "8px";
    }})
    .attr("font-weight","700")
    .attr("fill", d => textColor(cScale(d.data.score)));

// 점수
textG.append("text")
    .attr("x", d => (d.x1 - d.x0) / 2)
    .attr("y", d => {{
        const h = d.y1 - d.y0;
        return h > 50 ? h/2 + 14 : h/2 + 12;
    }})
    .attr("text-anchor","middle").attr("dominant-baseline","middle")
    .text(d => {{
        const w = d.x1 - d.x0, h = d.y1 - d.y0;
        return (w > 70 && h > 45) ? d.data.score + "점" : "";
    }})
    .attr("font-size","11px").attr("font-weight","600")
    .attr("fill", d => textColor(cScale(d.data.score)))
    .style("opacity",0.9);

// 섹터 라벨 배지 — leaves 위에 그려야 보임 (SVG z-order = DOM 순서)
(root.children || []).forEach(sector => {{
    const x0 = sector.x0;
    const y0 = sector.y0;
    const sW = sector.x1 - sector.x0;

    const avg = sector.data.avg_score != null ? sector.data.avg_score.toFixed(1) : "";
    const sc = sector.data.sector_score != null ? sector.data.sector_score.toFixed(1) : "";
    const label = `${{sector.data.name}} ${{avg}}점 (${{sc}})`;
    const labelW = Math.min(label.length * 8 + 16, sW);

    svg.append("rect")
        .attr("x", x0).attr("y", y0)
        .attr("width", labelW).attr("height", 22)
        .attr("fill", "rgba(15,23,42,0.85)")
        .attr("rx", 4).style("pointer-events","none");

    svg.append("text")
        .attr("x", x0 + 8).attr("y", y0 + 15)
        .text(sW > 80 ? label : sector.data.name)
        .attr("font-size","12px").attr("font-weight","700")
        .attr("fill","#FFFFFF").style("pointer-events","none");
}});

}})();
</script>'''


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
        margin=dict(t=120),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.07, xanchor='left', x=0),
        barmode='relative',
    )
    return _apply_dark(fig)


_MA_COLORS = {
    5:   '#38bdf8',  # sky-400
    10:  '#22d3ee',  # cyan-400
    20:  '#94a3b8',  # slate-400
    60:  '#a78bfa',  # violet-400
    120: '#fbbf24',  # amber-400
    240: '#fb923c',  # orange-400
}


def create_signal_ma_chart(
    df: pd.DataFrame,
    start_date: str = None,
    ma_periods: list = None,
) -> go.Figure:
    """외국인 MA 라인 + 골든/데드크로스 마커 (ma_periods로 표시할 MA 선택)"""
    if ma_periods is None:
        ma_periods = [5, 20]
    ma_periods = sorted(ma_periods)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title='시그널 & MA (데이터 없음)', height=420)
        return _apply_dark(fig)

    plot_df = df.copy()
    if start_date:
        plot_df = plot_df[plot_df['trade_date'] >= start_date]
    if plot_df.empty:
        plot_df = df.copy()

    fig = go.Figure()

    # 선택된 각 MA 계산 및 trace 추가
    ma_series = {}
    for p in ma_periods:
        col = f'ma{p}'
        if col in plot_df.columns:
            series = plot_df[col].fillna(float('nan'))
        else:
            series = plot_df['foreign_net_amount'].rolling(p).mean()
        ma_series[p] = series
        color = _MA_COLORS.get(p, '#e2e8f0')
        fig.add_trace(go.Scatter(
            x=plot_df['trade_date'], y=series / 1e8,
            name=f'MA{p}', mode='lines',
            line=dict(color=color, width=2 if p == min(ma_periods) else 1.5),
            hovertemplate=f'%{{x}}<br>MA{p}: %{{y:.1f}}억<extra></extra>',
        ))

    # 골든/데드크로스: MA가 정확히 2개일 때만
    if len(ma_periods) == 2:
        short_p, long_p = ma_periods[0], ma_periods[1]
        tmp = pd.DataFrame({
            'trade_date': plot_df['trade_date'].values,
            'short': ma_series[short_p].values,
            'long':  ma_series[long_p].values,
        }).dropna().reset_index(drop=True)

        golden_x, golden_y, dead_x, dead_y = [], [], [], []
        for i in range(1, len(tmp)):
            p_row = tmp.iloc[i - 1]
            c_row = tmp.iloc[i]
            diff_p = p_row['short'] - p_row['long']
            diff_c = c_row['short'] - c_row['long']
            if diff_p == diff_c or diff_p * diff_c > 0:
                continue
            t = diff_p / (diff_p - diff_c)
            p_date = pd.Timestamp(p_row['trade_date'])
            c_date = pd.Timestamp(c_row['trade_date'])
            x_cross = p_date + pd.Timedelta(seconds=t * (c_date - p_date).total_seconds())
            y_cross = (p_row['short'] + t * (c_row['short'] - p_row['short'])) / 1e8
            if diff_p < 0:
                golden_x.append(x_cross); golden_y.append(y_cross)
            else:
                dead_x.append(x_cross);   dead_y.append(y_cross)

        cross_label = f'MA{short_p}↑MA{long_p}'
        if golden_x:
            fig.add_trace(go.Scatter(
                x=golden_x, y=golden_y, name='골든크로스', mode='markers',
                marker=dict(symbol='triangle-up', color='#4ade80', size=9,
                            line=dict(width=1, color='#0f172a')),
                hovertemplate=f'골든크로스 ({cross_label})<br>%{{x}}<extra></extra>',
            ))
        if dead_x:
            fig.add_trace(go.Scatter(
                x=dead_x, y=dead_y, name='데드크로스', mode='markers',
                marker=dict(symbol='triangle-down', color='#f87171', size=9,
                            line=dict(width=1, color='#0f172a')),
                hovertemplate=f'데드크로스 (MA{short_p}↓MA{long_p})<br>%{{x}}<extra></extra>',
            ))

    fig.add_hline(y=0, line_color='#64748b', line_width=0.8)

    fig.update_yaxes(title_text='순매수 (억원)',
                     tickfont=dict(color=_MUTED), title_font=dict(color=_MUTED),
                     showgrid=True, gridcolor=_GRID)

    fig.update_layout(
        title='외국인 MA 시그널',
        xaxis_title='날짜',
        height=420,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    )
    return _apply_dark(fig)


def create_multiperiod_zscore_bar(zscore_row: "pd.Series") -> go.Figure:
    """7기간(5D/10D/20D/50D/100D/200D/500D) Z-Score 바차트 + ±2σ 기준선"""
    periods = ['5D', '10D', '20D', '50D', '100D', '200D', '500D']
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
