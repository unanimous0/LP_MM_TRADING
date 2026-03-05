"""
Stage 5-1: Streamlit 웹 대시보드 - 수급 메인 페이지

final_score 기반 단일 랭킹 + 드릴다운 분석.
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 등록
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from html import escape as _esc

from utils.data_loader import (
    run_analysis_pipeline_with_progress,
    get_date_range,
    get_abnormal_supply_data,
    get_market_cap_latest,
)
from utils.charts import (
    create_pattern_pie_chart,
    create_score_histogram,
    create_multiperiod_zscore_bar,
)

# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Whale Supply",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 글로벌 CSS — 위젯 테두리 가시성 + 카드 구분
# ---------------------------------------------------------------------------
st.markdown("""<style>
/* selectbox / multiselect 테두리 */
div[data-baseweb="select"] > div {
    border-color: #333 !important;
}
/* number_input / text_input 테두리 */
div[data-baseweb="input"] input,
div[data-baseweb="input"] > div {
    border-color: #333 !important;
}
/* date_input 테두리 */
[data-testid="stDateInput"] > div > div > div {
    border-color: #333 !important;
}
/* slider track 가시성 */
div[data-baseweb="slider"] div[role="slider"] {
    background: #4ade80 !important;
}
/* expander 테두리 */
[data-testid="stExpander"] {
    border-color: #222 !important;
}
</style>""", unsafe_allow_html=True)

st.title("Whale Supply")
st.caption("외국인/기관 투자자 수급 기반 종목 분석 시스템")

# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
min_date, max_date = get_date_range()
st.sidebar.markdown(f"**DB 기간**: {min_date} ~ {max_date}")

institution_weight = st.sidebar.slider(
    "기관 가중치", 0.0, 1.0, 0.3, step=0.05,
    key="w_institution_weight",
    help="""기관 수급이 외국인과 같은 방향일 때만 가중치가 반영됩니다.

[로직]
· 같은 방향(동반 매수/매도): combined = 외국인 + 기관 × weight
· 반대 방향: combined = 외국인만 (기관 무시)

[반대 방향 무시 이유]
기관이 외국인과 반대로 움직일 때 단순 합산하면 외국인의 강한 매수 신호가 희석되거나 뒤집힐 수 있습니다. 예) 외국인 +1,000억, 기관 -1,050억 → 합산 -50억(매도 신호)으로 분류되어 실제 외국인 강매수를 놓치게 됩니다. 기관의 역매매는 헤지·유동성 공급 등 외국인과 다른 목적일 수 있으므로 외국인 신호를 중심으로 해석합니다.

[값별 의미]
· 0.0 = 외국인 신호만 사용
· 0.3 = 기관 동조 시 30% 추가 반영 (기본값)
· 1.0 = 기관 동조 시 외국인과 동등하게 반영

※ 순수 외국인 관점으로 보려면 0으로 설정하세요.""",
)

_max_dt = datetime.strptime(max_date, "%Y-%m-%d")
end_date = st.sidebar.date_input(
    "기준 날짜",
    value=_max_dt,
    min_value=datetime.strptime(min_date, "%Y-%m-%d"),
    max_value=_max_dt.replace(month=12, day=31),
    help="해당 날짜 기준으로 분석합니다.",
)
end_date_str = end_date.strftime("%Y-%m-%d")

st.sidebar.divider()

direction = st.sidebar.selectbox(
    "분석 방향", ["매수 (Long)", "매도 (Short)"], index=0,
    help="매수: 외국인/기관 순매수 상위 종목 · 매도: 순매도 상위 종목",
)
_direction = 'long' if '매수' in direction else 'short'

mcap_filter = st.sidebar.selectbox(
    "시총 필터", ["전체", "시총 100위 이내", "시총 200위 이내", "시총 500위 이내"], index=0,
    help="대형주 위주로 필터링합니다.",
)

min_score_filter = st.sidebar.slider(
    "최소 종합점수", 0.0, 100.0, 60.0, step=5.0,
    help="종합점수(패턴점수 + 시그널수×5)가 이 값 이상인 종목만 표시합니다.",
)

top_n = st.sidebar.selectbox(
    "표시 종목 수", [10, 20, 30, 50, 100], index=1,
    help="수급 랭킹에 표시할 최대 종목 수",
)

# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
_prog = st.progress(0, text="분석 준비 중... 0%")
zscore_matrix, classified_df, signals_df, report_df = run_analysis_pipeline_with_progress(
    end_date=end_date_str,
    progress_bar=_prog,
    institution_weight=institution_weight,
    direction=_direction,
)

if report_df.empty:
    _prog.empty()
    st.warning("분석 데이터가 없습니다. DB를 확인하세요.")
    st.stop()

# 이상 수급 (KPI용)
_prog.progress(0.90, text="이상 수급 집계 중... 90%")
abnormal_buy = get_abnormal_supply_data(end_date=end_date_str, threshold=2.0, top_n=30, direction='buy', institution_weight=institution_weight)
abnormal_sell = get_abnormal_supply_data(end_date=end_date_str, threshold=2.0, top_n=30, direction='sell', institution_weight=institution_weight)
_prog.progress(1.0, text="완료 100%")
_prog.empty()

# ---------------------------------------------------------------------------
# final_score 계산 + 필터 + 정렬
# ---------------------------------------------------------------------------
report_df = report_df.copy()
if 'signal_count' in report_df.columns:
    report_df['final_score'] = report_df['score'] + report_df['signal_count'] * 5
else:
    report_df['final_score'] = report_df['score']

# 5D Z-Score 병합
if not classified_df.empty and '5D' in classified_df.columns:
    _z5d = classified_df[['stock_code', '5D']].drop_duplicates('stock_code')
    report_df = report_df.merge(_z5d, on='stock_code', how='left')

# 시총 데이터 병합 (필터링 전)
_mcap = get_market_cap_latest()
if not _mcap.empty:
    report_df = report_df.merge(
        _mcap[['stock_code', 'market_cap_str', 'market_cap_rank']],
        on='stock_code', how='left',
    )

# 필터 + 정렬
ranked_df = report_df[report_df['final_score'] >= min_score_filter].copy()

# 시총 필터 적용
_mcap_limits = {"시총 100위 이내": 100, "시총 200위 이내": 200, "시총 500위 이내": 500}
if mcap_filter in _mcap_limits and 'market_cap_rank' in ranked_df.columns:
    ranked_df = ranked_df[ranked_df['market_cap_rank'] <= _mcap_limits[mcap_filter]]

ranked_df = ranked_df.sort_values('final_score', ascending=False).head(top_n)

# ---------------------------------------------------------------------------
# 기준일 + KPI
# ---------------------------------------------------------------------------
_dir_label = "매수" if _direction == 'long' else "매도"
st.markdown(f"**기준일**: {end_date_str} · **방향**: {_dir_label}")

total = len(report_df)
high_score = len(report_df[(report_df['score'] >= 70) & (report_df['signal_count'] >= 2)])
signal_2plus = len(report_df[report_df['signal_count'] >= 2])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("분석 종목", f"{total}개")
col2.metric("고득점 종목", f"{high_score}개", help="점수 70+ & 시그널 2+")
col3.metric("강한 매수", f"{len(abnormal_buy)}개", help="Z-Score > 2σ")
col4.metric("강한 매도", f"{len(abnormal_sell)}개", help="Z-Score < -2σ")
col5.metric("시그널 2+", f"{signal_2plus}개", help="시그널 2개 이상 종목")

st.divider()

# ---------------------------------------------------------------------------
# 수급 TOP N 랭킹
# ---------------------------------------------------------------------------
st.subheader(f"{'순매수' if _direction == 'long' else '순매도'} 수급 TOP {min(top_n, len(ranked_df))}")
_mcap_note = " · 시총 500위 이내" if mcap_filter != "전체" else ""
st.caption(f"종합점수(패턴점수 + 시그널×5) 기준 내림차순 · 최소 {min_score_filter:.0f}점 이상 · {len(ranked_df)}개 종목{_mcap_note}")

if ranked_df.empty:
    st.info("조건에 맞는 종목이 없습니다. 사이드바에서 최소 종합점수를 낮춰보세요.")
else:
    _pat_col = 'pattern_label' if 'pattern_label' in ranked_df.columns else 'pattern'

    # 순위 컬럼 추가
    _display = ranked_df.reset_index(drop=True).copy()
    _display.insert(0, 'rank', range(1, len(_display) + 1))

    _show_cols = ['rank', 'stock_code', 'stock_name', 'sector', _pat_col,
                  'score', 'signal_count', '5D', 'final_score',
                  'market_cap_str', 'market_cap_rank']
    _show_cols = [c for c in _show_cols if c in _display.columns]

    _col_cfg = {
        'rank': st.column_config.NumberColumn('#', width='small'),
        'stock_code': st.column_config.TextColumn('종목코드'),
        'stock_name': st.column_config.TextColumn('종목명'),
        'sector': st.column_config.TextColumn('섹터'),
        'pattern': st.column_config.TextColumn('패턴'),
        'pattern_label': st.column_config.TextColumn('패턴'),
        'score': st.column_config.NumberColumn('패턴점수', format='%.1f'),
        'signal_count': st.column_config.NumberColumn('시그널', format='%d'),
        '5D': st.column_config.NumberColumn('5D Z', format='%.2f'),
        'final_score': st.column_config.ProgressColumn(
            '종합점수', min_value=0, max_value=115, format='%.1f점',
        ),
        'market_cap_str': st.column_config.TextColumn('시총'),
        'market_cap_rank': st.column_config.NumberColumn('시총순위', format='%d위'),
    }
    _col_cfg = {k: v for k, v in _col_cfg.items() if k in _show_cols}

    # 테이블 클릭 → 드릴다운 연동
    event = st.dataframe(
        _display[_show_cols],
        column_config=_col_cfg,
        use_container_width=True,
        hide_index=True,
        height=min(600, len(_display) * 40 + 40),
        on_select="rerun",
        selection_mode="single-row",
        key="ranking_table",
    )

    # 클릭된 행 → selectbox 동기화 (session_state 직접 업데이트)
    _drill_options = [
        f"#{i+1} {row['stock_name']} ({row['stock_code']}) — {row['final_score']:.1f}점"
        for i, (_, row) in enumerate(ranked_df.iterrows())
    ]
    _selected_rows = event.selection.rows if event.selection else []
    if _selected_rows:
        _drill_idx = _selected_rows[0]
        if _drill_idx < len(_drill_options):
            st.session_state['drill_select'] = _drill_options[_drill_idx]

    # ---------------------------------------------------------------------------
    # 드릴다운: 선택된 종목 분석
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("종목 드릴다운")

    _drill_sel = st.selectbox(
        "종목 선택", _drill_options, key="drill_select",
        help="테이블에서 행을 클릭하거나, 여기서 직접 선택할 수 있습니다.",
    )

    if _drill_sel:
        _drill_code = _drill_sel.split('(')[1].split(')')[0]
        _drill_row = ranked_df[ranked_df['stock_code'] == _drill_code].iloc[0]

        # 패턴 배너
        pattern = _drill_row.get('pattern', '기타')
        pattern_label = _drill_row.get('pattern_label', pattern)
        score = _drill_row.get('score', 0)
        final_score = _drill_row.get('final_score', 0)
        signal_count = int(_drill_row.get('signal_count', 0))
        signal_list = _drill_row.get('signal_list', '') or ''
        if isinstance(signal_list, list):
            signal_list = ', '.join(signal_list)

        _PATTERN_COLORS = {
            '급등형': '#f472b6',   # pink-400
            '지속형':   '#2dd4bf',   # teal-400
            '전환형':   '#fbbf24',   # amber-400
            '기타':     '#64748b',   # slate-500
        }
        pcolor = _PATTERN_COLORS.get(pattern, '#64748b')

        st.markdown(
            f'<div style="border-left:4px solid {pcolor}; padding:8px 16px; '
            f'background-color:#111111; border-radius:4px; margin:8px 0;">'
            f'<b>패턴:</b> {_esc(str(pattern_label))} &nbsp;|&nbsp; '
            f'<b>패턴점수:</b> {score:.1f} &nbsp;|&nbsp; '
            f'<b>시그널:</b> {signal_count}개 ({_esc(str(signal_list)) if signal_list else "없음"}) &nbsp;|&nbsp; '
            f'<b>종합:</b> {final_score:.1f}점'
            f'</div>',
            unsafe_allow_html=True,
        )

        # 드릴다운 메트릭 + Z-Score 바차트
        dc1, dc2 = st.columns([1, 2])

        with dc1:
            # 점수 산출 근거
            st.markdown("**점수 산출 근거**")

            _comps = {
                '최근수급 (recent)': _drill_row.get('recent', float('nan')),
                '단기이격 (short_divergence)': _drill_row.get('short_divergence', float('nan')),
                '중기이격 (mid_divergence)': _drill_row.get('mid_divergence', float('nan')),
                '장기이격 (long_divergence)': _drill_row.get('long_divergence', float('nan')),
                '가중평균 (weighted)': _drill_row.get('weighted', float('nan')),
                '단순평균 (average)': _drill_row.get('average', float('nan')),
            }
            for label, val in _comps.items():
                if pd.notna(val):
                    _c = '#4ade80' if val >= 0 else '#f87171'
                    st.markdown(
                        f'<span style="color:#94a3b8;font-size:13px;">{label}:</span> '
                        f'<span style="color:{_c};font-weight:600;">{val:+.2f}</span>',
                        unsafe_allow_html=True,
                    )

            tc = _drill_row.get('temporal_consistency', float('nan'))
            if pd.notna(tc):
                st.markdown(
                    f'<span style="color:#94a3b8;font-size:13px;">기간순서 일관성 (tc):</span> '
                    f'<span style="font-weight:600;">{tc:.2f}</span>',
                    unsafe_allow_html=True,
                )

            sub_type = _drill_row.get('sub_type', None)
            if sub_type and not (isinstance(sub_type, float) and pd.isna(sub_type)):
                st.markdown(
                    f'<span style="color:#94a3b8;font-size:13px;">복합패턴:</span> '
                    f'<span style="font-weight:600;">{_esc(str(sub_type))}</span>',
                    unsafe_allow_html=True,
                )

            # 이상수급 해당 여부
            _is_abnormal_buy = (
                not abnormal_buy.empty and _drill_code in abnormal_buy['stock_code'].values
            )
            _is_abnormal_sell = (
                not abnormal_sell.empty and _drill_code in abnormal_sell['stock_code'].values
            )
            if _is_abnormal_buy:
                st.success("⚡ 이상 수급 매수 (Z > 2σ)")
            elif _is_abnormal_sell:
                st.error("⚡ 이상 수급 매도 (Z < -2σ)")

            # 종목 상세 링크
            if st.button("📋 종목 상세 보기 →", key="drill_to_detail"):
                st.session_state['heatmap_selected_code'] = _drill_code
                st.switch_page("pages/5_📋_종목상세.py")

        with dc2:
            # 멀티기간 Z-Score 바차트
            if not classified_df.empty:
                _stock_z = classified_df[classified_df['stock_code'] == _drill_code]
                if not _stock_z.empty:
                    fig_bar = create_multiperiod_zscore_bar(_stock_z.iloc[0])
                    st.plotly_chart(fig_bar, width="stretch", theme=None)
                else:
                    st.info("Z-Score 데이터가 없습니다.")
            else:
                st.info("Z-Score 데이터가 없습니다.")

st.divider()

# ---------------------------------------------------------------------------
# 패턴 분석 요약 (2열)
# ---------------------------------------------------------------------------
st.subheader("패턴 분석 요약")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_pie = create_pattern_pie_chart(report_df)
    st.plotly_chart(fig_pie, width="stretch", theme=None)

with chart_col2:
    fig_hist = create_score_histogram(report_df)
    st.plotly_chart(fig_hist, width="stretch", theme=None)
