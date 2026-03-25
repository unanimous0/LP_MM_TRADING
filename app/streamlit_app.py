"""
Stage 5-1: Streamlit 웹 대시보드 - 수급 메인 페이지

final_score 기반 단일 랭킹 + 드릴다운 분석 + 이상수급 + 당일순위 + 고득점변동.
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
from typing import Dict
from html import escape as _esc

from utils.data_loader import (
    run_analysis_pipeline_with_progress,
    get_date_range,
    get_abnormal_supply_data,
    get_market_cap_latest,
    get_score_reference_distributions,
    rescale_scores,
    get_today_supply_ranking,
    snapshot_scores,
    get_score_change_alerts,
    is_in_watchlist, add_to_watchlist, remove_from_watchlist,
)
from utils.ui_constants import INSTITUTION_WEIGHT_HELP, WIDGET_BORDER_CSS
from utils.charts import (
    create_pattern_pie_chart,
    create_score_histogram,
    create_multiperiod_zscore_bar,
    create_abnormal_supply_chart,
    create_supply_ranking_chart,
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
# 글로벌 CSS — 위젯 테두리 가시성 + 카드 구분 + 매수/매도 섹션 색상
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
/* 매수 섹션 (green) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(
    [style*="4ade80"]
) { border-color: #4ade80 !important; }
/* 매도 섹션 (red) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(
    [style*="f87171"]
) { border-color: #f87171 !important; }
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
    help=INSTITUTION_WEIGHT_HELP,
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
    "시총 필터", ["전체", "시총 100위 이내", "시총 200위 이내", "시총 300위 이내", "시총 500위 이내"], index=4,
    help="대형주 위주로 필터링합니다.",
)

st.sidebar.divider()

min_score_filter = st.sidebar.slider(
    "최소 종합점수", 0.0, 100.0, 60.0, step=5.0,
    help="종합점수(패턴점수 + 시그널수×2)가 이 값 이상인 종목만 표시합니다.",
)

top_n = st.sidebar.selectbox(
    "표시 종목 수", [10, 20, 30, 50, 100], index=3,
    help="수급 랭킹에 표시할 최대 종목 수",
)

# 고급 설정 (하단)
st.sidebar.divider()
_rescale_on = st.sidebar.checkbox(
    "점수 보정", value=True,
    help="시총 필터 범위 종목의 최근 N거래일 점수 분포 기준으로 0~100 스케일을 재조정합니다. "
         "대형주의 점수가 낮게 나오는 문제를 해결합니다.",
)
_rescale_lookback = st.sidebar.number_input(
    "점수 보정 표본 기간", min_value=20, max_value=240, value=120, step=20,
    help="점수 보정 전용: 최근 N거래일의 점수 분포로 백분위 변환합니다.",
    disabled=not _rescale_on,
)
z_score_window = st.sidebar.slider(
    "이상수급 Z-Score 기간", min_value=20, max_value=240, value=60, step=10,
    help="이상수급 탭 전용: 이상 수급 판단 시 평균/표준편차 계산에 사용하는 과거 거래일 수.",
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

# 이상 수급 (KPI용 + 이상수급 탭용)
_prog.progress(0.85, text="이상 수급 집계 중... 85%")
abnormal_buy = get_abnormal_supply_data(
    end_date=end_date_str, threshold=2.0, top_n=30, direction='buy',
    institution_weight=institution_weight, z_score_window=z_score_window,
)
abnormal_sell = get_abnormal_supply_data(
    end_date=end_date_str, threshold=2.0, top_n=30, direction='sell',
    institution_weight=institution_weight, z_score_window=z_score_window,
)

# 당일 수급 순위
_prog.progress(0.92, text="당일 수급 순위 조회 중... 92%")
supply_ranking = get_today_supply_ranking()

_prog.progress(1.0, text="완료 100%")
_prog.empty()

# 고득점 변동 스냅샷 저장 (세션당 1회)
if not st.session_state.get('main_snapshot_done'):
    try:
        _, _latest_date = get_date_range()
        snapshot_scores(report_df, _latest_date)
        st.session_state['main_snapshot_done'] = True
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 기본 final_score (보정 전 raw) + 시총/Z-Score 병합
# ---------------------------------------------------------------------------
report_df = report_df.copy()
report_df['final_score'] = report_df['score'] + report_df.get('signal_count', 0) * 2

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

# ---------------------------------------------------------------------------
# 점수 보정 — Precomputer 1회 → 체급별 분포 (캐시)
# ---------------------------------------------------------------------------
_ref_dists: Dict = {}
if _rescale_on:
    with st.spinner("점수 보정 기준 계산 중... (첫 호출 시 ~20초, 이후 캐시)"):
        _ref_dists = get_score_reference_distributions(
            end_date=end_date_str,
            lookback_days=_rescale_lookback,
            institution_weight=institution_weight,
            direction=_direction,
        )

# raw 보존 (tier TOP에서 체급별 독립 보정용)
_report_raw = report_df

# 상세 랭킹용 보정: 시총 필터에 맞는 분포 사용
_mcap_to_ref_key = {
    "시총 100위 이내": "top_100", "시총 200위 이내": "top_200",
    "시총 300위 이내": "top_300", "시총 500위 이내": "all",
}
_ref_key = _mcap_to_ref_key.get(mcap_filter, "all")
_ref_list = _ref_dists.get(_ref_key, [])
if _ref_list:
    report_df = rescale_scores(report_df, _ref_list)

# 필터 + 정렬
_mcap_limits = {"시총 100위 이내": 100, "시총 200위 이내": 200, "시총 300위 이내": 300, "시총 500위 이내": 500}
ranked_df = report_df.copy()
if mcap_filter in _mcap_limits and 'market_cap_rank' in ranked_df.columns:
    ranked_df = ranked_df[ranked_df['market_cap_rank'] <= _mcap_limits[mcap_filter]]
ranked_df = ranked_df[ranked_df['final_score'] >= min_score_filter]
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

# ===========================================================================
# 4탭 구조: 수급 TOP / 이상수급 / 당일순위 / 고득점변동
# ===========================================================================
tab_top, tab_abnormal, tab_ranking, tab_alerts = st.tabs([
    "중단기 수급",
    "오늘 이상수급",
    "당일 순위",
    "고득점 변동",
])

# ─── 탭 1: 수급 TOP (기존 메인 콘텐츠) ──────────────────────────────────────
with tab_top:
    _dir_top_label = "순매수" if _direction == 'long' else "순매도"

    # 시총 구간별 수급 TOP
    st.subheader(f"시총 구간별 {_dir_top_label} TOP")
    st.caption("같은 체급 안에서 종합점수 기준 상위 종목")

    _TIER_TOP_N = 10
    _pat_col_tier = 'pattern_label' if 'pattern_label' in _report_raw.columns else 'pattern'

    _tier_configs = [
        ("대형주", "시총 100위 이내", 1, 100, "large"),
        ("중형주", "101~300위", 101, 300, "mid"),
        ("소형주", "301위 이하", 301, None, "small"),
    ]

    _tier_cols = st.columns(len(_tier_configs))
    for _tc, (_tier_name, _tier_desc, _rank_min, _rank_max, _tier_ref_key) in zip(_tier_cols, _tier_configs):
        with _tc:
            if 'market_cap_rank' in _report_raw.columns:
                _tier_df = _report_raw[_report_raw['market_cap_rank'] >= _rank_min].copy()
                if _rank_max is not None:
                    _tier_df = _tier_df[_tier_df['market_cap_rank'] <= _rank_max]
            else:
                _tier_df = _report_raw.copy()

            _tier_ref = _ref_dists.get(_tier_ref_key, [])
            if _tier_ref and not _tier_df.empty:
                _tier_df = rescale_scores(_tier_df, _tier_ref)

            _tier_df = _tier_df.sort_values('final_score', ascending=False).head(_TIER_TOP_N)

            st.markdown(f"**{_tier_name}** <span style='color:#94a3b8;font-size:13px;'>({_tier_desc})</span>",
                        unsafe_allow_html=True)

            if _tier_df.empty:
                st.caption("해당 조건 종목 없음")
            else:
                _rows_html = []
                for _rank_i, (_, _r) in enumerate(_tier_df.iterrows(), 1):
                    _s = _r['final_score']
                    _sc = '#4ade80' if _s >= 70 else ('#fbbf24' if _s >= 60 else '#94a3b8')
                    _pat = str(_r.get(_pat_col_tier, ''))
                    _name = str(_r.get('stock_name', ''))
                    _rows_html.append(
                        f'<tr>'
                        f'<td style="color:#64748b;width:24px;">{_rank_i}</td>'
                        f'<td style="max-width:100px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{_esc(_name)}</td>'
                        f'<td style="color:{_sc};font-weight:600;text-align:right;">{_s:.0f}</td>'
                        f'<td style="color:#94a3b8;font-size:12px;">{_esc(_pat)}</td>'
                        f'</tr>'
                    )
                st.markdown(
                    '<table style="width:100%;border-collapse:collapse;font-size:13px;line-height:1.8;">'
                    + ''.join(_rows_html)
                    + '</table>',
                    unsafe_allow_html=True,
                )

    st.divider()

    st.subheader(f"{_dir_top_label} 수급 상세 랭킹")
    _mcap_note = f" · {mcap_filter}" if mcap_filter != "전체" else ""
    st.caption(
        f"종합점수 = 수급강도(35%) + 수급추세(20%) + 종합수급(25%) + 수급지속(20%) + 시그널×2  ·  "
        f"최소 {min_score_filter:.0f}점 · {len(ranked_df)}개{_mcap_note}"
    )

    if ranked_df.empty:
        st.info("조건에 맞는 종목이 없습니다. 사이드바에서 최소 종합점수를 낮춰보세요.")
    else:
        _pat_col = 'pattern_label' if 'pattern_label' in ranked_df.columns else 'pattern'

        _display = ranked_df.reset_index(drop=True).copy()
        _display.insert(0, 'rank', range(1, len(_display) + 1))

        _show_cols = ['rank', 'stock_code', 'stock_name', 'sector', _pat_col,
                      'signal_count', '5D', 'final_score',
                      'market_cap_str', 'market_cap_rank']
        _show_cols = [c for c in _show_cols if c in _display.columns]

        _col_cfg = {
            'rank': st.column_config.NumberColumn('#', width='small'),
            'stock_code': st.column_config.TextColumn('종목코드'),
            'stock_name': st.column_config.TextColumn('종목명'),
            'sector': st.column_config.TextColumn('섹터'),
            'pattern': st.column_config.TextColumn('패턴'),
            'pattern_label': st.column_config.TextColumn('패턴'),
            'signal_count': st.column_config.NumberColumn('시그널', format='%d'),
            '5D': st.column_config.NumberColumn('5D Z', format='%.2f'),
            'final_score': st.column_config.ProgressColumn(
                '점수', min_value=0, max_value=100, format='%.1f점',
            ),
            'market_cap_str': st.column_config.TextColumn('시총'),
            'market_cap_rank': st.column_config.NumberColumn('시총순위', format='%d위'),
        }
        _col_cfg = {k: v for k, v in _col_cfg.items() if k in _show_cols}

        event = st.dataframe(
            _display[_show_cols],
            column_config=_col_cfg,
            use_container_width=True,
            hide_index=True,
            height=min(20 * 38 + 40, len(_display) * 38 + 40),
            on_select="rerun",
            selection_mode="single-row",
            key="ranking_table",
        )

        _drill_options = [
            f"#{i+1} {row['stock_name']} ({row['stock_code']}) — {row['final_score']:.1f}점"
            for i, (_, row) in enumerate(ranked_df.iterrows())
        ]
        _selected_rows = event.selection.rows if event.selection else []
        if _selected_rows:
            _drill_idx = _selected_rows[0]
            if _drill_idx < len(_drill_options):
                st.session_state['drill_select'] = _drill_options[_drill_idx]

        # 드릴다운: 선택된 종목 분석
        st.divider()
        st.subheader("종목 드릴다운")

        _drill_sel = st.selectbox(
            "종목 선택", _drill_options, key="drill_select",
            help="테이블에서 행을 클릭하거나, 여기서 직접 선택할 수 있습니다.",
        )

        if _drill_sel:
            _drill_code = _drill_sel.split('(')[1].split(')')[0]
            _drill_row = ranked_df[ranked_df['stock_code'] == _drill_code].iloc[0]

            pattern = _drill_row.get('pattern', '기타')
            pattern_label = _drill_row.get('pattern_label', pattern)
            score = _drill_row.get('score', 0)
            final_score = _drill_row.get('final_score', 0)
            signal_count = int(_drill_row.get('signal_count', 0))
            signal_list = _drill_row.get('signal_list', '') or ''
            if isinstance(signal_list, list):
                signal_list = ', '.join(signal_list)

            _PATTERN_COLORS = {
                '급등형': '#f472b6',
                '지속형':   '#2dd4bf',
                '전환형':   '#fbbf24',
                '기타':     '#64748b',
            }
            pcolor = _PATTERN_COLORS.get(pattern, '#64748b')

            st.markdown(
                f'<div style="border-left:4px solid {pcolor}; padding:8px 16px; '
                f'background-color:#111111; border-radius:4px; margin:8px 0;">'
                f'<b>패턴:</b> {_esc(str(pattern_label))} &nbsp;|&nbsp; '
                f'<b>점수:</b> {final_score:.1f}점 &nbsp;|&nbsp; '
                f'<b>시그널:</b> {signal_count}개 ({_esc(str(signal_list)) if signal_list else "없음"})'
                f'</div>',
                unsafe_allow_html=True,
            )

            dc1, dc2 = st.columns([1, 2])

            with dc1:
                st.markdown("**점수 산출 근거 (v2)**")

                _drill_direction = _drill_row.get('direction', 'long')
                _sp_col = 'supply_persistence_long' if _drill_direction == 'long' else 'supply_persistence_short'
                _sp_val = _drill_row.get(_sp_col, 0.0)
                if pd.isna(_sp_val):
                    _sp_val = 0.0
                _sp_norm = min(_sp_val * 2.0, 3.0)

                _comps = {
                    '수급강도 ×0.35': _drill_row.get('recent', float('nan')),
                    '수급추세 ×0.20': _drill_row.get('long_divergence', float('nan')),
                    '종합수급 ×0.25': _drill_row.get('weighted', float('nan')),
                }
                for label, val in _comps.items():
                    if pd.notna(val):
                        _c = '#4ade80' if val >= 0 else '#f87171'
                        st.markdown(
                            f'<span style="color:#94a3b8;font-size:13px;">{label}:</span> '
                            f'<span style="color:{_c};font-weight:600;">{val:+.2f}</span>',
                            unsafe_allow_html=True,
                        )
                _sp_c = '#4ade80' if _sp_norm > 0 else '#94a3b8'
                st.markdown(
                    f'<span style="color:#94a3b8;font-size:13px;">수급지속 ×0.20:</span> '
                    f'<span style="color:{_sp_c};font-weight:600;">{_sp_norm:+.2f}</span> '
                    f'<span style="color:#64748b;font-size:11px;">(raw {_sp_val:.3f})</span>',
                    unsafe_allow_html=True,
                )

                sub_type = _drill_row.get('sub_type', None)
                if sub_type and not (isinstance(sub_type, float) and pd.isna(sub_type)):
                    st.markdown(
                        f'<span style="color:#94a3b8;font-size:13px;">복합패턴:</span> '
                        f'<span style="font-weight:600;">{_esc(str(sub_type))}</span>',
                        unsafe_allow_html=True,
                    )

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

                _btn_cols = st.columns(2)
                if _btn_cols[0].button("📋 종목 상세 →", key="drill_to_detail", use_container_width=True):
                    st.session_state['heatmap_selected_code'] = _drill_code
                    st.switch_page("pages/5_📋_종목상세.py")
                _drill_name = _drill_row.get('stock_name', _drill_code)
                _drill_sector = _drill_row.get('sector', '')
                if is_in_watchlist(_drill_code):
                    if _btn_cols[1].button("⭐ 관심 해제", key="drill_wl", use_container_width=True):
                        remove_from_watchlist(_drill_code)
                        st.toast(f"'{_drill_name}' 관심종목에서 제거", icon="🗑️")
                        st.rerun()
                else:
                    if _btn_cols[1].button("☆ 관심 추가", key="drill_wl", use_container_width=True):
                        add_to_watchlist(_drill_code, str(_drill_name), str(_drill_sector))
                        st.toast(f"'{_drill_name}' 관심종목에 추가", icon="⭐")
                        st.rerun()

            with dc2:
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

    # 패턴 분석 요약 (2열)
    st.subheader("패턴 분석 요약")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_pie = create_pattern_pie_chart(report_df)
        st.plotly_chart(fig_pie, width="stretch", theme=None)

    with chart_col2:
        fig_hist = create_score_histogram(report_df)
        st.plotly_chart(fig_hist, width="stretch", theme=None)

# ─── 탭 2: 이상수급 (Z > 2σ) ────────────────────────────────────────────────
with tab_abnormal:
    st.caption(f"최근 {z_score_window}거래일 평균 수급 대비 2표준편차 이상 벗어난 종목")
    with st.expander("산출 방식 보기"):
        _w = z_score_window
        _iw = institution_weight
        _iw_pct = int(_iw * 100)
        st.markdown(
            "**1단계: 수급 강도 (Sff)** — 순매수금액을 유통시가총액으로 나눠 종목 간 비교 가능하게 정규화\n\n"
            "$$\\text{Sff} = \\frac{\\text{순매수금액}}{\\text{유통주식수} \\times \\text{종가}}$$\n\n"
            f"**2단계: 외국인 중심 합산** — 외국인 수급을 주(主)로, 기관은 동반 매수 시에만 {_iw_pct}% 반영\n\n"
            "$$\\text{Combined} = \\begin{cases}"
            f"\\text{{Foreign}} + \\text{{Institution}} \\times {_iw} & "
            "\\text{(같은 방향)} \\\\"
            "\\text{Foreign} & \\text{(반대 방향)}"
            "\\end{cases}$$\n\n"
            f"**3단계: Z-Score** — 최근 {_w}거래일 이동평균(μ)·표준편차(σ) 기준 오늘의 이탈도\n\n"
            f"$$Z = \\frac{{\\text{{오늘 Sff}} - \\mu_{{{_w}}}}}{{\\sigma_{{{_w}}}}}$$\n\n"
            f"Z > 2 이면 과거 {_w}일 대비 상위 ~2.3% 수준의 이례적 매수, Z < -2 이면 이례적 매도\n\n"
            "---\n"
            f"**사이드바에서 조정 가능한 파라미터**: "
            f"기관 가중치(현재 {_iw}) — 0이면 외국인만, 1이면 외국인·기관 동등 반영 / "
            f"Z-Score 기준 기간(현재 {_w}일) — 짧으면 최근 추세에 민감, 길면 장기 평균 기준\n\n"
            "---\n"
            f"**외국인 Z ≠ 종합 Z인 이유**: 외국인·기관·종합 Z-Score는 각각 **자기 Sff 시리즈의 {_w}일 μ/σ**로 독립 계산됩니다. "
            "오늘 기관이 반대 방향이라 종합 Sff = 외국인 Sff여도, "
            f"과거 {_w}일 중 동반 매수였던 날에는 종합 Sff에 기관×{_iw}이 포함되어 있어 μ·σ가 다릅니다. "
            "같은 오늘 값을 다른 기준으로 나누므로 Z-Score가 달라집니다."
        )

    buy_col, sell_col = st.columns(2)

    with buy_col:
        with st.container(border=True):
            st.markdown('<div style="color: #4ade80; font-weight: 600; font-size: 0.95rem;">'
                        '강한 매수 수급</div>', unsafe_allow_html=True)
            if abnormal_buy.empty:
                st.info("오늘 강한 매수 수급 종목이 없습니다.")
            else:
                fig_buy = create_abnormal_supply_chart(abnormal_buy.head(10), direction='buy')
                st.plotly_chart(fig_buy, width="stretch", theme=None)

                buy_display = abnormal_buy[
                    ['stock_name', 'sector', 'foreign_zscore', 'institution_zscore', 'combined_zscore']
                ].reset_index(drop=True)
                st.dataframe(
                    buy_display,
                    column_config={
                        'stock_name': st.column_config.TextColumn('종목명'),
                        'sector': st.column_config.TextColumn('섹터'),
                        'foreign_zscore': st.column_config.NumberColumn('외국인 Z', format='%.2f'),
                        'institution_zscore': st.column_config.NumberColumn('기관 Z', format='%.2f'),
                        'combined_zscore': st.column_config.NumberColumn('종합 Z', format='%.2f'),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

    with sell_col:
        with st.container(border=True):
            st.markdown('<div style="color: #f87171; font-weight: 600; font-size: 0.95rem;">'
                        '강한 매도 수급</div>', unsafe_allow_html=True)
            if abnormal_sell.empty:
                st.info("오늘 강한 매도 수급 종목이 없습니다.")
            else:
                fig_sell = create_abnormal_supply_chart(abnormal_sell.head(10), direction='sell')
                st.plotly_chart(fig_sell, width="stretch", theme=None)

                sell_display = abnormal_sell[
                    ['stock_name', 'sector', 'foreign_zscore', 'institution_zscore', 'combined_zscore']
                ].reset_index(drop=True)
                st.dataframe(
                    sell_display,
                    column_config={
                        'stock_name': st.column_config.TextColumn('종목명'),
                        'sector': st.column_config.TextColumn('섹터'),
                        'foreign_zscore': st.column_config.NumberColumn('외국인 Z', format='%.2f'),
                        'institution_zscore': st.column_config.NumberColumn('기관 Z', format='%.2f'),
                        'combined_zscore': st.column_config.NumberColumn('종합 Z', format='%.2f'),
                    },
                    use_container_width=True,
                    hide_index=True,
                )

# ─── 탭 3: 당일 수급 순위 ────────────────────────────────────────────────────
with tab_ranking:
    st.caption("당일 외국인/기관 순매수·순매도 금액 상위 종목 (원시 금액 기준, 정규화 미적용)")

    if supply_ranking.empty:
        st.info("당일 수급 데이터가 없습니다.")
    else:
        def _fmt_col(df, col):
            """금액 컬럼을 쉼표 포맷 문자열로 변환한 DataFrame 반환"""
            out = df[['stock_name', 'sector']].copy()
            out['순매수(원)'] = df[col].apply(lambda v: f'{int(v):,}' if pd.notna(v) else '-')
            return out.reset_index(drop=True)

        _foreign_buy = supply_ranking.nlargest(50, 'foreign_net_amount')
        _foreign_sell = supply_ranking.nsmallest(50, 'foreign_net_amount')
        _inst_buy = supply_ranking.nlargest(50, 'institution_net_amount')
        _inst_sell = supply_ranking.nsmallest(50, 'institution_net_amount')

        # --- 순매수 상위 ---
        st.markdown("##### 순매수 상위")
        fb_col, ib_col = st.columns(2)

        with fb_col:
            with st.container(border=True):
                st.markdown('<div style="color: #4ade80; font-weight: 600; font-size: 0.95rem;">'
                            '외국인 순매수</div>', unsafe_allow_html=True)
                fig = create_supply_ranking_chart(
                    _foreign_buy, 'foreign_net_amount', '외국인 순매수 Top 10', top_n=10,
                )
                st.plotly_chart(fig, width="stretch", theme=None)
                st.dataframe(
                    _fmt_col(_foreign_buy, 'foreign_net_amount'),
                    use_container_width=True,
                    hide_index=True,
                )

        with ib_col:
            with st.container(border=True):
                st.markdown('<div style="color: #4ade80; font-weight: 600; font-size: 0.95rem;">'
                            '기관 순매수</div>', unsafe_allow_html=True)
                fig = create_supply_ranking_chart(
                    _inst_buy, 'institution_net_amount', '기관 순매수 Top 10', top_n=10,
                )
                st.plotly_chart(fig, width="stretch", theme=None)
                st.dataframe(
                    _fmt_col(_inst_buy, 'institution_net_amount'),
                    use_container_width=True,
                    hide_index=True,
                )

        # --- 순매도 상위 ---
        st.markdown("##### 순매도 상위")
        fs_col, is_col = st.columns(2)

        with fs_col:
            with st.container(border=True):
                st.markdown('<div style="color: #f87171; font-weight: 600; font-size: 0.95rem;">'
                            '외국인 순매도</div>', unsafe_allow_html=True)
                fig = create_supply_ranking_chart(
                    _foreign_sell, 'foreign_net_amount', '외국인 순매도 Top 10', top_n=10,
                )
                st.plotly_chart(fig, width="stretch", theme=None)
                st.dataframe(
                    _fmt_col(_foreign_sell, 'foreign_net_amount'),
                    use_container_width=True,
                    hide_index=True,
                )

        with is_col:
            with st.container(border=True):
                st.markdown('<div style="color: #f87171; font-weight: 600; font-size: 0.95rem;">'
                            '기관 순매도</div>', unsafe_allow_html=True)
                fig = create_supply_ranking_chart(
                    _inst_sell, 'institution_net_amount', '기관 순매도 Top 10', top_n=10,
                )
                st.plotly_chart(fig, width="stretch", theme=None)
                st.dataframe(
                    _fmt_col(_inst_sell, 'institution_net_amount'),
                    use_container_width=True,
                    hide_index=True,
                )

# ─── 탭 4: 고득점 변동 알림 ──────────────────────────────────────────────────
with tab_alerts:
    st.caption(f"점수 {70}점 이상 종목의 신규 진입 / 급등(+5점) / 급락(-5점) / 이탈 이벤트")

    alerts_df = get_score_change_alerts(limit=100)
    if alerts_df.empty:
        st.info("기록된 변동 알림이 없습니다. 페이지를 다시 로드하면 오늘 분석 결과와 이전 분석을 비교합니다.")
    else:
        _ct_labels = {
            'new_entry':  '🆕 신규 진입',
            'score_up':   '📈 급등',
            'score_down': '📉 급락',
            'exit':       '🚪 이탈',
        }

        _ct_all = list(_ct_labels.keys())
        _ct_sel = st.multiselect(
            "이벤트 유형 필터",
            options=_ct_all,
            default=_ct_all,
            format_func=lambda x: _ct_labels.get(x, x),
            key="alert_type_filter",
        )

        filtered_alerts = alerts_df[alerts_df['change_type'].isin(_ct_sel)] if _ct_sel else alerts_df

        _al_cols = ['analysis_date', 'change_type', 'stock_code', 'stock_name',
                    'sector', 'pattern', 'score', 'prev_score', 'signal_count']
        _al_cols = [c for c in _al_cols if c in filtered_alerts.columns]

        _al_cfg = {
            'analysis_date': st.column_config.TextColumn('분석일'),
            'change_type':   st.column_config.TextColumn('변동 유형'),
            'stock_code':    st.column_config.TextColumn('종목코드'),
            'stock_name':    st.column_config.TextColumn('종목명'),
            'sector':        st.column_config.TextColumn('섹터'),
            'pattern':       st.column_config.TextColumn('패턴'),
            'score':         st.column_config.NumberColumn('현재 점수', format='%.1f'),
            'prev_score':    st.column_config.NumberColumn('이전 점수', format='%.1f'),
            'signal_count':  st.column_config.NumberColumn('시그널', format='%d'),
        }
        _al_cfg = {k: v for k, v in _al_cfg.items() if k in _al_cols}

        _disp_alerts = filtered_alerts[_al_cols].copy()
        _disp_alerts['change_type'] = _disp_alerts['change_type'].map(
            lambda x: _ct_labels.get(x, x)
        )

        st.dataframe(
            _disp_alerts.reset_index(drop=True),
            column_config=_al_cfg,
            use_container_width=True,
            hide_index=True,
            height=min(500, len(_disp_alerts) * 40 + 40),
        )
        st.caption(f"총 {len(_disp_alerts)}건 (최근 100건)")
