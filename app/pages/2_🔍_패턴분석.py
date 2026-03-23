"""
종목 탐색 페이지 - 종합점수 기준 종목 순위 + 섹터 분석 + 누적 수급

사이드바: 기관 가중치 / 기준 날짜 / 섹터 필터 / 최소 점수 / 누적 수급 설정
4개 탭: 점수 순위, 섹터 분석, 누적 수급, 패턴 가이드
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from html import escape as _esc

from utils.ui_constants import INSTITUTION_WEIGHT_HELP
from utils.data_loader import (
    run_analysis_pipeline_with_progress, get_sectors, get_date_range,
    get_stock_list, get_db_connection,
    get_watchlist, add_to_watchlist, remove_from_watchlist,
    get_market_cap_latest,
    get_score_reference_distributions, rescale_scores,
    get_cumulative_score_ranking,
    get_cumulative_sff_ranking,
)
from utils.charts import (
    create_sector_avg_score_chart,
    create_sector_concentration_chart,
    create_sector_treemap,
    create_cumulative_bar_chart,
)
from src.analyzer.integrated_report import IntegratedReport

st.set_page_config(page_title="종목 탐색", page_icon="🔍", layout="wide")
st.markdown('<style>div[data-baseweb="select"]>div{border-color:#333!important}div[data-baseweb="input"] input,div[data-baseweb="input"]>div{border-color:#333!important}[data-testid="stDateInput"]>div>div>div{border-color:#333!important}[data-testid="stExpander"]{border-color:#222!important}</style>', unsafe_allow_html=True)
st.title("종목 탐색")


# ---------------------------------------------------------------------------
# 점수 산출 툴팁 HTML 생성
# ---------------------------------------------------------------------------
def _build_tooltip_html(row, zscore_row=None):
    """종목의 점수 산출 과정을 HTML 툴팁 내용으로 생성 (v2: 4구성요소)."""
    pattern = row.get('pattern', '기타')
    sub_type = row.get('sub_type', None)
    if pd.isna(sub_type) if not isinstance(sub_type, str) else False:
        sub_type = None
    score = float(row.get('score', 0))
    pat_label = row.get('pattern_label', pattern)

    recent = float(row.get('recent', 0))
    long_divergence = float(row.get('long_divergence', 0))
    weighted = float(row.get('weighted', 0))
    persistence = float(row.get('persistence', 0))

    # supply_persistence (방향별)
    direction = row.get('direction', 'long')
    sp_col = 'supply_persistence_long' if direction == 'long' else 'supply_persistence_short'
    sp_val = float(row.get(sp_col, 0))
    sp_normalized = min(sp_val * 2.0, 3.0)

    sig_count = int(row.get('signal_count', 0))
    sig_list = str(row.get('signal_list', '') or '')
    final_score = score + sig_count * 2

    # Z-Score 값 추출
    zvals = {}
    periods = ['5D', '10D', '20D', '50D', '100D', '200D', '500D']
    if zscore_row is not None:
        for p in periods:
            if p in zscore_row.index and pd.notna(zscore_row[p]):
                zvals[p] = float(zscore_row[p])

    h = []
    h.append(f'<div class="tt-title">{_esc(str(pat_label))}</div>')

    # ── Z-Score 그리드 ──
    if zvals:
        grid = '<div class="zg">'
        for p in periods:
            if p in zvals:
                v = zvals[p]
                cls = 'zp' if v >= 0 else 'zn'
                grid += f'<div class="zg-cell"><div class="zg-period">{p}</div><div class="{cls}">{v:+.2f}</div></div>'
        grid += '</div>'
        h.append(grid)

    # ── 패턴 근거 ──
    h.append('<div class="tt-section">패턴 분류 근거</div>')
    tc = row.get('temporal_consistency', 0.5)
    tc = tc if pd.notna(tc) else 0.5
    if pattern == '급등형':
        h.append(
            f'<div class="tt-line">장기이격(<b>{long_divergence:+.2f}</b>)&gt;1.0 '
            f'&amp; 최근수급(<b>{recent:+.2f}</b>)&gt;0.5 '
            f'&amp; tc(<b>{tc:.2f}</b>)≥0.5</div>')
    elif pattern == '지속형':
        h.append(
            f'<div class="tt-line">가중평균(<b>{weighted:+.2f}</b>)&gt;0.8 '
            f'&amp; 지속성(<b>{persistence:.2f}</b>)&gt;0.7</div>')
    elif pattern == '전환형':
        h.append(
            f'<div class="tt-line">가중평균(<b>{weighted:+.2f}</b>)&gt;0.5 '
            f'&amp; 장기이격(<b>{long_divergence:+.2f}</b>)&lt;0</div>')
    else:
        h.append('<div class="tt-line">급등·지속·전환 조건 미충족 → 기타</div>')

    if sub_type:
        h.append(f'<div class="tt-line">복합: <b>{_esc(sub_type)}</b></div>')

    # ── 점수 산출 (v2: 4구성요소) ──
    h.append('<div class="tt-section">점수 산출 (v2)</div>')

    z5d = zvals.get('5D')
    z20d = zvals.get('20D')
    z100d = zvals.get('100D')
    z200d = zvals.get('200D')

    # 4개 카드
    cards = []

    # 수급강도 (recent)
    fml = f'(5D {z5d:+.2f} + 20D {z20d:+.2f}) / 2' if z5d is not None and z20d is not None else ''
    cards.append(
        f'<div class="cc"><div class="cc-h">수급강도 <span class="cc-w">×0.35</span></div>'
        f'{"<div class=cc-f>" + fml + "</div>" if fml else ""}'
        f'<div class="cc-v"><b>{recent:+.2f}</b> → {recent*0.35:+.3f}</div></div>'
    )

    # 수급추세 (long_divergence)
    fml2 = ''
    if z5d is not None:
        for lp, lv in [('200D', z200d), ('100D', z100d)]:
            if lv is not None:
                fml2 = f'5D {z5d:+.2f} − {lp} {lv:+.2f}'
                break
    cards.append(
        f'<div class="cc"><div class="cc-h">수급추세 <span class="cc-w">×0.20</span></div>'
        f'{"<div class=cc-f>" + fml2 + "</div>" if fml2 else ""}'
        f'<div class="cc-v"><b>{long_divergence:+.2f}</b> → {long_divergence*0.20:+.3f}</div></div>'
    )

    # 종합수급 (weighted)
    cards.append(
        f'<div class="cc"><div class="cc-h">종합수급 <span class="cc-w">×0.25</span></div>'
        f'<div class="cc-f">가중 평균 (최근 높은 비중)</div>'
        f'<div class="cc-v"><b>{weighted:+.2f}</b> → {weighted*0.25:+.3f}</div></div>'
    )

    # 수급지속 (supply_persistence)
    cards.append(
        f'<div class="cc"><div class="cc-h">수급지속 <span class="cc-w">×0.20</span></div>'
        f'<div class="cc-f">평균Sff×√연속일 = {sp_val:.3f} → ×2 cap3</div>'
        f'<div class="cc-v"><b>{sp_normalized:+.2f}</b> → {sp_normalized*0.20:+.3f}</div></div>'
    )

    h.append(f'<div class="cc-grid">{"".join(cards)}</div>')

    weighted_sum = recent * 0.35 + long_divergence * 0.20 + weighted * 0.25 + sp_normalized * 0.20
    base_score = float(np.clip(((weighted_sum + 3) / 6) * 100, 0, 100))
    h.append(f'<div class="tt-result">합산Z {weighted_sum:+.3f} → 패턴점수 <b>{base_score:.1f}</b>점</div>')

    if sig_count > 0:
        h.append(f'<div class="tt-adj">시그널 {sig_count}개 × 2 = +{sig_count*2} <span class="tt-dim">({_esc(sig_list)})</span></div>')
    else:
        h.append('<div class="tt-adj">시그널 0개</div>')

    h.append(f'<div class="tt-final">종합 = <b>{final_score:.1f}</b>점</div>')

    return ''.join(h)


def _build_stock_table_html(display_df, classified_df, pat_col):
    """종목 순위 HTML 테이블 생성 (5D Z / 종합점수 셀 호버 시 점수 분석 툴팁)."""

    # Z-Score 데이터 조인
    z_map = {}
    if not classified_df.empty and '5D' in classified_df.columns:
        for _, zr in classified_df.iterrows():
            z_map[zr['stock_code']] = zr

    css = """
<style>
.stk-wrap { overflow-x:auto; margin-bottom:8px; max-height:680px; overflow-y:auto; }
.stk-tbl { width:100%; border-collapse:collapse; font-size:14px; }
.stk-tbl th {
    padding:8px 10px; text-align:left;
    border-bottom:2px solid #222222; color:#94a3b8;
    font-weight:600; font-size:13px; position:sticky; top:0;
    background:#0a0a0a; z-index:2;
}
.stk-tbl th:last-child, .stk-tbl td:last-child { text-align:right; }
.stk-tbl td { padding:6px 10px; color:#e2e8f0; border-bottom:1px solid #111111; }
.stk-tbl tr:hover td { background:#111111; }
/* 툴팁 호스트 셀 */
.hc { position:relative; cursor:default; }
.hc .tt-pop { display:none; position:absolute; z-index:20;
    right:0; top:100%; width:620px;
    padding:16px 20px; background:#111111; border:1px solid #2a2a2a;
    border-radius:10px; box-shadow:0 8px 32px rgba(0,0,0,.55);
    font-size:14px; line-height:1.5; color:#cbd5e1; text-align:left !important;
}
.hc:hover .tt-pop { display:block; }
.tt-title { font-weight:700; font-size:15px; color:#4ade80; margin-bottom:8px;
    padding-bottom:6px; border-bottom:1px solid #222222; }
/* Z-Score 그리드 */
.zg { display:flex; gap:2px; margin-bottom:10px; padding-bottom:10px;
    border-bottom:1px solid #222222; }
.zg-cell { flex:1; text-align:center; padding:4px 2px;
    background:#0a0a0a; border-radius:4px; font-size:13px; line-height:1.4; }
.zg-period { color:#64748b; font-size:10px; }
.zg .zp { color:#4ade80; font-weight:600; }
.zg .zn { color:#f87171; font-weight:600; }
.tt-section { font-weight:600; color:#e2e8f0; margin-top:10px; margin-bottom:6px;
    border-bottom:1px solid #222222; padding-bottom:4px; font-size:13px; }
.tt-line { margin-left:4px; font-size:13px; margin-bottom:1px; }
/* 점수 항목 2열 그리드 */
.cc-grid { display:grid; grid-template-columns:1fr 1fr; gap:4px; }
.cc { padding:5px 8px; background:#0a0a0a; border-radius:5px; font-size:13px; }
.cc-h { font-weight:600; color:#e2e8f0; }
.cc-w { font-weight:400; color:#64748b; }
.cc-f { color:#94a3b8; font-size:11px; margin-top:1px; }
.cc-v { font-family:monospace; font-size:13px; margin-top:1px; }
.tt-result { margin-top:8px; color:#e2e8f0; font-size:14px; }
.tt-adj { margin:3px 0 3px 6px; font-family:monospace; font-size:13px; }
.tt-final { margin-top:10px; font-size:16px; font-weight:700; color:#4ade80;
    border-top:1px solid #2a2a2a; padding-top:8px; }
.tt-dim { color:#64748b; }
.score-bar { display:inline-block; height:6px; border-radius:3px; background:#4ade80; vertical-align:middle; margin-right:6px; }
.v-pos { color:#4ade80; }
.v-neg { color:#f87171; }
</style>
"""

    headers = ['종목코드', '종목명', '섹터', '패턴', '시그널', '5D Z', '종합점수', '시총', '시총순위']
    header_html = ''.join(f'<th>{h}</th>' for h in headers)

    rows_html = []
    for _, row in display_df.iterrows():
        code = _esc(str(row.get('stock_code', '')))
        name = _esc(str(row.get('stock_name', '')))
        sector = _esc(str(row.get('sector', '-') or '-'))
        pat = _esc(str(row.get(pat_col, '-')))
        sig = int(row.get('signal_count', 0))
        z5d = row.get('5D', float('nan'))
        fs = float(row.get('final_score', 0))

        # Tooltip HTML (shared by both hover cells)
        zrow = z_map.get(row.get('stock_code'), None)
        tooltip_html = _build_tooltip_html(row, zrow)
        tt_div = f'<div class="tt-pop">{tooltip_html}</div>'

        # 5D Z cell — hover triggers tooltip
        if pd.notna(z5d):
            zcls = 'v-pos' if z5d >= 0 else 'v-neg'
            z5d_td = f'<td class="hc"><span class="{zcls}">{z5d:+.2f}</span>{tt_div}</td>'
        else:
            z5d_td = f'<td class="hc">-{tt_div}</td>'

        # Score cell — hover triggers tooltip
        bar_w = max(0, min(100, fs / 115 * 100))
        score_td = f'<td class="hc"><span class="score-bar" style="width:{bar_w:.0f}px;"></span>{fs:.1f}{tt_div}</td>'

        # 시총 / 시총순위
        mcap_str = _esc(str(row.get('market_cap_str', '-') or '-'))
        mcap_rank = row.get('market_cap_rank', float('nan'))
        mcap_rank_str = f'{int(mcap_rank)}위' if pd.notna(mcap_rank) else '-'

        rows_html.append(
            f'<tr>'
            f'<td>{code}</td><td>{name}</td><td>{sector}</td>'
            f'<td>{pat}</td><td>{sig}</td>{z5d_td}{score_td}'
            f'<td>{mcap_str}</td><td>{mcap_rank_str}</td>'
            f'</tr>'
        )

    count_note = '<div style="color:#64748b;font-size:12px;margin-bottom:6px;">5D Z 또는 종합점수에 마우스를 올리면 점수 산출 분석이 표시됩니다</div>'
    table = (
        f'{css}{count_note}'
        f'<div class="stk-wrap">'
        f'<table class="stk-tbl"><thead><tr>{header_html}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table></div>'
    )
    return table


# ---------------------------------------------------------------------------
# 사이드바: 4개 (기관 가중치 / 기준 날짜 / 섹터 / 최소 점수)
# ---------------------------------------------------------------------------
min_date, max_date = get_date_range()
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
    help="해당 날짜 기준으로 패턴/시그널을 분석합니다. 과거 날짜를 선택하면 당시 상태를 볼 수 있습니다.",
)
end_date_str = end_date.strftime("%Y-%m-%d")

st.sidebar.divider()

direction = st.sidebar.selectbox(
    "분석 방향", ["매수 (Long)", "매도 (Short)"], index=0,
    help="매수: 외국인/기관 순매수 상위 종목 · 매도: 순매도 상위 종목",
)
_direction = 'long' if '매수' in direction else 'short'

mcap_filter = st.sidebar.selectbox(
    "시총 필터", ["전체", "시총 100위 이내", "시총 200위 이내", "시총 300위 이내", "시총 500위 이내"], index=0,
    help="대형주 위주로 필터링합니다.",
)

st.sidebar.divider()
_rescale_on = st.sidebar.checkbox(
    "점수 보정", value=True,
    help="시총 필터 범위 종목의 최근 N거래일 점수 분포 기준으로 0~100 스케일을 재조정합니다. "
         "대형주의 점수가 낮게 나오는 문제를 해결합니다.",
    key="pa_rescale_on",
)
_rescale_lookback = st.sidebar.number_input(
    "기준 기간 (거래일)", min_value=20, max_value=240, value=120, step=20,
    help="최근 M거래일의 점수 분포를 기준으로 스케일링합니다. 길수록 안정적, 짧으면 최근 시장 반영.",
    disabled=not _rescale_on,
    key="pa_rescale_lookback",
)
st.sidebar.divider()

sectors = get_sectors()
selected_sector = st.sidebar.selectbox("섹터", ["전체"] + sectors)

min_score = st.sidebar.slider("최소 종합점수", 0.0, 100.0, 60.0, step=5.0,
                               help="종합점수(패턴점수 + 시그널수×2)가 이 값 이상인 종목만 표시합니다.")

# 누적 수급 설정
st.sidebar.divider()
st.sidebar.markdown("**누적 수급 설정**")
_LOOKBACK_OPTIONS = {'5일': 5, '10일': 10, '20일': 20, '60일': 60, '120일': 120, '240일': 240}
lookback_label = st.sidebar.selectbox("누적 조회 기간", list(_LOOKBACK_OPTIONS.keys()), index=2, key="pa_lookback")
lookback_days = _LOOKBACK_OPTIONS[lookback_label]
top_rank_n = st.sidebar.slider("상위 기준 (Top N)", 10, 200, 50, step=10, key="pa_top_rank_n",
                                help="점수/수급 출현빈도 계산 시 '상위 N위 이내' 기준")

# ---------------------------------------------------------------------------
# 데이터 로드 & 필터링
# ---------------------------------------------------------------------------
_prog = st.progress(0, text="분석 준비 중... 0%")
zscore_matrix, classified_df, signals_df, report_df = run_analysis_pipeline_with_progress(
    end_date=end_date_str, progress_bar=_prog,
    institution_weight=institution_weight,
    direction=_direction,
)
_prog.empty()

if report_df.empty:
    st.warning("분석 데이터가 없습니다.")
    st.stop()

# final_score 계산 (보정 전 raw)
report_df = report_df.copy()
report_df['final_score'] = report_df['score'] + report_df.get('signal_count', 0) * 2

# 점수 보정 — 시총 필터 기준 분포로 백분위 변환
_ref_dists = {}
if _rescale_on:
    with st.spinner("점수 보정 기준 계산 중... (첫 호출 시 ~20초, 이후 캐시)"):
        _ref_dists = get_score_reference_distributions(
            end_date=end_date_str,
            lookback_days=_rescale_lookback,
            institution_weight=institution_weight,
            direction=_direction,
        )
_mcap_to_ref_key = {
    "시총 100위 이내": "top_100", "시총 200위 이내": "top_200",
    "시총 300위 이내": "top_300", "시총 500위 이내": "all",
}
_ref_list = _ref_dists.get(_mcap_to_ref_key.get(mcap_filter, "all"), [])
if _ref_list:
    report_df = rescale_scores(report_df, _ref_list)

# 섹터 + 최소 점수 필터
conn = get_db_connection()
report_gen = IntegratedReport(conn)
filtered_df = report_gen.filter_report(
    report_df,
    sector=selected_sector if selected_sector != '전체' else None,
    min_score=min_score if min_score > 0 else None,
)

# final_score가 filter_report 이후에도 유지되는지 확인
if 'final_score' not in filtered_df.columns and not filtered_df.empty:
    filtered_df['final_score'] = filtered_df['score'] + filtered_df.get('signal_count', 0) * 2

# 최소 점수를 final_score 기준으로 재필터
if min_score > 0 and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df['final_score'] >= min_score]

# 항상 종합점수 내림차순 정렬
filtered_df = filtered_df.sort_values('final_score', ascending=False)

# 5D Z-Score 병합
if not classified_df.empty and '5D' in classified_df.columns:
    _z5d = classified_df[['stock_code', '5D']].drop_duplicates('stock_code')
    filtered_df = filtered_df.merge(_z5d, on='stock_code', how='left')

# 시총 데이터 병합 + 필터
_mcap = get_market_cap_latest()
if not _mcap.empty:
    filtered_df = filtered_df.merge(
        _mcap[['stock_code', 'market_cap_str', 'market_cap_rank']],
        on='stock_code', how='left',
    )

_mcap_limits = {"시총 100위 이내": 100, "시총 200위 이내": 200, "시총 300위 이내": 300, "시총 500위 이내": 500}
if mcap_filter in _mcap_limits and 'market_cap_rank' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['market_cap_rank'] <= _mcap_limits[mcap_filter]]

_dir_label = "순매수" if _direction == 'long' else "순매도"
_mcap_note = " · 시총 500위 이내" if mcap_filter != "전체" else ""
st.caption(f"필터링 결과: {len(filtered_df)}개 종목 (전체 {len(report_df)}개) | {_dir_label} | 종합점수 내림차순{_mcap_note}")

# ---------------------------------------------------------------------------
# 4개 탭: 점수 순위 / 섹터 분석 / 누적 수급 / 패턴 가이드
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["점수 순위", "섹터 분석", "누적 수급", "패턴 가이드"])

_pat_col = 'pattern_label' if 'pattern_label' in filtered_df.columns else 'pattern'

# ---- Tab 1: 종목 순위 ----
with tab1:
    if filtered_df.empty:
        st.info("조건에 맞는 종목이 없습니다. 최소 종합점수를 낮춰보세요.")
    else:
        html = _build_stock_table_html(filtered_df, classified_df, _pat_col)
        st.markdown(html, unsafe_allow_html=True)

        # 관심종목 추가/제거 UI
        with st.expander("⭐ 관심종목 관리", expanded=False):
            _stock_opts = [
                f"{r['stock_name']} ({r['stock_code']})"
                for _, r in filtered_df.iterrows()
            ]
            _sel = st.multiselect(
                "추가할 종목 선택 (현재 필터 기준)",
                options=_stock_opts,
                placeholder="종목명 또는 코드로 검색...",
                key="wl_add_sel",
            )
            _c1, _c2 = st.columns(2)
            if _c1.button("⭐ 선택 종목 추가", use_container_width=True, disabled=not _sel):
                for _opt in _sel:
                    _scode = _opt.split('(')[-1].rstrip(')')
                    _sname = _opt.rsplit(' (', 1)[0]
                    _ssector = filtered_df[filtered_df['stock_code'] == _scode]['sector'].values
                    add_to_watchlist(_scode, _sname, str(_ssector[0]) if len(_ssector) else '')
                st.toast(f"{len(_sel)}개 종목을 관심종목에 추가했습니다.", icon="⭐")
                st.rerun()
            if _c2.button("🗑️ 선택 종목 제거", use_container_width=True, disabled=not _sel):
                for _opt in _sel:
                    _scode = _opt.split('(')[-1].rstrip(')')
                    remove_from_watchlist(_scode)
                st.toast(f"{len(_sel)}개 종목을 관심종목에서 제거했습니다.", icon="🗑️")
                st.rerun()

# ---- Tab 2: 섹터 분석 ----
with tab2:
    _src_df = filtered_df if not filtered_df.empty else report_df
    if _src_df.empty:
        st.info("데이터가 없습니다.")
    else:
        st.subheader("섹터 Treemap")
        st.caption("박스 크기: 종합점수 비례 | 색상: 빨강(낮음) → 초록(높음) | 섹터별 상위 10개 종목")
        fig_treemap = create_sector_treemap(_src_df)
        st.plotly_chart(fig_treemap, width='stretch', theme=None)

        st.divider()

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("섹터 평균 점수")
            fig_avg = create_sector_avg_score_chart(_src_df)
            st.plotly_chart(fig_avg, width="stretch", theme=None)
        with col_right:
            st.subheader("수급 집중도 TOP 10")
            st.caption("섹터점수 = 평균점수 × (1 + 고득점/전체) | 5개 이상 섹터만")
            fig_conc = create_sector_concentration_chart(_src_df)
            st.plotly_chart(fig_conc, width="stretch", theme=None)

        _cc = _src_df.copy()
        _cc['sector'] = _cc['sector'].fillna('기타')
        if 'final_score' not in _cc.columns:
            _cc['final_score'] = _cc['score'] + _cc.get('signal_count', 0) * 2
        _agg = _cc.groupby('sector').agg(
            평균점수=('final_score', 'mean'),
            종목수=('stock_code', 'size'),
        ).reset_index()
        _high = _cc[_cc['final_score'] >= 70].groupby('sector').size().reset_index(name='고득점')
        _agg = _agg.merge(_high, on='sector', how='left')
        _agg['고득점'] = _agg['고득점'].fillna(0).astype(int)
        _agg = _agg[_agg['종목수'] >= 5]
        _agg['섹터점수'] = (_agg['평균점수'] * (1 + _agg['고득점'] / _agg['종목수'])).round(1)
        _agg['평균점수'] = _agg['평균점수'].round(1)
        _agg = _agg.sort_values('섹터점수', ascending=False).head(10)
        _agg = _agg.rename(columns={'sector': '섹터'})
        st.dataframe(
            _agg[['섹터', '평균점수', '종목수', '고득점', '섹터점수']].reset_index(drop=True),
            use_container_width=True,
        )

# ---- Tab 3: 누적 수급 ----
with tab3:
    _cum_prog = st.empty()
    _cum_prog_bar = _cum_prog.progress(0, text="누적 수급 데이터 집계 중... 0%")

    # 시총 필터 → 숫자 변환
    _cum_mcap_top_n = _mcap_limits.get(mcap_filter, None)

    # A. SQL 기반 수급 집계
    _cum_prog_bar = _cum_prog.progress(0.05, text="수급 데이터 집계 중... 5%")
    _cum_sff_df = get_cumulative_sff_ranking(
        end_date=end_date_str,
        lookback_days=lookback_days,
        institution_weight=institution_weight,
        direction=_direction,
        top_rank_n=top_rank_n,
    )

    # B. Precomputer 기반 점수 집계
    _cum_prog_bar = _cum_prog.progress(0.10, text="종합점수 집계 중... 10%")
    _cum_score_df = get_cumulative_score_ranking(
        end_date=end_date_str,
        lookback_days=lookback_days,
        institution_weight=institution_weight,
        direction=_direction,
        top_rank_n=top_rank_n,
        market_cap_top_n=_cum_mcap_top_n or 500,
    )

    # Merge
    _cum_prog_bar = _cum_prog.progress(0.90, text="종목 정보 병합 중... 90%")
    if not _cum_sff_df.empty and not _cum_score_df.empty:
        _cum_merged = pd.merge(_cum_sff_df, _cum_score_df, on='stock_code', how='inner')
    elif not _cum_sff_df.empty:
        _cum_merged = _cum_sff_df.copy()
    elif not _cum_score_df.empty:
        _cum_merged = _cum_score_df.copy()
    else:
        _cum_prog.empty()
        st.info("누적 수급 데이터가 없습니다.")
        _cum_merged = pd.DataFrame()

    if not _cum_merged.empty:
        # 종목명/섹터 조인
        _cum_stock_list = get_stock_list()
        _cum_merged = _cum_merged.merge(
            _cum_stock_list[['stock_code', 'stock_name', 'sector']],
            on='stock_code', how='left',
        )

        # 시총 조인
        _cum_mcap = get_market_cap_latest()
        if not _cum_mcap.empty:
            _cum_merged = _cum_merged.merge(
                _cum_mcap[['stock_code', 'market_cap_rank', 'market_cap_str']],
                on='stock_code', how='left',
            )
        else:
            _cum_merged['market_cap_rank'] = None
            _cum_merged['market_cap_str'] = '-'

        # 시총 필터
        if _cum_mcap_top_n is not None and 'market_cap_rank' in _cum_merged.columns:
            _cum_merged = _cum_merged[_cum_merged['market_cap_rank'] <= _cum_mcap_top_n]

        # IPO 필터
        if 'trading_days' in _cum_merged.columns:
            _cum_merged = _cum_merged[_cum_merged['trading_days'] >= lookback_days * 0.5]

        # Fill NaN
        for _col in ['avg_score', 'max_score', 'score_top_n_ratio', 'appearance_ratio']:
            if _col in _cum_merged.columns:
                _cum_merged[_col] = _cum_merged[_col].fillna(0)
        for _col in ['cum_sff', 'avg_sff', 'positive_ratio', 'sff_top_n_ratio', 'cum_net_amount']:
            if _col in _cum_merged.columns:
                _cum_merged[_col] = _cum_merged[_col].fillna(0)

        # Composite 계산
        def _pct_rank(series):
            return series.rank(pct=True, na_option='bottom')

        _zero = pd.Series(0, index=_cum_merged.index)
        _cum_merged['pct_avg_score'] = _pct_rank(_cum_merged['avg_score']) if 'avg_score' in _cum_merged.columns else _zero
        _cum_merged['pct_cum_sff'] = _pct_rank(_cum_merged['cum_sff']) if 'cum_sff' in _cum_merged.columns else _zero
        _cum_merged['pct_positive_ratio'] = _pct_rank(_cum_merged['positive_ratio']) if 'positive_ratio' in _cum_merged.columns else _zero
        _cum_merged['pct_top_n_ratio'] = _pct_rank(_cum_merged['score_top_n_ratio']) if 'score_top_n_ratio' in _cum_merged.columns else _zero

        _cum_merged['composite'] = (
            _cum_merged['pct_avg_score'] * 0.35
            + _cum_merged['pct_cum_sff'] * 0.25
            + _cum_merged['pct_positive_ratio'] * 0.20
            + _cum_merged['pct_top_n_ratio'] * 0.20
        ) * 100

        # 비율 → 퍼센트
        for _col in ['positive_ratio', 'sff_top_n_ratio', 'score_top_n_ratio', 'appearance_ratio']:
            if _col in _cum_merged.columns:
                _cum_merged[_col + '_pct'] = _cum_merged[_col] * 100

        # display_name
        _cum_merged['display_name'] = _cum_merged.apply(
            lambda r: f"{r.get('stock_name', r['stock_code'])} ({r['stock_code']})", axis=1
        )

        _cum_prog.empty()

        # 표시
        _cum_dir_label = "매수" if _direction == 'long' else "매도"
        st.caption(
            f"Composite = 평균점수(35%) + 누적Sff(25%) + 양수비율(20%) + 출현빈도(20%) 백분위 가중평균  |  "
            f"**{lookback_label}** / {_cum_dir_label} / Top {top_rank_n}"
        )

        _cum_display_n = 50
        _cum_chart_top_n = 10
        _cum_sorted = _cum_merged.sort_values('composite', ascending=False).head(_cum_display_n)

        if not _cum_sorted.empty:
            _cum_color = '#38bdf8'  # sky-400

            fig_cum = create_cumulative_bar_chart(
                _cum_sorted, 'composite', title=f'종합 순위 Top {_cum_chart_top_n}',
                top_n=_cum_chart_top_n, value_format='.1f', color=_cum_color,
            )
            st.plotly_chart(fig_cum, use_container_width=True, theme=None)

            _cum_max_pos_pct = max(_cum_sorted['positive_ratio_pct'].max(), 1) if 'positive_ratio_pct' in _cum_sorted.columns else 100
            _cum_max_topn_pct = max(_cum_sorted['score_top_n_ratio_pct'].max(), 1) if 'score_top_n_ratio_pct' in _cum_sorted.columns else 100

            st.dataframe(
                _cum_sorted[['display_name', 'sector', 'composite',
                             'avg_score', 'cum_sff',
                             'positive_ratio_pct', 'score_top_n_ratio_pct',
                             'market_cap_str', 'market_cap_rank']].reset_index(drop=True),
                column_config={
                    'display_name': st.column_config.TextColumn('종목'),
                    'sector': st.column_config.TextColumn('섹터'),
                    'composite': st.column_config.ProgressColumn(
                        '종합 점수', min_value=0, max_value=100, format='%.1f점',
                    ),
                    'avg_score': st.column_config.NumberColumn('평균 점수', format='%.1f점'),
                    'cum_sff': st.column_config.NumberColumn('누적 Sff', format='%.4f'),
                    'positive_ratio_pct': st.column_config.ProgressColumn(
                        '양수비율', min_value=0, max_value=_cum_max_pos_pct, format='%.0f%%',
                    ),
                    'score_top_n_ratio_pct': st.column_config.ProgressColumn(
                        f'점수 Top{top_rank_n}', min_value=0, max_value=_cum_max_topn_pct, format='%.0f%%',
                    ),
                    'market_cap_str': st.column_config.TextColumn('시총'),
                    'market_cap_rank': st.column_config.NumberColumn('시총순위', format='%d위'),
                },
                use_container_width=True,
                hide_index=True,
                height=min(600, len(_cum_sorted) * 38 + 40),
            )
        else:
            st.info("데이터가 없습니다.")
    else:
        _cum_prog.empty()

# ---- Tab 4: 패턴 가이드 ----
with tab4:
    st.markdown("""
### 패턴 분류 체계

이 시스템은 외국인/기관 수급 데이터를 7개 기간(5D~500D) Z-Score로 변환한 뒤,
Z-Score 패턴의 형태에 따라 3가지 기본 패턴 + 7가지 복합 패턴으로 자동 분류합니다.

---

#### 기본 패턴 4종

| 패턴 | 조건 | 의미 | 투자 전략 |
|------|------|------|----------|
| **급등형** | 장기이격 > 1.0, 최근수급 > 0.5, tc ≥ 0.5 | 단기 수급이 장기 대비 급격히 강함 | 추격 매수, 단기 트레이딩 |
| **지속형** | 가중평균 > 0.8, 지속성 > 0.7 | 다수 기간에 걸쳐 일관된 매집 | 조정 시 분할 매수, 중장기 보유 |
| **전환형** | 가중평균 > 0.5, 장기이격 < 0 | 장기 매집은 있으나 최근 수급 약화 | 저점 매수 대기, 반등 시그널 확인 후 진입 |
| **기타** | 위 조건 모두 미충족 | 뚜렷한 수급 패턴 없음 | 관망 또는 다른 지표 참고 |

**핵심 지표 설명**:
- **최근수급** = (5D + 20D) / 2: 최근 단기 수급 강도
- **단기이격** = 5D - 20D: 1주 vs 1달 단기 방향
- **중기이격** = 5D - 100D: 1주 vs 5개월 중기 수급 개선도
- **장기이격** = 5D - max(200D, 100D, 50D, 20D): 단기와 장기의 차이. 양수가 클수록 최근 수급 폭발
- **가중평균**: 7개 기간 Z-Score의 가중 평균 (최근 기간에 높은 비중)
- **지속성**: 양수 Z-Score 기간의 비율 (0~1). 0.7이면 7개 중 5개 이상이 양수
- **tc (Temporal Consistency)**: 인접 기간이 순서대로인 비율 (5D≥10D≥...≥500D). 1.0이면 완벽한 순서

---

#### 복합 패턴 7종 (sub_type)

기본 패턴 위에 추가 한정자를 부여하여 같은 패턴 내에서도 품질 차이를 구분합니다.

**급등형 세부**:

| 복합 패턴 | 조건 | 해석 |
|----------|------|------|
| **장기기반** | 200D > 0.3 AND 100D > 0.3 | 장기간 매집 위에 단기 폭발 — 가장 신뢰도 높음 |
| **감속** | 단기이격 < -0.3 | 급등세 있으나 최근 속도 감소 중 |
| **단기반등** | 200D < -0.3 OR 100D < -0.3 | 장기 매도세 속 일시적 반등 — 함정 가능성 |

**지속형 세부**:

| 복합 패턴 | 조건 | 해석 |
|----------|------|------|
| **단기돌파** | 5D > 1.0 AND 단기이격 > 0.5 | 장기 매집 중 단기 수급 돌파 — 진입 타이밍 |
| **전면수급** | 전 기간 Z > 0 AND 변동성 < 0.5 | 모든 기간에서 꾸준한 매수 — 안정적 |
| **수급약화** | 단기이격 < -0.3 AND 5D < 20D | 매집은 지속되나 최근 수급 둔화 |

**전환형 세부**:

| 복합 패턴 | 조건 | 해석 |
|----------|------|------|
| **V자반등** | 5D > 1.0 AND 최근수급 > 0.5 | 급격한 반전 시그널 — 초기 진입 기회 |

> ※ 복합 패턴은 라벨(정보 제공)만, 점수 보정 없음 (v2)

---

#### 점수 산출 공식 (v2)

```
패턴점수 = ((가중합산Z + 3) / 6) × 100     ← Z∈[-3,3] → 점수∈[0,100]

가중합산Z = 수급강도 × 0.35    (= (5D + 20D) / 2)
          + 수급추세 × 0.20    (= 5D − 200D)
          + 종합수급 × 0.25    (= 가중평균)
          + 수급지속 × 0.20    (= 평균Sff × √연속일수, ×2 정규화, cap 3.0)

종합점수 = 패턴점수 + 시그널수 × 2
  · 시그널: MA골든크로스, 수급가속도, 외인기관동조 (각 2점)
  · 최대 ~106점 (패턴100 + 시그널3×2)
```

---

#### 종합점수 해석 기준

| 점수 구간 | 해석 | 권장 액션 |
|----------|------|----------|
| **75점 이상** | 강한 수급 + 지속성 높음 | 적극 검토, 진입 포인트 확인 |
| **65~74점** | 양호한 수급, 패턴 명확 | 관심종목 등록, 타이밍 탐색 |
| **55~64점** | 보통 수급 | 모니터링, 추가 확인 |
| **55점 미만** | 약한 수급 또는 방향 불명확 | 관망 |

---

#### 시그널 3종

| 시그널 | 조건 | 의미 |
|--------|------|------|
| **MA 골든크로스** | 외국인 5일 이동평균 > 20일 이동평균 | 단기 수급이 중기를 상회 — 상승 전환 |
| **수급 가속도** | 최근 5일 평균 Sff > 이전 5일 평균 Sff | 수급 강도가 증가하는 중 |
| **외인-기관 동조** | 외국인과 기관 모두 같은 방향 순매수 | 두 주체가 동시에 매수 — 확신도 높음 |

시그널이 2개 이상이면 진입 신뢰도가 높습니다 (백테스트 기준 승률 60%, 평균 +3~4%).
""")

st.divider()
st.caption("종목의 상세 분석은 사이드바의 '종목 상세' 페이지에서 확인할 수 있습니다.")
