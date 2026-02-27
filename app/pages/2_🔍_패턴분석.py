"""
패턴 분석 페이지 - 종합점수 기준 종목 순위 + 섹터 분석

사이드바: 기관 가중치 / 기준 날짜 / 섹터 필터 / 최소 점수
3개 탭: 종목 순위, 섹터 분석, 패턴 가이드
종목 상세: 개별 종목 정보 (패턴/점수/시그널/진입/손절)
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

from utils.data_loader import (
    run_analysis_pipeline_with_progress, get_sectors, get_date_range,
    get_stock_list, get_db_connection,
    get_watchlist, add_to_watchlist, remove_from_watchlist,
)
from utils.charts import (
    create_sector_avg_score_chart,
    create_sector_concentration_chart,
    create_sector_treemap,
)
from src.analyzer.integrated_report import IntegratedReport

st.set_page_config(page_title="패턴분석", page_icon="🔍", layout="wide")
st.title("패턴 분류 & 시그널 분석")


# ---------------------------------------------------------------------------
# 점수 산출 툴팁 HTML 생성
# ---------------------------------------------------------------------------
def _build_tooltip_html(row, zscore_row=None):
    """종목의 점수 산출 과정을 HTML 툴팁 내용으로 생성."""
    pattern = row.get('pattern', '기타')
    sub_type = row.get('sub_type', None)
    if pd.isna(sub_type) if not isinstance(sub_type, str) else False:
        sub_type = None
    score = float(row.get('score', 0))
    tc = row.get('temporal_consistency', 0.5)
    tc = tc if pd.notna(tc) else 0.5
    pat_label = row.get('pattern_label', pattern)

    recent = float(row.get('recent', 0))
    momentum = float(row.get('momentum', 0))
    weighted = float(row.get('weighted', 0))
    average = float(row.get('average', 0))
    short_trend = float(row.get('short_trend', 0))
    persistence = float(row.get('persistence', 0))

    is_sustained = (pattern == '지속형')

    if is_sustained:
        weights = {'recent': 0.25, 'momentum': 0.20, 'weighted': 0.30,
                   'average': 0.25, 'short_trend': 0.00}
    else:
        weights = {'recent': 0.25, 'momentum': 0.20, 'weighted': 0.30,
                   'average': 0.10, 'short_trend': 0.15}

    comps = {'recent': recent, 'momentum': momentum, 'weighted': weighted,
             'average': average, 'short_trend': short_trend}

    weighted_sum = sum(comps[k] * weights[k] for k in weights if weights[k] > 0)
    base_score = float(np.clip(((weighted_sum + 3) / 6) * 100, 0, 100))

    tc_bonus = 0.0
    if not is_sustained:
        tc_bonus = (tc - 0.5) * 20

    sub_bonus_map = {
        '장기기반': +5, '단기돌파': +5, 'V자반등': +3, '전면수급': +3,
        '모멘텀약화': -5, '감속': -5, '단기반등': -8,
    }
    sub_bonus = sub_bonus_map.get(sub_type, 0)

    sig_count = int(row.get('signal_count', 0))
    sig_list = str(row.get('signal_list', '') or '')
    final_score = score + sig_count * 5

    # Z-Score 값 추출 (점수 산출 근거에도 사용)
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
    if pattern == '모멘텀형':
        h.append(
            f'<div class="tt-line">모멘텀(<b>{momentum:+.2f}</b>)&gt;1.0 '
            f'&amp; 최근수급(<b>{recent:+.2f}</b>)&gt;0.5 '
            f'&amp; tc(<b>{tc:.2f}</b>)≥0.5</div>')
    elif pattern == '지속형':
        h.append(
            f'<div class="tt-line">가중평균(<b>{weighted:+.2f}</b>)&gt;0.8 '
            f'&amp; 지속성(<b>{persistence:.2f}</b>)&gt;0.7</div>')
    elif pattern == '전환형':
        h.append(
            f'<div class="tt-line">가중평균(<b>{weighted:+.2f}</b>)&gt;0.5 '
            f'&amp; 모멘텀(<b>{momentum:+.2f}</b>)&lt;0</div>')
    else:
        h.append('<div class="tt-line">모멘텀·지속·전환 조건 미충족 → 기타</div>')

    if sub_type:
        h.append(f'<div class="tt-line">복합: <b>{_esc(sub_type)}</b> ({sub_bonus:+d}점)</div>')

    # ── 점수 산출 (근거 공식 포함, 2열 그리드) ──
    h.append('<div class="tt-section">점수 산출</div>')

    z5d = zvals.get('5D')
    z20d = zvals.get('20D')
    z50d = zvals.get('50D')
    z100d = zvals.get('100D')
    z200d = zvals.get('200D')

    def _comp_card(label, key, w):
        v = comps[key]
        # 근거 공식
        formula = ''
        if key == 'recent' and z5d is not None and z20d is not None:
            formula = f'(5D {z5d:+.2f} + 20D {z20d:+.2f}) / 2'
        elif key == 'momentum' and z5d is not None:
            for lp, lv in [('200D', z200d), ('100D', z100d), ('50D', z50d), ('20D', z20d)]:
                if lv is not None:
                    formula = f'5D {z5d:+.2f} − {lp} {lv:+.2f}'
                    break
        elif key == 'weighted' and zvals:
            formula = '가중 평균 (최근 높은 비중)'
        elif key == 'average' and zvals:
            formula = f'{len(zvals)}개 기간 단순 평균'
        elif key == 'short_trend' and z5d is not None and z20d is not None:
            formula = f'5D {z5d:+.2f} − 20D {z20d:+.2f}'

        if w > 0:
            fml = f'<div class="cc-f">{formula}</div>' if formula else ''
            return (
                f'<div class="cc"><div class="cc-h">{label} <span class="cc-w">×{w:.2f}</span></div>'
                f'{fml}'
                f'<div class="cc-v"><b>{v:+.2f}</b> → {v*w:+.3f}</div></div>'
            )
        elif key == 'short_trend' and is_sustained:
            return (
                f'<div class="cc"><div class="cc-h">{label} <span class="tt-dim">×0 (지속형 제외)</span></div>'
                f'<div class="cc-v">{v:+.2f}</div></div>'
            )
        return ''

    cards = []
    for key, label, w in [('recent', '최근수급', 0.25), ('momentum', '모멘텀', 0.20),
                           ('weighted', '가중평균', 0.30), ('average', '단순평균', weights['average']),
                           ('short_trend', '단기모멘텀', weights['short_trend'])]:
        c = _comp_card(label, key, w)
        if c:
            cards.append(c)

    h.append(f'<div class="cc-grid">{"".join(cards)}</div>')
    h.append(f'<div class="tt-result">합산Z {weighted_sum:+.3f} → 기본 <b>{base_score:.1f}</b>점</div>')

    # 보정
    if not is_sustained:
        tc_desc = "순서일치" if tc >= 0.7 else ("혼조" if tc >= 0.4 else "역순")
        h.append(f'<div class="tt-adj">tc보너스 ({tc:.2f} − 0.5) × 20 = <b>{tc_bonus:+.1f}</b> <span class="tt-dim">({tc_desc})</span></div>')

    if sub_type:
        h.append(f'<div class="tt-adj">복합패턴 {_esc(sub_type)} = <b>{sub_bonus:+d}</b></div>')

    calc_total = float(np.clip(base_score + tc_bonus + sub_bonus, 0, 100))
    h.append(f'<div class="tt-result">패턴점수 = <b>{calc_total:.1f}</b>점</div>')

    if sig_count > 0:
        h.append(f'<div class="tt-adj">시그널 {sig_count}개 × 5 = +{sig_count*5} <span class="tt-dim">({_esc(sig_list)})</span></div>')
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
    border-bottom:2px solid #334155; color:#94a3b8;
    font-weight:600; font-size:13px; position:sticky; top:0;
    background:#0f172a; z-index:2;
}
.stk-tbl th:last-child, .stk-tbl td:last-child { text-align:right; }
.stk-tbl td { padding:6px 10px; color:#e2e8f0; border-bottom:1px solid #1e293b; }
.stk-tbl tr:hover td { background:#1e293b; }
/* 툴팁 호스트 셀 */
.hc { position:relative; cursor:default; }
.hc .tt-pop { display:none; position:absolute; z-index:20;
    right:0; top:100%; width:620px;
    padding:16px 20px; background:#1e293b; border:1px solid #475569;
    border-radius:10px; box-shadow:0 8px 32px rgba(0,0,0,.55);
    font-size:14px; line-height:1.5; color:#cbd5e1; text-align:left !important;
}
.hc:hover .tt-pop { display:block; }
.tt-title { font-weight:700; font-size:15px; color:#38bdf8; margin-bottom:8px;
    padding-bottom:6px; border-bottom:1px solid #334155; }
/* Z-Score 그리드 */
.zg { display:flex; gap:2px; margin-bottom:10px; padding-bottom:10px;
    border-bottom:1px solid #334155; }
.zg-cell { flex:1; text-align:center; padding:4px 2px;
    background:#0f172a; border-radius:4px; font-size:13px; line-height:1.4; }
.zg-period { color:#64748b; font-size:10px; }
.zg .zp { color:#4ade80; font-weight:600; }
.zg .zn { color:#f87171; font-weight:600; }
.tt-section { font-weight:600; color:#e2e8f0; margin-top:10px; margin-bottom:6px;
    border-bottom:1px solid #334155; padding-bottom:4px; font-size:13px; }
.tt-line { margin-left:4px; font-size:13px; margin-bottom:1px; }
/* 점수 항목 2열 그리드 */
.cc-grid { display:grid; grid-template-columns:1fr 1fr; gap:4px; }
.cc { padding:5px 8px; background:#0f172a; border-radius:5px; font-size:13px; }
.cc-h { font-weight:600; color:#e2e8f0; }
.cc-w { font-weight:400; color:#64748b; }
.cc-f { color:#94a3b8; font-size:11px; margin-top:1px; }
.cc-v { font-family:monospace; font-size:13px; margin-top:1px; }
.tt-result { margin-top:8px; color:#e2e8f0; font-size:14px; }
.tt-adj { margin:3px 0 3px 6px; font-family:monospace; font-size:13px; }
.tt-final { margin-top:10px; font-size:16px; font-weight:700; color:#38bdf8;
    border-top:1px solid #475569; padding-top:8px; }
.tt-dim { color:#64748b; }
.score-bar { display:inline-block; height:6px; border-radius:3px; background:#38bdf8; vertical-align:middle; margin-right:6px; }
.v-pos { color:#4ade80; }
.v-neg { color:#f87171; }
</style>
"""

    headers = ['종목코드', '종목명', '섹터', '패턴', '시그널', '5D Z', '종합점수']
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

        rows_html.append(
            f'<tr>'
            f'<td>{code}</td><td>{name}</td><td>{sector}</td>'
            f'<td>{pat}</td><td>{sig}</td>{z5d_td}{score_td}'
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
    help="해당 날짜 기준으로 패턴/시그널을 분석합니다. 과거 날짜를 선택하면 당시 상태를 볼 수 있습니다.",
)
end_date_str = end_date.strftime("%Y-%m-%d")

st.sidebar.divider()

sectors = get_sectors()
selected_sector = st.sidebar.selectbox("섹터", ["전체"] + sectors)

min_score = st.sidebar.slider("최소 종합점수", 0.0, 100.0, 60.0, step=5.0,
                               help="종합점수(패턴점수 + 시그널수×5)가 이 값 이상인 종목만 표시합니다.")

# ---------------------------------------------------------------------------
# 데이터 로드 & 필터링
# ---------------------------------------------------------------------------
_prog = st.progress(0, text="분석 준비 중... 0%")
zscore_matrix, classified_df, signals_df, report_df = run_analysis_pipeline_with_progress(
    end_date=end_date_str, progress_bar=_prog,
    institution_weight=institution_weight,
)
_prog.empty()

if report_df.empty:
    st.warning("분석 데이터가 없습니다.")
    st.stop()

# final_score 계산
report_df = report_df.copy()
if 'signal_count' in report_df.columns:
    report_df['final_score'] = report_df['score'] + report_df['signal_count'] * 5
else:
    report_df['final_score'] = report_df['score']

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
    if 'signal_count' in filtered_df.columns:
        filtered_df['final_score'] = filtered_df['score'] + filtered_df['signal_count'] * 5
    else:
        filtered_df['final_score'] = filtered_df['score']

# 최소 점수를 final_score 기준으로 재필터
if min_score > 0 and not filtered_df.empty:
    filtered_df = filtered_df[filtered_df['final_score'] >= min_score]

# 항상 종합점수 내림차순 정렬
filtered_df = filtered_df.sort_values('final_score', ascending=False)

# 5D Z-Score 병합
if not classified_df.empty and '5D' in classified_df.columns:
    _z5d = classified_df[['stock_code', '5D']].drop_duplicates('stock_code')
    filtered_df = filtered_df.merge(_z5d, on='stock_code', how='left')

st.caption(f"필터링 결과: {len(filtered_df)}개 종목 (전체 {len(report_df)}개) | 종합점수 내림차순")

# ---------------------------------------------------------------------------
# 3개 탭: 종목 순위 / 섹터 분석 / 패턴 가이드
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["종목 순위", "섹터 분석", "패턴 가이드"])

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
            _cc['final_score'] = _cc['score'] + _cc.get('signal_count', 0) * 5
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

# ---- Tab 3: 패턴 가이드 ----
with tab3:
    st.markdown("""
### 패턴 분류 체계

이 시스템은 외국인/기관 수급 데이터를 7개 기간(5D~500D) Z-Score로 변환한 뒤,
Z-Score 패턴의 형태에 따라 3가지 기본 패턴 + 7가지 복합 패턴으로 자동 분류합니다.

---

#### 기본 패턴 4종

| 패턴 | 조건 | 의미 | 투자 전략 |
|------|------|------|----------|
| **모멘텀형** | 모멘텀 > 1.0, 최근수급 > 0.5, tc ≥ 0.5 | 단기 수급이 장기 대비 급격히 강함 | 추격 매수, 단기 트레이딩 |
| **지속형** | 가중평균 > 0.8, 지속성 > 0.7 | 다수 기간에 걸쳐 일관된 매집 | 조정 시 분할 매수, 중장기 보유 |
| **전환형** | 가중평균 > 0.5, 모멘텀 < 0 | 장기 매집은 있으나 최근 수급 약화 | 저점 매수 대기, 반등 시그널 확인 후 진입 |
| **기타** | 위 조건 모두 미충족 | 뚜렷한 수급 패턴 없음 | 관망 또는 다른 지표 참고 |

**핵심 지표 설명**:
- **모멘텀** = 5D - max(200D, 100D, 50D, 20D): 단기와 장기의 차이. 양수가 클수록 최근 수급 폭발
- **최근수급** = (5D + 20D) / 2: 최근 단기 수급 강도
- **가중평균**: 7개 기간 Z-Score의 가중 평균 (최근 기간에 높은 비중)
- **지속성**: 양수 Z-Score 기간의 비율 (0~1). 0.7이면 7개 중 5개 이상이 양수
- **tc (Temporal Consistency)**: 인접 기간이 순서대로인 비율 (5D≥10D≥...≥500D). 1.0이면 완벽한 순서

---

#### 복합 패턴 7종 (sub_type)

기본 패턴 위에 추가 한정자를 부여하여 같은 패턴 내에서도 품질 차이를 구분합니다.

**모멘텀형 세부**:

| 복합 패턴 | 조건 | 점수 보정 | 해석 |
|----------|------|----------|------|
| **장기기반** | 200D > 0.3 AND 100D > 0.3 | **+5점** | 장기간 매집 위에 단기 폭발 — 가장 신뢰도 높음 |
| **감속** | 단기모멘텀 < -0.3 | **-5점** | 모멘텀은 있으나 최근 속도 감소 중 |
| **단기반등** | 200D < -0.3 OR 100D < -0.3 | **-8점** | 장기 매도세 속 일시적 반등 — 함정 가능성 |

**지속형 세부**:

| 복합 패턴 | 조건 | 점수 보정 | 해석 |
|----------|------|----------|------|
| **단기돌파** | 5D > 1.0 AND 단기모멘텀 > 0.5 | **+5점** | 장기 매집 중 단기 수급 돌파 — 진입 타이밍 |
| **전면수급** | 전 기간 Z > 0 AND 변동성 < 0.5 | **+3점** | 모든 기간에서 꾸준한 매수 — 안정적 |
| **모멘텀약화** | 단기모멘텀 < -0.3 AND 5D < 20D | **-5점** | 매집은 지속되나 최근 수급 둔화 |

**전환형 세부**:

| 복합 패턴 | 조건 | 점수 보정 | 해석 |
|----------|------|----------|------|
| **V자반등** | 5D > 1.0 AND 최근수급 > 0.5 | **+3점** | 급격한 반전 시그널 — 초기 진입 기회 |

---

#### 점수 산출 공식

```
기본점수 = ((가중합산Z + 3) / 6) × 100     ← Z∈[-3,3] → 점수∈[0,100]

가중합산Z = 최근수급 × 0.25
          + 모멘텀 × 0.20
          + 가중평균 × 0.30
          + 단순평균 × 0.10 (지속형: 0.25)
          + 단기모멘텀 × 0.15 (지속형: 0)

패턴점수 = 기본점수 + tc보너스 + 복합패턴보너스
  · tc보너스 = (tc - 0.5) × 20  (±10점, 지속형 제외)
  · 복합패턴보너스 = -8 ~ +5점

종합점수 = 패턴점수 + 시그널수 × 5
  · 시그널: MA골든크로스, 수급가속도, 외인기관동조 (각 5점)
  · 최대 115점 (패턴100 + 시그널3×5)
```

---

#### 종합점수 해석 기준

| 점수 구간 | 해석 | 권장 액션 |
|----------|------|----------|
| **80점 이상** | 매우 강한 수급 + 다중 시그널 | 적극 검토, 진입 포인트 확인 |
| **70~79점** | 강한 수급, 패턴 명확 | 관심종목 등록, 타이밍 탐색 |
| **60~69점** | 보통 수급, 패턴 존재 | 모니터링, 추가 시그널 대기 |
| **50~59점** | 약한 수급, 불확실 | 관망, 다른 지표와 교차 확인 |
| **50점 미만** | 중립 또는 매도 수급 | 롱 전략 부적합 |

---

#### 시그널 3종

| 시그널 | 조건 | 의미 |
|--------|------|------|
| **MA 골든크로스** | 외국인 5일 이동평균 > 20일 이동평균 | 단기 수급이 중기를 상회 — 상승 전환 |
| **수급 가속도** | 최근 5일 평균 Sff > 이전 5일 평균 Sff | 수급 강도가 증가하는 중 |
| **외인-기관 동조** | 외국인과 기관 모두 같은 방향 순매수 | 두 주체가 동시에 매수 — 확신도 높음 |

시그널이 2개 이상이면 진입 신뢰도가 높습니다 (백테스트 기준 승률 60%, 평균 +3~4%).
""")

# ---------------------------------------------------------------------------
# 종목 상세 (원래 하단 섹션)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("종목 상세 정보")

if not filtered_df.empty:
    stock_options = [
        f"{row['stock_name']} ({row['stock_code']})"
        for _, row in filtered_df.iterrows()
    ]

    selected = st.selectbox("종목 선택", stock_options)

    if selected:
        stock_code = selected.split('(')[-1].rstrip(')')
        row = filtered_df[filtered_df['stock_code'] == stock_code].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("패턴", row.get('pattern_label', row.get('pattern', '-')))
        col2.metric("종합점수", f"{row.get('final_score', 0):.1f}")
        col3.metric("시그널 수", f"{row.get('signal_count', 0):.0f}")
        col4.metric("섹터", row.get('sector', '-'))

        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.markdown("**진입 포인트**")
            st.info(row.get('entry_point', '-'))
        with detail_col2:
            st.markdown("**손절 기준**")
            st.warning(row.get('stop_loss', '-'))

        # Z-Score 데이터 표시
        if not classified_df.empty:
            stock_zscore = classified_df[classified_df['stock_code'] == stock_code]
            if not stock_zscore.empty:
                zscore_row = stock_zscore.iloc[0]
                period_cols = ['5D', '10D', '20D', '50D', '100D', '200D', '500D']
                existing_periods = [c for c in period_cols if c in zscore_row.index]

                if existing_periods:
                    st.markdown("**기간별 Z-Score**")
                    zscore_data = {col: [f"{zscore_row[col]:.2f}"] for col in existing_periods}
                    st.dataframe(pd.DataFrame(zscore_data), use_container_width=True)

        if 'signal_list' in row.index and row['signal_list']:
            st.markdown("**활성 시그널**")
            signals = row['signal_list'] if isinstance(row['signal_list'], str) else str(row['signal_list'])
            st.success(signals)
else:
    st.info("종목을 선택하려면 사이드바에서 최소 종합점수를 조정하세요.")
