"""
누적 수급 순위 페이지 — 종합점수 + 수급 흐름을 기간/누적 관점으로 분석

최근 N일간 꾸준히 상위권에 머문 종목을 식별.
4개 탭: 종합 순위 / 종합점수 평균 / 누적 수급강도 / 수급 안정성 & 출현빈도
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
from datetime import datetime

from utils.data_loader import (
    get_date_range,
    get_cumulative_score_ranking,
    get_cumulative_sff_ranking,
    get_stock_list,
    get_market_cap_latest,
)
from utils.charts import create_cumulative_bar_chart

st.set_page_config(page_title="누적 수급", page_icon="🏆", layout="wide")

# CSS: 위젯 테두리 가시성
st.markdown("""
<style>
div[data-baseweb="select"] > div { border-color: #333 !important; }
div[data-baseweb="input"] input, div[data-baseweb="input"] > div { border-color: #333 !important; }
[data-testid="stDateInput"] > div > div > div { border-color: #333 !important; }
</style>
""", unsafe_allow_html=True)

st.title("누적 수급 순위")
st.caption("최근 N일간 종합점수·수급강도를 누적하여 꾸준히 상위권에 머문 종목 식별")

# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
min_date, max_date = get_date_range()
_max_dt = datetime.strptime(max_date, "%Y-%m-%d")

institution_weight = st.sidebar.slider(
    "기관 가중치", 0.0, 1.0, 0.3, step=0.05,
    key="w_institution_weight",
    help="기관 수급이 외국인과 같은 방향일 때만 가중치 반영 (0=외국인만, 1=동등)",
)

end_date = st.sidebar.date_input(
    "기준 날짜",
    value=_max_dt,
    min_value=datetime.strptime(min_date, "%Y-%m-%d"),
    max_value=_max_dt.replace(month=12, day=31),
    help="분석 기준 종료일",
)
end_date_str = end_date.strftime("%Y-%m-%d")

_LOOKBACK_OPTIONS = {
    '5일': 5, '10일': 10, '20일': 20,
    '60일': 60, '120일': 120, '240일': 240,
}
lookback_label = st.sidebar.selectbox(
    "조회 기간", list(_LOOKBACK_OPTIONS.keys()), index=2,
    help="누적 집계할 최근 거래일 수",
)
lookback_days = _LOOKBACK_OPTIONS[lookback_label]

direction = st.sidebar.selectbox(
    "분석 방향",
    ['매수 (Long)', '매도 (Short)'],
    index=0,
    help="매수: 순매수 상위 종목 / 매도: 순매도 상위 종목",
)
direction_key = 'long' if 'Long' in direction else 'short'

_MCAP_OPTIONS = {'전체': None, '100위': 100, '200위': 200, '300위': 300, '500위': 500}
mcap_label = st.sidebar.selectbox("시총 필터", list(_MCAP_OPTIONS.keys()), index=3)
mcap_top_n = _MCAP_OPTIONS[mcap_label]

top_rank_n = st.sidebar.slider(
    "상위 기준 (Top N)", 10, 200, 50, step=10,
    help="점수/수급 출현빈도 계산 시 '상위 N위 이내' 기준",
)

_DISPLAY_OPTIONS = {20: '20개', 50: '50개', 100: '100개'}
display_n = st.sidebar.selectbox(
    "표시 종목 수",
    list(_DISPLAY_OPTIONS.keys()),
    index=1,
    format_func=lambda x: _DISPLAY_OPTIONS[x],
)

# 바차트 표시 종목 수 (고정 — 가독성 최적)
_CHART_TOP_N = 10

# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
_prog = st.progress(0, text="수급 데이터 집계 중... 0%")

# A. SQL 기반 수급 집계 (빠름, ~1초)
_prog.progress(0.05, text="수급 데이터 집계 중... 5%")
sff_df = get_cumulative_sff_ranking(
    end_date=end_date_str,
    lookback_days=lookback_days,
    institution_weight=institution_weight,
    direction=direction_key,
    top_rank_n=top_rank_n,
)

# B. Precomputer 기반 점수 집계 (무거움, ~2-10초)
_prog.progress(0.10, text="종합점수 집계 중... 10%")
score_df = get_cumulative_score_ranking(
    end_date=end_date_str,
    lookback_days=lookback_days,
    institution_weight=institution_weight,
    direction=direction_key,
    top_rank_n=top_rank_n,
    market_cap_top_n=mcap_top_n or 500,
)

# C. 종목 정보 + 시총
_prog.progress(0.90, text="종목 정보 병합 중... 90%")
stock_list = get_stock_list()
market_cap = get_market_cap_latest()

# Merge: inner join (양쪽 모두 있는 종목만 — 한쪽만 있으면 composite 왜곡)
if not sff_df.empty and not score_df.empty:
    merged = pd.merge(sff_df, score_df, on='stock_code', how='inner')
elif not sff_df.empty:
    merged = sff_df.copy()
elif not score_df.empty:
    merged = score_df.copy()
else:
    _prog.empty()
    st.warning("데이터가 없습니다. DB를 확인하세요.")
    st.stop()

# 종목명/섹터 조인
merged = merged.merge(stock_list[['stock_code', 'stock_name', 'sector']], on='stock_code', how='left')

# 시총 조인
if not market_cap.empty:
    merged = merged.merge(
        market_cap[['stock_code', 'market_cap_rank', 'market_cap_str']],
        on='stock_code', how='left',
    )
else:
    merged['market_cap_rank'] = None
    merged['market_cap_str'] = '-'

# 시총 필터
if mcap_top_n is not None and 'market_cap_rank' in merged.columns:
    merged = merged[merged['market_cap_rank'] <= mcap_top_n]

# IPO 필터: 조회 기간의 50% 이상 거래일이 있는 종목만
if 'trading_days' in merged.columns:
    merged = merged[merged['trading_days'] >= lookback_days * 0.5]

# Fill NaN
for col in ['avg_score', 'max_score', 'score_top_n_ratio', 'appearance_ratio']:
    if col in merged.columns:
        merged[col] = merged[col].fillna(0)
for col in ['cum_sff', 'avg_sff', 'positive_ratio', 'sff_top_n_ratio', 'cum_net_amount']:
    if col in merged.columns:
        merged[col] = merged[col].fillna(0)

# ---------------------------------------------------------------------------
# 종합점수 (Composite) 계산 — 백분위 순위 기반, 0~100 스케일
# ---------------------------------------------------------------------------
def _pct_rank(series):
    """0~1 백분위 순위 (NaN-safe)"""
    return series.rank(pct=True, na_option='bottom')

if len(merged) > 0:
    # 컬럼 부재 시 0으로 채운 Series 사용 (NaN 전파 방지)
    _zero = pd.Series(0, index=merged.index)
    merged['pct_avg_score'] = _pct_rank(merged['avg_score']) if 'avg_score' in merged.columns else _zero
    merged['pct_cum_sff'] = _pct_rank(merged['cum_sff']) if 'cum_sff' in merged.columns else _zero
    merged['pct_positive_ratio'] = _pct_rank(merged['positive_ratio']) if 'positive_ratio' in merged.columns else _zero
    merged['pct_top_n_ratio'] = _pct_rank(merged['score_top_n_ratio']) if 'score_top_n_ratio' in merged.columns else _zero

    # 0~100 스케일로 변환 (점수처럼 읽히도록)
    merged['composite'] = (
        merged['pct_avg_score'] * 0.35
        + merged['pct_cum_sff'] * 0.25
        + merged['pct_positive_ratio'] * 0.20
        + merged['pct_top_n_ratio'] * 0.20
    ) * 100
else:
    merged['composite'] = 0

# 비율 컬럼 → 퍼센트(0~100)로 변환 (단위 통일)
for col in ['positive_ratio', 'sff_top_n_ratio', 'score_top_n_ratio', 'appearance_ratio']:
    if col in merged.columns:
        merged[col + '_pct'] = merged[col] * 100

# display_name
merged['display_name'] = merged.apply(
    lambda r: f"{r.get('stock_name', r['stock_code'])} ({r['stock_code']})", axis=1
)

# 누적 Sff 정규화 (0~100, 바 표시용)
if 'cum_sff' in merged.columns and len(merged) > 0:
    _sff_max = merged['cum_sff'].max()
    merged['cum_sff_norm'] = (merged['cum_sff'] / _sff_max * 100) if _sff_max > 0 else 0

_prog.progress(1.0, text="완료 100%")
_prog.empty()

# ---------------------------------------------------------------------------
# KPI 카드
# ---------------------------------------------------------------------------
dir_label = "매수" if direction_key == 'long' else "매도"
st.markdown(f"**기준일**: {end_date_str}  |  **조회 기간**: {lookback_label}  |  **방향**: {dir_label}  |  **상위 기준**: Top {top_rank_n}")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("분석 종목 수", f"{len(merged):,}")

if 'avg_score' in merged.columns and len(merged) > 0:
    _top_score = merged.nlargest(1, 'avg_score')
    k2.metric("최고 평균 점수", f"{_top_score['avg_score'].iloc[0]:.1f}점",
              help=_top_score['display_name'].iloc[0])
else:
    k2.metric("최고 평균 점수", "-")

if 'cum_sff' in merged.columns and len(merged) > 0:
    _top_sff = merged.nlargest(1, 'cum_sff')
    k3.metric("최고 누적 Sff", f"{_top_sff['cum_sff'].iloc[0]:.3f}",
              help=_top_sff['display_name'].iloc[0])
else:
    k3.metric("최고 누적 Sff", "-")

if 'positive_ratio' in merged.columns and len(merged) > 0:
    _top_pos = merged.nlargest(1, 'positive_ratio')
    k4.metric("최고 양수비율", f"{_top_pos['positive_ratio_pct'].iloc[0]:.0f}%",
              help=_top_pos['display_name'].iloc[0])
else:
    k4.metric("최고 양수비율", "-")

if 'score_top_n_ratio' in merged.columns and len(merged) > 0:
    _top_freq = merged.nlargest(1, 'score_top_n_ratio')
    k5.metric(f"최고 출현빈도 (Top {top_rank_n})", f"{_top_freq['score_top_n_ratio_pct'].iloc[0]:.0f}%",
              help=_top_freq['display_name'].iloc[0])
else:
    k5.metric(f"최고 출현빈도 (Top {top_rank_n})", "-")

# ---------------------------------------------------------------------------
# 4개 탭
# ---------------------------------------------------------------------------
tab_composite, tab_score, tab_sff, tab_stability = st.tabs([
    "종합 순위",
    "종합점수 평균",
    "누적 수급강도",
    "수급 안정성 & 출현빈도",
])

# 색상 테마
_COLOR_COMPOSITE = '#38bdf8'  # sky-400
_COLOR_SCORE = '#4ade80'      # green-400
_COLOR_SFF = '#fbbf24'        # amber-400
_COLOR_STABILITY = '#a78bfa'  # violet-400


def _fmt_net_amount(val):
    """순매수금액 → 억/조 포맷"""
    eok = val / 1e8
    abs_eok = abs(eok)
    sign = '+' if eok >= 0 else '-'
    if abs_eok >= 10000:
        return f"{sign}{abs_eok / 10000:,.1f}조"
    return f"{sign}{int(abs_eok):,}억"


# ─── 탭 1: 종합 순위 ─────────────────────────────────────────────────────────
with tab_composite:
    st.caption("4개 지표를 백분위 순위로 변환 → 가중 평균 (점수 35% + 수급 25% + 양수비율 20% + 출현빈도 20%)")

    composite_sorted = merged.sort_values('composite', ascending=False).head(display_n)

    if not composite_sorted.empty:
        # ProgressColumn max_value: 표시 데이터 내 최댓값 기준 (바 비율 의미있게)
        _max_pos_pct = max(composite_sorted['positive_ratio_pct'].max(), 1)
        _max_topn_pct = max(composite_sorted['score_top_n_ratio_pct'].max(), 1)

        fig = create_cumulative_bar_chart(
            composite_sorted, 'composite', title=f'종합 순위 Top {_CHART_TOP_N}',
            top_n=_CHART_TOP_N, value_format='.1f', color=_COLOR_COMPOSITE,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

        st.dataframe(
            composite_sorted[['display_name', 'sector', 'composite',
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
                    '양수비율', min_value=0, max_value=_max_pos_pct, format='%.0f%%',
                ),
                'score_top_n_ratio_pct': st.column_config.ProgressColumn(
                    f'점수 Top{top_rank_n}', min_value=0, max_value=_max_topn_pct, format='%.0f%%',
                ),
                'market_cap_str': st.column_config.TextColumn('시총'),
                'market_cap_rank': st.column_config.NumberColumn('시총순위', format='%d위'),
            },
            use_container_width=True,
            hide_index=True,
            height=min(600, len(composite_sorted) * 38 + 40),
        )
    else:
        st.info("데이터가 없습니다.")

# ─── 탭 2: 종합점수 평균 ─────────────────────────────────────────────────────
with tab_score:
    st.caption(f"{lookback_label} 일별 final_score(= 점수 + 시그널×5)의 평균·최고·상위 출현빈도")

    if 'avg_score' in merged.columns:
        score_sorted = merged.sort_values('avg_score', ascending=False).head(display_n)

        if not score_sorted.empty:
            fig = create_cumulative_bar_chart(
                score_sorted, 'avg_score', title=f'평균 점수 Top {_CHART_TOP_N}',
                top_n=_CHART_TOP_N, value_format='.1f', color=_COLOR_SCORE,
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

            _tbl2_cols = ['display_name', 'sector', 'avg_score', 'max_score',
                          'score_top_n_ratio_pct', 'appearance_ratio_pct', 'market_cap_str']
            _tbl2_cols = [c for c in _tbl2_cols if c in score_sorted.columns]

            _max_topn2 = max(score_sorted['score_top_n_ratio_pct'].max(), 1) if 'score_top_n_ratio_pct' in score_sorted.columns else 100
            _max_appear = max(score_sorted['appearance_ratio_pct'].max(), 1) if 'appearance_ratio_pct' in score_sorted.columns else 100

            st.dataframe(
                score_sorted[_tbl2_cols].reset_index(drop=True),
                column_config={
                    'display_name': st.column_config.TextColumn('종목'),
                    'sector': st.column_config.TextColumn('섹터'),
                    'avg_score': st.column_config.NumberColumn('평균 점수', format='%.1f점'),
                    'max_score': st.column_config.NumberColumn('최고 점수', format='%.1f점'),
                    'score_top_n_ratio_pct': st.column_config.ProgressColumn(
                        f'Top{top_rank_n} 출현빈도', min_value=0, max_value=_max_topn2, format='%.0f%%',
                    ),
                    'appearance_ratio_pct': st.column_config.ProgressColumn(
                        '등장 비율', min_value=0, max_value=_max_appear, format='%.0f%%',
                    ),
                    'market_cap_str': st.column_config.TextColumn('시총'),
                },
                use_container_width=True,
                hide_index=True,
                height=min(600, len(score_sorted) * 38 + 40),
            )
        else:
            st.info("데이터가 없습니다.")
    else:
        st.info("점수 데이터가 없습니다.")

# ─── 탭 3: 누적 수급강도 ─────────────────────────────────────────────────────
with tab_sff:
    _sff_label = "누적 매수 강도" if direction_key == 'long' else "누적 매도 강도"
    st.caption(f"{lookback_label} combined_sff 합계 + 순매수금액 합계")

    if 'cum_sff' in merged.columns:
        sff_sorted = merged.sort_values('cum_sff', ascending=False).head(display_n)

        if not sff_sorted.empty:
            fig = create_cumulative_bar_chart(
                sff_sorted, 'cum_sff', title=f'{_sff_label} Top {_CHART_TOP_N}',
                top_n=_CHART_TOP_N, value_format='.4f', color=_COLOR_SFF,
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)

            # 순매수금액 포맷
            sff_display = sff_sorted.copy()
            if 'cum_net_amount' in sff_display.columns:
                sff_display['cum_net_amount_str'] = sff_display['cum_net_amount'].apply(_fmt_net_amount)

            _tbl3_cols = ['display_name', 'sector', 'cum_sff_norm', 'cum_sff',
                          'avg_sff', 'cum_net_amount_str', 'market_cap_str']
            _tbl3_cols = [c for c in _tbl3_cols if c in sff_display.columns]

            st.dataframe(
                sff_display[_tbl3_cols].reset_index(drop=True),
                column_config={
                    'display_name': st.column_config.TextColumn('종목'),
                    'sector': st.column_config.TextColumn('섹터'),
                    'cum_sff_norm': st.column_config.ProgressColumn(
                        '수급 강도', min_value=0, max_value=100, format='%.0f',
                    ),
                    'cum_sff': st.column_config.NumberColumn('누적 Sff', format='%.4f'),
                    'avg_sff': st.column_config.NumberColumn('평균 Sff', format='%.5f'),
                    'cum_net_amount_str': st.column_config.TextColumn('누적 순매수'),
                    'market_cap_str': st.column_config.TextColumn('시총'),
                },
                use_container_width=True,
                hide_index=True,
                height=min(600, len(sff_display) * 38 + 40),
            )
        else:
            st.info("데이터가 없습니다.")
    else:
        st.info("수급 데이터가 없습니다.")

# ─── 탭 4: 수급 안정성 & 출현빈도 ───────────────────────────────────────────
with tab_stability:
    _dir_sff_desc = "순매수(combined_sff > 0)" if direction_key == 'long' else "순매도(원본 combined_sff < 0)"
    st.caption(f"{_dir_sff_desc}인 날 비율 + 일별 Sff 상위 {top_rank_n}위 출현 비율")

    if 'positive_ratio' in merged.columns:
        stab_col1, stab_col2 = st.columns(2)

        with stab_col1:
            st.markdown(f"##### {'매수' if direction_key == 'long' else '매도'} 일수 비율")
            pos_sorted = merged.sort_values('positive_ratio', ascending=False).head(display_n)

            if not pos_sorted.empty:
                _max_pos4 = max(pos_sorted['positive_ratio_pct'].max(), 1)

                fig = create_cumulative_bar_chart(
                    pos_sorted, 'positive_ratio_pct',
                    title=f'양수 비율 Top {_CHART_TOP_N}',
                    top_n=_CHART_TOP_N, value_format='.0f', color=_COLOR_STABILITY,
                )
                st.plotly_chart(fig, use_container_width=True, theme=None)

                st.dataframe(
                    pos_sorted[['display_name', 'sector', 'positive_ratio_pct']].reset_index(drop=True),
                    column_config={
                        'display_name': st.column_config.TextColumn('종목'),
                        'sector': st.column_config.TextColumn('섹터'),
                        'positive_ratio_pct': st.column_config.ProgressColumn(
                            '양수 비율', min_value=0, max_value=_max_pos4, format='%.0f%%',
                        ),
                    },
                    use_container_width=True,
                    hide_index=True,
                    height=min(500, len(pos_sorted) * 38 + 40),
                )

        with stab_col2:
            st.markdown(f"##### Sff Top {top_rank_n} 출현 빈도")
            freq_sorted = merged.sort_values('sff_top_n_ratio', ascending=False).head(display_n)

            if not freq_sorted.empty:
                _max_freq4 = max(freq_sorted['sff_top_n_ratio_pct'].max(), 1)

                fig = create_cumulative_bar_chart(
                    freq_sorted, 'sff_top_n_ratio_pct',
                    title=f'Sff Top{top_rank_n} 출현 빈도 Top {_CHART_TOP_N}',
                    top_n=_CHART_TOP_N, value_format='.0f', color=_COLOR_STABILITY,
                )
                st.plotly_chart(fig, use_container_width=True, theme=None)

                st.dataframe(
                    freq_sorted[['display_name', 'sector', 'sff_top_n_ratio_pct']].reset_index(drop=True),
                    column_config={
                        'display_name': st.column_config.TextColumn('종목'),
                        'sector': st.column_config.TextColumn('섹터'),
                        'sff_top_n_ratio_pct': st.column_config.ProgressColumn(
                            f'Top{top_rank_n} 출현빈도', min_value=0, max_value=_max_freq4, format='%.0f%%',
                        ),
                    },
                    use_container_width=True,
                    hide_index=True,
                    height=min(500, len(freq_sorted) * 38 + 40),
                )
    else:
        st.info("안정성 데이터가 없습니다.")

# ---------------------------------------------------------------------------
# 종목 상세 이동
# ---------------------------------------------------------------------------
st.divider()
_detail_options = merged.sort_values('composite', ascending=False)['display_name'].tolist()
if _detail_options:
    _detail_col1, _detail_col2 = st.columns([3, 1])
    with _detail_col1:
        _detail_sel = st.selectbox("종목 선택", _detail_options, index=0, key="cum_detail_stock")
    with _detail_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📋 종목 상세 보기 →", key="cum_to_detail"):
            _code = _detail_sel.split('(')[-1].rstrip(')')
            st.session_state['heatmap_selected_code'] = _code
            st.switch_page("pages/5_📋_종목상세.py")
