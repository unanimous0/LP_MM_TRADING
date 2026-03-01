"""
이상 수급 페이지 — 이상수급 / 당일 수급순위 / 수급금액 / 고득점 변동알림

메인 페이지에서 분리된 참고 데이터 페이지.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from utils.data_loader import (
    run_analysis_pipeline_with_progress,
    get_date_range,
    get_abnormal_supply_data,
    get_today_supply_ranking,
    snapshot_scores,
    get_score_change_alerts,
    get_stock_list,
    get_stock_raw_history,
    get_stock_zscore_history,
)
from utils.charts import (
    create_abnormal_supply_chart,
    create_supply_ranking_chart,
    create_supply_amount_chart,
)

st.set_page_config(page_title="이상 수급", page_icon="⚡", layout="wide")

# CSS: 매수/매도 섹션 색상 테두리
st.markdown("""
<style>
/* 매수 섹션 (green) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(
    [style*="4ade80"]
) { border-color: #4ade80 !important; }
/* 매도 섹션 (red) */
div[data-testid="stVerticalBlockBorderWrapper"]:has(
    [style*="f87171"]
) { border-color: #f87171 !important; }
</style>
""", unsafe_allow_html=True)

st.title("이상 수급")
st.caption("Z-Score 기반 이상 수급 탐지 + 당일 수급 순위 + 수급 금액 조회 + 고득점 변동 알림")

# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
min_date, max_date = get_date_range()
_max_dt = datetime.strptime(max_date, "%Y-%m-%d")
end_date = st.sidebar.date_input(
    "이상 수급 기준일",
    value=_max_dt,
    min_value=datetime.strptime(min_date, "%Y-%m-%d"),
    max_value=_max_dt.replace(month=12, day=31),
    help="이상 수급 탐지 기준 날짜. 과거 날짜를 선택하면 해당 시점의 이상 수급을 볼 수 있습니다.",
)
end_date_str = end_date.strftime("%Y-%m-%d")
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
z_score_window = st.sidebar.slider(
    "Z-Score 기준 기간 (거래일)",
    min_value=20, max_value=240, value=60, step=10,
    help="이상 수급 판단 시 평균/표준편차 계산에 사용하는 과거 거래일 수 (기본 60일 = 약 3개월)",
)

# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
_prog = st.progress(0, text="분석 준비 중... 0%")
zscore_matrix, classified_df, signals_df, report_df = run_analysis_pipeline_with_progress(
    progress_bar=_prog,
    institution_weight=institution_weight,
    end_date=end_date_str,
)

if report_df.empty:
    _prog.empty()
    st.warning("분석 데이터가 없습니다. DB를 확인하세요.")
    st.stop()

# 이상 수급 데이터 로드
_prog.progress(0.90, text="이상 수급 탐지 중... 90%")
abnormal_buy = get_abnormal_supply_data(end_date=end_date_str, threshold=2.0, top_n=30, direction='buy', institution_weight=institution_weight, z_score_window=z_score_window)
abnormal_sell = get_abnormal_supply_data(end_date=end_date_str, threshold=2.0, top_n=30, direction='sell', institution_weight=institution_weight, z_score_window=z_score_window)

# 당일 수급 순위 데이터 로드
_prog.progress(0.95, text="당일 수급 순위 조회 중... 95%")
supply_ranking = get_today_supply_ranking()
_prog.progress(1.0, text="완료 100%")
_prog.empty()

# 분석 완료 후 고득점 변동 스냅샷 저장 (세션당 1회)
if not st.session_state.get('abnormal_snapshot_done'):
    try:
        _, _latest_date = get_date_range()
        snapshot_scores(report_df, _latest_date)
        st.session_state['abnormal_snapshot_done'] = True
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 기준일 표시
# ---------------------------------------------------------------------------
st.markdown(f"**기준일**: {end_date_str}")

# ---------------------------------------------------------------------------
# 4탭: 이상 수급 / 당일 수급 순위 / 수급 금액 / 고득점 변동 알림
# ---------------------------------------------------------------------------
tab_abnormal, tab_ranking, tab_supply_amount, tab_alerts = st.tabs([
    "이상 수급 (Z > 2σ)",
    "당일 수급 순위",
    "수급 금액",
    "고득점 변동 알림",
])

# ─── 탭 1: 이상 수급 ─────────────────────────────────────────────────────────
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

# ─── 탭 2: 당일 수급 순위 ────────────────────────────────────────────────────
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

# ─── 탭 3: 수급 금액 ──────────────────────────────────────────────────────────
with tab_supply_amount:
    st.caption("종목을 선택하면 외국인/기관/개인 순매수 금액 차트와 상세 테이블을 확인할 수 있습니다.")

    # 종목 선택 (이상수급 종목 + 전체 종목)
    _abnormal_codes = set()
    _quick_options = []
    if not abnormal_buy.empty:
        for _, r in abnormal_buy.iterrows():
            opt = f"{r['stock_name']} ({r['stock_code']})"
            if r['stock_code'] not in _abnormal_codes:
                _quick_options.append(opt)
                _abnormal_codes.add(r['stock_code'])
    if not abnormal_sell.empty:
        for _, r in abnormal_sell.iterrows():
            opt = f"{r['stock_name']} ({r['stock_code']})"
            if r['stock_code'] not in _abnormal_codes:
                _quick_options.append(opt)
                _abnormal_codes.add(r['stock_code'])

    # 전체 종목 리스트
    stock_list = get_stock_list()
    all_options = [
        f"{row['stock_name']} ({row['stock_code']})"
        for _, row in stock_list.iterrows()
    ]

    # 이상수급 종목을 앞에, 나머지를 뒤에 배치
    _remaining = [opt for opt in all_options if opt not in set(_quick_options)]
    combined_options = _quick_options + _remaining

    _sa_selected = st.selectbox(
        "종목 선택",
        combined_options,
        index=0 if combined_options else None,
        help="이상수급 탐지 종목이 상단에 표시됩니다. 아무 종목이나 검색 가능합니다.",
        key="abnormal_supply_stock",
    )

    if _sa_selected:
        _sa_code = _sa_selected.split('(')[-1].rstrip(')')
        _sa_name = _sa_selected.rsplit(' (', 1)[0]

        # 표시 기간
        _SA_PERIOD_DAYS = {"3개월": 90, "6개월": 180, "1년": 365, "전체": None}
        _sa_period = st.selectbox("표시 기간", list(_SA_PERIOD_DAYS.keys()), index=1, key="sa_period")
        _sa_days = _SA_PERIOD_DAYS[_sa_period]
        _sa_start = (end_date - timedelta(days=_sa_days)).strftime("%Y-%m-%d") if _sa_days else None

        # 데이터 로드
        raw_df = get_stock_raw_history(_sa_code, end_date_str)

        if raw_df.empty:
            st.warning(f"'{_sa_name}' ({_sa_code})의 수급 데이터가 없습니다.")
        else:
            # 수급 금액 차트
            fig = create_supply_amount_chart(raw_df, start_date=_sa_start)
            st.plotly_chart(fig, width="stretch", theme=None)

            # Z-Score 조인용 데이터
            _effective_window = min(z_score_window, len(raw_df))
            zscore_df = get_stock_zscore_history(_sa_code, end_date_str, institution_weight, _effective_window)

            # 수급 상세 테이블 (종목상세 페이지와 동일)
            display_raw = (raw_df if not _sa_start
                           else raw_df[raw_df['trade_date'] >= _sa_start]).copy()
            if not display_raw.empty:
                display_raw = display_raw.reset_index(drop=True)
                indiv = (display_raw.get('individual_net_amount',
                                         -(display_raw['foreign_net_amount']
                                           + display_raw['institution_net_amount'])
                                         ).fillna(0))
                f_eok = display_raw['foreign_net_amount'].fillna(0) / 1e8
                i_eok = display_raw['institution_net_amount'].fillna(0) / 1e8
                p_eok = indiv / 1e8

                # Z-Score 조인 (종합 Z 제외)
                z_display = (
                    zscore_df[['trade_date', 'foreign_zscore', 'institution_zscore']]
                    if not zscore_df.empty else pd.DataFrame(columns=['trade_date', 'foreign_zscore', 'institution_zscore'])
                )

                tbl = pd.DataFrame({
                    '날짜':        display_raw['trade_date'],
                    '외국인 순매수': f_eok.round(0).astype(int),
                    '외국인 누적':  f_eok.cumsum().round(0).astype(int),
                    '기관 순매수':  i_eok.round(0).astype(int),
                    '기관 누적':   i_eok.cumsum().round(0).astype(int),
                    '개인 순매수':  p_eok.round(0).astype(int),
                    '개인 누적':   p_eok.cumsum().round(0).astype(int),
                })

                if not z_display.empty:
                    tbl = tbl.merge(
                        z_display.rename(columns={
                            'trade_date': '날짜',
                            'foreign_zscore': '외국인 Z',
                            'institution_zscore': '기관 Z',
                        }),
                        on='날짜', how='left',
                    )
                    tbl = tbl[['날짜',
                                '외국인 순매수', '외국인 Z', '외국인 누적',
                                '기관 순매수',   '기관 Z',   '기관 누적',
                                '개인 순매수',   '개인 누적']]
                else:
                    tbl = tbl[['날짜',
                                '외국인 순매수', '외국인 누적',
                                '기관 순매수',   '기관 누적',
                                '개인 순매수',   '개인 누적']]

                tbl = tbl.sort_values('날짜', ascending=False).reset_index(drop=True)

                st.caption(f"단위: 억원 · 누적은 표시 기간 시작일 기준 · Z-Score 기준 기간: {_effective_window}거래일 · 순매수/누적: 🟢 양수  🔴 음수 · Z-Score: 🟡 ≥+2σ  🔵 ≤-2σ")

                # ── 그룹 정의 (종목상세와 동일한 HTML 테이블)
                _GROUPS = [
                    ('외국인', ['외국인 순매수', '외국인 Z', '외국인 누적'],  '#38bdf8', '#0d1e2c'),
                    ('기관',   ['기관 순매수',   '기관 Z',   '기관 누적'],   '#f472b6', '#1e0e1c'),
                    ('개인',   ['개인 순매수',   '개인 누적'],               '#fb923c', '#1e1408'),
                ]
                _GROUPS = [(g, [c for c in cols if c in tbl.columns], clr, bg)
                           for g, cols, clr, bg in _GROUPS]

                _col_group = {}
                for g, cols, clr, bg in _GROUPS:
                    for idx, c in enumerate(cols):
                        _col_group[c] = (clr, bg, idx == 0)

                _z_cols = {'외국인 Z', '기관 Z'}
                _num_cols = {'외국인 순매수', '외국인 누적', '기관 순매수', '기관 누적', '개인 순매수', '개인 누적'}

                # ── 헤더 2행
                _th_date = (
                    "padding:6px 10px; text-align:center; font-weight:700; font-size:14px;"
                    "background:#1e293b; color:#94a3b8; border-bottom:1px solid #334155;"
                    "white-space:nowrap; vertical-align:middle;"
                )
                group_header_cells = f'<th rowspan="2" style="{_th_date}">날짜</th>'
                for g, cols, clr, bg in _GROUPS:
                    if not cols:
                        continue
                    group_header_cells += (
                        f'<th colspan="{len(cols)}" style="'
                        f'padding:5px 10px; text-align:center; font-weight:700; font-size:13px;'
                        f'background:{bg}; color:{clr}; border-bottom:2px solid {clr};'
                        f'border-left:2px solid {clr}; white-space:nowrap;">{g}</th>'
                    )

                sub_header_cells = ""
                for c in tbl.columns:
                    if c == '날짜':
                        continue
                    clr, bg, is_first = _col_group.get(c, ('#e2e8f0', '#1e293b', False))
                    border_left = f"border-left:2px solid {clr};" if is_first else ""
                    sub_label = c.replace('외국인 ', '').replace('기관 ', '').replace('개인 ', '')
                    sub_header_cells += (
                        f'<th style="padding:5px 10px; text-align:center; font-weight:600; font-size:13px;'
                        f'background:#1e293b; color:#94a3b8; border-bottom:1px solid #334155;'
                        f'{border_left} white-space:nowrap;">{sub_label}</th>'
                    )

                # ── 데이터 행
                rows_html = ""
                for i, (_, row) in enumerate(tbl.iterrows()):
                    base_bg = '#162032' if i % 2 == 1 else 'transparent'
                    cells = ""
                    cells += (
                        f'<td style="padding:5px 10px; text-align:center; font-size:14px;'
                        f'color:#94a3b8; border-bottom:1px solid #1e293b; background:{base_bg};'
                        f'white-space:nowrap;">{row["날짜"]}</td>'
                    )
                    for c in tbl.columns:
                        if c == '날짜':
                            continue
                        v = row[c]
                        clr, bg, is_first = _col_group.get(c, ('#e2e8f0', 'transparent', False))
                        border_left = f"border-left:2px solid {clr};" if is_first else ""
                        cell_bg = base_bg
                        td_base = (
                            f"padding:5px 10px; text-align:center; font-size:14px;"
                            f"border-bottom:1px solid #1e293b; background:{cell_bg};"
                            f"{border_left} white-space:nowrap;"
                        )
                        if c in _num_cols:
                            val_color = '#4ade80' if int(v) > 0 else ('#f87171' if int(v) < 0 else '#e2e8f0')
                            cells += f'<td style="{td_base}color:{val_color};">{int(v):,}</td>'
                        elif c in _z_cols:
                            if pd.notna(v):
                                z_color = '#fbbf24' if v >= 2 else ('#7dd3fc' if v <= -2 else '#94a3b8')
                                z_bold  = 'font-weight:700;' if abs(v) >= 2 else ''
                                cells += f'<td style="{td_base}color:{z_color};{z_bold}">{v:.2f}</td>'
                            else:
                                cells += f'<td style="{td_base}color:#64748b;">-</td>'
                        else:
                            disp = v if pd.notna(v) else '-'
                            cells += f'<td style="{td_base}color:#e2e8f0;">{disp}</td>'
                    rows_html += f"<tr>{cells}</tr>"

                html_table = f"""
<div style="overflow-x:auto; max-height:420px; overflow-y:auto;
            border:1px solid #334155; border-radius:6px;">
  <table style="width:100%; border-collapse:collapse; font-size:14px;">
    <thead style="position:sticky; top:0; z-index:10;">
      <tr>{group_header_cells}</tr>
      <tr>{sub_header_cells}</tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
"""
                st.markdown(html_table, unsafe_allow_html=True)

# ─── 탭 4: 고득점 변동 알림 ────────────────────────────────────────────────────
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
