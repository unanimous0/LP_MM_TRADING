"""
Stage 5-1: Streamlit 웹 대시보드 - 홈 페이지

KPI 카드, 이상 수급, 수급 순위, 패턴 분포 차트, 관심 종목 테이블을 표시하는 대시보드 메인 페이지.
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 등록
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
from datetime import datetime

from utils.data_loader import (
    run_analysis_pipeline_with_progress,
    get_date_range,
    get_abnormal_supply_data,
    get_today_supply_ranking,
)
from utils.charts import (
    create_pattern_pie_chart,
    create_score_histogram,
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

# CSS: 이상 수급 섹션 색상 테두리
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

st.title("Whale Supply")
st.caption("외국인/기관 투자자 수급 기반 종목 분석 시스템")

# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
min_date, max_date = get_date_range()
st.sidebar.markdown(f"**DB 기간**: {min_date} ~ {max_date}")
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

_prog = st.progress(0, text="분석 준비 중... 0%")
zscore_matrix, classified_df, signals_df, report_df = run_analysis_pipeline_with_progress(
    progress_bar=_prog,
    institution_weight=institution_weight,
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

# ---------------------------------------------------------------------------
# 헤더 + 기준일
# ---------------------------------------------------------------------------
st.markdown(f"**기준일**: {end_date_str}")

# ---------------------------------------------------------------------------
# KPI 카드 (5개)
# ---------------------------------------------------------------------------
total = len(report_df)
watchlist_df = report_df[
    (report_df['score'] >= 70) & (report_df['signal_count'] >= 2)
].copy()
signal_2plus = len(report_df[report_df['signal_count'] >= 2])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("분석 종목", f"{total}개")
col2.metric("관심 종목", f"{len(watchlist_df)}개", help="점수 70+ & 시그널 2+")
col3.metric("강한 매수", f"{len(abnormal_buy)}개", help="Z-Score > 2σ")
col4.metric("강한 매도", f"{len(abnormal_sell)}개", help="Z-Score < -2σ")
col5.metric("시그널 2+", f"{signal_2plus}개", help="시그널 2개 이상 종목")

st.divider()

# ---------------------------------------------------------------------------
# 수급 탭: 이상 수급 / 수급 순위
# ---------------------------------------------------------------------------

tab_abnormal, tab_ranking = st.tabs([
    "이상 수급 (Z-Score > 2σ)",
    "당일 수급 순위",
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

st.divider()

# ---------------------------------------------------------------------------
# 관심 종목 테이블 (score>=70, signal_count>=2)
# ---------------------------------------------------------------------------
st.subheader("관심 종목 (점수 70+, 시그널 2+)")

if watchlist_df.empty:
    st.info("현재 조건을 만족하는 관심 종목이 없습니다.")
else:
    display_cols = [
        'stock_code', 'stock_name', 'sector', 'pattern',
        'score', 'signal_count', 'entry_point', 'stop_loss',
    ]
    display_cols = [c for c in display_cols if c in watchlist_df.columns]

    col_config = {
        'stock_code': st.column_config.TextColumn('종목코드'),
        'stock_name': st.column_config.TextColumn('종목명'),
        'sector': st.column_config.TextColumn('섹터'),
        'pattern': st.column_config.TextColumn('패턴'),
        'score': st.column_config.ProgressColumn(
            '최종점수', min_value=0, max_value=115, format='%d점',
        ),
        'signal_count': st.column_config.NumberColumn('시그널', format='%d개'),
        'entry_point': st.column_config.NumberColumn('진입가', format='₩%d'),
        'stop_loss': st.column_config.NumberColumn('손절가', format='₩%d'),
    }
    col_config = {k: v for k, v in col_config.items() if k in display_cols}

    st.dataframe(
        watchlist_df[display_cols].reset_index(drop=True),
        column_config=col_config,
        use_container_width=True,
        hide_index=True,
        height=min(500, len(watchlist_df) * 40 + 40),
    )

    st.caption(f"총 {len(watchlist_df)}개 종목")
