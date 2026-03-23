"""
히트맵 페이지 - Z-Score 인터랙티브 히트맵

고도화 기능:
  A. 히트맵 클릭 → 하단 미니 상세 (KPI + Z-Score 바차트 + 상세 페이지 이동 버튼)
  B. 호버에 패턴/점수/시그널 정보 표시
  D. 섹터 평균 히트맵 탭 추가
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
from datetime import datetime

from utils.data_loader import run_analysis_pipeline_with_progress, get_stock_list, get_sectors, get_date_range
from utils.charts import (
    create_zscore_heatmap,
    create_sector_zscore_heatmap,
    create_multiperiod_zscore_bar,
)

st.set_page_config(page_title="히트맵", page_icon="📊", layout="wide")
st.markdown('<style>div[data-baseweb="select"]>div{border-color:#333!important}div[data-baseweb="input"] input,div[data-baseweb="input"]>div{border-color:#333!important}[data-testid="stDateInput"]>div>div>div{border-color:#333!important}[data-testid="stExpander"]{border-color:#222!important}</style>', unsafe_allow_html=True)
st.title("Z-Score 수급 히트맵")

# ---------------------------------------------------------------------------
# 사이드바 필터
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
    help="해당 날짜 기준으로 Z-Score를 계산합니다. 과거 날짜를 선택하면 당시 수급 상태를 볼 수 있습니다.",
)
end_date_str = end_date.strftime("%Y-%m-%d")

st.sidebar.divider()

direction = st.sidebar.radio(
    "수급 방향",
    options=['buy', 'sell', 'both'],
    format_func=lambda x: {'buy': '매수 상위', 'sell': '매도 상위', 'both': '양쪽'}[x],
    horizontal=True,
    help="매수 상위: Z-Score 높은 순 / 매도 상위: Z-Score 낮은 순 / 양쪽: 각 절반씩",
)

sort_options = {
    'recent':          '최근 수급 강도',
    'long_divergence': '장기 대비 변화 (5D-200D)',
    'weighted':        '종합 수급 (가중 평균)',
}
sort_by = st.sidebar.selectbox(
    "정렬 기준",
    options=list(sort_options.keys()),
    format_func=lambda x: sort_options[x],
)

top_n = st.sidebar.slider("표시 종목 수", min_value=10, max_value=200, value=50, step=10)

st.sidebar.divider()

sectors = get_sectors()
selected_sector = st.sidebar.selectbox("섹터 필터", options=["전체"] + sectors)

# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
_prog = st.progress(0, text="분석 준비 중... 0%")
_pipeline_direction = 'long' if direction in ('buy', 'both') else 'short'
zscore_matrix, classified_df, signals_df, report_df = run_analysis_pipeline_with_progress(
    end_date=end_date_str, progress_bar=_prog,
    institution_weight=institution_weight,
    direction=_pipeline_direction,
)
_prog.empty()

if zscore_matrix.empty:
    st.warning("Z-Score 데이터가 없습니다.")
    st.stop()

stock_list = get_stock_list()

# 섹터 필터링
if selected_sector != "전체":
    sector_stocks = stock_list[stock_list['sector'] == selected_sector]['stock_code'].tolist()
    zscore_matrix = zscore_matrix[zscore_matrix['stock_code'].isin(sector_stocks)]
    if not report_df.empty:
        report_df = report_df[report_df['stock_code'].isin(sector_stocks)]
    if zscore_matrix.empty:
        st.info(f"'{selected_sector}' 섹터에 해당하는 종목이 없습니다.")
        st.stop()

# ---------------------------------------------------------------------------
# 통계 (히트맵 위)
# ---------------------------------------------------------------------------
period_cols = [c for c in zscore_matrix.columns if c != 'stock_code' and not c.startswith('_')]
if '5D' in period_cols:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("표시 종목 수", f"{min(top_n, len(zscore_matrix))}개")
    col2.metric("평균 5D Z-Score", f"{zscore_matrix['5D'].mean():.2f}")
    strong_buy = (zscore_matrix['5D'] > 2).sum()
    strong_sell = (zscore_matrix['5D'] < -2).sum()
    col3.metric("강한 매수 (Z>2)", f"{strong_buy}개")
    col4.metric("강한 매도 (Z<-2)", f"{strong_sell}개")

# ---------------------------------------------------------------------------
# D: 탭 구조 (종목별 히트맵 | 섹터 평균 히트맵)
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["종목별 히트맵", "섹터 평균 히트맵"])

with tab1:
    # B: report_df 전달 → 호버에 패턴/점수/시그널 표시
    fig = create_zscore_heatmap(
        zscore_matrix, sort_by=sort_by, top_n=top_n,
        stock_names=stock_list, direction=direction,
        report_df=report_df,
    )

    # A: on_select="rerun" — 클릭 시 하단 미니 상세 표시
    event = st.plotly_chart(
        fig, width="stretch", theme=None,
        on_select="rerun", selection_mode="points",
        key="heatmap_main",
    )

    # A: 클릭된 종목 코드 추출
    selected_code = None
    selected_label = None
    try:
        pts = event.selection.points
        if pts:
            y_label = str(pts[0].get('y', '') or '')
            # y_label 형식: "종목명(종목코드)" 또는 그냥 "종목코드"
            if '(' in y_label and y_label.endswith(')'):
                selected_code = y_label.split('(')[-1][:-1]
                selected_label = y_label
            elif y_label:
                selected_code = y_label
                selected_label = y_label
    except Exception:
        pass

    # A: 미니 상세 패널
    if selected_code:
        st.divider()

        # 종목명 조회
        _name_mask = stock_list['stock_code'] == selected_code
        _stock_name = (
            stock_list.loc[_name_mask, 'stock_name'].values[0]
            if _name_mask.any() else selected_code
        )

        _hdr_col, _btn_col = st.columns([4, 1])
        with _hdr_col:
            st.subheader(f"📌 {_stock_name} ({selected_code})")
        with _btn_col:
            if st.button("종목 상세 보기 →", type="primary", key="goto_detail"):
                st.session_state['heatmap_selected_code'] = selected_code
                st.switch_page("pages/5_📋_종목상세.py")

        # KPI 4개
        _m1, _m2, _m3, _m4 = st.columns(4)

        _zrow_mask = zscore_matrix['stock_code'] == selected_code
        if _zrow_mask.any():
            _zrow = zscore_matrix[_zrow_mask].iloc[0]
            _1w = float(_zrow['5D']) if '5D' in _zrow.index else float('nan')
            _m1.metric("5D Z-Score", f"{_1w:.2f}σ" if pd.notna(_1w) else "-")
        else:
            _m1.metric("5D Z-Score", "-")

        _rrow = None
        if not report_df.empty and selected_code in report_df['stock_code'].values:
            _rrow = report_df[report_df['stock_code'] == selected_code].iloc[0]
            _m2.metric("패턴", str(_rrow.get('pattern_label', _rrow.get('pattern', '-'))))
            _m3.metric("점수", f"{float(_rrow.get('score', 0)):.0f}")
            _m4.metric("시그널", f"{int(_rrow.get('signal_count', 0))}개")
        else:
            _m2.metric("패턴", "-")
            _m3.metric("점수", "-")
            _m4.metric("시그널", "-")

        # 멀티기간 Z-Score 바차트
        if _zrow_mask.any():
            _fig_bar = create_multiperiod_zscore_bar(_zrow)
            st.plotly_chart(_fig_bar, width="stretch", theme=None, key="mini_zscore_bar")

        # 활성 시그널 표시
        if _rrow is not None:
            _sig = _rrow.get('signal_list', '')
            if _sig and str(_sig) not in ('', 'nan', 'None'):
                st.success(f"**활성 시그널**: {_sig}")

with tab2:
    # D: 섹터 평균 히트맵
    st.caption("섹터별 종목들의 평균 Z-Score. 섹터 필터가 적용된 경우 해당 섹터만 표시됩니다.")
    fig_sector = create_sector_zscore_heatmap(
        zscore_matrix, stock_list=stock_list, sort_by=sort_by,
    )
    st.plotly_chart(fig_sector, width="stretch", theme=None)
