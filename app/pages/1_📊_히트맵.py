"""
히트맵 페이지 - Z-Score 인터랙티브 히트맵

사이드바: 정렬 기준, 표시 종목 수, 섹터 필터
메인: Plotly 인터랙티브 히트맵 (줌/호버)
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
from utils.charts import create_zscore_heatmap

st.set_page_config(page_title="히트맵", page_icon="📊", layout="wide")
st.title("Z-Score 수급 히트맵")

# ---------------------------------------------------------------------------
# 사이드바 필터
# ---------------------------------------------------------------------------
min_date, max_date = get_date_range()
institution_weight = st.sidebar.slider(
    "기관 가중치", 0.0, 1.0, 0.3, step=0.05,
    key="w_institution_weight",
    help="기관 수급 반영 비율 (0=외국인만, 0.3=기본, 1.0=동등)",
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

sort_options = {
    'recent': '최근 수급 (1W 기준)',
    'momentum': '모멘텀 (단기-장기 차이)',
    'weighted': '가중 평균 (최근 높은 비중)',
    'average': '단순 평균',
}
sort_by = st.sidebar.selectbox(
    "정렬 기준",
    options=list(sort_options.keys()),
    format_func=lambda x: sort_options[x],
)

top_n = st.sidebar.slider("표시 종목 수", min_value=10, max_value=200, value=50, step=10)

sectors = get_sectors()
selected_sector = st.sidebar.selectbox("섹터 필터", options=["전체"] + sectors)

# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
_prog = st.progress(0, text="분석 준비 중... 0%")
zscore_matrix, classified_df, signals_df, report_df = run_analysis_pipeline_with_progress(
    end_date=end_date_str, progress_bar=_prog,
    institution_weight=institution_weight,
)
_prog.empty()

if zscore_matrix.empty:
    st.warning("Z-Score 데이터가 없습니다.")
    st.stop()

# 섹터 필터링
if selected_sector != "전체":
    stock_list = get_stock_list()
    sector_stocks = stock_list[stock_list['sector'] == selected_sector]['stock_code'].tolist()
    zscore_matrix = zscore_matrix[zscore_matrix['stock_code'].isin(sector_stocks)]

    if zscore_matrix.empty:
        st.info(f"'{selected_sector}' 섹터에 해당하는 종목이 없습니다.")
        st.stop()

# ---------------------------------------------------------------------------
# 통계 (히트맵 위)
# ---------------------------------------------------------------------------
period_cols = [c for c in zscore_matrix.columns if c != 'stock_code']
if '1W' in period_cols:
    col1, col2, col3 = st.columns(3)
    col1.metric("표시 종목 수", f"{min(top_n, len(zscore_matrix))}개")
    col2.metric("평균 1W Z-Score", f"{zscore_matrix['1W'].mean():.2f}")
    strong_buy = (zscore_matrix['1W'] > 2).sum()
    col3.metric("강한 매수 (Z>2)", f"{strong_buy}개")

# ---------------------------------------------------------------------------
# 히트맵
# ---------------------------------------------------------------------------
stock_names = get_stock_list()
fig = create_zscore_heatmap(zscore_matrix, sort_by=sort_by, top_n=top_n, stock_names=stock_names)
st.plotly_chart(fig, width="stretch", theme=None)
