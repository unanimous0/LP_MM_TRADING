"""
Stage 5-1: Streamlit 웹 대시보드 - 홈 페이지

KPI 카드, 패턴 분포 차트, 관심 종목 테이블을 표시하는 대시보드 메인 페이지.
"""

import sys
from pathlib import Path

# 프로젝트 루트 경로 등록
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd

from utils.data_loader import run_analysis_pipeline_with_progress, get_date_range
from utils.charts import create_pattern_pie_chart, create_score_histogram

# ---------------------------------------------------------------------------
# 페이지 설정
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="수급 분석 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("수급 분석 대시보드")
st.caption("외국인/기관 투자자 수급 기반 종목 분석 시스템")

# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
min_date, max_date = get_date_range()
st.sidebar.markdown(f"**DB 기간**: {min_date} ~ {max_date}")

_prog = st.progress(0, text="분석 준비 중... 0%")
zscore_matrix, classified_df, signals_df, report_df = run_analysis_pipeline_with_progress(
    progress_bar=_prog,
)
_prog.empty()

if report_df.empty:
    st.warning("분석 데이터가 없습니다. DB를 확인하세요.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI 카드
# ---------------------------------------------------------------------------
total = len(report_df)
pattern_counts = report_df['pattern'].value_counts()

col1, col2, col3, col4 = st.columns(4)
col1.metric("전체 종목", f"{total}개")
col2.metric("모멘텀형", f"{pattern_counts.get('모멘텀형', 0)}개")
col3.metric("지속형", f"{pattern_counts.get('지속형', 0)}개")
col4.metric("전환형", f"{pattern_counts.get('전환형', 0)}개")

# ---------------------------------------------------------------------------
# 차트 (2열)
# ---------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fig_pie = create_pattern_pie_chart(report_df)
    st.plotly_chart(fig_pie, use_container_width=True, theme=None)

with chart_col2:
    fig_hist = create_score_histogram(report_df)
    st.plotly_chart(fig_hist, use_container_width=True, theme=None)

# ---------------------------------------------------------------------------
# 관심 종목 테이블 (score>=70, signal_count>=2)
# ---------------------------------------------------------------------------
st.subheader("관심 종목 (점수 70+, 시그널 2+)")

watchlist = report_df[
    (report_df['score'] >= 70) & (report_df['signal_count'] >= 2)
].copy()

if watchlist.empty:
    st.info("현재 조건을 만족하는 관심 종목이 없습니다.")
else:
    display_cols = [
        'stock_code', 'stock_name', 'sector', 'pattern',
        'score', 'signal_count', 'entry_point', 'stop_loss',
    ]
    display_cols = [c for c in display_cols if c in watchlist.columns]

    st.dataframe(
        watchlist[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=min(400, len(watchlist) * 40 + 40),
    )

    st.caption(f"총 {len(watchlist)}개 종목")
