"""
패턴 분석 페이지 - 패턴 분류 + 시그널 탐지 결과 조회

사이드바: 패턴/섹터/점수/시그널 필터
3개 탭: 종목 리스트, 패턴별 통계, 시그널 분석
종목 상세: 개별 종목 정보 (패턴/점수/시그널/진입/손절)
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
from datetime import datetime

from utils.data_loader import run_analysis_pipeline, get_sectors, get_date_range
from utils.charts import create_signal_distribution_chart
from src.analyzer.integrated_report import IntegratedReport
from utils.data_loader import get_db_connection

st.set_page_config(page_title="패턴분석", page_icon="🔍", layout="wide")
st.title("패턴 분류 & 시그널 분석")

# ---------------------------------------------------------------------------
# 사이드바 필터
# ---------------------------------------------------------------------------
min_date, max_date = get_date_range()
end_date = st.sidebar.date_input(
    "기준 날짜",
    value=datetime.strptime(max_date, "%Y-%m-%d"),
    min_value=datetime.strptime(min_date, "%Y-%m-%d"),
    max_value=datetime.strptime(max_date, "%Y-%m-%d"),
    help="해당 날짜 기준으로 패턴/시그널을 분석합니다. 과거 날짜를 선택하면 당시 상태를 볼 수 있습니다.",
)
end_date_str = end_date.strftime("%Y-%m-%d")

st.sidebar.divider()

pattern_options = ['전체', '모멘텀형', '지속형', '전환형', '기타']
selected_pattern = st.sidebar.selectbox("패턴", pattern_options)

sectors = get_sectors()
selected_sector = st.sidebar.selectbox("섹터", ["전체"] + sectors)

min_score = st.sidebar.slider("최소 점수", 0.0, 100.0, 0.0, step=5.0)
min_signals = st.sidebar.slider("최소 시그널 수", 0, 3, 0)

# ---------------------------------------------------------------------------
# 데이터 로드 & 필터링
# ---------------------------------------------------------------------------
zscore_matrix, classified_df, signals_df, report_df = run_analysis_pipeline(end_date=end_date_str)

if report_df.empty:
    st.warning("분석 데이터가 없습니다.")
    st.stop()

# IntegratedReport의 filter_report 사용
conn = get_db_connection()
report_gen = IntegratedReport(conn)
filtered_df = report_gen.filter_report(
    report_df,
    pattern=selected_pattern if selected_pattern != '전체' else None,
    sector=selected_sector if selected_sector != '전체' else None,
    min_score=min_score if min_score > 0 else None,
    min_signal_count=min_signals if min_signals > 0 else None,
)

st.caption(f"필터링 결과: {len(filtered_df)}개 종목 (전체 {len(report_df)}개)")

# ---------------------------------------------------------------------------
# 3개 탭
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["종목 리스트", "패턴별 통계", "시그널 분석"])

with tab1:
    if filtered_df.empty:
        st.info("조건에 맞는 종목이 없습니다.")
    else:
        display_df = filtered_df.copy()
        display_df['final_score'] = display_df['score'] + display_df.get('signal_count', 0) * 5

        display_cols = [
            'stock_code', 'stock_name', 'sector', 'pattern',
            'score', 'signal_count', 'final_score',
            'signal_list', 'entry_point', 'stop_loss',
        ]
        display_cols = [c for c in display_cols if c in display_df.columns]

        st.dataframe(
            display_df[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=min(600, len(display_df) * 40 + 40),
            column_config={
                "score": st.column_config.ProgressColumn(
                    "패턴 점수", min_value=0, max_value=100, format="%.0f",
                ),
                "signal_count": st.column_config.NumberColumn("시그널 수", format="%d"),
                "final_score": st.column_config.ProgressColumn(
                    "최종 점수", min_value=0, max_value=115, format="%.0f",
                ),
            },
        )

with tab2:
    summary_df = report_gen.get_pattern_summary_report(report_df)
    if summary_df.empty:
        st.info("패턴 통계 데이터가 없습니다.")
    else:
        st.dataframe(summary_df, use_container_width=True)

with tab3:
    fig_signal = create_signal_distribution_chart(report_df)
    st.plotly_chart(fig_signal, use_container_width=True)

# ---------------------------------------------------------------------------
# 종목 상세
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
        # 선택된 종목의 stock_code 추출
        stock_code = selected.split('(')[-1].rstrip(')')
        row = filtered_df[filtered_df['stock_code'] == stock_code].iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("패턴", row['pattern'])
        col2.metric("점수", f"{row['score']:.0f}")
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
                period_cols = ['1W', '1M', '3M', '6M', '1Y', '2Y']
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
    st.info("종목을 선택하려면 사이드바에서 필터 조건을 조정하세요.")
