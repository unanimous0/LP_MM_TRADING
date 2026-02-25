"""
패턴 분석 페이지 - 패턴 분류 + 시그널 탐지 결과 조회

사이드바: 정렬/방향/패턴/섹터/점수/시그널 필터
7개 탭: 종목 리스트, 패턴별 통계, 시그널 분석, 섹터 크로스 분석, 섹터 Z-Score 히트맵, 수급 집중도, Treemap
종목 상세: 개별 종목 정보 (패턴/점수/시그널/진입/손절)
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime

from utils.data_loader import (
    run_analysis_pipeline_with_progress, get_sectors, get_date_range,
    get_stock_list, get_db_connection,
)
from utils.charts import (
    create_signal_distribution_chart,
    create_sector_pattern_crosstab_chart,
    create_sector_avg_score_chart,
    create_sector_zscore_heatmap,
    create_sector_concentration_chart,
    create_sector_treemap_html,
)
from src.analyzer.integrated_report import IntegratedReport

st.set_page_config(page_title="패턴분석", page_icon="🔍", layout="wide")
st.title("패턴 분류 & 시그널 분석")

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
    help="해당 날짜 기준으로 패턴/시그널을 분석합니다. 과거 날짜를 선택하면 당시 상태를 볼 수 있습니다.",
)
end_date_str = end_date.strftime("%Y-%m-%d")

st.sidebar.divider()

# A: 정렬 기준
sort_options = {
    'score':        '패턴 점수',
    'final_score':  '최종 점수 (패턴+시그널)',
    'recent':       '최근 수급 (5D)',
    'momentum':     '모멘텀 (단기-장기)',
    'weighted':     '가중 평균 (최근 높은 비중)',
    'average':      '단순 평균 (7기간)',
    'short_trend':  '단기 모멘텀 (5D-20D)',
    'signal_count': '시그널 수',
}
sort_by = st.sidebar.selectbox(
    "정렬 기준",
    options=list(sort_options.keys()),
    format_func=lambda x: sort_options[x],
)

# B: 수급 방향
supply_direction = st.sidebar.radio(
    "수급 방향",
    options=['all', 'buy', 'sell'],
    format_func=lambda x: {'all': '전체', 'buy': '매수 상위', 'sell': '매도 상위'}[x],
    horizontal=True,
    help="5D Z-Score 기준: 매수(>0), 매도(<0)",
)

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
_prog = st.progress(0, text="분석 준비 중... 0%")
zscore_matrix, classified_df, signals_df, report_df = run_analysis_pipeline_with_progress(
    end_date=end_date_str, progress_bar=_prog,
    institution_weight=institution_weight,
)
_prog.empty()

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

# B: 수급 방향 필터 (classified_df의 5D 기준)
if supply_direction != 'all' and not classified_df.empty and '5D' in classified_df.columns:
    if supply_direction == 'buy':
        dir_codes = set(classified_df[classified_df['5D'] > 0]['stock_code'].tolist())
    else:
        dir_codes = set(classified_df[classified_df['5D'] < 0]['stock_code'].tolist())
    filtered_df = filtered_df[filtered_df['stock_code'].isin(dir_codes)]
    zscore_matrix = zscore_matrix[zscore_matrix['stock_code'].isin(dir_codes)]

# final_score 계산 (정렬용)
filtered_df = filtered_df.copy()
if 'signal_count' in filtered_df.columns:
    filtered_df['final_score'] = filtered_df['score'] + filtered_df['signal_count'] * 5
else:
    filtered_df['final_score'] = filtered_df['score']

# 정렬 적용
if sort_by in ('score', 'final_score', 'signal_count', 'recent', 'momentum', 'weighted', 'average', 'short_trend'):
    if sort_by in filtered_df.columns:
        filtered_df = filtered_df.sort_values(sort_by, ascending=False)

dir_label = {'all': '전체', 'buy': '매수 상위', 'sell': '매도 상위'}[supply_direction]
st.caption(f"필터링 결과: {len(filtered_df)}개 종목 (전체 {len(report_df)}개) | 방향: {dir_label} | 정렬: {sort_options[sort_by]}")

# 탭 4/6/7에서 공통 사용 (DRY)
_src_df = filtered_df if not filtered_df.empty else report_df

# ---------------------------------------------------------------------------
# 7개 탭
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "종목 리스트", "패턴별 통계", "시그널 분석",
    "섹터 크로스 분석", "섹터 Z-Score 히트맵", "수급 집중도", "Treemap",
])

# ---- Tab 1: 종목 리스트 ----
with tab1:
    if filtered_df.empty:
        st.info("조건에 맞는 종목이 없습니다.")
    else:
        # D: 정렬 키 컬럼 토글
        show_sort_cols = st.checkbox("정렬 키 컬럼 표시", value=False,
                                      help="recent/momentum/weighted/average/short_trend/temporal_consistency 수치를 표시합니다.")

        display_df = filtered_df.copy()

        display_cols = [
            'stock_code', 'stock_name', 'sector', 'pattern',
            'score', 'signal_count', 'final_score',
        ]
        if show_sort_cols:
            display_cols += ['recent', 'momentum', 'weighted', 'average', 'short_trend', 'temporal_consistency']
        display_cols += ['signal_list', 'entry_point', 'stop_loss']
        display_cols = [c for c in display_cols if c in display_df.columns]

        col_config = {
            "score": st.column_config.NumberColumn("패턴 점수", format="%.1f"),
            "signal_count": st.column_config.NumberColumn("시그널 수", format="%d"),
            "final_score": st.column_config.ProgressColumn(
                "최종 점수", min_value=0, max_value=115, format="%.1f",
            ),
        }
        if show_sort_cols:
            col_config.update({
                "recent": st.column_config.NumberColumn(
                    "최근수급", format="%.2f",
                    help="(5D+20D)/2 — 최근 단기 수급 강도",
                ),
                "momentum": st.column_config.NumberColumn(
                    "모멘텀", format="%.2f",
                    help="5D-500D — 단기-장기 차이 (양수=최근 개선)",
                ),
                "weighted": st.column_config.NumberColumn(
                    "가중평균", format="%.2f",
                    help="최근 기간에 높은 비중 — 단기 변화 민감",
                ),
                "average": st.column_config.NumberColumn(
                    "단순평균", format="%.2f",
                    help="7기간 Z-Score 단순 평균",
                ),
                "short_trend": st.column_config.NumberColumn(
                    "단기모멘텀", format="%.2f",
                    help="5D-20D — 양수=최근 5일이 20일보다 강함 (단기 가속), 음수=지속형에서 정상",
                ),
                "temporal_consistency": st.column_config.NumberColumn(
                    "시간순서", format="%.2f",
                    help="0~1 — 5D≥10D≥...≥500D 순서 일치 비율. 1.0=완전 모멘텀형, 0.0=완전 지속형",
                ),
            })

        st.dataframe(
            display_df[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=min(600, len(display_df) * 40 + 40),
            column_config=col_config,
        )

# ---- Tab 2: 패턴별 통계 ----
with tab2:
    summary_df = report_gen.get_pattern_summary_report(report_df)
    if summary_df.empty:
        st.info("패턴 통계 데이터가 없습니다.")
    else:
        st.dataframe(summary_df, use_container_width=True)

# ---- Tab 3: 시그널 분석 ----
with tab3:
    fig_signal = create_signal_distribution_chart(report_df)
    st.plotly_chart(fig_signal, width="stretch", theme=None)

# ---- Tab 4: 섹터 크로스 분석 ----
with tab4:
    if _src_df.empty:
        st.info("섹터 분석 데이터가 없습니다.")
    else:
        col_left, col_right = st.columns(2)
        with col_left:
            fig_ct = create_sector_pattern_crosstab_chart(_src_df)
            st.plotly_chart(fig_ct, width="stretch", theme=None)
        with col_right:
            fig_avg = create_sector_avg_score_chart(_src_df)
            st.plotly_chart(fig_avg, width="stretch", theme=None)

        # 교차 테이블 (crosstab)
        st.subheader("섹터 x 패턴 교차 테이블")
        _ct_df = _src_df.copy()
        _ct_df['sector'] = _ct_df['sector'].fillna('기타')
        _ct_df['pattern'] = _ct_df['pattern'].fillna('기타')
        ct = pd.crosstab(_ct_df['sector'], _ct_df['pattern'], margins=True, margins_name='합계')
        st.dataframe(ct, use_container_width=True)

        # 섹터별 시그널 통계
        if 'signal_count' in _src_df.columns:
            st.subheader("섹터별 시그널 통계")
            sig_stats = _src_df.groupby(_src_df['sector'].fillna('기타')).agg(
                종목수=('stock_code', 'size'),
                평균점수=('score', 'mean'),
                평균시그널=('signal_count', 'mean'),
                시그널2이상=('signal_count', lambda x: (x >= 2).sum()),
            ).round(1).sort_values('평균점수', ascending=False)
            st.dataframe(sig_stats, use_container_width=True)


# ---- Tab 5: 섹터 Z-Score 히트맵 ----
with tab5:
    if zscore_matrix.empty:
        st.info("히트맵 데이터가 없습니다.")
    else:
        st.caption("현재 필터 조건이 적용된 종목들의 섹터별 평균 Z-Score")
        stock_list = get_stock_list()
        _filtered_codes = set(filtered_df['stock_code'].tolist()) if not filtered_df.empty else set()
        _hm_matrix = zscore_matrix[zscore_matrix['stock_code'].isin(_filtered_codes)] if _filtered_codes else zscore_matrix
        if _hm_matrix.empty:
            st.info("필터 조건에 맞는 종목이 없습니다.")
        else:
            _hm_sort_map = {
                'score': 'weighted', 'final_score': 'weighted', 'signal_count': 'recent',
                'recent': 'recent', 'momentum': 'momentum', 'weighted': 'weighted', 'average': 'average',
            }
            fig_sector_hm = create_sector_zscore_heatmap(
                _hm_matrix, stock_list=stock_list,
                sort_by=_hm_sort_map.get(sort_by, 'weighted'),
            )
            st.plotly_chart(fig_sector_hm, width="stretch", theme=None)

# ---- Tab 6: 수급 집중도 ----
with tab6:
    if _src_df.empty:
        st.info("데이터가 없습니다.")
    else:
        st.caption("섹터점수 = 평균점수 × (1 + 고득점종목수/전체종목수) | 고득점 = 최종점수 ≥ 70 | 5개 이상 섹터만")
        fig_conc = create_sector_concentration_chart(_src_df)
        st.plotly_chart(fig_conc, width="stretch", theme=None)

        # 집중도 테이블
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

# ---- Tab 7: Treemap ----
with tab7:
    if _src_df.empty:
        st.info("데이터가 없습니다.")
    else:
        st.caption("박스 크기: 종합점수 비례 | 색상: 빨강(낮음) → 초록(높음) | 섹터별 상위 10개 종목")
        st.markdown('<style>.stElementContainer:has(iframe) iframe { width: 100% !important; }</style>', unsafe_allow_html=True)
        treemap_html = create_sector_treemap_html(_src_df)
        components.html(treemap_html, height=820, scrolling=False)

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
        col1.metric("패턴", row.get('pattern', '-'))
        col2.metric("점수", f"{row.get('score', 0):.0f}")
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
    st.info("종목을 선택하려면 사이드바에서 필터 조건을 조정하세요.")
