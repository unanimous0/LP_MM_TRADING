"""
종목 비교 페이지 — 최대 5종목 수급 패턴 병렬 비교

4개 섹션:
  ① Z-Score 추이 오버레이 (시계열)
  ② 멀티기간 Z-Score 그룹 바차트
  ③ 패턴 점수 레이더
  ④ 핵심 지표 비교 테이블
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd

from utils.ui_constants import INSTITUTION_WEIGHT_HELP
from utils.data_loader import (
    get_stock_list, get_date_range,
    get_stock_zscore_history,
    _stage_classify, _stage_report,
    get_watchlist,
)
from utils.charts import (
    create_compare_zscore_chart,
    create_compare_multiperiod_bar,
    create_compare_score_radar,
)

st.set_page_config(page_title="종목 비교", page_icon="🔀", layout="wide")
st.markdown('<style>div[data-baseweb="select"]>div{border-color:#333!important}div[data-baseweb="input"] input,div[data-baseweb="input"]>div{border-color:#333!important}[data-testid="stDateInput"]>div>div>div{border-color:#333!important}[data-testid="stExpander"]{border-color:#222!important}</style>', unsafe_allow_html=True)
st.title("종목 비교")
st.caption("최대 5종목을 선택해 Z-Score·패턴 점수·시그널을 나란히 비교합니다.")

# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
min_date, max_date = get_date_range()
_max_dt = datetime.strptime(max_date, "%Y-%m-%d")

end_date = st.sidebar.date_input(
    "기준 날짜",
    value=_max_dt,
    min_value=datetime.strptime(min_date, "%Y-%m-%d"),
    max_value=_max_dt.replace(month=12, day=31),
)
end_date_str = end_date.strftime("%Y-%m-%d")

_PERIOD_DAYS = {"3개월": 90, "6개월": 180, "1년": 365, "전체": None}
period_sel   = st.sidebar.selectbox("표시 기간", list(_PERIOD_DAYS.keys()), index=1)
period_days  = _PERIOD_DAYS[period_sel]
start_date_str = (
    (end_date - timedelta(days=period_days)).strftime("%Y-%m-%d")
    if period_days else None
)

z_score_window = st.sidebar.slider(
    "Z-Score 기준 기간 (거래일)", 20, 1300, 50, step=10,
)

institution_weight = st.sidebar.slider(
    "기관 가중치", 0.0, 1.0, 0.3, step=0.05,
    key="w_institution_weight",
    help=INSTITUTION_WEIGHT_HELP,
)

st.sidebar.divider()

# 종목 선택
stock_list = get_stock_list()
stock_options = {
    f"{r['stock_name']} ({r['stock_code']})": r['stock_code']
    for _, r in stock_list.iterrows()
}

# 관심종목 미리 불러오기 (빠른 접근용)
_wl_df = get_watchlist()
_wl_codes = set(_wl_df['stock_code'].tolist()) if not _wl_df.empty else set()
_wl_opts  = [
    f"{r['stock_name']} ({r['stock_code']})"
    for _, r in _wl_df.iterrows()
] if not _wl_df.empty else []

# 기본 선택 종목 (관심종목이 있으면 최대 2개 기본 선택)
_default_sel = _wl_opts[:2] if _wl_opts else []

selected_labels = st.sidebar.multiselect(
    "비교할 종목 (최대 5개)",
    options=list(stock_options.keys()),
    default=_default_sel,
    max_selections=5,
    placeholder="종목명 또는 코드로 검색...",
)

if not selected_labels:
    st.info("사이드바에서 비교할 종목을 선택하세요. 관심종목에 저장된 종목이 기본 선택됩니다.")
    st.stop()

# 선택된 종목 코드 & 라벨
selected_codes = [stock_options[lbl] for lbl in selected_labels]
labels = [lbl.rsplit(' (', 1)[0] for lbl in selected_labels]

# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
_prog = st.progress(0, text="분석 파이프라인 실행 중... 0%")
classified_df = _stage_classify(end_date=end_date_str, institution_weight=institution_weight)
_prog.progress(0.4, text="패턴 리포트 생성 중... 40%")
report_df = _stage_report(end_date=end_date_str, institution_weight=institution_weight)
_prog.progress(0.6, text="Z-Score 시계열 로드 중... 60%")

# 종목별 Z-Score 시계열 로드
zscore_data = {}
for i, (code, lbl) in enumerate(zip(selected_codes, labels)):
    df = get_stock_zscore_history(code, end_date_str, institution_weight, z_score_window)
    zscore_data[lbl] = df
    _prog.progress(0.6 + 0.4 * (i + 1) / len(selected_codes), text=f"Z-Score 로드 중... {lbl}")

_prog.progress(1.0, text="완료 100%")
_prog.empty()

# ---------------------------------------------------------------------------
# 패턴/점수 데이터 추출
# ---------------------------------------------------------------------------
period_cols = ['5D', '10D', '20D', '50D', '100D', '200D', '500D']

report_rows = []  # 비교 테이블용
multiperiod_rows = []  # 멀티기간 바차트용
radar_rows = []  # 레이더 차트용

for code, lbl in zip(selected_codes, labels):
    r_row = (
        report_df[report_df['stock_code'] == code].iloc[0]
        if (not report_df.empty and code in report_df['stock_code'].values) else None
    )
    c_row = (
        classified_df[classified_df['stock_code'] == code].iloc[0]
        if (not classified_df.empty and code in classified_df['stock_code'].values) else None
    )

    # 비교 테이블 행
    row = {'종목': lbl, '종목코드': code}
    if r_row is not None:
        row['패턴']       = r_row.get('pattern_label', r_row.get('pattern', '-'))
        row['패턴 점수']  = round(float(r_row.get('score', 0)), 1)
        row['시그널 수']  = int(r_row.get('signal_count', 0))
        row['진입가']     = r_row.get('entry_point', None)
        row['손절가']     = r_row.get('stop_loss', None)
    if c_row is not None:
        for p in period_cols:
            row[p] = round(float(c_row.get(p, 0) or 0), 2)

    report_rows.append(row)

    # 멀티기간 바차트 행
    mp_row = {'label': lbl}
    if c_row is not None:
        for p in period_cols:
            mp_row[p] = float(c_row.get(p, 0) or 0)
    multiperiod_rows.append(mp_row)

    # 레이더 차트 행
    rd_row = {'label': lbl}
    if c_row is not None:
        for k in ['recent', 'short_divergence', 'mid_divergence', 'long_divergence', 'weighted', 'average']:
            rd_row[k] = float(c_row.get(k, 0) or 0)
    radar_rows.append(rd_row)

# ---------------------------------------------------------------------------
# 차트 탭
# ---------------------------------------------------------------------------
tab_ts, tab_mp, tab_radar, tab_table = st.tabs([
    "📈 Z-Score 추이",
    "📊 멀티기간 비교",
    "🕸️ 패턴 점수 레이더",
    "📋 핵심 지표 테이블",
])

# ---- 탭 1: Z-Score 시계열 ----
with tab_ts:
    _z_col_opts = {
        '종합 Z-Score': 'combined_zscore',
        '외국인 Z-Score': 'foreign_zscore',
        '기관 Z-Score': 'institution_zscore',
    }
    _z_col_lbl = st.radio(
        "Z-Score 종류", list(_z_col_opts.keys()), horizontal=True, key="cmp_zcol",
    )
    _z_col = _z_col_opts[_z_col_lbl]

    fig_ts = create_compare_zscore_chart(
        zscore_data=zscore_data,
        z_col=_z_col,
        start_date=start_date_str,
    )
    st.plotly_chart(fig_ts, width="stretch", theme=None)

# ---- 탭 2: 멀티기간 바차트 ----
with tab_mp:
    fig_mp = create_compare_multiperiod_bar(multiperiod_rows, period_cols)
    st.plotly_chart(fig_mp, width="stretch", theme=None)
    st.caption("각 기간(5D~500D)의 Z-Score를 종목별로 나란히 비교합니다.")

# ---- 탭 3: 레이더 ----
with tab_radar:
    fig_radar = create_compare_score_radar(radar_rows)
    st.plotly_chart(fig_radar, width="stretch", theme=None)
    st.caption("최근수급/단기이격/중기이격/장기이격/가중평균/단순평균 — 패턴 점수 구성 비교")

# ---- 탭 4: 핵심 지표 테이블 ----
with tab_table:
    if report_rows:
        cmp_df = pd.DataFrame(report_rows)
        _tbl_cols = ['종목', '패턴', '패턴 점수', '시그널 수'] + period_cols + ['진입가', '손절가']
        _tbl_cols = [c for c in _tbl_cols if c in cmp_df.columns]
        _tbl_cfg = {
            '패턴 점수': st.column_config.NumberColumn('패턴 점수', format='%.1f'),
            '시그널 수': st.column_config.NumberColumn('시그널 수', format='%d'),
            '진입가':    st.column_config.NumberColumn('진입가', format='₩%d'),
            '손절가':    st.column_config.NumberColumn('손절가', format='₩%d'),
        }
        for p in period_cols:
            _tbl_cfg[p] = st.column_config.NumberColumn(p, format='%.2f')

        st.dataframe(
            cmp_df[_tbl_cols].set_index('종목'),
            column_config=_tbl_cfg,
            use_container_width=True,
        )
    else:
        st.info("비교 데이터를 불러올 수 없습니다.")
