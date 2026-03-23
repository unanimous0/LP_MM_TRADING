"""
종목 상세 페이지 — 단일 종목 수급 심층 분석

5탭: 주가 차트 / Z-Score 추이 / 수급 금액 / 시그널 & MA / 패턴 현황
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from utils.ui_constants import INSTITUTION_WEIGHT_HELP
from streamlit_lightweight_charts import renderLightweightCharts
from utils.data_loader import (
    get_stock_list, get_date_range,
    get_stock_zscore_history, get_stock_raw_history, get_stock_ohlcv,
    _stage_classify, _stage_report, _stage_signals,
    get_watchlist, is_in_watchlist, add_to_watchlist, remove_from_watchlist,
)
from utils.charts import (
    create_zscore_history_chart,
    create_supply_amount_chart,
    create_signal_ma_chart,
    create_multiperiod_zscore_bar,
)

st.set_page_config(page_title="종목 상세", page_icon="📋", layout="wide")
st.markdown('<style>div[data-baseweb="select"]>div{border-color:#333!important}div[data-baseweb="input"] input,div[data-baseweb="input"]>div{border-color:#333!important}[data-testid="stDateInput"]>div>div>div{border-color:#333!important}[data-testid="stExpander"]{border-color:#222!important}</style>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
min_date, max_date = get_date_range()
_max_dt = datetime.strptime(max_date, "%Y-%m-%d")

# 종목 선택 (최상단)
stock_list = get_stock_list()
if stock_list.empty:
    st.error("종목 정보를 불러올 수 없습니다.")
    st.stop()

stock_options = [
    f"{row['stock_name']} ({row['stock_code']})"
    for _, row in stock_list.iterrows()
]

# 삼성전자를 기본값으로 설정
default_idx = next(
    (i for i, s in enumerate(stock_options) if '005930' in s), 0
)

# 히트맵 클릭으로 넘어온 경우 해당 종목을 우선 선택
_from_heatmap = st.session_state.pop('heatmap_selected_code', None)
if _from_heatmap:
    default_idx = next(
        (i for i, s in enumerate(stock_options) if _from_heatmap in s), default_idx
    )

selected = st.sidebar.selectbox(
    "종목 선택", stock_options, index=default_idx,
    help="종목명 또는 종목코드로 검색 가능",
)

st.sidebar.divider()

end_date = st.sidebar.date_input(
    "기준 날짜",
    value=_max_dt,
    min_value=datetime.strptime(min_date, "%Y-%m-%d"),
    max_value=_max_dt.replace(month=12, day=31),
    help="이 날짜 이전 데이터를 기준으로 분석합니다.",
)
end_date_str = end_date.strftime("%Y-%m-%d")

_PERIOD_DAYS = {"3개월": 90, "6개월": 180, "1년": 365, "전체": None}
period_sel = st.sidebar.selectbox("표시 기간", list(_PERIOD_DAYS.keys()), index=1)
period_days = _PERIOD_DAYS[period_sel]

# 표시 기간의 start_date 계산
if period_days:
    start_date_str = (end_date - timedelta(days=period_days)).strftime("%Y-%m-%d")
else:
    start_date_str = None

st.sidebar.divider()

z_score_window = st.sidebar.slider(
    "Z-Score 기준 기간 (거래일)",
    min_value=20, max_value=1300, value=50, step=10,
    help="수급 금액 테이블의 Z-Score 계산 기준 기간 (최대 5년 = 약 1,300거래일). 실제 데이터 수보다 크면 자동으로 최대 데이터 기준으로 조정됩니다.",
)

institution_weight = st.sidebar.slider(
    "기관 가중치", 0.0, 1.0, 0.3, step=0.05,
    key="w_institution_weight",
    help=INSTITUTION_WEIGHT_HELP,
)

_direction_sel = st.sidebar.selectbox(
    "분석 방향", ["매수 (Long)", "매도 (Short)"], index=0,
    help="매수: 순매수 관점 패턴/점수 · 매도: 순매도 관점 패턴/점수",
    key="detail_direction",
)
_direction = 'long' if '매수' in _direction_sel else 'short'

# 선택 종목 파싱
stock_code = selected.split('(')[-1].rstrip(')')
stock_name = selected.rsplit(' (', 1)[0]

# 종목 기본 정보
info_row = stock_list[stock_list['stock_code'] == stock_code].iloc[0]
sector    = info_row.get('sector', '-') or '-'
market_id = info_row.get('market_id', '-') or '-'

# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
_prog = st.progress(0, text=f"💰 '{stock_name}' 수급 원시 데이터 로드 중... 0%")
raw_df = get_stock_raw_history(stock_code, end_date_str)

# Z-Score 기준 기간: 실제 데이터 수보다 크면 자동 캡
_data_count = len(raw_df) if not raw_df.empty else z_score_window
_effective_window = min(z_score_window, _data_count)
if _effective_window < z_score_window:
    st.sidebar.caption(f"⚠️ 데이터 {_data_count}거래일 기준 자동 적용")

_prog.progress(0.25, text=f"📡 Z-Score 계산 중... 25%")
zscore_df = get_stock_zscore_history(stock_code, end_date_str, institution_weight, _effective_window)

if zscore_df.empty and raw_df.empty:
    _prog.empty()
    st.title(stock_name)
    st.warning(f"'{stock_name}' ({stock_code}) 종목의 수급 데이터가 없습니다.")
    st.stop()

_prog.progress(0.50, text="📊 패턴 분류 중... 50%")
classified_df = _stage_classify(end_date=end_date_str, institution_weight=institution_weight, direction=_direction)

_prog.progress(0.70, text="📋 통합 리포트 생성 중... 70%")
report_df = _stage_report(end_date=end_date_str, institution_weight=institution_weight, direction=_direction)

_prog.progress(0.88, text="🔔 시그널 탐지 중... 88%")
signals_df = _stage_signals(end_date=end_date_str, institution_weight=institution_weight)

_prog.progress(1.0, text="✅ 완료 100%")
_prog.empty()

# 선택 종목 행 추출 (없으면 None)
stock_report = (
    report_df[report_df['stock_code'] == stock_code].iloc[0]
    if (not report_df.empty and stock_code in report_df['stock_code'].values)
    else None
)
stock_classified = (
    classified_df[classified_df['stock_code'] == stock_code].iloc[0]
    if (not classified_df.empty and stock_code in classified_df['stock_code'].values)
    else None
)
stock_signals = (
    signals_df[signals_df['stock_code'] == stock_code].iloc[0]
    if (not signals_df.empty and stock_code in signals_df['stock_code'].values)
    else None
)

# ---------------------------------------------------------------------------
# 헤더 + 관심종목 버튼
# ---------------------------------------------------------------------------
_h_col, _star_col = st.columns([8, 1])
_h_col.title(stock_name)
_h_col.caption(f"{sector} · 마켓 {market_id} · {stock_code}")

# ⭐ 관심종목 토글 버튼
_in_watchlist = is_in_watchlist(stock_code)
if _in_watchlist:
    if _star_col.button("⭐ 관심 해제", key="wl_toggle", use_container_width=True):
        remove_from_watchlist(stock_code)
        st.toast(f"'{stock_name}' 관심종목에서 제거했습니다.", icon="🗑️")
        st.rerun()
else:
    if _star_col.button("☆ 관심 추가", key="wl_toggle", use_container_width=True):
        add_to_watchlist(stock_code, stock_name, sector)
        st.toast(f"'{stock_name}'을(를) 관심종목에 추가했습니다.", icon="⭐")
        st.rerun()

# ---------------------------------------------------------------------------
# KPI 카드 5개
# ---------------------------------------------------------------------------
# 현재 Z-Score (zscore_df 마지막 행)
if not zscore_df.empty:
    latest_z = zscore_df.iloc[-1]
    combined_z     = latest_z.get('combined_zscore',     float('nan'))
    foreign_z      = latest_z.get('foreign_zscore',      float('nan'))
    institution_z  = latest_z.get('institution_zscore',  float('nan'))
else:
    combined_z = foreign_z = institution_z = float('nan')

# 현재가 (raw_df 마지막 행)
close_price = raw_df.iloc[-1]['close_price'] if not raw_df.empty else None

# 활성 시그널 수
signal_count = int(stock_report['signal_count']) if stock_report is not None else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric(
    "Z-Score 종합",
    f"{combined_z:.2f}" if pd.notna(combined_z) else "-",
)
col2.metric(
    "Z-Score 외국인",
    f"{foreign_z:.2f}" if pd.notna(foreign_z) else "-",
)
col3.metric(
    "Z-Score 기관",
    f"{institution_z:.2f}" if pd.notna(institution_z) else "-",
)
col4.metric(
    "현재가",
    f"{int(close_price):,}원" if (close_price and pd.notna(close_price)) else "-",
)
col5.metric("활성 시그널", f"{signal_count}개")

# ---------------------------------------------------------------------------
# 패턴 배너
# ---------------------------------------------------------------------------
if stock_report is not None:
    pattern       = stock_report.get('pattern', '-')
    pattern_label = stock_report.get('pattern_label', pattern)
    score         = stock_report.get('score', 0)
    signal_list   = stock_report.get('signal_list', '') or ''

    _PATTERN_COLORS = {
        '급등형': '#f472b6',   # pink-400
        '지속형':   '#2dd4bf',   # teal-400
        '전환형':   '#fbbf24',   # amber-400
        '기타':     '#64748b',   # slate-500
    }
    pcolor = _PATTERN_COLORS.get(pattern, '#64748b')

    signals_text = signal_list if signal_list else '없음'
    if isinstance(signals_text, list):
        signals_text = ', '.join(signals_text)

    st.markdown(
        f'<div style="border-left:4px solid {pcolor}; padding:8px 16px; '
        f'background-color:#111111; border-radius:4px; margin:8px 0;">'
        f'<b>현재 패턴:</b> {pattern_label} &nbsp;|&nbsp; '
        f'<b>패턴 점수:</b> {score:.0f}점 &nbsp;|&nbsp; '
        f'<b>시그널:</b> {signals_text}'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.info("현재 분석 기준일의 패턴 분류 결과가 없습니다. (DB 데이터 부족 또는 필터 미통과)")

# ---------------------------------------------------------------------------
# 4탭
# ---------------------------------------------------------------------------
tab_price, tab1, tab2, tab3, tab4 = st.tabs([
    "주가 차트",
    "Z-Score 추이",
    "수급 금액",
    "시그널 & MA",
    "패턴 현황",
])

# ── Tab Price: 주가 차트 (TradingView Lightweight Charts)
with tab_price:
    ohlcv_df = get_stock_ohlcv(stock_code, end_date_str, start_date_str)
    if ohlcv_df.empty:
        st.info("주가(OHLCV) 데이터가 없습니다.")
    else:
        # 캔들스틱 데이터
        candle_data = [
            {
                "time": row["date"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            }
            for _, row in ohlcv_df.iterrows()
        ]

        # 거래량 데이터 (양봉=green, 음봉=red)
        volume_data = [
            {
                "time": row["date"],
                "value": float(row["volume"]),
                "color": "#4ade8060" if row["close"] >= row["open"] else "#f8717160",
            }
            for _, row in ohlcv_df.iterrows()
        ]

        # 수급 오버레이 옵션 (중복 선택 가능)
        _ov_cols = st.columns([3, 2])
        _ov_investors = _ov_cols[0].multiselect(
            "수급 오버레이", ["외국인", "기관", "개인"],
            default=["외국인"], key="price_overlay_investors",
        )
        _ov_mode = _ov_cols[1].selectbox(
            "표시 방식", ["일별 순매수", "누적 순매수"],
            index=0, key="price_overlay_mode",
        )

        _INVESTOR_COL = {
            "외국인": "foreign_net_amount",
            "기관": "institution_net_amount",
            "개인": "individual_net_amount",
        }
        _INVESTOR_COLOR = {
            "외국인": "#38bdf8",  # sky-400
            "기관": "#a78bfa",    # violet-400
            "개인": "#fbbf24",    # amber-400
        }

        # 캔들스틱 (우측 Y축 = 주가)
        _series = [
            {
                "type": "Candlestick",
                "data": candle_data,
                "options": {
                    "upColor": "#4ade80",
                    "downColor": "#f87171",
                    "wickUpColor": "#4ade80",
                    "wickDownColor": "#f87171",
                    "borderVisible": False,
                    "priceScaleId": "right",
                    "priceFormat": {"type": "price", "precision": 0, "minMove": 1},
                },
            },
            {
                "type": "Histogram",
                "data": volume_data,
                "options": {
                    "priceFormat": {"type": "volume"},
                    "priceScaleId": "volume",
                },
            },
        ]

        # 수급 오버레이 라인들 (좌측 Y축 = 금액 억원)
        _ov_labels = []
        for _inv in _ov_investors:
            _col = _INVESTOR_COL.get(_inv)
            if not _col or _col not in raw_df.columns or raw_df.empty:
                continue
            _supply_raw = raw_df[['trade_date', _col]].copy()
            if start_date_str:
                _supply_raw = _supply_raw[_supply_raw['trade_date'] >= start_date_str]
            _supply_raw = _supply_raw.dropna(subset=[_col])
            if _supply_raw.empty:
                continue

            if _ov_mode == "누적 순매수":
                _supply_raw = _supply_raw.copy()
                _supply_raw[_col] = _supply_raw[_col].cumsum()

            _line_data = [
                {"time": str(r['trade_date']), "value": float(r[_col]) / 1e8}
                for _, r in _supply_raw.iterrows()
            ]
            _mode_label = "누적" if _ov_mode == "누적 순매수" else ""
            _title = f"{_inv}{_mode_label}(억)"
            _series.append({
                "type": "Line",
                "data": _line_data,
                "options": {
                    "color": _INVESTOR_COLOR.get(_inv, "#38bdf8"),
                    "lineWidth": 2,
                    "priceScaleId": "left",
                    "title": _title,
                    "priceFormat": {"type": "price", "precision": 0, "minMove": 1},
                },
            })
            _ov_labels.append(f"{_inv}")

        # Y축 단위 라벨 (차트 상단)
        _left_label = f"{'+'.join(_ov_labels)} {_ov_mode} (억원)" if _ov_labels else ""
        _axis_html = '<div style="display:flex;justify-content:space-between;font-size:12px;color:#94a3b8;margin-bottom:2px;">'
        _axis_html += f'<span>{_left_label}</span>' if _left_label else '<span></span>'
        _axis_html += '<span>주가 (원)</span></div>'
        st.markdown(_axis_html, unsafe_allow_html=True)

        renderLightweightCharts(
            [
                {
                    "chart": {
                        "layout": {
                            "background": {"color": "#0f172a"},
                            "textColor": "#e2e8f0",
                        },
                        "grid": {
                            "vertLines": {"color": "#334155"},
                            "horzLines": {"color": "#334155"},
                        },
                        "height": 500,
                        "crosshair": {"mode": 0},
                        "rightPriceScale": {"visible": True, "borderColor": "#334155"},
                        "leftPriceScale": {"visible": bool(_ov_labels), "borderColor": "#334155"},
                        "timeScale": {
                            "borderColor": "#334155",
                            "timeVisible": False,
                        },
                        "localization": {
                            "locale": "ko-KR",
                            "dateFormat": "yyyy-MM-dd",
                        },
                    },
                    "series": _series,
                }
            ],
            key="price_chart",
        )

        st.caption(f"표시 기간: {period_sel} ({len(ohlcv_df)}거래일) · 양봉 🟢 · 음봉 🔴")

# ── Tab 1: Z-Score 추이
with tab1:
    if zscore_df.empty:
        st.info("Z-Score 데이터가 없습니다.")
    else:
        fig1 = create_zscore_history_chart(zscore_df, start_date=start_date_str)
        st.plotly_chart(fig1, width="stretch", theme=None)

        # 데이터 포인트 수 안내
        display_df = zscore_df if not start_date_str else zscore_df[zscore_df['trade_date'] >= start_date_str]
        st.caption(
            f"표시 기간: {period_sel} ({len(display_df)}거래일) · "
            f"전체 이력: {len(zscore_df)}거래일 · "
            f"Z-Score 기준 기간: {_effective_window}거래일 · "
            f"기관 가중치: {institution_weight:.2f}"
        )

# ── Tab 2: 수급 금액
with tab2:
    if raw_df.empty:
        st.info("수급 데이터가 없습니다.")
    else:
        fig2 = create_supply_amount_chart(raw_df, start_date=start_date_str)
        st.plotly_chart(fig2, width="stretch", theme=None)

        # 수급 상세 테이블
        display_raw = (raw_df if not start_date_str
                       else raw_df[raw_df['trade_date'] >= start_date_str]).copy()
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

            # ── 그룹 정의 (컬럼명 → 그룹 색상/배경)
            _GROUPS = [
                ('외국인', ['외국인 순매수', '외국인 Z', '외국인 누적'],  '#4ade80', '#052e16'),
                ('기관',   ['기관 순매수',   '기관 Z',   '기관 누적'],   '#f472b6', '#1e0e1c'),
                ('개인',   ['개인 순매수',   '개인 누적'],               '#fb923c', '#1e1408'),
            ]
            # 실제 tbl 컬럼에 있는 것만 남김
            _GROUPS = [(g, [c for c in cols if c in tbl.columns], clr, bg)
                       for g, cols, clr, bg in _GROUPS]

            # 컬럼 → 그룹 정보 매핑
            _col_group = {}  # col_name → (color, bg, is_first_in_group)
            for g, cols, clr, bg in _GROUPS:
                for idx, c in enumerate(cols):
                    _col_group[c] = (clr, bg, idx == 0)

            _z_cols = {'외국인 Z', '기관 Z'}
            _num_cols = {'외국인 순매수', '외국인 누적', '기관 순매수', '기관 누적', '개인 순매수', '개인 누적'}

            # ── 헤더 2행: 그룹행 + 컬럼행
            _th_date = (
                "padding:6px 10px; text-align:center; font-weight:700; font-size:14px;"
                "background:#111111; color:#94a3b8; border-bottom:1px solid #222222;"
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
                clr, bg, is_first = _col_group.get(c, ('#e2e8f0', '#111111', False))
                border_left = f"border-left:2px solid {clr};" if is_first else ""
                sub_label = c.replace('외국인 ', '').replace('기관 ', '').replace('개인 ', '')
                sub_header_cells += (
                    f'<th style="padding:5px 10px; text-align:center; font-weight:600; font-size:13px;'
                    f'background:#111111; color:#94a3b8; border-bottom:1px solid #222222;'
                    f'{border_left} white-space:nowrap;">{sub_label}</th>'
                )

            # ── 데이터 행
            rows_html = ""
            for i, (_, row) in enumerate(tbl.iterrows()):
                base_bg = '#162032' if i % 2 == 1 else 'transparent'
                cells = ""
                # 날짜 셀
                cells += (
                    f'<td style="padding:5px 10px; text-align:center; font-size:14px;'
                    f'color:#94a3b8; border-bottom:1px solid #111111; background:{base_bg};'
                    f'white-space:nowrap;">{row["날짜"]}</td>'
                )
                for c in tbl.columns:
                    if c == '날짜':
                        continue
                    v = row[c]
                    clr, bg, is_first = _col_group.get(c, ('#e2e8f0', 'transparent', False))
                    border_left = f"border-left:2px solid {clr};" if is_first else ""
                    cell_bg = base_bg  # 줄무늬만 유지 (그룹 bg는 헤더에만)
                    td_base = (
                        f"padding:5px 10px; text-align:center; font-size:14px;"
                        f"border-bottom:1px solid #111111; background:{cell_bg};"
                        f"{border_left} white-space:nowrap;"
                    )
                    if c in _num_cols:
                        val_color = '#4ade80' if int(v) > 0 else ('#f87171' if int(v) < 0 else '#e2e8f0')
                        cells += f'<td style="{td_base}color:{val_color};">{int(v):,}</td>'
                    elif c in _z_cols:
                        if pd.notna(v):
                            # 순매수/누적(초록/빨강)과 구분: 강한매수=amber, 강한매도=indigo
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
            border:1px solid #222222; border-radius:6px;">
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

# ── Tab 3: 시그널 & MA
with tab3:
    if raw_df.empty:
        st.info("수급 데이터가 없습니다.")
    else:
        _MA_OPTIONS = [5, 10, 20, 60, 120, 240]
        selected_mas = st.multiselect(
            "MA 기간 선택",
            options=_MA_OPTIONS,
            default=[5, 20],
            format_func=lambda x: f"MA{x}",
            key="tab3_ma_periods",
            help="표시할 이동평균 기간을 선택하세요. 정확히 2개 선택 시 골든/데드크로스가 표시됩니다.",
        )

        if not selected_mas:
            st.info("MA 기간을 하나 이상 선택해주세요.")
        else:
            if len(selected_mas) != 2:
                st.caption(
                    f"💡 골든/데드크로스는 MA 2개 선택 시에만 표시됩니다. "
                    f"(현재 {len(selected_mas)}개 선택)"
                )
            fig3 = create_signal_ma_chart(raw_df, start_date=start_date_str, ma_periods=selected_mas)
            st.plotly_chart(fig3, width="stretch", theme=None)

        # 현재 시그널 상태 3개 메트릭
        st.markdown("##### 현재 시그널 상태")
        mc1, mc2, mc3 = st.columns(3)

        # MA 상태: raw_df 마지막 유효 행의 ma5 vs ma20 비교 (상태 기반)
        _ma_valid = raw_df.dropna(subset=['ma5', 'ma20'])
        if not _ma_valid.empty:
            _last = _ma_valid.iloc[-1]
            _ma5, _ma20 = _last['ma5'], _last['ma20']
            _is_golden = _ma5 > _ma20
            _is_dead   = _ma5 < _ma20
        else:
            _is_golden = _is_dead = None

        mc1.metric(
            "골든크로스",
            "🟢 활성" if _is_golden else ("❌ 비활성" if _is_golden is not None else "-"),
            help="MA5 > MA20 상태 유지 중",
        )
        mc2.metric(
            "데드크로스",
            "🔴 활성" if _is_dead else ("❌ 비활성" if _is_dead is not None else "-"),
            help="MA5 < MA20 상태 유지 중",
        )

        if stock_signals is not None:
            accel = stock_signals.get('acceleration', float('nan'))
            mc3.metric(
                "수급 가속도",
                f"{accel:.2f}x" if pd.notna(accel) else "-",
                help="최근 5일 평균 / 직전 5일 평균 (>1.5 = 가속)",
            )
        else:
            mc3.metric("수급 가속도", "-")
            st.caption("시그널 데이터가 없습니다. (데이터 부족 또는 필터 미통과)")

# ── Tab 4: 패턴 현황
with tab4:
    if stock_classified is None:
        st.info("패턴 분류 데이터가 없습니다. (분석 기준일 DB 데이터 부족 또는 필터 미통과)")
    else:
        fig4 = create_multiperiod_zscore_bar(stock_classified)
        st.plotly_chart(fig4, width="stretch", theme=None)

        # 진입/손절 안내
        if stock_report is not None:
            pc1, pc2 = st.columns(2)
            with pc1:
                entry = stock_report.get('entry_point', '-')
                st.info(f"**진입 포인트**: {entry}")
            with pc2:
                sl = stock_report.get('stop_loss', '-')
                st.warning(f"**손절 기준**: {sl}")

            # 시그널 목록
            sig_list = stock_report.get('signal_list', '')
            if sig_list:
                st.markdown("**활성 시그널 목록**")
                items = sig_list if isinstance(sig_list, list) else [sig_list]
                for item in items:
                    st.success(str(item))

        # 기간별 Z-Score 수치 테이블
        period_cols = [c for c in ['5D', '10D', '20D', '50D', '100D', '200D', '500D'] if c in stock_classified.index]
        if period_cols:
            st.markdown("**기간별 Z-Score 수치**")
            zscore_table = pd.DataFrame(
                {col: [f"{stock_classified[col]:.2f}"] for col in period_cols}
            )
            st.dataframe(zscore_table, use_container_width=True)
