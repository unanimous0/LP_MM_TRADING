"""
워크포워드 페이지 - Walk-Forward Analysis 실행 및 결과 시각화

사이드바: 기간/WF 설정/Optuna/탐색 범위/고정 조건
메인: KPI + 기간별 차트 + 파라미터 추이 + 통합 성과 + 거래 내역
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

from utils.data_loader import (
    run_walk_forward,
    get_date_range,
    get_metrics_from_result,
    _deserialize_trades,
)
from utils.charts import create_wf_period_returns_chart

st.set_page_config(page_title="워크포워드", page_icon="🔄", layout="wide")
st.markdown("""
<style>
section[data-testid="stSidebar"] { min-width: 340px !important; max-width: 340px !important; }
section[data-testid="stSidebar"] > div:first-child { width: 340px !important; }
div[data-baseweb="select"] > div { border-color: #333 !important; }
div[data-baseweb="input"] input, div[data-baseweb="input"] > div { border-color: #333 !important; }
[data-testid="stDateInput"] > div > div > div { border-color: #333 !important; }
[data-testid="stExpander"] { border-color: #222 !important; }
</style>
""", unsafe_allow_html=True)
st.title("Walk-Forward 분석")


# ---------------------------------------------------------------------------
# 사이드바: 기간 설정
# ---------------------------------------------------------------------------
min_date, max_date = get_date_range()
_max_dt = datetime.strptime(max_date, "%Y-%m-%d")
_max_value = _max_dt.replace(month=12, day=31)
_min_dt = datetime.strptime(min_date, "%Y-%m-%d")

st.sidebar.header("기간 설정")
start_date = st.sidebar.date_input(
    "시작일",
    value=datetime.strptime("2023-01-02", "%Y-%m-%d"),
    min_value=_min_dt,
    max_value=_max_value,
    key="wf_start",
)
end_date = st.sidebar.date_input(
    "종료일",
    value=_max_dt,
    min_value=_min_dt,
    max_value=_max_value,
    key="wf_end",
)

st.sidebar.divider()

# ---------------------------------------------------------------------------
# 사이드바: 워크포워드 설정
# ---------------------------------------------------------------------------
st.sidebar.header("워크포워드 설정")
train_months = st.sidebar.slider("학습 기간 (개월)", 1, 24, 6, key="wf_train_months")
val_months = st.sidebar.slider("검증 기간 (개월)", 1, 12, 1, key="wf_val_months")
step_months = st.sidebar.slider("스텝 (개월)", 1, 6, 1, key="wf_step_months")

st.sidebar.divider()

# ---------------------------------------------------------------------------
# 사이드바: Optuna 설정
# ---------------------------------------------------------------------------
st.sidebar.header("Optuna 설정")
n_trials = st.sidebar.slider("Trial 수 (기간당)", 10, 200, 100, step=10, key="wf_n_trials")
metric = st.sidebar.selectbox(
    "평가 지표",
    options=['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor'],
    index=0,
    format_func=lambda x: {
        'sharpe_ratio': 'Sharpe Ratio',
        'total_return': '총 수익률',
        'win_rate': '승률',
        'profit_factor': 'Profit Factor',
    }[x],
    key="wf_metric",
)

# ---------------------------------------------------------------------------
# 사이드바: 탐색 범위
# ---------------------------------------------------------------------------
with st.sidebar.expander("🔍 탐색 범위 설정", expanded=False):
    st.caption("Optuna가 각 기간마다 탐색할 파라미터 범위입니다.")

    st.markdown("**최소 점수 (min_score)**")
    ps_ms_low, ps_ms_high = st.slider(
        "min_score 범위", 30.0, 100.0, (50.0, 90.0), step=5.0, key="wf_ps_min_score",
        label_visibility="collapsed",
    )

    st.markdown("**최소 시그널 (min_signals)**")
    ps_sig_low, ps_sig_high = st.slider(
        "min_signals 범위", 1, 5, (1, 3), step=1, key="wf_ps_min_signals",
        label_visibility="collapsed",
    )

    st.markdown("**목표 수익률 (target_return)**")
    ps_tr_low, ps_tr_high = st.slider(
        "target_return 범위", 0.01, 0.50, (0.05, 0.25), step=0.01, key="wf_ps_target_return",
        label_visibility="collapsed", format="%.2f",
    )

    st.markdown("**손절 비율 (stop_loss)**")
    ps_sl_low, ps_sl_high = st.slider(
        "stop_loss 범위", -0.30, -0.01, (-0.15, -0.03), step=0.01, key="wf_ps_stop_loss",
        label_visibility="collapsed", format="%.2f",
    )

    st.markdown("**최대 포지션 (max_positions)**")
    ps_mp_low, ps_mp_high = st.slider(
        "max_positions 범위", 1, 100, (1, 50), step=1, key="wf_ps_max_positions",
        label_visibility="collapsed",
    )

    st.markdown("**최대 보유일 (max_hold_days)**")
    ps_mh_low, ps_mh_high = st.slider(
        "max_hold_days 범위", 1, 999, (1, 500), step=1, key="wf_ps_max_hold_days",
        label_visibility="collapsed",
    )

    st.markdown("**반대수급 청산 (reverse_threshold)**")
    ps_rt_low, ps_rt_high = st.slider(
        "reverse_threshold 범위", 0.0, 115.0, (0.0, 115.0), step=5.0, key="wf_ps_reverse_threshold",
        label_visibility="collapsed",
    )

st.sidebar.divider()

# ---------------------------------------------------------------------------
# 사이드바: 고정 조건
# ---------------------------------------------------------------------------
with st.sidebar.expander("🔒 고정 조건", expanded=False):
    st.caption("모든 기간에서 동일하게 적용되는 설정입니다.")

    strategy = st.selectbox(
        "전략 방향",
        options=['long', 'short', 'both'],
        format_func=lambda x: {'long': 'Long (순매수)', 'short': 'Short (순매도)', 'both': 'Long+Short (병행)'}[x],
        key="wf_strategy",
    )

    initial_capital_str = st.text_input(
        "초기 자본금 (원)", value="10,000,000", key="wf_capital_text",
    )
    try:
        initial_capital = float(int(initial_capital_str.replace(',', '').replace(' ', '')))
    except ValueError:
        initial_capital = 10_000_000.0

    institution_weight = st.slider(
        "기관 가중치", 0.0, 1.0, 0.3, step=0.05, key="wf_institution_weight",
    )

    use_tc = st.checkbox("Temporal Consistency", value=True, key="wf_use_tc")
    use_divergence = st.checkbox("Divergence", value=True, key="wf_use_divergence")

    st.markdown("**거래 비용**")
    tax_rate = st.number_input("증권거래세 (%)", 0.00, 1.00, 0.20, step=0.01, format="%.2f", key="wf_tax") / 100
    commission_rate = st.number_input("수수료 (%)", 0.000, 1.000, 0.015, step=0.001, format="%.3f", key="wf_commission") / 100
    slippage_rate = st.number_input("슬리피지 (%)", 0.00, 1.00, 0.10, step=0.01, format="%.2f", key="wf_slippage") / 100
    borrowing_rate = st.number_input("차입비용 (%/연)", 0.0, 20.0, 3.0, step=0.5, format="%.1f", key="wf_borrowing") / 100

    _wf_cap_options = {"전체": None, "50": 50, "100": 100, "200": 200, "300": 300, "500": 500}
    _wf_cap_selection = st.selectbox(
        "시총 필터 (유통시총 상위)",
        options=list(_wf_cap_options.keys()),
        index=3,
        key="wf_market_cap_top_n",
        help="각 검증 기간 종료일 기준 유통시총 상위 N종목만 대상으로 백테스트합니다.",
    )
    market_cap_top_n = _wf_cap_options[_wf_cap_selection]


# ---------------------------------------------------------------------------
# 실행 버튼
# ---------------------------------------------------------------------------
run_clicked = st.sidebar.button("워크포워드 실행", type="primary", use_container_width=True)

if run_clicked:
    # 탐색 공간 조립
    param_space = {
        'min_score':                {'type': 'float', 'low': ps_ms_low,  'high': ps_ms_high},
        'min_signals':              {'type': 'int',   'low': ps_sig_low, 'high': ps_sig_high},
        'target_return':            {'type': 'float', 'low': ps_tr_low,  'high': ps_tr_high},
        'stop_loss':                {'type': 'float', 'low': ps_sl_low,  'high': ps_sl_high},
        'max_positions':            {'type': 'int',   'low': ps_mp_low,  'high': ps_mp_high},
        'max_hold_days':            {'type': 'int',   'low': ps_mh_low,  'high': ps_mh_high},
        'reverse_signal_threshold': {'type': 'float', 'low': ps_rt_low,  'high': ps_rt_high},
    }

    _progress_bar = st.progress(0, text="워크포워드 준비 중...")

    def _wf_progress(current, total):
        pct = min(1.0, current / total)
        _progress_bar.progress(pct, text=f"기간 {current}/{total} 완료 ({pct*100:.0f}%)")

    wf_result = run_walk_forward(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        train_months=train_months,
        val_months=val_months,
        step_months=step_months,
        n_trials=n_trials,
        metric=metric,
        strategy=strategy,
        initial_capital=initial_capital,
        institution_weight=institution_weight,
        use_tc=use_tc,
        use_divergence=use_divergence,
        tax_rate=tax_rate,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        borrowing_rate=borrowing_rate,
        optuna_param_space=param_space,
        progress_callback=_wf_progress,
        market_cap_top_n=market_cap_top_n,
    )

    _progress_bar.empty()
    st.session_state['wf_result'] = wf_result
    st.rerun()


# ---------------------------------------------------------------------------
# 결과 표시
# ---------------------------------------------------------------------------
if 'wf_result' not in st.session_state:
    st.info("사이드바에서 설정 후 '워크포워드 실행' 버튼을 클릭하세요.")

    # 기존 CSV 업로드 기능 유지
    with st.expander("📁 기존 CSV 업로드 (CLI 결과)"):
        uploaded_file = st.file_uploader("Walk-Forward 결과 CSV", type=['csv'], key="wf_csv_upload")
        if uploaded_file is not None:
            wf_df = pd.read_csv(uploaded_file)
            st.success(f"업로드: {uploaded_file.name} ({len(wf_df)}개 기간)")
            st.dataframe(wf_df, use_container_width=True)

            # 간단한 KPI + 바차트
            for col in ['val_return', 'total_return', 'return']:
                if col in wf_df.columns:
                    returns = wf_df[col]
                    k1, k2, k3 = st.columns(3)
                    k1.metric("평균 수익률", f"{returns.mean():.2f}%")
                    k2.metric("양(+) 기간", f"{(returns > 0).sum()}/{len(returns)}")
                    k3.metric("중앙값 수익률", f"{returns.median():.2f}%")
                    fig = create_wf_period_returns_chart(wf_df)
                    st.plotly_chart(fig, use_container_width=True, theme=None)
                    break
    st.stop()


# ---------------------------------------------------------------------------
# 결과 렌더링
# ---------------------------------------------------------------------------
result = st.session_state['wf_result']
summary_df = result['summary']
periods = result['periods']
combined_daily = result['combined_daily_values']

if summary_df.empty:
    st.warning("유효한 학습/검증 기간이 없습니다. 전체 기간이 학습+검증 기간보다 짧습니다.")
    st.stop()

# ---------------------------------------------------------------------------
# KPI 행
# ---------------------------------------------------------------------------
avg_return = summary_df['total_return'].mean()
avg_sharpe = summary_df['sharpe_ratio'].mean() if 'sharpe_ratio' in summary_df.columns else 0
avg_winrate = summary_df['win_rate'].mean() if 'win_rate' in summary_df.columns else 0
positive_periods = (summary_df['total_return'] > 0).sum()
total_periods = len(summary_df)
total_trades = int(summary_df['total_trades'].sum()) if 'total_trades' in summary_df.columns else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("평균 수익률", f"{avg_return:+.2f}%")
k2.metric("평균 샤프", f"{avg_sharpe:.2f}")
k3.metric("평균 승률", f"{avg_winrate:.1f}%")
k4.metric("양(+) 기간", f"{positive_periods}/{total_periods}")
k5.metric("총 거래", f"{total_trades}건")

# ---------------------------------------------------------------------------
# 기간별 결과 테이블
# ---------------------------------------------------------------------------
st.subheader("기간별 결과")
display_summary = summary_df.copy()
# 보기 좋게 컬럼 순서 정리
_front_cols = ['val_start', 'val_end', 'total_return', 'sharpe_ratio', 'win_rate',
               'max_drawdown', 'profit_factor', 'total_trades']
_front_cols = [c for c in _front_cols if c in display_summary.columns]
_param_cols = [c for c in display_summary.columns if c.startswith('param_')]
_other_cols = [c for c in display_summary.columns if c not in _front_cols and c not in _param_cols]
display_summary = display_summary[_front_cols + _param_cols + _other_cols]
st.dataframe(
    display_summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "total_return": st.column_config.NumberColumn("수익률(%)", format="%.2f"),
        "sharpe_ratio": st.column_config.NumberColumn("샤프", format="%.2f"),
        "win_rate": st.column_config.NumberColumn("승률(%)", format="%.1f"),
        "max_drawdown": st.column_config.NumberColumn("MDD(%)", format="%.2f"),
        "profit_factor": st.column_config.NumberColumn("PF", format="%.2f"),
        "total_trades": st.column_config.NumberColumn("거래수", format="%d"),
    },
)

# ---------------------------------------------------------------------------
# 기간별 검증 수익률 바차트
# ---------------------------------------------------------------------------
st.subheader("기간별 검증 수익률")
fig_returns = create_wf_period_returns_chart(summary_df)
st.plotly_chart(fig_returns, use_container_width=True, theme=None)

# ---------------------------------------------------------------------------
# 파라미터 변화 추이
# ---------------------------------------------------------------------------
param_cols = [c for c in summary_df.columns if c.startswith('param_')]
if param_cols:
    st.subheader("최적 파라미터 변화 추이")

    period_labels = summary_df['val_start'].astype(str) if 'val_start' in summary_df.columns else [
        f"기간 {i+1}" for i in range(len(summary_df))
    ]

    # Plotly 서브플롯으로 파라미터 추이 표시
    varying_params = [c for c in param_cols if summary_df[c].nunique() > 1]
    if varying_params:
        cols_per_row = 2
        rows = (len(varying_params) + cols_per_row - 1) // cols_per_row
        chart_cols = st.columns(cols_per_row)
        for idx, pcol in enumerate(varying_params):
            with chart_cols[idx % cols_per_row]:
                label = pcol.replace('param_', '')
                fig_p = go.Figure(data=go.Scatter(
                    x=list(period_labels),
                    y=summary_df[pcol].values,
                    mode='lines+markers',
                    marker=dict(color='#38bdf8', size=6),
                    line=dict(color='#38bdf8', width=2),
                    hovertemplate=f'{label}: %{{y:.3f}}<extra></extra>',
                ))
                fig_p.update_layout(
                    title=label,
                    height=250,
                    margin=dict(t=30, b=30, l=40, r=20),
                    plot_bgcolor='#0f172a',
                    paper_bgcolor='#1e293b',
                    font=dict(color='#e2e8f0'),
                    xaxis=dict(gridcolor='#334155', tickangle=-45),
                    yaxis=dict(gridcolor='#334155'),
                )
                st.plotly_chart(fig_p, use_container_width=True, theme=None)
    else:
        st.caption("모든 기간에서 동일한 파라미터가 선택되었습니다.")

# ---------------------------------------------------------------------------
# 통합 성과 차트 (equity curve + drawdown)
# ---------------------------------------------------------------------------
combined_trades_list = result['combined_trade_dicts']
if combined_trades_list and not combined_daily.empty:
    st.subheader("통합 성과")
    trades_obj = _deserialize_trades(combined_trades_list)

    from src.backtesting.plotly_visualizer import PlotlyVisualizer
    pv = PlotlyVisualizer(
        trades=trades_obj,
        daily_values=combined_daily,
        initial_capital=result['initial_capital'],
    )

    tab_eq, tab_dd = st.tabs(["수익률 곡선", "낙폭"])
    with tab_eq:
        st.plotly_chart(pv.fig_equity_curve(), use_container_width=True, theme=None)
    with tab_dd:
        st.plotly_chart(pv.fig_drawdown(), use_container_width=True, theme=None)

    # 통합 KPI
    from src.backtesting.metrics import PerformanceMetrics
    if trades_obj:
        combined_metrics = PerformanceMetrics(
            trades=trades_obj,
            daily_values=combined_daily,
            initial_capital=result['initial_capital'],
        )
        cs = combined_metrics.summary()
        cm1, cm2, cm3, cm4, cm5 = st.columns(5)
        cm1.metric("통합 수익률", f"{cs['total_return']:+.2f}%")
        cm2.metric("통합 승률", f"{cs['win_rate']:.1f}%")
        cm3.metric("통합 MDD", f"{cs['max_drawdown']:.2f}%")
        cm4.metric("통합 샤프", f"{cs['sharpe_ratio']:.2f}")
        cm5.metric("통합 거래", f"{cs['total_trades']}건")

# ---------------------------------------------------------------------------
# 거래 내역 테이블
# ---------------------------------------------------------------------------
if combined_trades_list:
    st.subheader("통합 거래 내역")
    trade_df = pd.DataFrame(combined_trades_list)

    if not trade_df.empty:
        display_cols = [
            'stock_name', 'stock_code', 'pattern', 'direction',
            'entry_date', 'entry_price', 'exit_date', 'exit_price',
            'return_pct', 'hold_days', 'exit_reason', 'score', 'signal_count',
        ]
        display_cols = [c for c in display_cols if c in trade_df.columns]
        st.dataframe(
            trade_df[display_cols],
            use_container_width=True,
            height=min(600, len(trade_df) * 40 + 40),
            hide_index=True,
            column_config={
                "return_pct": st.column_config.NumberColumn("수익률(%)", format="%.2f"),
                "entry_price": st.column_config.NumberColumn("진입가", format="%,.0f"),
                "exit_price": st.column_config.NumberColumn("청산가", format="%,.0f"),
                "score": st.column_config.NumberColumn("점수", format="%.0f"),
                "signal_count": st.column_config.NumberColumn("시그널", format="%d"),
            },
        )

        csv = trade_df[display_cols].to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "거래 내역 CSV 다운로드",
            csv,
            file_name="walk_forward_trades.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# 기간별 결과 CSV 다운로드
# ---------------------------------------------------------------------------
st.divider()
csv_summary = summary_df.to_csv(index=False, encoding='utf-8-sig')
st.download_button(
    "기간별 결과 CSV 다운로드",
    csv_summary,
    file_name="walk_forward_summary.csv",
    mime="text/csv",
)

# ---------------------------------------------------------------------------
# 기존 CSV 업로드 (하단)
# ---------------------------------------------------------------------------
with st.expander("📁 기존 CSV 업로드 (CLI 결과)"):
    uploaded_file = st.file_uploader("Walk-Forward 결과 CSV", type=['csv'], key="wf_csv_upload_bottom")
    if uploaded_file is not None:
        wf_df = pd.read_csv(uploaded_file)
        st.success(f"업로드: {uploaded_file.name} ({len(wf_df)}개 기간)")
        st.dataframe(wf_df, use_container_width=True)
