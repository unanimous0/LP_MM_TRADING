"""
백테스트 페이지 - 백테스트 실행 및 결과 시각화

사이드바: BacktestConfig 파라미터 위젯 + Optuna 최적화
메인: KPI 카드 + 5개 Plotly 차트 탭 + 거래 내역
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
from datetime import datetime

from utils.data_loader import (
    run_backtest,
    run_optuna_optimization,
    get_metrics_from_result,
    get_trades_from_result,
    get_date_range,
)
from src.backtesting.plotly_visualizer import PlotlyVisualizer

st.set_page_config(page_title="백테스트", page_icon="📈", layout="wide")
st.title("백테스트 실행")


# ---------------------------------------------------------------------------
# 헬퍼: 위젯 step에 맞춰 반올림
# ---------------------------------------------------------------------------
def _snap(value, step, lo, hi):
    """값을 위젯 step에 맞춰 반올림하고 범위 내로 클램핑"""
    snapped = round(value / step) * step
    return max(lo, min(hi, round(snapped, 10)))


# ---------------------------------------------------------------------------
# 최적화된 파라미터 적용 (위젯 렌더링 전에 실행)
# ---------------------------------------------------------------------------
if 'pending_opt_params' in st.session_state:
    p = st.session_state['pending_opt_params']
    st.session_state['w_min_score'] = _snap(p.get('min_score', 60.0), 5.0, 0.0, 100.0)
    st.session_state['w_min_signals'] = int(max(0, min(3, p.get('min_signals', 1))))
    st.session_state['w_target_return'] = _snap(p.get('target_return', 0.15) * 100, 1.0, 1.0, 50.0)
    st.session_state['w_stop_loss'] = _snap(p.get('stop_loss', -0.075) * 100, 0.5, -30.0, -1.0)
    del st.session_state['pending_opt_params']


# ---------------------------------------------------------------------------
# 사이드바: 파라미터
# ---------------------------------------------------------------------------
st.sidebar.header("백테스트 설정")

min_date, max_date = get_date_range()

# ---------------------------------------------------------------------------
# 기간 설정
# ---------------------------------------------------------------------------
st.sidebar.subheader("기간")
use_split = st.sidebar.checkbox(
    "최적화 / 검증 기간 분리",
    value=False,
    help="최적화 기간에서 최적 파라미터를 탐색하고, 검증 기간에서 백테스트를 실행합니다. 과적합 없는 신뢰도 높은 결과를 얻을 수 있습니다.",
)

if use_split:
    st.sidebar.caption("🔧 최적화 기간 (파라미터 탐색)")
    opt_start_date = st.sidebar.date_input(
        "최적화 시작일",
        value=datetime.strptime("2025-01-01", "%Y-%m-%d"),
        min_value=datetime.strptime(min_date, "%Y-%m-%d"),
        max_value=datetime.strptime(max_date, "%Y-%m-%d"),
        key="w_opt_start",
    )
    opt_end_date = st.sidebar.date_input(
        "최적화 종료일",
        value=datetime.strptime("2025-09-30", "%Y-%m-%d"),
        min_value=datetime.strptime(min_date, "%Y-%m-%d"),
        max_value=datetime.strptime(max_date, "%Y-%m-%d"),
        key="w_opt_end",
    )
    st.sidebar.caption("✅ 검증 기간 (백테스트 실행)")
    val_start_date = st.sidebar.date_input(
        "검증 시작일",
        value=datetime.strptime("2025-10-01", "%Y-%m-%d"),
        min_value=datetime.strptime(min_date, "%Y-%m-%d"),
        max_value=datetime.strptime(max_date, "%Y-%m-%d"),
        key="w_val_start",
    )
    val_end_date = st.sidebar.date_input(
        "검증 종료일",
        value=datetime.strptime(max_date, "%Y-%m-%d"),
        min_value=datetime.strptime(min_date, "%Y-%m-%d"),
        max_value=datetime.strptime(max_date, "%Y-%m-%d"),
        key="w_val_end",
    )
else:
    _start = st.sidebar.date_input(
        "시작일",
        value=datetime.strptime("2025-01-01", "%Y-%m-%d"),
        min_value=datetime.strptime(min_date, "%Y-%m-%d"),
        max_value=datetime.strptime(max_date, "%Y-%m-%d"),
    )
    _end = st.sidebar.date_input(
        "종료일",
        value=datetime.strptime(max_date, "%Y-%m-%d"),
        min_value=datetime.strptime(min_date, "%Y-%m-%d"),
        max_value=datetime.strptime(max_date, "%Y-%m-%d"),
    )
    opt_start_date = val_start_date = _start
    opt_end_date = val_end_date = _end

# 전략
strategy = st.sidebar.selectbox(
    "전략 방향",
    options=['long', 'short', 'both'],
    format_func=lambda x: {'long': 'Long (순매수)', 'short': 'Short (순매도)', 'both': 'Long+Short (병행)'}[x],
)

# 진입 조건
st.sidebar.subheader("진입 조건")
min_score = st.sidebar.slider("최소 점수", 0.0, 100.0, 60.0, step=5.0, key="w_min_score")
min_signals = st.sidebar.slider("최소 시그널 수", 0, 3, 1, key="w_min_signals")

# 청산 조건
st.sidebar.subheader("청산 조건")
target_return = st.sidebar.slider("목표 수익률 (%)", 1.0, 50.0, 15.0, step=1.0, key="w_target_return") / 100
stop_loss = st.sidebar.slider("손절 비율 (%)", -30.0, -1.0, -7.5, step=0.5, key="w_stop_loss") / 100
max_hold_days = st.sidebar.number_input("최대 보유 기간 (일)", 1, 999, 999)
reverse_threshold = st.sidebar.slider("반대 수급 청산 점수", 0.0, 100.0, 60.0, step=5.0)

# 포트폴리오
st.sidebar.subheader("포트폴리오")
initial_capital = st.sidebar.number_input("초기 자본금 (원)", 1_000_000, 1_000_000_000, 10_000_000, step=1_000_000)
max_positions = st.sidebar.slider("최대 동시 포지션", 1, 20, 5)

# 고급 설정
with st.sidebar.expander("고급 설정"):
    institution_weight = st.slider("기관 가중치", 0.0, 1.0, 0.3, step=0.05, key="w_institution_weight")

# ---------------------------------------------------------------------------
# 실행 버튼
# ---------------------------------------------------------------------------
run_clicked = st.sidebar.button("백테스트 실행", type="primary", use_container_width=True)

if run_clicked:
    st.session_state['bt_result'] = run_backtest(
        start_date=val_start_date.strftime("%Y-%m-%d"),
        end_date=val_end_date.strftime("%Y-%m-%d"),
        strategy=strategy,
        min_score=min_score,
        min_signals=min_signals,
        target_return=target_return,
        stop_loss=stop_loss,
        max_hold_days=max_hold_days,
        initial_capital=float(initial_capital),
        max_positions=max_positions,
        institution_weight=institution_weight,
        reverse_threshold=reverse_threshold,
    )
    st.session_state['bt_use_split'] = use_split
    st.session_state['bt_opt_period'] = (opt_start_date.strftime("%Y-%m-%d"), opt_end_date.strftime("%Y-%m-%d"))
    st.session_state['bt_val_period'] = (val_start_date.strftime("%Y-%m-%d"), val_end_date.strftime("%Y-%m-%d"))

# ---------------------------------------------------------------------------
# Optuna 최적화 섹션
# ---------------------------------------------------------------------------
st.sidebar.divider()
with st.sidebar.expander("파라미터 최적화 (Optuna)"):
    opt_n_trials = st.slider("Trial 수", 10, 200, 30, step=10, key="w_opt_n_trials")
    opt_metric = st.selectbox(
        "평가 지표",
        options=['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor'],
        format_func=lambda x: {
            'sharpe_ratio': 'Sharpe Ratio',
            'total_return': '총 수익률',
            'win_rate': '승률',
            'profit_factor': 'Profit Factor',
        }[x],
        key="w_opt_metric",
    )
    if use_split:
        st.caption(f"최적화: {opt_start_date} ~ {opt_end_date}  →  검증: {val_start_date} ~ {val_end_date}")
    else:
        st.warning("⚠️ 같은 기간에서 최적화와 백테스트를 수행합니다. 이 결과로 전략의 실제 수익성을 판단할 수 없습니다. 위의 '최적화 / 검증 기간 분리'를 사용하세요.")
    opt_clicked = st.button("최적 파라미터 찾기", use_container_width=True)

if opt_clicked:
    st.session_state.pop('opt_result', None)
    spinner_msg = f"Optuna 최적화 실행 중... ({opt_n_trials} trials)"
    if use_split:
        spinner_msg += f"  |  최적화: {opt_start_date}~{opt_end_date}"
    with st.spinner(spinner_msg):
        opt_result = run_optuna_optimization(
            start_date=opt_start_date.strftime("%Y-%m-%d"),
            end_date=opt_end_date.strftime("%Y-%m-%d"),
            strategy=strategy,
            n_trials=opt_n_trials,
            metric=opt_metric,
            initial_capital=float(initial_capital),
            max_positions=max_positions,
            max_hold_days=max_hold_days,
            reverse_threshold=reverse_threshold,
        )
    if opt_result:
        st.session_state['opt_result'] = opt_result
        st.session_state['opt_metric'] = opt_metric
        st.session_state['pending_opt_params'] = opt_result['params']
        # 최적화된 파라미터로 검증 기간 백테스트 자동 실행
        params = opt_result['params']
        with st.spinner(f"검증 기간 백테스트 실행 중... ({val_start_date}~{val_end_date})"):
            st.session_state['bt_result'] = run_backtest(
                start_date=val_start_date.strftime("%Y-%m-%d"),
                end_date=val_end_date.strftime("%Y-%m-%d"),
                strategy=strategy,
                min_score=params['min_score'],
                min_signals=params['min_signals'],
                target_return=params['target_return'],
                stop_loss=params['stop_loss'],
                max_hold_days=max_hold_days,
                initial_capital=float(initial_capital),
                max_positions=max_positions,
                institution_weight=institution_weight,
                reverse_threshold=reverse_threshold,
            )
        st.session_state['bt_use_split'] = use_split
        st.session_state['bt_opt_period'] = (opt_start_date.strftime("%Y-%m-%d"), opt_end_date.strftime("%Y-%m-%d"))
        st.session_state['bt_val_period'] = (val_start_date.strftime("%Y-%m-%d"), val_end_date.strftime("%Y-%m-%d"))
        st.rerun()
    else:
        st.error("최적화 실패: 완료된 Trial이 없습니다. Trial 수를 늘리거나 기간을 조정해보세요.")

# ---------------------------------------------------------------------------
# 최적화 결과 표시
# ---------------------------------------------------------------------------
if 'opt_result' in st.session_state:
    opt_r = st.session_state['opt_result']
    opt_m = st.session_state.get('opt_metric', 'sharpe_ratio')

    with st.expander("최적화 결과", expanded=True):
        c1, c2, c3 = st.columns(3)
        metric_names = {
            'sharpe_ratio': 'Sharpe Ratio',
            'total_return': '총 수익률',
            'win_rate': '승률',
            'profit_factor': 'Profit Factor',
        }
        metric_val = opt_r.get(opt_m, 0)
        if opt_m in ('total_return', 'win_rate'):
            c1.metric(metric_names[opt_m], f"{metric_val:.2f}%")
        else:
            c1.metric(metric_names[opt_m], f"{metric_val:.4f}")
        c2.metric("완료 Trial", f"{opt_r['total_complete']}개")
        c3.metric("중단 Trial", f"{opt_r['total_pruned']}개")

        params = opt_r['params']
        param_labels = {
            'min_score': ('최소 점수', f"{params['min_score']:.1f}"),
            'min_signals': ('최소 시그널 수', f"{params['min_signals']}"),
            'target_return': ('목표 수익률', f"{params['target_return']*100:.1f}%"),
            'stop_loss': ('손절 비율', f"{params['stop_loss']*100:.1f}%"),
        }
        st.markdown("**최적 파라미터:**")
        cols = st.columns(4)
        for i, (key, (label, val)) in enumerate(param_labels.items()):
            cols[i].metric(label, val)

# ---------------------------------------------------------------------------
# 결과 표시
# ---------------------------------------------------------------------------
if 'bt_result' not in st.session_state:
    st.info("사이드바에서 파라미터를 설정하고 '백테스트 실행' 버튼을 클릭하세요.")
    st.stop()

result = st.session_state['bt_result']
trades = get_trades_from_result(result)
metrics = get_metrics_from_result(result)

if not trades:
    st.warning("백테스트 기간 내 거래가 발생하지 않았습니다. 파라미터를 조정해보세요.")
    st.stop()

# ---------------------------------------------------------------------------
# 기간 표시 배너
# ---------------------------------------------------------------------------
_use_split = st.session_state.get('bt_use_split', False)
_opt_p = st.session_state.get('bt_opt_period')
_val_p = st.session_state.get('bt_val_period')

if _use_split and _opt_p and _val_p:
    st.info(
        f"🔧 최적화 기간: **{_opt_p[0]} ~ {_opt_p[1]}** &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"✅ 검증 기간: **{_val_p[0]} ~ {_val_p[1]}**"
    )
elif _val_p:
    st.caption(f"백테스트 기간: {_val_p[0]} ~ {_val_p[1]}")

# ---------------------------------------------------------------------------
# KPI 행
# ---------------------------------------------------------------------------
summary = metrics.summary()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("총 수익률", f"{summary['total_return']:+.2f}%")
kpi2.metric("승률", f"{summary['win_rate']:.1f}%")
kpi3.metric("MDD", f"{summary['max_drawdown']:.2f}%")
kpi4.metric("샤프 비율", f"{summary['sharpe_ratio']:.2f}")
kpi5.metric("총 거래", f"{summary['total_trades']}건")

# ---------------------------------------------------------------------------
# 차트 탭 (PlotlyVisualizer 재사용)
# ---------------------------------------------------------------------------
pv = PlotlyVisualizer(
    trades=trades,
    daily_values=result['daily_values'],
    initial_capital=result['initial_capital'],
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "수익률 곡선", "낙폭", "월별 수익률", "수익률 분포", "패턴별 성과", "거래 내역",
])

with tab1:
    st.plotly_chart(pv.fig_equity_curve(), use_container_width=True)

with tab2:
    st.plotly_chart(pv.fig_drawdown(), use_container_width=True)

with tab3:
    st.plotly_chart(pv.fig_monthly_returns(), use_container_width=True)

with tab4:
    fig = pv.fig_return_distribution()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("거래 데이터가 없습니다.")

with tab5:
    fig = pv.fig_pattern_performance()
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("거래 데이터가 없습니다.")

with tab6:
    trade_df = pd.DataFrame([t.to_dict() for t in trades])
    display_cols = [
        'stock_name', 'stock_code', 'pattern', 'direction',
        'entry_date', 'entry_price', 'exit_date', 'exit_price',
        'return_pct', 'hold_days', 'exit_reason', 'signal_count',
    ]
    display_cols = [c for c in display_cols if c in trade_df.columns]
    st.dataframe(
        trade_df[display_cols],
        use_container_width=True,
        height=min(600, len(trade_df) * 40 + 40),
        column_config={
            "return_pct": st.column_config.NumberColumn("수익률 (%)", format="%.2f"),
            "entry_price": st.column_config.NumberColumn("진입가", format="%,.0f"),
            "exit_price": st.column_config.NumberColumn("청산가", format="%,.0f"),
        },
    )

    # 다운로드 버튼
    csv = trade_df[display_cols].to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        "거래 내역 CSV 다운로드",
        csv,
        file_name="backtest_trades.csv",
        mime="text/csv",
    )
