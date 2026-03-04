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
    run_backtest_with_progress,
    run_optuna_optimization,
    get_optuna_trial_count,
    get_metrics_from_result,
    get_trades_from_result,
    get_date_range,
    get_db_connection,
    save_backtest_history,
    get_backtest_history,
    delete_backtest_history,
)
from src.backtesting.plotly_visualizer import PlotlyVisualizer

st.set_page_config(page_title="백테스트", page_icon="📈", layout="wide")
st.title("백테스트 실행")

st.markdown("""
<style>
/* 사이드바 너비 확장 */
section[data-testid="stSidebar"] { min-width: 340px !important; max-width: 340px !important; }
section[data-testid="stSidebar"] > div:first-child { width: 340px !important; }

/* 슬라이더-입력박스 세로 중앙 정렬 (마지막 컬럼을 하단 정렬) */
section[data-testid="stSidebar"] div.stHorizontalBlock > div.stColumn:last-child {
    display: flex; flex-direction: column; justify-content: flex-end; padding-bottom: 0.4rem;
}
/* 위젯 테두리 가시성 */
div[data-baseweb="select"] > div { border-color: #333 !important; }
div[data-baseweb="input"] input, div[data-baseweb="input"] > div { border-color: #333 !important; }
[data-testid="stDateInput"] > div > div > div { border-color: #333 !important; }
[data-testid="stExpander"] { border-color: #222 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 헬퍼: 위젯 step에 맞춰 반올림
# ---------------------------------------------------------------------------
def _snap(value, step, lo, hi):
    """값을 위젯 step에 맞춰 반올림하고 범위 내로 클램핑"""
    snapped = round(value / step) * step
    return max(lo, min(hi, round(snapped, 10)))


def _synced_slider(label, min_val, max_val, step, key, is_int=False):
    """슬라이더(드래그) + 숫자 직접 입력 연동 위젯"""
    ni_key = f"{key}_ni"
    # number_input 키 초기화
    if ni_key not in st.session_state:
        st.session_state[ni_key] = st.session_state.get(key, min_val)

    def _on_slider():
        st.session_state[ni_key] = st.session_state[key]

    def _on_input():
        v = st.session_state[ni_key]
        if is_int:
            v = int(max(min_val, min(max_val, v)))
        else:
            v = float(max(min_val, min(max_val, v)))
        st.session_state[key] = v

    col_s, col_n = st.sidebar.columns([3, 1])
    with col_s:
        val = st.slider(label, min_val, max_val, step=step, key=key, on_change=_on_slider)
    with col_n:
        # 슬라이더 라벨 높이만큼 spacer → 세로 중앙 정렬
        st.markdown('<div style="height:1.65rem"></div>', unsafe_allow_html=True)
        st.number_input(
            "　", min_value=min_val, max_value=max_val, step=step,
            key=ni_key, on_change=_on_input, label_visibility="collapsed",
        )
    return val


# ---------------------------------------------------------------------------
# 위젯 기본값 초기화 (최초 1회만 - key+value 동시 지정 경고 방지)
# ---------------------------------------------------------------------------
_defaults = {
    'w_min_score': 60.0,
    'w_min_signals': 1,
    'w_target_return': 15.0,
    'w_stop_loss': -7.5,
    'w_reverse_threshold': 60.0,
    'w_max_positions': 5,
    'w_max_hold_days': 500,
    'w_initial_capital': 10_000_000,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# 초기 자본금 텍스트 입력 초기화 (쉼표 포맷)
if 'w_initial_capital_text' not in st.session_state:
    st.session_state['w_initial_capital_text'] = f"{st.session_state['w_initial_capital']:,}"

def _on_capital_change():
    raw = st.session_state['w_initial_capital_text']
    try:
        val = int(raw.replace(',', '').replace(' ', ''))
        val = max(1_000_000, min(1_000_000_000, val))
    except ValueError:
        val = st.session_state['w_initial_capital']
    st.session_state['w_initial_capital'] = val
    st.session_state['w_initial_capital_text'] = f"{val:,}"


# ---------------------------------------------------------------------------
# 최적화된 파라미터 적용 (위젯 렌더링 전에 실행)
# ---------------------------------------------------------------------------
if 'pending_opt_params' in st.session_state:
    p = st.session_state['pending_opt_params']
    st.session_state['w_min_score'] = _snap(p.get('min_score', 60.0), 5.0, 0.0, 100.0)
    st.session_state['w_min_signals'] = int(max(0, min(3, p.get('min_signals', 1))))
    st.session_state['w_target_return'] = _snap(p.get('target_return', 0.15) * 100, 1.0, 1.0, 200.0)
    st.session_state['w_stop_loss'] = _snap(p.get('stop_loss', -0.075) * 100, 0.5, -100.0, -1.0)
    st.session_state['w_max_positions'] = int(max(1, min(50, p.get('max_positions', 5))))
    st.session_state['w_max_hold_days'] = int(max(1, min(500, p.get('max_hold_days', 30))))
    st.session_state['w_reverse_threshold'] = _snap(p.get('reverse_signal_threshold', 60.0), 5.0, 0.0, 115.0)
    # number_input(_ni) 키도 슬라이더와 동기화
    for _k in ['w_min_score', 'w_min_signals', 'w_target_return', 'w_stop_loss',
               'w_max_positions', 'w_reverse_threshold']:
        st.session_state[f'{_k}_ni'] = st.session_state[_k]
    del st.session_state['pending_opt_params']


# ---------------------------------------------------------------------------
# 사이드바 ① 기간 분리 설정 (최상단 — 최적화·백테스트 공통)
# ---------------------------------------------------------------------------
min_date, max_date = get_date_range()

use_split = st.sidebar.checkbox(
    "최적화 / 검증 기간 분리",
    value=True,
    help="최적화 기간에서 최적 파라미터를 탐색하고, 검증 기간에서 백테스트를 실행합니다. 과적합 없는 신뢰도 높은 결과를 얻을 수 있습니다.",
)
if not use_split:
    st.sidebar.warning("⚠️ 같은 기간에서 최적화·백테스트를 수행합니다. 과적합으로 신뢰하기 어려운 결과가 나올 수 있습니다.")

if use_split:
    st.sidebar.caption("🔧 최적화 기간 (파라미터 탐색)")
    _max_dt = datetime.strptime(max_date, "%Y-%m-%d")
    _max_value = _max_dt.replace(month=12, day=31)
    opt_start_date = st.sidebar.date_input(
        "최적화 시작일",
        value=datetime.strptime("2023-01-02", "%Y-%m-%d"),
        min_value=datetime.strptime(min_date, "%Y-%m-%d"),
        max_value=_max_value,
        key="w_opt_start",
    )
    opt_end_date = st.sidebar.date_input(
        "최적화 종료일",
        value=datetime.strptime("2025-09-30", "%Y-%m-%d"),
        min_value=datetime.strptime(min_date, "%Y-%m-%d"),
        max_value=_max_value,
        key="w_opt_end",
    )
    st.sidebar.caption("✅ 검증 기간 (백테스트 실행)")
    val_start_date = st.sidebar.date_input(
        "검증 시작일",
        value=datetime.strptime("2025-10-01", "%Y-%m-%d"),
        min_value=datetime.strptime(min_date, "%Y-%m-%d"),
        max_value=_max_value,
        key="w_val_start",
    )
    val_end_date = st.sidebar.date_input(
        "검증 종료일",
        value=_max_dt,
        min_value=datetime.strptime(min_date, "%Y-%m-%d"),
        max_value=_max_value,
        key="w_val_end",
    )
else:
    _max_dt = datetime.strptime(max_date, "%Y-%m-%d")
    _max_value = _max_dt.replace(month=12, day=31)
    _start = st.sidebar.date_input(
        "시작일",
        value=datetime.strptime("2023-01-02", "%Y-%m-%d"),
        min_value=datetime.strptime(min_date, "%Y-%m-%d"),
        max_value=_max_value,
    )
    _end = st.sidebar.date_input(
        "종료일",
        value=_max_dt,
        min_value=datetime.strptime(min_date, "%Y-%m-%d"),
        max_value=_max_value,
    )
    opt_start_date = val_start_date = _start
    opt_end_date = val_end_date = _end

st.sidebar.divider()

# ---------------------------------------------------------------------------
# 사이드바 ② 파라미터 최적화
# ---------------------------------------------------------------------------
# strategy는 expander 안에서 get_optuna_trial_count()에 필요하므로 먼저 정의
strategy = st.sidebar.selectbox(
    "전략 방향",
    options=['long', 'short', 'both'],
    format_func=lambda x: {'long': 'Long (순매수)', 'short': 'Short (순매도)', 'both': 'Long+Short (병행)'}[x],
)

with st.sidebar.expander("⚡ 파라미터 최적화 (Optuna)"):
    opt_n_trials = st.slider("이번 추가 Trial 수", 10, 200, 100, step=10, key="w_opt_n_trials")
    opt_metric = st.selectbox(
        "평가 지표",
        options=['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor'],
        format_func=lambda x: {
            'sharpe_ratio': 'Sharpe Ratio',
            'total_return': '총 수익률',
            'win_rate': '승률',
            'profit_factor': 'Profit Factor',
        }[x],
        key="w_opt_metric",
    )
    opt_reset = st.checkbox(
        "누적 Trial 초기화 후 실행",
        value=False,
        help="체크 시 이전 누적 결과를 삭제하고 새로 시작합니다.",
    )
    # ① 현재 누적 현황 (DB에서 직접 읽기 — 새로고침/재시작 후에도 유지)
    if opt_reset:
        st.caption("🔄 초기화 후 새로 시작")
    else:
        _acc = get_optuna_trial_count(
            start_date=opt_start_date.strftime("%Y-%m-%d"),
            end_date=opt_end_date.strftime("%Y-%m-%d"),
            strategy=strategy,
            metric=opt_metric,
        )
        if _acc > 0:
            st.caption(f"📊 이전 누적 {_acc}회 → 실행 후 약 {_acc + opt_n_trials}회")
        else:
            st.caption("📊 첫 실행 (누적 없음)")
    opt_clicked = st.button("최적 파라미터 찾기", use_container_width=True, type="primary")

st.sidebar.divider()

# ---------------------------------------------------------------------------
# 사이드바 ③ 백테스트 설정
# ---------------------------------------------------------------------------
st.sidebar.header("백테스트 설정")

# 🧪 최적화 대상 파라미터
st.sidebar.subheader("🧪 최적화 대상 파라미터")
st.sidebar.caption("'최적 파라미터 찾기' 실행 시 Optuna가 자동 결정합니다. 수동으로 설정 후 '백테스트 실행'도 가능합니다.")
min_score = _synced_slider("최소 점수", 0.0, 100.0, 5.0, "w_min_score")
min_signals = _synced_slider("최소 시그널 수", 0, 3, 1, "w_min_signals", is_int=True)
target_return = _synced_slider("목표 수익률 (%)", 1.0, 200.0, 1.0, "w_target_return") / 100
stop_loss = _synced_slider("손절 비율 (%)", -100.0, -1.0, 0.5, "w_stop_loss") / 100
max_positions = _synced_slider("최대 동시 포지션", 1, 50, 1, "w_max_positions", is_int=True)
max_hold_days = st.sidebar.number_input("최대 보유 기간 (일)", 1, 500, key="w_max_hold_days")
reverse_threshold = _synced_slider("반대 수급 청산 점수", 0.0, 115.0, 5.0, "w_reverse_threshold")

st.sidebar.divider()

# 🔒 고정 조건
st.sidebar.subheader("🔒 고정 조건")
st.sidebar.caption("최적화·백테스트 모두 이 값으로 고정됩니다.")
st.sidebar.text_input("초기 자본금 (원)", key='w_initial_capital_text', on_change=_on_capital_change)
initial_capital = float(st.session_state['w_initial_capital'])
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
with st.sidebar.expander("스코어링 버전", expanded=False):
    st.caption("현재 스코어링 개선 항목(2026-02-25)의 적용 여부를 설정합니다. OFF 시 개선 이전 점수 체계로 백테스트해 효과를 비교할 수 있습니다.")
    use_tc = st.checkbox(
        "Temporal Consistency (tc)",
        value=True,
        key="w_use_tc",
        help="tc 기준: 5D≥10D≥…≥500D 순서 일관성 (0~1)\n"
             "ON: 급등형 진입조건(tc≥0.5) + 점수 보너스 ±10점 적용\n"
             "OFF: 조건 무시, 보너스 없음 (개선 이전 동작)",
    )
    use_divergence = st.checkbox(
        "Divergence (단기이격/중기이격)",
        value=True,
        key="w_use_divergence",
        help="단기/중기 이격도를 점수에 반영 (가중치 0.10+0.10)\n"
             "ON: 현재 가중치 (long_divergence 0.15 / average 0.10 / short_divergence 0.10 / mid_divergence 0.10)\n"
             "OFF: 레거시 가중치 (long_divergence 0.20 / average 0.20 / short_divergence 0.00 / mid_divergence 0.00)",
    )
    if not use_tc or not use_divergence:
        st.info("⚠️ 일부 OFF — 스코어링 개선 이전 동작으로 실행됩니다.")

with st.sidebar.expander("거래 비용", expanded=False):
    tax_rate = st.number_input("증권거래세 (%)", 0.00, 1.00, 0.20, step=0.01, format="%.2f", key="w_tax_rate") / 100
    commission_rate = st.number_input("수수료 (%)", 0.000, 1.000, 0.015, step=0.001, format="%.3f", key="w_commission_rate") / 100
    slippage_rate = st.number_input("슬리피지 (%)", 0.00, 1.00, 0.10, step=0.01, format="%.2f", key="w_slippage_rate") / 100
    borrowing_rate = st.number_input("공매도 차입비용 (%/연)", 0.0, 20.0, 3.0, step=0.5, format="%.1f", key="w_borrowing_rate") / 100

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
        tax_rate=tax_rate,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        borrowing_rate=borrowing_rate,
        use_tc=use_tc,
        use_divergence=use_divergence,
    )
    st.session_state['bt_use_split'] = use_split
    st.session_state['bt_opt_period'] = (opt_start_date.strftime("%Y-%m-%d"), opt_end_date.strftime("%Y-%m-%d"))
    st.session_state['bt_val_period'] = (val_start_date.strftime("%Y-%m-%d"), val_end_date.strftime("%Y-%m-%d"))
    st.session_state['bt_use_tc'] = use_tc
    st.session_state['bt_use_divergence'] = use_divergence

# ---------------------------------------------------------------------------
# Optuna 최적화 실행
# ---------------------------------------------------------------------------

if opt_clicked:
    # ② 이전 최고값 저장 (같은 metric일 때만, reset 시 제거)
    if opt_reset or st.session_state.get('opt_metric') != opt_metric:
        st.session_state.pop('opt_prev_best', None)
    elif 'opt_result' in st.session_state:
        st.session_state['opt_prev_best'] = st.session_state['opt_result'].get(opt_metric)
    st.session_state.pop('opt_result', None)
    _opt_progress_bar = st.progress(0, text="사전 계산 중...")
    _opt_status = st.empty()

    def _opt_progress_callback(current, total):
        if current == 0:
            _opt_progress_bar.progress(0.0, text="사전 계산 중... (패턴/시그널 벡터화)")
        else:
            pct = min(1.0, current / total)
            display = min(current, total)
            _opt_progress_bar.progress(pct, text=f"최적화 중... {display}/{total} trial ({pct*100:.0f}%)")

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
        institution_weight=institution_weight,
        progress_callback=_opt_progress_callback,
        reset_study=opt_reset,
        tax_rate=tax_rate,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
        borrowing_rate=borrowing_rate,
        use_tc=use_tc,
        use_divergence=use_divergence,
    )
    _opt_progress_bar.empty()
    _opt_status.empty()
    if opt_result:
        st.session_state['opt_result'] = opt_result
        st.session_state['opt_metric'] = opt_metric
        st.session_state['pending_opt_params'] = opt_result['params']
        # 최적화된 파라미터로 검증 기간 백테스트 자동 실행
        params = opt_result['params']
        _bt_progress_bar = st.progress(0, text="백테스트 준비 중...")

        def _bt_progress_callback(current, total):
            pct = min(1.0, current / total)
            display = min(current, total)
            _bt_progress_bar.progress(pct, text=f"백테스트 중... {display}/{total}일 ({pct*100:.0f}%)")

        st.session_state['bt_result'] = run_backtest_with_progress(
            start_date=val_start_date.strftime("%Y-%m-%d"),
            end_date=val_end_date.strftime("%Y-%m-%d"),
            strategy=strategy,
            min_score=params['min_score'],
            min_signals=params['min_signals'],
            target_return=params['target_return'],
            stop_loss=params['stop_loss'],
            max_hold_days=params['max_hold_days'],
            initial_capital=float(initial_capital),
            max_positions=params['max_positions'],
            institution_weight=institution_weight,
            reverse_threshold=params['reverse_signal_threshold'],
            progress_callback=_bt_progress_callback,
            tax_rate=tax_rate,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            borrowing_rate=borrowing_rate,
            use_tc=use_tc,
            use_divergence=use_divergence,
        )
        _bt_progress_bar.empty()
        st.session_state['bt_use_split'] = use_split
        st.session_state['bt_opt_period'] = (opt_start_date.strftime("%Y-%m-%d"), opt_end_date.strftime("%Y-%m-%d"))
        st.session_state['bt_val_period'] = (val_start_date.strftime("%Y-%m-%d"), val_end_date.strftime("%Y-%m-%d"))
        st.session_state['bt_use_tc'] = use_tc
        st.session_state['bt_use_divergence'] = use_divergence
        st.rerun()
    else:
        st.error("최적화 실패: 완료된 Trial이 없습니다. Trial 수를 늘리거나 기간을 조정해보세요.")

# ---------------------------------------------------------------------------
# 결과 박스 공통 CSS (최적화=주황, 검증=초록 — :has() 로 자동 구분)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has([style*="ff9800"]) {
    border-color: rgba(255, 152, 0, 0.4) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:has([style*="00c853"]) {
    border-color: rgba(0, 200, 83, 0.35) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 최적화 결과 표시
# ---------------------------------------------------------------------------
if 'opt_result' in st.session_state:
    opt_r = st.session_state['opt_result']
    opt_m = st.session_state.get('opt_metric', 'sharpe_ratio')
    _bt_opt_p = st.session_state.get('bt_opt_period')
    _opt_period_str = f"{_bt_opt_p[0]} ~ {_bt_opt_p[1]}" if _bt_opt_p else ""

    metric_names = {
        'sharpe_ratio': 'Sharpe Ratio',
        'total_return': '총 수익률',
        'win_rate': '승률',
        'profit_factor': 'Profit Factor',
    }
    metric_val = opt_r.get(opt_m, 0)
    metric_display = f"{metric_val:.2f}%" if opt_m in ('total_return', 'win_rate') else f"{metric_val:.4f}"

    with st.container(border=True):
        existing_before = opt_r.get('existing_before', 0)
        added_this_run = opt_r['total_complete'] - existing_before
        _add_str = f"&nbsp;(이번 +{added_this_run}회 추가)" if existing_before > 0 else ""
        st.markdown(
            '<div style="border-left:4px solid #ff9800;padding:10px 18px;margin-bottom:20px;background-color:rgba(255,152,0,0.07);border-radius:0 8px 8px 0;">'
            f'<div style="font-size:1.25rem;font-weight:700;color:#ff9800;margin-bottom:4px;">🔧 최적화 결과 (In-Sample)</div>'
            f'<div style="font-size:0.82rem;color:#888;line-height:1.5;">최적화 기간 <strong style="color:#aaa">{_opt_period_str}</strong>&nbsp;·&nbsp;누적 <strong style="color:#ff9800">{opt_r["total_complete"]}회</strong> trial{_add_str}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        params = opt_r['params']
        col_stats, col_div, col_params = st.columns([3, 0.08, 4])

        with col_stats:
            s1, s2, s3 = st.columns(3)
            # ② 이전 대비 delta 표시
            _prev_best = st.session_state.get('opt_prev_best')
            if _prev_best is not None:
                _delta = metric_val - _prev_best
                _delta_str = f"{_delta:+.2f}%" if opt_m in ('total_return', 'win_rate') else f"{_delta:+.4f}"
                s1.metric(metric_names[opt_m], metric_display, delta=_delta_str)
            else:
                s1.metric(metric_names[opt_m], metric_display)
            s2.metric("완료 Trial", f"{opt_r['total_complete']}개")
            s3.metric("중단 Trial", f"{opt_r['total_pruned']}개")

        with col_div:
            st.markdown(
                '<div style="border-left: 1px solid rgba(128,128,128,0.25); height: 72px; margin: 4px auto;"></div>',
                unsafe_allow_html=True,
            )

        with col_params:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("최소 점수", f"{params['min_score']:.1f}")
            p2.metric("최소 시그널", f"{params['min_signals']}")
            p3.metric("목표 수익률", f"{params['target_return']*100:.1f}%")
            p4.metric("손절", f"{params['stop_loss']*100:.1f}%")
            p5, p6, p7, _ = st.columns(4)
            p5.metric("최대 포지션", f"{params['max_positions']}개")
            p6.metric("최대 보유일", f"{params['max_hold_days']}일")
            p7.metric("반대수급 청산", f"{params['reverse_signal_threshold']:.0f}점")

        # ③ 누적 study 정보 (접힌 상태, 작게)
        existing_before = opt_r.get('existing_before', 0)
        added_this_run = opt_r['total_complete'] - existing_before
        _strategy_key = opt_r['params'].get('strategy', '?')
        _sd = _bt_opt_p[0].replace('-', '') if _bt_opt_p else ''
        _ed = _bt_opt_p[1].replace('-', '') if _bt_opt_p else ''
        _sname = f"opt__{_strategy_key}__{_sd}__{_ed}__{opt_m}"
        with st.expander("💾 누적 study 정보", expanded=False):
            st.caption(f"📁 저장: `data/optuna_studies.db`")
            st.caption(f"🔑 Study: `{_sname}`")
            st.caption(f"이번 실행 전 누적: {existing_before}회 · 이번 추가: +{added_this_run}회 · 총: {opt_r['total_complete']}회")

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

# 스코어링 버전 배너
_bt_use_tc = st.session_state.get('bt_use_tc', True)
_bt_use_divergence = st.session_state.get('bt_use_divergence', True)
if _bt_use_tc and _bt_use_divergence:
    st.caption("📐 스코어링: **현재 버전** (Temporal Consistency + Divergence 적용)")
else:
    _off_items = []
    if not _bt_use_tc:
        _off_items.append("Temporal Consistency OFF")
    if not _bt_use_divergence:
        _off_items.append("Divergence OFF")
    st.warning(f"📐 스코어링: **이전 버전** ({', '.join(_off_items)}) — 개선 효과 비교용")

# ---------------------------------------------------------------------------
# 기간 종료 청산 종목 표시
# ---------------------------------------------------------------------------
end_trades = [t for t in trades if t.exit_reason == 'end']
with st.expander(f"⚠️ 기간 종료 시 청산된 포지션: {len(end_trades)}개", expanded=False):
    if not end_trades:
        st.caption("기간 내 모든 포지션이 익절/손절/시간손절/반대수급으로 정상 청산되었습니다.")
    else:
        st.caption(
            "익절/손절 조건을 충족하지 못한 채 백테스트 종료일까지 보유되어 "
            "마지막 날 종가로 청산된 종목입니다. "
            "보유 기간이 짧아 전략 효과가 반영되지 않았을 수 있습니다."
        )
        end_df = pd.DataFrame([{
            '종목명': t.stock_name,
            '종목코드': t.stock_code,
            '진입일': t.entry_date,
            '보유일': t.hold_days,
            '수익률(%)': round(t.return_pct, 2),
            '진입가': int(t.entry_price),
            '청산가(종료일 종가)': int(t.exit_price),
        } for t in end_trades])
        st.dataframe(
            end_df,
            use_container_width=True,
            column_config={
                "수익률(%)": st.column_config.NumberColumn(format="%.2f"),
                "진입가": st.column_config.NumberColumn(format="%,d"),
                "청산가(종료일 종가)": st.column_config.NumberColumn(format="%,d"),
            },
            hide_index=True,
        )

# ---------------------------------------------------------------------------
# 검증 결과 컨테이너 (테두리)
# ---------------------------------------------------------------------------
with st.container(border=True):

    # 헤더
    if _use_split and _val_p:
        _opt_str = f"{_opt_p[0]} ~ {_opt_p[1]}" if _opt_p else ""
        st.markdown(
            '<div style="border-left:4px solid #00c853;padding:10px 18px;margin-bottom:20px;background-color:rgba(0,200,83,0.07);border-radius:0 8px 8px 0;">'
            '<div style="font-size:1.25rem;font-weight:700;color:#00c853;margin-bottom:4px;">✅ 검증 결과 (Out-of-Sample)</div>'
            f'<div style="font-size:0.82rem;color:#888;line-height:1.5;">🔧 최적화 기간 <strong style="color:#aaa">{_opt_str}</strong> 에서 찾은 최적 파라미터를 &nbsp;→&nbsp; 📅 검증 기간 <strong style="color:#aaa">{_val_p[0]} ~ {_val_p[1]}</strong> 에 적용한 실제 성과입니다.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.subheader("📊 백테스트 결과")

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
        st.plotly_chart(pv.fig_equity_curve(), width="stretch", theme=None)

    with tab2:
        st.plotly_chart(pv.fig_drawdown(), width="stretch", theme=None)

    with tab3:
        st.plotly_chart(pv.fig_monthly_returns(), width="stretch", theme=None)

    with tab4:
        fig = pv.fig_return_distribution()
        if fig:
            st.plotly_chart(fig, width="stretch", theme=None)
        else:
            st.info("거래 데이터가 없습니다.")

    with tab5:
        fig = pv.fig_pattern_performance()
        if fig:
            st.plotly_chart(fig, width="stretch", theme=None)
        else:
            st.info("거래 데이터가 없습니다.")

    with tab6:
        trade_df = pd.DataFrame([t.to_dict() for t in trades])
        # score = final_score (패턴점수 + 시그널수×5), pattern_score 역산
        if 'score' in trade_df.columns and 'signal_count' in trade_df.columns:
            trade_df['pattern_score'] = trade_df['score'] - trade_df['signal_count'] * 5
            trade_df = trade_df.rename(columns={'score': 'final_score'})

        # 보유기간 중 intraperiod 통계 (max/min 수익률, MDD)
        if not trade_df.empty:
            import numpy as _np
            _conn = get_db_connection()
            _codes = trade_df['stock_code'].unique().tolist()
            _min_d = trade_df['entry_date'].min()
            _max_d = trade_df['exit_date'].max()
            from sqlalchemy import text as _text
            _ph = ','.join([f':c{i}' for i in range(len(_codes))])
            _params = {f'c{i}': c for i, c in enumerate(_codes)}
            _params['min_d'] = _min_d
            _params['max_d'] = _max_d
            _price_df = pd.read_sql(
                _text(
                    f"SELECT stock_code, time AS trade_date, close_price FROM ohlcv_daily "
                    f"WHERE stock_code IN ({_ph}) AND time BETWEEN :min_d AND :max_d "
                    f"ORDER BY stock_code, time"
                ),
                _conn, params=_params,
            )
            if not _price_df.empty and 'trade_date' in _price_df.columns:
                _price_df['trade_date'] = _price_df['trade_date'].astype(str)
            _stats = []
            for _, _tr in trade_df.iterrows():
                _p = _price_df[
                    (_price_df['stock_code'] == _tr['stock_code']) &
                    (_price_df['trade_date'] >= _tr['entry_date']) &
                    (_price_df['trade_date'] <= _tr['exit_date'])
                ]['close_price'].dropna().values
                if len(_p) < 2 or _tr['entry_price'] <= 0:
                    _stats.append({'max_return_pct': float('nan'), 'min_return_pct': float('nan'), 'mdd_pct': float('nan')})
                    continue
                _rets = (_p / _tr['entry_price'] - 1) * 100
                _peak = _np.maximum.accumulate(_p)
                _mdd = float(((_p - _peak) / _peak * 100).min())
                _stats.append({'max_return_pct': float(_rets.max()), 'min_return_pct': float(_rets.min()), 'mdd_pct': _mdd})
            _stats_df = pd.DataFrame(_stats)
            trade_df = pd.concat([trade_df.reset_index(drop=True), _stats_df], axis=1)

        display_cols = [
            'stock_name', 'stock_code', 'pattern', 'direction',
            'entry_date', 'entry_price', 'exit_date', 'exit_price',
            'return_pct', 'max_return_pct', 'min_return_pct', 'mdd_pct',
            'hold_days', 'exit_reason',
            'pattern_score', 'signal_count', 'final_score',
        ]
        display_cols = [c for c in display_cols if c in trade_df.columns]
        st.dataframe(
            trade_df[display_cols],
            use_container_width=True,
            height=min(600, len(trade_df) * 40 + 40),
            column_config={
                "return_pct":     st.column_config.NumberColumn("수익률 (%)",     format="%.2f"),
                "max_return_pct": st.column_config.NumberColumn("max_ret (%)", format="%.2f"),
                "min_return_pct": st.column_config.NumberColumn("min_ret (%)", format="%.2f"),
                "mdd_pct":        st.column_config.NumberColumn("MDD (%)",        format="%.2f"),
                "entry_price":    st.column_config.NumberColumn("진입가",          format="%,.0f"),
                "exit_price":     st.column_config.NumberColumn("청산가",          format="%,.0f"),
                "pattern_score":  st.column_config.NumberColumn("패턴 점수",       format="%.0f"),
                "signal_count":   st.column_config.NumberColumn("시그널 수",       format="%d"),
                "final_score":    st.column_config.NumberColumn("최종 점수",       format="%.0f"),
            },
        )

        # 다운로드 버튼 (컬럼명 한글 변환)
        csv_df = trade_df[display_cols].rename(columns={
            'pattern_score': '패턴점수', 'signal_count': '시그널수', 'final_score': '최종점수',
            'max_return_pct': '최대수익률', 'min_return_pct': '최소수익률', 'mdd_pct': 'MDD',
        })
        csv = csv_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            "거래 내역 CSV 다운로드",
            csv,
            file_name="backtest_trades.csv",
            mime="text/csv",
        )

# ---------------------------------------------------------------------------
# 결과 저장 (히스토리)
# ---------------------------------------------------------------------------
st.divider()
with st.expander("💾 이 결과를 히스토리에 저장", expanded=False):
    _hist_label = st.text_input(
        "레이블 (선택)",
        placeholder="예: 현재 스코어링 롱 전략, 레거시 비교",
        key="hist_label_input",
    )
    _hist_note = st.text_area(
        "메모 (선택)", height=80,
        placeholder="파라미터 조정 내용, 특이사항 등",
        key="hist_note_input",
    )
    if st.button("📥 히스토리 저장", use_container_width=False):
        _hist_sd = str(st.session_state.get('bt_val_period', [None, None])[0] or '')
        _hist_ed = str(st.session_state.get('bt_val_period', [None, None])[1] or '')
        _row_id = save_backtest_history(
            result=result,
            start_date=_hist_sd,
            end_date=_hist_ed,
            note=_hist_note,
            label=_hist_label,
        )
        st.success(f"히스토리에 저장되었습니다. (ID: {_row_id})")
        st.rerun()

# ---------------------------------------------------------------------------
# 백테스트 히스토리 조회
# ---------------------------------------------------------------------------
st.divider()
with st.expander("📋 백테스트 히스토리", expanded=False):
    hist_df = get_backtest_history(limit=50)
    if hist_df.empty:
        st.info("저장된 히스토리가 없습니다. 위 '이 결과를 히스토리에 저장'으로 저장하세요.")
    else:
        _hist_cols = [
            'id', 'run_at', 'label', 'strategy',
            'start_date', 'end_date',
            'total_return', 'mdd', 'sharpe', 'calmar',
            'win_rate', 'total_trades',
        ]
        _hist_cols = [c for c in _hist_cols if c in hist_df.columns]
        _hist_cfg = {
            'id':           st.column_config.NumberColumn('ID', format='%d'),
            'run_at':       st.column_config.TextColumn('실행시각'),
            'label':        st.column_config.TextColumn('레이블'),
            'strategy':     st.column_config.TextColumn('전략'),
            'start_date':   st.column_config.TextColumn('시작일'),
            'end_date':     st.column_config.TextColumn('종료일'),
            'total_return': st.column_config.NumberColumn('총수익률(%)', format='%.2f'),
            'mdd':          st.column_config.NumberColumn('MDD(%)', format='%.2f'),
            'sharpe':       st.column_config.NumberColumn('샤프', format='%.2f'),
            'calmar':       st.column_config.NumberColumn('칼마', format='%.2f'),
            'win_rate':     st.column_config.NumberColumn('승률(%)', format='%.1f'),
            'total_trades': st.column_config.NumberColumn('거래수', format='%d'),
        }
        _hist_cfg = {k: v for k, v in _hist_cfg.items() if k in _hist_cols}

        st.dataframe(
            hist_df[_hist_cols].reset_index(drop=True),
            column_config=_hist_cfg,
            use_container_width=True,
            hide_index=True,
            height=min(500, len(hist_df) * 40 + 40),
        )

        # 삭제 UI
        _del_ids = st.multiselect(
            "삭제할 항목 ID 선택",
            options=hist_df['id'].tolist(),
            format_func=lambda x: f"ID {x} — {hist_df[hist_df['id']==x]['label'].values[0] or hist_df[hist_df['id']==x]['run_at'].values[0]}",
            key="hist_del_sel",
        )
        if st.button("🗑️ 선택 항목 삭제", disabled=not _del_ids):
            for _did in _del_ids:
                delete_backtest_history(_did)
            st.toast(f"{len(_del_ids)}개 삭제 완료", icon="🗑️")
            st.rerun()
