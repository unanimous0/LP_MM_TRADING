"""
워크포워드 페이지 - Walk-Forward 결과 조회

Walk-Forward는 실행 시간이 길어 CSV 업로드/자동 로드 방식.
KPI 행, 기간별 수익률 바차트, 파라미터 변화 추이 표시.
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from utils.charts import create_wf_period_returns_chart

st.set_page_config(page_title="워크포워드", page_icon="🔄", layout="wide")
st.title("Walk-Forward 분석 결과")

# ---------------------------------------------------------------------------
# CLI 가이드
# ---------------------------------------------------------------------------
with st.expander("Walk-Forward 실행 방법 (CLI)"):
    st.code("""
# 기본 실행 (6개월 학습, 1개월 검증, Optuna 50 trials)
python scripts/analysis/backtest_runner.py --walk-forward \\
  --start 2024-01-01 --end 2024-12-31 \\
  --wf-save-csv output/walk_forward.csv

# 100 trials, 4 workers 병렬
python scripts/analysis/backtest_runner.py --walk-forward \\
  --n-trials 100 --workers 4 \\
  --wf-save-csv output/walk_forward.csv
    """, language="bash")

# ---------------------------------------------------------------------------
# 데이터 로드 (업로드 또는 자동 감지)
# ---------------------------------------------------------------------------
wf_df = None

# 자동 감지 경로
auto_path = _PROJECT_ROOT / "output" / "walk_forward_results.csv"

uploaded_file = st.file_uploader("Walk-Forward 결과 CSV 업로드", type=['csv'])

if uploaded_file is not None:
    wf_df = pd.read_csv(uploaded_file)
    st.success(f"업로드된 파일: {uploaded_file.name} ({len(wf_df)}개 기간)")
elif auto_path.exists():
    wf_df = pd.read_csv(auto_path)
    st.info(f"자동 로드: {auto_path.name} ({len(wf_df)}개 기간)")
else:
    # output 디렉토리에서 walk_forward*.csv 자동 검색
    output_dir = _PROJECT_ROOT / "output"
    if output_dir.exists():
        wf_files = sorted(output_dir.glob("walk_forward*.csv"), reverse=True)
        if wf_files:
            wf_df = pd.read_csv(wf_files[0])
            st.info(f"자동 로드: {wf_files[0].name} ({len(wf_df)}개 기간)")

if wf_df is None or wf_df.empty:
    st.warning(
        "Walk-Forward 결과 파일이 없습니다.\n\n"
        "CLI에서 `--wf-save-csv` 옵션으로 결과를 저장한 후 업로드하거나, "
        "`output/` 디렉토리에 저장하세요."
    )
    st.stop()

# ---------------------------------------------------------------------------
# 결과 전체 테이블
# ---------------------------------------------------------------------------
st.subheader("기간별 결과")
st.dataframe(wf_df, use_container_width=True)

# ---------------------------------------------------------------------------
# KPI 행
# ---------------------------------------------------------------------------
return_col = None
for col in ['val_return', 'total_return', 'return']:
    if col in wf_df.columns:
        return_col = col
        break

if return_col:
    returns = wf_df[return_col]

    sharpe_col = None
    for col in ['val_sharpe', 'sharpe_ratio', 'sharpe']:
        if col in wf_df.columns:
            sharpe_col = col
            break

    win_col = None
    for col in ['val_win_rate', 'win_rate']:
        if col in wf_df.columns:
            win_col = col
            break

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("평균 수익률", f"{returns.mean():.2f}%")

    if sharpe_col:
        kpi2.metric("평균 샤프", f"{wf_df[sharpe_col].mean():.2f}")
    else:
        kpi2.metric("중앙값 수익률", f"{returns.median():.2f}%")

    if win_col:
        kpi3.metric("평균 승률", f"{wf_df[win_col].mean():.1f}%")
    else:
        kpi3.metric("수익 기간 비율", f"{(returns > 0).mean() * 100:.1f}%")

    kpi4.metric("양(+) 기간", f"{(returns > 0).sum()}/{len(returns)}")

    # ---------------------------------------------------------------------------
    # 기간별 수익률 바차트
    # ---------------------------------------------------------------------------
    st.subheader("기간별 검증 수익률")
    fig = create_wf_period_returns_chart(wf_df)
    st.plotly_chart(fig, width="stretch", theme=None)

# ---------------------------------------------------------------------------
# 파라미터 변화 추이
# ---------------------------------------------------------------------------
param_cols = [c for c in wf_df.columns if c.startswith(('best_', 'opt_'))]
if not param_cols:
    # 일반 파라미터 컬럼 감지
    known_params = ['min_score', 'min_signals', 'target_return', 'stop_loss']
    param_cols = [c for c in wf_df.columns if c in known_params]

if param_cols:
    st.subheader("최적 파라미터 변화 추이")

    period_labels = None
    for col in ['period', 'val_start', 'start']:
        if col in wf_df.columns:
            period_labels = wf_df[col].astype(str)
            break

    if period_labels is None:
        period_labels = [f"기간 {i+1}" for i in range(len(wf_df))]

    for pcol in param_cols:
        if wf_df[pcol].nunique() > 1:
            st.line_chart(
                pd.DataFrame({'기간': period_labels, pcol: wf_df[pcol]}).set_index('기간'),
                height=250,
            )
