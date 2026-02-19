"""
Walk-Forward Analysis 모듈 (Week 5)

WalkForwardAnalyzer 클래스:
- 학습/검증 기간 롤링으로 전략 과적합 방지 및 견고성 검증
- 각 검증 기간마다 학습 기간 최적 파라미터로 백테스트
- 전체 기간 통합 성과 분석
"""

import sqlite3
import calendar
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime, timedelta

from .engine import BacktestConfig, BacktestEngine
from .optimizer import ParameterOptimizer
from .metrics import PerformanceMetrics


def _add_months(dt: datetime, months: int) -> datetime:
    """
    개월 수 더하기 (stdlib만 사용, python-dateutil 불필요)

    Args:
        dt: 기준 날짜
        months: 더할 개월 수

    Returns:
        months개월 후 날짜 (월말 초과 시 말일로 조정)
    """
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


class WalkForwardConfig:
    """Walk-Forward Analysis 설정"""

    def __init__(self,
                 train_months: int = 6,
                 val_months: int = 1,
                 step_months: int = 1,
                 metric: str = 'sharpe_ratio',
                 top_n: int = 1,
                 workers: int = 1):
        """
        Args:
            train_months: 학습 기간 (개월)
            val_months: 검증 기간 (개월)
            step_months: 롤링 스텝 (개월)
            metric: 최적화 기준 지표 (sharpe_ratio/total_return/win_rate/profit_factor)
            top_n: 최적 파라미터 후보 수 (1이면 최적 1개만)
            workers: 병렬 처리 worker 수
        """
        self.train_months = train_months
        self.val_months = val_months
        self.step_months = step_months
        self.metric = metric
        self.top_n = top_n
        self.workers = workers


class WalkForwardAnalyzer:
    """Walk-Forward Analysis 클래스"""

    def __init__(self,
                 db_path: str,
                 start_date: str,
                 end_date: str,
                 wf_config: Optional[WalkForwardConfig] = None,
                 base_config: Optional[BacktestConfig] = None,
                 param_grid: Optional[dict] = None):
        """
        Args:
            db_path: SQLite DB 경로
            start_date: 전체 분석 시작일 (YYYY-MM-DD)
            end_date: 전체 분석 종료일 (YYYY-MM-DD)
            wf_config: Walk-Forward 설정 (None이면 기본값)
            base_config: 백테스트 기본 설정 (최적화 대상 외 파라미터)
            param_grid: 탐색 파라미터 그리드 (None이면 DEFAULT_PARAM_GRID)
        """
        self.db_path = db_path
        self.start_date = start_date
        self.end_date = end_date
        self.wf_config = wf_config or WalkForwardConfig()
        self.base_config = base_config or BacktestConfig()
        self.param_grid = param_grid  # None이면 grid_search()에서 DEFAULT_PARAM_GRID 사용

        self._results: List[Dict] = []
        self._combined_trades = []

    def split_periods(self) -> List[Dict]:
        """
        학습/검증 기간 분할 (롤링 윈도우)

        Returns:
            [
                {
                    'train_start': '2024-01-01',
                    'train_end': '2024-06-30',
                    'val_start': '2024-07-01',
                    'val_end': '2024-07-31',
                },
                ...
            ]
        """
        periods = []
        current = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')

        while True:
            train_start = current
            train_end = _add_months(current, self.wf_config.train_months) - timedelta(days=1)
            val_start = train_end + timedelta(days=1)
            val_end = _add_months(val_start, self.wf_config.val_months) - timedelta(days=1)

            if val_end > end:
                break

            periods.append({
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'val_start': val_start.strftime('%Y-%m-%d'),
                'val_end': val_end.strftime('%Y-%m-%d'),
            })
            current = _add_months(current, self.wf_config.step_months)

        return periods

    def _extract_best_params(self, row: pd.Series) -> Dict:
        """
        최적화 결과 행에서 BacktestConfig 파라미터 추출

        base_config의 값으로 시작 후 grid_search 결과로 덮어쓰기

        Args:
            row: grid_search() 반환 DataFrame의 단일 행

        Returns:
            BacktestConfig(**params) 호출 가능한 파라미터 딕셔너리
        """
        perf_cols = {'total_return', 'sharpe_ratio', 'win_rate',
                     'max_drawdown', 'profit_factor', 'total_trades'}

        # base_config의 모든 파라미터로 초기화
        params = {
            'initial_capital': self.base_config.initial_capital,
            'max_positions': self.base_config.max_positions,
            'min_score': self.base_config.min_score,
            'min_signals': self.base_config.min_signals,
            'target_return': self.base_config.target_return,
            'stop_loss': self.base_config.stop_loss,
            'max_hold_days': self.base_config.max_hold_days,
            'reverse_signal_threshold': self.base_config.reverse_signal_threshold,
            'strategy': self.base_config.strategy,
            'institution_weight': self.base_config.institution_weight,
            'force_exit_on_end': self.base_config.force_exit_on_end,
        }

        # grid_search 결과로 덮어쓰기 (성과 열 제외)
        for col in row.index:
            if col not in perf_cols and col in params:
                params[col] = row[col]

        return params

    def run(self, verbose: bool = True) -> Dict:
        """
        Walk-Forward 전체 실행

        각 검증 기간마다:
        1. 학습 기간에서 ParameterOptimizer.grid_search() → 최적 파라미터 추출
        2. 최적 파라미터로 검증 기간 BacktestEngine.run() 실행
        3. 결과 저장

        Args:
            verbose: 진행 상황 출력 여부

        Returns:
            {
                'periods': List[dict],          # 기간별 결과 (파라미터 + 메트릭)
                'combined_trades': List[Trade], # 전체 기간 통합 거래
                'combined_daily_values': pd.DataFrame,  # 전체 기간 일별 가치
            }
        """
        periods = self.split_periods()

        if not periods:
            if verbose:
                print("[WARN] 유효한 학습/검증 기간이 없습니다. "
                      "전체 기간이 train + val 기간보다 짧습니다.")
            self._results = []
            self._combined_trades = []
            return {
                'periods': [],
                'combined_trades': [],
                'combined_daily_values': pd.DataFrame(),
            }

        all_results = []
        combined_trades = []
        combined_daily_values = []

        if verbose:
            print(f"\n{'='*80}")
            print(f"🔄 Walk-Forward Analysis 시작")
            print(f"{'='*80}")
            print(f"전체 기간: {self.start_date} ~ {self.end_date}")
            print(f"학습: {self.wf_config.train_months}개월 | "
                  f"검증: {self.wf_config.val_months}개월 | "
                  f"스텝: {self.wf_config.step_months}개월")
            print(f"총 {len(periods)}개 기간\n")

        for i, period in enumerate(periods):
            if verbose:
                print(f"\n[{i+1}/{len(periods)}] "
                      f"학습: {period['train_start']}~{period['train_end']} "
                      f"→ 검증: {period['val_start']}~{period['val_end']}")

            # 1. 학습 기간: Grid Search로 최적 파라미터 탐색
            optimizer = ParameterOptimizer(
                db_path=self.db_path,
                start_date=period['train_start'],
                end_date=period['train_end'],
                base_config=self.base_config,
            )
            opt_results = optimizer.grid_search(
                param_grid=self.param_grid,
                metric=self.wf_config.metric,
                top_n=1,
                workers=self.wf_config.workers,
                verbose=False,
            )

            # 최적 파라미터 추출
            if opt_results.empty:
                if verbose:
                    print(f"  [SKIP] 학습 기간 최적화 결과 없음")
                continue

            best_params = self._extract_best_params(opt_results.iloc[0])

            if verbose:
                grid = self.param_grid or ParameterOptimizer.DEFAULT_PARAM_GRID
                param_str = " | ".join([
                    f"{k}={best_params[k]}" for k in grid.keys() if k in best_params
                ])
                print(f"  최적 파라미터: {param_str}")

            # 2. 검증 기간: 최적 파라미터로 백테스트
            conn = sqlite3.connect(self.db_path)
            try:
                val_config = BacktestConfig(**best_params)
                engine = BacktestEngine(conn, val_config)
                val_result = engine.run(
                    period['val_start'], period['val_end'],
                    verbose=False, preload_data=True
                )
            finally:
                conn.close()

            # 3. 검증 기간 성과 계산
            val_metrics = PerformanceMetrics(
                val_result['trades'],
                val_result['daily_values'],
                val_config.initial_capital
            ).summary()

            period_result = {
                **period,
                'best_params': best_params,
                **val_metrics,
            }
            all_results.append(period_result)
            combined_trades.extend(val_result['trades'])
            if not val_result['daily_values'].empty:
                combined_daily_values.append(val_result['daily_values'])

            if verbose:
                print(f"  검증 결과: "
                      f"수익률 {val_metrics.get('total_return', 0):+.2f}% | "
                      f"승률 {val_metrics.get('win_rate', 0):.1f}% | "
                      f"거래 {val_metrics.get('total_trades', 0):.0f}건")

        self._results = all_results
        self._combined_trades = combined_trades
        combined_df = (pd.concat(combined_daily_values, ignore_index=True)
                       if combined_daily_values else pd.DataFrame())

        if verbose:
            print(f"\n{'='*80}")
            print(f"✅ Walk-Forward Analysis 완료!")
            print(f"총 {len(all_results)}/{len(periods)} 기간 성공")
            print(f"통합 거래: {len(combined_trades)}건")
            print(f"{'='*80}\n")

        return {
            'periods': all_results,
            'combined_trades': combined_trades,
            'combined_daily_values': combined_df,
        }

    def summary(self) -> pd.DataFrame:
        """
        기간별 결과 요약 DataFrame 반환

        Columns: train_start, train_end, val_start, val_end,
                 param_* (최적 파라미터), 성과 메트릭

        Returns:
            pd.DataFrame: 각 검증 기간별 결과
        """
        if not self._results:
            return pd.DataFrame()

        rows = []
        for r in self._results:
            row = {
                'train_start': r.get('train_start', ''),
                'train_end': r.get('train_end', ''),
                'val_start': r.get('val_start', ''),
                'val_end': r.get('val_end', ''),
            }
            # best_params 펼치기 (param_ 접두사)
            if 'best_params' in r:
                for k, v in r['best_params'].items():
                    row[f'param_{k}'] = v
            # 성과 메트릭
            for k in ['total_return', 'sharpe_ratio', 'win_rate',
                      'max_drawdown', 'profit_factor', 'total_trades']:
                row[k] = r.get(k, None)
            rows.append(row)

        return pd.DataFrame(rows)

    def print_results(self):
        """Walk-Forward 결과 테이블 출력"""
        df = self.summary()

        if df.empty:
            print("[WARN] Walk-Forward 결과가 없습니다.")
            return

        print(f"\n{'='*80}")
        print(f"📊 Walk-Forward Analysis 결과")
        print(f"{'='*80}")

        for _, row in df.iterrows():
            print(f"\n검증 기간: {row['val_start']} ~ {row['val_end']}")
            print(f"  수익률: {row['total_return']:+.2f}% | "
                  f"샤프: {row['sharpe_ratio']:.2f} | "
                  f"승률: {row['win_rate']:.1f}% | "
                  f"MDD: {row['max_drawdown']:.2f}% | "
                  f"거래: {row['total_trades']:.0f}건")

        # 통합 통계
        print(f"\n[통합 성과]")
        print(f"평균 수익률: {df['total_return'].mean():+.2f}%")
        print(f"평균 샤프: {df['sharpe_ratio'].mean():.2f}")
        print(f"평균 승률: {df['win_rate'].mean():.1f}%")
        print(f"평균 MDD: {df['max_drawdown'].mean():.2f}%")
        print(f"양(+) 기간: {(df['total_return'] > 0).sum()}/{len(df)}")
        print()
