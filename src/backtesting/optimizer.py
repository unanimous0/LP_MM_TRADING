"""
파라미터 최적화 모듈 (Week 4)

ParameterOptimizer 클래스:
- Grid Search로 최적 파라미터 탐색
- multiprocessing 병렬 처리 지원
- institution_weight 포함 모든 BacktestConfig 파라미터 최적화
"""

import sqlite3
import itertools
import pandas as pd
from typing import Optional, List, Dict
from multiprocessing import Pool

from .engine import BacktestConfig, BacktestEngine
from .metrics import PerformanceMetrics


# ============================================================================
# 모듈 레벨 worker 함수 (multiprocessing pickle 호환)
# ============================================================================

def _run_backtest_worker(args: tuple) -> Optional[dict]:
    """
    단일 파라미터 조합 백테스트 실행 (worker 함수)

    multiprocessing.Pool에서 호출하므로 모듈 레벨에 정의 (pickle 가능)

    Args:
        args: (db_path, params, start_date, end_date) 튜플

    Returns:
        {'params': dict, ...성과 메트릭} 또는 None (실패 시)
    """
    db_path, params, start_date, end_date = args
    conn = sqlite3.connect(db_path)
    try:
        config = BacktestConfig(**params)
        engine = BacktestEngine(conn, config)
        result = engine.run(start_date, end_date, verbose=False)
        metrics = PerformanceMetrics(
            result['trades'],
            result['daily_values'],
            config.initial_capital
        )
        summary = metrics.summary()
        return {'params': params, **summary}
    except Exception:
        return None
    finally:
        conn.close()


# ============================================================================
# ParameterOptimizer 클래스
# ============================================================================

class ParameterOptimizer:
    """
    Grid Search 기반 파라미터 최적화 클래스

    탐색 대상 파라미터:
    - min_score: 최소 패턴 점수 (진입 조건)
    - min_signals: 최소 시그널 개수 (진입 조건)
    - target_return: 목표 수익률 (청산 조건)
    - stop_loss: 손절 비율 (청산 조건)
    - institution_weight: 기관 가중치 (normalizer 파라미터)
    """

    DEFAULT_PARAM_GRID = {
        'min_score': [60, 70, 80],
        'min_signals': [1, 2],
        'target_return': [0.10, 0.15, 0.20],
        'stop_loss': [-0.05, -0.075, -0.10],
        'institution_weight': [0.0, 0.1, 0.2, 0.3, 0.5],
    }

    def __init__(self, db_path: str, start_date: str, end_date: str,
                 base_config: Optional[BacktestConfig] = None):
        """
        초기화

        Args:
            db_path: SQLite DB 파일 경로 (worker에서 사용)
            start_date: 백테스트 시작일 (YYYY-MM-DD)
            end_date: 백테스트 종료일 (YYYY-MM-DD)
            base_config: 기본 BacktestConfig (None이면 기본값 사용)
                param_grid에 없는 파라미터는 base_config 값 사용
        """
        self.db_path = db_path
        self.start_date = start_date
        self.end_date = end_date
        self.base_config = base_config or BacktestConfig()

    def grid_search(self,
                    param_grid: Optional[Dict] = None,
                    metric: str = 'sharpe_ratio',
                    top_n: int = 10,
                    workers: int = 1,
                    verbose: bool = True) -> pd.DataFrame:
        """
        Grid Search 실행

        모든 파라미터 조합을 백테스트하고 metric 기준 top_n 결과 반환.

        Args:
            param_grid: 탐색할 파라미터 그리드
                예: {'min_score': [60, 70], 'institution_weight': [0.0, 0.3]}
                None이면 DEFAULT_PARAM_GRID 사용
            metric: 최적화 평가 지표
                'sharpe_ratio', 'total_return', 'win_rate', 'profit_factor'
            top_n: 상위 N개 결과 반환
            workers: 병렬 처리 worker 수 (1이면 순차 실행)
            verbose: 진행 상황 출력 여부

        Returns:
            pd.DataFrame: top_n 결과 (metric 기준 내림차순 정렬)
                - 파라미터 열: 탐색한 파라미터
                - 성과 열: total_return, sharpe_ratio, win_rate,
                           max_drawdown, profit_factor, total_trades
        """
        if param_grid is None:
            param_grid = self.DEFAULT_PARAM_GRID

        # 파라미터 조합 생성
        combinations = self._build_param_combinations(param_grid)

        if verbose:
            print(f"\n{'='*60}")
            print(f"🔍 Grid Search 시작")
            print(f"{'='*60}")
            print(f"기간: {self.start_date} ~ {self.end_date}")
            print(f"탐색 파라미터: {list(param_grid.keys())}")
            print(f"조합 수: {len(combinations)}개")
            print(f"평가 지표: {metric}")
            print(f"Workers: {workers}")
            print()

        # 백테스트 실행
        args_list = [
            (self.db_path, params, self.start_date, self.end_date)
            for params in combinations
        ]

        if workers > 1:
            with Pool(processes=workers) as pool:
                raw_results = pool.map(_run_backtest_worker, args_list)
        else:
            raw_results = []
            for i, args in enumerate(args_list):
                result = _run_backtest_worker(args)
                raw_results.append(result)
                if verbose and (i + 1) % 10 == 0:
                    print(f"  진행: {i+1}/{len(args_list)} 완료...")

        # 결과 정리
        valid_results = [r for r in raw_results if r is not None]

        if not valid_results:
            if verbose:
                print("[WARN] 유효한 결과가 없습니다.")
            return pd.DataFrame()

        # DataFrame 변환
        rows = []
        for r in valid_results:
            row = {}
            # 파라미터 열
            for k, v in r['params'].items():
                row[k] = v
            # 성과 열
            row['total_return'] = r.get('total_return', 0.0)
            row['sharpe_ratio'] = r.get('sharpe_ratio', 0.0)
            row['win_rate'] = r.get('win_rate', 0.0)
            row['max_drawdown'] = r.get('max_drawdown', 0.0)
            row['profit_factor'] = r.get('profit_factor', 0.0)
            row['total_trades'] = r.get('total_trades', 0)
            rows.append(row)

        df = pd.DataFrame(rows)

        # metric 기준 내림차순 정렬 (max_drawdown은 작을수록 좋으므로 예외 처리)
        if metric == 'max_drawdown':
            df = df.sort_values(metric, ascending=True)
        else:
            df = df.sort_values(metric, ascending=False)

        result_df = df.head(top_n).reset_index(drop=True)

        if verbose:
            print(f"\n✅ Grid Search 완료!")
            print(f"총 {len(valid_results)}개 조합 실행 완료")
            self.print_results(result_df, top_n=min(top_n, 5))

        return result_df

    def _build_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """
        base_config를 기반으로 파라미터 조합 생성

        param_grid에 있는 파라미터만 변경하고,
        나머지는 base_config 값을 유지.

        Args:
            param_grid: 탐색 파라미터 그리드

        Returns:
            파라미터 딕셔너리 리스트
        """
        # base_config에서 기본값 추출
        base_params = {
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

        # param_grid 키와 값 목록 추출
        keys = list(param_grid.keys())
        value_lists = [param_grid[k] for k in keys]

        # 모든 조합 생성
        combinations = []
        for values in itertools.product(*value_lists):
            params = base_params.copy()
            for k, v in zip(keys, values):
                params[k] = v
            combinations.append(params)

        return combinations

    def print_results(self, results_df: pd.DataFrame, top_n: int = 10):
        """
        최적화 결과 테이블 출력

        Args:
            results_df: grid_search() 반환 DataFrame
            top_n: 출력할 상위 N개
        """
        if results_df.empty:
            print("[WARN] 출력할 결과가 없습니다.")
            return

        df = results_df.head(top_n)

        print(f"\n{'='*60}")
        print(f"📊 최적화 결과 (상위 {len(df)}개)")
        print(f"{'='*60}")

        # 파라미터 열과 성과 열 분리
        perf_cols = ['total_return', 'sharpe_ratio', 'win_rate',
                     'max_drawdown', 'profit_factor', 'total_trades']
        param_cols = [c for c in df.columns if c not in perf_cols]

        for i, row in df.iterrows():
            print(f"\n[{i+1}위]")
            # 파라미터
            param_str = " | ".join([
                f"{c}={row[c]}" for c in param_cols if c in row
            ])
            print(f"  파라미터: {param_str}")
            # 성과
            print(f"  수익률: {row['total_return']:+.2f}% | "
                  f"샤프: {row['sharpe_ratio']:.2f} | "
                  f"승률: {row['win_rate']:.1f}% | "
                  f"MDD: {row['max_drawdown']:.2f}% | "
                  f"PF: {row['profit_factor']:.2f} | "
                  f"거래: {row['total_trades']:.0f}건")

        print()
