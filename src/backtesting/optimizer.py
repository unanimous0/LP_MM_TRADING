"""
파라미터 최적화 모듈 (Week 4 + Week 5)

ParameterOptimizer 클래스:
- Grid Search로 최적 파라미터 탐색 (--optimize)
- multiprocessing 병렬 처리 지원

OptunaOptimizer 클래스:
- Bayesian Optimization (Walk-Forward Analysis용)
- MedianPruner: 나쁜 Trial 조기 중단
- 2단계 탐색: Phase 1 (넓은 범위) → Phase 2 (좋은 구간 집중)
"""

import sqlite3
import itertools
import pandas as pd
from datetime import datetime
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


# ============================================================================
# OptunaOptimizer 클래스 (Walk-Forward용 Bayesian Optimization)
# ============================================================================

class OptunaOptimizer:
    """
    Optuna Bayesian Optimization 기반 파라미터 최적화 클래스

    Walk-Forward Analysis에서 Grid Search 대신 사용.
    - MedianPruner: 절반 기간 중간 평가로 나쁜 Trial 조기 중단
    - 2단계 탐색: Phase 1 (넓은 범위) → Phase 2 (좋은 구간 집중)

    파라미터 공간 형식:
        {
            'min_score':     {'type': 'float', 'low': 50.0, 'high': 90.0},
            'min_signals':   {'type': 'int',   'low': 1,    'high': 3},
        }
    """

    DEFAULT_PARAM_SPACE = {
        'min_score':          {'type': 'float', 'low': 50.0,  'high': 90.0},
        'min_signals':        {'type': 'int',   'low': 1,     'high': 3},
        'target_return':      {'type': 'float', 'low': 0.05,  'high': 0.25},
        'stop_loss':          {'type': 'float', 'low': -0.15, 'high': -0.03},
        'institution_weight': {'type': 'float', 'low': 0.0,   'high': 0.5},
    }

    def __init__(self, db_path: str, start_date: str, end_date: str,
                 base_config: Optional[BacktestConfig] = None):
        """
        Args:
            db_path: SQLite DB 파일 경로
            start_date: 백테스트 시작일 (YYYY-MM-DD)
            end_date: 백테스트 종료일 (YYYY-MM-DD)
            base_config: 기본 BacktestConfig (최적화 대상 외 파라미터)
        """
        self.db_path = db_path
        self.start_date = start_date
        self.end_date = end_date
        self.base_config = base_config or BacktestConfig()

    def _build_base_params(self) -> dict:
        """base_config에서 기본 파라미터 딕셔너리 생성"""
        c = self.base_config
        return {
            'initial_capital': c.initial_capital,
            'max_positions': c.max_positions,
            'min_score': c.min_score,
            'min_signals': c.min_signals,
            'target_return': c.target_return,
            'stop_loss': c.stop_loss,
            'max_hold_days': c.max_hold_days,
            'reverse_signal_threshold': c.reverse_signal_threshold,
            'strategy': c.strategy,
            'institution_weight': c.institution_weight,
            'force_exit_on_end': c.force_exit_on_end,
        }

    def _build_objective(self, param_space: dict, metric: str):
        """
        Optuna objective function 생성 (closure)

        MedianPruner 지원:
        - Step 0: 학습 기간 절반 평가 → trial.report() → prune 판단
        - 통과 시: 전체 기간 평가 → 최종 값 반환
        """
        import optuna as _optuna
        db_path = self.db_path
        start_date = self.start_date
        end_date = self.end_date
        build_base = self._build_base_params
        _TrialPruned = _optuna.exceptions.TrialPruned

        def objective(trial):
            # 파라미터 샘플링
            params = build_base()
            for name, spec in param_space.items():
                if spec['type'] == 'int':
                    params[name] = trial.suggest_int(
                        name, int(spec['low']), int(spec['high']))
                else:
                    params[name] = trial.suggest_float(
                        name, spec['low'], spec['high'])

            # 중간 평가 날짜 (전체 기간의 절반)
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            mid_dt = start_dt + (end_dt - start_dt) // 2
            mid_date = mid_dt.strftime('%Y-%m-%d')

            conn = sqlite3.connect(db_path)
            try:
                config = BacktestConfig(**params)

                # Step 0: 절반 기간 평가 → Pruning 판단
                if mid_date > start_date:
                    engine_half = BacktestEngine(conn, config)
                    half_result = engine_half.run(start_date, mid_date, verbose=False)
                    half_trades = half_result.get('trades', [])
                    half_daily = half_result.get('daily_values', pd.DataFrame())

                    if half_trades:
                        half_m = PerformanceMetrics(
                            half_trades, half_daily, config.initial_capital
                        ).summary()
                        intermediate = float(half_m.get(metric) or float('-inf'))
                    else:
                        intermediate = float('-inf')

                    trial.report(intermediate, step=0)
                    if trial.should_prune():
                        raise _TrialPruned()

                # 전체 기간 평가
                engine_full = BacktestEngine(conn, config)
                full_result = engine_full.run(start_date, end_date, verbose=False)
                full_trades = full_result.get('trades', [])
                full_daily = full_result.get('daily_values', pd.DataFrame())

                if not full_trades:
                    return float('-inf')

                full_m = PerformanceMetrics(
                    full_trades, full_daily, config.initial_capital
                ).summary()
                return float(full_m.get(metric) or float('-inf'))

            except _TrialPruned:
                raise
            except Exception:
                return float('-inf')
            finally:
                conn.close()

        return objective

    def _narrow_param_space(self, study, param_space: dict,
                            top_pct: float = 0.25,
                            margin: float = 0.25) -> dict:
        """
        Phase 1 결과에서 상위 Trial의 파라미터 범위를 좁혀서 반환

        Args:
            study: Phase 1 완료된 Optuna Study
            param_space: 현재 파라미터 공간
            top_pct: 상위 몇 % Trial 기준 (기본: 상위 25%)
            margin: 최솟값/최댓값 기준 여유 비율 (기본: 25%)

        Returns:
            좁혀진 파라미터 공간 (데이터 부족 시 원본 반환)
        """
        import optuna as _optuna
        complete = [
            t for t in study.trials
            if t.state == _optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if len(complete) < 4:
            return param_space  # 데이터 부족 → 좁히지 않음

        n_top = max(2, int(len(complete) * top_pct))
        top_trials = sorted(complete, key=lambda t: t.value, reverse=True)[:n_top]

        narrowed = {}
        for name, spec in param_space.items():
            values = [t.params[name] for t in top_trials if name in t.params]
            if len(values) < 2:
                narrowed[name] = spec
                continue

            v_min, v_max = min(values), max(values)
            total_range = spec['high'] - spec['low']
            # 탐색 범위의 최소 10% 폭 유지
            expansion = max(total_range * 0.1, (v_max - v_min) * margin)
            new_low = max(spec['low'], v_min - expansion)
            new_high = min(spec['high'], v_max + expansion)

            if spec['type'] == 'int':
                new_lo_i = max(int(spec['low']), int(new_low))
                new_hi_i = min(int(spec['high']), int(new_high) + 1)
                if new_lo_i < new_hi_i:
                    narrowed[name] = {'type': 'int', 'low': new_lo_i, 'high': new_hi_i}
                else:
                    narrowed[name] = spec
            else:
                if new_low < new_high - 1e-8:
                    narrowed[name] = {'type': 'float', 'low': new_low, 'high': new_high}
                else:
                    narrowed[name] = spec

        return narrowed

    def optimize(self, param_space: Optional[dict] = None,
                 n_trials: int = 50,
                 metric: str = 'sharpe_ratio',
                 verbose: bool = True) -> Optional[dict]:
        """
        2단계 Bayesian Optimization 실행

        Phase 1 (넓은 범위 탐색, n_trials//2 trials)
          → 상위 25% Trial로 탐색 범위 좁히기
          → Phase 2 (집중 탐색, 나머지 trials, Phase 1 최고값 seed)

        Args:
            param_space: 탐색 파라미터 공간
                None이면 DEFAULT_PARAM_SPACE 사용
                형식: {'param': {'type': 'float'/'int', 'low': ..., 'high': ...}}
            n_trials: 총 Trial 수 (Phase 1: n//2, Phase 2: 나머지)
            metric: 평가 지표
                'sharpe_ratio', 'total_return', 'win_rate',
                'profit_factor', 'max_drawdown'
            verbose: 진행 상황 출력 여부

        Returns:
            {
                'params': BacktestConfig 파라미터 딕셔너리,
                metric: float (최고 값),
                'total_complete': int,
                'total_pruned': int,
            }
            또는 None (완료 Trial 없음)
        """
        import optuna as _optuna
        from optuna.pruners import MedianPruner
        _optuna.logging.set_verbosity(_optuna.logging.WARNING)

        if param_space is None:
            param_space = self.DEFAULT_PARAM_SPACE

        phase1_n = max(0, n_trials // 2)
        phase2_n = n_trials - phase1_n

        if verbose:
            print(f"\n{'='*60}")
            print(f"🔮 Optuna Bayesian Optimization 시작")
            print(f"{'='*60}")
            print(f"기간: {self.start_date} ~ {self.end_date}")
            print(f"총 Trial: {n_trials} (Phase 1: {phase1_n} | Phase 2: {phase2_n})")
            print(f"평가 지표: {metric}")
            print(f"파라미터: {list(param_space.keys())}")

        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

        # ── Phase 1: 넓은 범위 탐색 ──────────────────────────────────────
        study1 = _optuna.create_study(direction='maximize', pruner=pruner)
        study2 = None

        if phase1_n > 0:
            if verbose:
                print(f"\n[Phase 1] 넓은 범위 탐색 ({phase1_n} trials)...")
            obj1 = self._build_objective(param_space, metric)
            study1.optimize(obj1, n_trials=phase1_n, show_progress_bar=False)

            p1_complete = sum(
                1 for t in study1.trials
                if t.state == _optuna.trial.TrialState.COMPLETE
            )
            p1_pruned = sum(
                1 for t in study1.trials
                if t.state == _optuna.trial.TrialState.PRUNED
            )
            if verbose:
                print(f"  완료: {p1_complete}개 | 중단(Pruned): {p1_pruned}개")
                try:
                    print(f"  Phase 1 최고 {metric}: {study1.best_value:.4f}")
                except ValueError:
                    pass

        # ── Phase 2: 좋은 구간 집중 탐색 ────────────────────────────────
        if phase2_n > 0:
            narrowed_space = self._narrow_param_space(study1, param_space)
            if verbose:
                changed = [k for k in narrowed_space
                           if narrowed_space[k] != param_space.get(k)]
                print(f"\n[Phase 2] 집중 탐색 ({phase2_n} trials)...")
                if changed:
                    print(f"  좁혀진 파라미터: {changed}")

            study2 = _optuna.create_study(direction='maximize', pruner=pruner)

            # Phase 1 최고 파라미터를 seed trial로 추가
            try:
                if study1.best_trial and study1.best_trial.params:
                    study2.enqueue_trial(study1.best_trial.params)
            except (ValueError, AttributeError):
                pass

            obj2 = self._build_objective(narrowed_space, metric)
            study2.optimize(obj2, n_trials=phase2_n, show_progress_bar=False)

            p2_complete = sum(
                1 for t in study2.trials
                if t.state == _optuna.trial.TrialState.COMPLETE
            )
            p2_pruned = sum(
                1 for t in study2.trials
                if t.state == _optuna.trial.TrialState.PRUNED
            )
            if verbose:
                print(f"  완료: {p2_complete}개 | 중단(Pruned): {p2_pruned}개")
                try:
                    print(f"  Phase 2 최고 {metric}: {study2.best_value:.4f}")
                except ValueError:
                    pass

        # ── 전체 결과에서 최고 Trial 선택 ────────────────────────────────
        all_complete = [
            t for t in study1.trials
            if t.state == _optuna.trial.TrialState.COMPLETE
        ]
        if study2:
            all_complete += [
                t for t in study2.trials
                if t.state == _optuna.trial.TrialState.COMPLETE
            ]

        if not all_complete:
            if verbose:
                print("\n[WARN] 완료된 Trial이 없습니다.")
            return None

        best_trial = max(all_complete, key=lambda t: t.value)

        # best_trial 파라미터 → BacktestConfig 파라미터 딕셔너리
        best_params = self._build_base_params()
        for name in param_space:
            if name in best_trial.params:
                best_params[name] = best_trial.params[name]

        all_trials = study1.trials + (study2.trials if study2 else [])
        total_pruned = sum(
            1 for t in all_trials
            if t.state == _optuna.trial.TrialState.PRUNED
        )

        if verbose:
            print(f"\n{'='*60}")
            print(f"✅ Optuna 최적화 완료!")
            print(f"완료 Trial: {len(all_complete)}개 | 중단 Trial: {total_pruned}개")
            print(f"최고 {metric}: {best_trial.value:.4f}")
            param_parts = []
            for k in param_space:
                v = best_params[k]
                param_parts.append(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                )
            print(f"최적 파라미터: {' | '.join(param_parts)}")
            print(f"{'='*60}\n")

        return {
            'params': best_params,
            metric: best_trial.value,
            'total_complete': len(all_complete),
            'total_pruned': total_pruned,
        }

    def print_results(self, result: Optional[dict], metric: str = 'sharpe_ratio'):
        """최적화 결과 출력"""
        if result is None:
            print("[WARN] 출력할 결과가 없습니다.")
            return

        print(f"\n{'='*60}")
        print(f"📊 Optuna 최적화 결과")
        print(f"{'='*60}")
        print(f"최고 {metric}: {result.get(metric, 'N/A')}")
        print(f"완료 Trial: {result.get('total_complete', 'N/A')}")
        print(f"중단 Trial: {result.get('total_pruned', 'N/A')}")
        params = result.get('params', {})
        print(f"\n최적 파라미터:")
        for k, v in params.items():
            if k not in {'initial_capital', 'force_exit_on_end'}:
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print()
