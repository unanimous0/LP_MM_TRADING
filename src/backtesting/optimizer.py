"""
파라미터 최적화 모듈

OptunaOptimizer 클래스:
- Bayesian Optimization (--optimize 및 Walk-Forward Analysis 공용)
- MedianPruner: 나쁜 Trial 조기 중단
- 2단계 탐색: Phase 1 (넓은 범위) → Phase 2 (좋은 구간 집중)
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Optional, Dict

from .engine import BacktestConfig, BacktestEngine
from .precomputer import BacktestPrecomputer
from .metrics import PerformanceMetrics


# ============================================================================
# OptunaOptimizer 클래스 (Bayesian Optimization)
# ============================================================================

class OptunaOptimizer:
    """
    Optuna Bayesian Optimization 기반 파라미터 최적화 클래스

    --optimize 및 Walk-Forward Analysis 공용.
    - MedianPruner: 절반 기간 중간 평가로 나쁜 Trial 조기 중단
    - 2단계 탐색: Phase 1 (넓은 범위) → Phase 2 (좋은 구간 집중)

    파라미터 공간 형식:
        {
            'min_score':     {'type': 'float', 'low': 50.0, 'high': 90.0},
            'min_signals':   {'type': 'int',   'low': 1,    'high': 3},
        }
    """

    DEFAULT_PARAM_SPACE = {
        'min_score':     {'type': 'float', 'low': 50.0,  'high': 90.0},
        'min_signals':   {'type': 'int',   'low': 1,     'high': 3},
        'target_return': {'type': 'float', 'low': 0.05,  'high': 0.25},
        'stop_loss':     {'type': 'float', 'low': -0.15, 'high': -0.03},
        # institution_weight는 분석 철학 파라미터 (전략 최적화 대상 아님)
        # BacktestConfig의 고정 파라미터로 관리 (기본값: 0.3)
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

    def _build_objective(self, param_space: dict, metric: str,
                         precomputed=None):
        """
        Optuna objective function 생성 (closure)

        MedianPruner 지원:
        - Step 0: 학습 기간 절반 평가 → trial.report() → prune 판단
        - 통과 시: 전체 기간 평가 → 최종 값 반환

        Args:
            precomputed: PrecomputeResult (외부 주입 시 Trial 간 공유, None이면 Trial마다 계산)
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
                # precomputed 주입 시: 전체 기간 사전 계산 데이터를 절반 루프에도 재사용
                # (rolling은 backward-looking이므로 미래 누수 없음)
                if mid_date > start_date:
                    engine_half = BacktestEngine(conn, config)
                    half_result = engine_half.run(
                        start_date, mid_date, verbose=False,
                        preload_data=(precomputed is None),
                        precomputed=precomputed,
                    )
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
                full_result = engine_full.run(
                    start_date, end_date, verbose=False,
                    preload_data=(precomputed is None),
                    precomputed=precomputed,
                )
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

        # ── Precomputer 1회 실행 (모든 Trial 공유) ────────────────────────
        # institution_weight는 base_config 고정값 사용 (최적화 대상 아님)
        # → Phase 1/2 전체 Trial이 동일한 Precomputed 데이터를 재사용
        if verbose:
            print(f"\n[Precompute] 사전 계산 중 (모든 Trial 공유)...")
        conn_pre = sqlite3.connect(self.db_path)
        try:
            pc = BacktestPrecomputer(conn_pre, self.base_config.institution_weight)
            shared_precomputed = pc.precompute(
                self.end_date, start_date=self.start_date, verbose=verbose
            )
        finally:
            conn_pre.close()
        if verbose:
            print(f"[Precompute] 완료 → {n_trials} Trial에 공유\n")

        # ── Phase 1: 넓은 범위 탐색 ──────────────────────────────────────
        sampler = _optuna.samplers.TPESampler(seed=42)
        study1 = _optuna.create_study(direction='maximize', pruner=pruner,
                                      sampler=sampler)
        study2 = None

        if phase1_n > 0:
            if verbose:
                print(f"[Phase 1] 넓은 범위 탐색 ({phase1_n} trials)...")
            obj1 = self._build_objective(param_space, metric,
                                         precomputed=shared_precomputed)
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

            study2 = _optuna.create_study(direction='maximize', pruner=pruner,
                                          sampler=_optuna.samplers.TPESampler(seed=42))

            # Phase 1 최고 파라미터를 seed trial로 추가
            try:
                if study1.best_trial and study1.best_trial.params:
                    study2.enqueue_trial(study1.best_trial.params)
            except (ValueError, AttributeError):
                pass

            obj2 = self._build_objective(narrowed_space, metric,
                                          precomputed=shared_precomputed)
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
