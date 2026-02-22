"""
OptunaOptimizer 모듈 테스트

테스트 항목:
1. BacktestConfig institution_weight 저장 확인
2. BacktestEngine이 normalizer에 weight 전달 확인
3. BacktestEngine이 signal_detector에 institution_weight 전달 확인
4. OptunaOptimizer DEFAULT_PARAM_SPACE 필수 키 확인 (institution_weight 제외)
5. optimize() 반환 구조 확인
6. optimize() 모든 Trial 실패 시 None 반환
7. optimize() 최고 metric 값 반환 확인
8. optimize() n_trials 파라미터 동작 확인
"""

import pytest
import sqlite3
import pandas as pd
from unittest.mock import patch, MagicMock

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.optimizer import OptunaOptimizer


class TestInstitutionWeightConfig:
    """institution_weight 설정 테스트"""

    def test_institution_weight_in_config_default(self):
        """BacktestConfig 기본값 institution_weight=0.3 저장 확인"""
        config = BacktestConfig()
        assert config.institution_weight == 0.3

    def test_institution_weight_in_config_custom(self):
        """BacktestConfig 사용자 정의 institution_weight 저장 확인"""
        config = BacktestConfig(institution_weight=0.0)
        assert config.institution_weight == 0.0

        config2 = BacktestConfig(institution_weight=0.5)
        assert config2.institution_weight == 0.5

    def test_institution_weight_passed_to_normalizer(self):
        """BacktestEngine이 normalizer에 institution_weight 전달 확인"""
        conn = MagicMock()
        conn.execute = MagicMock()

        config = BacktestConfig(institution_weight=0.1)

        with patch('src.backtesting.engine.SupplyNormalizer') as MockNormalizer:
            with patch('src.backtesting.engine.OptimizedMultiPeriodCalculator'):
                with patch('src.backtesting.engine.PatternClassifier'):
                    with patch('src.backtesting.engine.SignalDetector'):
                        engine = BacktestEngine(conn, config)

            call_kwargs = MockNormalizer.call_args
            passed_config = call_kwargs[1]['config'] if call_kwargs[1] else call_kwargs[0][1]
            assert passed_config['institution_weight'] == 0.1

    def test_institution_weight_passed_to_signal_detector(self):
        """BacktestEngine이 SignalDetector에 institution_weight 전달 확인"""
        conn = MagicMock()
        conn.execute = MagicMock()

        config = BacktestConfig(institution_weight=0.2)

        with patch('src.backtesting.engine.SupplyNormalizer'):
            with patch('src.backtesting.engine.OptimizedMultiPeriodCalculator'):
                with patch('src.backtesting.engine.PatternClassifier'):
                    with patch('src.backtesting.engine.SignalDetector') as MockDetector:
                        engine = BacktestEngine(conn, config)

            call_kwargs = MockDetector.call_args
            # SignalDetector(conn, institution_weight=0.2) 형태로 호출되는지 확인
            passed_weight = call_kwargs[1].get('institution_weight') if call_kwargs[1] else None
            assert passed_weight == 0.2


class TestOptunaOptimizer:
    """OptunaOptimizer 테스트"""

    def _make_optimizer(self, db_path=':memory:'):
        """테스트용 optimizer 생성 (짧은 기간)"""
        return OptunaOptimizer(
            db_path=db_path,
            start_date='2024-06-01',
            end_date='2024-06-30',
        )

    def test_default_param_space_keys(self):
        """DEFAULT_PARAM_SPACE 필수 키 확인 (institution_weight 제외됨)"""
        required_keys = {
            'min_score', 'min_signals', 'target_return', 'stop_loss',
            'max_positions', 'max_hold_days', 'reverse_signal_threshold',
        }
        assert required_keys == set(OptunaOptimizer.DEFAULT_PARAM_SPACE.keys())
        # institution_weight는 분석 철학 파라미터 + Precomputer 공유 불가 → 최적화 대상 아님
        assert 'institution_weight' not in OptunaOptimizer.DEFAULT_PARAM_SPACE

        # 각 파라미터에 type, low, high 존재 확인
        for key, spec in OptunaOptimizer.DEFAULT_PARAM_SPACE.items():
            assert 'type' in spec, f"{key}: type 없음"
            assert 'low' in spec, f"{key}: low 없음"
            assert 'high' in spec, f"{key}: high 없음"
            assert spec['type'] in ('float', 'int'), f"{key}: 잘못된 type {spec['type']}"

    def test_optimize_returns_expected_structure(self):
        """optimize() 반환 구조 (params, metric, total_complete, total_pruned)"""
        optimizer = self._make_optimizer()

        mock_trades = [MagicMock(return_pct=5.0, hold_days=3, pattern='모멘텀형',
                                  signal_count=2, direction='long')]
        mock_daily = pd.DataFrame({
            'date': ['2024-06-15'], 'value': [10_500_000],
            'cash': [5_000_000], 'position_count': [1], 'total_trades': [1],
        })
        mock_summary = {
            'total_return': 5.0, 'sharpe_ratio': 1.5, 'win_rate': 100.0,
            'max_drawdown': -1.0, 'profit_factor': 999.0, 'total_trades': 1,
        }

        with patch('src.backtesting.optimizer.BacktestPrecomputer') as MockPC:
            with patch('src.backtesting.optimizer.BacktestEngine') as MockEngine:
                with patch('src.backtesting.optimizer.PerformanceMetrics') as MockMetrics:
                    MockPC.return_value.precompute.return_value = MagicMock()
                    mock_engine_inst = MagicMock()
                    mock_engine_inst.run.return_value = {
                        'trades': mock_trades,
                        'daily_values': mock_daily,
                    }
                    MockEngine.return_value = mock_engine_inst
                    MockMetrics.return_value.summary.return_value = mock_summary

                    result = optimizer.optimize(
                        n_trials=4,
                        metric='sharpe_ratio',
                        verbose=False,
                    )

        assert result is not None
        assert 'params' in result
        assert 'sharpe_ratio' in result
        assert 'total_complete' in result
        assert 'total_pruned' in result
        assert isinstance(result['params'], dict)
        assert result['total_complete'] > 0

    def test_optimize_returns_none_on_failure(self):
        """완료된 Trial이 없으면 None 반환"""
        import optuna

        optimizer = self._make_optimizer()

        with patch('src.backtesting.optimizer.BacktestPrecomputer') as MockPC:
            MockPC.return_value.precompute.return_value = MagicMock()

            with patch.object(optimizer, '_build_objective') as mock_build:
                def pruning_objective(trial):
                    trial.report(-999, step=0)
                    raise optuna.exceptions.TrialPruned()

                mock_build.return_value = pruning_objective

                result = optimizer.optimize(
                    n_trials=4,
                    metric='sharpe_ratio',
                    verbose=False,
                )

        assert result is None

    def test_optimize_best_metric_value(self):
        """최고 metric 값 반환 확인"""
        optimizer = self._make_optimizer()

        call_count = [0]

        def make_summary():
            call_count[0] += 1
            values = [0.5, 1.5, 0.8, 1.2, 0.3, 2.0, 0.1, 0.9]
            val = values[(call_count[0] - 1) % len(values)]
            return {
                'total_return': val * 2, 'sharpe_ratio': val,
                'win_rate': 50.0, 'max_drawdown': -5.0,
                'profit_factor': 1.0, 'total_trades': 3,
            }

        mock_trades = [MagicMock(return_pct=2.0, hold_days=3, pattern='모멘텀형',
                                  signal_count=1, direction='long')]
        mock_daily = pd.DataFrame({
            'date': ['2024-06-15'], 'value': [10_200_000],
            'cash': [5_000_000], 'position_count': [1], 'total_trades': [1],
        })

        with patch('src.backtesting.optimizer.BacktestPrecomputer') as MockPC:
            with patch('src.backtesting.optimizer.BacktestEngine') as MockEngine:
                with patch('src.backtesting.optimizer.PerformanceMetrics') as MockMetrics:
                    MockPC.return_value.precompute.return_value = MagicMock()
                    mock_engine_inst = MagicMock()
                    mock_engine_inst.run.return_value = {
                        'trades': mock_trades,
                        'daily_values': mock_daily,
                    }
                    MockEngine.return_value = mock_engine_inst
                    MockMetrics.return_value.summary = make_summary

                    result = optimizer.optimize(
                        n_trials=8,
                        metric='sharpe_ratio',
                        verbose=False,
                    )

        assert result is not None
        assert result['sharpe_ratio'] > 0

    def test_optimize_n_trials(self):
        """n_trials 파라미터 동작 확인"""
        optimizer = self._make_optimizer()

        mock_trades = [MagicMock(return_pct=1.0, hold_days=2, pattern='지속형',
                                  signal_count=1, direction='long')]
        mock_daily = pd.DataFrame({
            'date': ['2024-06-15'], 'value': [10_100_000],
            'cash': [5_000_000], 'position_count': [1], 'total_trades': [1],
        })
        mock_summary = {
            'total_return': 1.0, 'sharpe_ratio': 0.5, 'win_rate': 100.0,
            'max_drawdown': -0.5, 'profit_factor': 999.0, 'total_trades': 1,
        }

        with patch('src.backtesting.optimizer.BacktestPrecomputer') as MockPC:
            with patch('src.backtesting.optimizer.BacktestEngine') as MockEngine:
                with patch('src.backtesting.optimizer.PerformanceMetrics') as MockMetrics:
                    MockPC.return_value.precompute.return_value = MagicMock()
                    mock_engine_inst = MagicMock()
                    mock_engine_inst.run.return_value = {
                        'trades': mock_trades,
                        'daily_values': mock_daily,
                    }
                    MockEngine.return_value = mock_engine_inst
                    MockMetrics.return_value.summary.return_value = mock_summary

                    result = optimizer.optimize(
                        n_trials=10,
                        metric='sharpe_ratio',
                        verbose=False,
                    )

        assert result is not None
        total = result['total_complete'] + result['total_pruned']
        assert total <= 10

    def test_precomputer_called_once_per_optimize(self):
        """optimize() 1회 호출 시 Precomputer가 1회만 실행되는지 확인"""
        optimizer = self._make_optimizer()

        mock_trades = [MagicMock(return_pct=1.0, hold_days=2, pattern='모멘텀형',
                                  signal_count=2, direction='long')]
        mock_daily = pd.DataFrame({
            'date': ['2024-06-15'], 'value': [10_100_000],
            'cash': [5_000_000], 'position_count': [1], 'total_trades': [1],
        })
        mock_summary = {
            'total_return': 1.0, 'sharpe_ratio': 0.8, 'win_rate': 100.0,
            'max_drawdown': -0.3, 'profit_factor': 999.0, 'total_trades': 1,
        }

        with patch('src.backtesting.optimizer.BacktestPrecomputer') as MockPC:
            with patch('src.backtesting.optimizer.BacktestEngine') as MockEngine:
                with patch('src.backtesting.optimizer.PerformanceMetrics') as MockMetrics:
                    MockPC.return_value.precompute.return_value = MagicMock()
                    mock_engine_inst = MagicMock()
                    mock_engine_inst.run.return_value = {
                        'trades': mock_trades,
                        'daily_values': mock_daily,
                    }
                    MockEngine.return_value = mock_engine_inst
                    MockMetrics.return_value.summary.return_value = mock_summary

                    optimizer.optimize(n_trials=6, metric='sharpe_ratio', verbose=False)

            # BacktestPrecomputer는 optimize() 1회당 1번만 인스턴스 생성
            assert MockPC.call_count == 1
            assert MockPC.return_value.precompute.call_count == 1
