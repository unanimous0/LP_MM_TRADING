"""
ParameterOptimizer 모듈 테스트 (Week 4)

테스트 항목:
1. BacktestConfig institution_weight 저장 확인
2. BacktestEngine이 normalizer에 weight 전달 확인
3. 파라미터 조합 수 정확성 검증
4. grid_search 반환 타입 + 컬럼 구조 검증
5. metric 기준 내림차순 정렬 확인
6. top_n 제한 동작 확인
"""

import pytest
import sqlite3
import pandas as pd
from unittest.mock import patch, MagicMock

from src.backtesting.engine import BacktestConfig, BacktestEngine
from src.backtesting.optimizer import ParameterOptimizer, _run_backtest_worker


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
        # read_sql이 필요할 수 있으므로 mock
        conn.execute = MagicMock()

        config = BacktestConfig(institution_weight=0.1)

        with patch('src.backtesting.engine.SupplyNormalizer') as MockNormalizer:
            with patch('src.backtesting.engine.OptimizedMultiPeriodCalculator'):
                with patch('src.backtesting.engine.PatternClassifier'):
                    with patch('src.backtesting.engine.SignalDetector'):
                        engine = BacktestEngine(conn, config)

            # SupplyNormalizer가 institution_weight=0.1을 포함한 config로 초기화되었는지 확인
            call_kwargs = MockNormalizer.call_args
            passed_config = call_kwargs[1]['config'] if call_kwargs[1] else call_kwargs[0][1]
            assert passed_config['institution_weight'] == 0.1


class TestParameterOptimizer:
    """ParameterOptimizer 테스트"""

    def _make_optimizer(self, db_path=':memory:'):
        """테스트용 optimizer 생성 (짧은 기간)"""
        return ParameterOptimizer(
            db_path=db_path,
            start_date='2024-06-01',
            end_date='2024-06-30',
        )

    def test_build_param_combinations_count(self):
        """파라미터 조합 수 정확성 검증"""
        optimizer = self._make_optimizer()

        param_grid = {
            'min_score': [60, 70],       # 2가지
            'min_signals': [1, 2],        # 2가지
            'institution_weight': [0.0, 0.3, 0.5],  # 3가지
        }

        combinations = optimizer._build_param_combinations(param_grid)

        # 2 × 2 × 3 = 12가지 조합
        assert len(combinations) == 12

    def test_build_param_combinations_values(self):
        """파라미터 조합 값 정확성 검증"""
        optimizer = self._make_optimizer()

        param_grid = {
            'min_score': [60, 80],
            'institution_weight': [0.0, 0.3],
        }

        combinations = optimizer._build_param_combinations(param_grid)

        # 4가지 조합
        assert len(combinations) == 4

        # 각 조합에 base_config 기본값이 포함되는지 확인
        for combo in combinations:
            assert 'initial_capital' in combo
            assert 'max_positions' in combo
            assert 'strategy' in combo

        # min_score 값 확인
        scores = [c['min_score'] for c in combinations]
        assert 60 in scores
        assert 80 in scores

    def test_grid_search_returns_dataframe(self):
        """grid_search 반환 타입 + 컬럼 구조 검증"""
        optimizer = self._make_optimizer()

        # 작은 param_grid로 빠르게 테스트
        small_grid = {
            'min_score': [60],
            'min_signals': [1],
        }

        # BacktestEngine.run을 mock으로 대체
        mock_result = {
            'trades': [],
            'daily_values': pd.DataFrame({'date': [], 'value': [], 'cash': [],
                                          'position_count': [], 'total_trades': []}),
            'portfolio': MagicMock(),
            'config': BacktestConfig(),
        }

        with patch('src.backtesting.optimizer._run_backtest_worker') as mock_worker:
            mock_worker.return_value = {
                'params': {'min_score': 60, 'min_signals': 1},
                'total_return': 1.5,
                'sharpe_ratio': 0.8,
                'win_rate': 55.0,
                'max_drawdown': -5.0,
                'profit_factor': 1.2,
                'total_trades': 10,
            }

            result_df = optimizer.grid_search(
                param_grid=small_grid,
                metric='sharpe_ratio',
                top_n=5,
                workers=1,
                verbose=False,
            )

        # 반환 타입 확인
        assert isinstance(result_df, pd.DataFrame)

        # 필수 성과 컬럼 존재 확인
        required_cols = ['total_return', 'sharpe_ratio', 'win_rate',
                         'max_drawdown', 'profit_factor', 'total_trades']
        for col in required_cols:
            assert col in result_df.columns, f"컬럼 없음: {col}"

    def test_grid_search_sorted_by_metric(self):
        """metric 기준 내림차순 정렬 확인"""
        optimizer = self._make_optimizer()

        # 다양한 sharpe_ratio 값을 가진 mock 결과
        mock_results = [
            {'params': {'min_score': 60}, 'total_return': 1.0, 'sharpe_ratio': 0.5,
             'win_rate': 50.0, 'max_drawdown': -5.0, 'profit_factor': 1.1, 'total_trades': 5},
            {'params': {'min_score': 70}, 'total_return': 2.0, 'sharpe_ratio': 1.5,
             'win_rate': 60.0, 'max_drawdown': -3.0, 'profit_factor': 1.5, 'total_trades': 8},
            {'params': {'min_score': 80}, 'total_return': 0.5, 'sharpe_ratio': 0.2,
             'win_rate': 45.0, 'max_drawdown': -8.0, 'profit_factor': 0.9, 'total_trades': 3},
        ]

        with patch('src.backtesting.optimizer._run_backtest_worker',
                   side_effect=mock_results):
            result_df = optimizer.grid_search(
                param_grid={'min_score': [60, 70, 80]},
                metric='sharpe_ratio',
                top_n=10,
                workers=1,
                verbose=False,
            )

        # sharpe_ratio 내림차순 정렬 확인
        sharpe_values = result_df['sharpe_ratio'].tolist()
        assert sharpe_values == sorted(sharpe_values, reverse=True), \
            f"내림차순 정렬 실패: {sharpe_values}"

    def test_grid_search_top_n(self):
        """top_n 제한 동작 확인"""
        optimizer = self._make_optimizer()

        # 5개 결과가 나올 mock 설정
        def mock_worker(args):
            db_path, params, start, end = args
            return {
                'params': params,
                'total_return': params.get('min_score', 60) * 0.1,
                'sharpe_ratio': params.get('min_score', 60) * 0.01,
                'win_rate': 50.0,
                'max_drawdown': -5.0,
                'profit_factor': 1.0,
                'total_trades': 5,
            }

        with patch('src.backtesting.optimizer._run_backtest_worker', side_effect=mock_worker):
            result_df = optimizer.grid_search(
                param_grid={'min_score': [60, 65, 70, 75, 80]},  # 5개 조합
                metric='sharpe_ratio',
                top_n=3,  # 상위 3개만
                workers=1,
                verbose=False,
            )

        # top_n=3이므로 최대 3행
        assert len(result_df) <= 3
