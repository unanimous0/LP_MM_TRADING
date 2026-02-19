"""
WalkForwardAnalyzer 모듈 테스트 (Week 5)

테스트 항목:
1. WalkForwardConfig 기본값 검증
2. split_periods() 기간 분할 수 정확성
3. split_periods() 학습/검증 기간 중복 없음
4. split_periods() 연속성 (스텝 이동 정확성)
5. split_periods() 데이터 부족 시 빈 리스트 반환
6. run() 반환 딕셔너리 키 구조 검증
7. run() 각 기간 결과에 성과 메트릭 포함 확인
8. summary() DataFrame 반환 확인
9. SupplyNormalizer preload() / clear_preload() 동작
10. preload() end_date 필터링 정확성
"""

import pytest
import sqlite3
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.backtesting.walk_forward import WalkForwardAnalyzer, WalkForwardConfig
from src.backtesting.engine import BacktestConfig
from src.analyzer.normalizer import SupplyNormalizer


# ============================================================================
# 픽스처
# ============================================================================

@pytest.fixture
def sample_db():
    """테스트용 인메모리 SQLite DB (investor_flows + stocks 테이블)"""
    conn = sqlite3.connect(':memory:')
    conn.execute("""
        CREATE TABLE investor_flows (
            trade_date TEXT,
            stock_code TEXT,
            foreign_net_amount REAL,
            institution_net_amount REAL,
            close_price REAL,
            free_float_shares REAL
        )
    """)
    conn.execute("""
        CREATE TABLE stocks (
            stock_code TEXT,
            stock_name TEXT,
            sector TEXT
        )
    """)
    # 2024-01-02 ~ 2024-01-05 데이터
    data = [
        ('2024-01-02', '005930', 1_000_000, 500_000, 70_000, 5_000_000_000),
        ('2024-01-03', '005930', -500_000, 200_000, 71_000, 5_000_000_000),
        ('2024-01-04', '005930', 800_000, 300_000, 72_000, 5_000_000_000),
        ('2024-01-05', '005930', -200_000, -100_000, 71_500, 5_000_000_000),
        ('2024-01-02', '000660', 200_000, 100_000, 150_000, 1_000_000_000),
        ('2024-01-03', '000660', -100_000, -50_000, 148_000, 1_000_000_000),
        ('2024-01-04', '000660', 300_000, 150_000, 152_000, 1_000_000_000),
    ]
    conn.executemany(
        "INSERT INTO investor_flows VALUES (?, ?, ?, ?, ?, ?)", data
    )
    conn.execute(
        "INSERT INTO stocks VALUES ('005930', '삼성전자', '반도체')"
    )
    conn.commit()
    yield conn
    conn.close()


def _make_analyzer(start, end, train=6, val=1, step=1):
    """테스트용 WalkForwardAnalyzer 생성"""
    return WalkForwardAnalyzer(
        db_path=':memory:',
        start_date=start,
        end_date=end,
        wf_config=WalkForwardConfig(
            train_months=train,
            val_months=val,
            step_months=step,
        ),
    )


# ============================================================================
# 테스트 클래스
# ============================================================================

class TestWalkForwardConfig:
    """WalkForwardConfig 테스트"""

    def test_wf_config_defaults(self):
        """WalkForwardConfig 기본값 검증"""
        config = WalkForwardConfig()

        assert config.train_months == 6
        assert config.val_months == 1
        assert config.step_months == 1
        assert config.metric == 'sharpe_ratio'
        assert config.top_n == 1
        assert config.workers == 1

    def test_wf_config_custom(self):
        """WalkForwardConfig 사용자 정의 값 저장 확인"""
        config = WalkForwardConfig(
            train_months=3,
            val_months=2,
            step_months=2,
            metric='total_return',
            workers=4,
        )

        assert config.train_months == 3
        assert config.val_months == 2
        assert config.metric == 'total_return'


class TestSplitPeriods:
    """split_periods() 테스트"""

    def test_split_periods_count_12months(self):
        """12개월 데이터, train=6, val=1, step=1 → 6 기간"""
        analyzer = _make_analyzer('2024-01-01', '2024-12-31')
        periods = analyzer.split_periods()

        assert len(periods) == 6

    def test_split_periods_count_1period(self):
        """7개월 데이터, train=6, val=1, step=1 → 정확히 1 기간"""
        # train(6개월) + val(1개월) = 7개월 필요 → 딱 맞음
        analyzer = _make_analyzer('2024-01-01', '2024-07-31')
        periods = analyzer.split_periods()

        assert len(periods) == 1

    def test_split_periods_no_overlap(self):
        """학습/검증 기간 중복 없음 확인"""
        analyzer = _make_analyzer('2024-01-01', '2024-12-31')
        periods = analyzer.split_periods()

        assert len(periods) > 0
        for p in periods:
            # val_start는 train_end보다 1일 후여야 함 (중복 없음)
            train_end = datetime.strptime(p['train_end'], '%Y-%m-%d')
            val_start = datetime.strptime(p['val_start'], '%Y-%m-%d')
            from datetime import timedelta
            assert val_start == train_end + timedelta(days=1), \
                f"중복 발생: train_end={p['train_end']}, val_start={p['val_start']}"

    def test_split_periods_continuous(self):
        """연속성 - 연속된 period의 train_start 차이가 step_months와 일치"""
        analyzer = _make_analyzer('2024-01-01', '2024-12-31', step=1)
        periods = analyzer.split_periods()

        assert len(periods) >= 2

        # 연속된 두 period 확인
        first_start = datetime.strptime(periods[0]['train_start'], '%Y-%m-%d')
        second_start = datetime.strptime(periods[1]['train_start'], '%Y-%m-%d')

        # step=1개월: 두 번째 train_start가 첫 번째보다 1개월 후
        # (월말일 처리 등으로 정확한 day는 다를 수 있으나 month 차이는 1)
        expected_month = first_start.month % 12 + 1
        assert second_start.month == expected_month, \
            f"step 불일치: {first_start.strftime('%Y-%m-%d')} → {second_start.strftime('%Y-%m-%d')}"

    def test_split_periods_insufficient_data(self):
        """데이터 부족 시 빈 리스트 반환"""
        # 5개월 데이터: train(6개월) + val(1개월) = 7개월 필요 → 부족
        analyzer = _make_analyzer('2024-01-01', '2024-05-31')
        periods = analyzer.split_periods()

        assert len(periods) == 0


class TestWalkForwardAnalyzerRun:
    """WalkForwardAnalyzer.run() 테스트"""

    def _mock_opt_result(self):
        """최적화 결과 Mock DataFrame"""
        return pd.DataFrame([{
            'min_score': 60.0,
            'min_signals': 1.0,
            'target_return': 0.15,
            'stop_loss': -0.075,
            'institution_weight': 0.3,
            'total_return': 1.5,
            'sharpe_ratio': 0.8,
            'win_rate': 55.0,
            'max_drawdown': -5.0,
            'profit_factor': 1.2,
            'total_trades': 10,
        }])

    def _mock_engine_result(self):
        """BacktestEngine.run() Mock 결과"""
        return {
            'trades': [],
            'daily_values': pd.DataFrame({
                'date': pd.Series([], dtype=str),
                'value': pd.Series([], dtype=float),
                'cash': pd.Series([], dtype=float),
                'position_count': pd.Series([], dtype=int),
                'total_trades': pd.Series([], dtype=int),
            }),
            'portfolio': MagicMock(),
            'config': BacktestConfig(),
        }

    def test_run_returns_expected_keys(self):
        """run() 반환 딕셔너리 키 구조 검증"""
        analyzer = _make_analyzer('2024-01-01', '2024-12-31')

        with patch('src.backtesting.walk_forward.ParameterOptimizer') as MockOpt, \
             patch('src.backtesting.walk_forward.BacktestEngine') as MockEngine, \
             patch('src.backtesting.walk_forward.PerformanceMetrics') as MockMetrics:

            # Optimizer mock
            MockOpt.return_value.grid_search.return_value = self._mock_opt_result()

            # Engine mock
            MockEngine.return_value.run.return_value = self._mock_engine_result()

            # PerformanceMetrics mock
            mock_metrics = MagicMock()
            mock_metrics.summary.return_value = {
                'total_return': 1.5, 'sharpe_ratio': 0.8,
                'win_rate': 55.0, 'max_drawdown': -5.0,
                'profit_factor': 1.2, 'total_trades': 10,
            }
            MockMetrics.return_value = mock_metrics

            result = analyzer.run(verbose=False)

        assert 'periods' in result
        assert 'combined_trades' in result
        assert 'combined_daily_values' in result
        assert isinstance(result['combined_daily_values'], pd.DataFrame)

    def test_run_periods_have_metrics(self):
        """각 기간 결과에 성과 메트릭 포함 확인"""
        # step=6으로 줄여서 기간 수 제한
        analyzer = _make_analyzer('2024-01-01', '2024-12-31', train=6, val=1, step=6)

        with patch('src.backtesting.walk_forward.ParameterOptimizer') as MockOpt, \
             patch('src.backtesting.walk_forward.BacktestEngine') as MockEngine, \
             patch('src.backtesting.walk_forward.PerformanceMetrics') as MockMetrics:

            MockOpt.return_value.grid_search.return_value = self._mock_opt_result()
            MockEngine.return_value.run.return_value = self._mock_engine_result()

            mock_metrics = MagicMock()
            mock_metrics.summary.return_value = {
                'total_return': 2.0, 'sharpe_ratio': 1.0,
                'win_rate': 60.0, 'max_drawdown': -3.0,
                'profit_factor': 1.5, 'total_trades': 5,
            }
            MockMetrics.return_value = mock_metrics

            result = analyzer.run(verbose=False)

        assert len(result['periods']) >= 1
        for period_result in result['periods']:
            assert 'total_return' in period_result
            assert 'sharpe_ratio' in period_result
            assert 'win_rate' in period_result
            assert 'max_drawdown' in period_result

    def test_summary_returns_dataframe(self):
        """summary() → DataFrame 반환 및 필수 컬럼 확인"""
        analyzer = _make_analyzer('2024-01-01', '2024-12-31', train=6, val=1, step=6)

        with patch('src.backtesting.walk_forward.ParameterOptimizer') as MockOpt, \
             patch('src.backtesting.walk_forward.BacktestEngine') as MockEngine, \
             patch('src.backtesting.walk_forward.PerformanceMetrics') as MockMetrics:

            MockOpt.return_value.grid_search.return_value = self._mock_opt_result()
            MockEngine.return_value.run.return_value = self._mock_engine_result()

            mock_metrics = MagicMock()
            mock_metrics.summary.return_value = {
                'total_return': 1.0, 'sharpe_ratio': 0.5,
                'win_rate': 50.0, 'max_drawdown': -4.0,
                'profit_factor': 1.1, 'total_trades': 3,
            }
            MockMetrics.return_value = mock_metrics

            analyzer.run(verbose=False)

        df = analyzer.summary()

        assert isinstance(df, pd.DataFrame)
        # 필수 컬럼 확인
        for col in ['val_start', 'val_end', 'total_return', 'sharpe_ratio', 'win_rate']:
            assert col in df.columns, f"필수 컬럼 없음: {col}"

    def test_summary_empty_when_no_results(self):
        """결과 없을 때 summary() 빈 DataFrame 반환"""
        analyzer = _make_analyzer('2024-01-01', '2024-12-31')
        # run() 호출 없이 summary() 호출
        df = analyzer.summary()

        assert isinstance(df, pd.DataFrame)
        assert df.empty


class TestNormalizerPreload:
    """SupplyNormalizer preload / clear_preload 테스트"""

    def test_normalizer_preload_activates(self, sample_db):
        """preload() 후 _preload_raw가 None이 아님을 확인"""
        normalizer = SupplyNormalizer(sample_db)

        assert normalizer._preload_raw is None  # 초기값

        normalizer.preload()

        assert normalizer._preload_raw is not None
        assert isinstance(normalizer._preload_raw, pd.DataFrame)
        assert len(normalizer._preload_raw) > 0

    def test_normalizer_clear_preload(self, sample_db):
        """clear_preload() 후 _preload_raw가 None으로 복구됨"""
        normalizer = SupplyNormalizer(sample_db)
        normalizer.preload()
        assert normalizer._preload_raw is not None  # 로드 확인

        normalizer.clear_preload()
        assert normalizer._preload_raw is None  # 해제 확인

    def test_normalizer_preload_filters_by_end_date(self, sample_db):
        """preload(end_date) 후 해당 날짜 이후 데이터가 포함되지 않음"""
        normalizer = SupplyNormalizer(sample_db)

        # 2024-01-03까지만 로드
        normalizer.preload(end_date='2024-01-03')

        assert normalizer._preload_raw is not None
        # 2024-01-04, 2024-01-05 데이터가 없어야 함
        dates_loaded = normalizer._preload_raw['trade_date'].unique()
        assert '2024-01-04' not in dates_loaded, "end_date 이후 데이터가 포함됨"
        assert '2024-01-05' not in dates_loaded, "end_date 이후 데이터가 포함됨"
        assert '2024-01-02' in dates_loaded
        assert '2024-01-03' in dates_loaded

    def test_normalizer_preload_used_in_calculate_sff(self, sample_db):
        """preload 활성화 시 calculate_sff()가 메모리 필터링 사용"""
        normalizer = SupplyNormalizer(sample_db)

        # preload 없이 계산
        result_no_preload = normalizer.calculate_sff(end_date='2024-01-03')

        # preload 후 계산
        normalizer.preload()
        result_with_preload = normalizer.calculate_sff(end_date='2024-01-03')

        # 결과가 동일해야 함
        assert len(result_no_preload) == len(result_with_preload)
        assert set(result_no_preload.columns) == set(result_with_preload.columns)
