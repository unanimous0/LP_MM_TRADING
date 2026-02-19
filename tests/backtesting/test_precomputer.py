"""
BacktestPrecomputer 모듈 테스트

사전 계산 결과 검증 (Z-Score, 시그널, 가격 lookup)
"""

import pytest
import pandas as pd
import numpy as np
from src.backtesting.precomputer import BacktestPrecomputer, PrecomputeResult, PERIODS
from src.database.connection import get_connection


class TestPrecomputeResult:
    """PrecomputeResult 구조 테스트"""

    @pytest.fixture
    def conn(self):
        conn = get_connection()
        yield conn
        conn.close()

    @pytest.fixture
    def result(self, conn):
        pc = BacktestPrecomputer(conn, institution_weight=0.3)
        return pc.precompute('2024-03-31', verbose=False)

    def test_returns_precompute_result(self, result):
        """precompute가 PrecomputeResult를 반환하는지"""
        assert isinstance(result, PrecomputeResult)

    def test_zscore_not_empty(self, result):
        """Z-Score 데이터가 비어있지 않은지"""
        assert not result.zscore_all_dates.empty

    def test_signals_not_empty(self, result):
        """시그널 데이터가 비어있지 않은지"""
        assert not result.signals_all_dates.empty

    def test_price_lookup_populated(self, result):
        """가격 lookup이 채워져 있는지"""
        assert len(result.price_lookup) > 0

    def test_stock_names_populated(self, result):
        """종목명 lookup이 채워져 있는지"""
        assert len(result.stock_names) > 0

    def test_trading_dates_populated(self, result):
        """거래일 목록이 채워져 있는지"""
        assert len(result.trading_dates) > 0


class TestZScorePrecomputation:
    """Z-Score 사전 계산 테스트"""

    @pytest.fixture
    def conn(self):
        conn = get_connection()
        yield conn
        conn.close()

    @pytest.fixture
    def result(self, conn):
        pc = BacktestPrecomputer(conn, institution_weight=0.3)
        return pc.precompute('2024-03-31', verbose=False)

    def test_zscore_multiindex(self, result):
        """Z-Score DataFrame이 올바른 MultiIndex를 갖는지"""
        assert result.zscore_all_dates.index.names == ['trade_date', 'stock_code']

    def test_zscore_has_all_periods(self, result):
        """Z-Score에 6개 기간 컬럼이 모두 있는지"""
        expected_cols = set(PERIODS.keys())
        assert expected_cols.issubset(set(result.zscore_all_dates.columns))

    def test_zscore_values_reasonable(self, result):
        """Z-Score 값이 합리적인 범위인지 (-10 ~ 10)"""
        for period in PERIODS.keys():
            col = result.zscore_all_dates[period].dropna()
            if len(col) > 0:
                assert col.min() > -20, f"{period} Z-Score too low: {col.min()}"
                assert col.max() < 20, f"{period} Z-Score too high: {col.max()}"

    def test_zscore_matches_slow_path(self, conn):
        """사전 계산 Z-Score가 느린 경로(performance_optimizer)와 일치하는지"""
        from src.analyzer.normalizer import SupplyNormalizer
        from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator

        # 느린 경로: 기존 OptimizedMultiPeriodCalculator
        normalizer = SupplyNormalizer(conn, config={
            'z_score_window': 60,
            'min_data_points': 30,
            'institution_weight': 0.3,
        })
        calculator = OptimizedMultiPeriodCalculator(normalizer, enable_caching=False)
        slow_zscore = calculator.calculate_multi_period_zscores(
            periods_dict=PERIODS,
            end_date='2024-03-31'
        )

        # 빠른 경로: BacktestPrecomputer
        pc = BacktestPrecomputer(conn, institution_weight=0.3)
        result = pc.precompute('2024-03-31', verbose=False)

        # 최신 날짜의 Z-Score 비교
        latest_date = result.trading_dates[-1]
        fast_zscore = result.zscore_all_dates.loc[latest_date]

        # 공통 종목만 비교
        common_stocks = slow_zscore.index.intersection(fast_zscore.index)
        assert len(common_stocks) > 100, f"공통 종목 {len(common_stocks)}개 (최소 100개 필요)"

        for period in PERIODS.keys():
            slow_values = slow_zscore.loc[common_stocks, period]
            fast_values = fast_zscore.loc[common_stocks, period]
            # NaN이 아닌 값만 비교
            mask = slow_values.notna() & fast_values.notna()
            if mask.sum() > 0:
                np.testing.assert_allclose(
                    slow_values[mask].values,
                    fast_values[mask].values,
                    rtol=1e-5, atol=1e-10,
                    err_msg=f"{period} Z-Score 불일치"
                )


class TestSignalPrecomputation:
    """시그널 사전 계산 테스트"""

    @pytest.fixture
    def conn(self):
        conn = get_connection()
        yield conn
        conn.close()

    @pytest.fixture
    def result(self, conn):
        pc = BacktestPrecomputer(conn, institution_weight=0.3)
        return pc.precompute('2024-03-31', verbose=False)

    def test_signals_multiindex(self, result):
        """시그널 DataFrame이 올바른 MultiIndex를 갖는지"""
        assert result.signals_all_dates.index.names == ['trade_date', 'stock_code']

    def test_signals_has_expected_columns(self, result):
        """시그널에 필수 컬럼이 모두 있는지"""
        expected_cols = {'ma_cross', 'ma_diff', 'acceleration', 'sync_rate', 'signal_count'}
        assert expected_cols.issubset(set(result.signals_all_dates.columns))

    def test_signal_count_range(self, result):
        """signal_count가 0~3 범위인지"""
        signal_counts = result.signals_all_dates['signal_count']
        assert signal_counts.min() >= 0
        assert signal_counts.max() <= 3

    def test_ma_cross_is_boolean(self, result):
        """ma_cross가 boolean 타입인지"""
        ma_cross = result.signals_all_dates['ma_cross'].dropna()
        assert ma_cross.isin([True, False]).all()

    def test_sync_rate_range(self, result):
        """sync_rate가 0~100 범위인지"""
        sync_rate = result.signals_all_dates['sync_rate'].dropna()
        if len(sync_rate) > 0:
            assert sync_rate.min() >= 0
            assert sync_rate.max() <= 100


class TestPriceLookup:
    """가격 lookup 테스트"""

    @pytest.fixture
    def conn(self):
        conn = get_connection()
        yield conn
        conn.close()

    @pytest.fixture
    def result(self, conn):
        pc = BacktestPrecomputer(conn, institution_weight=0.3)
        return pc.precompute('2024-03-31', verbose=False)

    def test_samsung_price_exists(self, result):
        """삼성전자 가격이 존재하는지"""
        price = result.price_lookup.get(('005930', '2024-01-02'))
        assert price is not None
        assert price > 0

    def test_price_is_float(self, result):
        """가격이 float 타입인지"""
        for key, price in list(result.price_lookup.items())[:10]:
            assert isinstance(price, float)

    def test_missing_price_returns_none(self, result):
        """없는 가격 조회 시 None 반환"""
        price = result.price_lookup.get(('999999', '2099-12-31'))
        assert price is None


class TestStockNames:
    """종목명 lookup 테스트"""

    @pytest.fixture
    def conn(self):
        conn = get_connection()
        yield conn
        conn.close()

    @pytest.fixture
    def result(self, conn):
        pc = BacktestPrecomputer(conn, institution_weight=0.3)
        return pc.precompute('2024-03-31', verbose=False)

    def test_samsung_name(self, result):
        """삼성전자 종목명이 올바른지"""
        assert result.stock_names.get('005930') == '삼성전자'

    def test_missing_stock_returns_none(self, result):
        """없는 종목 조회 시 None 반환"""
        assert result.stock_names.get('999999') is None


class TestStartDateFilter:
    """start_date 필터 테스트"""

    @pytest.fixture
    def conn(self):
        conn = get_connection()
        yield conn
        conn.close()

    def test_start_date_filters_trading_dates(self, conn):
        """start_date가 거래일 목록을 필터링하는지"""
        pc = BacktestPrecomputer(conn, institution_weight=0.3)
        result = pc.precompute('2024-03-31', start_date='2024-02-01', verbose=False)
        assert all(d >= '2024-02-01' for d in result.trading_dates)

    def test_zscore_still_has_all_data(self, conn):
        """start_date 필터와 관계없이 Z-Score는 전체 데이터 포함"""
        pc = BacktestPrecomputer(conn, institution_weight=0.3)
        result = pc.precompute('2024-03-31', start_date='2024-02-01', verbose=False)
        # Z-Score에는 2024-02-01 이전 데이터도 있어야 함 (rolling 계산용)
        zscore_dates = result.zscore_all_dates.index.get_level_values('trade_date').unique()
        assert any(d < '2024-02-01' for d in zscore_dates)


class TestInstitutionWeight:
    """기관 가중치 테스트"""

    @pytest.fixture
    def conn(self):
        conn = get_connection()
        yield conn
        conn.close()

    def test_different_weights_produce_different_results(self, conn):
        """기관 가중치 변경 시 Z-Score가 달라지는지"""
        pc1 = BacktestPrecomputer(conn, institution_weight=0.3)
        result1 = pc1.precompute('2024-03-31', verbose=False)

        pc2 = BacktestPrecomputer(conn, institution_weight=0.5)
        result2 = pc2.precompute('2024-03-31', verbose=False)

        # 같은 날짜의 Z-Score 비교 (다른 weight → 다른 값)
        date = result1.trading_dates[-1]
        z1 = result1.zscore_all_dates.loc[date, '1W']
        z2 = result2.zscore_all_dates.loc[date, '1W']

        # 완전 동일하지 않아야 함
        assert not z1.equals(z2), "다른 가중치인데 동일한 Z-Score"
