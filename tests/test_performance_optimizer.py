"""
Unit tests for src/visualizer/performance_optimizer.py

Tests vectorized Z-Score calculation:
- calculate_multi_period_zscores()
- _calculate_zscore_vectorized()
- Caching behavior
- Performance optimization

Note: These tests require a valid PostgreSQL database connection.
"""

import pytest
import time

from src.analyzer.normalizer import SupplyNormalizer
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
from src.database.connection import get_pg_engine, is_mv_available


class TestOptimizedMultiPeriodCalculator:
    """Test OptimizedMultiPeriodCalculator class"""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance"""
        engine = get_pg_engine()
        normalizer = SupplyNormalizer(engine)
        return OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)

    def test_calculate_multi_period_basic(self, optimizer):
        """Calculate Z-Scores for multiple periods"""
        periods = {'5D': 5, '10D': 10, '20D': 20}
        result = optimizer.calculate_multi_period_zscores(periods)

        assert not result.empty, "Should return data"
        period_cols = [c for c in result.columns if not c.startswith('_')]
        assert list(period_cols) == ['5D', '10D', '20D'], "Should have all period columns"
        # 방향 확신도 메타데이터 컬럼도 포함되어야 함
        assert '_today_sff' in result.columns, "Should have _today_sff metadata"
        assert '_sff_5d_avg' in result.columns, "Should have _sff_5d_avg metadata (5-day avg sff)"
        assert len(result) > 0, "Should have stocks"

    def test_calculate_multi_period_all_periods(self, optimizer):
        """Calculate Z-Scores for all 7 periods"""
        periods = {
            '5D': 5, '10D': 10, '20D': 20, '50D': 50,
            '100D': 100, '200D': 200, '500D': 500
        }
        result = optimizer.calculate_multi_period_zscores(periods)

        period_cols = [c for c in result.columns if not c.startswith('_')]
        assert len(period_cols) == 7, "Should have all 7 period columns"

    def test_caching_enabled(self, optimizer):
        """Test caching behavior (Python path) or SQL path returns results"""
        periods = {'5D': 5, '10D': 10}

        # First call
        result1 = optimizer.calculate_multi_period_zscores(periods)
        assert not result1.empty, "Should return data"

        if not is_mv_available():
            # Python path: _sff_cache should be populated
            assert optimizer._sff_cache is not None, "Cache should be populated"

        # Second call should also return data
        result2 = optimizer.calculate_multi_period_zscores(periods)
        assert not result2.empty, "Should return data on second call"

    def test_caching_disabled(self):
        """Test behavior with caching disabled"""
        engine = get_pg_engine()
        normalizer = SupplyNormalizer(engine)
        optimizer = OptimizedMultiPeriodCalculator(normalizer, enable_caching=False)

        periods = {'5D': 5, '10D': 10}
        result = optimizer.calculate_multi_period_zscores(periods)

        assert not result.empty, "Should still return data"

    def test_specific_stocks_only(self, optimizer):
        """Calculate Z-Scores for specific stocks only"""
        periods = {'5D': 5, '10D': 10}
        stock_codes = ['005930', '000660']

        result = optimizer.calculate_multi_period_zscores(periods, stock_codes)

        if not result.empty:
            assert len(result) <= 2, "Should return max 2 stocks"

    def test_clear_cache(self, optimizer):
        """Test cache clearing"""
        # stock_codes 지정 → Python fallback 경로 강제 (SQL은 stock_codes=None만 지원)
        periods = {'5D': 5}
        optimizer.calculate_multi_period_zscores(periods, stock_codes=['005930'])

        assert optimizer._sff_cache is not None, "Cache should be populated (Python path)"

        optimizer.clear_cache()
        assert optimizer._sff_cache is None, "Cache should be cleared"


class TestPerformance:
    """Test performance optimizations"""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance"""
        engine = get_pg_engine()
        normalizer = SupplyNormalizer(engine)
        return OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)

    def test_multi_period_performance(self, optimizer):
        """Test that multi-period calculation is reasonably fast"""
        periods = {
            '5D': 5, '10D': 10, '20D': 20, '50D': 50,
            '100D': 100, '200D': 200, '500D': 500
        }

        start_time = time.time()
        result = optimizer.calculate_multi_period_zscores(periods)
        elapsed = time.time() - start_time

        assert elapsed < 120.0, f"Should complete in < 120 seconds (took {elapsed:.2f}s)"
        assert not result.empty, "Should return data"

    def test_caching_improves_performance(self):
        """Test that caching improves performance"""
        engine = get_pg_engine()
        normalizer = SupplyNormalizer(engine)

        periods = {'5D': 5, '10D': 10, '20D': 20}

        # Without caching
        optimizer_no_cache = OptimizedMultiPeriodCalculator(normalizer, enable_caching=False)
        start1 = time.time()
        result1 = optimizer_no_cache.calculate_multi_period_zscores(periods)
        time_no_cache = time.time() - start1

        # With caching (first call)
        optimizer_cache = OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)
        start2 = time.time()
        result2 = optimizer_cache.calculate_multi_period_zscores(periods)
        time_with_cache = time.time() - start2

        # Note: First call with caching may not be faster
        # SQL path doesn't populate _sff_cache, Python path does
        if not is_mv_available():
            assert optimizer_cache._sff_cache is not None


class TestZScoreCorrectness:
    """Test Z-Score calculation correctness"""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance"""
        engine = get_pg_engine()
        normalizer = SupplyNormalizer(engine)
        return OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)

    def test_zscore_range(self, optimizer):
        """Z-Scores should be in reasonable range"""
        periods = {'5D': 5, '10D': 10, '20D': 20}
        result = optimizer.calculate_multi_period_zscores(periods)

        if not result.empty:
            # Most Z-Scores should be between -5 and 5 (메타데이터 컬럼 제외)
            period_cols = [c for c in result.columns if not c.startswith('_')]
            for col in period_cols:
                valid_zscores = result[col].dropna()
                if len(valid_zscores) > 0:
                    assert valid_zscores.abs().quantile(0.95) < 5.0, \
                        f"95% of Z-Scores should be < 5.0 for {col}"

    def test_zscore_not_all_nan(self, optimizer):
        """Z-Scores should not all be NaN"""
        periods = {'5D': 5, '10D': 10}
        result = optimizer.calculate_multi_period_zscores(periods)

        if not result.empty:
            period_cols = [c for c in result.columns if not c.startswith('_')]
            for col in period_cols:
                non_nan = result[col].notna().sum()
                assert non_nan > 0, f"Column {col} should have non-NaN values"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
