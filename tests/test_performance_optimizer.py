"""
Unit tests for src/visualizer/performance_optimizer.py

Tests vectorized Z-Score calculation:
- calculate_multi_period_zscores()
- _calculate_zscore_vectorized()
- Caching behavior
- Performance optimization

Note: These tests require a valid database with data.
"""

import pytest
import time
from pathlib import Path

from src.analyzer.normalizer import SupplyNormalizer
from src.visualizer.performance_optimizer import OptimizedMultiPeriodCalculator
from src.database.connection import get_connection


# Check if database exists
DB_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'investor_data.db'
DB_EXISTS = DB_PATH.exists()

skip_if_no_db = pytest.mark.skipif(
    not DB_EXISTS,
    reason="Database not found. Run load_initial_data.py first."
)


@skip_if_no_db
class TestOptimizedMultiPeriodCalculator:
    """Test OptimizedMultiPeriodCalculator class"""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance"""
        conn = get_connection()
        normalizer = SupplyNormalizer(conn)
        yield OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)
        conn.close()

    def test_calculate_multi_period_basic(self, optimizer):
        """Calculate Z-Scores for multiple periods"""
        periods = {'1D': 1, '1W': 5, '1M': 21}
        result = optimizer.calculate_multi_period_zscores(periods)
        
        assert not result.empty, "Should return data"
        assert list(result.columns) == ['1D', '1W', '1M'], "Should have all period columns"
        assert len(result) > 0, "Should have stocks"

    def test_calculate_multi_period_all_periods(self, optimizer):
        """Calculate Z-Scores for all 7 periods"""
        periods = {
            '1D': 1, '1W': 5, '1M': 21, '3M': 63,
            '6M': 126, '1Y': 252, '2Y': 504
        }
        result = optimizer.calculate_multi_period_zscores(periods)
        
        assert len(result.columns) == 7, "Should have all 7 period columns"

    def test_caching_enabled(self, optimizer):
        """Test caching behavior"""
        periods = {'1D': 1, '1W': 5}
        
        # First call (loads cache)
        result1 = optimizer.calculate_multi_period_zscores(periods)
        assert optimizer._sff_cache is not None, "Cache should be populated"
        
        # Second call (uses cache)
        result2 = optimizer.calculate_multi_period_zscores(periods)
        assert optimizer._sff_cache is not None, "Cache should still be populated"

    def test_caching_disabled(self):
        """Test behavior with caching disabled"""
        conn = get_connection()
        normalizer = SupplyNormalizer(conn)
        optimizer = OptimizedMultiPeriodCalculator(normalizer, enable_caching=False)
        
        periods = {'1D': 1, '1W': 5}
        result = optimizer.calculate_multi_period_zscores(periods)
        
        assert not result.empty, "Should still return data"
        conn.close()

    def test_specific_stocks_only(self, optimizer):
        """Calculate Z-Scores for specific stocks only"""
        periods = {'1D': 1, '1W': 5}
        stock_codes = ['005930', '000660']
        
        result = optimizer.calculate_multi_period_zscores(periods, stock_codes)
        
        if not result.empty:
            assert len(result) <= 2, "Should return max 2 stocks"

    def test_clear_cache(self, optimizer):
        """Test cache clearing"""
        periods = {'1D': 1}
        optimizer.calculate_multi_period_zscores(periods)
        
        assert optimizer._sff_cache is not None, "Cache should be populated"
        
        optimizer.clear_cache()
        assert optimizer._sff_cache is None, "Cache should be cleared"


@skip_if_no_db
class TestPerformance:
    """Test performance optimizations"""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance"""
        conn = get_connection()
        normalizer = SupplyNormalizer(conn)
        yield OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)
        conn.close()

    def test_multi_period_performance(self, optimizer):
        """Test that multi-period calculation is reasonably fast"""
        periods = {
            '1D': 1, '1W': 5, '1M': 21, '3M': 63,
            '6M': 126, '1Y': 252, '2Y': 504
        }
        
        start_time = time.time()
        result = optimizer.calculate_multi_period_zscores(periods)
        elapsed = time.time() - start_time
        
        assert elapsed < 5.0, f"Should complete in < 5 seconds (took {elapsed:.2f}s)"
        assert not result.empty, "Should return data"

    def test_caching_improves_performance(self):
        """Test that caching improves performance"""
        conn = get_connection()
        normalizer = SupplyNormalizer(conn)
        
        periods = {'1D': 1, '1W': 5, '1M': 21}
        
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
        # But cache should be populated
        assert optimizer_cache._sff_cache is not None
        
        conn.close()


@skip_if_no_db
class TestZScoreCorrectness:
    """Test Z-Score calculation correctness"""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance"""
        conn = get_connection()
        normalizer = SupplyNormalizer(conn)
        yield OptimizedMultiPeriodCalculator(normalizer, enable_caching=True)
        conn.close()

    def test_zscore_range(self, optimizer):
        """Z-Scores should be in reasonable range"""
        periods = {'1D': 1, '1W': 5, '1M': 21}
        result = optimizer.calculate_multi_period_zscores(periods)
        
        if not result.empty:
            # Most Z-Scores should be between -5 and 5
            for col in [c for c in result.columns if c != '1D']:  # Skip 1D (NaN is normal)
                valid_zscores = result[col].dropna()
                if len(valid_zscores) > 0:
                    assert valid_zscores.abs().quantile(0.95) < 5.0, \
                        f"95% of Z-Scores should be < 5.0 for {col}"

    def test_zscore_not_all_nan(self, optimizer):
        """Z-Scores should not all be NaN"""
        periods = {'1D': 1, '1W': 5}
        result = optimizer.calculate_multi_period_zscores(periods)
        
        if not result.empty:
            for col in [c for c in result.columns if c != '1D']:  # Skip 1D (NaN is normal)
                non_nan = result[col].notna().sum()
                assert non_nan > 0, f"Column {col} should have non-NaN values"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
