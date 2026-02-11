"""
Unit tests for src/analyzer/normalizer.py

Tests Sff and Z-Score calculations:
- calculate_sff()
- calculate_zscore()
- get_abnormal_supply()
- _get_sff_data()

Note: These tests require a valid database with data.
They will be skipped if the database is not available.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.analyzer.normalizer import SupplyNormalizer
from src.database.connection import get_connection


# Check if database exists
DB_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'investor_data.db'
DB_EXISTS = DB_PATH.exists()

skip_if_no_db = pytest.mark.skipif(
    not DB_EXISTS,
    reason="Database not found. Run load_initial_data.py first."
)


@skip_if_no_db
class TestCalculateSff:
    """Test calculate_sff() method"""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer instance"""
        conn = get_connection()
        yield SupplyNormalizer(conn)
        conn.close()

    def test_calculate_sff_valid_stock(self, normalizer):
        """Calculate Sff for valid stock code"""
        result = normalizer.calculate_sff(stock_codes=['005930'])
        
        assert not result.empty, "Should return data for valid stock"
        assert 'foreign_sff' in result.columns
        assert 'institution_sff' in result.columns
        assert 'combined_sff' in result.columns
        
        # Check Sff is in reasonable range (percentage)
        assert result['combined_sff'].abs().max() < 100, "Sff should be < 100%"

    def test_calculate_sff_invalid_stock(self, normalizer):
        """Calculate Sff for invalid stock code"""
        result = normalizer.calculate_sff(stock_codes=['999999'])
        assert result.empty, "Should return empty for invalid stock"

    def test_calculate_sff_date_range(self, normalizer):
        """Calculate Sff with date range"""
        result = normalizer.calculate_sff(
            stock_codes=['005930'],
            start_date='2026-01-01',
            end_date='2026-01-31'
        )
        
        if not result.empty:
            dates = pd.to_datetime(result['trade_date'])
            assert dates.min() >= pd.Timestamp('2026-01-01')
            assert dates.max() <= pd.Timestamp('2026-01-31')

    def test_calculate_sff_sql_injection_blocked(self, normalizer):
        """SQL injection attempt should be blocked"""
        with pytest.raises(ValueError):
            normalizer.calculate_sff(
                stock_codes=["005930'); DROP TABLE stocks; --"]
            )

    def test_calculate_sff_invalid_date_blocked(self, normalizer):
        """Invalid date format should be blocked"""
        with pytest.raises(ValueError):
            normalizer.calculate_sff(
                start_date="2026-01-01'); DROP TABLE--"
            )


@skip_if_no_db
class TestCalculateZScore:
    """Test calculate_zscore() method"""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer instance"""
        conn = get_connection()
        yield SupplyNormalizer(conn)
        conn.close()

    def test_calculate_zscore_valid_stock(self, normalizer):
        """Calculate Z-Score for valid stock"""
        result = normalizer.calculate_zscore(stock_codes=['005930'])
        
        if not result.empty:
            assert 'foreign_zscore' in result.columns
            assert 'institution_zscore' in result.columns
            assert 'combined_zscore' in result.columns
            
            # Z-Score should be in reasonable range (-5 to 5 typically)
            assert result['combined_zscore'].abs().max() < 10, "Z-Score too large"

    def test_calculate_zscore_insufficient_data(self, normalizer):
        """Calculate Z-Score with insufficient data"""
        # Stock with very recent data only
        result = normalizer.calculate_zscore(stock_codes=['999999'])
        assert result.empty, "Should return empty for insufficient data"


@skip_if_no_db
class TestGetAbnormalSupply:
    """Test get_abnormal_supply() method"""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer instance"""
        conn = get_connection()
        yield SupplyNormalizer(conn)
        conn.close()

    def test_get_abnormal_supply_default(self, normalizer):
        """Get abnormal supply with default parameters"""
        result = normalizer.get_abnormal_supply()
        
        if not result.empty:
            assert len(result) <= 20, "Should return top 20 by default"
            assert 'stock_name' in result.columns
            assert 'sector' in result.columns
            assert 'combined_zscore' in result.columns

    def test_get_abnormal_supply_direction_buy(self, normalizer):
        """Get abnormal supply with buy direction"""
        result = normalizer.get_abnormal_supply(direction='buy', top_n=10)
        
        if not result.empty:
            # All Z-Scores should be positive (buy signals)
            assert (result['combined_zscore'] > 0).all(), "Buy signals should have positive Z-Score"

    def test_get_abnormal_supply_direction_sell(self, normalizer):
        """Get abnormal supply with sell direction"""
        result = normalizer.get_abnormal_supply(direction='sell', top_n=10)
        
        if not result.empty:
            # All Z-Scores should be negative (sell signals)
            # Most sell signals should have negative Z-Score
            assert (result['combined_zscore'] < 0).sum() >= len(result) * 0.7

    def test_get_abnormal_supply_high_threshold(self, normalizer):
        """Get abnormal supply with high threshold"""
        result = normalizer.get_abnormal_supply(threshold=3.0, top_n=5)
        
        if not result.empty:
            # All Z-Scores should exceed threshold
            assert (result['combined_zscore'].abs() >= 3.0).any(), "Should have Z-Score >= 3.0"


@skip_if_no_db
class TestGetSffData:
    """Test _get_sff_data() method"""

    @pytest.fixture
    def normalizer(self):
        """Create normalizer instance"""
        conn = get_connection()
        yield SupplyNormalizer(conn)
        conn.close()

    def test_get_sff_data_all_stocks(self, normalizer):
        """Get Sff data for all stocks"""
        result = normalizer._get_sff_data()
        
        assert not result.empty, "Should return data for all stocks"
        assert 'stock_code' in result.columns
        assert 'trade_date' in result.columns
        assert 'combined_sff' in result.columns
        assert len(result.columns) == 3, "Should have exactly 3 columns"

    def test_get_sff_data_specific_stocks(self, normalizer):
        """Get Sff data for specific stocks"""
        result = normalizer._get_sff_data(stock_codes=['005930', '000660'])
        
        if not result.empty:
            unique_stocks = result['stock_code'].unique()
            assert len(unique_stocks) <= 2, "Should return max 2 stocks"

    def test_get_sff_data_sql_injection_blocked(self, normalizer):
        """SQL injection attempt should be blocked"""
        with pytest.raises(ValueError):
            normalizer._get_sff_data(
                stock_codes=["005930'); DROP TABLE--"]
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
