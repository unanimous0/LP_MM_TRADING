"""
Unit tests for src/analyzer/signal_detector.py

Tests signal detection logic:
- detect_ma_crossover()
- calculate_acceleration()
- calculate_sync_rate()
- detect_all_signals()

Note: These tests require a valid database with data.
They will be skipped if the database is not available.
"""

import pytest
import pandas as pd
from pathlib import Path

from src.analyzer.signal_detector import SignalDetector
from src.database.connection import get_connection


# Check if database exists
DB_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'investor_data.db'
DB_EXISTS = DB_PATH.exists()

skip_if_no_db = pytest.mark.skipif(
    not DB_EXISTS,
    reason="Database not found. Run load_initial_data.py first."
)


@skip_if_no_db
class TestSignalDetector:
    """Test SignalDetector class"""

    @pytest.fixture
    def detector(self):
        """Create detector instance"""
        conn = get_connection()
        yield SignalDetector(conn)
        conn.close()

    def test_detect_ma_crossover(self, detector):
        """Test MA golden cross detection"""
        result = detector.detect_ma_crossover()

        # Check columns exist
        expected_cols = ['stock_code', 'trade_date', 'ma_short', 'ma_long',
                        'ma_diff', 'is_golden_cross']

        for col in expected_cols:
            assert col in result.columns

        # If crosses found, check validity
        if len(result) > 0:
            assert (result['is_golden_cross'] == True).all()
            assert (result['ma_short'] > result['ma_long']).all()
            assert (result['ma_diff'] > 0).all()

    def test_calculate_acceleration(self, detector):
        """Test supply acceleration calculation"""
        result = detector.calculate_acceleration()

        # Check columns exist
        expected_cols = ['stock_code', 'trade_date', 'recent_avg',
                        'prev_avg', 'acceleration']

        for col in expected_cols:
            assert col in result.columns

        # Check no inf values
        assert not result['acceleration'].isin([float('inf'), float('-inf')]).any()

    def test_calculate_sync_rate(self, detector):
        """Test foreign-institution sync rate calculation"""
        result = detector.calculate_sync_rate()

        # Check columns exist
        expected_cols = ['stock_code', 'trade_date', 'sync_days',
                        'total_days', 'sync_rate']

        for col in expected_cols:
            assert col in result.columns

        # Check sync_rate is percentage (0~100)
        if len(result) > 0:
            assert (result['sync_rate'] >= 0).all()
            assert (result['sync_rate'] <= 100).all()

            # sync_days <= total_days
            assert (result['sync_days'] <= result['total_days']).all()

    def test_detect_all_signals(self, detector):
        """Test integrated signal detection"""
        result = detector.detect_all_signals()

        # Check columns exist
        expected_cols = ['stock_code', 'ma_cross', 'ma_diff', 'acceleration',
                        'sync_rate', 'signal_count', 'signal_list']

        for col in expected_cols:
            assert col in result.columns

        # Check signal_count is non-negative
        assert (result['signal_count'] >= 0).all()

        # Check signal_count <= 3 (max signals)
        assert (result['signal_count'] <= 3).all()

        # Check signal_list is list type
        assert result['signal_list'].apply(lambda x: isinstance(x, list)).all()

    def test_get_strong_signals(self, detector):
        """Test strong signal filtering"""
        result = detector.get_strong_signals(min_signal_count=2)

        # All results should have signal_count >= 2
        if len(result) > 0:
            assert (result['signal_count'] >= 2).all()

        # Should be sorted by signal_count descending
        if len(result) > 1:
            assert (result['signal_count'].diff().dropna() <= 0).all()

    def test_custom_config(self):
        """Test custom configuration"""
        custom_config = {
            'ma_short': 3,
            'ma_long': 10,
            'acceleration_window': 3,
            'sync_threshold': 100,
            'sync_window': 10,
        }

        conn = get_connection()
        detector = SignalDetector(conn, config=custom_config)

        # Check config applied
        assert detector.config['ma_short'] == 3
        assert detector.config['ma_long'] == 10
        assert detector.config['acceleration_window'] == 3

        conn.close()

    def test_specific_stock_codes(self, detector):
        """Test signal detection for specific stock codes"""
        stock_codes = ['005930', '000660']
        result = detector.detect_all_signals(stock_codes=stock_codes)

        # Check only requested stocks are returned
        if len(result) > 0:
            assert result['stock_code'].isin(stock_codes).all()

    def test_empty_stock_codes(self, detector):
        """Test signal detection with empty stock codes"""
        result = detector.detect_all_signals(stock_codes=[])

        # Should return empty DataFrame
        assert len(result) == 0

    def test_invalid_stock_codes(self, detector):
        """Test signal detection with invalid stock codes"""
        result = detector.detect_all_signals(stock_codes=['999999'])

        # Should return empty or valid DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_ma_crossover_specific_stock(self, detector):
        """Test MA crossover for specific stock"""
        result = detector.detect_ma_crossover(stock_codes=['005930'])

        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_acceleration_specific_stock(self, detector):
        """Test acceleration for specific stock"""
        result = detector.calculate_acceleration(stock_codes=['005930'])

        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_sync_rate_specific_stock(self, detector):
        """Test sync rate for specific stock"""
        result = detector.calculate_sync_rate(stock_codes=['005930'])

        # Should return DataFrame
        assert isinstance(result, pd.DataFrame)

    def test_signal_list_content(self, detector):
        """Test signal_list contains valid signal names"""
        result = detector.detect_all_signals()

        valid_signals = ['MA크로스', '가속도', '동조율']

        for signal_list in result['signal_list']:
            for signal in signal_list:
                # Check if signal contains valid keywords
                assert any(keyword in signal for keyword in valid_signals)

    def test_ma_diff_consistency(self, detector):
        """Test ma_diff consistency with ma_short and ma_long"""
        result = detector.detect_ma_crossover()

        if len(result) > 0:
            # ma_diff should equal ma_short - ma_long
            calculated_diff = result['ma_short'] - result['ma_long']
            pd.testing.assert_series_equal(
                result['ma_diff'],
                calculated_diff,
                check_names=False,
                rtol=1e-5
            )
