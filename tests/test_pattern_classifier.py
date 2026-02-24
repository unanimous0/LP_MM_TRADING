"""
Unit tests for src/analyzer/pattern_classifier.py

Tests pattern classification logic:
- calculate_features()
- calculate_sort_keys()
- classify_pattern()
- classify_all()
- filter_by_pattern()

Note: These tests use synthetic data and do not require a database.
"""

import pytest
import pandas as pd
import numpy as np

from src.analyzer.pattern_classifier import PatternClassifier


class TestPatternClassifier:
    """Test PatternClassifier class"""

    @pytest.fixture
    def classifier(self):
        """Create classifier instance"""
        return PatternClassifier()

    @pytest.fixture
    def sample_zscore_matrix(self):
        """Create sample Z-Score matrix for testing"""
        data = {
            'stock_code': ['005930', '000660', '035720', '035420', '051910'],
            '5D': [1.5, -0.5, 2.0, 0.8, -1.2],
            '10D': [1.3, -0.4, 1.0, 0.9, -1.0],
            '20D': [1.2, -0.3, 0.1, 1.0, -0.8],
            '50D': [0.8, 0.2, 0.0, 1.2, 0.5],
            '100D': [0.5, 0.5, -0.1, 1.5, 0.8],
            '200D': [0.3, 0.8, 0.1, 1.8, 1.0],
            '500D': [0.1, 1.0, 0.0, 2.0, 1.5],
        }
        return pd.DataFrame(data)

    def test_calculate_features(self, classifier, sample_zscore_matrix):
        """Test feature calculation"""
        result = classifier.calculate_features(sample_zscore_matrix)

        # Check new columns added
        assert 'volatility' in result.columns
        assert 'persistence' in result.columns
        assert 'sl_ratio' in result.columns

        # Check volatility is positive
        assert (result['volatility'] >= 0).all()

        # Check persistence is between 0 and 1
        assert (result['persistence'] >= 0).all()
        assert (result['persistence'] <= 1).all()

    def test_calculate_sort_keys(self, classifier, sample_zscore_matrix):
        """Test sort key calculation"""
        result = classifier.calculate_sort_keys(sample_zscore_matrix)

        # Check new columns added
        assert 'recent' in result.columns
        assert 'momentum' in result.columns
        assert 'weighted' in result.columns
        assert 'average' in result.columns

        # Check recent = (5D + 20D) / 2
        expected_recent = (result['5D'] + result['20D']) / 2
        pd.testing.assert_series_equal(
            result['recent'],
            expected_recent,
            check_names=False
        )

        # Check momentum = 5D - 500D
        expected_momentum = result['5D'] - result['500D']
        pd.testing.assert_series_equal(
            result['momentum'],
            expected_momentum,
            check_names=False
        )

    def test_classify_pattern_momentum_breakout(self, classifier):
        """Test momentum breakout pattern classification"""
        row = pd.Series({
            'recent': 1.0,
            'momentum': 1.5,
            'weighted': 0.5,
            'persistence': 0.5
        })

        pattern = classifier.classify_pattern(row)
        assert pattern == '모멘텀형'

    def test_classify_pattern_sustained_accumulation(self, classifier):
        """Test sustained accumulation pattern classification"""
        row = pd.Series({
            'recent': 0.7,
            'momentum': 0.3,
            'weighted': 0.9,
            'persistence': 0.8
        })

        pattern = classifier.classify_pattern(row)
        assert pattern == '지속형'

    def test_classify_pattern_pullback_bounce(self, classifier):
        """Test pullback bounce pattern classification"""
        row = pd.Series({
            'recent': 0.4,
            'momentum': -0.5,
            'weighted': 0.7,
            'persistence': 0.6
        })

        pattern = classifier.classify_pattern(row)
        assert pattern == '전환형'

    def test_classify_pattern_other(self, classifier):
        """Test other pattern classification"""
        row = pd.Series({
            'recent': 0.2,
            'momentum': 0.1,
            'weighted': 0.3,
            'persistence': 0.4
        })

        pattern = classifier.classify_pattern(row)
        assert pattern == '기타'

    def test_calculate_pattern_score(self, classifier):
        """Test pattern score calculation"""
        row = pd.Series({
            'recent': 1.0,
            'momentum': 1.5,
            'weighted': 0.8,
            'average': 0.6
        })

        score = classifier.calculate_pattern_score(row)

        # Check score is in valid range
        assert 0 <= score <= 100

        # Score should be > 50 for positive values
        assert score > 50

    def test_classify_all(self, classifier, sample_zscore_matrix):
        """Test full classification pipeline"""
        result = classifier.classify_all(sample_zscore_matrix)

        # Check all required columns exist
        assert 'stock_code' in result.columns
        assert 'pattern' in result.columns
        assert 'score' in result.columns
        assert 'recent' in result.columns
        assert 'momentum' in result.columns
        assert 'weighted' in result.columns
        assert 'average' in result.columns

        # Check row count
        assert len(result) == len(sample_zscore_matrix)

        # Check pattern values
        valid_patterns = ['모멘텀형', '지속형', '전환형', '기타']
        assert result['pattern'].isin(valid_patterns).all()

        # Check score range
        assert (result['score'] >= 0).all()
        assert (result['score'] <= 100).all()

    def test_get_pattern_summary(self, classifier, sample_zscore_matrix):
        """Test pattern summary"""
        classified_df = classifier.classify_all(sample_zscore_matrix)
        summary = classifier.get_pattern_summary(classified_df)

        # Check summary is a dict
        assert isinstance(summary, dict)

        # Check total count matches
        total_count = sum(summary.values())
        assert total_count == len(sample_zscore_matrix)

    def test_filter_by_pattern(self, classifier, sample_zscore_matrix):
        """Test pattern filtering"""
        classified_df = classifier.classify_all(sample_zscore_matrix)

        # Filter by pattern
        patterns = classified_df['pattern'].unique()
        if len(patterns) > 0:
            test_pattern = patterns[0]
            filtered = classifier.filter_by_pattern(classified_df, test_pattern)

            # All rows should have the same pattern
            assert (filtered['pattern'] == test_pattern).all()

    def test_filter_by_score(self, classifier, sample_zscore_matrix):
        """Test score filtering"""
        classified_df = classifier.classify_all(sample_zscore_matrix)

        # Filter by score
        min_score = 60
        filtered = classifier.filter_by_pattern(
            classified_df,
            pattern=classified_df['pattern'].iloc[0],
            min_score=min_score
        )

        # All scores should be >= min_score
        if len(filtered) > 0:
            assert (filtered['score'] >= min_score).all()

    def test_get_top_picks(self, classifier, sample_zscore_matrix):
        """Test top picks extraction"""
        classified_df = classifier.classify_all(sample_zscore_matrix)
        top_picks = classifier.get_top_picks(classified_df, top_n_per_pattern=2)

        # Check return type
        assert isinstance(top_picks, dict)

        # Check patterns
        expected_patterns = ['모멘텀형', '지속형', '전환형']
        assert all(pattern in top_picks for pattern in expected_patterns)

        # Check top N limit
        for pattern, df in top_picks.items():
            if len(df) > 0:
                assert len(df) <= 2

    def test_missing_columns_error(self, classifier):
        """Test error handling for missing columns"""
        invalid_df = pd.DataFrame({
            'stock_code': ['005930'],
            '5D': [1.0]
            # Missing other period columns
        })

        with pytest.raises(ValueError):
            classifier.calculate_sort_keys(invalid_df)

    def test_nan_handling(self, classifier):
        """Test NaN handling in features"""
        df_with_nan = pd.DataFrame({
            'stock_code': ['005930'],
            '5D': [np.nan],
            '10D': [0.8],
            '20D': [1.0],
            '50D': [0.5],
            '100D': [0.3],
            '200D': [0.2],
            '500D': [0.1],
        })

        result = classifier.calculate_features(df_with_nan)

        # Should not crash, NaN should be handled
        assert len(result) == 1

    def test_custom_config(self):
        """Test custom configuration"""
        custom_config = {
            'pattern_thresholds': {
                'momentum_breakout': {
                    'momentum_min': 2.0,
                    'recent_min': 1.0,
                },
                'sustained_accumulation': {
                    'weighted_min': 1.0,
                    'persistence_min': 0.8,
                },
                'pullback_bounce': {
                    'weighted_min': 0.7,
                    'momentum_max': -0.5,
                }
            },
            'score_weights': {
                'recent': 0.3,
                'momentum': 0.3,
                'weighted': 0.2,
                'average': 0.2,
            },
            'feature_config': {
                'volatility_periods': ['5D', '10D', '20D', '50D', '100D', '200D', '500D'],
                'persistence_threshold': 0,
            }
        }

        classifier = PatternClassifier(config=custom_config)

        # Check config applied
        assert classifier.config['pattern_thresholds']['momentum_breakout']['momentum_min'] == 2.0
        assert classifier.config['score_weights']['recent'] == 0.3

    def test_classify_all_short_direction(self, classifier):
        """Test classify_all with short direction (순매도 탐지)"""
        # Negative Z-Score data (순매도)
        data = {
            'stock_code': ['005930', '000660', '035720'],
            '5D': [-1.5, -0.5, -2.0],  # 음수 = 순매도
            '10D': [-1.3, -0.4, -1.0],
            '20D': [-1.2, -0.3, -0.1],
            '50D': [-0.8, -0.2, 0.0],
            '100D': [-0.5, 0.0, 0.1],
            '200D': [-0.3, 0.2, 0.2],
            '500D': [-0.1, 0.3, 0.3],
        }
        df = pd.DataFrame(data)

        # Short direction 분류
        result = classifier.classify_all(df, direction='short')

        assert 'direction' in result.columns
        assert (result['direction'] == 'short').all()
        assert 'pattern' in result.columns
        assert 'score' in result.columns

        # Short일 때도 패턴 이름은 동일 (모멘텀형/지속형/전환형/기타)
        assert result['pattern'].isin(['모멘텀형', '지속형', '전환형', '기타']).all()

    def test_classify_all_long_vs_short_same_pattern_names(self, classifier):
        """Test that long and short use same pattern names (모멘텀형/지속형/전환형)"""
        # Same absolute values, opposite signs
        long_data = {
            'stock_code': ['005930'],
            '5D': [1.5],
            '10D': [1.3],
            '20D': [1.2],
            '50D': [0.8],
            '100D': [0.5],
            '200D': [0.3],
            '500D': [0.1],
        }

        short_data = {
            'stock_code': ['000660'],
            '5D': [-1.5],
            '10D': [-1.3],
            '20D': [-1.2],
            '50D': [-0.8],
            '100D': [-0.5],
            '200D': [-0.3],
            '500D': [-0.1],
        }

        long_df = pd.DataFrame(long_data)
        short_df = pd.DataFrame(short_data)

        long_result = classifier.classify_all(long_df, direction='long')
        short_result = classifier.classify_all(short_df, direction='short')

        # 패턴 이름이 동일해야 함 (부호만 다름)
        # Z-Score 절댓값이 같으면 같은 패턴으로 분류되어야 함
        assert long_result.iloc[0]['pattern'] == short_result.iloc[0]['pattern']

        # direction만 다름
        assert long_result.iloc[0]['direction'] == 'long'
        assert short_result.iloc[0]['direction'] == 'short'

    def test_classify_all_invalid_direction(self, classifier, sample_zscore_matrix):
        """Test invalid direction parameter"""
        with pytest.raises(ValueError, match="direction must be 'long' or 'short'"):
            classifier.classify_all(sample_zscore_matrix, direction='invalid')
