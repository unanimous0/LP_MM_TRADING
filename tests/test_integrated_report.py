"""
Unit tests for src/analyzer/integrated_report.py

Tests integrated report generation:
- generate_report()
- filter_report()
- get_pattern_summary_report()
- export_to_csv()

Note: These tests require a valid database with data.
They will be skipped if the database is not available.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.analyzer.integrated_report import IntegratedReport
from src.database.connection import get_connection


# Check if database exists
DB_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'investor_data.db'
DB_EXISTS = DB_PATH.exists()

skip_if_no_db = pytest.mark.skipif(
    not DB_EXISTS,
    reason="Database not found. Run load_initial_data.py first."
)


@skip_if_no_db
class TestIntegratedReport:
    """Test IntegratedReport class"""

    @pytest.fixture
    def report_gen(self):
        """Create report generator instance"""
        conn = get_connection()
        yield IntegratedReport(conn)
        conn.close()

    @pytest.fixture
    def sample_classified_df(self):
        """Create sample classified DataFrame"""
        data = {
            'stock_code': ['005930', '000660', '035720', '035420', '051910'],
            '5D': [1.5, -0.5, 2.0, 0.8, -1.2],
            '10D': [1.3, -0.4, 1.0, 0.9, -1.0],
            '20D': [1.2, -0.3, 0.1, 1.0, -0.8],
            '50D': [0.8, 0.2, 0.0, 1.2, 0.5],
            '100D': [0.5, 0.5, -0.1, 1.5, 0.8],
            '200D': [0.3, 0.8, 0.1, 1.8, 1.0],
            '500D': [0.1, 1.0, 0.0, 2.0, 1.5],
            'recent': [1.35, -0.4, 1.05, 0.9, -1.0],
            'momentum': [1.4, -1.5, 2.0, -1.2, -2.7],
            'weighted': [0.9, 0.4, 0.6, 1.3, 0.0],
            'average': [0.73, 0.28, 0.35, 1.38, 0.05],
            'pattern': ['모멘텀형', '전환형', '모멘텀형', '지속형', '기타'],
            'score': [85, 45, 78, 92, 38]
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def sample_signals_df(self):
        """Create sample signals DataFrame"""
        data = {
            'stock_code': ['005930', '000660', '035720', '035420', '051910'],
            'signal_count': [2, 0, 3, 1, 0],
            'signal_list': [
                ['MA크로스', '가속도 1.8배'],
                [],
                ['MA크로스', '가속도 2.1배', '동조율 75%'],
                ['동조율 72%'],
                []
            ]
        }
        return pd.DataFrame(data)

    def test_generate_entry_stop_recommendation(self, report_gen):
        """Test entry/stop recommendation generation"""
        row = pd.Series({'pattern': '모멘텀형'})
        result = report_gen.generate_entry_stop_recommendation(row)

        assert 'entry_point' in result
        assert 'stop_loss' in result
        assert isinstance(result['entry_point'], str)
        assert isinstance(result['stop_loss'], str)

    def test_generate_report(self, report_gen, sample_classified_df, sample_signals_df):
        """Test integrated report generation"""
        result = report_gen.generate_report(sample_classified_df, sample_signals_df)

        # Check required columns exist
        expected_cols = ['stock_code', 'pattern', 'score', 'signal_count',
                        'entry_point', 'stop_loss']

        for col in expected_cols:
            assert col in result.columns

        # Check row count matches
        assert len(result) == len(sample_classified_df)

        # Check sorted by score descending
        if len(result) > 1:
            assert (result['score'].diff().dropna() <= 0).all()

    def test_generate_report_without_signals(self, report_gen, sample_classified_df):
        """Test report generation without signals"""
        result = report_gen.generate_report(sample_classified_df, signals_df=None)

        # Should still work
        assert len(result) == len(sample_classified_df)
        assert 'signal_count' in result.columns
        assert (result['signal_count'] == 0).all()

    def test_filter_report_by_pattern(self, report_gen, sample_classified_df):
        """Test report filtering by pattern"""
        report_df = report_gen.generate_report(sample_classified_df)

        filtered = report_gen.filter_report(report_df, pattern='모멘텀형')

        # All rows should have the same pattern
        if len(filtered) > 0:
            assert (filtered['pattern'] == '모멘텀형').all()

    def test_filter_report_by_score(self, report_gen, sample_classified_df):
        """Test report filtering by score"""
        report_df = report_gen.generate_report(sample_classified_df)

        min_score = 70
        filtered = report_gen.filter_report(report_df, min_score=min_score)

        # All scores should be >= min_score
        if len(filtered) > 0:
            assert (filtered['score'] >= min_score).all()

    def test_filter_report_by_signal_count(self, report_gen, sample_classified_df, sample_signals_df):
        """Test report filtering by signal count"""
        report_df = report_gen.generate_report(sample_classified_df, sample_signals_df)

        min_signals = 2
        filtered = report_gen.filter_report(report_df, min_signal_count=min_signals)

        # All signal counts should be >= min_signals
        if len(filtered) > 0:
            assert (filtered['signal_count'] >= min_signals).all()

    def test_filter_report_multiple_filters(self, report_gen, sample_classified_df, sample_signals_df):
        """Test report filtering with multiple filters"""
        report_df = report_gen.generate_report(sample_classified_df, sample_signals_df)

        filtered = report_gen.filter_report(
            report_df,
            pattern='모멘텀형',
            min_score=70,
            min_signal_count=1,
            top_n=5
        )

        # Check all filters applied
        if len(filtered) > 0:
            assert (filtered['pattern'] == '모멘텀형').all()
            assert (filtered['score'] >= 70).all()
            assert (filtered['signal_count'] >= 1).all()
            assert len(filtered) <= 5

    def test_get_pattern_summary_report(self, report_gen, sample_classified_df):
        """Test pattern summary report"""
        report_df = report_gen.generate_report(sample_classified_df)
        summary = report_gen.get_pattern_summary_report(report_df)

        # Check columns exist
        expected_cols = ['pattern', 'count', 'avg_score', 'avg_signal_count']
        for col in expected_cols:
            assert col in summary.columns

        # Check total count matches
        total_count = summary['count'].sum()
        assert total_count == len(sample_classified_df)

        # Check sorted by count descending
        if len(summary) > 1:
            assert (summary['count'].diff().dropna() <= 0).all()

    def test_export_to_csv(self, report_gen, sample_classified_df):
        """Test CSV export"""
        report_df = report_gen.generate_report(sample_classified_df)

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name

        try:
            # Export to CSV
            report_gen.export_to_csv(report_df, temp_path)

            # Check file exists
            assert Path(temp_path).exists()

            # Read back and verify
            df_loaded = pd.read_csv(temp_path)
            assert len(df_loaded) == len(report_df)

        finally:
            # Clean up
            Path(temp_path).unlink()

    def test_export_to_csv_all_columns(self, report_gen, sample_classified_df):
        """Test CSV export with all columns"""
        report_df = report_gen.generate_report(sample_classified_df)

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = f.name

        try:
            report_gen.export_to_csv(report_df, temp_path, include_all_columns=True)

            df_loaded = pd.read_csv(temp_path)
            assert len(df_loaded.columns) >= len(report_df.columns)

        finally:
            Path(temp_path).unlink()

    def test_get_watchlist(self, report_gen, sample_classified_df, sample_signals_df):
        """Test watchlist extraction"""
        report_df = report_gen.generate_report(sample_classified_df, sample_signals_df)

        watchlist = report_gen.get_watchlist(report_df, min_score=70, min_signal_count=1)

        # Check return type
        assert isinstance(watchlist, dict)

        # Check patterns
        expected_patterns = ['모멘텀형', '지속형', '전환형']
        assert all(pattern in watchlist for pattern in expected_patterns)

        # Check filters applied
        for pattern, df_watch in watchlist.items():
            if len(df_watch) > 0:
                assert (df_watch['pattern'] == pattern).all()
                assert (df_watch['score'] >= 70).all()
                assert (df_watch['signal_count'] >= 1).all()

    def test_print_summary_card(self, report_gen, sample_classified_df, capsys):
        """Test summary card printing"""
        report_df = report_gen.generate_report(sample_classified_df)

        # Should not crash
        report_gen.print_summary_card(report_df, top_n=3)

        # Check output
        captured = capsys.readouterr()
        assert '종목별 요약 카드' in captured.out

    def test_custom_config(self):
        """Test custom configuration"""
        custom_config = {
            'entry_rules': {
                '모멘텀형': {
                    'condition': '테스트 진입',
                    'description': '테스트 설명'
                }
            },
            'stop_loss_rules': {
                '모멘텀형': -3
            },
            'display': {
                'max_rows': 100,
                'min_score': 60,
            }
        }

        conn = get_connection()
        report_gen = IntegratedReport(conn, config=custom_config)

        # Check config applied
        assert report_gen.config['entry_rules']['모멘텀형']['condition'] == '테스트 진입'
        assert report_gen.config['stop_loss_rules']['모멘텀형'] == -3

        conn.close()

    def test_entry_point_all_patterns(self, report_gen):
        """Test entry point generation for all patterns"""
        patterns = ['모멘텀형', '지속형', '전환형', '기타']

        for pattern in patterns:
            row = pd.Series({'pattern': pattern})
            result = report_gen.generate_entry_stop_recommendation(row)

            assert 'entry_point' in result
            assert 'stop_loss' in result
            assert len(result['entry_point']) > 0
            assert len(result['stop_loss']) > 0

    def test_empty_report(self, report_gen):
        """Test empty report handling"""
        empty_df = pd.DataFrame()

        result = report_gen.generate_report(empty_df)

        # Should handle empty input gracefully
        assert isinstance(result, pd.DataFrame)
