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

        # temporal_consistency is computed in classify_all, not calculate_features
        # (that's by design — it must run before tanh)

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

        # Check momentum = 5D - 200D (500D는 참고용, 모멘텀 계산에서 제외)
        expected_momentum = result['5D'] - result['200D']
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
            'persistence': 0.5,
            'temporal_consistency': 0.6,  # 기준 0.5 충족
        })

        pattern = classifier.classify_pattern(row)
        assert pattern == '모멘텀형'

    def test_classify_pattern_sustained_accumulation(self, classifier):
        """Test sustained accumulation pattern classification"""
        row = pd.Series({
            'recent': 0.7,
            'momentum': 0.3,
            'weighted': 0.9,
            'persistence': 0.8,
            'temporal_consistency': 0.7,  # 기준 0.6 충족
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
        assert 'short_trend' in result.columns
        assert 'temporal_consistency' in result.columns

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

    # ------------------------------------------------------------------
    # 신규 테스트: Temporal Consistency + Short Trend (8개)
    # ------------------------------------------------------------------

    def test_temporal_consistency_perfect_order(self, classifier):
        """5D > 10D > ... > 500D 완전 순서 일치 → tc = 1.0"""
        df = pd.DataFrame({
            'stock_code': ['A'],
            '5D':   [3.0],
            '10D':  [2.5],
            '20D':  [2.0],
            '50D':  [1.5],
            '100D': [1.0],
            '200D': [0.5],
            '500D': [0.0],
        })
        tc = PatternClassifier._compute_temporal_consistency(df)
        assert tc.iloc[0] == pytest.approx(1.0)

    def test_temporal_consistency_reverse_order(self, classifier):
        """5D < 10D < ... < 500D 완전 역순 → tc = 0.0"""
        df = pd.DataFrame({
            'stock_code': ['A'],
            '5D':   [0.0],
            '10D':  [0.5],
            '20D':  [1.0],
            '50D':  [1.5],
            '100D': [2.0],
            '200D': [2.5],
            '500D': [3.0],
        })
        tc = PatternClassifier._compute_temporal_consistency(df)
        assert tc.iloc[0] == pytest.approx(0.0)

    def test_temporal_consistency_mixed(self, classifier):
        """절반 정도 순서 일치 → tc ≈ 0.5"""
        df = pd.DataFrame({
            'stock_code': ['A'],
            '5D':   [2.0],  # 5D>=10D ✓
            '10D':  [1.0],  # 10D<20D ✗
            '20D':  [1.5],  # 20D>=50D ✓
            '50D':  [1.0],  # 50D<100D ✗
            '100D': [1.2],  # 100D>=200D ✓
            '200D': [1.0],  # 200D<500D ✗
            '500D': [1.1],
        })
        tc = PatternClassifier._compute_temporal_consistency(df)
        assert tc.iloc[0] == pytest.approx(0.5)

    def test_temporal_consistency_with_nan(self, classifier):
        """NaN 포함 시 유효 쌍으로만 계산 (NaN 쌍은 건너뜀)"""
        df = pd.DataFrame({
            'stock_code': ['A'],
            '5D':   [2.0],
            '10D':  [1.0],   # 5D>=10D ✓
            '20D':  [np.nan],  # 10D~20D 쌍 무효
            '50D':  [np.nan],  # 20D~50D 쌍 무효
            '100D': [1.5],
            '200D': [1.0],   # 100D>=200D ✓
            '500D': [0.5],   # 200D>=500D ✓
        })
        tc = PatternClassifier._compute_temporal_consistency(df)
        # 유효 쌍: (5D,10D) ✓, (100D,200D) ✓, (200D,500D) ✓ → 3/3 = 1.0
        assert tc.iloc[0] == pytest.approx(1.0)

    def test_short_trend_in_output(self, classifier, sample_zscore_matrix):
        """classify_all 결과의 short_trend = 출력 5D - 출력 20D (원본 Z-Score 기준)"""
        result = classifier.classify_all(sample_zscore_matrix)
        assert 'short_trend' in result.columns
        # 출력 5D/20D는 원본 Z-Score 복원값 → short_trend도 원본 기준이어야 함 (Fix #3)
        expected_st = result['5D'] - result['20D']
        pd.testing.assert_series_equal(
            result['short_trend'].reset_index(drop=True),
            expected_st.reset_index(drop=True),
            check_names=False,
        )

    def test_temporal_consistency_in_output(self, classifier, sample_zscore_matrix):
        """classify_all 결과에 temporal_consistency 컬럼 존재 및 범위 확인"""
        result = classifier.classify_all(sample_zscore_matrix)
        assert 'temporal_consistency' in result.columns
        assert (result['temporal_consistency'] >= 0.0).all()
        assert (result['temporal_consistency'] <= 1.0).all()

    def test_momentum_pattern_blocks_low_tc(self, classifier):
        """tc 미달(0.3 < 0.5) 종목은 모멘텀형 → 기타로 분류"""
        row = pd.Series({
            'recent': 1.0,
            'momentum': 1.5,
            'weighted': 0.4,
            'persistence': 0.4,
            'temporal_consistency': 0.3,  # 기준 0.5 미달
        })
        pattern = classifier.classify_pattern(row)
        # 모멘텀형 tc 기준(0.5) 미달이므로 기타
        assert pattern != '모멘텀형'

    def test_score_increases_with_consistency(self, classifier):
        """tc=1.0이 tc=0.0보다 점수가 10점 이상 높음 (비-지속형 패턴)"""
        base = {
            'recent': 1.0,
            'momentum': 0.5,
            'weighted': 0.8,
            'average': 0.6,
            'short_trend': 0.3,
            # 'pattern' 없음 → '' → tc_bonus 적용됨
        }
        row_high_tc = pd.Series({**base, 'temporal_consistency': 1.0})
        row_low_tc  = pd.Series({**base, 'temporal_consistency': 0.0})

        score_high = classifier.calculate_pattern_score(row_high_tc)
        score_low  = classifier.calculate_pattern_score(row_low_tc)

        # tc=1.0 → +10점, tc=0.0 → -10점 → 차이 = 20점
        assert score_high - score_low == pytest.approx(20.0, abs=0.1)

    def test_sustained_pattern_with_low_tc(self, classifier):
        """지속형은 tc 조건이 없으므로 tc=0.1도 지속형으로 분류됨"""
        row = pd.Series({
            'recent': 0.7,
            'momentum': 0.3,
            'weighted': 0.9,
            'persistence': 0.8,
            'temporal_consistency': 0.1,  # 낮은 tc → 지속형은 0.0 이상이면 통과
        })
        pattern = classifier.classify_pattern(row)
        assert pattern == '지속형'

    def test_sustained_score_no_short_trend_penalty(self, classifier):
        """지속형은 short_trend 음수여도 점수 패널티 없음 (가중치 0 처리)"""
        base = {
            'recent': 0.7,
            'momentum': 0.3,
            'weighted': 0.9,
            'average': 0.6,
            'temporal_consistency': 0.2,
            'pattern': '지속형',
        }
        row_neg_st = pd.Series({**base, 'short_trend': -1.0})  # 음수: 지속형에서 정상
        row_pos_st = pd.Series({**base, 'short_trend':  1.0})  # 양수: 단기 모멘텀 있음

        score_neg = classifier.calculate_pattern_score(row_neg_st)
        score_pos = classifier.calculate_pattern_score(row_pos_st)

        # 지속형은 short_trend 가중치 = 0 → 음수/양수 상관없이 동일 점수
        assert score_neg == pytest.approx(score_pos, abs=0.01)

    def test_sustained_no_tc_bonus(self, classifier):
        """지속형은 tc 보너스 없음 — tc=1.0 vs tc=0.0 점수 동일"""
        base = {
            'recent': 0.7,
            'momentum': 0.3,
            'weighted': 0.9,
            'average': 0.6,
            'short_trend': -0.5,
            'pattern': '지속형',
        }
        row_high_tc = pd.Series({**base, 'temporal_consistency': 1.0})
        row_low_tc  = pd.Series({**base, 'temporal_consistency': 0.0})

        score_high = classifier.calculate_pattern_score(row_high_tc)
        score_low  = classifier.calculate_pattern_score(row_low_tc)

        # 지속형: tc_bonus 미적용 → 두 점수 동일
        assert score_high == pytest.approx(score_low, abs=0.01)


class TestScoringToggle:
    """use_tc / use_short_trend 토글 파라미터 검증 (스코어링 개선 before/after 비교용)"""

    @pytest.fixture
    def classifier_default(self):
        """기본값 (use_tc=True, use_short_trend=True) — 현재 스코어링"""
        return PatternClassifier()

    @pytest.fixture
    def classifier_no_tc(self):
        """use_tc=False — tc 조건/보너스 비활성"""
        return PatternClassifier(use_tc=False)

    @pytest.fixture
    def classifier_no_st(self):
        """use_short_trend=False — 레거시 가중치 사용"""
        return PatternClassifier(use_short_trend=False)

    @pytest.fixture
    def classifier_legacy(self):
        """use_tc=False, use_short_trend=False — 스코어링 개선 이전 완전 재현"""
        return PatternClassifier(use_tc=False, use_short_trend=False)

    def test_use_tc_false_bypasses_momentum_tc_condition(self, classifier_no_tc):
        """use_tc=False이면 tc<0.5여도 모멘텀형으로 분류됨"""
        row = pd.Series({
            'recent': 1.0,
            'momentum': 1.5,
            'weighted': 0.4,
            'persistence': 0.4,
            'temporal_consistency': 0.2,  # 기준 0.5 미달이지만 use_tc=False
        })
        assert classifier_no_tc.classify_pattern(row) == '모멘텀형'

    def test_use_tc_true_still_blocks_low_tc(self, classifier_default):
        """기본값(use_tc=True)에서는 tc<0.5면 모멘텀형 탈락 — 기존 동작 유지"""
        row = pd.Series({
            'recent': 1.0,
            'momentum': 1.5,
            'weighted': 0.4,
            'persistence': 0.4,
            'temporal_consistency': 0.2,
        })
        assert classifier_default.classify_pattern(row) != '모멘텀형'

    def test_use_tc_false_no_tc_bonus_in_score(self, classifier_no_tc):
        """use_tc=False이면 tc_bonus(±10점)가 적용되지 않음"""
        base = {
            'recent': 1.0,
            'momentum': 0.5,
            'weighted': 0.8,
            'average': 0.6,
            'short_trend': 0.3,
        }
        score_high_tc = classifier_no_tc.calculate_pattern_score(
            pd.Series({**base, 'temporal_consistency': 1.0})
        )
        score_low_tc = classifier_no_tc.calculate_pattern_score(
            pd.Series({**base, 'temporal_consistency': 0.0})
        )
        # tc_bonus 없으면 두 점수 동일
        assert score_high_tc == pytest.approx(score_low_tc, abs=0.01)

    def test_use_short_trend_false_uses_legacy_weights(
        self, classifier_default, classifier_no_st
    ):
        """use_short_trend=False이면 short_trend가 점수에 영향을 주지 않음"""
        base = {
            'recent': 1.0,
            'momentum': 0.5,
            'weighted': 0.8,
            'average': 0.6,
            'temporal_consistency': 0.5,
        }
        # short_trend 값만 다른 두 행
        score_pos = classifier_no_st.calculate_pattern_score(
            pd.Series({**base, 'short_trend': 2.0})
        )
        score_neg = classifier_no_st.calculate_pattern_score(
            pd.Series({**base, 'short_trend': -2.0})
        )
        # 레거시 가중치에서 short_trend=0.00 → 두 점수 동일
        assert score_pos == pytest.approx(score_neg, abs=0.01)

    def test_use_short_trend_true_affects_score(self, classifier_default):
        """기본값(use_short_trend=True)에서는 short_trend가 점수에 영향"""
        base = {
            'recent': 1.0,
            'momentum': 0.5,
            'weighted': 0.8,
            'average': 0.6,
            'temporal_consistency': 0.5,
        }
        score_pos = classifier_default.calculate_pattern_score(
            pd.Series({**base, 'short_trend': 2.0})
        )
        score_neg = classifier_default.calculate_pattern_score(
            pd.Series({**base, 'short_trend': -2.0})
        )
        # short_trend=0.15 가중치 → 두 점수 달라야 함
        assert score_pos != pytest.approx(score_neg, abs=0.1)

    def test_legacy_classifier_weights_sum_to_one(self, classifier_legacy):
        """레거시 가중치 합계 = 1.00"""
        weights = PatternClassifier._LEGACY_SCORE_WEIGHTS
        total = sum(weights.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_default_and_legacy_score_differ_for_high_short_trend(
        self, classifier_default, classifier_legacy
    ):
        """동일 데이터에서 현재 vs 이전 스코어링 결과가 다름 (검증 비교 가능 확인)"""
        row = pd.Series({
            'recent': 1.2,
            'momentum': 0.8,
            'weighted': 0.9,
            'average': 0.7,
            'short_trend': 1.5,   # 강한 단기 모멘텀 (현재만 반영)
            'temporal_consistency': 0.8,  # 높은 tc (현재만 반영)
            'pattern': '모멘텀형',
        })
        score_current = classifier_default.calculate_pattern_score(row)
        score_legacy = classifier_legacy.calculate_pattern_score(row)
        # 현재 버전이 레거시보다 높아야 함 (tc_bonus + short_trend 가중치 효과)
        assert score_current > score_legacy


class TestSubType:
    """복합 패턴(sub_type) 분류 테스트"""

    @pytest.fixture
    def classifier(self):
        return PatternClassifier()

    def test_sub_type_momentum_long_base(self, classifier):
        """① 모멘텀형 + 200D>0.3 AND 100D>0.3 → sub_type='장기기반'"""
        row = pd.Series({
            '5D': 2.0, '10D': 1.5, '20D': 1.0, '50D': 0.8,
            '100D': 0.5, '200D': 0.5, '500D': 0.2,
            'recent': 1.5, 'momentum': 1.8, 'weighted': 1.0,
            'average': 0.8, 'short_trend': 1.0, 'persistence': 0.8,
            'temporal_consistency': 0.8,
        })
        result = PatternClassifier._classify_sub_type('모멘텀형', row, classifier.config)
        assert result == '장기기반'

    def test_sub_type_sustained_breakout(self, classifier):
        """② 지속형 + 5D>1.0 AND short_trend>0.5 → sub_type='단기돌파'"""
        row = pd.Series({
            '5D': 1.5, '10D': 1.0, '20D': 0.5, '50D': 0.8,
            '100D': 0.9, '200D': 1.0, '500D': 0.8,
            'recent': 1.0, 'momentum': 0.7, 'weighted': 0.9,
            'average': 0.8, 'short_trend': 1.0, 'persistence': 0.8,
            'temporal_consistency': 0.2,
        })
        result = PatternClassifier._classify_sub_type('지속형', row, classifier.config)
        assert result == '단기돌파'

    def test_sub_type_reversal_v_bounce(self, classifier):
        """③ 전환형 + 5D>1.0 AND recent>0.5 → sub_type='V자반등'"""
        row = pd.Series({
            '5D': 1.5, '10D': 0.5, '20D': -0.2, '50D': -0.3,
            '100D': -0.5, '200D': -0.2, '500D': 0.1,
            'recent': 0.65, 'momentum': -0.5, 'weighted': 0.6,
            'average': 0.1, 'short_trend': 1.7, 'persistence': 0.5,
        })
        result = PatternClassifier._classify_sub_type('전환형', row, classifier.config)
        assert result == 'V자반등'

    def test_sub_type_sustained_uniform(self, classifier):
        """④ 지속형 + 5D~200D 모두 양수 & std<0.5 → sub_type='전면수급'"""
        row = pd.Series({
            '5D': 0.8, '10D': 0.9, '20D': 0.7, '50D': 0.8,
            '100D': 0.9, '200D': 0.8, '500D': 0.7,
            'recent': 0.75, 'momentum': 0.1, 'weighted': 0.85,
            'average': 0.8, 'short_trend': 0.1, 'persistence': 1.0,
            'temporal_consistency': 0.3,
        })
        result = PatternClassifier._classify_sub_type('지속형', row, classifier.config)
        assert result == '전면수급'

    def test_sub_type_sustained_weakening(self, classifier):
        """⑤ 지속형 + short_trend<-0.3 AND 5D<20D → sub_type='모멘텀약화'
        주의: 전면수급 조건(모든 Z>0 + std<0.5) 불충족하도록 std>0.5 설정
        """
        row = pd.Series({
            '5D': 0.3, '10D': 0.5, '20D': 0.8, '50D': 1.0,
            '100D': 1.5, '200D': 2.0, '500D': 1.0,  # std > 0.5 (전면수급 불충족)
            'recent': 0.55, 'momentum': -0.7, 'weighted': 0.9,
            'average': 0.8, 'short_trend': -0.5, 'persistence': 1.0,
            'temporal_consistency': 0.0,
        })
        result = PatternClassifier._classify_sub_type('지속형', row, classifier.config)
        assert result == '모멘텀약화'

    def test_sub_type_momentum_decel(self, classifier):
        """⑥ 모멘텀형 + short_trend<-0.3 → sub_type='감속'"""
        row = pd.Series({
            '5D': 1.0, '10D': 1.5, '20D': 1.8, '50D': 0.5,
            '100D': 0.3, '200D': 0.2, '500D': 0.1,
            'recent': 1.4, 'momentum': 0.9, 'weighted': 0.7,
            'average': 0.6, 'short_trend': -0.8, 'persistence': 0.7,
            'temporal_consistency': 0.6,
        })
        result = PatternClassifier._classify_sub_type('모멘텀형', row, classifier.config)
        assert result == '감속'

    def test_sub_type_momentum_dead_cat(self, classifier):
        """⑦ 모멘텀형 + 200D<-0.3 → sub_type='단기반등' (위험 먼저 체크)"""
        row = pd.Series({
            '5D': 2.0, '10D': 1.5, '20D': 1.0, '50D': 0.5,
            '100D': -0.2, '200D': -0.5, '500D': -0.3,
            'recent': 1.5, 'momentum': 2.3, 'weighted': 0.5,
            'average': 0.4, 'short_trend': 1.0, 'persistence': 0.5,
            'temporal_consistency': 0.6,
        })
        result = PatternClassifier._classify_sub_type('모멘텀형', row, classifier.config)
        assert result == '단기반등'

    def test_sub_type_none_for_other_pattern(self, classifier):
        """기타 패턴은 항상 sub_type=None"""
        row = pd.Series({
            '5D': 0.2, '10D': 0.1, '20D': 0.3, '50D': 0.1,
            '100D': 0.0, '200D': -0.1, '500D': 0.0,
        })
        result = PatternClassifier._classify_sub_type('기타', row, classifier.config)
        assert result is None

    def test_pattern_label_format(self, classifier):
        """pattern_label = 'pattern(sub_type)' 또는 'pattern' (sub_type None)"""
        data = {
            'stock_code': ['A', 'B'],
            '5D':   [2.0, 0.2],
            '10D':  [1.5, 0.1],
            '20D':  [1.0, 0.3],
            '50D':  [0.8, 0.1],
            '100D': [0.5, 0.0],
            '200D': [0.5, -0.1],
            '500D': [0.2, 0.0],
        }
        result = classifier.classify_all(pd.DataFrame(data))
        for _, row in result.iterrows():
            if pd.notna(row['sub_type']):
                assert row['pattern_label'] == f"{row['pattern']}({row['sub_type']})"
            else:
                assert row['pattern_label'] == row['pattern']

    def test_sub_type_score_bonus_applied(self, classifier):
        """sub_type 점수 보정이 적용됨 (장기기반 +5, 단기반등 -8)"""
        # 장기기반 데이터 (모멘텀형 + 200D>0.3 + 100D>0.3)
        data_good = {
            'stock_code': ['A'],
            '5D': [2.0], '10D': [1.5], '20D': [1.0], '50D': [0.8],
            '100D': [0.5], '200D': [0.5], '500D': [0.2],
        }
        # 단기반등 데이터 (모멘텀형 + 200D<-0.3)
        data_bad = {
            'stock_code': ['B'],
            '5D': [2.0], '10D': [1.5], '20D': [1.0], '50D': [0.5],
            '100D': [-0.2], '200D': [-0.5], '500D': [-0.3],
        }
        result_good = classifier.classify_all(pd.DataFrame(data_good))
        result_bad = classifier.classify_all(pd.DataFrame(data_bad))
        assert result_good.iloc[0]['sub_type'] == '장기기반'
        assert result_bad.iloc[0]['sub_type'] == '단기반등'
        # 장기기반 점수가 단기반등보다 높아야 함 (+5 vs -8 = 13점 차이 최소)
        assert result_good.iloc[0]['score'] > result_bad.iloc[0]['score']

    def test_classify_all_has_sub_type_columns(self, classifier):
        """classify_all() 결과에 sub_type, pattern_label 컬럼 존재"""
        data = {
            'stock_code': ['A', 'B', 'C'],
            '5D': [1.5, -0.5, 2.0], '10D': [1.3, -0.4, 1.0],
            '20D': [1.2, -0.3, 0.1], '50D': [0.8, 0.2, 0.0],
            '100D': [0.5, 0.5, -0.1], '200D': [0.3, 0.8, 0.1],
            '500D': [0.1, 1.0, 0.0],
        }
        result = classifier.classify_all(pd.DataFrame(data))
        assert 'sub_type' in result.columns
        assert 'pattern_label' in result.columns

    def test_sub_type_short_direction(self, classifier):
        """direction='short' 시 sub_type이 정상 분류되는지 확인"""
        # 강한 매도세: 음수 Z-Score (short에서 부호 반전 → 양수)
        data = {
            'stock_code': ['A'],
            '5D': [-2.0], '10D': [-1.5], '20D': [-1.0], '50D': [-0.8],
            '100D': [-0.5], '200D': [-0.5], '500D': [-0.2],
            '_today_sff': [-0.5], '_sff_5d_avg': [-0.5],
            '_std_5D': [0.3], '_std_10D': [0.3], '_std_20D': [0.3],
            '_std_50D': [0.3], '_std_100D': [0.3], '_std_200D': [0.3],
            '_std_500D': [0.3],
        }
        result = classifier.classify_all(pd.DataFrame(data), direction='short')
        assert 'sub_type' in result.columns
        assert 'pattern_label' in result.columns
        # short 방향: Z 부호 반전 후 200D>0.3, 100D>0.3 → 장기기반 가능
        sub = result.iloc[0]['sub_type']
        # sub_type이 None이 아니면 유효한 값이어야 함
        valid_sub_types = {'장기기반', '단기돌파', 'V자반등', '전면수급',
                          '모멘텀약화', '감속', '단기반등', None}
        assert sub in valid_sub_types or pd.isna(sub)

    def test_sub_type_with_scoring_toggles(self, classifier):
        """use_tc=False, use_short_trend=False에서도 sub_type 정상 분류"""
        data = {
            'stock_code': ['A'],
            '5D': [2.0], '10D': [1.5], '20D': [1.0], '50D': [0.8],
            '100D': [0.5], '200D': [0.5], '500D': [0.2],
        }
        # 기본 (tc+short_trend ON)
        clf_default = PatternClassifier(use_tc=True, use_short_trend=True)
        r_default = clf_default.classify_all(pd.DataFrame(data))

        # 레거시 (tc+short_trend OFF)
        clf_legacy = PatternClassifier(use_tc=False, use_short_trend=False)
        r_legacy = clf_legacy.classify_all(pd.DataFrame(data))

        # 두 경우 모두 sub_type/pattern_label 컬럼 존재
        assert 'sub_type' in r_default.columns
        assert 'sub_type' in r_legacy.columns
        assert 'pattern_label' in r_legacy.columns

        # sub_type 값 자체는 동일해야 함 (tc/short_trend는 점수에만 영향, 분류 조건 무관)
        assert r_default.iloc[0]['sub_type'] == r_legacy.iloc[0]['sub_type']

    def test_sub_type_priority_sustained_breakout_over_uniform(self, classifier):
        """지속형에서 단기돌파 + 전면수급 동시 충족 시 단기돌파 우선"""
        row = pd.Series({
            # 단기돌파 조건: 5D>1.0 AND short_trend>0.5
            '5D': 1.5, '10D': 0.9, '20D': 0.8, '50D': 0.7,
            '100D': 0.6, '200D': 0.5, '500D': 0.4,
            'short_trend': 0.7,
            # 전면수급 조건: all>0 (충족), std < 0.5
            # std([1.5,0.9,0.8,0.7,0.6,0.5]) ≈ 0.33 < 0.5 (충족)
            'recent': 1.15, 'momentum': 1.1, 'weighted': 0.8,
            'average': 0.77, 'persistence': 0.8,
            'temporal_consistency': 0.8,
        })
        result = PatternClassifier._classify_sub_type('지속형', row, classifier.config)
        assert result == '단기돌파', f"단기돌파가 전면수급보다 우선이어야 함, got: {result}"
