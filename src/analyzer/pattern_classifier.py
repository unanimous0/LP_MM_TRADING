"""
Pattern-based classification module

Implements Stage 3 pattern classification:
- 3개 바구니 자동 분류 (모멘텀형, 지속형, 전환형)
- 4가지 정렬 키 통합 (Recent, Momentum, Weighted, Average)
- 추가 특성 추출 (변동성, 지속성, 가속도)
- 패턴 강도 점수 계산 (0~100)

용도:
    Stage 2 히트맵 결과(4가지 정렬 키)를 통합하여
    투자 스타일별 종목을 자동 분류하고 패턴 강도를 점수화합니다.
    -> Find "What"
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple


class PatternClassifier:
    """수급 패턴 분류 클래스"""

    def __init__(self, config: Optional[dict] = None):
        """
        Args:
            config: 패턴 분류 설정
                - pattern_thresholds: 패턴별 임계값
                - score_weights: 점수 가중치
        """
        self.config = config or self._get_default_config()

    @staticmethod
    def _get_default_config() -> dict:
        """기본 설정값 반환"""
        return {
            # 패턴 분류 임계값
            'pattern_thresholds': {
                # 모멘텀형: 단기 모멘텀 매우 강함
                'momentum': {
                    'momentum_min': 1.0,      # 1W-2Y > 1.0
                    'recent_min': 0.5,        # (1W+1M)/2 > 0.5
                },
                # 지속형: 장기간 일관된 추세
                'sustained': {
                    'weighted_min': 0.8,      # 가중 평균 > 0.8
                    'persistence_min': 0.7,   # 양수 기간 비율 > 70%
                },
                # 전환형: 추세 약화, 반대 방향 전환 대기
                'reversal': {
                    'weighted_min': 0.5,      # 가중 평균 > 0.5
                    'momentum_max': 0,        # 1W-2Y < 0 (최근 약화)
                }
            },

            # 점수 계산 가중치 (0~100 스케일)
            'score_weights': {
                'recent': 0.25,       # 현재 강도
                'momentum': 0.25,     # 전환점
                'weighted': 0.30,     # 중장기 트렌드
                'average': 0.20,      # 일관성
            },

            # 특성 계산 설정
            'feature_config': {
                'volatility_periods': ['1W', '1M', '3M', '6M', '1Y', '2Y'],
                'persistence_threshold': 0,  # 양수 기준
            }
        }

    def calculate_features(self, zscore_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        추가 특성 계산

        Args:
            zscore_matrix: Stage 2 출력 (stock_code, 1W~2Y, _sort_key)

        Returns:
            pd.DataFrame: 특성 추가된 데이터
                - volatility: 6개 기간 표준편차 (변동성)
                - persistence: 양수 기간 비율 (지속성)
                - sl_ratio: 단기/장기 비율 (가속도)
        """
        df = zscore_matrix.copy()

        periods = self.config['feature_config']['volatility_periods']
        threshold = self.config['feature_config']['persistence_threshold']

        # 1. 변동성 (Volatility): 6개 기간 표준편차
        if all(p in df.columns for p in periods):
            df['volatility'] = df[periods].std(axis=1)
        else:
            df['volatility'] = np.nan

        # 2. 지속성 (Persistence): 양수 기간 비율
        if all(p in df.columns for p in periods):
            df['persistence'] = (df[periods] > threshold).sum(axis=1) / len(periods)
        else:
            df['persistence'] = np.nan

        # 3. 단기/장기 비율 (Short/Long Ratio): 최근 가속도
        if all(p in df.columns for p in ['1W', '1M', '1Y', '2Y']):
            numerator = df['1W'] + df['1M']
            denominator = df['1Y'] + df['2Y'] + 1e-6  # 0으로 나누기 방지
            df['sl_ratio'] = numerator / denominator
        else:
            df['sl_ratio'] = np.nan

        return df

    def calculate_sort_keys(self, zscore_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        4가지 정렬 키 계산

        Args:
            zscore_matrix: Stage 2 출력 (stock_code, 1W~2Y)

        Returns:
            pd.DataFrame: 4가지 정렬 키 추가
                - recent: (1W+1M)/2 - 현재 강도
                - momentum: 1W-2Y - 수급 개선도
                - weighted: 가중 평균 - 중장기 트렌드
                - average: 단순 평균 - 전체 일관성
        """
        df = zscore_matrix.copy()

        # 필수 컬럼 확인
        required_cols = ['1W', '1M', '3M', '6M', '1Y', '2Y']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")

        # 1. Recent: (1W+1M)/2
        df['recent'] = (df['1W'] + df['1M']) / 2

        # 2. Momentum: 1W-2Y
        df['momentum'] = df['1W'] - df['2Y']

        # 3. Weighted: 가중 평균 (최근 기간 높은 가중치)
        weights = {'1W': 3.0, '1M': 2.5, '3M': 2.0, '6M': 1.5, '1Y': 1.0, '2Y': 0.5}
        total_weight = sum(weights.values())

        df['weighted'] = (
            df['1W'] * weights['1W'] +
            df['1M'] * weights['1M'] +
            df['3M'] * weights['3M'] +
            df['6M'] * weights['6M'] +
            df['1Y'] * weights['1Y'] +
            df['2Y'] * weights['2Y']
        ) / total_weight

        # 4. Average: 단순 평균
        df['average'] = df[required_cols].mean(axis=1)

        return df

    def classify_pattern(self, row: pd.Series) -> str:
        """
        패턴 분류 규칙 적용

        Args:
            row: 종목별 데이터 행 (recent, momentum, weighted, persistence 포함)

        Returns:
            str: 패턴명 ('모멘텀형', '지속형', '전환형', '기타')

        Rules:
            1. 모멘텀형: momentum > 1.0 AND recent > 0.5
               → 단기 모멘텀 매우 강함 (추격 매수)
               → 매수: 급상승 중, 단기 추격 전략

            2. 지속형: weighted > 0.8 AND persistence > 0.7
               → 장기간 일관된 추세 (조정 후 진입)
               → 매수: 장기 매집, 5~10% 조정 시 분할 매수

            3. 전환형: weighted > 0.5 AND momentum < 0
               → 추세 약화, 반대 방향 전환 대기
               → 매수: 고점에서 조정 중, 저점 매수 대기 (반등 시그널 확인)
               → 매도(미래): 저점에서 반등 중, 고점 매도 대기 (조정 시그널 확인)

            4. 기타: 위 조건 미충족
        """
        thresholds = self.config['pattern_thresholds']

        # Pattern 1: 모멘텀형
        if (row['momentum'] > thresholds['momentum']['momentum_min'] and
            row['recent'] > thresholds['momentum']['recent_min']):
            return '모멘텀형'

        # Pattern 2: 지속형
        if (row['weighted'] > thresholds['sustained']['weighted_min'] and
            row['persistence'] > thresholds['sustained']['persistence_min']):
            return '지속형'

        # Pattern 3: 전환형
        if (row['weighted'] > thresholds['reversal']['weighted_min'] and
            row['momentum'] < thresholds['reversal']['momentum_max']):
            return '전환형'

        # Pattern 4: 기타
        return '기타'

    def calculate_pattern_score(self, row: pd.Series) -> float:
        """
        패턴 강도 점수 계산 (0~100)

        Args:
            row: 종목별 데이터 행 (recent, momentum, weighted, average 포함)

        Returns:
            float: 패턴 강도 점수 (0~100)

        Formula:
            Score = Σ(정렬키 × 가중치) × 정규화 계수
            - Z-Score 범위 [-3, 3]을 [0, 100]으로 변환
            - 음수 값은 0으로 클리핑
        """
        weights = self.config['score_weights']

        # 가중 합계 계산
        weighted_sum = (
            row['recent'] * weights['recent'] +
            row['momentum'] * weights['momentum'] +
            row['weighted'] * weights['weighted'] +
            row['average'] * weights['average']
        )

        # Z-Score 범위 [-3, 3] → [0, 100] 변환
        # Z=3일 때 100점, Z=0일 때 50점, Z=-3일 때 0점
        score = ((weighted_sum + 3) / 6) * 100

        # 클리핑 (0~100 범위)
        return np.clip(score, 0, 100)

    def classify_all(self, zscore_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        전체 종목 패턴 분류 (메인 메서드)

        Args:
            zscore_matrix: Stage 2 출력 (stock_code, 1W~2Y)

        Returns:
            pd.DataFrame: 패턴 분류 결과
                - stock_code: 종목코드
                - 1W~2Y: 6개 기간 Z-Score
                - recent, momentum, weighted, average: 4가지 정렬 키
                - volatility, persistence, sl_ratio: 추가 특성
                - pattern: 패턴명
                - score: 패턴 강도 점수 (0~100)

        Example:
            >>> classifier = PatternClassifier()
            >>> result = classifier.classify_all(zscore_matrix)
            >>> print(result[['stock_code', 'pattern', 'score']])
        """
        # 1. 4가지 정렬 키 계산
        df = self.calculate_sort_keys(zscore_matrix)

        # 2. 추가 특성 계산
        df = self.calculate_features(df)

        # 3. 패턴 분류
        df['pattern'] = df.apply(self.classify_pattern, axis=1)

        # 4. 패턴 강도 점수 계산
        df['score'] = df.apply(self.calculate_pattern_score, axis=1)

        # 5. 컬럼 순서 정리
        base_cols = ['stock_code']
        period_cols = ['1W', '1M', '3M', '6M', '1Y', '2Y']
        sort_key_cols = ['recent', 'momentum', 'weighted', 'average']
        feature_cols = ['volatility', 'persistence', 'sl_ratio']
        result_cols = ['pattern', 'score']

        # 존재하는 컬럼만 선택 (유연성)
        output_cols = []
        for col in (base_cols + period_cols + sort_key_cols + feature_cols + result_cols):
            if col in df.columns:
                output_cols.append(col)

        return df[output_cols]

    def get_pattern_summary(self, classified_df: pd.DataFrame) -> Dict[str, int]:
        """
        패턴별 종목 수 요약

        Args:
            classified_df: classify_all() 결과

        Returns:
            dict: 패턴별 종목 수
                {'모멘텀형': 12, '지속형': 45, '전환형': 33, '기타': 255}
        """
        return classified_df['pattern'].value_counts().to_dict()

    def filter_by_pattern(self,
                         classified_df: pd.DataFrame,
                         pattern: str,
                         min_score: float = 0,
                         top_n: Optional[int] = None) -> pd.DataFrame:
        """
        패턴별 필터링 + 점수 임계값 적용

        Args:
            classified_df: classify_all() 결과
            pattern: 패턴명 ('모멘텀형', '지속형', '전환형', '기타')
            min_score: 최소 점수 (0~100)
            top_n: 상위 N개만 반환 (None이면 전체)

        Returns:
            pd.DataFrame: 필터링된 결과 (점수 내림차순 정렬)

        Example:
            >>> # 모멘텀형 중 점수 70점 이상, 상위 10개
            >>> df_momentum = classifier.filter_by_pattern(
            ...     classified_df, '모멘텀형', min_score=70, top_n=10
            ... )
        """
        # 패턴 필터링
        df_filtered = classified_df[classified_df['pattern'] == pattern].copy()

        # 점수 필터링
        df_filtered = df_filtered[df_filtered['score'] >= min_score]

        # 점수 내림차순 정렬
        df_filtered = df_filtered.sort_values('score', ascending=False)

        # 상위 N개
        if top_n is not None:
            df_filtered = df_filtered.head(top_n)

        return df_filtered

    def get_top_picks(self,
                     classified_df: pd.DataFrame,
                     top_n_per_pattern: int = 10) -> Dict[str, pd.DataFrame]:
        """
        패턴별 베스트 픽 추출

        Args:
            classified_df: classify_all() 결果
            top_n_per_pattern: 패턴당 상위 N개

        Returns:
            dict: 패턴별 베스트 픽
                {
                    '모멘텀형': DataFrame (top 10),
                    '지속형': DataFrame (top 10),
                    '전환형': DataFrame (top 10)
                }

        Example:
            >>> top_picks = classifier.get_top_picks(classified_df, top_n_per_pattern=10)
            >>> print(top_picks['모멘텀형'][['stock_code', 'score']])
        """
        patterns = ['모멘텀형', '지속형', '전환형']
        result = {}

        for pattern in patterns:
            result[pattern] = self.filter_by_pattern(
                classified_df,
                pattern=pattern,
                top_n=top_n_per_pattern
            )

        return result
