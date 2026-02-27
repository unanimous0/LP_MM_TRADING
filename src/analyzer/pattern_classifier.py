"""
Pattern-based classification module

Implements Stage 3 pattern classification:
- 3개 바구니 자동 분류 (모멘텀형, 지속형, 전환형)
- 5가지 정렬 키 통합 (Recent, Momentum, Weighted, Average, ShortTrend)
- 추가 특성 추출 (변동성, 지속성, 가속도, 시간 순서 일관성)
- 패턴 강도 점수 계산 (0~100)

용도:
    Stage 2 히트맵 결과(5가지 정렬 키)를 통합하여
    투자 스타일별 종목을 자동 분류하고 패턴 강도를 점수화합니다.
    -> Find "What"

메트릭 계산 순서 (중요):
    1. Z-Score 부호 반전 (short 방향)
    2. temporal_consistency 계산 (tanh 이전 필수 — tanh 후 0>=0 오류 방지)
    3. tanh 방향 확신도 적용
    4. short_trend 계산 (tanh 이후 — sort key와 스케일 일치)
    5. 정렬키/특성/패턴/점수 계산
    6. 원본 Z-Score 복원
    7. short_trend 재계산 (복원된 Z-Score 기준 — 출력 표시용)
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple


class PatternClassifier:
    """수급 패턴 분류 클래스"""

    # 스코어링 개선(tc + short_trend) 이전 가중치 — use_short_trend=False 시 사용
    # 변경 이력: momentum 0.25→0.20, average 0.20→0.10, short_trend 0.15 신규
    _LEGACY_SCORE_WEIGHTS = {
        'recent':      0.25,
        'momentum':    0.25,   # short_trend 추가 이전 값
        'weighted':    0.30,
        'average':     0.20,   # short_trend 추가 이전 값
        'short_trend': 0.00,
    }

    def __init__(self, config: Optional[dict] = None,
                 use_tc: bool = True,
                 use_short_trend: bool = True):
        """
        Args:
            config: 패턴 분류 설정
                - pattern_thresholds: 패턴별 임계값
                - score_weights: 점수 가중치
            use_tc: Temporal Consistency 적용 여부
                True(기본): tc 임계값 조건 + tc_bonus ±10점 적용
                False: tc 조건 무시 (스코어링 개선 이전 동작)
            use_short_trend: Short Trend 점수 반영 여부
                True(기본): 현재 가중치 (short_trend=0.15) 사용
                False: 레거시 가중치 사용 (short_trend=0.00, 스코어링 개선 이전 동작)
        """
        self.config = config or self._get_default_config()
        self.use_tc = use_tc
        self.use_short_trend = use_short_trend

    @staticmethod
    def _get_default_config() -> dict:
        """기본 설정값 반환"""
        return {
            # 패턴 분류 임계값
            'pattern_thresholds': {
                # 모멘텀형: 단기 모멘텀 매우 강함
                'momentum': {
                    'momentum_min': 1.0,      # 5D-200D > 1.0
                    'recent_min': 0.5,        # (5D+20D)/2 > 0.5
                    'temporal_consistency_min': 0.5,  # 6쌍 중 ≥3쌍 순서 일치
                },
                # 지속형: 장기간 일관된 추세
                'sustained': {
                    'weighted_min': 0.8,      # 가중 평균 > 0.8
                    'persistence_min': 0.7,   # 양수 기간 비율 > 70%
                    'temporal_consistency_min': 0.0,  # 지속형은 tc 조건 없음
                    # ↑ 이유: 지속형의 이상적 패턴은 5D<10D<...<500D (장기>단기)
                    #   즉, tc=0.0도 완전히 정상 → tc 기준 적용 시 오히려 우량 지속형 탈락
                },
                # 전환형: 추세 약화, 반대 방향 전환 대기
                'reversal': {
                    'weighted_min': 0.5,      # 가중 평균 > 0.5
                    'momentum_max': 0,        # 5D-200D < 0 (최근 약화)
                }
            },

            # 점수 계산 가중치 (합계 = 1.00)
            'score_weights': {
                'recent': 0.25,       # 현재 강도 (유지)
                'momentum': 0.20,     # 장기 개선도 (0.25 → 0.20, 장기전환 과대평가 완화)
                'weighted': 0.30,     # 중장기 트렌드 (유지)
                'average': 0.10,      # 전체 일관성 (0.20 → 0.10, short_trend로 대체)
                'short_trend': 0.15,  # 단기 모멘텀 방향 (5D-20D)
            },

            # 특성 계산 설정
            'feature_config': {
                'volatility_periods': ['5D', '10D', '20D', '50D', '100D', '200D', '500D'],
                'persistence_threshold': 0,  # 양수 기준
            },

            # 복합 패턴(sub_type) 임계값
            'sub_type_thresholds': {
                'long_base_200d': 0.3,          # ① 장기기반: 200D 최소값
                'long_base_100d': 0.3,          # ① 장기기반: 100D 최소값
                'breakout_5d': 1.0,             # ② 단기돌파: 5D 최소값
                'breakout_short_trend': 0.5,    # ② 단기돌파: short_trend 최소값
                'v_bounce_5d': 1.0,             # ③ V자반등: 5D 최소값
                'v_bounce_recent': 0.5,         # ③ V자반등: recent 최소값
                'uniform_std_max': 0.5,         # ④ 전면수급: 5D~200D std 최대값
                'weakening_short_trend': -0.3,  # ⑤ 모멘텀약화: short_trend 최대값
                'decel_short_trend': -0.3,      # ⑥ 감속: short_trend 최대값
                'dead_cat_long_z': -0.3,        # ⑦ 단기반등: 200D/100D 최대값
            },

            # sub_type별 점수 보정
            'sub_type_score_bonus': {
                '장기기반': 5,
                '단기돌파': 5,
                'V자반등': 3,
                '전면수급': 3,
                '모멘텀약화': -5,
                '감속': -5,
                '단기반등': -8,
            },
        }

    def calculate_features(self, zscore_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        추가 특성 계산

        Args:
            zscore_matrix: Stage 2 출력 (stock_code, 5D~500D, _sort_key)

        Returns:
            pd.DataFrame: 특성 추가된 데이터
                - volatility: 7개 기간 표준편차 (변동성)
                - persistence: 양수 기간 비율 (지속성)
                - sl_ratio: 단기/장기 비율 (가속도)
        """
        df = zscore_matrix.copy()

        periods = self.config['feature_config']['volatility_periods']
        threshold = self.config['feature_config']['persistence_threshold']

        # 1. 변동성 (Volatility): 7개 기간 표준편차
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
        if all(p in df.columns for p in ['5D', '20D', '200D', '500D']):
            numerator = df['5D'] + df['20D']
            denominator = df['200D'] + df['500D'] + 1e-6  # 0으로 나누기 방지
            df['sl_ratio'] = numerator / denominator
        else:
            df['sl_ratio'] = np.nan

        return df

    def calculate_sort_keys(self, zscore_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        4가지 정렬 키 계산

        Args:
            zscore_matrix: Stage 2 출력 (stock_code, 5D~500D)

        Returns:
            pd.DataFrame: 4가지 정렬 키 추가
                - recent: (5D+20D)/2 - 현재 강도
                - momentum: 5D-200D - 수급 개선도
                - weighted: 가중 평균 - 중장기 트렌드
                - average: 단순 평균 - 전체 일관성
        """
        df = zscore_matrix.copy()

        # 필수 컬럼 확인
        required_cols = ['5D', '10D', '20D', '50D', '100D', '200D', '500D']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns. Expected: {required_cols}")

        # 1. Recent: (5D+20D)/2
        df['recent'] = (df['5D'] + df['20D']) / 2

        # 2. Momentum: 5D - 가장 긴 유효 기간 (200D→100D→50D→20D 순서 폴백)
        # 500D는 참고용 — 모멘텀 계산에서 제외 (장기 기준 = 200D/100D)
        # 백테스트 초기처럼 DB 데이터 부족 시 긴 기간이 NaN일 수 있으므로 폴백 적용
        _longest = (df['200D']
                    .fillna(df['100D'])
                    .fillna(df['50D'])
                    .fillna(df['20D']))
        df['momentum'] = df['5D'] - _longest

        # 3. Weighted: NaN-robust 가중 평균 (데이터 없는 기간 자동 제외)
        # 백테스트 초기에 200D/500D가 NaN이어도 사용 가능한 기간으로 계산
        weights = {'5D': 3.5, '10D': 3.0, '20D': 2.5, '50D': 2.0, '100D': 1.5, '200D': 1.0, '500D': 0.5}
        _w_num = sum(df[p].fillna(0) * w for p, w in weights.items())
        _w_den = sum(df[p].notna().astype(float) * w for p, w in weights.items())
        df['weighted'] = np.where(_w_den > 0, _w_num / _w_den, np.nan)

        # 4. Average: 단순 평균 (skipna=True, pandas 기본값)
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
        tc = row.get('temporal_consistency', 0.5)
        tc = tc if pd.notna(tc) else 0.5

        # Pattern 1: 모멘텀형
        # use_tc=False이면 tc 임계값 조건을 무시 (스코어링 개선 이전 동작)
        tc_ok_momentum = (not self.use_tc) or (
            tc >= thresholds['momentum'].get('temporal_consistency_min', 0.0)
        )
        if (row['momentum'] > thresholds['momentum']['momentum_min'] and
            row['recent'] > thresholds['momentum']['recent_min'] and
            tc_ok_momentum):
            return '모멘텀형'

        # Pattern 2: 지속형
        # 지속형은 tc_min=0.0 (항상 통과) → use_tc 토글 무관
        if (row['weighted'] > thresholds['sustained']['weighted_min'] and
            row['persistence'] > thresholds['sustained']['persistence_min'] and
            tc >= thresholds['sustained'].get('temporal_consistency_min', 0.0)):
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
            row: 종목별 데이터 행 (recent, momentum, weighted, average, short_trend,
                 temporal_consistency, pattern 포함)

        Returns:
            float: 패턴 강도 점수 (0~100)

        Formula:
            Score = Σ(정렬키 × 패턴별 가중치) × 정규화 계수
            - Z-Score 범위 [-3, 3]을 [0, 100]으로 변환
            - 지속형은 short_trend 제외 (가중치 0, average로 흡수)
            - 지속형은 tc_bonus 없음 (tc=0.0이 정상 패턴이므로 페널티 부적절)
            - 모멘텀형/전환형/기타: tc_bonus ±10점 적용

        Pattern-aware weights:
            지속형: short_trend 가중치=0 (average로 재분배)
            기타 패턴: 설정값 그대로 사용
        """
        # use_short_trend=False → 레거시 가중치 (short_trend 도입 이전)
        # use_short_trend=True  → 현재 가중치 (short_trend=0.15 포함)
        weights = self.config['score_weights'] if self.use_short_trend else self._LEGACY_SCORE_WEIGHTS

        # 패턴별 가중치 조정
        # 지속형: 이상적 패턴이 5D<10D<...<500D (장기>단기)이므로
        #   short_trend(=5D-20D)가 음수가 정상 → 패널티 방지를 위해 가중치 0으로 설정
        #   해제된 short_trend 가중치는 average로 재분배하여 total_w=1.0 유지
        pattern = row.get('pattern', '')
        if pattern == '지속형':
            st_w = weights.get('short_trend', 0)
            effective_weights = dict(weights)
            effective_weights['short_trend'] = 0.0
            effective_weights['average'] = weights.get('average', 0) + st_w
        else:
            effective_weights = weights

        # NaN-robust 가중 합계 계산 (데이터 부족 기간 대응)
        components = [
            (row['recent'],                          effective_weights['recent']),
            (row['momentum'],                        effective_weights['momentum']),
            (row['weighted'],                        effective_weights['weighted']),
            (row['average'],                         effective_weights['average']),
            (row.get('short_trend', np.nan),         effective_weights.get('short_trend', 0)),
        ]
        valid = [(v, w) for v, w in components if pd.notna(v) and w > 0]
        if not valid:
            return np.nan

        # 유효 가중치 합계로 정규화하여 원래 스케일 유지
        original_total_w = sum(w for _, w in components)
        valid_total_w = sum(w for _, w in valid)
        weighted_sum = sum(v * w for v, w in valid) / valid_total_w * original_total_w

        # Z-Score 범위 [-3, 3] → [0, 100] 변환
        # Z=3일 때 100점, Z=0일 때 50점, Z=-3일 때 0점
        base_score = ((weighted_sum + 3) / 6) * 100
        base_score = float(np.clip(base_score, 0, 100))

        # Temporal consistency 보너스: ±10점 (지속형 제외, use_tc=True일 때만)
        # 지속형은 tc=0.0(장기>단기)이 이상적이므로 tc 보너스 부적절
        # 모멘텀형/전환형/기타: tc=1.0 → +10점, tc=0.5 → 0점, tc=0.0 → -10점
        # use_tc=False이면 보너스 미적용 (스코어링 개선 이전 동작)
        if self.use_tc and pattern != '지속형':
            tc = row.get('temporal_consistency', 0.5)
            if pd.notna(tc):
                tc_bonus = (tc - 0.5) * 20
                base_score += tc_bonus

        return float(np.clip(base_score, 0, 100))

    @staticmethod
    def _compute_temporal_consistency(df: pd.DataFrame) -> pd.Series:
        """
        인접 기간 Z-Score의 시간 순서 연속성 (0~1)
        long 방향: 5D ≥ 10D ≥ 20D ≥ 50D ≥ 100D ≥ 200D ≥ 500D 이상적

        tanh 적용 전 원본 Z-Score로 계산해야 정확 (호출 위치 중요).
        tanh 이후 매도 종목의 Z-Score가 0으로 zeroed-out되면
        0 >= 0 이 항상 True → tc=1.0 오류 발생.

        Returns:
            pd.Series: temporal_consistency (0.0~1.0)
                0.0: 모든 인접 쌍이 역순
                0.5: 절반만 순서 일치 (유효 쌍 없을 때 기본값)
                1.0: 모든 인접 쌍이 순서 일치 (꾸준한 상승 추세)
        """
        pairs = [('5D', '10D'), ('10D', '20D'), ('20D', '50D'),
                 ('50D', '100D'), ('100D', '200D'), ('200D', '500D')]

        scores = pd.Series(0.0, index=df.index)
        valid_counts = pd.Series(0.0, index=df.index)

        for short_col, long_col in pairs:
            if short_col not in df.columns or long_col not in df.columns:
                continue
            mask = df[short_col].notna() & df[long_col].notna()
            scores += np.where(mask & (df[short_col] >= df[long_col]), 1.0, 0.0)
            valid_counts += mask.astype(float)

        return pd.Series(
            np.where(valid_counts > 0, scores / valid_counts, 0.5),
            index=df.index
        )

    @staticmethod
    def _classify_sub_type(pattern: str, row: pd.Series, config: dict) -> Optional[str]:
        """
        복합 패턴 sub_type 판정

        기본 패턴(모멘텀형/지속형/전환형) 위에 추가 한정자를 부여한다.
        위험 패턴을 먼저 체크 (단기반등 → 감속 → 장기기반 순서).

        Args:
            pattern: 기본 패턴 문자열 ('모멘텀형', '지속형', '전환형', '기타')
            row: Z-Score/feature 행 (5D~500D + sort keys + features)
            config: sub_type_thresholds 포함 설정 딕셔너리

        Returns:
            sub_type 문자열 또는 None (해당 없음)
        """
        th = config.get('sub_type_thresholds', {})
        if not th:
            return None

        def _get(key, default=np.nan):
            v = row.get(key, default)
            return v if pd.notna(v) else default

        if pattern == '모멘텀형':
            # ⑦ 단기반등 (위험 먼저): 장기 매도 속 단기 반등 함정
            z200 = _get('200D', 0.0)
            z100 = _get('100D', 0.0)
            if z200 < th.get('dead_cat_long_z', -0.3) or z100 < th.get('dead_cat_long_z', -0.3):
                return '단기반등'

            # ⑥ 감속: 모멘텀 피크아웃
            st = _get('short_trend', 0.0)
            if st < th.get('decel_short_trend', -0.3):
                return '감속'

            # ① 장기기반: 장기매집 위에 단기폭발
            if z200 > th.get('long_base_200d', 0.3) and z100 > th.get('long_base_100d', 0.3):
                return '장기기반'

            return None

        elif pattern == '지속형':
            # ② 단기돌파: 매집→돌파 시작
            z5 = _get('5D', 0.0)
            st = _get('short_trend', 0.0)
            if z5 > th.get('breakout_5d', 1.0) and st > th.get('breakout_short_trend', 0.5):
                return '단기돌파'

            # ④ 전면수급: 5D~200D 모두 양수 & 표준편차 낮음
            check_cols = ['5D', '10D', '20D', '50D', '100D', '200D']
            vals = [_get(c, np.nan) for c in check_cols]
            valid_vals = [v for v in vals if not np.isnan(v)]
            if valid_vals and all(v > 0 for v in valid_vals) and len(valid_vals) >= 4:
                if np.std(valid_vals) < th.get('uniform_std_max', 0.5):
                    return '전면수급'

            # ⑤ 모멘텀약화: 매집세 약화 중
            z20 = _get('20D', 0.0)
            if st < th.get('weakening_short_trend', -0.3) and z5 < z20:
                return '모멘텀약화'

            return None

        elif pattern == '전환형':
            # ③ V자반등: 강한 V자 반등
            z5 = _get('5D', 0.0)
            recent = _get('recent', 0.0)
            if z5 > th.get('v_bounce_5d', 1.0) and recent > th.get('v_bounce_recent', 0.5):
                return 'V자반등'

            return None

        # 기타 패턴은 sub_type 없음
        return None

    @staticmethod
    def _apply_direction_confidence(df: pd.DataFrame, direction: str) -> pd.DataFrame:
        """
        방향 확신도를 Z-Score에 적용하여 실제 수급 방향을 반영

        Z-Score는 "평균 대비 이탈도"를 측정하므로 매도 중인 종목도
        양수 Z-Score를 가질 수 있다 (매도가 줄어들면 평균보다 높음).
        이를 보정하기 위해 confidence = tanh(today_sff / rolling_std)를 곱한다.

        - today_sff > 0 (실제 매수): confidence > 0 → Z-Score 유지
        - today_sff ≈ 0 (노이즈):    confidence ≈ 0 → Z-Score 감쇠
        - today_sff < 0 (실제 매도): confidence < 0 → 롱에서 제외

        tanh는 자기 변동성(rolling_std) 대비 상대적으로 정규화하므로
        하드코딩된 임계값 없이 종목별 자동 조정된다.

        방향 판단 기준: _sff_5d_avg (5일 평균 sff) 우선, 없으면 _today_sff 폴백.
        오늘 하루 소폭 매도가 있어도 최근 5일 평균이 매수면 long 신호 보존.

        Args:
            df: Z-Score 컬럼 + (_sff_5d_avg 또는 _today_sff) + _std_*D 메타데이터 포함 DataFrame
            direction: 'long' (현재 Z-Score 부호 그대로) 또는 'short' (이미 반전됨)

        Returns:
            pd.DataFrame: Z-Score 컬럼이 confidence 적용된 DataFrame
        """
        # 방향 판단 기준: _sff_5d_avg 우선, 없으면 _today_sff 폴백
        # _sff_5d_avg를 쓰는 이유: 오늘 하루 소폭 매도가 있어도 최근 5일이 순매수면
        # 장기 매수 신호를 보존. today_sff만 보면 오늘 sff<0 → confidence=0 → 신호 소거
        if '_sff_5d_avg' not in df.columns and '_today_sff' not in df.columns:
            return df

        sff_col = '_sff_5d_avg' if '_sff_5d_avg' in df.columns else '_today_sff'
        period_cols = ['5D', '10D', '20D', '50D', '100D', '200D', '500D']

        for col in period_cols:
            std_col = f'_std_{col}'
            if col not in df.columns or std_col not in df.columns:
                continue

            # confidence = tanh(sff_5d_avg / rolling_std)
            # rolling_std가 0이면 confidence도 0 (안전 처리)
            std_safe = df[std_col].replace(0, np.nan)
            confidence = np.tanh(df[sff_col] / std_safe).fillna(0)

            if direction == 'long':
                # 양수 sff(매수)만 통과, 음수 sff(매도) 제거
                df[col] = df[col] * np.maximum(confidence, 0)
            else:  # 'short' — Z-Score는 이미 반전됨, sff는 원본
                # 음수 sff(매도)만 통과, 양수 sff(매수) 제거
                df[col] = df[col] * np.maximum(-confidence, 0)

        return df

    def classify_all(self, zscore_matrix: pd.DataFrame,
                     direction: str = 'long') -> pd.DataFrame:
        """
        전체 종목 패턴 분류 (메인 메서드)

        Args:
            zscore_matrix: Stage 2 출력 (stock_code, 5D~500D)
            direction: 'long' (순매수, Z>0) 또는 'short' (순매도, Z<0)

        Returns:
            pd.DataFrame: 패턴 분류 결과
                - stock_code: 종목코드
                - 5D~500D: 7개 기간 Z-Score (원본 값, 출력용)
                - recent, momentum, weighted, average, short_trend: 5가지 정렬 키
                - volatility, persistence, sl_ratio, temporal_consistency: 추가 특성
                - pattern: 패턴명 (모멘텀형/지속형/전환형/기타)
                - score: 패턴 강도 점수 (0~100)
                - direction: 'long' 또는 'short'

        Note:
            - direction='long': 양수 Z-Score 분석 (순매수)
            - direction='short': 음수 Z-Score 분석 (순매도)
            - 패턴 이름은 방향과 무관하게 동일 (모멘텀형/지속형/전환형)
            - short일 때 Z-Score 부호 반전하여 분석
            - temporal_consistency: tanh 이전 계산 (0>=0 오류 방지)
            - short_trend: tanh 이후 계산 (sort key와 스케일 일치)
            - 출력 short_trend: 원본 Z-Score 복원 후 재계산 (표시 일관성)

        Example:
            >>> classifier = PatternClassifier()
            >>> # 순매수 패턴
            >>> long_result = classifier.classify_all(zscore_matrix, direction='long')
            >>> # 순매도 패턴 (Z-Score 음수)
            >>> short_result = classifier.classify_all(zscore_matrix, direction='short')
        """
        if direction not in ['long', 'short']:
            raise ValueError(f"direction must be 'long' or 'short', got: {direction}")

        df = zscore_matrix.copy()

        # 원본 Z-Score 보존 (출력용 — 히트맵 셀 값 등에 사용)
        period_cols = ['5D', '10D', '20D', '50D', '100D', '200D', '500D']
        original_zscores = {col: df[col].copy() for col in period_cols if col in df.columns}

        # Short 전략: Z-Score 부호 반전 (음수 → 양수 변환하여 패턴 분류)
        if direction == 'short':
            for col in period_cols:
                if col in df.columns:
                    df[col] = -df[col]

        # [temporal_consistency] — tanh 적용 전에 계산 (중요!)
        # tanh 이후에는 매도 종목의 Z-Score가 0으로 zeroed-out되어
        # 0>=0 이 항상 True → tc=1.0 오류 발생하므로 반드시 이전에 계산
        df['temporal_consistency'] = self._compute_temporal_consistency(df)

        # 방향 확신도(direction confidence) 적용
        # Z-Score는 편차를 측정하므로, 매도 중인 종목도 Z>0이 될 수 있음 (매도 완화)
        # confidence = tanh(today_sff / rolling_std)로 실제 수급 방향을 반영
        df = self._apply_direction_confidence(df, direction)

        # [short_trend] — tanh 적용 후에 계산 (중요!)
        # recent/momentum/weighted/average와 동일한 post-tanh 스케일을 사용해야
        # 점수 계산 시 가중치가 의도한 비율로 반영됨 (pre-tanh 값은 스케일이 다름)
        if '5D' in df.columns and '20D' in df.columns:
            df['short_trend'] = df['5D'] - df['20D']
        else:
            df['short_trend'] = np.nan

        # 1. 5가지 정렬 키 계산
        df = self.calculate_sort_keys(df)

        # 2. 추가 특성 계산
        df = self.calculate_features(df)

        # 3. 패턴 분류
        df['pattern'] = df.apply(self.classify_pattern, axis=1)

        # 3.5. 복합 패턴 sub_type 판정
        # Note: 기본 패턴과 동일하게 post-tanh(방향 확신도 보정) 값으로 평가
        df['sub_type'] = df.apply(
            lambda r: self._classify_sub_type(r['pattern'], r, self.config),
            axis=1,
        )
        df['pattern_label'] = df.apply(
            lambda r: f"{r['pattern']}({r['sub_type']})" if pd.notna(r['sub_type']) else r['pattern'],
            axis=1,
        )

        # 4. 패턴 강도 점수 계산
        df['score'] = df.apply(self.calculate_pattern_score, axis=1)

        # 4.5. sub_type 점수 보정 (tc_bonus와 별개)
        sub_bonus_map = self.config.get('sub_type_score_bonus', {})
        if sub_bonus_map:
            df['score'] = df.apply(
                lambda r: float(np.clip(
                    r['score'] + sub_bonus_map.get(r['sub_type'], 0),
                    0, 100,
                )) if pd.notna(r['sub_type']) else r['score'],
                axis=1,
            )

        # 5. direction 컬럼 추가
        df['direction'] = direction

        # 원본 Z-Score 복원 (출력 = 편차 원본값, 분류/점수는 이미 confidence 반영됨)
        for col, original in original_zscores.items():
            df[col] = original

        # 출력용 short_trend 재계산: 복원된 원본 Z-Score 기준 (표시 일관성, Fix #3)
        # 분류/점수는 이미 post-tanh 값으로 계산 완료 → 이 재계산은 출력 컬럼 전용
        if '5D' in df.columns and '20D' in df.columns:
            df['short_trend'] = df['5D'] - df['20D']

        # 6. 컬럼 순서 정리
        base_cols = ['stock_code']
        period_cols = ['5D', '10D', '20D', '50D', '100D', '200D', '500D']
        sort_key_cols = ['recent', 'momentum', 'weighted', 'average', 'short_trend']
        feature_cols = ['volatility', 'persistence', 'sl_ratio', 'temporal_consistency']
        result_cols = ['pattern', 'sub_type', 'pattern_label', 'score', 'direction']

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
