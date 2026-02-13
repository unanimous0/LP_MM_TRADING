"""
Integrated report module

Implements Stage 3 final report generation:
- Stage 1 (이상 수급) + Stage 2 (히트맵) + Stage 3 (패턴) 통합
- 종목별 1줄 요약 카드 생성
- 진입/청산 포인트 제시
- 시그널 통합 스코어링

용도:
    전체 분석 파이프라인 결과를 통합하여
    투자 의사결정을 위한 최종 리포트를 생성합니다.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from src.analyzer.pattern_classifier import PatternClassifier
from src.analyzer.signal_detector import SignalDetector


class IntegratedReport:
    """통합 리포트 생성 클래스"""

    def __init__(self, conn, config: Optional[dict] = None):
        """
        Args:
            conn: 데이터베이스 연결
            config: 리포트 생성 설정
                - entry_rules: 진입 규칙
                - stop_loss_rules: 손절 규칙
        """
        self.conn = conn
        self.config = config or self._get_default_config()
        self.classifier = PatternClassifier()
        self.detector = SignalDetector(conn)

    @staticmethod
    def _get_default_config() -> dict:
        """기본 설정값 반환"""
        return {
            # 진입 포인트 규칙
            'entry_rules': {
                '전환돌파형': {
                    'condition': '현재가 진입 가능',
                    'description': '단기 추격 매수, 모멘텀 확인 후 진입'
                },
                '지속매집형': {
                    'condition': '조정 후 재진입',
                    'description': '5~10% 조정 시 분할 매수'
                },
                '조정반등형': {
                    'condition': '저가 매수',
                    'description': '현재가 또는 추가 조정 시 진입'
                },
                '기타': {
                    'condition': '관망',
                    'description': '명확한 패턴 확인 후 진입'
                }
            },

            # 손절 규칙
            'stop_loss_rules': {
                '전환돌파형': -5,    # -5% 손절
                '지속매집형': -10,   # -10% 손절
                '조정반등형': -7,    # -7% 손절
                '기타': -5           # -5% 손절
            },

            # 리포트 표시 옵션
            'display': {
                'max_rows': 50,           # 최대 표시 종목 수
                'min_score': 50,          # 최소 점수 (0~100)
                'include_signals': True,  # 시그널 포함 여부
                'include_entry_stop': True,  # 진입/손절 포함 여부
            }
        }

    def _load_stock_info(self, stock_codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        종목 정보 로드 (내부 헬퍼 메서드)

        Args:
            stock_codes: 종목 코드 리스트 (None이면 전체)

        Returns:
            pd.DataFrame: (stock_code, stock_name, sector)
        """
        if stock_codes:
            codes_str = "','".join(stock_codes)
            where_sql = f"WHERE stock_code IN ('{codes_str}')"
        else:
            where_sql = ""

        query = f"""
        SELECT stock_code, stock_name, sector
        FROM stocks
        {where_sql}
        """

        df = pd.read_sql(query, self.conn)

        if df.empty:
            print("[WARN] No stock info found")
            return pd.DataFrame()

        return df

    def generate_entry_stop_recommendation(self, row: pd.Series) -> Dict[str, str]:
        """
        진입/청산 포인트 생성

        Args:
            row: 종목별 데이터 행 (pattern, signal_count 포함)

        Returns:
            dict: {
                'entry_point': 진입 포인트 설명,
                'stop_loss': 손절 포인트 설명
            }
        """
        pattern = row['pattern']
        signal_count = row.get('signal_count', 0)

        # 시그널 2개 이상 = 강한 진입 신호 (패턴 무관)
        if signal_count >= 2:
            return {
                'entry_point': f'즉시 진입 가능 (시그널 {signal_count}개, 강한 매수 타이밍)',
                'stop_loss': '-7% 손절'
            }

        # 시그널 1개 = 패턴별 전략 + 시그널 참고
        if signal_count == 1:
            if pattern == '전환돌파형':
                entry_point = '현재가 진입 가능 (단기 모멘텀 + 시그널 1개)'
                stop_loss = '-5% 손절'
            elif pattern == '지속매집형':
                entry_point = '현재가 또는 소폭 조정 시 진입 (시그널 1개 발생)'
                stop_loss = '-8% 손절'
            elif pattern == '조정반등형':
                entry_point = '단기 반등 진입 (시그널 확인됨)'
                stop_loss = '-7% 손절'
            else:  # 기타
                entry_point = '신중 진입 (시그널 있으나 패턴 불명확)'
                stop_loss = '-5% 손절'

            return {
                'entry_point': entry_point,
                'stop_loss': stop_loss
            }

        # 시그널 0개 = 원래 패턴별 전략
        entry_rule = self.config['entry_rules'].get(pattern, self.config['entry_rules']['기타'])
        entry_point = f"{entry_rule['condition']} ({entry_rule['description']})"

        stop_loss_pct = self.config['stop_loss_rules'].get(pattern, -5)
        stop_loss = f"{stop_loss_pct}% 손절"

        return {
            'entry_point': entry_point,
            'stop_loss': stop_loss
        }

    def generate_report(self,
                       classified_df: pd.DataFrame,
                       signals_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        통합 리포트 생성 (메인 메서드)

        Args:
            classified_df: PatternClassifier.classify_all() 결과
            signals_df: SignalDetector.detect_all_signals() 결과 (Optional)

        Returns:
            pd.DataFrame: 통합 리포트
                - stock_code: 종목코드
                - stock_name: 종목명
                - sector: 섹터
                - pattern: 패턴명
                - score: 패턴 강도 점수 (0~100)
                - recent, momentum, weighted, average: 4가지 정렬 키
                - signal_count: 시그널 개수
                - signal_list: 시그널 리스트
                - entry_point: 진입 포인트
                - stop_loss: 손절 포인트

        Example:
            >>> from src.analyzer.integrated_report import IntegratedReport
            >>> report = IntegratedReport(conn)
            >>> df_report = report.generate_report(classified_df, signals_df)
        """
        # 빈 DataFrame 처리
        if classified_df.empty:
            return pd.DataFrame(columns=[
                'stock_code', 'stock_name', 'sector', 'pattern', 'score',
                'signal_count', 'signal_list', 'entry_point', 'stop_loss'
            ])

        # 1. 종목 정보 추가
        stock_codes = classified_df['stock_code'].astype(str).tolist()
        df_stocks = self._load_stock_info(stock_codes)

        # stock_code 타입 통일 (문자열)
        df_stocks['stock_code'] = df_stocks['stock_code'].astype(str)
        classified_df['stock_code'] = classified_df['stock_code'].astype(str)

        df_report = classified_df.merge(df_stocks, on='stock_code', how='left')

        # 2. 시그널 통합 (Optional)
        if signals_df is not None and not signals_df.empty:
            signals_df['stock_code'] = signals_df['stock_code'].astype(str)
            df_report = df_report.merge(
                signals_df[['stock_code', 'signal_count', 'signal_list']],
                on='stock_code',
                how='left'
            )

            # NaN 처리
            df_report['signal_count'] = df_report['signal_count'].fillna(0).astype(int)
            df_report['signal_list'] = df_report['signal_list'].fillna('').apply(
                lambda x: x if isinstance(x, list) else []
            )
        else:
            df_report['signal_count'] = 0
            df_report['signal_list'] = [[] for _ in range(len(df_report))]

        # 3. 진입/청산 포인트 생성
        entry_stop = df_report.apply(self.generate_entry_stop_recommendation, axis=1)
        df_report['entry_point'] = entry_stop.apply(lambda x: x['entry_point'])
        df_report['stop_loss'] = entry_stop.apply(lambda x: x['stop_loss'])

        # 4. 컬럼 순서 정리
        base_cols = ['stock_code', 'stock_name', 'sector']
        pattern_cols = ['pattern', 'score']
        sort_key_cols = ['recent', 'momentum', 'weighted', 'average']
        signal_cols = ['signal_count', 'signal_list']
        action_cols = ['entry_point', 'stop_loss']

        output_cols = []
        for col in (base_cols + pattern_cols + sort_key_cols + signal_cols + action_cols):
            if col in df_report.columns:
                output_cols.append(col)

        # 5. 점수 내림차순 정렬
        df_report = df_report.sort_values('score', ascending=False)

        return df_report[output_cols]

    def filter_report(self,
                     report_df: pd.DataFrame,
                     pattern: Optional[str] = None,
                     sector: Optional[str] = None,
                     min_score: Optional[float] = None,
                     min_signal_count: Optional[int] = None,
                     top_n: Optional[int] = None) -> pd.DataFrame:
        """
        리포트 필터링

        Args:
            report_df: generate_report() 결과
            pattern: 패턴 필터 (None이면 전체)
            sector: 섹터 필터 (None이면 전체)
            min_score: 최소 점수 (None이면 전체)
            min_signal_count: 최소 시그널 개수 (None이면 전체)
            top_n: 상위 N개 (None이면 전체)

        Returns:
            pd.DataFrame: 필터링된 리포트

        Example:
            >>> # 전환돌파형 + 점수 70점 이상 + 시그널 2개 이상, 상위 10개
            >>> df_filtered = report.filter_report(
            ...     report_df,
            ...     pattern='전환돌파형',
            ...     min_score=70,
            ...     min_signal_count=2,
            ...     top_n=10
            ... )
        """
        df = report_df.copy()

        # 패턴 필터
        if pattern:
            df = df[df['pattern'] == pattern]

        # 섹터 필터
        if sector and 'sector' in df.columns:
            df = df[df['sector'] == sector]

        # 점수 필터
        if min_score is not None:
            df = df[df['score'] >= min_score]

        # 시그널 필터
        if min_signal_count is not None:
            df = df[df['signal_count'] >= min_signal_count]

        # 상위 N개
        if top_n is not None:
            df = df.head(top_n)

        return df

    def get_pattern_summary_report(self, report_df: pd.DataFrame) -> pd.DataFrame:
        """
        패턴별 요약 리포트

        Args:
            report_df: generate_report() 결과

        Returns:
            pd.DataFrame: 패턴별 통계
                - pattern: 패턴명
                - count: 종목 수
                - avg_score: 평균 점수
                - avg_signal_count: 평균 시그널 개수
                - top_sector: 최다 섹터

        Example:
            >>> summary = report.get_pattern_summary_report(report_df)
        """
        summary = []

        for pattern in report_df['pattern'].unique():
            df_pattern = report_df[report_df['pattern'] == pattern]

            # 최다 섹터
            if 'sector' in df_pattern.columns:
                top_sector = df_pattern['sector'].value_counts().index[0] if len(df_pattern) > 0 else 'N/A'
            else:
                top_sector = 'N/A'

            summary.append({
                'pattern': pattern,
                'count': len(df_pattern),
                'avg_score': df_pattern['score'].mean(),
                'avg_signal_count': df_pattern['signal_count'].mean(),
                'top_sector': top_sector
            })

        df_summary = pd.DataFrame(summary)

        # 종목 수 내림차순 정렬
        df_summary = df_summary.sort_values('count', ascending=False)

        return df_summary

    def print_summary_card(self, report_df: pd.DataFrame, top_n: int = 10) -> None:
        """
        종목별 요약 카드 출력 (콘솔)

        Args:
            report_df: generate_report() 결과
            top_n: 출력할 종목 수 (기본: 10개)

        Example:
            >>> report.print_summary_card(report_df, top_n=5)

            ========================================
            [1] 232140 와이씨 (전환돌파형, 점수: 85)
            ========================================
            섹터: 전기전자
            정렬 키: Recent=0.91, Momentum=1.70, Weighted=0.52, Average=0.32
            시그널: MA크로스, 가속도 1.8배 (2개)
            진입: 현재가 진입 가능 (단기 추격 매수, 모멘텀 확인 후 진입)
            손절: -5% 손절
        """
        df_top = report_df.head(top_n)

        print("\n" + "="*80)
        print(f"📊 종목별 요약 카드 (상위 {top_n}개)")
        print("="*80 + "\n")

        for idx, (_, row) in enumerate(df_top.iterrows(), 1):
            print(f"{'='*80}")
            print(f"[{idx}] {row['stock_code']} {row.get('stock_name', 'N/A')} "
                  f"({row['pattern']}, 점수: {row['score']:.0f})")
            print(f"{'='*80}")

            if 'sector' in row:
                print(f"섹터: {row['sector']}")

            # 정렬 키
            if all(k in row for k in ['recent', 'momentum', 'weighted', 'average']):
                print(f"정렬 키: Recent={row['recent']:.2f}, Momentum={row['momentum']:.2f}, "
                      f"Weighted={row['weighted']:.2f}, Average={row['average']:.2f}")

            # 시그널
            signals = row.get('signal_list', [])
            signal_count = row.get('signal_count', 0)
            if signals:
                print(f"시그널: {', '.join(signals)} ({signal_count}개)")
            else:
                print(f"시그널: 없음")

            # 진입/청산
            print(f"진입: {row.get('entry_point', 'N/A')}")
            print(f"손절: {row.get('stop_loss', 'N/A')}")
            print()

        print("="*80 + "\n")

    def export_to_csv(self,
                     report_df: pd.DataFrame,
                     output_path: str,
                     include_all_columns: bool = False) -> None:
        """
        리포트를 CSV로 저장

        Args:
            report_df: generate_report() 결과
            output_path: 저장 경로 (예: 'output/integrated_report.csv')
            include_all_columns: 모든 컬럼 포함 여부 (기본: False, 핵심 컬럼만)

        Example:
            >>> report.export_to_csv(report_df, 'output/integrated_report.csv')
        """
        if include_all_columns:
            df_export = report_df
        else:
            # 핵심 컬럼만 선택
            core_cols = ['stock_code', 'stock_name', 'sector', 'pattern', 'score',
                        'recent', 'momentum', 'weighted', 'average',
                        'signal_count', 'entry_point', 'stop_loss']

            # 존재하는 컬럼만 선택
            export_cols = [col for col in core_cols if col in report_df.columns]
            df_export = report_df[export_cols]

        # CSV 저장
        df_export.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"[INFO] Report saved to {output_path} ({len(df_export)} rows)")

    def get_watchlist(self,
                     report_df: pd.DataFrame,
                     min_score: float = 70,
                     min_signal_count: int = 2) -> Dict[str, pd.DataFrame]:
        """
        패턴별 관심 종목 리스트 추출

        Args:
            report_df: generate_report() 결과
            min_score: 최소 점수 (기본: 70점)
            min_signal_count: 최소 시그널 개수 (기본: 2개)

        Returns:
            dict: 패턴별 관심 종목
                {
                    '전환돌파형': DataFrame (고점수 + 강시그널),
                    '지속매집형': DataFrame,
                    '조정반등형': DataFrame
                }

        Example:
            >>> watchlist = report.get_watchlist(report_df, min_score=75, min_signal_count=2)
            >>> print(watchlist['전환돌파형'][['stock_name', 'score', 'signal_count']])
        """
        patterns = ['전환돌파형', '지속매집형', '조정반등형']
        watchlist = {}

        for pattern in patterns:
            df_filtered = self.filter_report(
                report_df,
                pattern=pattern,
                min_score=min_score,
                min_signal_count=min_signal_count
            )
            watchlist[pattern] = df_filtered

        return watchlist
