#!/usr/bin/env python3
"""
엑셀 리포트 생성기

종합 추천 순위와 다양한 기준별 순위를 엑셀 파일로 생성합니다.
"""

import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path

try:
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("[WARN] openpyxl not installed. Excel formatting will be basic.")


def calculate_combined_score(df: pd.DataFrame, signal_bonus: int = 5) -> pd.DataFrame:
    """
    종합 점수 계산 (A안: 단순 보너스 방식)

    종합점수 = 원래 점수 + (시그널 개수 × 5점)

    Args:
        df: 데이터프레임
        signal_bonus: 시그널 1개당 보너스 점수 (기본 5점)

    Returns:
        종합점수가 추가된 데이터프레임
    """
    df = df.copy()
    df['combined_score'] = df['score'] + (df['signal_count'] * signal_bonus)
    return df


def format_excel_sheet(ws, df, title=None):
    """엑셀 시트 포맷팅"""
    if not HAS_OPENPYXL:
        return

    # 헤더 스타일
    header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
    header_font = Font(bold=True, color="FFFFFF", size=11)

    # 제목 추가 (있는 경우)
    if title:
        ws.insert_rows(1)
        ws['A1'] = title
        ws['A1'].font = Font(bold=True, size=14)
        ws.merge_cells(f'A1:{chr(64 + len(df.columns))}1')
        start_row = 2
    else:
        start_row = 1

    # 헤더 포맷팅
    for cell in ws[start_row]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal='center', vertical='center')

    # 열 너비 자동 조정 (한글 고려)
    def get_text_width(text):
        """텍스트 너비 계산 (한글은 2배, 영문은 1배)"""
        if text is None:
            return 0
        width = 0
        for char in str(text):
            # 한글, 한자, 일본어 등 (유니코드 범위)
            if '\uac00' <= char <= '\ud7a3' or '\u4e00' <= char <= '\u9fff':
                width += 2  # 한글/한자는 2배 너비
            else:
                width += 1  # 영문/숫자는 1배 너비
        return width

    for col_idx, column in enumerate(ws.columns, 1):
        max_width = 0
        column_letter = chr(64 + col_idx)  # A, B, C, ...
        for cell in column:
            try:
                # 제목 행(병합된 셀) 스킵 - 헤더 행부터 계산
                if title and cell.row < start_row:
                    continue

                # MergedCell 스킵
                if hasattr(cell, 'value') and cell.value is not None:
                    cell_width = get_text_width(cell.value)
                    if cell_width > max_width:
                        max_width = cell_width
            except:
                pass
        # 여유 공간 추가 (+3) 및 최대 너비 제한 (70)
        adjusted_width = min(max_width + 3, 70) if max_width > 0 else 12
        ws.column_dimensions[column_letter].width = adjusted_width

    # 숫자 포맷팅 및 정렬
    for row_idx, row in enumerate(ws.iter_rows(min_row=start_row+1), start=start_row+1):
        for col_idx, cell in enumerate(row, start=1):
            if isinstance(cell.value, (int, float)):
                # 오른쪽 정렬
                cell.alignment = Alignment(horizontal='right')

                # 컬럼명 가져오기
                header_cell = ws.cell(row=start_row, column=col_idx)
                header = str(header_cell.value) if header_cell.value else ''

                # 정수로 표시할 컬럼 (순위, 시그널, 종목수 등)
                integer_columns = ['순위', '시그널', '종목수', '종목 수']

                if any(int_col in header for int_col in integer_columns):
                    # 정수 포맷 (소수점 없음)
                    cell.number_format = '0'
                else:
                    # 실수 포맷 (소수점 둘째자리)
                    cell.number_format = '0.00'


def create_glossary_sheet(writer, signal_bonus: int):
    """
    용어 설명 시트 생성 (가로 펼침 표 형식)

    Args:
        writer: ExcelWriter 객체
        signal_bonus: 시그널 보너스 점수
    """
    # 용어 설명 데이터 준비 (가로로 펼친 형식)
    glossary_data = [
        ['📊 수급 레짐 스캐너 - 용어 설명', '', '', '', ''],
        ['', '', '', '', ''],

        # 보고서 개요
        ['■ 보고서 개요', '', '', '', ''],
        ['분석 목적', '외국인/기관 투자자 수급 흐름을 정량화하여 매수 강도가 높은 종목을 발굴', '', '', ''],
        ['분석 기간', '1주(1W) ~ 2년(2Y)까지 6개 기간을 분석하여 단기/중기/장기 트렌드를 종합', '', '', ''],
        ['데이터 출처', 'KOSPI200 + KOSDAQ150 총 345개 핵심 종목 (2024-01-02 ~ 2026-01-20)', '', '', ''],
        ['', '', '', '', ''],

        # 핵심 지표 (가로 배치)
        ['■ 핵심 지표', '', '', '', ''],
        ['', '📌 Sff (Supply Float Factor)', '', '📌 Z-Score', ''],
        ['정의', 'Sff = (순매수 금액 / 유통시총) × 100', '', 'Z-Score = (현재값 - 60일 평균) / 60일 표준편차', ''],
        ['의미', '시가총액 왜곡 제거, 유통물량 대비 매수 강도 정규화', '', '변동성 보정하여 이상 수급(평소와 다른 매수/매도) 탐지', ''],
        ['해석', '값이 클수록 유통물량 대비 순매수 금액이 큼', '', '|Z| > 2.0: 이상 수급, +값: 강한 매수, -값: 강한 매도', ''],
        ['', '', '', '', ''],

        # 패턴 분류 (가로 3열 배치)
        ['■ 패턴 분류 (3가지 유형)', '', '', '', ''],
        ['', '🔥 모멘텀형', '📈 지속형', '🔄 전환형', ''],
        ['특징', '단기 모멘텀이 매우 강한 종목', '장기간 일관된 상승 추세 종목', '과거 강했으나 최근 약화 → 전환 대기', ''],
        ['조건', '1W-2Y > 1.0 AND (1W+1M)/2 > 0.5', '가중평균 > 0.8 AND 양수 기간 > 70%', '가중평균 > 0.5 AND 1W-2Y < 0', ''],
        ['투자 스타일', '단기 트레이딩, 돌파 매매', '중장기 추세 추종, 포지션 트레이딩', '저가 매수 기회 포착, 역추세 매매', ''],
        ['위험도', '높음 (변동성 큼, 손절 엄격 필요)', '중간 (안정적 상승, 장기 보유 가능)', '높음 (추세 전환 실패 가능성, 신중 진입)', ''],
        ['', '', '', '', ''],

        # 시그널 (가로 3열 배치)
        ['■ 시그널 (3가지 타이밍 지표)', '', '', '', ''],
        ['', '✅ MA 골든크로스', '⚡ 수급 가속도', '🤝 외인-기관 동조율', ''],
        ['정의', '외국인 5일MA > 20일MA 돌파', '(최근5일 평균) / (직전5일 평균) > 1.5배', '최근 20일 중 동시 매수 비율 > 50%', ''],
        ['의미', '단기 수급이 장기 추세 상향 돌파 → 매수', '수급 강도 급증 → 모멘텀 가속', '두 투자 주체 동시 매수 → 확신도 높음', ''],
        ['', '', '', '', ''],

        # 점수 메트릭 (가로 4열 배치)
        ['■ 점수 메트릭 (4가지 정렬 기준)', '', '', '', ''],
        ['', '📍 Recent (현재 강도)', '📍 Momentum (개선도)', '📍 Weighted (가중 평균)', '📍 Average (일관성)'],
        ['계산식', '(1W + 1M) / 2', '1W - 2Y', '1W×0.30 + 1M×0.25 + 3M×0.20 + 6M×0.15 + 1Y×0.07 + 2Y×0.03', '(1W + 1M + 3M + 6M + 1Y + 2Y) / 6'],
        ['의미', '최근 1주~1개월 수급 강도 평균', '단기 vs 장기 수급 격차 → 전환점', '최근에 높은 가중치 부여한 중장기 트렌드', '전 기간 단순 평균 → 일관된 수급'],
        ['활용', '현재 진행형 매수세 파악', '과거 대비 수급 개선 여부 판단', '중장기 추세 방향 판단', '전 기간 고르게 강한 종목 발굴'],
        ['', '', '', '', ''],

        # 종합점수 & 추천 기준
        ['■ 종합점수 & 추천 기준', '', '', '', ''],
        ['종합점수', f'패턴 점수 + (시그널 개수 × {signal_bonus}점)', '', '', ''],
        ['패턴 점수', 'Recent×25% + Momentum×25% + Weighted×30% + Average×20% (0~100점)', '', '', ''],
        ['시그널 보너스', f'시그널 1개당 +{signal_bonus}점 (최대 {signal_bonus*3}점)', '', '', ''],
        ['', '', '', '', ''],
        ['추천 등급', '⭐⭐⭐ 강력 추천', '⭐⭐ 추천', '⭐ 관심', ''],
        ['기준', '종합점수 80+ AND 시그널 2개 이상', '종합점수 70+ AND 시그널 1개 이상', '종합점수 60+ OR 시그널 2개 이상', ''],
        ['', '', '', '', ''],

        # 진입/청산 기준
        ['■ 진입/청산 기준', '', '', '', ''],
        ['진입 전략', '시그널 발생 시점에서 당일 종가 또는 익일 시초가 매수', '', '', ''],
        ['손절 기준', '진입가 대비 -7% 도달 시 무조건 청산', '', '', ''],
        ['목표 수익률', '+15% 달성 시 50% 익절, +25% 달성 시 전량 청산', '', '', ''],
        ['최대 보유 기간', '30일 경과 시 수익/손실 무관 전량 청산', '', '', ''],
        ['', '', '', '', ''],
        ['', '', '', '', ''],
        ['📌 보고서 작성일: 2026-02-14  |  분석 시스템: 수급 레짐 스캐너 v3.1 (Stage 3 완료)', '', '', '', ''],
    ]

    # DataFrame으로 변환
    df_glossary = pd.DataFrame(glossary_data)

    # 시트에 쓰기 (헤더 없이)
    df_glossary.to_excel(writer, sheet_name='0.용어설명', index=False, header=False)

    # 포맷팅
    if HAS_OPENPYXL:
        ws = writer.sheets['0.용어설명']
        from openpyxl.styles import Font, Alignment, PatternFill

        # 열 너비 설정 (가로로 넓게)
        ws.column_dimensions['A'].width = 22
        ws.column_dimensions['B'].width = 40
        ws.column_dimensions['C'].width = 40
        ws.column_dimensions['D'].width = 40
        ws.column_dimensions['E'].width = 40

        # 전체 셀 포맷팅
        for row in ws.iter_rows():
            for cell in row:
                cell.font = Font(name='맑은 고딕', size=9)
                cell.alignment = Alignment(vertical='top', wrap_text=False)  # 자동 줄바꿈 OFF

                # 제목 (첫 행)
                if cell.row == 1:
                    cell.font = Font(name='맑은 고딕', size=14, bold=True, color='FFFFFF')
                    cell.fill = PatternFill(start_color='1F4E78', end_color='1F4E78', fill_type='solid')
                    cell.alignment = Alignment(horizontal='left', vertical='center')

                # 섹션 헤더 (■ 포함)
                if cell.value and '■' in str(cell.value):
                    cell.font = Font(name='맑은 고딕', size=11, bold=True, color='FFFFFF')
                    cell.fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
                    cell.alignment = Alignment(horizontal='left', vertical='center')

                # 패턴/시그널 이름 (이모지 포함)
                if cell.value and any(emoji in str(cell.value) for emoji in ['🔥', '📈', '🔄', '✅', '⚡', '🤝', '📍', '📌']):
                    cell.font = Font(name='맑은 고딕', size=10, bold=True, color='C00000')

        # 첫 행 병합 (제목)
        ws.merge_cells('A1:E1')

        # 높이 조정
        ws.row_dimensions[1].height = 25


def create_excel_report(csv_path: str, output_path: str, signal_bonus: int = 5):
    """
    엑셀 리포트 생성

    Args:
        csv_path: 입력 CSV 파일 경로
        output_path: 출력 엑셀 파일 경로
        signal_bonus: 시그널 1개당 보너스 점수
    """
    # CSV 읽기 (종목코드를 문자열로 읽어 앞의 0 보존)
    df = pd.read_csv(csv_path, encoding='utf-8-sig', dtype={'stock_code': str})

    # 종목코드 6자리로 패딩 후 'A' 붙이기 (엑셀에서 0으로 시작하는 코드 보호)
    df['stock_code'] = df['stock_code'].str.zfill(6)  # 6자리로 패딩 (예: 5930 → 005930)
    df['stock_code'] = 'A' + df['stock_code']  # A 접두사 추가 (예: 005930 → A005930)

    # 종합 점수 계산
    df = calculate_combined_score(df, signal_bonus)

    print(f"[INFO] Loaded {len(df)} stocks")
    print(f"[INFO] Stock codes prefixed with 'A' to preserve leading zeros in Excel")
    print(f"[INFO] Combined score formula: 원래점수 + (시그널 × {signal_bonus}점)")

    # 엑셀 파일 생성
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

        # ========================================
        # 시트 0: 용어 설명
        # ========================================
        create_glossary_sheet(writer, signal_bonus)

        # ========================================
        # 시트 1: 최종 결론 (종합 추천 순위)
        # ========================================
        df_final = df.nlargest(20, 'combined_score')[[
            'stock_code', 'stock_name', 'sector', 'pattern',
            'score', 'signal_count', 'combined_score', 'signal_list',
            'entry_point', 'stop_loss'
        ]].copy()

        df_final.columns = ['종목코드', '종목명', '섹터', '패턴',
                           '점수', '시그널', '종합점수', '시그널내용',
                           '진입전략', '손절']
        df_final.insert(0, '순위', range(1, len(df_final) + 1))

        df_final.to_excel(writer, sheet_name='1.최종결론', index=False)
        format_excel_sheet(writer.sheets['1.최종결론'], df_final,
                          f"📊 종합 추천 순위 TOP 20 (종합점수 = 점수 + 시그널×{signal_bonus})")

        # ========================================
        # 시트 2: 점수 순위 (종목의 질)
        # ========================================
        df_score = df.nlargest(30, 'score')[[
            'stock_code', 'stock_name', 'sector', 'pattern',
            'score', 'signal_count', 'recent', 'momentum', 'weighted', 'average'
        ]].copy()

        df_score.columns = ['종목코드', '종목명', '섹터', '패턴',
                           '점수', '시그널', 'Recent', 'Momentum', 'Weighted', 'Average']
        df_score.insert(0, '순위', range(1, len(df_score) + 1))

        df_score.to_excel(writer, sheet_name='2.점수순위', index=False)
        format_excel_sheet(writer.sheets['2.점수순위'], df_score,
                          "📈 점수 순위 TOP 30 (종목의 질)")

        # ========================================
        # 시트 3: 시그널 순위 (진입 타이밍)
        # ========================================
        df_signal = df.sort_values(['signal_count', 'score'], ascending=[False, False]).head(30)[[
            'stock_code', 'stock_name', 'sector', 'pattern',
            'score', 'signal_count', 'signal_list', 'entry_point'
        ]].copy()

        df_signal.columns = ['종목코드', '종목명', '섹터', '패턴',
                            '점수', '시그널', '시그널내용', '진입전략']
        df_signal.insert(0, '순위', range(1, len(df_signal) + 1))

        df_signal.to_excel(writer, sheet_name='3.시그널순위', index=False)
        format_excel_sheet(writer.sheets['3.시그널순위'], df_signal,
                          "🚨 시그널 순위 TOP 30 (진입 타이밍)")

        # ========================================
        # 시트 4: 패턴별 순위
        # ========================================
        patterns = ['모멘텀형', '지속형', '전환형']
        df_patterns_list = []

        for pattern in patterns:
            df_p = df[df['pattern'] == pattern].nlargest(10, 'combined_score')[[
                'stock_code', 'stock_name', 'sector',
                'score', 'signal_count', 'combined_score'
            ]].copy()

            if len(df_p) > 0:
                df_p.insert(0, '패턴', pattern)
                df_p.insert(1, '순위', range(1, len(df_p) + 1))
                df_patterns_list.append(df_p)

        if df_patterns_list:
            df_patterns = pd.concat(df_patterns_list, ignore_index=True)
            df_patterns.columns = ['패턴', '순위', '종목코드', '종목명', '섹터',
                                  '점수', '시그널', '종합점수']

            df_patterns.to_excel(writer, sheet_name='4.패턴별순위', index=False)
            format_excel_sheet(writer.sheets['4.패턴별순위'], df_patterns,
                              "🎯 패턴별 TOP 10")

        # ========================================
        # 시트 5: 섹터별 상위
        # ========================================
        sectors = df.groupby('sector')['combined_score'].max().nlargest(20).index
        df_sectors_list = []

        for sector in sectors:
            df_s = df[df['sector'] == sector].nlargest(3, 'combined_score')[[
                'stock_code', 'stock_name', 'pattern',
                'score', 'signal_count', 'combined_score'
            ]].copy()

            df_s.insert(0, '섹터', sector)
            df_sectors_list.append(df_s)

        df_sectors = pd.concat(df_sectors_list, ignore_index=True)
        df_sectors.columns = ['섹터', '종목코드', '종목명', '패턴',
                             '점수', '시그널', '종합점수']

        df_sectors.to_excel(writer, sheet_name='5.섹터별상위', index=False)
        format_excel_sheet(writer.sheets['5.섹터별상위'], df_sectors,
                          "🏢 섹터별 상위 종목 (섹터당 TOP 3)")

        # ========================================
        # 시트 6: 전체 데이터
        # ========================================
        df_all = df.sort_values('combined_score', ascending=False).copy()
        df_all.insert(0, '순위', range(1, len(df_all) + 1))

        # 컬럼명 한글화
        column_mapping = {
            'stock_code': '종목코드',
            'stock_name': '종목명',
            'sector': '섹터',
            'pattern': '패턴',
            'score': '점수',
            'recent': 'Recent',
            'momentum': 'Momentum',
            'weighted': 'Weighted',
            'average': 'Average',
            'signal_count': '시그널',
            'signal_list': '시그널내용',
            'combined_score': '종합점수',
            'entry_point': '진입전략',
            'stop_loss': '손절'
        }

        df_all = df_all.rename(columns=column_mapping)

        df_all.to_excel(writer, sheet_name='6.전체데이터', index=False)
        format_excel_sheet(writer.sheets['6.전체데이터'], df_all,
                          f"📋 전체 데이터 ({len(df_all)}개 종목)")

    print(f"✅ Excel report saved: {output_path}")
    print(f"\n시트 구성:")
    print(f"  0. 용어설명 - 패턴/시그널/메트릭 정의")
    print(f"  1. 최종결론 - 종합 추천 순위 TOP 20")
    print(f"  2. 점수순위 - 종목의 질 TOP 30")
    print(f"  3. 시그널순위 - 진입 타이밍 TOP 30")
    print(f"  4. 패턴별순위 - 패턴별 TOP 10")
    print(f"  5. 섹터별상위 - 섹터별 TOP 3")
    print(f"  6. 전체데이터 - 전체 {len(df_all)}개 종목")

    # ========================================
    # CSV 파일도 함께 저장
    # ========================================
    csv_output_path = output_path.replace('.xlsx', '.csv')
    df_all.to_csv(csv_output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ CSV file saved: {csv_output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='수급 레짐 스캐너 - 엑셀 리포트 생성',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 실행 (CSV → Excel)
  python scripts/analysis/generate_excel_report.py

  # 입력/출력 파일 지정
  python scripts/analysis/generate_excel_report.py --input output/regime_report.csv --output output/report.xlsx

  # 시그널 보너스 점수 변경 (기본 5점)
  python scripts/analysis/generate_excel_report.py --signal-bonus 10
        """
    )

    parser.add_argument(
        '--input', '-i',
        default='output/regime_report.csv',
        help='입력 CSV 파일 (기본: output/regime_report.csv)'
    )

    parser.add_argument(
        '--output', '-o',
        default=None,
        help='출력 엑셀 파일 (기본: output/regime_report_YYYYMMDD_HHMMSS.xlsx)'
    )

    parser.add_argument(
        '--signal-bonus',
        type=int,
        default=5,
        help='시그널 1개당 보너스 점수 (기본: 5)'
    )

    args = parser.parse_args()

    # 입력 파일 확인
    if not Path(args.input).exists():
        print(f"[ERROR] Input file not found: {args.input}")
        print("\n먼저 regime_scanner.py를 실행하여 CSV 파일을 생성하세요:")
        print("  python scripts/analysis/regime_scanner.py --save-csv")
        return 1

    # 출력 파일명 생성
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'output/regime_report_{timestamp}.xlsx'

    # 디렉토리 생성
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # 엑셀 리포트 생성
    print("="*80)
    print("📊 엑셀 리포트 생성")
    print("="*80)
    print(f"입력: {args.input}")
    print(f"출력: {args.output}")
    print(f"시그널 보너스: {args.signal_bonus}점")
    print()

    create_excel_report(args.input, args.output, args.signal_bonus)

    print()
    print("="*80)
    print("✅ 완료!")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
