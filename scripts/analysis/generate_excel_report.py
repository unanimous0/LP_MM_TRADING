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

    # 열 너비 자동 조정
    for col_idx, column in enumerate(ws.columns, 1):
        max_length = 0
        column_letter = chr(64 + col_idx)  # A, B, C, ...
        for cell in column:
            try:
                # MergedCell 스킵
                if hasattr(cell, 'value') and cell.value is not None:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
            except:
                pass
        adjusted_width = min(max_length + 2, 50) if max_length > 0 else 12
        ws.column_dimensions[column_letter].width = adjusted_width

    # 숫자 정렬 (점수 관련 컬럼)
    for row in ws.iter_rows(min_row=start_row+1):
        for cell in row:
            if isinstance(cell.value, (int, float)):
                cell.alignment = Alignment(horizontal='right')


def create_excel_report(csv_path: str, output_path: str, signal_bonus: int = 5):
    """
    엑셀 리포트 생성

    Args:
        csv_path: 입력 CSV 파일 경로
        output_path: 출력 엑셀 파일 경로
        signal_bonus: 시그널 1개당 보너스 점수
    """
    # CSV 읽기
    df = pd.read_csv(csv_path, encoding='utf-8-sig')

    # 종목코드 앞에 'A' 붙이기 (엑셀에서 0으로 시작하는 코드 보호)
    df['stock_code'] = 'A' + df['stock_code'].astype(str)

    # 종합 점수 계산
    df = calculate_combined_score(df, signal_bonus)

    print(f"[INFO] Loaded {len(df)} stocks")
    print(f"[INFO] Stock codes prefixed with 'A' to preserve leading zeros in Excel")
    print(f"[INFO] Combined score formula: 원래점수 + (시그널 × {signal_bonus}점)")

    # 엑셀 파일 생성
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:

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
        patterns = ['전환돌파형', '지속매집형', '조정반등형']
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
    print(f"  1. 최종결론 - 종합 추천 순위 TOP 20")
    print(f"  2. 점수순위 - 종목의 질 TOP 30")
    print(f"  3. 시그널순위 - 진입 타이밍 TOP 30")
    print(f"  4. 패턴별순위 - 패턴별 TOP 10")
    print(f"  5. 섹터별상위 - 섹터별 TOP 3")
    print(f"  6. 전체데이터 - 전체 {len(df_all)}개 종목")


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
