#!/usr/bin/env python3
"""
HTML 리포트 생성기 (인터랙티브 대시보드)

Tailwind CSS + Chart.js를 사용한 웹 기반 분석 리포트
"""

import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
import json


def calculate_combined_score(df: pd.DataFrame, signal_bonus: int = 5) -> pd.DataFrame:
    """종합 점수 계산"""
    df = df.copy()
    df['combined_score'] = df['score'] + (df['signal_count'] * signal_bonus)
    return df


def generate_html_report(csv_path: str, output_path: str, signal_bonus: int = 5):
    """HTML 리포트 생성"""

    # CSV 읽기
    df = pd.read_csv(csv_path, encoding='utf-8-sig', dtype={'stock_code': str})
    df['stock_code'] = df['stock_code'].str.zfill(6)

    # 종합 점수 계산
    df = calculate_combined_score(df, signal_bonus)

    print(f"[INFO] Loaded {len(df)} stocks")

    # 패턴별 통계
    pattern_stats = df.groupby('pattern').agg({
        'stock_code': 'count',
        'combined_score': 'mean'
    }).reset_index()
    pattern_stats.columns = ['pattern', 'count', 'avg_score']

    # 섹터별 통계 (TOP 10)
    sector_stats = df.groupby('sector').agg({
        'combined_score': 'mean',
        'stock_code': 'count'
    }).reset_index()
    sector_stats.columns = ['sector', 'avg_score', 'count']
    sector_stats = sector_stats.nlargest(10, 'avg_score')

    # 최종 결론 TOP 20
    df_final = df.nlargest(20, 'combined_score')[[
        'stock_code', 'stock_name', 'sector', 'pattern',
        'score', 'signal_count', 'combined_score', 'signal_list',
        'entry_point', 'stop_loss'
    ]].copy()

    # 섹터별 수급 집중도
    high_score_counts = df[df['combined_score'] >= 70].groupby('sector').size().reset_index(name='high_score_count')
    sector_concentration = df.groupby('sector').agg({
        'combined_score': 'mean',
        'stock_code': 'count'
    }).reset_index()
    sector_concentration.columns = ['sector', 'avg_score', 'total_count']
    sector_concentration = sector_concentration.merge(high_score_counts, on='sector', how='left')
    sector_concentration['high_score_count'] = sector_concentration['high_score_count'].fillna(0).astype(int)
    sector_concentration['sector_score'] = sector_concentration['avg_score'] * (
        1 + sector_concentration['high_score_count'] / sector_concentration['total_count']
    )
    sector_concentration = sector_concentration.nlargest(10, 'sector_score')

    # HTML 생성
    html = generate_html_template(
        df_final=df_final,
        pattern_stats=pattern_stats,
        sector_stats=sector_stats,
        sector_concentration=sector_concentration,
        total_stocks=len(df),
        signal_bonus=signal_bonus
    )

    # 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ HTML report saved: {output_path}")


def generate_html_template(df_final, pattern_stats, sector_stats, sector_concentration, total_stocks, signal_bonus):
    """HTML 템플릿 생성"""

    # 데이터를 JSON으로 변환
    pattern_labels = pattern_stats['pattern'].tolist()
    pattern_values = pattern_stats['count'].tolist()

    sector_labels = sector_stats['sector'].tolist()
    sector_values = sector_stats['avg_score'].tolist()

    # 최종 결론 테이블 HTML
    final_table_rows = ""
    for idx, row in df_final.iterrows():
        signal_list = row['signal_list'] if pd.notna(row['signal_list']) else '-'
        pattern_color = {
            '모멘텀형': 'bg-red-100 text-red-800',
            '지속형': 'bg-blue-100 text-blue-800',
            '전환형': 'bg-yellow-100 text-yellow-800'
        }.get(row['pattern'], 'bg-gray-100 text-gray-800')

        final_table_rows += f"""
        <tr class="hover:bg-gray-50">
            <td class="px-4 py-3 text-sm text-gray-900 font-medium">{row['stock_code']}</td>
            <td class="px-4 py-3 text-sm text-gray-900 font-semibold">{row['stock_name']}</td>
            <td class="px-4 py-3 text-sm text-gray-600">{row['sector']}</td>
            <td class="px-4 py-3">
                <span class="px-2 py-1 text-xs font-semibold rounded-full {pattern_color}">
                    {row['pattern']}
                </span>
            </td>
            <td class="px-4 py-3 text-sm text-gray-900 text-right">{row['score']:.1f}</td>
            <td class="px-4 py-3 text-sm text-gray-900 text-center">{int(row['signal_count'])}</td>
            <td class="px-4 py-3 text-sm text-blue-600 font-semibold text-right">{row['combined_score']:.1f}</td>
            <td class="px-4 py-3 text-sm text-gray-600">{signal_list}</td>
        </tr>
        """

    # 섹터별 수급 집중도 테이블
    sector_conc_rows = ""
    for idx, row in sector_concentration.iterrows():
        sector_conc_rows += f"""
        <tr class="hover:bg-gray-50">
            <td class="px-4 py-3 text-sm text-gray-900 font-medium">{row['sector']}</td>
            <td class="px-4 py-3 text-sm text-gray-900 text-right">{row['avg_score']:.1f}</td>
            <td class="px-4 py-3 text-sm text-gray-600 text-center">{int(row['total_count'])}</td>
            <td class="px-4 py-3 text-sm text-green-600 font-semibold text-center">{int(row['high_score_count'])}</td>
            <td class="px-4 py-3 text-sm text-blue-600 font-bold text-right">{row['sector_score']:.1f}</td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>수급 레짐 스캐너 - 분석 리포트</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-gray-50">
    <!-- 헤더 -->
    <header class="bg-gradient-to-r from-blue-600 to-blue-800 text-white shadow-lg">
        <div class="max-w-7xl mx-auto px-4 py-6">
            <h1 class="text-3xl font-bold">📊 수급 레짐 스캐너</h1>
            <p class="text-blue-100 mt-2">외국인/기관 투자자 수급 분석 리포트</p>
            <p class="text-blue-200 text-sm mt-1">생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 py-8">
        <!-- 요약 카드 -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="bg-white rounded-lg shadow p-6">
                <div class="text-sm text-gray-500 mb-1">전체 종목</div>
                <div class="text-3xl font-bold text-gray-900">{total_stocks}</div>
            </div>
            <div class="bg-white rounded-lg shadow p-6">
                <div class="text-sm text-gray-500 mb-1">모멘텀형</div>
                <div class="text-3xl font-bold text-red-600">{pattern_stats[pattern_stats['pattern']=='모멘텀형']['count'].values[0] if len(pattern_stats[pattern_stats['pattern']=='모멘텀형']) > 0 else 0}</div>
            </div>
            <div class="bg-white rounded-lg shadow p-6">
                <div class="text-sm text-gray-500 mb-1">지속형</div>
                <div class="text-3xl font-bold text-blue-600">{pattern_stats[pattern_stats['pattern']=='지속형']['count'].values[0] if len(pattern_stats[pattern_stats['pattern']=='지속형']) > 0 else 0}</div>
            </div>
            <div class="bg-white rounded-lg shadow p-6">
                <div class="text-sm text-gray-500 mb-1">전환형</div>
                <div class="text-3xl font-bold text-yellow-600">{pattern_stats[pattern_stats['pattern']=='전환형']['count'].values[0] if len(pattern_stats[pattern_stats['pattern']=='전환형']) > 0 else 0}</div>
            </div>
        </div>

        <!-- 차트 섹션 -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <!-- 패턴별 분포 -->
            <div class="bg-white rounded-lg shadow p-6">
                <h2 class="text-xl font-bold text-gray-900 mb-4">패턴별 분포</h2>
                <canvas id="patternChart"></canvas>
            </div>

            <!-- 섹터별 평균 점수 -->
            <div class="bg-white rounded-lg shadow p-6">
                <h2 class="text-xl font-bold text-gray-900 mb-4">섹터별 평균 점수 (TOP 10)</h2>
                <canvas id="sectorChart"></canvas>
            </div>
        </div>

        <!-- 최종 결론 테이블 -->
        <div class="bg-white rounded-lg shadow mb-8">
            <div class="px-6 py-4 border-b border-gray-200">
                <h2 class="text-xl font-bold text-gray-900">🎯 종합 추천 순위 TOP 20</h2>
                <p class="text-sm text-gray-600 mt-1">종합점수 = 패턴 점수 + (시그널 개수 × {signal_bonus}점)</p>
            </div>
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">종목코드</th>
                            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">종목명</th>
                            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">섹터</th>
                            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">패턴</th>
                            <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">점수</th>
                            <th class="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">시그널</th>
                            <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">종합점수</th>
                            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">시그널내용</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        {final_table_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 섹터별 수급 집중도 -->
        <div class="bg-white rounded-lg shadow">
            <div class="px-6 py-4 border-b border-gray-200">
                <h2 class="text-xl font-bold text-gray-900">🔥 섹터별 수급 집중도 (TOP 10)</h2>
                <p class="text-sm text-gray-600 mt-1">섹터점수 = 평균점수 × (1 + 고득점종목수/전체종목수)</p>
            </div>
            <div class="overflow-x-auto">
                <table class="min-w-full divide-y divide-gray-200">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">섹터</th>
                            <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">평균 점수</th>
                            <th class="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">종목 수</th>
                            <th class="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">고득점 종목</th>
                            <th class="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">섹터 점수</th>
                        </tr>
                    </thead>
                    <tbody class="bg-white divide-y divide-gray-200">
                        {sector_conc_rows}
                    </tbody>
                </table>
            </div>
        </div>
    </main>

    <!-- 푸터 -->
    <footer class="bg-gray-800 text-white mt-12">
        <div class="max-w-7xl mx-auto px-4 py-6 text-center">
            <p class="text-gray-400">수급 레짐 스캐너 v3.1 | Stage 3 완료</p>
            <p class="text-gray-500 text-sm mt-2">KOSPI200 + KOSDAQ150 (345개 종목) | 2024-01-02 ~ 2026-01-20</p>
        </div>
    </footer>

    <!-- Chart.js 스크립트 -->
    <script>
        // 패턴별 분포 차트
        const patternCtx = document.getElementById('patternChart').getContext('2d');
        new Chart(patternCtx, {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(pattern_labels)},
                datasets: [{{
                    data: {json.dumps(pattern_values)},
                    backgroundColor: ['#EF4444', '#3B82F6', '#F59E0B'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});

        // 섹터별 평균 점수 차트
        const sectorCtx = document.getElementById('sectorChart').getContext('2d');
        new Chart(sectorCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(sector_labels)},
                datasets: [{{
                    label: '평균 점수',
                    data: {json.dumps(sector_values)},
                    backgroundColor: '#3B82F6',
                    borderColor: '#2563EB',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                indexAxis: 'y',
                plugins: {{
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    x: {{
                        beginAtZero: true,
                        max: 100
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description='수급 레짐 스캐너 - HTML 리포트 생성',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--input', '-i', default='output/regime_report.csv')
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--signal-bonus', type=int, default=5)

    args = parser.parse_args()

    # 입력 파일 확인
    if not Path(args.input).exists():
        print(f"[ERROR] Input file not found: {args.input}")
        return 1

    # 출력 파일명 생성
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f'output/regime_report_{timestamp}.html'

    # 디렉토리 생성
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("📊 HTML 리포트 생성")
    print("="*80)
    print(f"입력: {args.input}")
    print(f"출력: {args.output}")
    print()

    generate_html_report(args.input, args.output, args.signal_bonus)

    print()
    print("="*80)
    print("✅ 완료!")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
