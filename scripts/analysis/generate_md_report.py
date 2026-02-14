#!/usr/bin/env python3
"""
Markdown 리포트 생성기

GitHub/Obsidian 등에서 읽기 좋은 Markdown 형식 리포트
"""

import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path


def calculate_combined_score(df: pd.DataFrame, signal_bonus: int = 5) -> pd.DataFrame:
    """종합 점수 계산"""
    df = df.copy()
    df['combined_score'] = df['score'] + (df['signal_count'] * signal_bonus)
    return df


def generate_md_report(csv_path: str, output_path: str, signal_bonus: int = 5):
    """Markdown 리포트 생성"""

    # CSV 읽기
    df = pd.read_csv(csv_path, encoding='utf-8-sig', dtype={'stock_code': str})
    df['stock_code'] = df['stock_code'].str.zfill(6)

    # 종합 점수 계산
    df = calculate_combined_score(df, signal_bonus)

    print(f"[INFO] Loaded {len(df)} stocks")

    # Markdown 생성
    md = generate_markdown_content(df, signal_bonus)

    # 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)

    print(f"✅ Markdown report saved: {output_path}")


def generate_markdown_content(df, signal_bonus):
    """Markdown 콘텐츠 생성"""

    # 패턴별 통계
    pattern_stats = df.groupby('pattern').agg({
        'stock_code': 'count',
        'combined_score': 'mean',
        'signal_count': 'mean'
    }).reset_index()

    # 최종 결론 TOP 20
    df_final = df.nlargest(20, 'combined_score')

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

    md = f"""# 📊 수급 레짐 스캐너 - 분석 리포트

**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**분석 시스템**: 수급 레짐 스캐너 v3.1 (Stage 3 완료)
**데이터**: KOSPI200 + KOSDAQ150 (345개 종목) | 2024-01-02 ~ 2026-01-20

---

## 📈 요약 통계

| 항목 | 값 |
|------|-----|
| 전체 종목 수 | {len(df):,}개 |
| 모멘텀형 | {len(df[df['pattern']=='모멘텀형']):,}개 |
| 지속형 | {len(df[df['pattern']=='지속형']):,}개 |
| 전환형 | {len(df[df['pattern']=='전환형']):,}개 |
| 평균 종합점수 | {df['combined_score'].mean():.1f}점 |
| 시그널 발생 종목 | {len(df[df['signal_count'] > 0]):,}개 |

---

## 🎯 종합 추천 순위 TOP 20

> **종합점수** = 패턴 점수 + (시그널 개수 × {signal_bonus}점)

| 순위 | 종목코드 | 종목명 | 섹터 | 패턴 | 점수 | 시그널 | 종합점수 | 시그널내용 |
|:----:|:--------:|--------|------|:----:|-----:|:------:|--------:|-----------|
"""

    for idx, (_, row) in enumerate(df_final.iterrows(), 1):
        pattern_emoji = {'모멘텀형': '🔥', '지속형': '📈', '전환형': '🔄'}.get(row['pattern'], '❓')
        signal_list = row['signal_list'] if pd.notna(row['signal_list']) else '-'

        md += f"| {idx} | `{row['stock_code']}` | **{row['stock_name']}** | {row['sector']} | {pattern_emoji} {row['pattern']} | {row['score']:.1f} | {int(row['signal_count'])} | **{row['combined_score']:.1f}** | {signal_list} |\n"

    md += f"""
---

## 📊 패턴별 통계

| 패턴 | 종목 수 | 평균 종합점수 | 평균 시그널 |
|------|--------:|--------------:|------------:|
"""

    for _, row in pattern_stats.iterrows():
        pattern_emoji = {'모멘텀형': '🔥', '지속형': '📈', '전환형': '🔄'}.get(row['pattern'], '❓')
        md += f"| {pattern_emoji} {row['pattern']} | {int(row['stock_code']):,}개 | {row['combined_score']:.1f}점 | {row['signal_count']:.2f}개 |\n"

    md += f"""
### 패턴별 특징

#### 🔥 모멘텀형 (Momentum Pattern)
- **특징**: 단기 모멘텀이 매우 강한 종목 (1주일 전환점 포착)
- **조건**: 1W-2Y > 1.0 AND (1W+1M)/2 > 0.5
- **투자 스타일**: 단기 트레이딩, 돌파 매매
- **위험도**: ⚠️ 높음 (변동성 큼, 손절 엄격 필요)

#### 📈 지속형 (Sustained Pattern)
- **특징**: 장기간 일관된 상승 추세를 보이는 종목
- **조건**: 가중평균 > 0.8 AND 양수 기간 비율 > 70%
- **투자 스타일**: 중장기 추세 추종, 포지션 트레이딩
- **위험도**: ✅ 중간 (안정적 상승, 장기 보유 가능)

#### 🔄 전환형 (Reversal Pattern)
- **특징**: 과거 강했으나 최근 약화 → 반대 방향 전환 대기
- **조건**: 가중평균 > 0.5 AND 1W-2Y < 0
- **투자 스타일**: 저가 매수 기회 포착, 역추세 매매
- **위험도**: ⚠️ 높음 (추세 전환 실패 가능성, 신중한 진입 필요)

---

## 🔥 섹터별 수급 집중도 (TOP 10)

> **섹터점수** = 평균점수 × (1 + 고득점종목수/전체종목수)

| 순위 | 섹터 | 평균 점수 | 종목 수 | 고득점 종목 | 섹터 점수 |
|:----:|------|----------:|--------:|------------:|----------:|
"""

    for idx, (_, row) in enumerate(sector_concentration.iterrows(), 1):
        md += f"| {idx} | {row['sector']} | {row['avg_score']:.1f} | {int(row['total_count'])}개 | {int(row['high_score_count'])}개 | **{row['sector_score']:.1f}** |\n"

    # 각 섹터별 대표 종목
    md += "\n### 섹터별 대표 종목 (TOP 3)\n\n"

    for _, sector_row in sector_concentration.head(5).iterrows():
        sector = sector_row['sector']
        sector_stocks = df[df['sector'] == sector].nlargest(3, 'combined_score')

        md += f"#### {sector}\n\n"
        md += "| 종목명 | 패턴 | 종합점수 | 시그널내용 |\n"
        md += "|--------|:----:|---------:|-----------|\n"

        for _, stock in sector_stocks.iterrows():
            pattern_emoji = {'모멘텀형': '🔥', '지속형': '📈', '전환형': '🔄'}.get(stock['pattern'], '❓')
            signal_list = stock['signal_list'] if pd.notna(stock['signal_list']) else '-'
            md += f"| **{stock['stock_name']}** | {pattern_emoji} {stock['pattern']} | {stock['combined_score']:.1f} | {signal_list} |\n"

        md += "\n"

    md += f"""
---

## 📚 용어 설명

### 핵심 지표

#### Sff (Supply Float Factor)
```
Sff = (순매수 금액 / 유통시총) × 100
```
- 시가총액 왜곡을 제거하고 유통물량 대비 매수 강도를 정규화

#### Z-Score
```
Z-Score = (현재값 - 60일 평균) / 60일 표준편차
```
- 변동성을 보정하여 이상 수급(평소와 다른 매수/매도)을 탐지
- |Z| > 2.0: 이상 수급 발생

### 시그널 (3가지)

| 시그널 | 정의 | 의미 |
|--------|------|------|
| ✅ MA 골든크로스 | 외국인 5일MA > 20일MA 돌파 | 단기 수급이 장기 추세를 상향 돌파 → 매수 타이밍 |
| ⚡ 수급 가속도 | (최근 5일 평균) / (직전 5일 평균) > 1.5배 | 수급 강도가 급격히 증가 → 모멘텀 가속 |
| 🤝 외인-기관 동조율 | 최근 20일 중 동시 매수 비율 > 50% | 두 투자 주체가 동시 매수 → 확신도 높음 |

### 점수 메트릭 (4가지)

| 메트릭 | 계산식 | 의미 | 활용 |
|--------|--------|------|------|
| **Recent** | (1W + 1M) / 2 | 최근 1주~1개월 수급 강도의 평균 | 현재 진행형 매수세 파악 |
| **Momentum** | 1W - 2Y | 단기(1주) vs 장기(2년) 수급 격차 | 전환점 포착, 수급 개선도 |
| **Weighted** | 1W×0.30 + 1M×0.25 + ... | 최근에 높은 가중치를 부여한 트렌드 | 중장기 추세 방향 판단 |
| **Average** | (1W + 1M + ... + 2Y) / 6 | 전체 기간의 단순 평균 | 일관된 수급 파악 |

### 종합점수 & 추천 기준

**종합점수 계산**
```
종합점수 = 패턴 점수 + (시그널 개수 × {signal_bonus}점)
```

**추천 등급**

| 등급 | 기준 |
|------|------|
| ⭐⭐⭐ 강력 추천 | 종합점수 80+ AND 시그널 2개 이상 |
| ⭐⭐ 추천 | 종합점수 70+ AND 시그널 1개 이상 |
| ⭐ 관심 | 종합점수 60+ OR 시그널 2개 이상 |

### 진입/청산 기준

| 구분 | 기준 |
|------|------|
| **진입 전략** | 시그널 발생 시점에서 당일 종가 또는 익일 시초가 매수 |
| **손절 기준** | 진입가 대비 -7% 도달 시 무조건 청산 |
| **목표 수익률** | +15% 달성 시 50% 익절, +25% 달성 시 전량 청산 |
| **최대 보유 기간** | 30일 경과 시 수익/손실 무관 전량 청산 |

---

## 📌 면책 조항

본 리포트는 투자 참고용으로 제공되며, 투자 권유가 아닙니다.
모든 투자 결정과 그에 따른 책임은 투자자 본인에게 있습니다.

---

**Generated by 수급 레짐 스캐너 v3.1** | [GitHub](https://github.com/unanimous0/LP_MM_TRADING)
"""

    return md


def main():
    parser = argparse.ArgumentParser(
        description='수급 레짐 스캐너 - Markdown 리포트 생성',
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
        args.output = f'output/regime_report_{timestamp}.md'

    # 디렉토리 생성
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("📝 Markdown 리포트 생성")
    print("="*80)
    print(f"입력: {args.input}")
    print(f"출력: {args.output}")
    print()

    generate_md_report(args.input, args.output, args.signal_bonus)

    print()
    print("="*80)
    print("✅ 완료!")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
