"""
백테스트 결과 시각화 모듈

matplotlib 기반 차트 생성
"""

from typing import List, Optional, Dict
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

from .portfolio import Trade


class BacktestVisualizer:
    """백테스트 결과 시각화 클래스"""

    # 색상 테마
    COLORS = {
        'long': '#2E86AB',      # 파랑 (매수)
        'short': '#A23B72',     # 보라 (공매도)
        'both': '#F18F01',      # 주황 (혼합)
        'profit': '#06A77D',    # 녹색 (수익)
        'loss': '#D62828',      # 빨강 (손실)
        'benchmark': '#6C757D', # 회색 (코스피)
        'neutral': '#DDDDDD',   # 연회색
    }

    def __init__(self, trades: List[Trade], daily_values: pd.DataFrame,
                 initial_capital: float, benchmark_returns: Optional[pd.DataFrame] = None):
        """
        초기화

        Args:
            trades: 거래 내역 리스트
            daily_values: 일별 포트폴리오 가치
                         컬럼: date (str), portfolio_value (float)
            initial_capital: 초기 자본금
            benchmark_returns: 벤치마크 수익률 (선택, 향후 구현)
        """
        self.trades = trades
        self.daily_values = daily_values.copy()
        self.initial_capital = initial_capital
        self.benchmark_returns = benchmark_returns

        # 날짜를 datetime으로 변환
        if 'date' in self.daily_values.columns:
            self.daily_values['date'] = pd.to_datetime(self.daily_values['date'])
            self.daily_values = self.daily_values.sort_values('date')

        # 누적 수익률 계산
        self.daily_values['return_pct'] = (
            (self.daily_values['value'] / self.initial_capital - 1) * 100
        )

        # 스타일 설정
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'AppleGothic'  # macOS 한글 폰트
        plt.rcParams['axes.unicode_minus'] = False    # 마이너스 기호 깨짐 방지

    def plot_equity_curve(self, save_path: Optional[str] = None,
                         show: bool = True) -> Optional[plt.Figure]:
        """
        누적 수익률 곡선 그리기

        Args:
            save_path: 저장 경로 (None이면 저장 안 함)
            show: 화면에 표시할지 여부

        Returns:
            Figure 객체 (show=False인 경우)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # 수익률 곡선
        ax.plot(self.daily_values['date'],
                self.daily_values['return_pct'],
                color=self.COLORS['both'],
                linewidth=2,
                label='전략 수익률')

        # 0% 기준선
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # 축 설정
        ax.set_xlabel('날짜', fontsize=12)
        ax.set_ylabel('누적 수익률 (%)', fontsize=12)
        ax.set_title('백테스트 누적 수익률 곡선', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        # 날짜 포맷
        fig.autofmt_xdate()

        # 여백 조정
        plt.tight_layout()

        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 수익률 곡선 저장: {save_path}")

        # 표시
        if show:
            plt.show()
        else:
            return fig

    def plot_drawdown(self, save_path: Optional[str] = None,
                     show: bool = True) -> Optional[plt.Figure]:
        """
        낙폭(Drawdown) 추이 그리기

        Args:
            save_path: 저장 경로
            show: 화면에 표시할지 여부

        Returns:
            Figure 객체 (show=False인 경우)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # 낙폭 계산
        cumulative_returns = self.daily_values['value'].values
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max * 100

        # 낙폭 그리기
        ax.fill_between(self.daily_values['date'],
                        drawdown,
                        0,
                        color=self.COLORS['loss'],
                        alpha=0.3,
                        label='낙폭')
        ax.plot(self.daily_values['date'],
                drawdown,
                color=self.COLORS['loss'],
                linewidth=1.5)

        # 최대 낙폭 표시
        max_dd_idx = np.argmin(drawdown)
        max_dd_value = drawdown[max_dd_idx]
        max_dd_date = self.daily_values['date'].iloc[max_dd_idx]

        ax.scatter([max_dd_date], [max_dd_value],
                  color=self.COLORS['loss'],
                  s=100,
                  zorder=5,
                  label=f'최대 낙폭: {max_dd_value:.2f}%')

        # 축 설정
        ax.set_xlabel('날짜', fontsize=12)
        ax.set_ylabel('낙폭 (%)', fontsize=12)
        ax.set_title('포트폴리오 낙폭(Drawdown) 추이', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # 날짜 포맷
        fig.autofmt_xdate()

        # 여백 조정
        plt.tight_layout()

        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 낙폭 차트 저장: {save_path}")

        # 표시
        if show:
            plt.show()
        else:
            return fig

    def plot_monthly_returns(self, save_path: Optional[str] = None,
                            show: bool = True) -> Optional[plt.Figure]:
        """
        월별 수익률 히트맵

        Args:
            save_path: 저장 경로
            show: 화면에 표시할지 여부

        Returns:
            Figure 객체 (show=False인 경우)
        """
        # 월별 수익률 계산
        df = self.daily_values.copy()
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month

        # 각 월의 마지막 날 포트폴리오 가치로 수익률 계산
        monthly = df.groupby(['year', 'month']).agg({
            'value': 'last',
            'date': 'last'
        }).reset_index()

        # 월별 수익률 계산 (전월 대비)
        monthly['monthly_return'] = monthly['value'].pct_change() * 100

        # 피벗 테이블 생성 (년 x 월)
        pivot = monthly.pivot(index='year', columns='month', values='monthly_return')

        # 히트맵 생성
        fig, ax = plt.subplots(figsize=(12, max(4, len(pivot) * 0.6)))

        sns.heatmap(pivot,
                   annot=True,
                   fmt='.2f',
                   cmap='RdYlGn',
                   center=0,
                   cbar_kws={'label': '수익률 (%)'},
                   linewidths=0.5,
                   ax=ax)

        # 축 설정
        ax.set_xlabel('월', fontsize=12)
        ax.set_ylabel('연도', fontsize=12)
        ax.set_title('월별 수익률 히트맵', fontsize=14, fontweight='bold')

        # 월 이름 표시 (실제 존재하는 월만)
        month_labels = [f'{int(col)}월' for col in pivot.columns]
        ax.set_xticklabels(month_labels)

        # 여백 조정
        plt.tight_layout()

        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 월별 수익률 히트맵 저장: {save_path}")

        # 표시
        if show:
            plt.show()
        else:
            return fig

    def plot_return_distribution(self, save_path: Optional[str] = None,
                                show: bool = True) -> Optional[plt.Figure]:
        """
        거래별 수익률 분포 히스토그램

        Args:
            save_path: 저장 경로
            show: 화면에 표시할지 여부

        Returns:
            Figure 객체 (show=False인 경우)
        """
        if not self.trades:
            print("⚠️  거래 내역이 없습니다.")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        # 거래별 수익률 추출
        returns = [trade.return_pct for trade in self.trades]

        # 승리/패배 구분
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        # 히스토그램
        bins = np.linspace(min(returns), max(returns), 30)

        ax.hist(wins, bins=bins, color=self.COLORS['profit'],
                alpha=0.7, label=f'승리 ({len(wins)}건)', edgecolor='black')
        ax.hist(losses, bins=bins, color=self.COLORS['loss'],
                alpha=0.7, label=f'패배 ({len(losses)}건)', edgecolor='black')

        # 평균/중앙값 표시
        mean_return = np.mean(returns)
        median_return = np.median(returns)

        ax.axvline(mean_return, color='blue', linestyle='--',
                  linewidth=2, label=f'평균: {mean_return:.2f}%')
        ax.axvline(median_return, color='green', linestyle='--',
                  linewidth=2, label=f'중앙값: {median_return:.2f}%')
        ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

        # 축 설정
        ax.set_xlabel('수익률 (%)', fontsize=12)
        ax.set_ylabel('거래 횟수', fontsize=12)
        ax.set_title('거래별 수익률 분포', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        # 여백 조정
        plt.tight_layout()

        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 수익률 분포 저장: {save_path}")

        # 표시
        if show:
            plt.show()
        else:
            return fig

    def plot_pattern_performance(self, save_path: Optional[str] = None,
                                show: bool = True) -> Optional[plt.Figure]:
        """
        패턴별 성과 바차트

        Args:
            save_path: 저장 경로
            show: 화면에 표시할지 여부

        Returns:
            Figure 객체 (show=False인 경우)
        """
        if not self.trades:
            print("⚠️  거래 내역이 없습니다.")
            return None

        # 패턴별 통계 계산
        df = pd.DataFrame([t.to_dict() for t in self.trades])

        pattern_stats = df.groupby('pattern').agg({
            'return_pct': ['count', 'mean', lambda x: (x > 0).sum() / len(x) * 100]
        }).reset_index()

        pattern_stats.columns = ['pattern', 'count', 'avg_return', 'win_rate']
        pattern_stats = pattern_stats.sort_values('avg_return', ascending=False)

        # 차트 생성
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # 1. 평균 수익률
        colors1 = [self.COLORS['profit'] if x > 0 else self.COLORS['loss']
                   for x in pattern_stats['avg_return']]
        ax1.barh(pattern_stats['pattern'], pattern_stats['avg_return'], color=colors1)
        ax1.set_xlabel('평균 수익률 (%)', fontsize=11)
        ax1.set_title('패턴별 평균 수익률', fontsize=12, fontweight='bold')
        ax1.axvline(0, color='black', linestyle='-', linewidth=1)
        ax1.grid(True, alpha=0.3, axis='x')

        # 2. 승률
        ax2.barh(pattern_stats['pattern'], pattern_stats['win_rate'],
                color=self.COLORS['both'])
        ax2.set_xlabel('승률 (%)', fontsize=11)
        ax2.set_title('패턴별 승률', fontsize=12, fontweight='bold')
        ax2.axvline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='x')

        # 3. 거래 횟수
        ax3.barh(pattern_stats['pattern'], pattern_stats['count'],
                color=self.COLORS['neutral'])
        ax3.set_xlabel('거래 횟수', fontsize=11)
        ax3.set_title('패턴별 거래 횟수', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 여백 조정
        plt.tight_layout()

        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 패턴별 성과 차트 저장: {save_path}")

        # 표시
        if show:
            plt.show()
        else:
            return fig

    def plot_all(self, save_dir: Optional[str] = None,
                save_pdf: Optional[str] = None,
                show: bool = False) -> None:
        """
        모든 차트 생성

        Args:
            save_dir: PNG 저장 디렉토리 (None이면 저장 안 함)
            save_pdf: PDF 저장 경로 (None이면 저장 안 함)
            show: 화면에 표시할지 여부
        """
        # PNG 저장
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            self.plot_equity_curve(save_path=save_dir / 'equity_curve.png', show=show)
            self.plot_drawdown(save_path=save_dir / 'drawdown.png', show=show)
            self.plot_monthly_returns(save_path=save_dir / 'monthly_returns.png', show=show)
            self.plot_return_distribution(save_path=save_dir / 'return_distribution.png', show=show)
            self.plot_pattern_performance(save_path=save_dir / 'pattern_performance.png', show=show)

            print(f"\n✅ 모든 차트 PNG 저장 완료: {save_dir}")

        # PDF 저장
        if save_pdf:
            pdf_path = Path(save_pdf)
            pdf_path.parent.mkdir(parents=True, exist_ok=True)

            with PdfPages(pdf_path) as pdf:
                # 각 차트를 PDF에 추가
                fig1 = self.plot_equity_curve(show=False)
                if fig1:
                    pdf.savefig(fig1)
                    plt.close(fig1)

                fig2 = self.plot_drawdown(show=False)
                if fig2:
                    pdf.savefig(fig2)
                    plt.close(fig2)

                fig3 = self.plot_monthly_returns(show=False)
                if fig3:
                    pdf.savefig(fig3)
                    plt.close(fig3)

                fig4 = self.plot_return_distribution(show=False)
                if fig4:
                    pdf.savefig(fig4)
                    plt.close(fig4)

                fig5 = self.plot_pattern_performance(show=False)
                if fig5:
                    pdf.savefig(fig5)
                    plt.close(fig5)

            print(f"\n✅ PDF 리포트 저장 완료: {save_pdf}")

        # show=True인 경우 모든 차트 표시
        if show and not save_dir and not save_pdf:
            self.plot_equity_curve(show=True)
            self.plot_drawdown(show=True)
            self.plot_monthly_returns(show=True)
            self.plot_return_distribution(show=True)
            self.plot_pattern_performance(show=True)
