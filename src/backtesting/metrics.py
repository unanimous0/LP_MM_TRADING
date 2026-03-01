"""
백테스트 성과 분석 모듈

PerformanceMetrics 클래스:
- 전체 성과 (승률, 수익률)
- 리스크 지표 (MDD, 샤프 비율)
- 세부 분석 (패턴별, 시그널별, 월별)
- 벤치마크 비교 (코스피 대비 알파)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from .portfolio import Trade


class PerformanceMetrics:
    """백테스트 성과 분석 클래스"""

    def __init__(self, trades: List[Trade], daily_values: pd.DataFrame,
                 initial_capital: float, benchmark_returns: Optional[pd.Series] = None):
        """
        초기화

        Args:
            trades: 완료된 거래 리스트
            daily_values: 일별 포트폴리오 가치 DataFrame
                columns: ['date', 'value', 'cash', 'position_count', 'total_trades']
            initial_capital: 초기 자본금
            benchmark_returns: 벤치마크 수익률 (일별, 선택)
                index: date (YYYY-MM-DD)
                values: 수익률 (소수, 예: 0.01 = 1%)
        """
        self.trades = trades
        self.daily_values = daily_values.copy()
        self.initial_capital = initial_capital
        self.benchmark_returns = benchmark_returns

        # 일별 수익률 계산
        if not daily_values.empty:
            self.daily_values['returns'] = self.daily_values['value'].pct_change()

    # ============================================================================
    # 1. 전체 성과 메트릭
    # ============================================================================

    def total_return(self) -> float:
        """
        누적 수익률 (%)

        Returns:
            누적 수익률 (예: 34.2 = +34.2%)
        """
        if self.daily_values.empty:
            return 0.0

        final_value = self.daily_values.iloc[-1]['value']
        return (final_value / self.initial_capital - 1) * 100

    def win_rate(self) -> float:
        """
        승률 (%)

        Returns:
            승률 (예: 58.6 = 58.6%)
        """
        if not self.trades:
            return 0.0

        wins = sum(1 for t in self.trades if t.return_pct > 0)
        return (wins / len(self.trades)) * 100

    def avg_return(self) -> float:
        """
        평균 수익률 (%)

        Returns:
            평균 수익률 (예: 8.9 = +8.9%)
        """
        if not self.trades:
            return 0.0

        return sum(t.return_pct for t in self.trades) / len(self.trades)

    def avg_win(self) -> float:
        """
        평균 승리 수익률 (%)

        Returns:
            평균 승리 수익률 (승리한 거래만, 예: 12.3 = +12.3%)
        """
        wins = [t.return_pct for t in self.trades if t.return_pct > 0]
        if not wins:
            return 0.0

        return sum(wins) / len(wins)

    def avg_loss(self) -> float:
        """
        평균 손실 수익률 (%)

        Returns:
            평균 손실 수익률 (손실한 거래만, 예: -5.2 = -5.2%)
        """
        losses = [t.return_pct for t in self.trades if t.return_pct <= 0]
        if not losses:
            return 0.0

        return sum(losses) / len(losses)

    def profit_factor(self) -> float:
        """
        Profit Factor (총 이익 / 총 손실)

        Returns:
            Profit Factor (예: 1.5 = 이익이 손실의 1.5배)
            손실이 0이면 inf 반환
        """
        total_profit = sum(t.profit for t in self.trades if t.profit > 0)
        total_loss = abs(sum(t.profit for t in self.trades if t.profit <= 0))

        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0.0

        return total_profit / total_loss

    # ============================================================================
    # 2. 리스크 지표
    # ============================================================================

    def max_drawdown(self) -> Dict[str, float]:
        """
        최대 낙폭 (MDD, Maximum Drawdown)

        Returns:
            {
                'mdd': -18.3,  # MDD (%)
                'start_date': '2025-08-12',  # 최고점 날짜
                'end_date': '2025-09-20',    # 최저점 날짜
                'recovery_date': '2025-10-15' or None  # 회복 날짜
            }
        """
        if self.daily_values.empty:
            return {'mdd': 0.0, 'start_date': None, 'end_date': None, 'recovery_date': None}

        # 누적 최고점 계산
        self.daily_values['cummax'] = self.daily_values['value'].cummax()

        # 낙폭 계산 (%)
        self.daily_values['drawdown'] = (
            (self.daily_values['value'] / self.daily_values['cummax'] - 1) * 100
        )

        # 최대 낙폭
        mdd_idx = self.daily_values['drawdown'].idxmin()
        mdd = self.daily_values.loc[mdd_idx, 'drawdown']

        # 최고점 날짜 (낙폭 시작점)
        start_idx = self.daily_values.loc[:mdd_idx, 'cummax'].idxmax()
        start_date = self.daily_values.loc[start_idx, 'date']

        # 최저점 날짜
        end_date = self.daily_values.loc[mdd_idx, 'date']

        # 회복 날짜 (최고점 갱신)
        peak_value = self.daily_values.loc[start_idx, 'value']
        recovery_df = self.daily_values.loc[mdd_idx:][
            self.daily_values.loc[mdd_idx:, 'value'] >= peak_value
        ]

        recovery_date = recovery_df.iloc[0]['date'] if not recovery_df.empty else None

        return {
            'mdd': mdd,
            'start_date': start_date,
            'end_date': end_date,
            'recovery_date': recovery_date
        }

    def sharpe_ratio(self, risk_free_rate: float = 0.03) -> float:
        """
        샤프 비율 (연환산)

        Sharpe Ratio = (연평균 수익률 - 무위험 수익률) / 연변동성

        Args:
            risk_free_rate: 무위험 수익률 (연율, 기본 3%)

        Returns:
            샤프 비율 (예: 1.24)
        """
        if self.daily_values.empty or len(self.daily_values) < 2:
            return 0.0

        # 일별 수익률
        returns = self.daily_values['returns'].dropna()

        if len(returns) == 0:
            return 0.0

        # 연평균 수익률 (240 거래일 기준)
        annual_return = returns.mean() * 240

        # 연변동성
        annual_volatility = returns.std() * np.sqrt(240)

        if annual_volatility == 0:
            return 0.0

        # 샤프 비율
        sharpe = (annual_return - risk_free_rate) / annual_volatility

        return sharpe

    def max_consecutive_losses(self) -> int:
        """
        최대 연속 손실 횟수

        Returns:
            최대 연속 손실 횟수 (예: 5)
        """
        if not self.trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in self.trades:
            if trade.return_pct <= 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def calmar_ratio(self) -> float:
        """
        칼마 비율 (Calmar Ratio)

        Calmar Ratio = 연평균 수익률 / abs(MDD)

        Returns:
            칼마 비율 (예: 1.2)
        """
        mdd_info = self.max_drawdown()
        mdd = abs(mdd_info['mdd'])

        if mdd == 0:
            return 0.0

        # 연평균 수익률
        total_ret = self.total_return() / 100  # % → 소수
        days = len(self.daily_values)
        annual_return = (1 + total_ret) ** (240 / days) - 1 if days > 0 else 0

        return (annual_return * 100) / mdd

    # ============================================================================
    # 3. 세부 분석
    # ============================================================================

    def performance_by_pattern(self) -> pd.DataFrame:
        """
        패턴별 성과 분석

        Returns:
            pd.DataFrame:
                columns: ['pattern', 'trades', 'win_rate', 'avg_return', 'avg_hold_days']
                예:
                    pattern  trades  win_rate  avg_return  avg_hold_days
                    급등형     45      62.1        12.3            8.2
                    지속형       72      54.8        18.7           21.3
                    전환형       35      47.6         9.1           14.1
        """
        if not self.trades:
            return pd.DataFrame(columns=['pattern', 'trades', 'win_rate', 'avg_return', 'avg_hold_days'])

        patterns = {}
        for trade in self.trades:
            if trade.pattern not in patterns:
                patterns[trade.pattern] = []
            patterns[trade.pattern].append(trade)

        results = []
        for pattern, trades in patterns.items():
            wins = sum(1 for t in trades if t.return_pct > 0)
            win_rate = (wins / len(trades)) * 100
            avg_return = sum(t.return_pct for t in trades) / len(trades)
            avg_hold_days = sum(t.hold_days for t in trades) / len(trades)

            results.append({
                'pattern': pattern,
                'trades': len(trades),
                'win_rate': win_rate,
                'avg_return': avg_return,
                'avg_hold_days': avg_hold_days
            })

        return pd.DataFrame(results).sort_values('avg_return', ascending=False)

    def performance_by_signal_count(self) -> pd.DataFrame:
        """
        시그널 개수별 성과 분석

        Returns:
            pd.DataFrame:
                columns: ['signal_count', 'trades', 'win_rate', 'avg_return']
                예:
                    signal_count  trades  win_rate  avg_return
                    3              25      72.0        18.7
                    2              48      61.3        13.4
                    1              54      52.1         8.9
                    0              25      44.8         5.2
        """
        if not self.trades:
            return pd.DataFrame(columns=['signal_count', 'trades', 'win_rate', 'avg_return'])

        signals = {}
        for trade in self.trades:
            count = trade.signal_count
            if count not in signals:
                signals[count] = []
            signals[count].append(trade)

        results = []
        for count, trades in signals.items():
            wins = sum(1 for t in trades if t.return_pct > 0)
            win_rate = (wins / len(trades)) * 100
            avg_return = sum(t.return_pct for t in trades) / len(trades)

            results.append({
                'signal_count': count,
                'trades': len(trades),
                'win_rate': win_rate,
                'avg_return': avg_return
            })

        return pd.DataFrame(results).sort_values('signal_count', ascending=False)

    def monthly_returns(self) -> pd.DataFrame:
        """
        월별 수익률 분석

        Returns:
            pd.DataFrame:
                columns: ['year_month', 'return']
                index: 0, 1, 2, ...
                예:
                    year_month  return
                    2024-01      8.2
                    2024-02     12.5
                    2024-03     -3.1
        """
        if self.daily_values.empty:
            return pd.DataFrame(columns=['year_month', 'return'])

        # 날짜를 datetime으로 변환
        df = self.daily_values.copy()
        df['date'] = pd.to_datetime(df['date'])

        # 연-월 추출
        df['year_month'] = df['date'].dt.to_period('M')

        # 월별 시작/종료 가치
        monthly = df.groupby('year_month').agg({
            'value': ['first', 'last']
        }).reset_index()

        monthly.columns = ['year_month', 'start_value', 'end_value']

        # 월별 수익률 계산
        monthly['return'] = (monthly['end_value'] / monthly['start_value'] - 1) * 100

        # year_month를 문자열로 변환
        monthly['year_month'] = monthly['year_month'].astype(str)

        return monthly[['year_month', 'return']]

    def trade_duration_stats(self) -> Dict[str, float]:
        """
        거래 보유 기간 통계

        Returns:
            {
                'avg': 14.3,    # 평균 보유 기간
                'min': 1,       # 최소 보유 기간
                'max': 30,      # 최대 보유 기간
                'median': 12    # 중앙값
            }
        """
        if not self.trades:
            return {'avg': 0, 'min': 0, 'max': 0, 'median': 0}

        hold_days = [t.hold_days for t in self.trades]

        return {
            'avg': np.mean(hold_days),
            'min': np.min(hold_days),
            'max': np.max(hold_days),
            'median': np.median(hold_days)
        }

    # ============================================================================
    # 4. 벤치마크 비교
    # ============================================================================

    def alpha(self) -> Optional[float]:
        """
        알파 (전략 수익률 - 벤치마크 수익률)

        Returns:
            알파 (%) 또는 None (벤치마크 없으면)
            예: 21.7 = 전략이 벤치마크보다 21.7% 더 수익
        """
        if self.benchmark_returns is None or self.benchmark_returns.empty:
            return None

        # 전략 수익률
        strategy_return = self.total_return()

        # 벤치마크 수익률 (누적)
        benchmark_cumulative = (1 + self.benchmark_returns).prod() - 1
        benchmark_return = benchmark_cumulative * 100

        return strategy_return - benchmark_return

    def beta(self) -> Optional[float]:
        """
        베타 (시장 민감도)

        Beta = Cov(전략 수익률, 벤치마크 수익률) / Var(벤치마크 수익률)

        Returns:
            베타 또는 None (벤치마크 없으면)
            예: 0.85 = 시장이 1% 움직일 때 전략은 0.85% 움직임
        """
        if self.benchmark_returns is None or self.benchmark_returns.empty:
            return None

        if self.daily_values.empty:
            return None

        # 날짜 매칭
        df = self.daily_values.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # 공통 날짜만
        common_dates = df.index.intersection(self.benchmark_returns.index)

        if len(common_dates) < 2:
            return None

        strategy_returns = df.loc[common_dates, 'returns'].dropna()
        bench_returns = self.benchmark_returns.loc[strategy_returns.index]

        # 길이 확인
        if len(strategy_returns) != len(bench_returns) or len(strategy_returns) < 2:
            return None

        # 베타 계산
        covariance = np.cov(strategy_returns.values, bench_returns.values)[0, 1]
        benchmark_variance = np.var(bench_returns.values, ddof=1)

        if benchmark_variance == 0:
            return None

        return covariance / benchmark_variance

    # ============================================================================
    # 5. 종합 리포트
    # ============================================================================

    def summary(self) -> Dict:
        """
        종합 성과 요약

        Returns:
            Dict: 모든 주요 메트릭 포함
        """
        mdd_info = self.max_drawdown()

        return {
            # 전체 성과
            'total_return': self.total_return(),
            'win_rate': self.win_rate(),
            'avg_return': self.avg_return(),
            'avg_win': self.avg_win(),
            'avg_loss': self.avg_loss(),
            'profit_factor': self.profit_factor(),
            'total_trades': len(self.trades),

            # 리스크 지표
            'max_drawdown': mdd_info['mdd'],
            'mdd_start': mdd_info['start_date'],
            'mdd_end': mdd_info['end_date'],
            'sharpe_ratio': self.sharpe_ratio(),
            'calmar_ratio': self.calmar_ratio(),
            'max_consecutive_losses': self.max_consecutive_losses(),

            # 거래 통계
            'avg_hold_days': self.trade_duration_stats()['avg'],

            # 벤치마크
            'alpha': self.alpha(),
            'beta': self.beta(),
        }
