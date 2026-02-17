"""
백테스팅 모듈

Stage 4: 백테스팅 시스템
- BacktestEngine: 롤링 윈도우 시뮬레이션
- Portfolio: 포지션 관리
- PerformanceMetrics: 성과 분석
- ParameterOptimizer: 파라미터 최적화
"""

from .portfolio import Trade, Position, Portfolio
from .engine import BacktestEngine, BacktestConfig
from .metrics import PerformanceMetrics

__all__ = [
    'Trade',
    'Position',
    'Portfolio',
    'BacktestEngine',
    'BacktestConfig',
    'PerformanceMetrics',
]
