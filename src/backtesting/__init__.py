"""
백테스팅 모듈

Stage 4: 백테스팅 시스템
- BacktestEngine: 롤링 윈도우 시뮬레이션
- Portfolio: 포지션 관리
- PerformanceMetrics: 성과 분석
- BacktestVisualizer: 결과 시각화 (Week 3)
- BacktestPrecomputer: 사전 계산 (속도 최적화)
"""

from .portfolio import Trade, Position, Portfolio
from .engine import BacktestEngine, BacktestConfig
from .metrics import PerformanceMetrics
from .visualizer import BacktestVisualizer
from .precomputer import BacktestPrecomputer, PrecomputeResult

__all__ = [
    'Trade',
    'Position',
    'Portfolio',
    'BacktestEngine',
    'BacktestConfig',
    'PerformanceMetrics',
    'BacktestVisualizer',
    'BacktestPrecomputer',
    'PrecomputeResult',
]
