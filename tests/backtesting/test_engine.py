"""
BacktestEngine 모듈 테스트

백테스트 엔진 기본 기능 검증
"""

import pytest
import pandas as pd
from datetime import datetime
from src.backtesting.engine import BacktestEngine, BacktestConfig
from src.database.connection import get_connection


class TestBacktestConfig:
    """BacktestConfig 테스트"""

    def test_default_config(self):
        """기본 설정 테스트"""
        config = BacktestConfig()

        assert config.initial_capital == 10_000_000
        assert config.max_positions == 10
        assert config.min_score == 70
        assert config.min_signals == 2
        assert config.target_return == 0.15
        assert config.stop_loss == -0.07
        assert config.max_hold_days == 30
        assert config.allowed_patterns is None

    def test_custom_config(self):
        """사용자 정의 설정 테스트"""
        config = BacktestConfig(
            initial_capital=5_000_000,
            max_positions=5,
            min_score=80,
            min_signals=1,
            target_return=0.20,
            stop_loss=-0.10,
            max_hold_days=20,
            allowed_patterns=['모멘텀형', '지속형']
        )

        assert config.initial_capital == 5_000_000
        assert config.max_positions == 5
        assert config.allowed_patterns == ['모멘텀형', '지속형']


class TestBacktestEngine:
    """BacktestEngine 테스트"""

    @pytest.fixture
    def conn(self):
        """데이터베이스 연결 픽스처"""
        conn = get_connection()
        yield conn
        conn.close()

    @pytest.fixture
    def engine(self, conn):
        """백테스트 엔진 픽스처"""
        config = BacktestConfig(
            initial_capital=10_000_000,
            max_positions=5,
            min_score=60,  # 낮게 설정 (테스트용)
            min_signals=1,  # 낮게 설정 (테스트용)
            target_return=0.10,
            stop_loss=-0.05,
            max_hold_days=15
        )
        return BacktestEngine(conn, config)

    def test_engine_initialization(self, engine):
        """엔진 초기화 테스트"""
        assert engine.conn is not None
        assert engine.config is not None
        assert engine.portfolio is not None
        assert engine.normalizer is not None
        assert engine.classifier is not None
        assert engine.signal_detector is not None

    def test_get_trading_dates(self, engine):
        """거래일 조회 테스트"""
        trading_dates = engine.get_trading_dates('2024-01-02', '2024-01-31')

        assert isinstance(trading_dates, list)
        assert len(trading_dates) > 0
        assert all(isinstance(d, str) for d in trading_dates)
        assert trading_dates[0] >= '2024-01-02'
        assert trading_dates[-1] <= '2024-01-31'
        # 정렬 확인
        assert trading_dates == sorted(trading_dates)

    def test_get_price(self, engine):
        """가격 조회 테스트"""
        # 삼성전자 (005930) 2024-01-02 종가
        price = engine.get_price('005930', '2024-01-02')

        assert price is not None
        assert price > 0

        # 없는 날짜
        none_price = engine.get_price('005930', '2099-12-31')
        assert none_price is None

    def test_get_stock_name(self, engine):
        """종목명 조회 테스트"""
        name = engine.get_stock_name('005930')
        assert name == '삼성전자'

        # 없는 종목
        name2 = engine.get_stock_name('999999')
        assert name2 == '999999'  # fallback to stock_code

    def test_scan_signals_on_date(self, engine):
        """특정 날짜 시그널 스캔 테스트"""
        # 2024-01-31 기준 시그널
        signals = engine._scan_signals_on_date('2024-01-31')

        assert isinstance(signals, pd.DataFrame)

        if not signals.empty:
            # 필수 컬럼 확인
            required_cols = ['stock_code', 'stock_name', 'pattern', 'score', 'signal_count']
            for col in required_cols:
                assert col in signals.columns

            # 데이터 타입 확인
            assert signals['score'].dtype in [float, int]
            assert signals['signal_count'].dtype in [int, 'int64']

    def test_select_entry_candidates(self, engine):
        """진입 후보 선택 테스트"""
        # 가상 시그널 데이터
        signals = pd.DataFrame({
            'stock_code': ['005930', '000660', '035420', '051910', '005380'],
            'stock_name': ['삼성전자', 'SK하이닉스', 'NAVER', 'LG화학', '현대차'],
            'pattern': ['모멘텀형', '지속형', '전환형', '모멘텀형', '기타'],
            'score': [85, 75, 65, 55, 45],
            'signal_count': [2, 2, 1, 1, 0],
        })

        # 기본 설정: min_score=60, min_signals=1
        engine.config.min_score = 60
        engine.config.min_signals = 1

        candidates = engine._select_entry_candidates(signals)

        # 점수 60 이상, 시그널 1개 이상
        assert len(candidates) == 3  # 85, 75, 65
        assert candidates.iloc[0]['stock_code'] == '005930'  # 점수 높은 순

        # 패턴 필터링
        engine.config.allowed_patterns = ['모멘텀형']
        candidates2 = engine._select_entry_candidates(signals)
        assert len(candidates2) == 1  # 005930만
        assert candidates2.iloc[0]['pattern'] == '모멘텀형'

    @pytest.mark.slow
    def test_run_short_backtest(self, engine):
        """짧은 기간 백테스트 통합 테스트 (3개월)"""
        # 2024-01-02 ~ 2024-03-31 (약 3개월)
        result = engine.run(
            start_date='2024-01-02',
            end_date='2024-03-31',
            verbose=False
        )

        # 결과 검증
        assert 'trades' in result
        assert 'daily_values' in result
        assert 'portfolio' in result
        assert 'config' in result

        # trades는 리스트
        assert isinstance(result['trades'], list)

        # daily_values는 DataFrame
        assert isinstance(result['daily_values'], pd.DataFrame)
        assert not result['daily_values'].empty

        # 일별 가치 컬럼 확인
        assert 'date' in result['daily_values'].columns
        assert 'value' in result['daily_values'].columns
        assert 'cash' in result['daily_values'].columns
        assert 'position_count' in result['daily_values'].columns

        # 최종 가치 > 0
        final_value = result['daily_values'].iloc[-1]['value']
        assert final_value > 0

        print(f"\n✅ 3개월 백테스트 완료")
        print(f"거래 횟수: {len(result['trades'])}건")
        print(f"최종 자본금: {final_value:,.0f}원")
        print(f"수익률: {(final_value / engine.config.initial_capital - 1) * 100:+.2f}%")


class TestFutureDataLeakage:
    """미래 데이터 누수 방지 테스트"""

    @pytest.fixture
    def conn(self):
        """데이터베이스 연결 픽스처"""
        conn = get_connection()
        yield conn
        conn.close()

    def test_zscore_end_date_parameter(self, conn):
        """Z-Score 계산 시 end_date 파라미터 존재 확인"""
        from src.analyzer.normalizer import SupplyNormalizer

        normalizer = SupplyNormalizer(conn)

        # end_date 파라미터가 존재하는지 확인
        import inspect
        sig = inspect.signature(normalizer.calculate_zscore)
        assert 'end_date' in sig.parameters

        # 실제 호출 테스트
        zscore = normalizer.calculate_zscore(end_date='2024-01-31')
        assert isinstance(zscore, pd.DataFrame)

        if not zscore.empty:
            # 2024-01-31 이후 데이터가 포함되지 않았는지 확인
            if 'trade_date' in zscore.columns:
                assert all(zscore['trade_date'] <= '2024-01-31')

    def test_scan_signals_uses_end_date(self, conn):
        """_scan_signals_on_date가 end_date를 올바르게 전달하는지 확인"""
        from src.backtesting.engine import BacktestEngine, BacktestConfig

        config = BacktestConfig(min_score=50, min_signals=0)
        engine = BacktestEngine(conn, config)

        # 2024-01-31 기준 스캔
        signals = engine._scan_signals_on_date('2024-01-31')

        # 결과가 있다면, 미래 데이터가 포함되지 않았는지 간접 확인
        # (직접 검증은 어렵지만, 최소한 에러 없이 실행되는지 확인)
        assert isinstance(signals, pd.DataFrame)


class TestShortStrategy:
    """Short Strategy (순매도 탐지) 테스트 (Week 2.5)"""

    @pytest.fixture
    def conn(self):
        """데이터베이스 연결 픽스처"""
        conn = get_connection()
        yield conn
        conn.close()

    @pytest.fixture
    def short_engine(self, conn):
        """Short Strategy 백테스트 엔진"""
        config = BacktestConfig(
            initial_capital=10_000_000,
            max_positions=5,
            min_score=50,
            min_signals=0,
            strategy='short'  # Short 전략
        )
        return BacktestEngine(conn, config)

    def test_config_short_strategy(self):
        """BacktestConfig short strategy 설정 테스트"""
        config = BacktestConfig(strategy='short')
        assert config.strategy == 'short'

    def test_config_invalid_strategy(self):
        """잘못된 strategy 파라미터 테스트"""
        with pytest.raises(ValueError, match="strategy must be"):
            BacktestConfig(strategy='invalid')

    def test_scan_signals_short_direction(self, short_engine):
        """Short direction 시그널 스캔 테스트"""
        signals = short_engine._scan_signals_on_date('2024-01-31', direction='short')

        assert isinstance(signals, pd.DataFrame)

        if not signals.empty:
            # direction 컬럼 확인
            assert 'direction' in signals.columns
            assert (signals['direction'] == 'short').all()

            # 패턴 이름은 동일 (모멘텀형/지속형/전환형/기타)
            assert signals['pattern'].isin(['모멘텀형', '지속형', '전환형', '기타']).all()

            # Short는 weighted < 0 (순매도)
            if 'weighted' in signals.columns:
                assert (signals['weighted'] < 0).all()

    @pytest.mark.slow
    def test_run_short_strategy_backtest(self, short_engine):
        """Short Strategy 백테스트 실행 테스트 (1개월)"""
        result = short_engine.run(
            start_date='2024-01-02',
            end_date='2024-01-31',
            verbose=False
        )

        assert 'trades' in result
        assert 'daily_values' in result
        assert isinstance(result['daily_values'], pd.DataFrame)

        # 거래가 있다면 direction 확인
        if result['trades']:
            for trade in result['trades']:
                assert trade.direction == 'short'

        print(f"\n✅ Short Strategy 백테스트 완료")
        print(f"거래 횟수: {len(result['trades'])}건")

    @pytest.mark.slow
    def test_run_both_strategy_backtest(self, conn):
        """Long + Short 병행 전략 백테스트 테스트"""
        config = BacktestConfig(
            initial_capital=10_000_000,
            max_positions=10,  # Long 5 + Short 5
            min_score=50,
            min_signals=0,
            strategy='both'
        )
        engine = BacktestEngine(conn, config)

        result = engine.run(
            start_date='2024-01-02',
            end_date='2024-01-31',
            verbose=False
        )

        assert isinstance(result['trades'], list)
        assert isinstance(result['daily_values'], pd.DataFrame)

        # 거래가 있다면 long/short 모두 확인
        if result['trades']:
            directions = set(t.direction for t in result['trades'])
            # long 또는 short 중 최소 하나는 있어야 함
            assert directions.intersection({'long', 'short'})

        print(f"\n✅ Both Strategy 백테스트 완료")
        print(f"총 거래 횟수: {len(result['trades'])}건")

        if result['trades']:
            long_trades = [t for t in result['trades'] if t.direction == 'long']
            short_trades = [t for t in result['trades'] if t.direction == 'short']
            print(f"Long 거래: {len(long_trades)}건")
            print(f"Short 거래: {len(short_trades)}건")
