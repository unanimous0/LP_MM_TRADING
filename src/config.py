"""
Stage 2 전역 설정 파일

8개 기간 정의, 시각화 파라미터, 필터링 옵션 등을 중앙 관리
CLI 오버라이드 지원 (argparse → .env → 기본값 순서)
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


# 기본 설정값 (Baseline)
DEFAULT_CONFIG = {
    # ============================================================================
    # 6개 기간 정의 (영업일 기준)
    # 주의: 1D(1일)는 표준편차 계산 불가로 제외
    # ============================================================================
    'periods': {
        '5D': 5,
        '10D': 10,
        '20D': 20,
        '50D': 50,
        '100D': 100,
        '200D': 200,
        '500D': 500,
    },

    # ============================================================================
    # 시각화 설정
    # ============================================================================
    'visualization': {
        # Z-Score 색상 임계값
        'zscore_thresholds': [-3, -2, -1, 0, 1, 2, 3],

        # 색상 스케일 (RdYlGn: 빨강=매도, 노랑=중립, 초록=매수)
        'colormap': 'RdYlGn',

        # 차트 크기 (인치)
        'figsize': (20, 12),

        # 해상도 (DPI)
        'dpi': 150,

        # Y축 정렬 기준
        # 'recent': 최근 기간(5D 또는 5D+20D) 우선 (추천!)
        # 'momentum': 수급 모멘텀(5D - 200D) - 전환점 포착
        # 'weighted': 가중 평균 (최근 높은 가중치)
        # 'average': 단순 평균 (deprecated, 과거에 강했던 종목 우선)
        'sort_by': 'recent',

        # 정렬 순서 (True: 내림차순 = 상단에 높은 값)
        'descending': True,

        # Z-Score 범위 고정 (colorbar vmin/vmax)
        'zscore_vmin': -3,
        'zscore_vmax': 3,
    },

    # ============================================================================
    # 성능 최적화
    # ============================================================================
    'performance': {
        # Sff 캐싱 활성화 (1회 DB 쿼리 → 8개 기간 재사용)
        'enable_caching': True,

        # 벡터화 Z-Score 계산 (groupby.transform)
        'use_vectorized_zscore': True,
    },

    # ============================================================================
    # 필터링 옵션
    # ============================================================================
    'filtering': {
        # 최소 시가총액 (억원 단위)
        'min_market_cap': 0,

        # 특정 섹터만 (빈 리스트 = 전체)
        'sectors': [],

        # 상위 N개 종목만 (None = 전체)
        'top_n_stocks': None,

        # 특정 시장만 (None = 전체, 'KOSPI200' 또는 'KOSDAQ150')
        'market': None,
    },

    # ============================================================================
    # 기타
    # ============================================================================
    'output': {
        # 기본 저장 경로
        'default_path': 'output/heatmap.png',

        # 출력 디렉토리 자동 생성
        'auto_create_dir': True,
    }
}


def load_config(env_file: Optional[str] = '.env',
                cli_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    3계층 설정 로딩 (우선순위: CLI > .env > 기본값)

    Args:
        env_file: 환경변수 파일 경로 (.env)
        cli_overrides: CLI 인자로 전달된 오버라이드 (argparse의 vars(args))

    Returns:
        병합된 최종 설정 딕셔너리

    Example:
        >>> # CLI 사용 예시
        >>> parser.add_argument('--threshold', type=float)
        >>> args = parser.parse_args()
        >>> config = load_config(cli_overrides=vars(args))
    """
    import copy

    # 1) 기본값 복사
    config = copy.deepcopy(DEFAULT_CONFIG)

    # 2) .env 파일 로드 (존재할 경우)
    if env_file and os.path.exists(env_file):
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)

            # 환경변수 매핑 (예: HEATMAP_COLORMAP=coolwarm)
            if os.getenv('HEATMAP_COLORMAP'):
                config['visualization']['colormap'] = os.getenv('HEATMAP_COLORMAP')

            if os.getenv('HEATMAP_DPI'):
                config['visualization']['dpi'] = int(os.getenv('HEATMAP_DPI'))

        except ImportError:
            pass  # python-dotenv 미설치 시 무시

    # 3) CLI 오버라이드 적용
    if cli_overrides:
        _apply_cli_overrides(config, cli_overrides)

    # 4) 설정값 검증
    try:
        _validate_config(config)
    except ValueError as e:
        print(f"[ERROR] Configuration validation failed: {e}")
        raise

    return config


def _apply_cli_overrides(config: Dict[str, Any], cli_args: Dict[str, Any]) -> None:
    """
    CLI 인자를 config 딕셔너리에 병합

    지원하는 오버라이드:
    - --threshold → visualization.zscore_vmin/vmax
    - --colormap → visualization.colormap
    - --figsize → visualization.figsize
    - --dpi → visualization.dpi
    - --sort-by → visualization.sort_by
    - --sector → filtering.sectors
    - --top → filtering.top_n_stocks
    - --min-cap → filtering.min_market_cap
    - --no-cache → performance.enable_caching
    """
    # 색상 임계값
    if cli_args.get('threshold'):
        threshold = cli_args['threshold']
        config['visualization']['zscore_vmin'] = -threshold
        config['visualization']['zscore_vmax'] = threshold

    # Colormap
    if cli_args.get('colormap'):
        config['visualization']['colormap'] = cli_args['colormap']

    # Figure 크기
    if cli_args.get('figsize'):
        config['visualization']['figsize'] = tuple(cli_args['figsize'])

    # 해상도
    if cli_args.get('dpi'):
        config['visualization']['dpi'] = cli_args['dpi']

    # 정렬 기준
    if cli_args.get('sort_by'):
        config['visualization']['sort_by'] = cli_args['sort_by']

    # 섹터 필터링
    if cli_args.get('sector'):
        config['filtering']['sectors'] = [cli_args['sector']]

    # 상위 N개
    if cli_args.get('top'):
        config['filtering']['top_n_stocks'] = cli_args['top']

    # 최소 시총
    if cli_args.get('min_cap'):
        config['filtering']['min_market_cap'] = cli_args['min_cap']

    # 캐싱 비활성화
    if cli_args.get('no_cache'):
        config['performance']['enable_caching'] = False

    # 출력 경로
    if cli_args.get('output'):
        config['output']['default_path'] = cli_args['output']


def _validate_config(config: Dict[str, Any]) -> None:
    """
    설정값 검증 (입력 오류 방지)

    Args:
        config: 검증할 설정 딕셔너리

    Raises:
        ValueError: 유효하지 않은 설정값

    Validates:
        - periods: 영업일 수 > 0
        - zscore_vmin < 0 < zscore_vmax
        - dpi: 50 ≤ dpi ≤ 1000
        - figsize: width, height > 0
        - colormap: matplotlib 지원 확인
        - top_n_stocks: N > 0 (if set)
    """
    # 1. 기간 검증
    for period_name, days in config['periods'].items():
        if not isinstance(days, int) or days <= 0:
            raise ValueError(
                f"Invalid period '{period_name}': {days} days. "
                f"Must be a positive integer."
            )

    # 2. Z-Score 임계값 검증
    vmin = config['visualization']['zscore_vmin']
    vmax = config['visualization']['zscore_vmax']

    if vmin >= 0:
        raise ValueError(
            f"zscore_vmin must be negative, got {vmin}. "
            f"Example: -3 for ±3σ range"
        )

    if vmax <= 0:
        raise ValueError(
            f"zscore_vmax must be positive, got {vmax}. "
            f"Example: 3 for ±3σ range"
        )

    if vmin >= vmax:
        raise ValueError(
            f"zscore_vmin ({vmin}) must be less than zscore_vmax ({vmax})"
        )

    # 3. DPI 검증
    dpi = config['visualization']['dpi']
    if not isinstance(dpi, int) or not (50 <= dpi <= 1000):
        raise ValueError(
            f"Invalid DPI: {dpi}. Must be between 50 and 1000. "
            f"Common values: 150 (screen), 300 (print)"
        )

    # 4. Figure 크기 검증
    figsize = config['visualization']['figsize']
    if not isinstance(figsize, (tuple, list)) or len(figsize) != 2:
        raise ValueError(
            f"Invalid figsize: {figsize}. Must be (width, height) tuple."
        )

    width, height = figsize
    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid figsize: ({width}, {height}). "
            f"Width and height must be positive."
        )

    # 5. Colormap 검증
    colormap = config['visualization']['colormap']
    try:
        import matplotlib.pyplot as plt
        plt.get_cmap(colormap)
    except ValueError:
        raise ValueError(
            f"Invalid colormap: '{colormap}'. "
            f"Use matplotlib.pyplot.colormaps() to see available options. "
            f"Common: 'RdYlGn', 'coolwarm', 'seismic'"
        )

    # 6. 필터링 옵션 검증
    top_n = config['filtering'].get('top_n_stocks')
    if top_n is not None:
        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError(
                f"Invalid top_n_stocks: {top_n}. Must be a positive integer."
            )

    min_cap = config['filtering'].get('min_market_cap')
    if min_cap is not None:
        if not isinstance(min_cap, (int, float)) or min_cap < 0:
            raise ValueError(
                f"Invalid min_market_cap: {min_cap}. Must be non-negative."
            )

    # 7. 정렬 기준 검증
    sort_by = config['visualization'].get('sort_by')
    valid_sort_modes = ['recent', 'momentum', 'weighted', 'average']
    if sort_by and sort_by not in valid_sort_modes:
        raise ValueError(
            f"Invalid sort_by: '{sort_by}'. "
            f"Must be one of: {', '.join(valid_sort_modes)}"
        )


def get_project_root() -> Path:
    """프로젝트 루트 디렉토리 반환"""
    return Path(__file__).parent.parent


def get_db_path() -> Path:
    """데이터베이스 파일 경로 반환"""
    return get_project_root() / 'data' / 'processed' / 'investor_data.db'


# 모듈 임포트 시 기본 설정 제공
config = DEFAULT_CONFIG
