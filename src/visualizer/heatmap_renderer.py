"""
Stage 2 히트맵 렌더링 모듈

350×8 매트릭스 히트맵 생성 (종목 × 기간)
- Y축: Z-Score 강도순 정렬 (상단 = 강한 매수)
- X축: 8개 기간 (1D, 1W, 1M, 3M, 6M, 1Y, 2Y)
- 색상: RdYlGn (빨강=매도, 노랑=중립, 초록=매수)
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import platform
import warnings


class HeatmapRenderer:
    """히트맵 시각화 클래스"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: 시각화 설정 딕셔너리 (config.py의 DEFAULT_CONFIG)
        """
        self.config = config
        self.vis_config = config['visualization']

        # 한글 폰트 설정 (OS별 자동 감지)
        self._setup_korean_font()

    def _setup_korean_font(self):
        """
        한글 폰트 설정 (OS별 자동 감지)

        matplotlib의 기본 폰트는 한글을 지원하지 않아 경고가 발생합니다.
        이 메서드는 OS별로 시스템에 설치된 한글 폰트를 자동 감지하여 설정합니다.

        지원 OS:
        - macOS: AppleGothic, AppleSDGothicNeo
        - Windows: Malgun Gothic, Gulim
        - Linux: NanumGothic, UnDotum

        Note:
            폰트를 찾지 못하면 경고만 출력하고 계속 진행합니다.
            (히트맵은 생성되지만 한글이 깨져 보일 수 있습니다)
        """
        system = platform.system()

        # OS별 한글 폰트 우선순위 리스트
        font_candidates = {
            'Darwin': ['AppleGothic', 'AppleSDGothicNeo'],  # macOS
            'Windows': ['Malgun Gothic', 'Gulim', 'Batang'],  # Windows
            'Linux': ['NanumGothic', 'UnDotum', 'NanumBarunGothic']  # Linux
        }

        # 현재 OS의 폰트 후보 가져오기
        candidates = font_candidates.get(system, [])

        # matplotlib에서 사용 가능한 폰트 목록 가져오기
        try:
            import matplotlib.font_manager as fm
            available_fonts = [f.name for f in fm.fontManager.ttflist]

            # 후보 중 설치된 폰트 찾기
            for font in candidates:
                if font in available_fonts:
                    plt.rcParams['font.family'] = font
                    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
                    print(f"[INFO] Korean font set to: {font}")
                    return

            # 폰트를 찾지 못한 경우
            warnings.warn(
                f"Korean font not found on {system}. "
                f"Tried: {', '.join(candidates)}. "
                f"Korean text may not display correctly. "
                f"Install a Korean font to fix this issue."
            )

        except Exception as e:
            warnings.warn(f"Failed to setup Korean font: {e}")

    def render_multi_period_heatmap(
        self,
        zscore_matrix: pd.DataFrame,
        output_path: str,
        stock_names: pd.DataFrame = None
    ) -> None:
        """
        350×8 히트맵 렌더링

        Args:
            zscore_matrix: pd.DataFrame
                - index: stock_code (종목 코드)
                - columns: ['1D', '1W', '1M', ...] (기간)
                - values: Z-Score
            output_path: 저장 경로 (예: 'output/heatmap.png')
            stock_names: 종목명 매핑 (옵션, stock_code → stock_name)

        Layout:
            Y축: 350 종목 (Z-Score 강도순, 상단 = 강한 매수)
            X축: 8개 기간 (1D, 1W, 1M, 3M, 6M, 1Y, 2Y)
            색상: RdYlGn (빨강=매도, 초록=매수)
        """
        # 출력 디렉토리 생성
        output_path = Path(output_path)
        if self.config['output']['auto_create_dir']:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Y축 정렬 (Z-Score 강도순)
        sort_by_column = self._get_sort_column(zscore_matrix)
        zscore_matrix_sorted = zscore_matrix.sort_values(
            by=sort_by_column,
            ascending=not self.vis_config['descending']  # True면 내림차순
        )

        print(f"[INFO] Sorted by {sort_by_column} ({len(zscore_matrix_sorted)} stocks)")

        # Step 2: Figure 생성
        fig, ax = plt.subplots(
            figsize=self.vis_config['figsize'],
            dpi=self.vis_config['dpi']
        )

        # Step 3: 히트맵 렌더링
        cmap = plt.cm.get_cmap(self.vis_config['colormap'])
        vmin = self.vis_config['zscore_vmin']
        vmax = self.vis_config['zscore_vmax']

        # NaN을 회색으로 처리
        cmap_with_nan = cmap.copy()
        cmap_with_nan.set_bad(color='lightgray')

        im = ax.imshow(
            zscore_matrix_sorted.values,
            cmap=cmap_with_nan,
            aspect='auto',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest'
        )

        # Step 4: X축 레이블 (기간)
        periods = list(zscore_matrix.columns)
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, fontsize=12, fontweight='bold')
        ax.set_xlabel('Period (Lookback Window)', fontsize=14, fontweight='bold')

        # Step 5: Y축 레이블 (종목 수)
        n_stocks = len(zscore_matrix_sorted)
        ax.set_ylabel(f'Stocks (n={n_stocks}, sorted by Z-Score)', fontsize=14, fontweight='bold')

        # Y축 틱: 상/중/하 3개만 표시
        y_ticks = [0, n_stocks // 2, n_stocks - 1]
        y_labels = ['Top (Strong Buy)', 'Mid', 'Bottom (Strong Sell)']
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels, fontsize=10)

        # Step 6: Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Z-Score (σ)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

        # Colorbar 틱 레이블
        cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
        cbar.set_ticklabels(['-3σ', '-2σ', '-1σ', '0', '+1σ', '+2σ', '+3σ'])

        # Step 7: 타이틀
        title = 'Supply-Demand Regime Heatmap (Multi-Period Z-Score)'
        if self.config['filtering'].get('sectors'):
            sectors_str = ', '.join(self.config['filtering']['sectors'])
            title += f'\nSector: {sectors_str}'

        plt.title(title, fontsize=16, fontweight='bold', pad=20)

        # Step 8: 그리드 추가 (가독성)
        ax.set_xticks(np.arange(len(periods)) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_stocks) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)

        # Step 9: 저장
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.vis_config['dpi'], bbox_inches='tight')
        plt.close()

        print(f"[OK] Heatmap saved to: {output_path}")

    def _get_sort_column(self, zscore_matrix: pd.DataFrame) -> str:
        """
        Y축 정렬 기준 컬럼 결정

        기본: 가장 최근 기간 (1D)
        설정에서 변경 가능
        """
        sort_by = self.vis_config.get('sort_by', 'combined_zscore')

        # 기간 컬럼 중 가장 최근 것 (첫 번째 컬럼)
        if sort_by == 'combined_zscore' and not zscore_matrix.empty:
            return zscore_matrix.columns[0]  # 첫 번째 기간 (1D)

        return zscore_matrix.columns[0]

    def render_sector_comparison(
        self,
        sector_heatmaps: Dict[str, pd.DataFrame],
        output_path: str
    ) -> None:
        """
        여러 섹터의 히트맵을 한 화면에 비교

        Args:
            sector_heatmaps: {섹터명: zscore_matrix}
            output_path: 저장 경로

        Layout:
            세로로 섹터별 히트맵 배치
        """
        n_sectors = len(sector_heatmaps)

        fig, axes = plt.subplots(
            nrows=n_sectors,
            ncols=1,
            figsize=(20, 6 * n_sectors),
            dpi=self.vis_config['dpi']
        )

        if n_sectors == 1:
            axes = [axes]

        cmap = plt.cm.get_cmap(self.vis_config['colormap'])
        vmin = self.vis_config['zscore_vmin']
        vmax = self.vis_config['zscore_vmax']

        for idx, (sector_name, zscore_matrix) in enumerate(sector_heatmaps.items()):
            ax = axes[idx]

            # 정렬
            sort_by_column = zscore_matrix.columns[0]
            zscore_matrix_sorted = zscore_matrix.sort_values(
                by=sort_by_column,
                ascending=False
            )

            # 렌더링
            im = ax.imshow(
                zscore_matrix_sorted.values,
                cmap=cmap,
                aspect='auto',
                vmin=vmin,
                vmax=vmax,
                interpolation='nearest'
            )

            # X축
            periods = list(zscore_matrix.columns)
            ax.set_xticks(range(len(periods)))
            ax.set_xticklabels(periods, fontsize=10)

            # Y축
            ax.set_ylabel(f'{sector_name}\n({len(zscore_matrix_sorted)} stocks)', fontsize=12)
            ax.set_yticks([])

            # 타이틀
            ax.set_title(f'Sector: {sector_name}', fontsize=14, fontweight='bold')

            # Colorbar
            if idx == 0:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Z-Score (σ)', rotation=270, labelpad=20, fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=self.vis_config['dpi'], bbox_inches='tight')
        plt.close()

        print(f"[OK] Sector comparison saved to: {output_path}")
