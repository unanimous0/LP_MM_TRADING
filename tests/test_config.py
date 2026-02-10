"""
Unit tests for src/config.py (Fixed version)
"""

import pytest
import copy
from src.config import load_config, _validate_config, DEFAULT_CONFIG


class TestDefaultConfig:
    """Test DEFAULT_CONFIG validity"""

    def test_default_config_is_valid(self):
        _validate_config(DEFAULT_CONFIG)

    def test_all_periods_positive(self):
        for period, days in DEFAULT_CONFIG['periods'].items():
            assert days > 0

    def test_zscore_range(self):
        vmin = DEFAULT_CONFIG['visualization']['zscore_vmin']
        vmax = DEFAULT_CONFIG['visualization']['zscore_vmax']
        assert vmin < 0
        assert vmax > 0


class TestValidateConfigPeriods:
    def test_negative_period(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['periods']['INVALID'] = -10
        with pytest.raises(ValueError):
            _validate_config(config)

    def test_zero_period(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['periods']['ZERO'] = 0
        with pytest.raises(ValueError):
            _validate_config(config)


class TestValidateConfigZScore:
    def test_positive_vmin(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['visualization']['zscore_vmin'] = 2.0
        with pytest.raises(ValueError):
            _validate_config(config)

    def test_negative_vmax(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['visualization']['zscore_vmax'] = -2.0
        with pytest.raises(ValueError):
            _validate_config(config)


class TestValidateConfigDPI:
    def test_dpi_too_low(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['visualization']['dpi'] = 30
        with pytest.raises(ValueError):
            _validate_config(config)

    def test_dpi_too_high(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['visualization']['dpi'] = 2000
        with pytest.raises(ValueError):
            _validate_config(config)

    def test_dpi_in_range(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        for dpi in [50, 150, 300, 1000]:
            config['visualization']['dpi'] = dpi
            _validate_config(config)


class TestValidateConfigFigsize:
    def test_negative_width(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['visualization']['figsize'] = (-10, 12)
        with pytest.raises(ValueError):
            _validate_config(config)

    def test_negative_height(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['visualization']['figsize'] = (20, -8)
        with pytest.raises(ValueError):
            _validate_config(config)


class TestValidateConfigColormap:
    def test_invalid_colormap(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['visualization']['colormap'] = 'INVALID_CMAP'
        with pytest.raises(ValueError):
            _validate_config(config)

    def test_valid_colormaps(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        for cmap in ['RdYlGn', 'coolwarm', 'seismic']:
            config['visualization']['colormap'] = cmap
            _validate_config(config)


class TestValidateConfigFiltering:
    def test_negative_top_n(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['filtering']['top_n_stocks'] = -5
        with pytest.raises(ValueError):
            _validate_config(config)

    def test_negative_min_cap(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config['filtering']['min_market_cap'] = -100
        with pytest.raises(ValueError):
            _validate_config(config)


class TestLoadConfig:
    def test_load_default(self):
        config = load_config()
        assert config is not None

    def test_cli_override_dpi(self):
        config = load_config(cli_overrides={'dpi': 300})
        assert config['visualization']['dpi'] == 300

    def test_cli_override_invalid_dpi(self):
        with pytest.raises(ValueError):
            load_config(cli_overrides={'dpi': 5000})


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
