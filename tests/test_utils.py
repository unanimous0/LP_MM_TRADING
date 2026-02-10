"""
Unit tests for src/utils.py

Tests input validation functions:
- validate_stock_code()
- validate_stock_codes()
- validate_date_format()
- sanitize_sector_name()
"""

import pytest
from src.utils import (
    validate_stock_code,
    validate_stock_codes,
    validate_date_format,
    sanitize_sector_name
)


class TestValidateStockCode:
    """Test validate_stock_code() function"""

    def test_valid_codes(self):
        """Valid 6-digit stock codes"""
        assert validate_stock_code('005930') is True  # 삼성전자
        assert validate_stock_code('000660') is True  # SK하이닉스
        assert validate_stock_code('123456') is True

    def test_invalid_length(self):
        """Invalid length (not 6 digits)"""
        assert validate_stock_code('12345') is False   # 5 digits
        assert validate_stock_code('1234567') is False  # 7 digits
        assert validate_stock_code('') is False         # empty

    def test_invalid_characters(self):
        """Invalid characters (not digits)"""
        assert validate_stock_code('00593A') is False  # contains letter
        assert validate_stock_code('005-30') is False  # contains dash
        assert validate_stock_code('005 30') is False  # contains space

    def test_sql_injection_attempts(self):
        """SQL injection attempts"""
        assert validate_stock_code("005930'); DROP TABLE--") is False
        assert validate_stock_code("000660' OR '1'='1") is False

    def test_invalid_types(self):
        """Invalid input types"""
        assert validate_stock_code(123456) is False     # int
        assert validate_stock_code(None) is False       # None
        assert validate_stock_code(['005930']) is False # list


class TestValidateStockCodes:
    """Test validate_stock_codes() function"""

    def test_all_valid(self):
        """All codes are valid"""
        result = validate_stock_codes(['005930', '000660'])
        assert result == ['005930', '000660']

    def test_mixed_valid_invalid(self):
        """Mix of valid and invalid codes"""
        result = validate_stock_codes(['005930', 'INVALID', '000660'])
        assert result == ['005930', '000660']

    def test_all_invalid(self):
        """All codes are invalid"""
        with pytest.raises(ValueError) as exc_info:
            validate_stock_codes(['INVALID1', 'INVALID2'])
        assert "No valid stock codes found" in str(exc_info.value)

    def test_empty_list(self):
        """Empty list"""
        result = validate_stock_codes([])
        assert result == []

    def test_sql_injection_filtering(self):
        """SQL injection attempts should be filtered"""
        result = validate_stock_codes([
            '005930',
            "000660'); DROP TABLE stocks; --",
            '035720'
        ])
        assert result == ['005930', '035720']


class TestValidateDateFormat:
    """Test validate_date_format() function"""

    def test_valid_dates(self):
        """Valid YYYY-MM-DD format"""
        assert validate_date_format('2026-02-10') is True
        assert validate_date_format('2024-01-01') is True
        assert validate_date_format('2025-12-31') is True

    def test_invalid_format(self):
        """Invalid date formats"""
        assert validate_date_format('2026/02/10') is False  # slash
        assert validate_date_format('02-10-2026') is False  # wrong order
        assert validate_date_format('2026-2-10') is False   # missing zero
        assert validate_date_format('20260210') is False    # no dashes

    def test_sql_injection_attempts(self):
        """SQL injection attempts"""
        assert validate_date_format("2026-02-10'); DROP TABLE--") is False
        assert validate_date_format("2026-02-10' OR '1'='1") is False

    def test_invalid_types(self):
        """Invalid input types"""
        assert validate_date_format(20260210) is False  # int
        assert validate_date_format(None) is False      # None


class TestSanitizeSectorName:
    """Test sanitize_sector_name() function"""

    def test_valid_sector_names(self):
        """Valid sector names"""
        assert sanitize_sector_name('반도체') == '반도체'
        assert sanitize_sector_name('반도체 및 관련장비') == '반도체 및 관련장비'
        assert sanitize_sector_name('  의료  ') == '의료'  # stripped

    def test_sql_injection_attempts(self):
        """SQL injection attempts should raise ValueError"""
        with pytest.raises(ValueError):
            sanitize_sector_name("반도체'; DROP TABLE")

        with pytest.raises(ValueError):
            sanitize_sector_name("반도체\" OR \"1\"=\"1")

        with pytest.raises(ValueError):
            sanitize_sector_name("반도체; DELETE FROM stocks")

    def test_invalid_types(self):
        """Invalid input types"""
        with pytest.raises(ValueError):
            sanitize_sector_name(123)

        with pytest.raises(ValueError):
            sanitize_sector_name(None)

    def test_length_limit(self):
        """Sector name length limit (max 50)"""
        long_name = 'A' * 51
        with pytest.raises(ValueError) as exc_info:
            sanitize_sector_name(long_name)
        assert "too long" in str(exc_info.value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
