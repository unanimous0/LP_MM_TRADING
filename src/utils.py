"""
유틸리티 함수 모듈

입력 검증, 보안 관련 헬퍼 함수 제공
"""

import re
from typing import List, Optional


def validate_stock_code(stock_code: str) -> bool:
    """
    종목 코드 검증 (SQL 인젝션 방지)

    한국 주식 종목코드는 6자리 숫자로 구성
    예: '005930' (삼성전자), '000660' (SK하이닉스)

    Args:
        stock_code: 검증할 종목 코드

    Returns:
        bool: 유효하면 True, 아니면 False

    Examples:
        >>> validate_stock_code('005930')
        True
        >>> validate_stock_code('000660')
        True
        >>> validate_stock_code('12345')
        False
        >>> validate_stock_code('005930\'); DROP TABLE stocks; --')
        False
    """
    if not isinstance(stock_code, str):
        return False

    # 정확히 6자리 숫자만 허용
    return bool(re.match(r'^[0-9]{6}$', stock_code))


def validate_stock_codes(stock_codes: List[str]) -> List[str]:
    """
    종목 코드 리스트 검증 및 필터링

    Args:
        stock_codes: 검증할 종목 코드 리스트

    Returns:
        List[str]: 검증된 종목 코드 리스트 (유효한 것만)

    Raises:
        ValueError: 모든 종목 코드가 유효하지 않을 경우

    Examples:
        >>> validate_stock_codes(['005930', '000660'])
        ['005930', '000660']
        >>> validate_stock_codes(['005930', 'INVALID', '000660'])
        ['005930', '000660']  # 유효한 것만 반환
    """
    if not stock_codes:
        return []

    # 유효한 종목 코드만 필터링
    validated = [code for code in stock_codes if validate_stock_code(code)]

    # 하나도 유효하지 않으면 예외
    if not validated:
        raise ValueError(
            f"No valid stock codes found. "
            f"Stock codes must be 6-digit numbers (e.g., '005930'). "
            f"Invalid codes: {stock_codes}"
        )

    # 일부 무효한 코드가 있으면 경고
    if len(validated) < len(stock_codes):
        invalid = [code for code in stock_codes if code not in validated]
        print(f"[WARN] Skipped invalid stock codes: {invalid}")

    return validated


def validate_date_format(date_str: str) -> bool:
    """
    날짜 형식 검증 (YYYY-MM-DD)

    Args:
        date_str: 검증할 날짜 문자열

    Returns:
        bool: 유효하면 True, 아니면 False

    Examples:
        >>> validate_date_format('2026-02-10')
        True
        >>> validate_date_format('2026/02/10')
        False
        >>> validate_date_format('2026-02-10\'); DROP TABLE stocks; --')
        False
    """
    if not isinstance(date_str, str):
        return False

    # YYYY-MM-DD 형식만 허용
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', date_str))


def sanitize_sector_name(sector: str) -> str:
    """
    섹터 이름 검증 및 정제

    Args:
        sector: 섹터 이름

    Returns:
        str: 정제된 섹터 이름

    Raises:
        ValueError: 유효하지 않은 섹터 이름

    Examples:
        >>> sanitize_sector_name('반도체')
        '반도체'
        >>> sanitize_sector_name('반도체\'; DROP TABLE')
        ValueError: Invalid sector name
    """
    if not isinstance(sector, str):
        raise ValueError("Sector name must be a string")

    # 길이 제한 (최대 50자)
    if len(sector) > 50:
        raise ValueError("Sector name too long (max 50 characters)")

    # 위험한 문자 체크 (SQL 특수문자)
    dangerous_chars = ["'", '"', ';', '--', '/*', '*/', 'DROP', 'DELETE', 'INSERT', 'UPDATE']
    for char in dangerous_chars:
        if char.upper() in sector.upper():
            raise ValueError(f"Invalid sector name: contains dangerous pattern '{char}'")

    return sector.strip()
