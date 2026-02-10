"""
엑셀 데이터 수집 모듈

Multi-level header를 가진 엑셀 파일에서 투자자 수급 데이터를 파싱하고
Wide format을 Long format으로 변환합니다.
"""

import pandas as pd


class ExcelCollector:
    """엑셀 파일에서 투자자 수급 데이터 수집"""

    def load_stock_mapping(self, excel_path: str) -> pd.DataFrame:
        """
        종목코드-이름 매핑 로드

        Args:
            excel_path: 종목 매핑 엑셀 파일 경로

        Returns:
            pd.DataFrame: 종목코드, 종목명 컬럼을 포함한 DataFrame
        """
        df = pd.read_excel(excel_path)
        df.columns = ['stock_code', 'stock_name']
        df['stock_code'] = df['stock_code'].astype(str).str.zfill(6)  # 6자리 패딩
        return df

    def load_investor_flows(self, excel_path: str, sheet_name: str) -> pd.DataFrame:
        """
        투자자 수급 데이터 로드 및 정규화

        Multi-level header (날짜 | 종목명 | 외국인순매수량 | 외국인순매수금액 | ...)
        형태의 데이터를 Long format으로 변환

        Args:
            excel_path: 투자자 수급 엑셀 파일 경로
            sheet_name: 시트명 ('KOSPI200' 또는 'KOSDAQ150')

        Returns:
            pd.DataFrame: 정규화된 투자자 수급 데이터 (금액은 원 단위)
        """
        # Multi-level header 읽기
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=[0, 1])

        # 날짜 컬럼 (첫 번째 컬럼)
        date_col = df.columns[0]
        dates = pd.to_datetime(df[date_col]).dt.date

        # Wide → Long 변환
        records = []

        # 4개씩 건너뛰며 종목명 추출 (날짜 컬럼 제외)
        stock_names = []
        for i in range(1, len(df.columns), 4):
            stock_name = df.columns[i][0]
            stock_names.append(stock_name)

        for stock_name in stock_names:
            # 해당 종목의 4개 컬럼 찾기
            cols = [col for col in df.columns if col[0] == stock_name]

            if len(cols) != 4:
                print(f"[WARN] {stock_name} has {len(cols)} columns (expected 4)")
                continue

            for i, date in enumerate(dates):
                # 엑셀 파일의 실제 순서: 기관 → 외국인
                # cols[0]: 기관 순매수 수량
                # cols[1]: 기관 순매수 금액
                # cols[2]: 외국인 순매수 수량
                # cols[3]: 외국인 순매수 금액
                records.append({
                    'trade_date': date,
                    'stock_name': stock_name,
                    'institution_net_volume': df[cols[0]].iloc[i],  # 기관 수량
                    'institution_net_amount': df[cols[1]].iloc[i],  # 기관 금액
                    'foreign_net_volume': df[cols[2]].iloc[i],      # 외국인 수량
                    'foreign_net_amount': df[cols[3]].iloc[i]       # 외국인 금액
                })

        df_result = pd.DataFrame(records)

        # 엑셀 파일은 천원 단위이므로 원 단위로 변환 (2026-02-09 추가)
        df_result['foreign_net_volume'] = df_result['foreign_net_volume'] * 1000
        df_result['foreign_net_amount'] = df_result['foreign_net_amount'] * 1000
        df_result['institution_net_volume'] = df_result['institution_net_volume'] * 1000
        df_result['institution_net_amount'] = df_result['institution_net_amount'] * 1000

        return df_result

    def load_market_caps(self, excel_path: str, sheet_name: str) -> pd.DataFrame:
        """
        시가총액 데이터 로드

        Args:
            excel_path: 시가총액 엑셀 파일 경로
            sheet_name: 시트명 ('KOSPI200' 또는 'KOSDAQ150')

        Returns:
            pd.DataFrame: 시가총액 데이터 (trade_date, stock_name, market_cap)
        """
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=[0, 1])

        # 날짜 컬럼
        date_col = df.columns[0]
        dates = pd.to_datetime(df[date_col]).dt.date

        # Wide → Long 변환
        records = []
        for col in df.columns[1:]:
            stock_name = col[0]
            for i, date in enumerate(dates):
                records.append({
                    'trade_date': date,
                    'stock_name': stock_name,
                    'market_cap': df[col].iloc[i]
                })

        df_result = pd.DataFrame(records)

        # 시가총액도 천원 단위이므로 원 단위로 변환
        df_result['market_cap'] = df_result['market_cap'] * 1000

        return df_result

    def load_stock_prices(self, excel_path: str, sheet_name: str) -> pd.DataFrame:
        """
        주가 및 거래량 데이터 로드

        Multi-level header format (same as investor flows):
        Row 1: 날짜 | 삼성전자 | 삼성전자 | 삼성전자 | SK하이닉스 | ...
        Row 2:      | 종가    | 거래량   | 거래대금  | 종가       | ...

        Args:
            excel_path: 주가 데이터 엑셀 파일 경로
            sheet_name: 시트명 ('KOSPI200' 또는 'KOSDAQ150')

        Returns:
            pd.DataFrame: (trade_date, stock_name, close_price, trading_volume, trading_value)
        """
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=[0, 1])

        # 날짜 컬럼 (첫 번째 컬럼)
        date_col = df.columns[0]
        dates = pd.to_datetime(df[date_col]).dt.date

        # Wide → Long 변환
        records = []

        # 3개씩 건너뛰며 종목명 추출 (날짜 컬럼 제외)
        stock_names = []
        for i in range(1, len(df.columns), 3):
            stock_name = df.columns[i][0]
            stock_names.append(stock_name)

        for stock_name in stock_names:
            # 해당 종목의 3개 컬럼 찾기 (종가, 거래량, 거래대금)
            cols = [col for col in df.columns if col[0] == stock_name]

            if len(cols) != 3:
                print(f"[WARN] {stock_name} has {len(cols)} columns (expected 3)")
                continue

            for i, date in enumerate(dates):
                records.append({
                    'trade_date': date,
                    'stock_name': stock_name,
                    'close_price': df[cols[0]].iloc[i],      # 종가
                    'trading_volume': df[cols[1]].iloc[i],   # 거래량
                    'trading_value': df[cols[2]].iloc[i]     # 거래대금
                })

        df_result = pd.DataFrame(records)

        # 거래대금은 천원 단위이므로 원 단위로 변환
        df_result['trading_value'] = df_result['trading_value'] * 1000

        return df_result

    def load_free_float(self, excel_path: str, sheet_name: str) -> pd.DataFrame:
        """
        유통주식수 데이터 로드

        Simple table format:
        종목명 | 유통주식수 | 유통비율
        삼성전자 | 5233233233 | 83.25

        Args:
            excel_path: 유통주식 데이터 엑셀 파일 경로
            sheet_name: 시트명 ('KOSPI200' 또는 'KOSDAQ150')

        Returns:
            pd.DataFrame: (stock_name, free_float_shares, free_float_ratio)
        """
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        # 첫 3개 컬럼만 사용 (종목명, 유통주식수, 유통비율)
        df = df.iloc[:, :3]
        df.columns = ['stock_name', 'free_float_shares', 'free_float_ratio']

        # 쉼표로 구분된 숫자 처리
        if df['free_float_shares'].dtype == 'object':
            df['free_float_shares'] = (df['free_float_shares']
                                      .astype(str)
                                      .str.replace(',', '')
                                      .astype(float)
                                      .astype('Int64'))

        # NaN 값 제거
        df = df.dropna(subset=['stock_name'])

        return df
