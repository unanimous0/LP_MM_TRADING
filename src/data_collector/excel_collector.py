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

        return pd.DataFrame(records)
