"""
Historical price/volume/free float backfill script

Updates existing investor_flows records with price, volume, and free float data.

Usage:
    python scripts/load_price_volume_backfill.py \\
        data/주가_200_150.xlsx \\
        data/유통주식_200_150.xlsx
"""

import sys
from pathlib import Path
import pandas as pd

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collector.excel_collector import ExcelCollector
from src.database.connection import get_connection


def backfill_prices(conn, price_file: str):
    """Load price data and UPDATE investor_flows"""

    print("\n" + "=" * 70)
    print("Step 1: Backfill Price/Volume Data")
    print("=" * 70)

    collector = ExcelCollector()

    # Load both markets
    print("[INFO] Loading KOSPI200 price data...")
    df_kospi = collector.load_stock_prices(price_file, 'KOSPI200')
    print(f"[OK]   Loaded {len(df_kospi):,} records")

    print("[INFO] Loading KOSDAQ150 price data...")
    df_kosdaq = collector.load_stock_prices(price_file, 'KOSDAQ150')
    print(f"[OK]   Loaded {len(df_kosdaq):,} records")

    df_all = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
    print(f"[INFO] Total price records: {len(df_all):,}")

    # Map stock names to codes
    print("[INFO] Mapping stock names to codes...")
    df_stocks = pd.read_sql('SELECT stock_code, stock_name FROM stocks', conn)
    df_all = df_all.merge(df_stocks, on='stock_name', how='left')

    unmapped = df_all['stock_code'].isna().sum()
    if unmapped > 0:
        print(f"[WARN] {unmapped} records could not be mapped")
        unmapped_stocks = df_all[df_all['stock_code'].isna()]['stock_name'].unique()
        print(f"[WARN] Unmapped stocks: {list(unmapped_stocks)[:5]}...")

    df_all = df_all.dropna(subset=['stock_code'])
    print(f"[OK]   Mapped {len(df_all):,} records")

    # Update existing records
    print(f"[INFO] Updating investor_flows table...")
    cursor = conn.cursor()
    updated_count = 0
    skipped_count = 0

    for idx, row in df_all.iterrows():
        cursor.execute("""
            UPDATE investor_flows
            SET close_price = ?,
                trading_volume = ?,
                trading_value = ?
            WHERE trade_date = ? AND stock_code = ?
        """, (
            row['close_price'],
            row['trading_volume'],
            row['trading_value'],
            str(row['trade_date']),
            row['stock_code']
        ))

        if cursor.rowcount > 0:
            updated_count += 1
        else:
            skipped_count += 1

        # Progress indicator
        if (idx + 1) % 10000 == 0:
            print(f"  ... processed {idx + 1:,} / {len(df_all):,} records")

    conn.commit()
    print(f"[OK] Updated {updated_count:,} records")
    print(f"[INFO] Skipped {skipped_count:,} records (no matching trade_date/stock_code)")

    return updated_count


def backfill_free_float(conn, ff_file: str):
    """Load free float data and UPDATE all dates for each stock"""

    print("\n" + "=" * 70)
    print("Step 2: Backfill Free Float Data")
    print("=" * 70)

    collector = ExcelCollector()

    # Load both markets
    print("[INFO] Loading KOSPI200 free float data...")
    df_kospi = collector.load_free_float(ff_file, 'KOSPI200')
    print(f"[OK]   Loaded {len(df_kospi)} stocks")

    print("[INFO] Loading KOSDAQ150 free float data...")
    df_kosdaq = collector.load_free_float(ff_file, 'KOSDAQ150')
    print(f"[OK]   Loaded {len(df_kosdaq)} stocks")

    df_all = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
    print(f"[INFO] Total stocks with free float data: {len(df_all)}")

    # Map to stock codes
    print("[INFO] Mapping stock names to codes...")
    df_stocks = pd.read_sql('SELECT stock_code, stock_name FROM stocks', conn)
    df_all = df_all.merge(df_stocks, on='stock_name', how='left')

    unmapped = df_all['stock_code'].isna().sum()
    if unmapped > 0:
        print(f"[WARN] {unmapped} stocks could not be mapped")
        unmapped_stocks = df_all[df_all['stock_code'].isna()]['stock_name'].unique()
        print(f"[WARN] Unmapped stocks: {list(unmapped_stocks)}")

    df_all = df_all.dropna(subset=['stock_code'])
    print(f"[OK]   Mapped {len(df_all)} stocks")

    # Update ALL records for each stock (free float is static per period)
    print(f"[INFO] Updating all records for {len(df_all)} stocks...")
    cursor = conn.cursor()
    total_updated = 0

    for idx, row in df_all.iterrows():
        cursor.execute("""
            UPDATE investor_flows
            SET free_float_shares = ?,
                free_float_ratio = ?
            WHERE stock_code = ?
        """, (
            int(row['free_float_shares']) if pd.notna(row['free_float_shares']) else None,
            float(row['free_float_ratio']) if pd.notna(row['free_float_ratio']) else None,
            row['stock_code']
        ))

        total_updated += cursor.rowcount

        # Progress indicator
        if (idx + 1) % 50 == 0:
            print(f"  ... processed {idx + 1} / {len(df_all)} stocks")

    conn.commit()
    print(f"[OK] Updated {total_updated:,} records across {len(df_all)} stocks")

    return total_updated


def verify_backfill(conn):
    """Verify backfill results"""

    print("\n" + "=" * 70)
    print("Verification")
    print("=" * 70)

    cursor = conn.cursor()

    # Total records
    cursor.execute("SELECT COUNT(*) FROM investor_flows")
    total = cursor.fetchone()[0]

    # Records with price data
    cursor.execute("SELECT COUNT(*) FROM investor_flows WHERE close_price IS NOT NULL")
    price_not_null = cursor.fetchone()[0]

    # Records with free float
    cursor.execute("SELECT COUNT(*) FROM investor_flows WHERE free_float_shares IS NOT NULL")
    ff_not_null = cursor.fetchone()[0]

    # Records with complete data
    cursor.execute("""
        SELECT COUNT(*)
        FROM investor_flows
        WHERE close_price IS NOT NULL
          AND free_float_shares IS NOT NULL
    """)
    complete = cursor.fetchone()[0]

    print(f"Total records: {total:,}")
    print(f"Records with price data: {price_not_null:,} ({price_not_null/total*100:.1f}%)")
    print(f"Records with free float: {ff_not_null:,} ({ff_not_null/total*100:.1f}%)")
    print(f"Records with complete data: {complete:,} ({complete/total*100:.1f}%)")

    # Sample check: Show a few records
    print("\n[INFO] Sample data (recent records for 삼성전자):")
    df_sample = pd.read_sql("""
        SELECT
            s.stock_name,
            i.trade_date,
            i.close_price,
            i.trading_volume,
            i.free_float_shares,
            i.free_float_ratio
        FROM investor_flows i
        JOIN stocks s ON i.stock_code = s.stock_code
        WHERE s.stock_name = '삼성전자'
        ORDER BY i.trade_date DESC
        LIMIT 5
    """, conn)

    print(df_sample.to_string(index=False))

    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Backfill historical price/volume/free float data')
    parser.add_argument('price_file', help='Path to price Excel file (e.g., 주가_200_150.xlsx)')
    parser.add_argument('ff_file', help='Path to free float Excel file (e.g., 유통주식_200_150.xlsx)')

    args = parser.parse_args()

    # Validate files exist
    price_path = Path(args.price_file)
    ff_path = Path(args.ff_file)

    if not price_path.exists():
        print(f"[ERROR] Price file not found: {price_path}")
        sys.exit(1)

    if not ff_path.exists():
        print(f"[ERROR] Free float file not found: {ff_path}")
        sys.exit(1)

    print("=" * 70)
    print("Price/Volume/Free Float Backfill")
    print("=" * 70)
    print(f"Price file: {price_path.name}")
    print(f"Free float file: {ff_path.name}")
    print("=" * 70)

    conn = get_connection()

    try:
        # Step 1: Backfill prices/volumes
        price_count = backfill_prices(conn, str(price_path))

        # Step 2: Backfill free float
        ff_count = backfill_free_float(conn, str(ff_path))

        # Step 3: Verification
        verify_backfill(conn)

        print("\n[SUCCESS] Backfill completed successfully")
        print("\nNext steps:")
        print("  1. Verify data quality with sample queries")
        print("  2. Run normalization calculations (Sff, Z-Score)")
        print("  3. Update daily data loader to include new data sources")

    except Exception as e:
        print(f"\n[ERROR] Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
