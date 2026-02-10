"""Add price/volume/free float columns to existing investor_flows table

This migration extends the investor_flows table with 5 new columns:
- close_price: 종가 (closing price)
- trading_volume: 거래량 (trading volume)
- trading_value: 거래대금 (trading value in won)
- free_float_shares: 유통주식수 (free float shares)
- free_float_ratio: 유통비율 (free float ratio as percentage)

Usage:
    python scripts/migrations/migrate_add_columns.py
"""

import sqlite3
from pathlib import Path


def migrate():
    """Add new columns to investor_flows table"""

    db_path = Path(__file__).parent.parent.parent / 'data/processed/investor_data.db'

    if not db_path.exists():
        print(f"[ERROR] Database not found at {db_path}")
        print("Please run load_initial_data.py first")
        return False

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("=" * 70)
    print("Database Schema Migration: Add Columns")
    print("=" * 70)

    # Check current schema
    cursor.execute("PRAGMA table_info(investor_flows)")
    existing_columns = [row[1] for row in cursor.fetchall()]
    print(f"\n[INFO] Current columns: {len(existing_columns)}")

    # Define new columns
    new_columns = [
        ('close_price', 'REAL'),
        ('trading_volume', 'BIGINT'),
        ('trading_value', 'BIGINT'),
        ('free_float_shares', 'BIGINT'),
        ('free_float_ratio', 'REAL')
    ]

    print(f"[INFO] Adding {len(new_columns)} new columns...\n")

    # Add columns (SQLite allows ALTER TABLE ADD COLUMN)
    added_count = 0
    skipped_count = 0

    for col_name, col_type in new_columns:
        if col_name in existing_columns:
            print(f"  ⊘ {col_name} - already exists, skipping")
            skipped_count += 1
        else:
            try:
                cursor.execute(f'ALTER TABLE investor_flows ADD COLUMN {col_name} {col_type}')
                print(f"  ✓ {col_name} - added successfully")
                added_count += 1
            except sqlite3.OperationalError as e:
                print(f"  ✗ {col_name} - failed: {e}")

    conn.commit()

    # Verify final schema
    cursor.execute("PRAGMA table_info(investor_flows)")
    final_columns = [row[1] for row in cursor.fetchall()]

    print("\n" + "=" * 70)
    print("Migration Summary")
    print("=" * 70)
    print(f"Columns added: {added_count}")
    print(f"Columns skipped: {skipped_count}")
    print(f"Total columns: {len(final_columns)}")
    print("=" * 70)

    # Verify all new columns present
    all_present = all(col_name in final_columns for col_name, _ in new_columns)

    if all_present:
        print("\n[SUCCESS] Migration completed successfully")
        print("\nNext steps:")
        print("  1. Run backfill script to populate historical data")
        print("  2. Update daily data loader to include new data sources")
    else:
        print("\n[WARNING] Some columns may be missing")
        print("Please check the output above")

    conn.close()
    return all_present


if __name__ == '__main__':
    migrate()
