"""
외국인 ↔ 기관 데이터 Swap 마이그레이션

ExcelCollector의 컬럼 매핑 오류로 인해
외국인과 기관 데이터가 서로 바뀌어 저장됨.
이를 수정하는 마이그레이션 스크립트.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_connection


def swap_data():
    conn = get_connection()
    cursor = conn.cursor()

    print("\n" + "=" * 70)
    print("외국인 ↔ 기관 데이터 Swap 마이그레이션")
    print("=" * 70)

    # 백업 확인
    backup_path = Path('data/processed/investor_data.db.backup_before_swap')
    if backup_path.exists():
        print(f"[OK] 백업 파일 존재: {backup_path}")
    else:
        print("[WARN] 백업 파일이 없습니다.")
        response = input("백업을 생성하시겠습니까? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.copy('data/processed/investor_data.db', backup_path)
            print(f"[OK] 백업 생성 완료: {backup_path}")
        else:
            print("[ERROR] 백업 없이 진행할 수 없습니다.")
            return

    # Swap 전 샘플 데이터
    print("\n[INFO] Swap 전 샘플 데이터 (삼성전자 2026-01-20):")
    cursor.execute("""
        SELECT foreign_net_amount, institution_net_amount
        FROM investor_flows
        WHERE stock_code = '005930' AND trade_date = '2026-01-20'
    """)
    before = cursor.fetchone()
    if before:
        print(f"  foreign_net_amount (잘못 저장됨): {before[0]:,}원")
        print(f"  institution_net_amount (잘못 저장됨): {before[1]:,}원")
        print(f"  → 외국인: {before[0] / 100_000_000:,.2f}억원")
        print(f"  → 기관: {before[1] / 100_000_000:,.2f}억원")

    # 총 레코드 수 확인
    cursor.execute("SELECT COUNT(*) FROM investor_flows")
    total_records = cursor.fetchone()[0]
    print(f"\n[INFO] 총 {total_records:,}개 레코드를 Swap합니다.")

    print("\n" + "=" * 70)
    print("[WARNING] 외국인 ↔ 기관 데이터를 서로 바꿉니다!")
    print("=" * 70)

    # Swap 실행 (임시 컬럼 사용)
    print("\n[INFO] Swap 실행 중...")

    # 1. 임시 컬럼 추가
    cursor.execute("ALTER TABLE investor_flows ADD COLUMN temp_foreign_volume BIGINT")
    cursor.execute("ALTER TABLE investor_flows ADD COLUMN temp_foreign_amount BIGINT")

    # 2. 외국인 → 임시 저장
    cursor.execute("""
        UPDATE investor_flows
        SET
            temp_foreign_volume = foreign_net_volume,
            temp_foreign_amount = foreign_net_amount
    """)

    # 3. 기관 → 외국인
    cursor.execute("""
        UPDATE investor_flows
        SET
            foreign_net_volume = institution_net_volume,
            foreign_net_amount = institution_net_amount
    """)

    # 4. 임시 → 기관
    cursor.execute("""
        UPDATE investor_flows
        SET
            institution_net_volume = temp_foreign_volume,
            institution_net_amount = temp_foreign_amount
    """)

    # 5. 임시 컬럼 삭제
    cursor.execute("ALTER TABLE investor_flows DROP COLUMN temp_foreign_volume")
    cursor.execute("ALTER TABLE investor_flows DROP COLUMN temp_foreign_amount")

    conn.commit()
    print("[OK] Swap 완료!")

    # Swap 후 샘플 데이터
    print("\n[INFO] Swap 후 샘플 데이터 (삼성전자 2026-01-20):")
    cursor.execute("""
        SELECT foreign_net_amount, institution_net_amount
        FROM investor_flows
        WHERE stock_code = '005930' AND trade_date = '2026-01-20'
    """)
    after = cursor.fetchone()
    if after:
        print(f"  foreign_net_amount (수정됨): {after[0]:,}원")
        print(f"  institution_net_amount (수정됨): {after[1]:,}원")
        print(f"  → 외국인: {after[0] / 100_000_000:,.2f}억원")
        print(f"  → 기관: {after[1] / 100_000_000:,.2f}억원")

        # 검증
        if before and after[0] == before[1] and after[1] == before[0]:
            print("\n[SUCCESS] Swap 검증 완료! ✅")
            print(f"  외국인: {before[0]:,} → {after[0]:,}")
            print(f"  기관: {before[1]:,} → {after[1]:,}")
        else:
            print("\n[ERROR] Swap 검증 실패!")

    print("\n" + "=" * 70)
    print("다음 단계:")
    print("1. 분석 스크립트 재실행")
    print("2. 결과 확인")
    print("=" * 70 + "\n")

    conn.close()


if __name__ == '__main__':
    swap_data()
