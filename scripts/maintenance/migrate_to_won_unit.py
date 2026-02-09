"""
데이터 단위 마이그레이션: 천원 → 원

한 번만 실행하는 마이그레이션 스크립트
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_connection


def migrate():
    conn = get_connection()
    cursor = conn.cursor()

    print("\n" + "=" * 70)
    print("데이터 단위 마이그레이션: 천원 → 원")
    print("=" * 70)

    # 백업 확인
    backup_path = Path('data/processed/investor_data.db.backup_before_migration')
    if not backup_path.exists():
        print("[ERROR] 백업 파일이 없습니다. 먼저 백업을 생성하세요.")
        print("실행: cp data/processed/investor_data.db data/processed/investor_data.db.backup_before_migration")
        return

    print(f"[OK] 백업 파일 확인: {backup_path}")

    # 마이그레이션 전 샘플 데이터
    print("\n[INFO] 마이그레이션 전 샘플 데이터 (삼성전자 2026-01-20):")
    cursor.execute("""
        SELECT foreign_net_amount, institution_net_amount
        FROM investor_flows
        WHERE stock_code = '005930' AND trade_date = '2026-01-20'
    """)
    before = cursor.fetchone()
    if before:
        print(f"  외국인 순매수: {before[0]:,} (천원 단위)")
        print(f"  기관 순매수: {before[1]:,} (천원 단위)")
        print(f"  → 억원: {before[0] / 100_000:,.2f}억원, {before[1] / 100_000:,.2f}억원")
    else:
        print("[WARN] 샘플 데이터를 찾을 수 없습니다.")

    # 총 레코드 수 확인
    cursor.execute("SELECT COUNT(*) FROM investor_flows")
    total_records = cursor.fetchone()[0]
    print(f"\n[INFO] 총 {total_records:,}개 레코드를 마이그레이션합니다.")

    # 사용자 확인
    print("\n" + "=" * 70)
    print("[WARNING] 이 작업은 불가역적입니다!")
    print("마이그레이션을 진행하면 모든 금액 데이터가 1,000배 증가합니다.")
    print("=" * 70)

    # 마이그레이션 실행
    print("\n[INFO] 마이그레이션 실행 중...")
    cursor.execute("""
        UPDATE investor_flows
        SET
            foreign_net_amount = foreign_net_amount * 1000,
            foreign_net_volume = foreign_net_volume * 1000,
            institution_net_amount = institution_net_amount * 1000,
            institution_net_volume = institution_net_volume * 1000
    """)

    conn.commit()
    print("[OK] 마이그레이션 완료!")

    # 마이그레이션 후 샘플 데이터
    print("\n[INFO] 마이그레이션 후 샘플 데이터 (삼성전자 2026-01-20):")
    cursor.execute("""
        SELECT foreign_net_amount, institution_net_amount
        FROM investor_flows
        WHERE stock_code = '005930' AND trade_date = '2026-01-20'
    """)
    after = cursor.fetchone()
    if after:
        print(f"  외국인 순매수: {after[0]:,} (원 단위)")
        print(f"  기관 순매수: {after[1]:,} (원 단위)")
        print(f"  → 억원: {after[0] / 100_000_000:,.2f}억원, {after[1] / 100_000_000:,.2f}억원")

        # 검증
        if before and after[0] == before[0] * 1000 and after[1] == before[1] * 1000:
            print("\n[SUCCESS] 마이그레이션 검증 완료! ✅")
            print(f"  외국인: {before[0]:,} → {after[0]:,} (× 1000)")
            print(f"  기관: {before[1]:,} → {after[1]:,} (× 1000)")
        else:
            print("\n[ERROR] 마이그레이션 검증 실패!")
            print("백업에서 복원을 권장합니다.")
    else:
        print("[WARN] 샘플 데이터를 찾을 수 없습니다.")

    print("\n" + "=" * 70)
    print("다음 단계:")
    print("1. ExcelCollector 수정 (× 1000 추가)")
    print("2. 분석 스크립트 재실행")
    print("3. 문서 업데이트")
    print("=" * 70 + "\n")

    conn.close()


if __name__ == '__main__':
    migrate()
