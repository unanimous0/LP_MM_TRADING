#!/bin/bash
# 수급 레짐 스캐너 - 전체 실행 스크립트

echo "================================================================================"
echo "🔍 수급 레짐 스캐너 - 전체 실행"
echo "================================================================================"
echo ""

# 1단계: CSV 리포트 생성
echo "📊 [1/2] CSV 리포트 생성 중..."
python3 scripts/analysis/regime_scanner.py --save-csv output/regime_report.csv --verbose

if [ $? -ne 0 ]; then
    echo "❌ CSV 생성 실패"
    exit 1
fi

echo ""
echo "✅ CSV 리포트 생성 완료"
echo ""

# 2단계: 엑셀 리포트 생성
echo "📊 [2/2] 엑셀 리포트 생성 중..."
python3 scripts/analysis/generate_excel_report.py \
    --input output/regime_report.csv \
    --output output/regime_report_final.xlsx

if [ $? -ne 0 ]; then
    echo "❌ 엑셀 생성 실패"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✅ 모든 작업 완료!"
echo "================================================================================"
echo ""
echo "📁 생성된 파일:"
echo "  - output/regime_report.csv"
echo "  - output/regime_report_final.xlsx"
echo ""
echo "💡 엑셀 파일을 열어서 확인하세요!"
echo ""
