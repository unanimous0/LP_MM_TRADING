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
echo "📊 [2/4] 엑셀 리포트 생성 중..."
python3 scripts/analysis/generate_excel_report.py \
    --input output/regime_report.csv \
    --output output/regime_report_final.xlsx

if [ $? -ne 0 ]; then
    echo "❌ 엑셀 생성 실패"
    exit 1
fi

echo ""
echo "✅ 엑셀 리포트 생성 완료"
echo ""

# 3단계: HTML 리포트 생성
echo "🌐 [3/4] HTML 리포트 생성 중..."
python3 scripts/analysis/generate_html_report.py \
    --input output/regime_report.csv \
    --output output/regime_report_final.html

if [ $? -ne 0 ]; then
    echo "❌ HTML 생성 실패"
    exit 1
fi

echo ""
echo "✅ HTML 리포트 생성 완료"
echo ""

# 4단계: Markdown 리포트 생성
echo "📝 [4/4] Markdown 리포트 생성 중..."
python3 scripts/analysis/generate_md_report.py \
    --input output/regime_report.csv \
    --output output/regime_report_final.md

if [ $? -ne 0 ]; then
    echo "❌ Markdown 생성 실패"
    exit 1
fi

echo ""
echo "✅ Markdown 리포트 생성 완료"
echo ""

# 중간 파일 삭제 (최종 파일만 남기기)
rm -f output/regime_report.csv

echo ""
echo "================================================================================"
echo "✅ 모든 작업 완료!"
echo "================================================================================"
echo ""
echo "📁 생성된 파일:"
echo "  - output/regime_report_final.xlsx  (📊 엑셀 리포트 - 용어 설명 포함, 7개 시트)"
echo "  - output/regime_report_final.csv   (📄 CSV 데이터 - 전체 데이터)"
echo "  - output/regime_report_final.html  (🌐 HTML 대시보드 - 인터랙티브 차트)"
echo "  - output/regime_report_final.md    (📝 Markdown 리포트 - GitHub/Obsidian)"
echo ""
echo "💡 사용 가이드:"
echo "  📊 엑셀: 0.용어설명 시트에서 패턴/시그널/메트릭 정의 확인"
echo "  🌐 HTML: 브라우저에서 열기 (차트 인터랙티브, 정렬 가능)"
echo "  📝 MD:   GitHub/Obsidian 등에서 읽기 (텍스트 기반)"
echo ""
