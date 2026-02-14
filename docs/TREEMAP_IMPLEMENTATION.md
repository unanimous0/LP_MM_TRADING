# Treemap 시각화 구현 가이드

## 📋 목차
- [개요](#개요)
- [구현 과정](#구현-과정)
  - [1단계: 초기 Grid 히트맵 구현](#1단계-초기-grid-히트맵-구현)
  - [2단계: Treemap으로 전환](#2단계-treemap으로-전환)
  - [3단계: 스타일 개선](#3단계-스타일-개선)
- [최종 결과](#최종-결과)
- [기술 스택](#기술-스택)
- [주요 코드 설명](#주요-코드-설명)
- [파일 구조](#파일-구조)

---

## 개요

수급 레짐 스캐너 HTML 리포트에 **D3.js Treemap** 시각화를 추가하여 섹터별 종목 분포와 점수를 한눈에 볼 수 있도록 개선했습니다.

### 목표
- ✅ 섹터별 수급 집중도를 시각적으로 표현
- ✅ 종합점수에 비례하는 박스 크기
- ✅ 직관적인 색상 코딩 (빨강 → 노랑 → 초록)
- ✅ 인터랙티브 툴팁으로 상세 정보 제공

---

## 구현 과정

### 1단계: 초기 Grid 히트맵 구현

#### 날짜
2026-02-14 (초기 구현)

#### 구현 내용
- D3.js v7을 사용한 기본 Grid 히트맵
- X축: 섹터 내 종목들 (순서대로 나열)
- Y축: 섹터명
- 색상: YlOrRd 스케일 (40~100점 범위)
- 툴팁: 종목명, 코드, 패턴, 점수, 시그널

#### 문제점
- 사용자가 원하는 형태가 Grid가 아닌 **Treemap** 형식이었음
- 박스 크기가 모두 동일하여 점수 차이 표현 부족
- 섹터 구분이 명확하지 않음

---

### 2단계: Treemap으로 전환

#### 날짜
2026-02-14 (리팩토링)

#### 구현 내용

**데이터 구조 변경**
```python
# Before: 플랫 리스트
heatmap_data = [
    {
        'sector': '반도체',
        'stock_code': '005930',
        'stock_name': '삼성전자',
        'combined_score': 85.5,
        ...
    },
    ...
]

# After: 계층 구조
treemap_data = {
    'name': 'root',
    'children': [
        {
            'name': '반도체',
            'children': [
                {
                    'name': '삼성전자',
                    'value': 85.5,  # 박스 크기 결정
                    'stock_code': '005930',
                    'combined_score': 85.5,
                    ...
                },
                ...
            ]
        },
        ...
    ]
}
```

**D3 Treemap 레이아웃 적용**
```javascript
// 계층 구조 생성
const root = d3.hierarchy(treemapData)
    .sum(d => d.value)  // 박스 크기 = 종합점수
    .sort((a, b) => b.value - a.value);  // 점수 높은 순

// Treemap 레이아웃
const treemap = d3.treemap()
    .size([width, height - margin.top])
    .padding(1)  // 박스 간 간격
    .paddingOuter(2)
    .paddingTop(24)  // 섹터 이름 공간
    .round(true);

treemap(root);
```

**주요 특징**
- ✅ 박스 크기가 종합점수에 비례
- ✅ 섹터별로 그룹화된 레이아웃
- ✅ 색상 스케일: 빨강(낮음) → 노랑(중간) → 초록(높음)
- ✅ 둥근 모서리 (border-radius: 3px)

#### 남은 문제점
1. **가독성 저하**: 모든 텍스트가 흰색이라 밝은 배경(초록/노랑)에서 보기 어려움
2. **범례 위치**: 우측 상단에 있어 섹터명과 겹침
3. **전체적인 디자인**: 세련되지 못함

---

### 3단계: 스타일 개선

#### 날짜
2026-02-14 (최종 개선)

#### 개선사항 1: 텍스트 가독성 향상

**배경 명도에 따른 자동 색상 조정**
```javascript
// 배경색 명도 계산 → 글자색 자동 선택
function getTextColor(bgColor) {
    const color = d3.color(bgColor);
    const luminance = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;
    return luminance > 140 ? '#1F2937' : '#FFFFFF';
    // 밝으면 진한 회색, 어두우면 흰색
}

// 적용
.attr("fill", d => getTextColor(colorScale(d.data.combined_score)))
```

**결과**
- 빨강 배경 (낮은 점수) → 흰색 글자
- 초록 배경 (높은 점수) → 검은색 글자
- 노랑 배경 (중간 점수) → 검은색 글자

**텍스트 배치 개선**
```javascript
// Before: 왼쪽 상단 정렬
.attr("x", 4)
.attr("y", 16)
.attr("text-anchor", "start")

// After: 중앙 정렬
.attr("x", d => (d.x1 - d.x0) / 2)
.attr("y", d => (d.y1 - d.y0) / 2)
.attr("text-anchor", "middle")
.attr("dominant-baseline", "middle")
```

#### 개선사항 2: 범례 위치 최적화

**Before: 우측 상단**
```javascript
const legend = svg.append("g")
    .attr("transform", `translate(${width - legendWidth - 20}, 20)`);
```

**After: 상단 중앙**
```javascript
const legend = svg.append("g")
    .attr("transform", `translate(${(width - legendWidth) / 2}, 20)`);
```

**범례 구성**
```
┌─────────────────────────────────────┐
│          종합점수 (타이틀)              │
├─────────────────────────────────────┤
│  [빨강 ─ 주황 ─ 노랑 ─ 연두 ─ 초록]    │ ← 그라데이션 바
├─────────────────────────────────────┤
│ 낮음(40점)    중간      높음(100점)    │ ← 3단계 레이블
└─────────────────────────────────────┘
```

#### 개선사항 3: 전체 스타일링

**섹터 레이블**
```javascript
// 반투명 검정 배경 추가
mainGroup.append("rect")
    .attr("x", x0)
    .attr("y", y0)
    .attr("width", labelWidth)
    .attr("height", 22)
    .attr("fill", "rgba(31, 41, 55, 0.85)")  // 반투명 검정
    .attr("rx", 4);  // 둥근 모서리

// 흰색 굵은 글씨
mainGroup.append("text")
    .attr("x", x0 + 8)
    .attr("y", y0 + 14)
    .text(sector.data.name)
    .attr("font-size", "12px")
    .attr("font-weight", "700")
    .attr("fill", "#FFFFFF");
```

**호버 효과**
```javascript
.on("mouseover", function(event, d) {
    d3.select(this)
        .attr("stroke", "#000000")
        .attr("stroke-width", 2.5)
        .style("filter", "brightness(1.1)");  // 밝기 증가

    tooltip.transition().duration(150).style("opacity", 1);
    tooltip.html(/* 상세 정보 */);
})
.on("mouseout", function() {
    d3.select(this)
        .attr("stroke", "#FFFFFF")
        .attr("stroke-width", 1.5)
        .style("filter", "none");

    tooltip.transition().duration(300).style("opacity", 0);
})
```

**툴팁 디자인**
```html
<!-- 구분선으로 제목 강조 -->
<div style="border-bottom: 2px solid #60A5FA; padding-bottom: 8px; margin-bottom: 8px;">
    <strong style="font-size: 17px; color: #60A5FA;">삼성전자</strong>
</div>

<!-- 정보 구조화 -->
<div style="line-height: 1.8;">
    <span style="color: #9CA3AF;">종목코드:</span>
    <strong style="color: #E5E7EB;">005930</strong><br/>

    <span style="color: #9CA3AF;">섹터:</span>
    <strong style="color: #E5E7EB;">반도체</strong><br/>

    <span style="color: #9CA3AF;">종합점수:</span>
    <strong style="color: #34D399; font-size: 16px;">85.5점</strong><br/>

    ...
</div>
```

**박스 크기별 텍스트 조정**
```javascript
// 박스 크기에 따라 텍스트 표시/생략
.text(d => {
    const width = d.x1 - d.x0;
    const height = d.y1 - d.y0;

    if (width > 70 && height > 35) {
        // 큰 박스: 전체 종목명
        return d.data.name.length > 8
            ? d.data.name.substring(0, 7) + '...'
            : d.data.name;
    } else if (width > 45 && height > 25) {
        // 중간 박스: 축약
        return d.data.name.length > 5
            ? d.data.name.substring(0, 4) + '...'
            : d.data.name;
    }
    // 작은 박스: 생략
    return '';
})

// 폰트 크기도 자동 조정
.attr("font-size", d => {
    const width = d.x1 - d.x0;
    if (width > 120) return "14px";
    if (width > 80) return "12px";
    if (width > 50) return "10px";
    return "8px";
})
```

---

## 최종 결과

### 주요 특징

#### 1. 시각적 표현
- **박스 크기**: 종합점수에 비례 (점수 높을수록 큰 박스)
- **색상 코딩**:
  - 빨강 (40~60점): 낮은 점수
  - 노랑 (60~80점): 중간 점수
  - 초록 (80~100점): 높은 점수
- **섹터 구분**: 각 섹터가 하나의 큰 영역으로 묶임

#### 2. 가독성
- **자동 색상 조정**: 배경색에 따라 텍스트 색상 자동 선택
- **중앙 정렬**: 종목명과 점수를 박스 중앙에 배치
- **크기별 최적화**: 박스 크기에 따라 텍스트 표시/생략

#### 3. 인터랙티브
- **호버 효과**: 마우스 올리면 밝아지고 테두리 강조
- **상세 툴팁**: 종목명, 코드, 섹터, 패턴, 점수, 시그널 정보 표시
- **시각적 피드백**: 마우스 오버/아웃 시 부드러운 전환 애니메이션

#### 4. 디자인
- **둥근 모서리**: 모든 박스에 3px border-radius
- **깔끔한 테두리**: 얇은 흰색 테두리 (1.5px)
- **섹터 레이블**: 반투명 검정 배경 + 흰색 굵은 글씨
- **범례**: 상단 중앙 배치로 섹터명과 겹치지 않음

### 생성 파일
```
output/regime_report_final.html  (55KB)
  ↳ 섹터별 Treemap 시각화 포함
  ↳ Chart.js 차트 (패턴별 분포, 섹터별 평균 점수)
  ↳ 종합 추천 순위 TOP 20 테이블
  ↳ 섹터별 수급 집중도 테이블
```

---

## 기술 스택

### 라이브러리
- **D3.js v7**: Treemap 레이아웃 및 시각화
- **Tailwind CSS (CDN)**: 반응형 레이아웃 및 스타일링
- **Chart.js (CDN)**: 패턴별/섹터별 차트

### 주요 D3 API
```javascript
d3.hierarchy()         // 계층 구조 생성
d3.treemap()          // Treemap 레이아웃
d3.scaleSequential()  // 연속형 색상 스케일
d3.interpolateRdYlGn  // 빨강-노랑-초록 그라데이션
d3.color()            // 색상 파싱 및 변환
```

### 색상 팔레트
```javascript
// 종합점수 스케일
const colorScale = d3.scaleSequential()
    .domain([40, 100])
    .interpolator(d3.interpolateRdYlGn);

// 패턴 색상
const patternColors = {
    '모멘텀형': '#EF4444',  // 빨강
    '지속형': '#3B82F6',    // 파랑
    '전환형': '#F59E0B'     // 주황
};

// 텍스트 색상
밝은 배경: '#1F2937'  // 진한 회색
어두운 배경: '#FFFFFF'  // 흰색
```

---

## 주요 코드 설명

### 1. 데이터 준비 (Python)

**파일**: `scripts/analysis/generate_html_report.py`

```python
# Treemap 데이터 준비 (계층 구조: 섹터 > 종목)
treemap_data = {
    'name': 'root',
    'children': []
}

for _, sector_row in sector_concentration.iterrows():
    sector = sector_row['sector']
    sector_stocks = df[df['sector'] == sector].nlargest(10, 'combined_score')

    sector_node = {
        'name': sector,
        'children': []
    }

    for _, stock in sector_stocks.iterrows():
        sector_node['children'].append({
            'name': stock['stock_name'],
            'value': float(stock['combined_score']),  # 박스 크기
            'stock_code': stock['stock_code'],
            'pattern': stock['pattern'],
            'combined_score': float(stock['combined_score']),
            'signal_count': int(stock['signal_count']),
            'signal_list': stock['signal_list'] if pd.notna(stock['signal_list']) else '-'
        })

    treemap_data['children'].append(sector_node)
```

### 2. Treemap 레이아웃 (JavaScript)

```javascript
// 계층 구조 생성
const root = d3.hierarchy(treemapData)
    .sum(d => d.value)  // value 필드를 기준으로 박스 크기 계산
    .sort((a, b) => b.value - a.value);  // 내림차순 정렬

// Treemap 레이아웃 설정
const treemap = d3.treemap()
    .size([width, height - margin.top])
    .padding(1)        // 박스 간 간격 1px
    .paddingOuter(2)   // 외부 여백 2px
    .paddingTop(24)    // 상단 여백 24px (섹터 이름 공간)
    .round(true);      // 좌표를 정수로 반올림

// 레이아웃 적용
treemap(root);
```

### 3. 박스 렌더링

```javascript
// 메인 그룹 (상단 마진 적용)
const mainGroup = svg.append("g")
    .attr("transform", `translate(0, ${margin.top})`);

// 종목 박스 그리기
const leaves = mainGroup.selectAll("g")
    .data(root.leaves())  // 리프 노드만 선택 (종목)
    .enter()
    .append("g")
    .attr("transform", d => `translate(${d.x0},${d.y0})`);

// 박스 사각형
leaves.append("rect")
    .attr("width", d => d.x1 - d.x0)
    .attr("height", d => d.y1 - d.y0)
    .attr("fill", d => colorScale(d.data.combined_score))
    .attr("stroke", "#FFFFFF")
    .attr("stroke-width", 1.5)
    .attr("rx", 3)  // 둥근 모서리
    .style("cursor", "pointer")
    .on("mouseover", /* 호버 효과 */)
    .on("mouseout", /* 원상복구 */);
```

### 4. 텍스트 색상 자동 조정

```javascript
// 배경색의 명도(luminance)를 계산하여 텍스트 색상 결정
function getTextColor(bgColor) {
    const color = d3.color(bgColor);

    // RGB를 명도로 변환 (ITU-R BT.709 표준)
    const luminance = 0.299 * color.r + 0.587 * color.g + 0.114 * color.b;

    // 임계값 140 기준으로 색상 선택
    return luminance > 140 ? '#1F2937' : '#FFFFFF';
}

// 종목명 텍스트에 적용
leaves.append("text")
    .attr("fill", d => getTextColor(colorScale(d.data.combined_score)))
    // ... 기타 속성
```

### 5. 섹터 레이블

```javascript
// 각 섹터별로 레이블 추가
const sectorGroups = root.children;
sectorGroups.forEach(sector => {
    const sectorLeaves = sector.leaves();
    if (sectorLeaves.length === 0) return;

    // 섹터 영역의 좌상단 좌표
    const x0 = Math.min(...sectorLeaves.map(d => d.x0));
    const y0 = Math.min(...sectorLeaves.map(d => d.y0));
    const x1 = Math.max(...sectorLeaves.map(d => d.x1));

    // 섹터 이름 길이에 따라 배경 너비 조정
    const labelWidth = Math.min(sector.data.name.length * 10 + 16, x1 - x0);

    // 반투명 검정 배경
    mainGroup.append("rect")
        .attr("x", x0)
        .attr("y", y0)
        .attr("width", labelWidth)
        .attr("height", 22)
        .attr("fill", "rgba(31, 41, 55, 0.85)")
        .attr("rx", 4);

    // 흰색 텍스트
    mainGroup.append("text")
        .attr("x", x0 + 8)
        .attr("y", y0 + 14)
        .text(sector.data.name)
        .attr("font-size", "12px")
        .attr("font-weight", "700")
        .attr("fill", "#FFFFFF");
});
```

### 6. 범례 (상단 중앙)

```javascript
const legendWidth = 400;
const legendHeight = 20;

const legend = svg.append("g")
    .attr("transform", `translate(${(width - legendWidth) / 2}, 20)`);

// 타이틀
legend.append("text")
    .attr("x", legendWidth / 2)
    .attr("y", 0)
    .attr("text-anchor", "middle")
    .style("font-size", "16px")
    .style("font-weight", "700")
    .attr("fill", "#1F2937")
    .text("종합점수");

// 그라데이션 정의
const defs = svg.append("defs");
const linearGradient = defs.append("linearGradient")
    .attr("id", "legend-gradient")
    .attr("x1", "0%")
    .attr("x2", "100%");

linearGradient.selectAll("stop")
    .data(d3.range(0, 1.1, 0.1))
    .enter()
    .append("stop")
    .attr("offset", d => `${d * 100}%`)
    .attr("stop-color", d => colorScale(40 + d * 60));

// 그라데이션 바
legend.append("rect")
    .attr("y", 15)
    .attr("width", legendWidth)
    .attr("height", legendHeight)
    .style("fill", "url(#legend-gradient)")
    .attr("stroke", "#D1D5DB")
    .attr("stroke-width", 1)
    .attr("rx", 4);

// 레이블 (낮음/중간/높음)
legend.append("text")
    .attr("x", 0)
    .attr("y", legendHeight + 35)
    .attr("text-anchor", "start")
    .style("font-size", "13px")
    .style("font-weight", "600")
    .attr("fill", "#DC2626")  // 빨강
    .text("낮음 (40점)");

legend.append("text")
    .attr("x", legendWidth / 2)
    .attr("y", legendHeight + 35)
    .attr("text-anchor", "middle")
    .style("font-size", "13px")
    .style("font-weight", "600")
    .attr("fill", "#F59E0B")  // 주황
    .text("중간");

legend.append("text")
    .attr("x", legendWidth)
    .attr("y", legendHeight + 35)
    .attr("text-anchor", "end")
    .style("font-size", "13px")
    .style("font-weight", "600")
    .attr("fill", "#059669")  // 초록
    .text("높음 (100점)");
```

---

## 파일 구조

### 수정된 파일

```
scripts/analysis/generate_html_report.py
  ↳ Treemap 데이터 준비 로직 (Python)
  ↳ HTML 템플릿 생성 함수
  ↳ D3.js Treemap 스크립트 임베딩
```

### 관련 파일

```
scripts/analysis/
├── regime_scanner.py          # CSV 리포트 생성
├── generate_excel_report.py   # 엑셀 리포트 생성
├── generate_html_report.py    # HTML 리포트 생성 (Treemap 포함)
├── generate_md_report.py      # Markdown 리포트 생성
└── run_all.sh                 # 전체 실행 스크립트

output/
├── regime_report.csv          # 중간 CSV 파일 (자동 삭제)
├── regime_report_final.csv    # 최종 CSV 데이터
├── regime_report_final.xlsx   # 최종 엑셀 리포트
├── regime_report_final.html   # 최종 HTML 리포트 (Treemap 포함)
└── regime_report_final.md     # 최종 Markdown 리포트
```

---

## 실행 방법

### 전체 리포트 생성
```bash
# 모든 형식 (Excel, CSV, HTML, Markdown) 생성
bash scripts/analysis/run_all.sh
```

### HTML만 생성
```bash
# 1. CSV 생성
python scripts/analysis/regime_scanner.py --save-csv output/regime_report.csv

# 2. HTML 생성
python scripts/analysis/generate_html_report.py \
    --input output/regime_report.csv \
    --output output/regime_report_final.html
```

### 브라우저에서 열기
```bash
# Mac
open output/regime_report_final.html

# Linux
xdg-open output/regime_report_final.html

# Windows
start output/regime_report_final.html
```

---

## Git 커밋 히스토리

### 1차 커밋: Grid 히트맵 구현
```bash
git commit -m "[HTML 리포트] D3.js 히트맵 추가"
```

### 2차 커밋: Treemap 전환
```bash
git commit -m "[HTML 리포트] D3 Grid 히트맵 → Treemap 변경

- 계층 구조 데이터로 변경 (섹터 > 종목)
- 박스 크기: 종합점수에 비례
- 색상: 빨강(낮음) → 노랑(중간) → 초록(높음)
- 각 박스에 종목명 + 점수 표시
- 툴팁으로 상세 정보 (섹터, 패턴, 시그널) 표시
- 섹터 레이블 각 영역 상단에 배치"
```

### 3차 커밋: 스타일 개선
```bash
git commit -m "[HTML 리포트] Treemap 스타일 대폭 개선

개선사항:
1. 텍스트 가독성 향상
   - 배경 명도에 따라 글자색 자동 조정 (밝으면 검정, 어두우면 흰색)
   - 중앙 정렬로 가독성 개선

2. 범례 위치 개선
   - 우측 상단 → 상단 중앙으로 이동
   - 섹터명과 겹치지 않도록 배치
   - 타이틀 + 그라데이션 바 + 3단계 레이블 (낮음/중간/높음)

3. 전체 스타일링 개선
   - 둥근 모서리 (border-radius)
   - 섹터 레이블에 반투명 검정 배경 추가
   - 테두리 두께 최적화 (1.5px)
   - 호버 효과 개선 (brightness + stroke 강조)
   - 툴팁 디자인 개선 (구분선, 색상 강조)
   - 폰트 크기 및 가중치 최적화"
```

---

## 성능 고려사항

### 데이터 최적화
- 각 섹터당 상위 10개 종목만 표시
- 총 100개 내외의 박스로 렌더링 부하 최소화

### 렌더링 최적화
- D3 `.round(true)` 옵션으로 정수 좌표 사용
- 불필요한 DOM 조작 최소화
- 호버 효과에 transition 지속시간 최적화 (150~300ms)

### 반응형 디자인
- 고정 크기 (1200x800px) SVG
- 스크롤 가능한 컨테이너로 작은 화면 대응
- CSS `overflow-x: auto` 적용

---

## 향후 개선 방향

### 기능 추가
- [ ] 패턴별 필터링 (클릭 시 특정 패턴만 표시)
- [ ] 확대/축소 기능 (Zoom & Pan)
- [ ] 섹터 클릭 시 상세 정보 패널 표시
- [ ] 애니메이션 전환 효과

### 성능 개선
- [ ] 가상화 (Virtualization) 적용 (종목 수 증가 대비)
- [ ] WebGL 렌더링 검토 (수천 개 데이터 처리 시)

### 디자인 개선
- [ ] 다크 모드 지원
- [ ] 색상 테마 커스터마이징
- [ ] 모바일 반응형 레이아웃

---

## 참고 자료

### D3.js 공식 문서
- [D3 Treemap](https://d3js.org/d3-hierarchy/treemap)
- [D3 Hierarchy](https://d3js.org/d3-hierarchy/hierarchy)
- [D3 Scale Sequential](https://d3js.org/d3-scale/sequential)

### 예제 및 영감
- [Observable HQ - Treemap Examples](https://observablehq.com/@d3/treemap)
- [빅파이낸스 히트맵](https://finance.naver.com/) - 디자인 참고

### 색상 접근성
- [WCAG 2.1 Color Contrast](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html)
- [Coolors - Color Palette Generator](https://coolors.co/)

---

**문서 작성**: 2026-02-14
**버전**: 1.0
**작성자**: Claude Sonnet 4.5 + unanimous0
