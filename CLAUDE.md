# 한국 주식 외국인/기관 투자자 수급 분석 프로그램

## [Status]
- 현재 작업: 프로젝트 초기 설정 완료
- 마지막 업데이트: 2026-02-09
- 다음 시작점: 요구사항 정리 후 데이터 수집 모듈 설계
- 현재 브랜치: main

## [Progress]
- ✅ 2026-02-09: GitHub 저장소 연결 완료
- ✅ 2026-02-09: 워크플로우 체계 수립
- ✅ 2026-02-09: 프로젝트 초기 구조 설정

## [Next Steps]
1. Infomax API 연동 가능 여부 확인
2. 분석 요구사항 상세 정리
3. 데이터 수집 모듈 설계 및 구현

## [Tech Stack]
- Python 3.10+
- 데이터 수집: Infomax API / pandas (엑셀 파일)
- 데이터 분석: pandas, numpy
- 시각화: matplotlib, seaborn (추후 추가 가능)
- 버전 관리: Git & GitHub

## [Project Structure]
```
Investor_Analysis/
├── CLAUDE.md              # 프로젝트 문서 및 작업 상태
├── README.md              # 프로젝트 소개
├── requirements.txt       # 의존성 목록
├── .gitignore            # Git 제외 파일
├── data/                 # 데이터 저장 (엑셀, CSV 등)
├── src/                  # 소스 코드
│   ├── data_collector/   # 데이터 수집 모듈
│   ├── analyzer/         # 분석 모듈
│   └── visualizer/       # 시각화 모듈
└── tests/                # 테스트 코드
```

## [Workflow - 작업 시작 시]
1. `git pull origin main`
2. CLAUDE.md [Status] 확인
3. Claude에게: "CLAUDE.md 읽고 작업 이어서 해줘"

## [Workflow - 작업 종료 시]
1. Claude에게: "작업 상태 CLAUDE.md에 업데이트 해줘"
2. `git add .`
3. `git commit -m "[집/회사] 작업 내용"`
4. `git push origin main`

## [Data Source]
- **Primary**: Infomax API (Python 지원 확인 필요)
- **Fallback**: 엑셀 파일 직접 제공
- 기간: 유연하게 선택 가능하도록 설계
