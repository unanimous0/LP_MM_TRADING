# 한국 주식 외국인/기관 투자자 수급 분석 프로그램

한국 주식 시장의 외국인 및 기관 투자자 수급 데이터를 수집하고 분석하는 Python 프로그램입니다.

## 프로젝트 개요

이 프로젝트는 한국 주식 시장에서 외국인과 기관 투자자의 매매 동향을 분석하여 투자 인사이트를 제공합니다.

## 주요 기능 (개발 예정)

- 외국인/기관 투자자 수급 데이터 수집
- 종목별 매매 동향 분석
- 시각화 및 리포트 생성

## 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/unanimous0/Investor_Analysis.git
cd Investor_Analysis
```

### 2. 가상환경 생성 및 활성화
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

> **Note**: 현재 프로젝트는 초기 설정 단계입니다. 구체적인 사용 방법은 추후 업데이트됩니다.

## 프로젝트 구조

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

## 데이터 소스

- **Primary**: Infomax API (연동 확인 중)
- **Fallback**: 엑셀 파일 직접 제공

## 기술 스택

- Python 3.10+
- pandas, numpy
- openpyxl (엑셀 파일 처리)
- matplotlib, seaborn (시각화, 추후 추가)

## 개발 현황

현재 상태 및 다음 작업 계획은 [CLAUDE.md](CLAUDE.md)를 참조하세요.

## 라이선스

MIT License

## 문의

프로젝트 관련 문의사항은 GitHub Issues를 이용해 주세요.
