"""공통 UI 상수 — 여러 페이지에서 반복되는 텍스트/CSS를 한 곳에서 관리"""

INSTITUTION_WEIGHT_HELP = """기관 수급이 외국인과 같은 방향일 때만 가중치가 반영됩니다.

· 0.0 = 외국인 신호만 사용
· 0.3 = 기관 동조 시 30% 추가 반영 (기본값)
· 1.0 = 기관 동조 시 외국인과 동등하게 반영"""

WIDGET_BORDER_CSS = """<style>
div[data-baseweb="select"]>div{border-color:#333!important}
div[data-baseweb="input"] input,div[data-baseweb="input"]>div{border-color:#333!important}
[data-testid="stDateInput"]>div>div>div{border-color:#333!important}
[data-testid="stExpander"]{border-color:#222!important}
</style>"""
