#!/bin/bash
# Streamlit 실행 래퍼 — Tailscale URL 안내 포함

TAILSCALE_IP="100.64.229.73"
PORT=8501

echo ""
echo "  Whale Supply Streamlit 대시보드"
echo "  ────────────────────────────────"
echo "  Local URL:     http://localhost:${PORT}"
echo "  Tailscale URL: http://${TAILSCALE_IP}:${PORT}"
echo ""

cd "$(dirname "$0")"
exec venv/bin/streamlit run app/streamlit_app.py --server.port=${PORT}
