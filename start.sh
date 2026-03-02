#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  LLM Explorer - Plataforma Educacional"
echo "=========================================="

# Backend
echo ""
echo "[1/2] Iniciando backend FastAPI..."
cd "$ROOT_DIR/backend"
if [ ! -d ".venv" ]; then
  echo "  Criando virtualenv..."
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt
echo "  Backend: http://localhost:8000"
echo "  Docs:    http://localhost:8000/docs"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Frontend
echo ""
echo "[2/2] Iniciando frontend React..."
cd "$ROOT_DIR/frontend"
if [ ! -d "node_modules" ]; then
  echo "  Instalando dependencias npm..."
  npm install
fi
echo "  Frontend: http://localhost:5173"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "=========================================="
echo "  Tudo pronto!"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000/docs"
echo "  Ctrl+C para encerrar"
echo "=========================================="

cleanup() {
  echo ""
  echo "Encerrando..."
  kill $BACKEND_PID 2>/dev/null
  kill $FRONTEND_PID 2>/dev/null
  exit 0
}

trap cleanup SIGINT SIGTERM
wait
