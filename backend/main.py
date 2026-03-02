"""Ponto de entrada da API — plataforma educacional de LLMs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Garantir que o diretório pai está no sys.path para imports relativos
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import (
    attention,
    embeddings,
    inference,
    models,
    setup,
    tokenization,
    training,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Criação da aplicação
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Plataforma Educacional de Transformers",
    description=(
        "API para demonstração interativa dos componentes de modelos Transformer: "
        "tokenização, embeddings, mecanismo de atenção, treinamento e inferência. "
        "Desenvolvida como recurso educacional para visualização de LLMs."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

API_PREFIX = "/api"

app.include_router(tokenization.router, prefix=API_PREFIX)
app.include_router(embeddings.router, prefix=API_PREFIX)
app.include_router(attention.router, prefix=API_PREFIX)
app.include_router(training.router, prefix=API_PREFIX)
app.include_router(inference.router, prefix=API_PREFIX)
app.include_router(models.router, prefix=API_PREFIX)
app.include_router(setup.router, prefix=API_PREFIX)

# ---------------------------------------------------------------------------
# Endpoint raiz
# ---------------------------------------------------------------------------


@app.get("/", tags=["saúde"])
async def raiz() -> dict[str, str]:
    """Health check da API.

    Returns:
        Dicionário com status e versão da API.
    """
    return {
        "status": "online",
        "servico": "Plataforma Educacional de Transformers",
        "versao": "1.0.0",
        "documentacao": "/docs",
    }


@app.get("/health", tags=["saúde"])
async def health_check() -> dict[str, str]:
    """Endpoint de health check para monitoramento.

    Returns:
        Dicionário confirmando que o serviço está operacional.
    """
    return {"status": "healthy"}
