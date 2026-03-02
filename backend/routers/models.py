"""Router de modelos: gerenciamento de modelos HuggingFace."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.model_manager import get_model_manager, MODELOS_DISPONIVEIS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["modelos"])

_gerenciador = get_model_manager()


# ---------------------------------------------------------------------------
# Modelos de requisição e resposta
# ---------------------------------------------------------------------------


class ModeloInfo(BaseModel):
    """Informações sobre um modelo disponível."""

    nome: str
    descricao: str
    num_camadas: int
    num_cabecas: int
    d_model: int
    tipo: str
    carregado: bool


class ListarModelosResponse(BaseModel):
    """Lista de modelos disponíveis."""

    modelos: list[ModeloInfo]
    total: int
    nota: str


class CarregarModeloRequest(BaseModel):
    """Requisição para carregar um modelo."""

    nome_modelo: str = Field(
        default="distilbert-base-uncased",
        description="Nome do modelo no HuggingFace Hub",
        examples=["distilbert-base-uncased", "bert-base-uncased", "gpt2"],
    )


class CarregarModeloResponse(BaseModel):
    """Resposta após carregar (ou tentar carregar) um modelo."""

    nome: str
    carregado: bool
    descricao: str
    num_camadas: int
    num_cabecas: int
    d_model: int
    tipo: str
    erro: str | None
    mensagem: str


class AtencaoRealRequest(BaseModel):
    """Requisição para obter pesos de atenção reais."""

    nome_modelo: str = Field(
        default="distilbert-base-uncased",
        description="Nome do modelo carregado",
    )
    texto: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Texto para análise de atenção",
        examples=["O gato senta no tapete"],
    )
    camada: int | None = Field(
        default=None,
        ge=1,
        description="Camada específica a retornar (None = todas)",
    )


class PesosAtencaoResponse(BaseModel):
    """Pesos de atenção reais de um modelo HuggingFace."""

    modelo: str
    texto: str
    tokens: list[str]
    num_tokens: int
    camadas: list[dict[str, object]]
    explicacao: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/available-models", response_model=ListarModelosResponse)
async def listar_modelos_disponiveis() -> ListarModelosResponse:
    """Lista modelos HuggingFace disponíveis para carregamento.

    Inclui metadados arquiteturais e status de carregamento atual.
    """
    infos = _gerenciador.listar_modelos_disponiveis()
    modelos = [
        ModeloInfo(
            nome=info.nome,
            descricao=info.descricao,
            num_camadas=info.num_camadas,
            num_cabecas=info.num_cabecas,
            d_model=info.d_model,
            tipo=info.tipo,
            carregado=info.carregado,
        )
        for info in infos
    ]

    return ListarModelosResponse(
        modelos=modelos,
        total=len(modelos),
        nota=(
            "O carregamento inicial pode levar alguns minutos pois baixa os pesos do modelo. "
            "Modelos carregados ficam em cache na memória para requisições subsequentes. "
            "Use DistilBERT para demonstrações mais rápidas."
        ),
    )


@router.post("/load-model", response_model=CarregarModeloResponse)
async def carregar_modelo(
    body: CarregarModeloRequest,
) -> CarregarModeloResponse:
    """Carrega um modelo HuggingFace de forma assíncrona.

    O download e carregamento ocorrem em uma thread separada para não
    bloquear o event loop. Modelos são cacheados após o primeiro carregamento.
    """
    if body.nome_modelo not in MODELOS_DISPONIVEIS:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Modelo '{body.nome_modelo}' não está na lista de modelos suportados. "
                f"Disponíveis: {list(MODELOS_DISPONIVEIS.keys())}"
            ),
        )

    logger.info("Requisição para carregar modelo.", extra={"modelo": body.nome_modelo})
    info = await _gerenciador.carregar_modelo(body.nome_modelo)

    mensagem = (
        f"Modelo '{info.nome}' carregado com sucesso e pronto para inferência."
        if info.carregado
        else f"Falha ao carregar modelo '{info.nome}': {info.erro}"
    )

    return CarregarModeloResponse(
        nome=info.nome,
        carregado=info.carregado,
        descricao=info.descricao,
        num_camadas=info.num_camadas,
        num_cabecas=info.num_cabecas,
        d_model=info.d_model,
        tipo=info.tipo,
        erro=info.erro,
        mensagem=mensagem,
    )


@router.post("/real-attention", response_model=PesosAtencaoResponse)
async def atencao_real(body: AtencaoRealRequest) -> PesosAtencaoResponse:
    """Extrai pesos de atenção reais de um modelo HuggingFace carregado.

    Diferente das simulações, esses são os pesos reais aprendidos durante
    o pré-treinamento, mostrando o que o modelo realmente aprendeu a focar.

    Requer que o modelo tenha sido carregado via /load-model primeiro.
    """
    try:
        resultado = await _gerenciador.obter_pesos_atencao_reais(
            body.nome_modelo, body.texto
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    camadas = resultado.camadas
    if body.camada is not None:
        camadas_filtradas = [
            c for c in resultado.camadas if c["camada"] == body.camada
        ]
        if not camadas_filtradas:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"Camada {body.camada} não encontrada. "
                    f"O modelo '{body.nome_modelo}' tem {len(resultado.camadas)} camadas "
                    f"(1 a {len(resultado.camadas)})."
                ),
            )
        camadas = camadas_filtradas

    info_modelo = MODELOS_DISPONIVEIS.get(body.nome_modelo, {})

    return PesosAtencaoResponse(
        modelo=resultado.modelo,
        texto=resultado.texto,
        tokens=resultado.tokens,
        num_tokens=resultado.num_tokens,
        camadas=camadas,
        explicacao=(
            f"Pesos de atenção reais do modelo '{body.nome_modelo}' "
            f"({'encoder bidirecional' if info_modelo.get('tipo') == 'encoder' else 'decoder causal'}). "
            f"O texto foi tokenizado em {resultado.num_tokens} tokens "
            f"(incluindo tokens especiais como [CLS] e [SEP]). "
            "Camadas iniciais tendem a capturar padrões sintáticos locais; "
            "camadas finais capturam relações semânticas de longa distância."
        ),
    )
