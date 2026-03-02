"""Router de atenção: self-attention, multi-head e análise de importância."""

from __future__ import annotations

import logging

import numpy as np

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.multi_head_attention import MultiHeadAttention
from core.transformer_simulator import TransformerSimulator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/attention", tags=["atenção"])


# ---------------------------------------------------------------------------
# Modelos de requisição e resposta
# ---------------------------------------------------------------------------


class SelfAttentionRequest(BaseModel):
    """Requisição para self-attention passo a passo."""

    tokens: list[str] = Field(
        ...,
        min_length=1,
        description="Lista de tokens para processar",
        examples=[["o", "gato", "vê", "o", "rato"]],
    )
    d_model: int = Field(
        default=64,
        ge=8,
        le=256,
        description="Dimensão do modelo",
    )


class SelfAttentionResponse(BaseModel):
    """Resultado completo do self-attention passo a passo."""

    Q: list[list[float]]
    K: list[list[float]]
    V: list[list[float]]
    scores_escalados: list[list[float]]
    pesos_atencao: list[list[float]]
    saida: list[list[float]]
    tokens: list[str]
    num_tokens_reais: int
    fator_escala: float
    passos_explicados: list[dict[str, str]]


class MultiHeadAttentionRequest(BaseModel):
    """Requisição para multi-head attention."""

    tokens: list[str] = Field(
        ...,
        min_length=1,
        description="Lista de tokens para processar",
    )
    d_model: int = Field(
        default=64,
        ge=8,
        le=256,
        description="Dimensão do modelo (deve ser divisível por num_cabecas)",
    )
    num_cabecas: int = Field(
        default=8,
        ge=1,
        le=16,
        description="Número de cabeças de atenção",
    )


class MultiHeadAttentionResponse(BaseModel):
    """Resultado do multi-head attention com padrões por cabeça."""

    cabecas: list[dict[str, object]]
    saida_final: list[list[float]]
    tokens: list[str]
    num_tokens_reais: int
    d_model: int
    num_cabecas: int
    d_k: int
    explicacao: str


class AttentionFlowRequest(BaseModel):
    """Requisição para fluxo de atenção de um token específico."""

    tokens: list[str] = Field(
        ...,
        min_length=2,
        description="Lista de tokens da sequência",
    )
    indice_token: int = Field(
        default=0,
        ge=0,
        description="Índice do token alvo (0-based)",
    )
    d_model: int = Field(
        default=64,
        ge=8,
        le=256,
        description="Dimensão do modelo",
    )


class AttentionFlowResponse(BaseModel):
    """Fluxo de atenção para um token específico."""

    token_alvo: str
    indice_token: int
    pesos: list[float]
    tokens: list[str]
    conexoes_significativas: list[dict[str, object]]
    explicacao: str


class TokenImportanceRequest(BaseModel):
    """Requisição para métricas de importância de tokens."""

    tokens: list[str] = Field(
        ...,
        min_length=1,
        description="Lista de tokens para análise",
    )
    d_model: int = Field(
        default=64,
        ge=8,
        le=256,
        description="Dimensão do modelo",
    )


class TokenImportanceResponse(BaseModel):
    """Métricas de importância por token."""

    tokens: list[str]
    importancia_recebida: list[float]
    importancia_dada: list[float]
    importancia_combinada: list[float]
    token_mais_importante: str
    explicacao: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _criar_simulador_e_embeddings(
    tokens: list[str], d_model: int
) -> tuple[TransformerSimulator, "np.ndarray", list[str], int]:
    """Cria simulador, computa embeddings e retorna dados prontos."""
    seq_length = max(10, len(tokens))
    simulador = TransformerSimulator(d_model=d_model, seq_length=seq_length)
    resultado_emb = simulador.compute_embeddings(tokens)
    embeddings = np.array(resultado_emb.embeddings_finais)
    return simulador, embeddings, resultado_emb.tokens, resultado_emb.num_tokens_reais


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/self-attention", response_model=SelfAttentionResponse)
async def self_attention(body: SelfAttentionRequest) -> SelfAttentionResponse:
    """Computa self-attention passo a passo retornando Q, K, V, scores e pesos.

    Implementa Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    de forma transparente para fins educacionais.
    """
    simulador, embeddings, tokens, num_tokens_reais = _criar_simulador_e_embeddings(
        body.tokens, body.d_model
    )
    resultado = simulador.compute_attention_step_by_step(
        embeddings, tokens, num_tokens_reais
    )

    passos_explicados = [
        {
            "passo": "1. Projeção Linear",
            "descricao": (
                f"Multiplicamos os embeddings por matrizes treináveis W_Q, W_K, W_V "
                f"(shape: {body.d_model}x{body.d_model}), criando as representações "
                "Query (o que procuro?), Key (o que ofereço?) e Value (o que transmito?)."
            ),
        },
        {
            "passo": "2. Produto Escalar Q·K^T",
            "descricao": (
                "Calculamos a compatibilidade entre cada Query e cada Key via produto "
                "escalar. Um score alto indica que o token Q deve prestar atenção ao token K."
            ),
        },
        {
            "passo": f"3. Escalonamento por √{body.d_model} = {resultado.fator_escala:.2f}",
            "descricao": (
                "Dividimos os scores pela raiz da dimensão para evitar que gradientes "
                "fiquem muito pequenos em espaços de alta dimensão (problema de vanishing gradients)."
            ),
        },
        {
            "passo": "4. Softmax",
            "descricao": (
                "Aplicamos softmax para converter scores em probabilidades que somam 1. "
                "Isso transforma a matriz de compatibilidade em pesos de atenção interpretáveis."
            ),
        },
        {
            "passo": "5. Média Ponderada dos Values",
            "descricao": (
                "Multiplicamos os pesos pelos Values. Cada token de saída é uma "
                "combinação ponderada de todos os outros tokens, permitindo que "
                "informação de qualquer posição influencie qualquer outra."
            ),
        },
    ]

    return SelfAttentionResponse(
        Q=resultado.Q,
        K=resultado.K,
        V=resultado.V,
        scores_escalados=resultado.scores_escalados,
        pesos_atencao=resultado.pesos_atencao,
        saida=resultado.saida,
        tokens=resultado.tokens,
        num_tokens_reais=resultado.num_tokens_reais,
        fator_escala=resultado.fator_escala,
        passos_explicados=passos_explicados,
    )


@router.post("/multi-head-attention", response_model=MultiHeadAttentionResponse)
async def multi_head_attention(
    body: MultiHeadAttentionRequest,
) -> MultiHeadAttentionResponse:
    """Computa multi-head attention com padrões por cabeça.

    Cada cabeça opera em subespaços diferentes, permitindo que o modelo
    capture simultâneamente relações sintáticas, semânticas e pragmáticas.
    """
    if body.d_model % body.num_cabecas != 0:
        divisores_validos = [h for h in range(1, 17) if body.d_model % h == 0]
        raise HTTPException(
            status_code=422,
            detail=(
                f"d_model ({body.d_model}) deve ser divisível por num_cabecas ({body.num_cabecas}). "
                f"Valores válidos para num_cabecas: {divisores_validos}"
            ),
        )

    seq_length = max(10, len(body.tokens))
    simulador_emb = TransformerSimulator(
        d_model=body.d_model, seq_length=seq_length
    )
    resultado_emb = simulador_emb.compute_embeddings(body.tokens)
    embeddings = np.array(resultado_emb.embeddings_finais)

    mha = MultiHeadAttention(
        d_model=body.d_model,
        num_heads=body.num_cabecas,
        seq_length=seq_length,
    )
    resultado = mha.compute_multi_head_attention(
        embeddings,
        resultado_emb.tokens,
        resultado_emb.num_tokens_reais,
    )

    n = resultado.num_tokens_reais
    tokens_reais = resultado_emb.tokens[:n]

    cabecas_serialized = [
        {
            "cabeca": cabeca.indice_cabeca + 1,
            "tokens": tokens_reais,
            "pesos_atencao": [row[:n] for row in cabeca.pesos_atencao[:n]],
        }
        for cabeca in resultado.cabecas
    ]

    return MultiHeadAttentionResponse(
        cabecas=cabecas_serialized,
        saida_final=resultado.saida_final,
        tokens=resultado_emb.tokens,
        num_tokens_reais=resultado.num_tokens_reais,
        d_model=resultado.d_model,
        num_cabecas=resultado.num_cabecas,
        d_k=resultado.d_k,
        explicacao=(
            f"Com {body.num_cabecas} cabeças e d_model={body.d_model}, cada cabeça "
            f"opera em subespaços de dimensão d_k={body.d_model // body.num_cabecas}. "
            "Cada cabeça aprende a focar em diferentes tipos de relações: "
            "algumas capturam relações sintáticas (sujeito-verbo), outras semânticas "
            "(co-referência) e outras pragmáticas (discurso). As saídas são "
            "concatenadas e projetadas de volta a d_model via W_O."
        ),
    )


@router.post("/attention-flow", response_model=AttentionFlowResponse)
async def attention_flow(body: AttentionFlowRequest) -> AttentionFlowResponse:
    """Computa o fluxo de atenção para um token específico.

    Mostra quanta atenção um determinado token dá a cada outro token
    na sequência, ilustrando o conceito de contexto dinâmico.
    """
    simulador, embeddings, tokens, num_tokens_reais = _criar_simulador_e_embeddings(
        body.tokens, body.d_model
    )

    if body.indice_token >= num_tokens_reais:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Índice {body.indice_token} fora do intervalo. "
                f"A sequência tem {num_tokens_reais} tokens reais (0 a {num_tokens_reais - 1})."
            ),
        )

    resultado_att = simulador.compute_attention_step_by_step(
        embeddings, tokens, num_tokens_reais
    )
    pesos = np.array(resultado_att.pesos_atencao)

    flow = simulador.compute_attention_flow(
        pesos, tokens, body.indice_token, num_tokens_reais
    )

    return AttentionFlowResponse(
        token_alvo=flow.token_alvo,
        indice_token=flow.indice_token,
        pesos=flow.pesos,
        tokens=flow.tokens,
        conexoes_significativas=flow.conexoes_significativas,
        explicacao=(
            f'O token "{flow.token_alvo}" distribui sua atenção por toda a sequência. '
            "Conexões com peso > 5% são consideradas significativas. "
            "Isso demonstra como o modelo captura dependências de longa distância: "
            "um verbo pode prestar atenção ao seu sujeito mesmo que separados por muitas palavras."
        ),
    )


@router.post("/token-importance", response_model=TokenImportanceResponse)
async def token_importance(body: TokenImportanceRequest) -> TokenImportanceResponse:
    """Computa métricas de importância de tokens baseadas na atenção.

    Calcula três métricas:
    - Importância Recebida: quanto outros tokens prestam atenção a este
    - Importância Dada: quanto este token presta atenção aos outros
    - Importância Combinada: média das duas métricas
    """
    simulador, embeddings, tokens, num_tokens_reais = _criar_simulador_e_embeddings(
        body.tokens, body.d_model
    )

    resultado_att = simulador.compute_attention_step_by_step(
        embeddings, tokens, num_tokens_reais
    )
    pesos = np.array(resultado_att.pesos_atencao)

    importancia = simulador._calcular_importancia(pesos, tokens, num_tokens_reais)

    idx_mais_importante = int(
        np.argmax(np.array(importancia.importancia_combinada))
    )
    token_mais_importante = importancia.tokens[idx_mais_importante]

    return TokenImportanceResponse(
        tokens=importancia.tokens,
        importancia_recebida=importancia.importancia_recebida,
        importancia_dada=importancia.importancia_dada,
        importancia_combinada=importancia.importancia_combinada,
        token_mais_importante=token_mais_importante,
        explicacao=(
            "Importância Recebida (Attention IN): soma de quanto todos os outros "
            "tokens prestam atenção a este token. Alta importância recebida indica "
            "que o token é semanticamente central. "
            "Importância Dada (Attention OUT): quanto este token 'consulta' os outros. "
            "Alta importância dada indica que o token precisa de muito contexto para "
            "ser compreendido (ex: pronomes anafóricos)."
        ),
    )
