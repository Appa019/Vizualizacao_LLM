"""Router de embeddings: embeddings, positional encoding e visualização 3D."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.transformer_simulator import TransformerSimulator
from utils.math_utils import reduce_dimensions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


# ---------------------------------------------------------------------------
# Modelos de requisição e resposta
# ---------------------------------------------------------------------------


class EmbeddingsRequest(BaseModel):
    """Requisição para computar embeddings."""

    tokens: list[str] = Field(
        ...,
        min_length=1,
        description="Lista de tokens para computar embeddings",
        examples=[["o", "gato", "caminha", "devagar"]],
    )
    d_model: int = Field(
        default=64,
        ge=8,
        le=256,
        description="Dimensão do modelo (tamanho dos vetores de embedding)",
    )


class EmbeddingsResponse(BaseModel):
    """Resposta com embeddings e positional encoding."""

    token_embeddings: list[list[float]]
    positional_encoding: list[list[float]]
    embeddings_finais: list[list[float]]
    tokens: list[str]
    num_tokens_reais: int
    d_model: int
    padroes_sinusoidais: list[list[float]]
    explicacao: str


class PositionalEncodingRequest(BaseModel):
    """Requisição para positional encoding."""

    seq_length: int = Field(
        default=10,
        ge=2,
        le=50,
        description="Comprimento da sequência",
    )
    d_model: int = Field(
        default=64,
        ge=8,
        le=256,
        description="Dimensão do modelo",
    )


class PositionalEncodingResponse(BaseModel):
    """Resposta com positional encoding."""

    encoding: list[list[float]]
    seq_length: int
    d_model: int
    padroes_por_dimensao: list[list[float]]
    explicacao: str


class EmbeddingSpaceRequest(BaseModel):
    """Requisição para visualização 3D do espaço de embeddings."""

    tokens: list[str] = Field(
        ...,
        min_length=2,
        description="Lista de tokens para visualizar no espaço 3D",
    )
    metodo: Literal["pca", "tsne"] = Field(
        default="pca",
        description="Método de redução de dimensionalidade",
    )
    d_model: int = Field(
        default=64,
        ge=8,
        le=256,
        description="Dimensão do modelo",
    )


class EmbeddingSpacePoint(BaseModel):
    """Ponto no espaço de embeddings 3D."""

    word: str
    position: list[float]
    category: str


class EmbeddingSpaceResponse(BaseModel):
    """Resposta com coordenadas 3D para visualização."""

    pontos: list[EmbeddingSpacePoint]
    metodo: str
    variancia_explicada: list[float] | None = None
    explicacao: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/embeddings", response_model=EmbeddingsResponse)
async def computar_embeddings(body: EmbeddingsRequest) -> EmbeddingsResponse:
    """Computa token embeddings e positional encoding para os tokens fornecidos.

    Demonstra como cada token é convertido em um vetor denso e como
    a informação posicional é adicionada via funções seno/cosseno.
    """
    simulador = TransformerSimulator(
        d_model=body.d_model, seq_length=max(10, len(body.tokens))
    )
    resultado = simulador.compute_embeddings(body.tokens)

    return EmbeddingsResponse(
        token_embeddings=resultado.token_embeddings,
        positional_encoding=resultado.positional_encoding,
        embeddings_finais=resultado.embeddings_finais,
        tokens=resultado.tokens,
        num_tokens_reais=resultado.num_tokens_reais,
        d_model=body.d_model,
        padroes_sinusoidais=resultado.padroes_sinusoidais,
        explicacao=(
            "Token Embeddings: vetores aprendidos que representam o significado "
            "de cada token. Positional Encoding: funções seno/cosseno que informam "
            "ao modelo a posição de cada token na sequência. "
            "O embedding final é a soma dos dois."
        ),
    )


@router.post("/positional-encoding", response_model=PositionalEncodingResponse)
async def positional_encoding(
    body: PositionalEncodingRequest,
) -> PositionalEncodingResponse:
    """Computa o positional encoding sinusoidal.

    Demonstra os padrões de seno/cosseno que permitem ao Transformer
    distinguir posições em uma sequência.
    """
    simulador = TransformerSimulator(
        d_model=body.d_model, seq_length=body.seq_length
    )
    pe = simulador.create_positional_encoding()

    n_dims_exibir = min(10, body.d_model)
    padroes = pe[:, :n_dims_exibir].tolist()

    return PositionalEncodingResponse(
        encoding=pe.tolist(),
        seq_length=body.seq_length,
        d_model=body.d_model,
        padroes_por_dimensao=padroes,
        explicacao=(
            "O Positional Encoding usa funções seno (dimensões pares) e cosseno "
            "(dimensões ímpares) com diferentes frequências. Isso garante que "
            "cada posição tenha um padrão único e que posições próximas tenham "
            "representações similares. A fórmula é: "
            "PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), "
            "PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))."
        ),
    )


@router.post("/embedding-space", response_model=EmbeddingSpaceResponse)
async def embedding_space(
    body: EmbeddingSpaceRequest,
) -> EmbeddingSpaceResponse:
    """Retorna coordenadas 3D para visualização do espaço de embeddings.

    Usa PCA ou t-SNE para reduzir a alta dimensionalidade dos embeddings
    a 3 dimensões para visualização interativa.
    """
    simulador = TransformerSimulator(
        d_model=body.d_model, seq_length=max(10, len(body.tokens))
    )
    resultado = simulador.compute_embeddings(body.tokens)

    embeddings_array = np.array(resultado.embeddings_finais)
    tokens_reais = resultado.tokens[: resultado.num_tokens_reais]
    embeddings_reais = embeddings_array[: resultado.num_tokens_reais]

    if len(tokens_reais) < 2:
        raise HTTPException(
            status_code=422,
            detail="São necessários pelo menos 2 tokens para visualização 3D.",
        )

    coordenadas = reduce_dimensions(
        embeddings_reais, method=body.metodo, n_components=3
    )

    # Categorias semânticas para visualização
    _categorias: dict[str, str] = {
        "gato": "animal", "cachorro": "animal", "peixe": "animal",
        "cavalo": "animal", "passaro": "animal", "rato": "animal",
        "vaca": "animal", "leao": "animal", "tigre": "animal",
        "vermelho": "cor", "azul": "cor", "verde": "cor",
        "amarelo": "cor", "roxo": "cor", "laranja": "cor",
        "preto": "cor", "branco": "cor", "rosa": "cor",
        "correr": "verbo", "andar": "verbo", "nadar": "verbo",
        "pular": "verbo", "voar": "verbo", "comer": "verbo",
        "dormir": "verbo", "falar": "verbo", "pensar": "verbo",
        "casa": "substantivo", "predio": "substantivo", "escola": "substantivo",
        "cidade": "substantivo", "rua": "substantivo", "mesa": "substantivo",
        "carro": "substantivo", "livro": "substantivo", "porta": "substantivo",
    }

    pontos = [
        EmbeddingSpacePoint(
            word=token,
            position=coordenadas[i],
            category=_categorias.get(token.lower(), "default"),
        )
        for i, token in enumerate(tokens_reais)
    ]

    descricao_metodo = (
        "PCA (Análise de Componentes Principais): redução linear que preserva "
        "a maior variância dos dados. Rápido e determinístico."
        if body.metodo == "pca"
        else "t-SNE: redução não-linear que preserva estrutura local. "
        "Ideal para visualizar clusters, mas não-determinístico."
    )

    return EmbeddingSpaceResponse(
        pontos=pontos,
        metodo=body.metodo,
        explicacao=(
            f"Embeddings de {body.d_model} dimensões reduzidos a 3D usando {body.metodo.upper()}. "
            f"{descricao_metodo} "
            "Tokens semanticamente similares tendem a aparecer próximos no espaço."
        ),
    )
