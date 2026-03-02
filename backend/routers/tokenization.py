"""Router de tokenização: demonstra diferentes abordagens de tokenização."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from utils.tokenizer import (
    BPEMergeStep,
    VocabularioStats,
    calcular_estatisticas_vocabulario,
    comparar_abordagens_tokenizacao,
    executar_bpe_passos,
    tokenizar_simples,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/tokenization", tags=["tokenização"])


# ---------------------------------------------------------------------------
# Modelos de requisição e resposta
# ---------------------------------------------------------------------------


class TokenizarRequest(BaseModel):
    """Requisição de tokenização."""

    texto: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Texto a ser tokenizado",
        examples=["O gato caminha pela floresta"],
    )
    max_tokens: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Número máximo de tokens a retornar",
    )


class TokenizarResponse(BaseModel):
    """Resposta de tokenização."""

    tokens: list[str]
    num_tokens: int
    texto_original: str
    estatisticas: dict[str, object]


class BPEStepsRequest(BaseModel):
    """Requisição para passos de BPE."""

    textos: list[str] = Field(
        ...,
        min_length=1,
        description="Lista de textos para construir vocabulário BPE",
        examples=[["O gato caminha", "A gata corre pela rua"]],
    )
    num_mesclagens: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Número de operações de mesclagem BPE a realizar",
    )


class BPEStepResponse(BaseModel):
    """Um passo da mesclagem BPE."""

    passo: int
    par_mesclado: list[str]
    novo_token: str
    frequencia: int
    tamanho_vocabulario: int
    amostra_corpus: list[list[str]]


class BPEStepsResponse(BaseModel):
    """Resposta com todos os passos de BPE."""

    passos: list[BPEStepResponse]
    num_passos: int
    explicacao: str


class CompararTokenizadoresResponse(BaseModel):
    """Comparação entre abordagens de tokenização."""

    texto: str
    abordagens: dict[str, list[str]]
    comparacao: list[dict[str, object]]
    explicacao: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/tokenize", response_model=TokenizarResponse)
async def tokenizar(body: TokenizarRequest) -> TokenizarResponse:
    """Tokeniza texto usando abordagem regex simples.

    Divide o texto em palavras e pontuação, retornando tokens em minúsculo.
    """
    tokens = tokenizar_simples(body.texto, max_tokens=body.max_tokens)
    stats = calcular_estatisticas_vocabulario(tokens)

    return TokenizarResponse(
        tokens=tokens,
        num_tokens=len(tokens),
        texto_original=body.texto,
        estatisticas={
            "tokens_unicos": stats.tokens_unicos,
            "tamanho_medio_token": stats.tamanho_medio_token,
            "tokens_mais_comuns": [
                {"token": t, "frequencia": f}
                for t, f in stats.tokens_mais_comuns
            ],
        },
    )


@router.post("/bpe-steps", response_model=BPEStepsResponse)
async def bpe_passos(body: BPEStepsRequest) -> BPEStepsResponse:
    """Retorna passos do algoritmo BPE para animação no frontend.

    O BPE (Byte Pair Encoding) constrói um vocabulário de subpalavras
    iterativamente mesclando os pares de símbolos mais frequentes.
    """
    passos_bpe: list[BPEMergeStep] = executar_bpe_passos(
        body.textos, num_mesclagens=body.num_mesclagens
    )

    passos_resposta = [
        BPEStepResponse(
            passo=p.passo,
            par_mesclado=list(p.par_mesclado),
            novo_token=p.novo_token,
            frequencia=p.frequencia,
            tamanho_vocabulario=len(p.vocabulario_atual),
            amostra_corpus=p.corpus_atual,
        )
        for p in passos_bpe
    ]

    return BPEStepsResponse(
        passos=passos_resposta,
        num_passos=len(passos_resposta),
        explicacao=(
            "O BPE começa com caracteres individuais e iterativamente mescla "
            "os pares mais frequentes. Isso permite representar palavras raras "
            "como combinações de subpalavras conhecidas, equilibrando "
            "tamanho de vocabulário e cobertura."
        ),
    )


@router.get("/compare-tokenizers", response_model=CompararTokenizadoresResponse)
async def comparar_tokenizadores(
    texto: str = "O transformer revolucionou o processamento de linguagem natural",
) -> CompararTokenizadoresResponse:
    """Compara diferentes abordagens de tokenização no mesmo texto.

    Demonstra as diferenças entre tokenização por palavra, subpalavra e caractere.
    """
    abordagens = comparar_abordagens_tokenizacao(texto)

    comparacao = [
        {
            "abordagem": "palavra",
            "tokens": abordagens["palavra"],
            "num_tokens": len(abordagens["palavra"]),
            "descricao": "Divide por espaços e pontuação. Simples, mas não lida bem com palavras raras.",
            "vantagem": "Intuitivo e eficiente para vocabulários conhecidos",
            "desvantagem": "Vocabulário muito grande; palavras fora do vocab são [UNK]",
        },
        {
            "abordagem": "subpalavra",
            "tokens": abordagens["subpalavra"],
            "num_tokens": len(abordagens["subpalavra"]),
            "descricao": "Divide palavras longas em prefixo + sufixo (simulando BPE/WordPiece).",
            "vantagem": "Lida com palavras raras e morfologia complexa",
            "desvantagem": "Tokens mais abstratos, difíceis de interpretar",
        },
        {
            "abordagem": "caractere",
            "tokens": abordagens["caractere"],
            "num_tokens": len(abordagens["caractere"]),
            "descricao": "Cada caractere é um token. Vocabulário mínimo (< 100 tokens).",
            "vantagem": "Sem palavras desconhecidas; vocabulário minúsculo",
            "desvantagem": "Sequências muito longas; difícil capturar semântica",
        },
    ]

    return CompararTokenizadoresResponse(
        texto=texto,
        abordagens=abordagens,
        comparacao=comparacao,
        explicacao=(
            "A escolha do método de tokenização impacta diretamente o tamanho "
            "do vocabulário, a capacidade de generalização e o comprimento das "
            "sequências processadas. Modelos modernos como BERT e GPT usam "
            "tokenização por subpalavras (WordPiece/BPE) como equilíbrio ideal."
        ),
    )
