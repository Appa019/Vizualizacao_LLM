"""Router de inferência: geração de texto com diferentes estratégias."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.text_generator import TextGenerator, VOCABULARIO_PT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/inference", tags=["inferência"])

_gerador = TextGenerator(vocabulario=VOCABULARIO_PT)


# ---------------------------------------------------------------------------
# Modelos de requisição e resposta
# ---------------------------------------------------------------------------


class GerarRequest(BaseModel):
    """Requisição para geração de texto."""

    prompt: list[str] = Field(
        default=["o"],
        min_length=0,
        description="Tokens de contexto inicial",
        examples=[["o", "gato"]],
    )
    estrategia: str = Field(
        default="temperatura",
        description="Estratégia: 'greedy', 'temperatura', 'top_k', 'top_p'",
    )
    max_tokens: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Número máximo de tokens a gerar",
    )
    temperatura: float = Field(
        default=1.0,
        gt=0,
        le=5.0,
        description="Temperatura para amostragem (apenas para estratégias estocásticas)",
    )
    k: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Número de candidatos para top-k",
    )
    p: float = Field(
        default=0.9,
        gt=0,
        le=1.0,
        description="Limiar de probabilidade acumulada para top-p (nucleus)",
    )


class TokenGeradoResponse(BaseModel):
    """Informações sobre um token gerado."""

    token: str
    probabilidade: float
    logit: float
    top_5_tokens: list[dict[str, object]]


class GerarResponse(BaseModel):
    """Resposta de geração de texto."""

    texto_gerado: str
    tokens_gerados: list[str]
    historico_probabilidades: list[TokenGeradoResponse]
    estrategia: str
    parametros: dict[str, object]
    explicacao: str


class TemperaturaDemoRequest(BaseModel):
    """Requisição para demonstração de temperatura."""

    temperaturas: list[float] = Field(
        default=[0.1, 0.5, 1.0, 1.5, 2.0],
        min_length=1,
        description="Lista de temperaturas a comparar",
    )


class TemperaturaDemoResponse(BaseModel):
    """Demonstração do efeito da temperatura."""

    logits_originais: list[float]
    tokens: list[str]
    distribuicoes: list[dict[str, object]]
    explicacao: str


class SamplingDemoRequest(BaseModel):
    """Requisição para demonstração de estratégias de amostragem."""

    prompt: list[str] = Field(
        default=["o"],
        description="Tokens de contexto",
    )
    max_tokens: int = Field(
        default=8,
        ge=1,
        le=30,
        description="Número de tokens a gerar por estratégia",
    )
    num_beams: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Número de beams para beam search",
    )


class SamplingDemoResponse(BaseModel):
    """Comparação entre estratégias de amostragem."""

    greedy: dict[str, object]
    temperatura_baixa: dict[str, object]
    temperatura_alta: dict[str, object]
    top_k: dict[str, object]
    top_p: dict[str, object]
    beam_search: list[dict[str, object]]
    explicacao: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resultado_para_dict(resultado: object) -> dict[str, object]:
    """Serializa GeracaoResult para dicionário JSON-safe."""
    from core.text_generator import GeracaoResult

    r: GeracaoResult = resultado  # type: ignore[assignment]
    return {
        "texto_gerado": r.texto_gerado,
        "tokens_gerados": r.tokens_gerados,
        "estrategia": r.estrategia,
        "parametros": r.parametros,
        "num_tokens": len(r.tokens_gerados),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/generate", response_model=GerarResponse)
async def gerar(body: GerarRequest) -> GerarResponse:
    """Gera texto token a token com a estratégia especificada.

    Retorna as probabilidades de cada token para visualização interativa.
    """
    estrategias_validas = {"greedy", "temperatura", "top_k", "top_p"}
    if body.estrategia not in estrategias_validas:
        raise HTTPException(
            status_code=422,
            detail=f"Estratégia inválida. Opções: {sorted(estrategias_validas)}",
        )

    if body.estrategia == "greedy":
        resultado = _gerador.gerar_greedy(body.prompt, body.max_tokens)
        explicacao = (
            "Geração greedy: em cada passo, escolhe o token com maior probabilidade. "
            "Determinístico, mas pode produzir textos repetitivos ou sem criatividade."
        )
    elif body.estrategia == "temperatura":
        resultado = _gerador.gerar_com_temperatura(
            body.prompt, body.temperatura, body.max_tokens
        )
        explicacao = (
            f"Temperatura = {body.temperatura}: "
            + (
                "distribuição concentrada nos tokens mais prováveis."
                if body.temperatura < 0.7
                else (
                    "distribuição mais uniforme, maior criatividade."
                    if body.temperatura > 1.3
                    else "distribuição original do modelo."
                )
            )
        )
    elif body.estrategia == "top_k":
        resultado = _gerador.gerar_top_k(
            body.prompt, body.k, body.temperatura, body.max_tokens
        )
        explicacao = (
            f"Top-k (k={body.k}): restringe a amostragem aos {body.k} tokens mais prováveis. "
            "Equilibra qualidade e diversidade ao eliminar tokens improváveis."
        )
    else:  # top_p
        resultado = _gerador.gerar_top_p(
            body.prompt, body.p, body.temperatura, body.max_tokens
        )
        explicacao = (
            f"Nucleus sampling (p={body.p}): mantém o menor conjunto de tokens "
            f"cuja probabilidade acumulada é ≥ {body.p}. "
            "Mais adaptativo que top-k: usa mais candidatos quando a distribuição "
            "é uniforme e menos quando é concentrada."
        )

    historico = [
        TokenGeradoResponse(
            token=t.token,
            probabilidade=round(t.probabilidade, 4),
            logit=round(t.logit, 4),
            top_5_tokens=[
                {"token": item["token"], "probabilidade": round(float(item["probabilidade"]), 4)}
                for item in t.top_5_tokens
            ],
        )
        for t in resultado.historico_probabilidades
    ]

    return GerarResponse(
        texto_gerado=resultado.texto_gerado,
        tokens_gerados=resultado.tokens_gerados,
        historico_probabilidades=historico,
        estrategia=resultado.estrategia,
        parametros=resultado.parametros,  # type: ignore[arg-type]
        explicacao=explicacao,
    )


@router.post("/temperature-demo", response_model=TemperaturaDemoResponse)
async def temperature_demo(
    body: TemperaturaDemoRequest,
) -> TemperaturaDemoResponse:
    """Demonstra o efeito da temperatura na distribuição de probabilidade.

    Visualiza como temperaturas diferentes transformam a mesma distribuição:
    baixas temperaturas tornam-na mais determinística, altas mais aleatória.
    """
    for temp in body.temperaturas:
        if temp <= 0:
            raise HTTPException(
                status_code=422,
                detail="Todas as temperaturas devem ser maiores que zero.",
            )

    demo = _gerador.demonstrar_temperatura(body.temperaturas)

    return TemperaturaDemoResponse(
        logits_originais=demo.logits_originais,
        tokens=demo.tokens,
        distribuicoes=[
            {
                "temperatura": d["temperatura"],
                "probabilidades": [round(p, 4) for p in d["probabilidades"]],  # type: ignore[index]
                "entropia": round(float(d["entropia"]), 4),  # type: ignore[arg-type]
                "descricao": d["descricao"],
            }
            for d in demo.distribuicoes
        ],
        explicacao=(
            "A temperatura T modifica os logits antes do softmax: logits / T. "
            "T < 1: sharpening — amplifica as diferenças, torna a distribuição mais concentrada. "
            "T = 1: distribuição original do modelo. "
            "T > 1: smoothing — suaviza as diferenças, torna a distribuição mais uniforme. "
            "A entropia quantifica a incerteza: alta entropia = mais aleatório."
        ),
    )


@router.post("/sampling-demo", response_model=SamplingDemoResponse)
async def sampling_demo(body: SamplingDemoRequest) -> SamplingDemoResponse:
    """Demonstra e compara as principais estratégias de amostragem.

    Gera texto com greedy, temperatura baixa, temperatura alta, top-k,
    top-p e beam search para mostrar as diferenças na prática.
    """
    greedy = _resultado_para_dict(
        _gerador.gerar_greedy(body.prompt, body.max_tokens)
    )
    temp_baixa = _resultado_para_dict(
        _gerador.gerar_com_temperatura(body.prompt, temperatura=0.3, max_tokens=body.max_tokens)
    )
    temp_alta = _resultado_para_dict(
        _gerador.gerar_com_temperatura(body.prompt, temperatura=1.5, max_tokens=body.max_tokens)
    )
    topk = _resultado_para_dict(
        _gerador.gerar_top_k(body.prompt, k=10, temperatura=1.0, max_tokens=body.max_tokens)
    )
    topp = _resultado_para_dict(
        _gerador.gerar_top_p(body.prompt, p=0.9, temperatura=1.0, max_tokens=body.max_tokens)
    )
    beams = _gerador.beam_search(
        body.prompt, num_beams=body.num_beams, max_tokens=body.max_tokens
    )

    return SamplingDemoResponse(
        greedy=greedy,
        temperatura_baixa=temp_baixa,
        temperatura_alta=temp_alta,
        top_k=topk,
        top_p=topp,
        beam_search=beams,
        explicacao=(
            "Greedy: determinístico, escolhe sempre o token mais provável. Rápido mas limitado. "
            "Temperatura baixa (0.3): quase greedy, preferência forte pelos tokens mais prováveis. "
            "Temperatura alta (1.5): mais criativo e diverso, mas pode ser incoerente. "
            "Top-k (k=10): elimina tokens improváveis, mantém diversidade razoável. "
            "Top-p (p=0.9): nucleus adaptativo, número variável de candidatos por contexto. "
            f"Beam search ({body.num_beams} beams): explora múltiplos caminhos em paralelo "
            "e retorna os mais prováveis globalmente. Mais coerente mas menos diverso."
        ),
    )
