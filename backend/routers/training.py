"""Router de treinamento: demonstra conceitos de otimização e treinamento."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel, Field

from core.mini_trainer import (
    MiniTransformerNet,
    computar_superficie_loss,
    gerar_dado_treino,
    gerar_dados_objetivos_treino,
    simular_gradient_descent,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["treinamento"])

# Estado global simples — adequado para demonstração (não produção)
_rede_global: MiniTransformerNet | None = None


def _obter_ou_criar_rede(
    tamanho_vocab: int = 20,
    taxa_aprendizado: float = 0.01,
) -> MiniTransformerNet:
    """Retorna a rede neural global, criando-a se necessário."""
    global _rede_global
    if _rede_global is None:
        _rede_global = MiniTransformerNet(
            tamanho_entrada=tamanho_vocab,
            tamanho_oculta=32,
            tamanho_saida=tamanho_vocab,
            taxa_aprendizado=taxa_aprendizado,
        )
    return _rede_global


# ---------------------------------------------------------------------------
# Modelos de requisição e resposta
# ---------------------------------------------------------------------------


class TrainStepRequest(BaseModel):
    """Requisição para um passo de treinamento."""

    taxa_aprendizado: float = Field(
        default=0.01,
        gt=0,
        le=1.0,
        description="Taxa de aprendizado para SGD",
    )
    tamanho_vocab: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Tamanho do vocabulário simulado",
    )
    resetar: bool = Field(
        default=False,
        description="Se True, reinicia a rede do zero",
    )


class TrainStepResponse(BaseModel):
    """Resultado de um passo de treinamento."""

    passo: int
    loss: float
    acuracia: float
    gradientes_norma: dict[str, float]
    historico_loss: list[float]
    predicao_top5: list[dict[str, object]]
    alvo: int
    explicacao: str


class LossSurfaceRequest(BaseModel):
    """Requisição para superfície de loss 3D."""

    resolucao: int = Field(
        default=30,
        ge=10,
        le=60,
        description="Resolução da grade (resolucao x resolucao pontos)",
    )


class LossSurfaceResponse(BaseModel):
    """Dados para visualização 3D da superfície de loss."""

    w1_valores: list[float]
    w2_valores: list[float]
    loss_grid: list[list[float]]
    ponto_otimo: dict[str, float]
    explicacao: str


class GradientDescentRequest(BaseModel):
    """Requisição para simulação de gradient descent."""

    taxa_aprendizado: float = Field(
        default=0.1,
        gt=0,
        le=2.0,
        description="Taxa de aprendizado",
    )
    num_iteracoes: int = Field(
        default=50,
        ge=5,
        le=200,
        description="Número de iterações",
    )
    w1_inicial: float = Field(
        default=2.5,
        ge=-3.0,
        le=3.0,
        description="Valor inicial de w1",
    )
    w2_inicial: float = Field(
        default=2.5,
        ge=-3.0,
        le=3.0,
        description="Valor inicial de w2",
    )


class GradientDescentResponse(BaseModel):
    """Resultado da simulação de gradient descent."""

    passos: list[dict[str, float]]
    taxa_aprendizado: float
    loss_inicial: float
    loss_final: float
    convergiu: bool
    reducao_percentual: float
    explicacao: str


class TrainingObjectivesResponse(BaseModel):
    """Dados de demonstração de objetivos de treinamento MLM e CLM."""

    frase_original: str
    tokens: list[str]
    mlm: dict[str, object]
    clm: dict[str, object]
    comparacao: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/train-step", response_model=TrainStepResponse)
async def train_step(body: TrainStepRequest) -> TrainStepResponse:
    """Executa um passo de treinamento na mini rede neural.

    Demonstra forward pass, cálculo de loss (cross-entropy) e
    backpropagation com atualização de pesos via SGD.
    """
    global _rede_global

    if body.resetar or _rede_global is None:
        _rede_global = MiniTransformerNet(
            tamanho_entrada=body.tamanho_vocab,
            tamanho_oculta=32,
            tamanho_saida=body.tamanho_vocab,
            taxa_aprendizado=body.taxa_aprendizado,
        )

    rede = _rede_global

    entrada, alvo = gerar_dado_treino(tamanho_vocab=body.tamanho_vocab)
    resultado = rede.train_step(entrada, alvo)

    top5_idx = sorted(
        range(len(resultado.predicao)),
        key=lambda i: resultado.predicao[i],
        reverse=True,
    )[:5]

    predicao_top5 = [
        {
            "indice": idx,
            "token": f"token_{idx}",
            "probabilidade": round(resultado.predicao[idx], 4),
            "correto": idx == resultado.alvo,
        }
        for idx in top5_idx
    ]

    return TrainStepResponse(
        passo=resultado.passo,
        loss=round(resultado.loss, 4),
        acuracia=resultado.acuracia,
        gradientes_norma={k: round(v, 4) for k, v in resultado.gradientes_norma.items()},
        historico_loss=[round(l, 4) for l in rede.historico_loss[-50:]],
        predicao_top5=predicao_top5,
        alvo=resultado.alvo,
        explicacao=(
            f"Passo {resultado.passo}: loss = {resultado.loss:.4f}. "
            f"A rede previu uma distribuição de probabilidade sobre {body.tamanho_vocab} tokens. "
            "O backpropagation calculou gradientes e atualizou os pesos para "
            "minimizar a cross-entropy loss entre previsão e alvo."
        ),
    )


@router.post("/loss-surface", response_model=LossSurfaceResponse)
async def loss_surface(body: LossSurfaceRequest) -> LossSurfaceResponse:
    """Computa superfície de loss 2D para visualização 3D.

    Demonstra a geometria da otimização: vales (mínimos), picos (máximos)
    e como o gradient descent navega essa paisagem.
    """
    resultado = computar_superficie_loss(resolucao=body.resolucao)

    return LossSurfaceResponse(
        w1_valores=resultado.w1_valores,
        w2_valores=resultado.w2_valores,
        loss_grid=resultado.loss_grid,
        ponto_otimo=resultado.ponto_otimo,
        explicacao=(
            "A superfície de loss representa como o valor da função de custo varia "
            "para diferentes valores dos pesos do modelo. O objetivo do treinamento "
            "é encontrar o ponto mais baixo (mínimo global). Na prática, a superfície "
            "tem milhões de dimensões, mas este exemplo 2D ilustra os conceitos de "
            "vale, platô, mínimo local e mínimo global."
        ),
    )


@router.post("/gradient-descent-demo", response_model=GradientDescentResponse)
async def gradient_descent_demo(
    body: GradientDescentRequest,
) -> GradientDescentResponse:
    """Simula gradient descent e retorna a trajetória de otimização.

    Demonstra como diferentes taxas de aprendizado afetam a convergência:
    taxa muito alta pode oscilar, taxa muito baixa converge lentamente.
    """
    resultado = simular_gradient_descent(
        taxa_aprendizado=body.taxa_aprendizado,
        num_iteracoes=body.num_iteracoes,
        w1_inicial=body.w1_inicial,
        w2_inicial=body.w2_inicial,
    )

    passos_serializados = [
        {
            "iteracao": p.iteracao,
            "w1": round(p.w1, 4),
            "w2": round(p.w2, 4),
            "loss": round(p.loss, 4),
            "gradiente_w1": round(p.gradiente_w1, 4),
            "gradiente_w2": round(p.gradiente_w2, 4),
        }
        for p in resultado.passos
    ]

    reducao = (
        (resultado.loss_inicial - resultado.loss_final) / resultado.loss_inicial * 100
        if resultado.loss_inicial > 0
        else 0.0
    )

    status = "convergiu" if resultado.convergiu else "não convergiu completamente"

    return GradientDescentResponse(
        passos=passos_serializados,
        taxa_aprendizado=resultado.taxa_aprendizado,
        loss_inicial=round(resultado.loss_inicial, 4),
        loss_final=round(resultado.loss_final, 4),
        convergiu=resultado.convergiu,
        reducao_percentual=round(reducao, 2),
        explicacao=(
            f"Com taxa de aprendizado = {body.taxa_aprendizado}, o otimizador {status} "
            f"em {len(resultado.passos)} iterações. "
            f"O loss reduziu {reducao:.1f}% ({resultado.loss_inicial:.4f} → {resultado.loss_final:.4f}). "
            "Em cada iteração, o gradiente indica a direção de maior subida, "
            "e damos um passo na direção oposta (descida)."
        ),
    )


@router.get("/training-objectives", response_model=TrainingObjectivesResponse)
async def training_objectives() -> TrainingObjectivesResponse:
    """Retorna dados de demonstração dos objetivos de treinamento MLM e CLM.

    Compara Masked Language Modeling (BERT) e Causal Language Modeling (GPT),
    mostrando como cada paradigma ensina o modelo a entender linguagem.
    """
    dados = gerar_dados_objetivos_treino()

    return TrainingObjectivesResponse(
        frase_original=str(dados["frase_original"]),
        tokens=list(dados["tokens"]),  # type: ignore[arg-type]
        mlm=dict(dados["mlm"]),  # type: ignore[arg-type]
        clm=dict(dados["clm"]),  # type: ignore[arg-type]
        comparacao=(
            "MLM (BERT): mascara tokens aleatórios e os prevê usando contexto bidirecional. "
            "Isso força o modelo a entender relações em ambas as direções da frase. "
            "Ideal para tarefas de compreensão (classificação, NER, QA). "
            "CLM (GPT): prevê o próximo token dado apenas o contexto anterior (causal). "
            "Isso torna o modelo naturalmente gerativo, mas limita o contexto à esquerda. "
            "Ideal para geração de texto, completar frases e chatbots."
        ),
    )
