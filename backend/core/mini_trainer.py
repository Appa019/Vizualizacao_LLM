"""Mini treinador de rede neural para demonstração de conceitos de treinamento."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from utils.math_utils import softmax

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]


@dataclass
class TrainStepResult:
    """Resultado de um passo de treinamento."""

    passo: int
    loss: float
    gradientes_norma: dict[str, float]
    acuracia: float
    predicao: list[float]
    alvo: int


@dataclass
class LossSurfaceResult:
    """Dados para visualização 3D da superfície de loss."""

    w1_valores: list[float]
    w2_valores: list[float]
    loss_grid: list[list[float]]
    ponto_otimo: dict[str, float]


@dataclass
class GradientDescentStep:
    """Um passo na simulação de gradient descent."""

    iteracao: int
    w1: float
    w2: float
    loss: float
    gradiente_w1: float
    gradiente_w2: float


@dataclass
class GradientDescentResult:
    """Resultado completo da simulação de gradient descent."""

    passos: list[GradientDescentStep]
    taxa_aprendizado: float
    loss_inicial: float
    loss_final: float
    convergiu: bool


class MiniTransformerNet:
    """Rede neural de 2 camadas para demonstrar treinamento.

    Arquitetura simplificada: entrada -> oculta (ReLU) -> saída (softmax).
    Treinada em previsão de próxima palavra em um vocabulário pequeno.

    Args:
        tamanho_entrada: Dimensão do vetor de entrada (one-hot).
        tamanho_oculta: Dimensão da camada oculta.
        tamanho_saida: Dimensão da camada de saída (vocabulário).
        taxa_aprendizado: Taxa de aprendizado para SGD.
    """

    def __init__(
        self,
        tamanho_entrada: int = 20,
        tamanho_oculta: int = 32,
        tamanho_saida: int = 20,
        taxa_aprendizado: float = 0.01,
    ) -> None:
        self.tamanho_entrada = tamanho_entrada
        self.tamanho_oculta = tamanho_oculta
        self.tamanho_saida = tamanho_saida
        self.taxa_aprendizado = taxa_aprendizado
        self.passo_atual = 0

        rng = np.random.default_rng(42)
        escala1 = np.sqrt(2.0 / tamanho_entrada)
        escala2 = np.sqrt(2.0 / tamanho_oculta)

        self.W1: FloatArray = rng.standard_normal(
            (tamanho_entrada, tamanho_oculta)
        ) * escala1
        self.b1: FloatArray = np.zeros(tamanho_oculta)
        self.W2: FloatArray = rng.standard_normal(
            (tamanho_oculta, tamanho_saida)
        ) * escala2
        self.b2: FloatArray = np.zeros(tamanho_saida)

        # Histórico de loss para visualização
        self.historico_loss: list[float] = []

    def _forward(
        self, x: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Passagem forward.

        Args:
            x: Vetor de entrada (tamanho_entrada,).

        Returns:
            Tupla (logits, probabilidades, ativacao_oculta).
        """
        h: FloatArray = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        logits: FloatArray = h @ self.W2 + self.b2
        probs = softmax(logits)
        return logits, probs, h

    def _cross_entropy_loss(self, probs: FloatArray, alvo: int) -> float:
        """Calcula cross-entropy loss.

        Args:
            probs: Distribuição de probabilidade prevista.
            alvo: Índice da classe correta.

        Returns:
            Valor do loss.
        """
        return float(-np.log(np.clip(probs[alvo], 1e-10, 1.0)))

    def train_step(self, x: FloatArray, alvo: int) -> TrainStepResult:
        """Executa um passo de treinamento com backpropagation.

        Args:
            x: Vetor de entrada.
            alvo: Índice do token alvo.

        Returns:
            TrainStepResult com loss, gradientes e acurácia.
        """
        self.passo_atual += 1

        # Forward
        logits, probs, h = self._forward(x)
        loss = self._cross_entropy_loss(probs, alvo)
        self.historico_loss.append(loss)

        # Backward (gradiente da cross-entropy + softmax)
        delta_output: FloatArray = probs.copy()
        delta_output[alvo] -= 1.0  # dL/d_logits

        dW2: FloatArray = h[:, np.newaxis] * delta_output[np.newaxis, :]
        db2: FloatArray = delta_output

        delta_hidden: FloatArray = delta_output @ self.W2.T
        delta_hidden[h <= 0] = 0.0  # gradiente ReLU

        dW1: FloatArray = x[:, np.newaxis] * delta_hidden[np.newaxis, :]
        db1: FloatArray = delta_hidden

        # Atualização SGD
        self.W2 -= self.taxa_aprendizado * dW2
        self.b2 -= self.taxa_aprendizado * db2
        self.W1 -= self.taxa_aprendizado * dW1
        self.b1 -= self.taxa_aprendizado * db1

        acuracia = float(int(np.argmax(probs) == alvo))

        return TrainStepResult(
            passo=self.passo_atual,
            loss=loss,
            gradientes_norma={
                "W1": float(np.linalg.norm(dW1)),
                "W2": float(np.linalg.norm(dW2)),
                "b1": float(np.linalg.norm(db1)),
                "b2": float(np.linalg.norm(db2)),
            },
            acuracia=acuracia,
            predicao=probs.tolist(),
            alvo=alvo,
        )

    def predict(self, x: FloatArray) -> FloatArray:
        """Executa inferência.

        Args:
            x: Vetor de entrada.

        Returns:
            Distribuição de probabilidade sobre o vocabulário.
        """
        _, probs, _ = self._forward(x)
        return probs


def gerar_dado_treino(
    tamanho_vocab: int = 20,
) -> tuple[FloatArray, int]:
    """Gera um par (entrada, alvo) sintético para demonstração.

    Simula a tarefa de prever o próximo token: dado um token atual (one-hot),
    o alvo é o token seguinte em uma sequência circular.

    Args:
        tamanho_vocab: Tamanho do vocabulário.

    Returns:
        Tupla (vetor_entrada, indice_alvo).
    """
    idx_entrada = np.random.randint(0, tamanho_vocab)
    idx_alvo = (idx_entrada + 1) % tamanho_vocab

    entrada = np.zeros(tamanho_vocab)
    entrada[idx_entrada] = 1.0

    return entrada, idx_alvo


def computar_superficie_loss(
    resolucao: int = 30,
) -> LossSurfaceResult:
    """Computa uma superfície de loss 2D para visualização 3D.

    Usa uma função de loss analítica simples (bowl convexa com ruído)
    para ilustrar o conceito de superfície de otimização.

    Args:
        resolucao: Número de pontos em cada dimensão da grade.

    Returns:
        LossSurfaceResult com coordenadas e valores de loss.
    """
    w1_range = np.linspace(-3.0, 3.0, resolucao)
    w2_range = np.linspace(-3.0, 3.0, resolucao)

    rng = np.random.default_rng(0)
    noise = rng.standard_normal((resolucao, resolucao)) * 0.2

    # Superfície: parábola 2D + ruído para aparência realista
    W1, W2 = np.meshgrid(w1_range, w2_range)
    loss_grid: FloatArray = (
        0.5 * W1**2
        + 2.0 * W2**2
        + 0.3 * W1 * W2
        + noise
    )
    # Garante que todos os valores de loss sejam positivos
    loss_grid = loss_grid - loss_grid.min() + 0.1

    idx_min = np.unravel_index(np.argmin(loss_grid), loss_grid.shape)

    return LossSurfaceResult(
        w1_valores=w1_range.tolist(),
        w2_valores=w2_range.tolist(),
        loss_grid=loss_grid.tolist(),
        ponto_otimo={
            "w1": float(w1_range[idx_min[1]]),
            "w2": float(w2_range[idx_min[0]]),
            "loss": float(loss_grid[idx_min]),
        },
    )


def simular_gradient_descent(
    taxa_aprendizado: float = 0.1,
    num_iteracoes: int = 50,
    w1_inicial: float = 2.5,
    w2_inicial: float = 2.5,
) -> GradientDescentResult:
    """Simula gradient descent em uma superfície de loss analítica.

    Usa a mesma função de loss da superfície para manter consistência visual.

    Args:
        taxa_aprendizado: Tamanho do passo de gradient descent.
        num_iteracoes: Número de iterações.
        w1_inicial: Valor inicial de w1.
        w2_inicial: Valor inicial de w2.

    Returns:
        GradientDescentResult com trajetória completa de otimização.
    """

    def loss_fn(w1: float, w2: float) -> float:
        return 0.5 * w1**2 + 2.0 * w2**2 + 0.3 * w1 * w2 + 0.1

    def grad_fn(w1: float, w2: float) -> tuple[float, float]:
        dw1 = w1 + 0.15 * w2
        dw2 = 4.0 * w2 + 0.3 * w1
        return dw1, dw2

    w1, w2 = w1_inicial, w2_inicial
    passos: list[GradientDescentStep] = []
    limiar_convergencia = 1e-4

    for i in range(num_iteracoes):
        loss = loss_fn(w1, w2)
        dw1, dw2 = grad_fn(w1, w2)

        passos.append(
            GradientDescentStep(
                iteracao=i,
                w1=w1,
                w2=w2,
                loss=loss,
                gradiente_w1=dw1,
                gradiente_w2=dw2,
            )
        )

        w1 -= taxa_aprendizado * dw1
        w2 -= taxa_aprendizado * dw2

        if abs(dw1) < limiar_convergencia and abs(dw2) < limiar_convergencia:
            break

    convergiu = (
        abs(passos[-1].gradiente_w1) < limiar_convergencia
        and abs(passos[-1].gradiente_w2) < limiar_convergencia
    )

    return GradientDescentResult(
        passos=passos,
        taxa_aprendizado=taxa_aprendizado,
        loss_inicial=passos[0].loss,
        loss_final=passos[-1].loss,
        convergiu=convergiu,
    )


def gerar_dados_objetivos_treino() -> dict[str, object]:
    """Gera dados de exemplo para Masked LM e Causal LM.

    Returns:
        Dicionário com exemplos de MLM e CLM em português.
    """
    frase_exemplo = "O modelo de linguagem aprende padrões do texto"
    tokens = frase_exemplo.split()

    # MLM: mascara tokens aleatórios (15% como no BERT)
    rng = np.random.default_rng(7)
    indices_mascarados = rng.choice(
        len(tokens),
        size=max(1, int(len(tokens) * 0.15)),
        replace=False,
    ).tolist()

    tokens_mlm = list(tokens)
    for idx in indices_mascarados:
        tokens_mlm[idx] = "[MASK]"

    # CLM: cada token prevê o próximo
    pares_clm = [
        {"entrada": tokens[:i], "alvo": tokens[i]}
        for i in range(1, len(tokens))
    ]

    return {
        "frase_original": frase_exemplo,
        "tokens": tokens,
        "mlm": {
            "descricao": "Masked Language Modeling (BERT): mascara tokens e os prevê a partir do contexto bidirecional",
            "tokens_com_mascara": tokens_mlm,
            "indices_mascarados": indices_mascarados,
            "tokens_originais": [tokens[i] for i in indices_mascarados],
        },
        "clm": {
            "descricao": "Causal Language Modeling (GPT): prevê o próximo token dado o contexto anterior",
            "pares_entrada_alvo": pares_clm,
        },
    }
