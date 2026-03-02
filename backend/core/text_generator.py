"""Geração de texto com diferentes estratégias de amostragem."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from heapq import heappush, heappop

import numpy as np
from numpy.typing import NDArray

from utils.math_utils import softmax, entropy

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

# Vocabulário simulado em português para demonstração
VOCABULARIO_PT: list[str] = [
    "o", "a", "de", "que", "e", "do", "da", "em", "um", "para",
    "com", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos",
    "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua",
    "ou", "ser", "quando", "muito", "há", "nos", "já", "está", "também",
    "só", "pelo", "pela", "até", "isso", "ela", "entre", "era", "depois",
    "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse",
    "eles", "estão", "você", "tinha", "foram", "essa", "num", "nem",
    "suas", "meu", "às", "minha", "numa", "pelos", "elas", "havia",
    "seja", "qual", "será", "nós", "tenho", "lhe", "deles", "essas",
    "esses", "pelas", "este", "dele", "tu", "te", "vocês", "vos",
    "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso",
    "nossa", "nossos", "nossas", "dela", "delas", "esta", "estes",
    "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo",
    "gato", "cachorro", "casa", "mundo", "tempo", "dia", "homem",
    "mulher", "vida", "mão", "parte", "lugar", "forma", "novo", "grande",
]


@dataclass
class TokenGeradoInfo:
    """Informações sobre um token gerado."""

    token: str
    probabilidade: float
    logit: float
    top_5_tokens: list[dict[str, float | str]]


@dataclass
class GeracaoResult:
    """Resultado de um processo de geração de texto."""

    texto_gerado: str
    tokens_gerados: list[str]
    historico_probabilidades: list[TokenGeradoInfo]
    estrategia: str
    parametros: dict[str, float | int | str]


@dataclass
class TemperaturaDemo:
    """Demonstração do efeito da temperatura nas probabilidades."""

    logits_originais: list[float]
    tokens: list[str]
    distribuicoes: list[dict[str, object]]  # por temperatura


@dataclass
class BeamNode:
    """Nó na árvore de busca do beam search."""

    tokens: list[str]
    log_prob: float
    probabilidade: float

    def __lt__(self, other: BeamNode) -> bool:
        return self.log_prob > other.log_prob  # max-heap por log_prob


class TextGenerator:
    """Gerador de texto com múltiplas estratégias de amostragem.

    Usa um modelo de linguagem simulado baseado em bigramas para geração.
    Implementa greedy, temperatura, top-k, top-p e beam search.

    Args:
        vocabulario: Lista de tokens do vocabulário.
        seed: Semente para reprodutibilidade.
    """

    def __init__(
        self,
        vocabulario: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self.vocabulario = vocabulario or VOCABULARIO_PT
        self.vocab_size = len(self.vocabulario)
        self.token_para_idx: dict[str, int] = {
            t: i for i, t in enumerate(self.vocabulario)
        }
        self.seed = seed

        # Matriz de transição simulada (modelo de bigrama)
        rng = np.random.default_rng(seed)
        raw = rng.standard_normal((self.vocab_size, self.vocab_size))
        self._matriz_transicao: FloatArray = raw

    def _obter_logits(self, contexto: list[str]) -> FloatArray:
        """Obtém logits baseados no último token do contexto.

        Args:
            contexto: Lista de tokens do contexto.

        Returns:
            Array de logits (vocab_size,).
        """
        if not contexto:
            rng = np.random.default_rng(self.seed)
            return rng.standard_normal(self.vocab_size)

        ultimo_token = contexto[-1]
        idx = self.token_para_idx.get(ultimo_token, 0)
        return self._matriz_transicao[idx].copy()

    def _aplicar_temperatura(
        self, logits: FloatArray, temperatura: float
    ) -> FloatArray:
        """Aplica temperatura aos logits.

        Args:
            logits: Logits originais.
            temperatura: Fator de temperatura (> 0). Menor = mais determinístico.

        Returns:
            Probabilidades após temperatura e softmax.
        """
        if temperatura <= 0:
            raise ValueError("Temperatura deve ser maior que zero")
        return softmax(logits / temperatura)

    def _aplicar_top_k(
        self, probs: FloatArray, k: int
    ) -> FloatArray:
        """Aplica filtragem top-k às probabilidades.

        Args:
            probs: Distribuição de probabilidade.
            k: Número de tokens a manter.

        Returns:
            Probabilidades renormalizadas com apenas os top-k tokens.
        """
        if k <= 0 or k >= len(probs):
            return probs

        indices_ordenados = np.argsort(probs)[::-1]
        mascara = np.zeros_like(probs)
        mascara[indices_ordenados[:k]] = 1.0
        probs_filtradas = probs * mascara
        soma = probs_filtradas.sum()
        return probs_filtradas / soma if soma > 0 else probs

    def _aplicar_top_p(
        self, probs: FloatArray, p: float
    ) -> FloatArray:
        """Aplica filtragem nucleus (top-p) às probabilidades.

        Args:
            probs: Distribuição de probabilidade.
            p: Limiar de probabilidade acumulada.

        Returns:
            Probabilidades renormalizadas mantendo o nucleus.
        """
        if p >= 1.0:
            return probs

        indices_ordenados = np.argsort(probs)[::-1]
        probs_ordenadas = probs[indices_ordenados]
        probs_cumulativas = np.cumsum(probs_ordenadas)

        # Remove tokens além do nucleus
        indices_remover = probs_cumulativas > p
        # Shift para incluir o primeiro token que ultrapassa p
        indices_remover[1:] = indices_remover[:-1].copy()
        indices_remover[0] = False

        indices_fora = indices_ordenados[indices_remover]
        probs_filtradas = probs.copy()
        probs_filtradas[indices_fora] = 0.0
        soma = probs_filtradas.sum()
        return probs_filtradas / soma if soma > 0 else probs

    def _amostrar(
        self, probs: FloatArray, rng: np.random.Generator
    ) -> int:
        """Amostra um índice da distribuição de probabilidade.

        Args:
            probs: Distribuição de probabilidade.
            rng: Gerador de números aleatórios.

        Returns:
            Índice amostrado.
        """
        return int(rng.choice(self.vocab_size, p=probs / probs.sum()))

    def gerar_greedy(
        self,
        prompt: list[str],
        max_tokens: int = 10,
    ) -> GeracaoResult:
        """Geração greedy: sempre escolhe o token mais provável.

        Args:
            prompt: Tokens de contexto inicial.
            max_tokens: Número máximo de tokens a gerar.

        Returns:
            GeracaoResult com texto e probabilidades.
        """
        contexto = list(prompt)
        historico: list[TokenGeradoInfo] = []

        for _ in range(max_tokens):
            logits = self._obter_logits(contexto)
            probs = softmax(logits)

            idx_escolhido = int(np.argmax(probs))
            token = self.vocabulario[idx_escolhido]

            top5_idx = np.argsort(probs)[::-1][:5]
            top5 = [
                {"token": self.vocabulario[i], "probabilidade": float(probs[i])}
                for i in top5_idx
            ]

            historico.append(
                TokenGeradoInfo(
                    token=token,
                    probabilidade=float(probs[idx_escolhido]),
                    logit=float(logits[idx_escolhido]),
                    top_5_tokens=top5,
                )
            )
            contexto.append(token)

        tokens_gerados = [h.token for h in historico]
        return GeracaoResult(
            texto_gerado=" ".join(prompt + tokens_gerados),
            tokens_gerados=tokens_gerados,
            historico_probabilidades=historico,
            estrategia="greedy",
            parametros={"max_tokens": max_tokens},
        )

    def gerar_com_temperatura(
        self,
        prompt: list[str],
        temperatura: float = 1.0,
        max_tokens: int = 10,
    ) -> GeracaoResult:
        """Geração com temperatura: controla aleatoriedade.

        Args:
            prompt: Tokens de contexto inicial.
            temperatura: Fator de temperatura (0 < T <= 2.0).
            max_tokens: Número máximo de tokens a gerar.

        Returns:
            GeracaoResult com texto e probabilidades.
        """
        temperatura = max(0.01, min(temperatura, 5.0))
        contexto = list(prompt)
        historico: list[TokenGeradoInfo] = []
        rng = np.random.default_rng(self.seed)

        for _ in range(max_tokens):
            logits = self._obter_logits(contexto)
            probs = self._aplicar_temperatura(logits, temperatura)

            idx_escolhido = self._amostrar(probs, rng)
            token = self.vocabulario[idx_escolhido]

            top5_idx = np.argsort(probs)[::-1][:5]
            top5 = [
                {"token": self.vocabulario[i], "probabilidade": float(probs[i])}
                for i in top5_idx
            ]

            historico.append(
                TokenGeradoInfo(
                    token=token,
                    probabilidade=float(probs[idx_escolhido]),
                    logit=float(logits[idx_escolhido]),
                    top_5_tokens=top5,
                )
            )
            contexto.append(token)

        tokens_gerados = [h.token for h in historico]
        return GeracaoResult(
            texto_gerado=" ".join(prompt + tokens_gerados),
            tokens_gerados=tokens_gerados,
            historico_probabilidades=historico,
            estrategia="temperatura",
            parametros={"temperatura": temperatura, "max_tokens": max_tokens},
        )

    def gerar_top_k(
        self,
        prompt: list[str],
        k: int = 50,
        temperatura: float = 1.0,
        max_tokens: int = 10,
    ) -> GeracaoResult:
        """Geração top-k: restringe a amostragem aos k tokens mais prováveis.

        Args:
            prompt: Tokens de contexto inicial.
            k: Número de tokens candidatos.
            temperatura: Fator de temperatura.
            max_tokens: Número máximo de tokens a gerar.

        Returns:
            GeracaoResult com texto e probabilidades.
        """
        contexto = list(prompt)
        historico: list[TokenGeradoInfo] = []
        rng = np.random.default_rng(self.seed)

        for _ in range(max_tokens):
            logits = self._obter_logits(contexto)
            probs = self._aplicar_temperatura(logits, temperatura)
            probs_k = self._aplicar_top_k(probs, k)

            idx_escolhido = self._amostrar(probs_k, rng)
            token = self.vocabulario[idx_escolhido]

            top5_idx = np.argsort(probs_k)[::-1][:5]
            top5 = [
                {"token": self.vocabulario[i], "probabilidade": float(probs_k[i])}
                for i in top5_idx
            ]

            historico.append(
                TokenGeradoInfo(
                    token=token,
                    probabilidade=float(probs_k[idx_escolhido]),
                    logit=float(logits[idx_escolhido]),
                    top_5_tokens=top5,
                )
            )
            contexto.append(token)

        tokens_gerados = [h.token for h in historico]
        return GeracaoResult(
            texto_gerado=" ".join(prompt + tokens_gerados),
            tokens_gerados=tokens_gerados,
            historico_probabilidades=historico,
            estrategia="top_k",
            parametros={"k": k, "temperatura": temperatura, "max_tokens": max_tokens},
        )

    def gerar_top_p(
        self,
        prompt: list[str],
        p: float = 0.9,
        temperatura: float = 1.0,
        max_tokens: int = 10,
    ) -> GeracaoResult:
        """Geração nucleus (top-p): mantém tokens até a probabilidade acumulada p.

        Args:
            prompt: Tokens de contexto inicial.
            p: Limiar de probabilidade acumulada (nucleus).
            temperatura: Fator de temperatura.
            max_tokens: Número máximo de tokens a gerar.

        Returns:
            GeracaoResult com texto e probabilidades.
        """
        contexto = list(prompt)
        historico: list[TokenGeradoInfo] = []
        rng = np.random.default_rng(self.seed)

        for _ in range(max_tokens):
            logits = self._obter_logits(contexto)
            probs = self._aplicar_temperatura(logits, temperatura)
            probs_p = self._aplicar_top_p(probs, p)

            idx_escolhido = self._amostrar(probs_p, rng)
            token = self.vocabulario[idx_escolhido]

            top5_idx = np.argsort(probs_p)[::-1][:5]
            top5 = [
                {"token": self.vocabulario[i], "probabilidade": float(probs_p[i])}
                for i in top5_idx
            ]

            historico.append(
                TokenGeradoInfo(
                    token=token,
                    probabilidade=float(probs_p[idx_escolhido]),
                    logit=float(logits[idx_escolhido]),
                    top_5_tokens=top5,
                )
            )
            contexto.append(token)

        tokens_gerados = [h.token for h in historico]
        return GeracaoResult(
            texto_gerado=" ".join(prompt + tokens_gerados),
            tokens_gerados=tokens_gerados,
            historico_probabilidades=historico,
            estrategia="top_p",
            parametros={"p": p, "temperatura": temperatura, "max_tokens": max_tokens},
        )

    def beam_search(
        self,
        prompt: list[str],
        num_beams: int = 3,
        max_tokens: int = 8,
    ) -> list[dict[str, object]]:
        """Beam search: mantém os num_beams caminhos mais prováveis.

        Args:
            prompt: Tokens de contexto inicial.
            num_beams: Número de feixes (beams) a manter.
            max_tokens: Número máximo de tokens a gerar.

        Returns:
            Lista de candidatos finais com texto e log-probabilidade.
        """
        # Inicializa beams: (log_prob, tokens)
        beams: list[tuple[float, list[str]]] = [(0.0, list(prompt))]

        for _ in range(max_tokens):
            novos_candidatos: list[tuple[float, list[str]]] = []

            for log_prob, tokens in beams:
                logits = self._obter_logits(tokens)
                probs = softmax(logits)

                # Expande cada beam com os top-num_beams tokens
                top_idx = np.argsort(probs)[::-1][:num_beams]
                for idx in top_idx:
                    novo_log_prob = log_prob + float(
                        np.log(np.clip(probs[idx], 1e-10, 1.0))
                    )
                    novos_candidatos.append(
                        (novo_log_prob, tokens + [self.vocabulario[idx]])
                    )

            # Mantém apenas os num_beams melhores
            novos_candidatos.sort(key=lambda x: x[0], reverse=True)
            beams = novos_candidatos[:num_beams]

        return [
            {
                "texto": " ".join(tokens[len(prompt):]),
                "texto_completo": " ".join(tokens),
                "log_probabilidade": log_prob,
                "probabilidade": float(np.exp(log_prob)),
                "tokens": tokens[len(prompt):],
            }
            for log_prob, tokens in sorted(beams, key=lambda x: x[0], reverse=True)
        ]

    def demonstrar_temperatura(
        self,
        temperaturas: list[float] | None = None,
    ) -> TemperaturaDemo:
        """Demonstra como a temperatura afeta a distribuição de probabilidade.

        Args:
            temperaturas: Lista de temperaturas a comparar.

        Returns:
            TemperaturaDemo com distribuições para cada temperatura.
        """
        if temperaturas is None:
            temperaturas = [0.1, 0.5, 1.0, 1.5, 2.0]

        # Usa logits fixos para demonstração consistente
        rng = np.random.default_rng(0)
        logits = rng.standard_normal(self.vocab_size)

        # Seleciona os 10 tokens mais relevantes para exibição
        probs_base = softmax(logits)
        top10_idx = np.argsort(probs_base)[::-1][:10]
        tokens_exibidos = [self.vocabulario[i] for i in top10_idx]
        logits_top10 = logits[top10_idx].tolist()

        distribuicoes: list[dict[str, object]] = []
        for temp in temperaturas:
            probs_temp = self._aplicar_temperatura(logits, temp)
            probs_top10 = probs_temp[top10_idx]

            distribuicoes.append(
                {
                    "temperatura": temp,
                    "probabilidades": probs_top10.tolist(),
                    "entropia": entropy(probs_temp),
                    "descricao": _descrever_temperatura(temp),
                }
            )

        return TemperaturaDemo(
            logits_originais=logits_top10,
            tokens=tokens_exibidos,
            distribuicoes=distribuicoes,
        )


def _descrever_temperatura(temperatura: float) -> str:
    """Retorna descrição educacional de um valor de temperatura."""
    if temperatura < 0.3:
        return "Muito determinístico: o modelo quase sempre escolhe o token mais provável"
    elif temperatura < 0.7:
        return "Pouco aleatório: preferência por tokens prováveis, com alguma variação"
    elif temperatura < 1.3:
        return "Balanceado: distribuição próxima do treinamento original"
    elif temperatura < 1.8:
        return "Criativo: mais aleatoriedade, textos mais variados mas menos coerentes"
    else:
        return "Muito aleatório: distribuição quase uniforme, saída geralmente incoerente"
