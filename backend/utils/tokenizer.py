"""Utilitários de tokenização: regex simples e demonstração de BPE."""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BPEMergeStep:
    """Representa um passo da mesclagem BPE."""

    passo: int
    par_mesclado: tuple[str, str]
    novo_token: str
    frequencia: int
    vocabulario_atual: list[str]
    corpus_atual: list[list[str]]


@dataclass
class VocabularioStats:
    """Estatísticas de um vocabulário."""

    total_tokens: int
    tokens_unicos: int
    tokens_mais_comuns: list[tuple[str, int]]
    tamanho_medio_token: float
    tokens_oov: int = 0


def tokenizar_simples(texto: str, max_tokens: int = 10) -> list[str]:
    """Tokeniza texto usando regex simples (palavras e pontuação).

    Args:
        texto: Texto de entrada.
        max_tokens: Número máximo de tokens a retornar.

    Returns:
        Lista de tokens em minúsculo.
    """
    tokens = re.findall(r"\b\w+\b|[.,!?;:]", texto.lower())
    return tokens[:max_tokens]


def tokenizar_subpalavras(texto: str) -> list[str]:
    """Tokenização por subpalavras: divide palavras longas por heurística.

    Args:
        texto: Texto de entrada.

    Returns:
        Lista de subpalavras.
    """
    palavras = re.findall(r"\b\w+\b", texto.lower())
    tokens: list[str] = []
    for palavra in palavras:
        if len(palavra) > 6:
            # Divide em prefixo + sufixo simulando BPE real
            meio = len(palavra) // 2
            tokens.extend([palavra[:meio], "##" + palavra[meio:]])
        else:
            tokens.append(palavra)
    return tokens


def tokenizar_caracteres(texto: str) -> list[str]:
    """Tokeniza texto em nível de caractere.

    Args:
        texto: Texto de entrada.

    Returns:
        Lista de caracteres (excluindo espaços).
    """
    return [c for c in texto.lower() if not c.isspace()]


def comparar_abordagens_tokenizacao(
    texto: str,
) -> dict[str, list[str]]:
    """Compara diferentes abordagens de tokenização para o mesmo texto.

    Args:
        texto: Texto de entrada.

    Returns:
        Dicionário mapeando nome da abordagem para lista de tokens.
    """
    return {
        "palavra": tokenizar_simples(texto, max_tokens=50),
        "subpalavra": tokenizar_subpalavras(texto),
        "caractere": tokenizar_caracteres(texto),
    }


# ---------------------------------------------------------------------------
# Algoritmo BPE demonstrativo
# ---------------------------------------------------------------------------


def _construir_corpus_inicial(textos: list[str]) -> list[list[str]]:
    """Constrói corpus inicial dividindo palavras em caracteres + </w>."""
    corpus: list[list[str]] = []
    for texto in textos:
        palavras = re.findall(r"\b\w+\b", texto.lower())
        for palavra in palavras:
            # Representação BPE: cada caractere separado, </w> marca fim de palavra
            chars = list(palavra) + ["</w>"]
            corpus.append(chars)
    return corpus


def _contar_pares(corpus: list[list[str]]) -> Counter[tuple[str, str]]:
    """Conta a frequência de cada par de símbolos adjacentes no corpus."""
    contagem: Counter[tuple[str, str]] = Counter()
    for sequencia in corpus:
        for i in range(len(sequencia) - 1):
            contagem[(sequencia[i], sequencia[i + 1])] += 1
    return contagem


def _mesclar_par(
    corpus: list[list[str]],
    par: tuple[str, str],
) -> list[list[str]]:
    """Aplica a mesclagem de um par em todo o corpus."""
    novo_corpus: list[list[str]] = []
    bigrama = list(par)
    novo_simbolo = "".join(par)

    for sequencia in corpus:
        nova_seq: list[str] = []
        i = 0
        while i < len(sequencia):
            if (
                i < len(sequencia) - 1
                and sequencia[i] == bigrama[0]
                and sequencia[i + 1] == bigrama[1]
            ):
                nova_seq.append(novo_simbolo)
                i += 2
            else:
                nova_seq.append(sequencia[i])
                i += 1
        novo_corpus.append(nova_seq)
    return novo_corpus


def executar_bpe_passos(
    textos: list[str],
    num_mesclagens: int = 10,
) -> list[BPEMergeStep]:
    """Executa BPE e retorna cada passo para animação.

    Args:
        textos: Lista de textos para construir vocabulário.
        num_mesclagens: Número de operações de mesclagem a realizar.

    Returns:
        Lista de passos BPE, cada um descrevendo a mesclagem realizada.
    """
    corpus = _construir_corpus_inicial(textos)
    passos: list[BPEMergeStep] = []

    for passo in range(1, num_mesclagens + 1):
        pares = _contar_pares(corpus)
        if not pares:
            break

        melhor_par = pares.most_common(1)[0]
        par, frequencia = melhor_par

        corpus = _mesclar_par(corpus, par)

        vocab_atual = sorted({token for seq in corpus for token in seq})

        passos.append(
            BPEMergeStep(
                passo=passo,
                par_mesclado=par,
                novo_token="".join(par),
                frequencia=frequencia,
                vocabulario_atual=vocab_atual,
                corpus_atual=[list(seq) for seq in corpus[:5]],  # amostra
            )
        )

    return passos


def calcular_estatisticas_vocabulario(tokens: list[str]) -> VocabularioStats:
    """Calcula estatísticas descritivas de uma lista de tokens.

    Args:
        tokens: Lista de tokens.

    Returns:
        Objeto com estatísticas do vocabulário.
    """
    contador = Counter(tokens)
    tamanho_medio = (
        sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0
    )
    return VocabularioStats(
        total_tokens=len(tokens),
        tokens_unicos=len(contador),
        tokens_mais_comuns=contador.most_common(10),
        tamanho_medio_token=round(tamanho_medio, 2),
    )
