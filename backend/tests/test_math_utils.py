"""Testes abrangentes para utils/math_utils.py.

Cobre: softmax, cosine_similarity, entropy, reduce_pca, reduce_tsne,
reduce_dimensions.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest
from numpy.typing import NDArray

from utils.math_utils import (
    cosine_similarity,
    entropy,
    reduce_dimensions,
    reduce_pca,
    reduce_tsne,
    softmax,
)

FloatArray = NDArray[np.float64]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embeddings_10x8() -> FloatArray:
    """10 amostras com 8 dimensões — suficiente para PCA e t-SNE."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((10, 8)).astype(np.float64)


@pytest.fixture
def embeddings_2x8() -> FloatArray:
    """2 amostras com 8 dimensões — n_samples < n_components para testar padding."""
    rng = np.random.default_rng(1)
    return rng.standard_normal((2, 8)).astype(np.float64)


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------


class TestSoftmax:
    """Testes para a função softmax."""

    def test_1d_soma_um(self) -> None:
        """Softmax 1D deve somar exatamente 1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        resultado = softmax(x)
        np.testing.assert_allclose(resultado.sum(), 1.0, atol=1e-7)

    def test_2d_cada_linha_soma_um(self) -> None:
        """Softmax 2D com axis=-1: cada linha deve somar 1.0."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal((5, 8)).astype(np.float64)
        resultado = softmax(x)
        somas = resultado.sum(axis=-1)
        np.testing.assert_allclose(somas, np.ones(5), atol=1e-7)

    def test_estabilidade_numerica_valores_grandes(self) -> None:
        """Softmax com valores muito grandes não deve produzir NaN ou Inf."""
        x = np.array([1000.0, 1000.0, 1000.0], dtype=np.float64)
        resultado = softmax(x)
        assert not np.any(np.isnan(resultado)), "softmax retornou NaN"
        assert not np.any(np.isinf(resultado)), "softmax retornou Inf"
        np.testing.assert_allclose(resultado.sum(), 1.0, atol=1e-7)

    def test_estabilidade_numerica_valores_muito_diferentes(self) -> None:
        """Softmax não produz NaN mesmo com grande spread entre valores."""
        x = np.array([0.0, 1000.0, -1000.0], dtype=np.float64)
        resultado = softmax(x)
        assert not np.any(np.isnan(resultado))
        assert not np.any(np.isinf(resultado))
        np.testing.assert_allclose(resultado.sum(), 1.0, atol=1e-7)

    def test_valores_negativos(self) -> None:
        """Softmax deve funcionar corretamente com entradas negativas."""
        x = np.array([-3.0, -1.0, -2.0], dtype=np.float64)
        resultado = softmax(x)
        np.testing.assert_allclose(resultado.sum(), 1.0, atol=1e-7)
        # -1.0 é o maior valor, deve ter a maior probabilidade
        assert resultado[1] > resultado[0]
        assert resultado[1] > resultado[2]

    def test_valores_uniformes_distribuicao_uniforme(self) -> None:
        """Entradas iguais devem produzir probabilidades iguais."""
        x = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
        resultado = softmax(x)
        np.testing.assert_allclose(resultado, np.full(4, 0.25), atol=1e-7)

    def test_shape_preservado(self) -> None:
        """Shape de saída deve ser igual ao shape de entrada."""
        x = np.zeros((3, 4, 5), dtype=np.float64)
        assert softmax(x).shape == x.shape


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    """Testes para a função cosine_similarity."""

    def test_vetores_identicos_retorna_1(self) -> None:
        """Vetores idênticos têm similaridade de cosseno 1.0."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        resultado = cosine_similarity(v, v)
        assert resultado == pytest.approx(1.0, abs=1e-7)

    def test_vetores_identicos_escala_diferente(self) -> None:
        """Vetores na mesma direção (escala diferente) têm similaridade 1.0."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        b = a * 5.0
        resultado = cosine_similarity(a, b)
        assert resultado == pytest.approx(1.0, abs=1e-7)

    def test_vetores_ortogonais_retorna_0(self) -> None:
        """Vetores ortogonais têm similaridade de cosseno 0.0."""
        a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        resultado = cosine_similarity(a, b)
        assert resultado == pytest.approx(0.0, abs=1e-7)

    def test_vetores_opostos_retorna_menos_1(self) -> None:
        """Vetores opostos têm similaridade de cosseno -1.0."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        resultado = cosine_similarity(v, -v)
        assert resultado == pytest.approx(-1.0, abs=1e-7)

    def test_vetor_zero_retorna_0(self) -> None:
        """Qualquer vetor comparado com vetor zero deve retornar 0.0."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        zero = np.zeros(3, dtype=np.float64)
        assert cosine_similarity(v, zero) == 0.0
        assert cosine_similarity(zero, v) == 0.0
        assert cosine_similarity(zero, zero) == 0.0

    def test_retorno_e_float_python(self) -> None:
        """Deve retornar float Python, não numpy scalar."""
        a = np.array([1.0, 0.0], dtype=np.float64)
        b = np.array([0.0, 1.0], dtype=np.float64)
        resultado = cosine_similarity(a, b)
        assert isinstance(resultado, float)

    def test_intervalo_entre_menos1_e_1(self) -> None:
        """Resultado deve estar sempre no intervalo [-1, 1]."""
        rng = np.random.default_rng(7)
        for _ in range(50):
            a = rng.standard_normal(16).astype(np.float64)
            b = rng.standard_normal(16).astype(np.float64)
            resultado = cosine_similarity(a, b)
            assert -1.0 - 1e-7 <= resultado <= 1.0 + 1e-7


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------


class TestEntropy:
    """Testes para a função entropy."""

    def test_distribuicao_uniforme_entropia_maxima(self) -> None:
        """Distribuição uniforme deve ter entropia máxima log(n)."""
        n = 4
        probs = np.full(n, 1.0 / n, dtype=np.float64)
        esperado = math.log(n)
        resultado = entropy(probs)
        assert resultado == pytest.approx(esperado, rel=1e-5)

    def test_distribuicao_deterministica_entropia_zero(self) -> None:
        """Distribuição one-hot deve ter entropia próxima de zero."""
        probs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        resultado = entropy(probs)
        # O clip em 1e-10 faz com que zeros contribuam minimamente,
        # mas o resultado deve ser muito próximo de zero.
        assert resultado == pytest.approx(0.0, abs=1e-4)

    def test_entropia_entre_zero_e_maximo(self) -> None:
        """Entropia de distribuição geral deve estar entre 0 e log(n)."""
        probs = np.array([0.5, 0.3, 0.2], dtype=np.float64)
        n = len(probs)
        resultado = entropy(probs)
        assert 0.0 <= resultado <= math.log(n) + 1e-9

    def test_retorno_e_float_python(self) -> None:
        """Deve retornar float Python, não numpy scalar."""
        probs = np.array([0.5, 0.5], dtype=np.float64)
        assert isinstance(entropy(probs), float)

    def test_sem_nan_com_zeros(self) -> None:
        """Probabilidades zero não devem causar NaN (graças ao clip)."""
        probs = np.array([0.0, 0.5, 0.5], dtype=np.float64)
        resultado = entropy(probs)
        assert not math.isnan(resultado)

    def test_entropia_cresce_com_incerteza(self) -> None:
        """Entropia maior para distribuição mais incerta."""
        mais_certa = np.array([0.9, 0.05, 0.05], dtype=np.float64)
        mais_incerta = np.array([0.4, 0.3, 0.3], dtype=np.float64)
        assert entropy(mais_incerta) > entropy(mais_certa)


# ---------------------------------------------------------------------------
# reduce_pca
# ---------------------------------------------------------------------------


class TestReducePca:
    """Testes para a função reduce_pca."""

    def test_shape_saida_correto(self, embeddings_10x8: FloatArray) -> None:
        """Saída deve ter shape (n_amostras, n_components)."""
        resultado = reduce_pca(embeddings_10x8, n_components=3)
        assert len(resultado) == 10
        assert all(len(linha) == 3 for linha in resultado)

    def test_retorno_e_lista_de_listas(self, embeddings_10x8: FloatArray) -> None:
        """Retorno deve ser list[list[float]], não ndarray."""
        resultado = reduce_pca(embeddings_10x8, n_components=3)
        assert isinstance(resultado, list)
        assert isinstance(resultado[0], list)
        assert isinstance(resultado[0][0], float)

    def test_determinismo_mesma_entrada(self, embeddings_10x8: FloatArray) -> None:
        """Mesma entrada deve produzir mesma saída (PCA é determinístico)."""
        r1 = reduce_pca(embeddings_10x8, n_components=3)
        r2 = reduce_pca(embeddings_10x8, n_components=3)
        np.testing.assert_allclose(
            np.array(r1), np.array(r2), atol=1e-10
        )

    def test_poucos_samples_padding_com_zeros(self, embeddings_2x8: FloatArray) -> None:
        """n_samples < n_components deve completar as colunas extras com zeros."""
        n_components = 3
        resultado = reduce_pca(embeddings_2x8, n_components=n_components)
        # Shape correto
        assert len(resultado) == 2
        assert all(len(linha) == n_components for linha in resultado)
        # A terceira coluna (índice 2) deve ser zeros (padding)
        arr = np.array(resultado)
        np.testing.assert_allclose(arr[:, 2], np.zeros(2), atol=1e-10)

    def test_n_components_diferente(self, embeddings_10x8: FloatArray) -> None:
        """Deve respeitar n_components=2."""
        resultado = reduce_pca(embeddings_10x8, n_components=2)
        assert all(len(linha) == 2 for linha in resultado)

    def test_sem_nan_na_saida(self, embeddings_10x8: FloatArray) -> None:
        """Saída não deve conter NaN."""
        arr = np.array(reduce_pca(embeddings_10x8, n_components=3))
        assert not np.any(np.isnan(arr))


# ---------------------------------------------------------------------------
# reduce_tsne
# ---------------------------------------------------------------------------


class TestReduceTsne:
    """Testes para a função reduce_tsne."""

    def test_shape_saida_correto(self, embeddings_10x8: FloatArray) -> None:
        """Saída deve ter shape (n_amostras, n_components)."""
        resultado = reduce_tsne(embeddings_10x8, n_components=3)
        assert len(resultado) == 10
        assert all(len(linha) == 3 for linha in resultado)

    def test_retorno_e_lista_de_listas(self, embeddings_10x8: FloatArray) -> None:
        """Retorno deve ser list[list[float]], não ndarray."""
        resultado = reduce_tsne(embeddings_10x8, n_components=3)
        assert isinstance(resultado, list)
        assert isinstance(resultado[0], list)
        assert isinstance(resultado[0][0], float)

    def test_determinismo_mesma_semente(self, embeddings_10x8: FloatArray) -> None:
        """Mesma semente deve produzir a mesma saída."""
        r1 = reduce_tsne(embeddings_10x8, n_components=3, random_state=42)
        r2 = reduce_tsne(embeddings_10x8, n_components=3, random_state=42)
        np.testing.assert_allclose(
            np.array(r1), np.array(r2), atol=1e-6
        )

    def test_sem_nan_na_saida(self, embeddings_10x8: FloatArray) -> None:
        """Saída não deve conter NaN."""
        arr = np.array(reduce_tsne(embeddings_10x8, n_components=3))
        assert not np.any(np.isnan(arr))

    def test_perplexidade_ajustada_para_poucos_samples(self) -> None:
        """Perplexidade deve ser ajustada automaticamente quando há poucos samples."""
        rng = np.random.default_rng(2)
        # 4 amostras, perplexidade padrão 5 seria >= n_samples
        emb = rng.standard_normal((4, 8)).astype(np.float64)
        # Não deve lançar exceção
        resultado = reduce_tsne(emb, n_components=2, perplexity=5.0)
        assert len(resultado) == 4
        assert all(len(linha) == 2 for linha in resultado)


# ---------------------------------------------------------------------------
# reduce_dimensions (dispatcher)
# ---------------------------------------------------------------------------


class TestReduceDimensions:
    """Testes para a função reduce_dimensions (dispatcher)."""

    def test_metodo_pca_chama_reduce_pca(self, embeddings_10x8: FloatArray) -> None:
        """method='pca' deve delegar para reduce_pca."""
        with patch(
            "utils.math_utils.reduce_pca", wraps=reduce_pca
        ) as mock_pca:
            reduce_dimensions(embeddings_10x8, method="pca", n_components=3)
            mock_pca.assert_called_once_with(embeddings_10x8, n_components=3)

    def test_metodo_tsne_chama_reduce_tsne(self, embeddings_10x8: FloatArray) -> None:
        """method='tsne' deve delegar para reduce_tsne."""
        with patch(
            "utils.math_utils.reduce_tsne", wraps=reduce_tsne
        ) as mock_tsne:
            reduce_dimensions(embeddings_10x8, method="tsne", n_components=3)
            mock_tsne.assert_called_once_with(embeddings_10x8, n_components=3)

    def test_padrao_e_pca(self, embeddings_10x8: FloatArray) -> None:
        """Método padrão deve ser PCA."""
        resultado_dispatcher = reduce_dimensions(embeddings_10x8, n_components=3)
        resultado_pca = reduce_pca(embeddings_10x8, n_components=3)
        np.testing.assert_allclose(
            np.array(resultado_dispatcher), np.array(resultado_pca), atol=1e-10
        )

    def test_saida_pca_shape_correto(self, embeddings_10x8: FloatArray) -> None:
        """Dispatcher com PCA deve retornar shape (n_samples, n_components)."""
        resultado = reduce_dimensions(embeddings_10x8, method="pca", n_components=3)
        assert len(resultado) == 10
        assert all(len(linha) == 3 for linha in resultado)

    def test_saida_tsne_shape_correto(self, embeddings_10x8: FloatArray) -> None:
        """Dispatcher com t-SNE deve retornar shape (n_samples, n_components)."""
        resultado = reduce_dimensions(embeddings_10x8, method="tsne", n_components=3)
        assert len(resultado) == 10
        assert all(len(linha) == 3 for linha in resultado)
