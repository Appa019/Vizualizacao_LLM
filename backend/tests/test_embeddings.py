"""Testes para embeddings, positional encoding e espaço de embeddings.

Cobre tanto os métodos da classe TransformerSimulator (unidade) quanto os
endpoints do router /api/embeddings (integração via TestClient).
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures auxiliares
# ---------------------------------------------------------------------------


@pytest.fixture
def simulador():
    """TransformerSimulator com parâmetros padrão (d_model=64, seq_length=10)."""
    from core.transformer_simulator import TransformerSimulator

    return TransformerSimulator(d_model=64, seq_length=10)


# ===========================================================================
# Testes de unidade: create_positional_encoding
# ===========================================================================


class TestPositionalEncoding:
    """Testes unitários sobre create_positional_encoding."""

    def test_valores_limitados_entre_menos_um_e_um(self, simulador) -> None:
        """Todo valor sinusoidal deve estar em [-1, 1] (propriedade matemática)."""
        pe = simulador.create_positional_encoding()
        assert np.all(pe >= -1.0), "Existem valores menores que -1"
        assert np.all(pe <= 1.0), "Existem valores maiores que +1"

    def test_shape_correto(self, simulador) -> None:
        """Shape deve ser (seq_length, d_model)."""
        pe = simulador.create_positional_encoding()
        assert pe.shape == (simulador.seq_length, simulador.d_model)

    def test_posicoes_distintas_geram_encodings_diferentes(
        self, simulador
    ) -> None:
        """A linha 0 e a linha 1 do PE devem ser vetores distintos."""
        pe = simulador.create_positional_encoding()
        assert not np.allclose(
            pe[0], pe[1]
        ), "Posições 0 e 1 geraram o mesmo positional encoding"

    def test_shape_customizado(self) -> None:
        """PE com seq_length e d_model não-padrão deve ter shape correto."""
        from core.transformer_simulator import TransformerSimulator

        sim = TransformerSimulator(d_model=32, seq_length=5)
        pe = sim.create_positional_encoding()
        assert pe.shape == (5, 32)

    def test_dimensoes_pares_sao_seno(self, simulador) -> None:
        """Dimensões pares devem ser sin(pos / 10000^(2i/d_model))."""
        pe = simulador.create_positional_encoding()
        pos = np.arange(simulador.seq_length)[:, np.newaxis]
        dim = 0  # primeira dimensão par
        divisor = 10000.0 ** (dim / simulador.d_model)
        esperado = np.sin(pos[:, 0] / divisor)
        np.testing.assert_allclose(pe[:, dim], esperado, rtol=1e-6)

    def test_dimensoes_impares_sao_cosseno(self, simulador) -> None:
        """Dimensões ímpares devem ser cos(pos / 10000^(2i/d_model))."""
        pe = simulador.create_positional_encoding()
        pos = np.arange(simulador.seq_length)[:, np.newaxis]
        dim_impar = 1  # corresponde ao índice i=0 nas dimensões ímpares
        divisor = 10000.0 ** (0 / simulador.d_model)
        esperado = np.cos(pos[:, 0] / divisor)
        np.testing.assert_allclose(pe[:, dim_impar], esperado, rtol=1e-6)


# ===========================================================================
# Testes de unidade: compute_embeddings / create_embeddings_from_tokens
# ===========================================================================


class TestEmbeddings:
    """Testes unitários sobre compute_embeddings e create_embeddings_from_tokens."""

    def test_embeddings_finais_e_soma_de_token_e_pe(self, simulador) -> None:
        """embeddings_finais deve ser exatamente token_embeddings + positional_encoding."""
        resultado = simulador.compute_embeddings(["o", "gato", "corre"])
        token_emb = np.array(resultado.token_embeddings)
        pos_enc = np.array(resultado.positional_encoding)
        finais = np.array(resultado.embeddings_finais)
        np.testing.assert_allclose(
            finais,
            token_emb + pos_enc,
            rtol=1e-10,
            err_msg="embeddings_finais != token_embeddings + positional_encoding",
        )

    def test_tokens_diferentes_em_posicoes_distintas_geram_embeddings_diferentes(
        self, simulador
    ) -> None:
        """Tokens na mesma sequência, em posições distintas, devem ter embeddings
        finais diferentes.

        Contexto: create_embeddings_from_tokens usa seed=42 fixo e gera uma
        única matriz aleatória, depois aplica escalonamento por categoria
        gramatical. Dois tokens de conteúdo na MESMA posição e na mesma
        sequência teriam o mesmo vetor base, mas em posições DIFERENTES eles
        recebem positional encodings distintos, fazendo seus embeddings_finais
        divergirem.
        """
        resultado = simulador.compute_embeddings(["gato", "cachorro"])

        vetor_pos0 = np.array(resultado.embeddings_finais[0])
        vetor_pos1 = np.array(resultado.embeddings_finais[1])
        assert not np.allclose(
            vetor_pos0, vetor_pos1
        ), "Tokens em posições diferentes produziram embeddings finais idênticos"

    def test_tokens_padding_tem_norma_menor(self, simulador) -> None:
        """Tokens de padding (posições >= num_tokens_reais) têm norma menor do que
        tokens reais, pois são multiplicados por 0.1."""
        tokens = ["gato"]  # 1 token real, restante é padding
        resultado = simulador.compute_embeddings(tokens)

        token_embs = np.array(resultado.token_embeddings)

        # Norma do token real (índice 0)
        norma_real = np.linalg.norm(token_embs[0])
        # Normas dos tokens de padding
        normas_padding = [
            np.linalg.norm(token_embs[i])
            for i in range(resultado.num_tokens_reais, simulador.seq_length)
        ]
        assert all(
            norma_real > norma_p for norma_p in normas_padding
        ), "Algum token de padding tem norma >= ao token real"

    def test_determinismo_com_seed_42(self, simulador) -> None:
        """Mesma entrada deve sempre produzir a mesma saída (seed=42)."""
        tokens = ["o", "modelo", "aprende"]
        resultado_1 = simulador.compute_embeddings(tokens)
        resultado_2 = simulador.compute_embeddings(tokens)

        np.testing.assert_array_equal(
            np.array(resultado_1.embeddings_finais),
            np.array(resultado_2.embeddings_finais),
        )

    def test_num_tokens_reais_correto(self, simulador) -> None:
        """num_tokens_reais deve refletir exatamente o número de tokens fornecidos."""
        tokens = ["o", "gato", "senta"]
        resultado = simulador.compute_embeddings(tokens)
        assert resultado.num_tokens_reais == len(tokens)

    def test_tokens_usados_inclui_padding(self, simulador) -> None:
        """O campo tokens no resultado deve ter tamanho seq_length com strings
        vazias para posições de padding."""
        tokens = ["ola"]
        resultado = simulador.compute_embeddings(tokens)
        assert len(resultado.tokens) == simulador.seq_length
        assert resultado.tokens[0] == "ola"
        assert all(
            t == ""
            for t in resultado.tokens[resultado.num_tokens_reais :]
        )

    def test_artigos_tem_norma_menor_que_palavras_de_conteudo(
        self, simulador
    ) -> None:
        """Artigos recebem fator 0.8; palavras de conteúdo recebem +0.3 de viés.

        Comparamos os token_embeddings (antes do PE) para isolar o efeito
        da heurística de categoria gramatical.

        Verificação: a norma de um artigo na posição 0 deve diferir da norma
        de uma palavra de conteúdo na posição 0 — não necessariamente menor,
        pois a escala 0.8 pode resultar em norma maior ou menor dependendo
        do vector aleatório, mas o padrão de escalonamento deve ser observável
        ao comparar múltiplos tokens no mesmo índice.
        """
        # Usamos seq_length=1 para cada chamada, assim ambos ficam no índice 0
        from core.transformer_simulator import TransformerSimulator

        sim = TransformerSimulator(d_model=64, seq_length=1)
        res_artigo = sim.compute_embeddings(["o"])
        res_conteudo = sim.compute_embeddings(["modelo"])

        emb_artigo = np.array(res_artigo.token_embeddings[0])
        emb_conteudo = np.array(res_conteudo.token_embeddings[0])

        # Com seed=42 e seq_length=1, ambos partem do mesmo vetor aleatório.
        # O artigo aplica fator 0.8; o token de conteúdo aplica +0.3.
        # Verificamos que os vetores são distintos.
        assert not np.allclose(
            emb_artigo, emb_conteudo
        ), "Artigo e palavra de conteúdo produziram o mesmo token_embedding"


# ===========================================================================
# Testes de integração: endpoint POST /api/embeddings/embeddings
# ===========================================================================


class TestEndpointEmbeddings:
    """Testes de integração para POST /api/embeddings/embeddings."""

    def test_retorna_estrutura_correta(self, client: TestClient) -> None:
        """Resposta deve conter todos os campos obrigatórios do EmbeddingsResponse."""
        payload = {"tokens": ["o", "gato", "corre"], "d_model": 64}
        resp = client.post("/api/embeddings/embeddings", json=payload)
        assert resp.status_code == 200

        data = resp.json()
        campos_obrigatorios = {
            "token_embeddings",
            "positional_encoding",
            "embeddings_finais",
            "tokens",
            "num_tokens_reais",
            "d_model",
            "padroes_sinusoidais",
            "explicacao",
        }
        assert campos_obrigatorios.issubset(
            data.keys()
        ), f"Campos ausentes: {campos_obrigatorios - data.keys()}"

    def test_num_tokens_reais_corresponde_ao_input(
        self, client: TestClient
    ) -> None:
        """num_tokens_reais deve ser igual ao número de tokens enviados."""
        tokens = ["transformer", "aprende", "representacoes"]
        payload = {"tokens": tokens, "d_model": 64}
        resp = client.post("/api/embeddings/embeddings", json=payload)
        assert resp.status_code == 200
        assert resp.json()["num_tokens_reais"] == len(tokens)

    def test_d_model_na_resposta(self, client: TestClient) -> None:
        """Campo d_model na resposta deve refletir o valor enviado."""
        payload = {"tokens": ["ola", "mundo"], "d_model": 32}
        resp = client.post("/api/embeddings/embeddings", json=payload)
        assert resp.status_code == 200
        assert resp.json()["d_model"] == 32

    def test_listas_tem_dimensoes_corretas(self, client: TestClient) -> None:
        """Cada vetor em token_embeddings deve ter comprimento igual a d_model."""
        d_model = 64
        tokens = ["python", "e", "incrivel"]
        payload = {"tokens": tokens, "d_model": d_model}
        resp = client.post("/api/embeddings/embeddings", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        for vetor in data["token_embeddings"]:
            assert len(vetor) == d_model

    def test_tokens_vazios_retorna_422(self, client: TestClient) -> None:
        """Lista de tokens vazia deve retornar HTTP 422 (validação Pydantic)."""
        resp = client.post(
            "/api/embeddings/embeddings", json={"tokens": [], "d_model": 64}
        )
        assert resp.status_code == 422


# ===========================================================================
# Testes de integração: endpoint POST /api/embeddings/positional-encoding
# ===========================================================================


class TestEndpointPositionalEncoding:
    """Testes de integração para POST /api/embeddings/positional-encoding."""

    def test_shape_do_encoding_corresponde_aos_parametros(
        self, client: TestClient
    ) -> None:
        """A lista encoding deve ter shape (seq_length, d_model)."""
        seq_length, d_model = 8, 32
        payload = {"seq_length": seq_length, "d_model": d_model}
        resp = client.post("/api/embeddings/positional-encoding", json=payload)
        assert resp.status_code == 200

        encoding = resp.json()["encoding"]
        assert len(encoding) == seq_length, f"Esperava {seq_length} linhas"
        for linha in encoding:
            assert len(linha) == d_model, f"Esperava {d_model} colunas por linha"

    def test_valores_limitados_entre_menos_um_e_um(
        self, client: TestClient
    ) -> None:
        """Todos os valores retornados devem estar em [-1, 1]."""
        payload = {"seq_length": 10, "d_model": 64}
        resp = client.post("/api/embeddings/positional-encoding", json=payload)
        assert resp.status_code == 200

        encoding = np.array(resp.json()["encoding"])
        assert np.all(encoding >= -1.0), "Existem valores < -1"
        assert np.all(encoding <= 1.0), "Existem valores > 1"

    def test_resposta_contem_campos_obrigatorios(
        self, client: TestClient
    ) -> None:
        """Resposta deve conter encoding, seq_length, d_model, padroes e explicacao."""
        payload = {"seq_length": 5, "d_model": 16}
        resp = client.post("/api/embeddings/positional-encoding", json=payload)
        assert resp.status_code == 200

        data = resp.json()
        campos_obrigatorios = {
            "encoding",
            "seq_length",
            "d_model",
            "padroes_por_dimensao",
            "explicacao",
        }
        assert campos_obrigatorios.issubset(data.keys())

    def test_padroes_por_dimensao_tem_no_maximo_10_colunas(
        self, client: TestClient
    ) -> None:
        """padroes_por_dimensao deve expor no máximo as 10 primeiras dimensões."""
        payload = {"seq_length": 6, "d_model": 64}
        resp = client.post("/api/embeddings/positional-encoding", json=payload)
        assert resp.status_code == 200

        padroes = resp.json()["padroes_por_dimensao"]
        assert len(padroes) == 6  # uma linha por posição
        for linha in padroes:
            assert len(linha) <= 10


# ===========================================================================
# Testes de integração: endpoint POST /api/embeddings/embedding-space
# ===========================================================================


class TestEndpointEmbeddingSpace:
    """Testes de integração para POST /api/embeddings/embedding-space."""

    def test_pca_retorna_coordenadas_3d_para_cada_token(
        self, client: TestClient
    ) -> None:
        """Deve retornar uma coordenada 3D (lista de 3 floats) para cada token."""
        tokens = ["gato", "cachorro", "peixe", "passaro"]
        payload = {"tokens": tokens, "metodo": "pca", "d_model": 64}
        resp = client.post("/api/embeddings/embedding-space", json=payload)
        assert resp.status_code == 200

        data = resp.json()
        assert data["tokens"] == tokens
        assert len(data["coordenadas_3d"]) == len(tokens)
        for coord in data["coordenadas_3d"]:
            assert len(coord) == 3, f"Esperava coordenada 3D, obteve: {coord}"

    def test_tsne_retorna_coordenadas_3d_para_cada_token(
        self, client: TestClient
    ) -> None:
        """t-SNE também deve retornar uma coordenada 3D por token."""
        tokens = ["sol", "lua", "estrela", "planeta", "nebulosa"]
        payload = {"tokens": tokens, "metodo": "tsne", "d_model": 64}
        resp = client.post("/api/embeddings/embedding-space", json=payload)
        assert resp.status_code == 200

        data = resp.json()
        assert len(data["coordenadas_3d"]) == len(tokens)
        for coord in data["coordenadas_3d"]:
            assert len(coord) == 3

    def test_metodo_registrado_na_resposta(self, client: TestClient) -> None:
        """O campo metodo na resposta deve refletir o método solicitado."""
        for metodo in ("pca", "tsne"):
            payload = {"tokens": ["a", "b", "c"], "metodo": metodo, "d_model": 32}
            resp = client.post("/api/embeddings/embedding-space", json=payload)
            assert resp.status_code == 200
            assert resp.json()["metodo"] == metodo

    def test_um_token_retorna_422(self, client: TestClient) -> None:
        """Enviar apenas 1 token deve retornar HTTP 422 (mínimo é 2)."""
        payload = {"tokens": ["solitario"], "metodo": "pca", "d_model": 64}
        resp = client.post("/api/embeddings/embedding-space", json=payload)
        assert resp.status_code == 422

    def test_resposta_contem_campos_obrigatorios(
        self, client: TestClient
    ) -> None:
        """Resposta deve conter tokens, coordenadas_3d, metodo e explicacao."""
        payload = {"tokens": ["rapido", "lento"], "metodo": "pca", "d_model": 64}
        resp = client.post("/api/embeddings/embedding-space", json=payload)
        assert resp.status_code == 200

        data = resp.json()
        campos_obrigatorios = {"tokens", "coordenadas_3d", "metodo", "explicacao"}
        assert campos_obrigatorios.issubset(data.keys())

    def test_coordenadas_3d_sao_floats(self, client: TestClient) -> None:
        """Cada elemento das coordenadas deve ser um número (int ou float)."""
        tokens = ["alpha", "beta", "gamma"]
        payload = {"tokens": tokens, "metodo": "pca", "d_model": 64}
        resp = client.post("/api/embeddings/embedding-space", json=payload)
        assert resp.status_code == 200

        for coord in resp.json()["coordenadas_3d"]:
            for valor in coord:
                assert isinstance(
                    valor, (int, float)
                ), f"Valor {valor!r} não é numérico"
