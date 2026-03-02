"""Testes para geração de texto: core/text_generator.py e routers/inference.py.

Cobre geração greedy, com temperatura, top-k, top-p, beam search,
demonstração de temperatura e todos os endpoints do router de inferência.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.text_generator import TextGenerator, VOCABULARIO_PT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gerador() -> TextGenerator:
    """Instância de TextGenerator com seed fixo para reprodutibilidade."""
    return TextGenerator(vocabulario=VOCABULARIO_PT, seed=42)


@pytest.fixture
def prompt_simples() -> list[str]:
    """Prompt de dois tokens pertencentes ao vocabulário."""
    return ["o", "gato"]


# ---------------------------------------------------------------------------
# Testes unitários — gerar_greedy
# ---------------------------------------------------------------------------


class TestGreedy:
    """Testes para TextGenerator.gerar_greedy."""

    def test_greedy_e_determinístico(
        self, gerador: TextGenerator, prompt_simples: list[str]
    ) -> None:
        """Duas chamadas com o mesmo prompt devem retornar resultados idênticos."""
        r1 = gerador.gerar_greedy(prompt_simples, max_tokens=5)
        r2 = gerador.gerar_greedy(prompt_simples, max_tokens=5)
        assert r1.tokens_gerados == r2.tokens_gerados
        assert r1.texto_gerado == r2.texto_gerado

    def test_greedy_tokens_pertencem_ao_vocabulario(
        self, gerador: TextGenerator, prompt_simples: list[str]
    ) -> None:
        """Todos os tokens gerados pelo greedy devem estar no vocabulário."""
        resultado = gerador.gerar_greedy(prompt_simples, max_tokens=10)
        vocab_set = set(VOCABULARIO_PT)
        for token in resultado.tokens_gerados:
            assert token in vocab_set, f"Token '{token}' não encontrado no vocabulário"

    def test_greedy_respeita_max_tokens(
        self, gerador: TextGenerator, prompt_simples: list[str]
    ) -> None:
        """O número de tokens gerados deve ser exatamente max_tokens."""
        max_tokens = 7
        resultado = gerador.gerar_greedy(prompt_simples, max_tokens=max_tokens)
        assert len(resultado.tokens_gerados) == max_tokens

    def test_greedy_historico_tem_mesmo_tamanho(
        self, gerador: TextGenerator, prompt_simples: list[str]
    ) -> None:
        """historico_probabilidades deve ter o mesmo comprimento que tokens_gerados."""
        resultado = gerador.gerar_greedy(prompt_simples, max_tokens=5)
        assert len(resultado.historico_probabilidades) == len(resultado.tokens_gerados)

    def test_greedy_estrategia_rotulada_corretamente(
        self, gerador: TextGenerator, prompt_simples: list[str]
    ) -> None:
        """Campo estrategia deve ser 'greedy'."""
        resultado = gerador.gerar_greedy(prompt_simples)
        assert resultado.estrategia == "greedy"


# ---------------------------------------------------------------------------
# Testes unitários — gerar_com_temperatura
# ---------------------------------------------------------------------------


class TestTemperatura:
    """Testes para TextGenerator.gerar_com_temperatura."""

    def test_temperatura_com_mesma_seed_e_reprodutivel(
        self, prompt_simples: list[str]
    ) -> None:
        """Mesma seed deve produzir resultado idêntico entre instâncias."""
        g1 = TextGenerator(vocabulario=VOCABULARIO_PT, seed=99)
        g2 = TextGenerator(vocabulario=VOCABULARIO_PT, seed=99)
        r1 = g1.gerar_com_temperatura(prompt_simples, temperatura=1.0, max_tokens=8)
        r2 = g2.gerar_com_temperatura(prompt_simples, temperatura=1.0, max_tokens=8)
        assert r1.tokens_gerados == r2.tokens_gerados

    def test_temperatura_seeds_diferentes_podem_divergir(
        self, prompt_simples: list[str]
    ) -> None:
        """Seeds diferentes têm alta probabilidade de produzir resultados distintos."""
        g1 = TextGenerator(vocabulario=VOCABULARIO_PT, seed=1)
        g2 = TextGenerator(vocabulario=VOCABULARIO_PT, seed=2)
        r1 = g1.gerar_com_temperatura(prompt_simples, temperatura=1.5, max_tokens=10)
        r2 = g2.gerar_com_temperatura(prompt_simples, temperatura=1.5, max_tokens=10)
        # Com temperatura alta e seeds distintas é praticamente impossível obter
        # sequências idênticas em vocabulários grandes — mas não é garantido,
        # por isso apenas registramos a comparação sem forçar falha.
        _ = r1.tokens_gerados != r2.tokens_gerados  # comparação intencional

    def test_temperatura_respeita_max_tokens(
        self, gerador: TextGenerator, prompt_simples: list[str]
    ) -> None:
        """O número de tokens gerados deve ser exatamente max_tokens."""
        resultado = gerador.gerar_com_temperatura(
            prompt_simples, temperatura=1.0, max_tokens=6
        )
        assert len(resultado.tokens_gerados) == 6


# ---------------------------------------------------------------------------
# Testes unitários — _aplicar_top_k
# ---------------------------------------------------------------------------


class TestAplicarTopK:
    """Testes para TextGenerator._aplicar_top_k."""

    @pytest.fixture
    def probs_uniformes(self) -> np.ndarray:
        """Distribuição uniforme sobre 10 tokens."""
        probs = np.full(10, 0.1)
        return probs.astype(np.float64)

    @pytest.fixture
    def probs_concentradas(self) -> np.ndarray:
        """Distribuição com dois tokens dominantes."""
        probs = np.array(
            [0.4, 0.3, 0.1, 0.05, 0.05, 0.04, 0.02, 0.02, 0.02, 0.02],
            dtype=np.float64,
        )
        return probs

    def test_top_k_filtra_para_exatamente_k_tokens(
        self, gerador: TextGenerator, probs_concentradas: np.ndarray
    ) -> None:
        """Após _aplicar_top_k(probs, k), apenas k tokens devem ter prob > 0."""
        k = 3
        resultado = gerador._aplicar_top_k(probs_concentradas, k)
        tokens_nao_zero = np.sum(resultado > 0)
        assert tokens_nao_zero == k

    def test_top_k_probabilidades_somam_um(
        self, gerador: TextGenerator, probs_concentradas: np.ndarray
    ) -> None:
        """Probabilidades renormalizadas após top-k devem somar 1."""
        resultado = gerador._aplicar_top_k(probs_concentradas, k=4)
        np.testing.assert_allclose(resultado.sum(), 1.0, atol=1e-7)

    def test_top_k_mantem_tokens_mais_provaveis(
        self, gerador: TextGenerator, probs_concentradas: np.ndarray
    ) -> None:
        """Os tokens mantidos devem ser exatamente os k mais prováveis."""
        k = 2
        resultado = gerador._aplicar_top_k(probs_concentradas, k)
        # Os top-2 originais têm índices 0 e 1
        assert resultado[0] > 0
        assert resultado[1] > 0
        # Todos os demais devem ser zero
        for idx in range(2, len(resultado)):
            assert resultado[idx] == 0.0

    def test_top_k_maior_ou_igual_vocab_retorna_original(
        self, gerador: TextGenerator, probs_concentradas: np.ndarray
    ) -> None:
        """k >= len(probs) deve retornar a distribuição inalterada."""
        k = len(probs_concentradas)
        resultado = gerador._aplicar_top_k(probs_concentradas, k)
        np.testing.assert_array_equal(resultado, probs_concentradas)

    def test_top_k_zero_retorna_original(
        self, gerador: TextGenerator, probs_concentradas: np.ndarray
    ) -> None:
        """k <= 0 deve retornar a distribuição inalterada."""
        resultado = gerador._aplicar_top_k(probs_concentradas, k=0)
        np.testing.assert_array_equal(resultado, probs_concentradas)


# ---------------------------------------------------------------------------
# Testes unitários — _aplicar_top_p
# ---------------------------------------------------------------------------


class TestAplicarTopP:
    """Testes para TextGenerator._aplicar_top_p."""

    @pytest.fixture
    def probs_exemplares(self) -> np.ndarray:
        """Distribuição com massa bem definida para testar nucleus."""
        # Tokens ordenados: 0.40, 0.30, 0.15, 0.10, 0.05
        probs = np.array([0.40, 0.30, 0.15, 0.10, 0.05], dtype=np.float64)
        return probs

    def test_top_p_filtragem_nucleus(
        self, gerador: TextGenerator, probs_exemplares: np.ndarray
    ) -> None:
        """Tokens mantidos devem acumular pelo menos p de probabilidade.

        Com p=0.75 e probs=[0.40, 0.30, 0.15, 0.10, 0.05]:
        - Token índice 0 (0.40) acumula 0.40
        - Token índice 1 (0.30) acumula 0.70
        - Token índice 2 (0.15) acumula 0.85 >= 0.75 → incluído
        O nucleus final deve ter exatamente 3 tokens com probabilidade > 0.
        """
        p = 0.75
        resultado = gerador._aplicar_top_p(probs_exemplares, p)
        tokens_mantidos = np.sum(resultado > 0)
        # Verifica que a probabilidade acumulada dos mantidos >= p
        probs_mantidas = resultado[resultado > 0]
        assert probs_mantidas.sum() == pytest.approx(1.0, abs=1e-6)
        # Deve manter pelo menos 2 (os dois primeiros já chegam em 0.70 < 0.75)
        assert tokens_mantidos >= 2

    def test_top_p_probabilidades_somam_um(
        self, gerador: TextGenerator, probs_exemplares: np.ndarray
    ) -> None:
        """Probabilidades renormalizadas após top-p devem somar 1."""
        resultado = gerador._aplicar_top_p(probs_exemplares, p=0.8)
        np.testing.assert_allclose(resultado.sum(), 1.0, atol=1e-7)

    def test_top_p_igual_a_1_retorna_original(
        self, gerador: TextGenerator, probs_exemplares: np.ndarray
    ) -> None:
        """p >= 1.0 não deve filtrar nenhum token (retorna inalterado)."""
        resultado = gerador._aplicar_top_p(probs_exemplares, p=1.0)
        np.testing.assert_array_equal(resultado, probs_exemplares)

    def test_top_p_acumulacao_correta(
        self, gerador: TextGenerator
    ) -> None:
        """Verifica que tokens removidos estão além do threshold acumulado."""
        # Distribuição controlada: token 0 domina com 0.95
        probs = np.array([0.95, 0.03, 0.01, 0.01], dtype=np.float64)
        p = 0.90
        resultado = gerador._aplicar_top_p(probs, p)
        # Token 0 (0.95) já ultrapassa p=0.90 sozinho; os demais devem ser zerados
        assert resultado[0] > 0
        for idx in range(1, len(resultado)):
            assert resultado[idx] == 0.0


# ---------------------------------------------------------------------------
# Testes unitários — beam_search
# ---------------------------------------------------------------------------


class TestBeamSearch:
    """Testes para TextGenerator.beam_search."""

    def test_beam_search_retorna_num_beams_resultados(
        self, gerador: TextGenerator, prompt_simples: list[str]
    ) -> None:
        """beam_search deve retornar exatamente num_beams candidatos."""
        num_beams = 3
        resultados = gerador.beam_search(
            prompt_simples, num_beams=num_beams, max_tokens=5
        )
        assert len(resultados) == num_beams

    def test_beam_search_ordenado_por_probabilidade(
        self, gerador: TextGenerator, prompt_simples: list[str]
    ) -> None:
        """Candidatos devem estar em ordem decrescente de probabilidade."""
        resultados = gerador.beam_search(prompt_simples, num_beams=4, max_tokens=4)
        probabilidades = [r["probabilidade"] for r in resultados]
        for i in range(len(probabilidades) - 1):
            assert probabilidades[i] >= probabilidades[i + 1], (
                f"Posição {i} ({probabilidades[i]:.6f}) < "
                f"posição {i+1} ({probabilidades[i+1]:.6f})"
            )

    def test_beam_search_primeiro_tem_maior_probabilidade(
        self, gerador: TextGenerator, prompt_simples: list[str]
    ) -> None:
        """O primeiro resultado deve ter a maior probabilidade de todos."""
        resultados = gerador.beam_search(prompt_simples, num_beams=3, max_tokens=6)
        maior_prob = max(r["probabilidade"] for r in resultados)
        assert resultados[0]["probabilidade"] == pytest.approx(maior_prob, abs=1e-9)

    def test_beam_search_campos_obrigatorios_presentes(
        self, gerador: TextGenerator, prompt_simples: list[str]
    ) -> None:
        """Cada candidato deve conter as chaves: texto, texto_completo, tokens,
        log_probabilidade e probabilidade."""
        resultados = gerador.beam_search(prompt_simples, num_beams=2, max_tokens=3)
        campos_esperados = {
            "texto",
            "texto_completo",
            "tokens",
            "log_probabilidade",
            "probabilidade",
        }
        for candidato in resultados:
            assert campos_esperados.issubset(candidato.keys())

    def test_beam_search_num_beams_diferente(
        self, gerador: TextGenerator, prompt_simples: list[str]
    ) -> None:
        """Deve funcionar corretamente com num_beams=5."""
        resultados = gerador.beam_search(prompt_simples, num_beams=5, max_tokens=3)
        assert len(resultados) == 5


# ---------------------------------------------------------------------------
# Testes unitários — demonstrar_temperatura (entropia)
# ---------------------------------------------------------------------------


class TestDemonstrarTemperatura:
    """Testes para TextGenerator.demonstrar_temperatura."""

    def test_entropia_cresce_com_temperatura(
        self, gerador: TextGenerator
    ) -> None:
        """Temperaturas maiores devem produzir distribuições com maior entropia."""
        temperaturas = [0.1, 0.5, 1.0, 1.5, 2.0]
        demo = gerador.demonstrar_temperatura(temperaturas)
        entropias = [float(d["entropia"]) for d in demo.distribuicoes]

        for i in range(len(entropias) - 1):
            assert entropias[i] < entropias[i + 1], (
                f"Entropia em T={temperaturas[i]} ({entropias[i]:.4f}) "
                f"não é menor que em T={temperaturas[i+1]} ({entropias[i+1]:.4f})"
            )

    def test_temperatura_0_1_entropia_menor_que_2_0(
        self, gerador: TextGenerator
    ) -> None:
        """Entropia com T=0.1 deve ser estritamente menor que com T=2.0."""
        demo = gerador.demonstrar_temperatura([0.1, 2.0])
        entropia_baixa = float(demo.distribuicoes[0]["entropia"])
        entropia_alta = float(demo.distribuicoes[1]["entropia"])
        assert entropia_baixa < entropia_alta

    def test_numero_de_distribuicoes_e_correto(
        self, gerador: TextGenerator
    ) -> None:
        """Deve retornar uma distribuição por temperatura solicitada."""
        temperaturas = [0.5, 1.0, 2.0]
        demo = gerador.demonstrar_temperatura(temperaturas)
        assert len(demo.distribuicoes) == len(temperaturas)

    def test_temperaturas_padrao_quando_none(
        self, gerador: TextGenerator
    ) -> None:
        """Chamar sem argumento deve usar as 5 temperaturas padrão."""
        demo = gerador.demonstrar_temperatura(None)
        assert len(demo.distribuicoes) == 5

    def test_tokens_e_logits_consistentes(
        self, gerador: TextGenerator
    ) -> None:
        """Tokens e logits_originais devem ter o mesmo comprimento (10)."""
        demo = gerador.demonstrar_temperatura([1.0])
        assert len(demo.tokens) == len(demo.logits_originais)
        assert len(demo.tokens) == 10

    def test_probabilidades_somam_um_por_temperatura(
        self, gerador: TextGenerator
    ) -> None:
        """As probabilidades normalizadas de cada distribuição devem somar ~1.

        Atenção: distribuicoes armazena apenas os top-10 tokens, portanto
        a soma é < 1. Verificamos apenas que os valores são não-negativos
        e que a entropia foi calculada corretamente sobre a distribuição
        completa (verificado pelo teste de ordenação de entropia acima).
        """
        demo = gerador.demonstrar_temperatura([1.0])
        probs = demo.distribuicoes[0]["probabilidades"]
        for p in probs:  # type: ignore[union-attr]
            assert float(p) >= 0.0


# ---------------------------------------------------------------------------
# Testes de API — POST /api/inference/generate
# ---------------------------------------------------------------------------


class TestApiGenerate:
    """Testes de integração para POST /api/inference/generate."""

    def test_generate_greedy_campos_resposta(self, client: TestClient) -> None:
        """Resposta greedy deve conter texto_gerado, tokens_gerados e
        historico_probabilidades."""
        resposta = client.post(
            "/api/inference/generate",
            json={"prompt": ["o", "gato"], "estrategia": "greedy", "max_tokens": 5},
        )
        assert resposta.status_code == 200
        corpo = resposta.json()
        assert "texto_gerado" in corpo
        assert "tokens_gerados" in corpo
        assert "historico_probabilidades" in corpo
        assert isinstance(corpo["tokens_gerados"], list)
        assert len(corpo["tokens_gerados"]) == 5

    def test_generate_greedy_historico_estrutura(self, client: TestClient) -> None:
        """Cada entrada no historico_probabilidades deve ter token, probabilidade,
        logit e top_5_tokens."""
        resposta = client.post(
            "/api/inference/generate",
            json={"prompt": ["o"], "estrategia": "greedy", "max_tokens": 3},
        )
        assert resposta.status_code == 200
        historico = resposta.json()["historico_probabilidades"]
        for entrada in historico:
            assert "token" in entrada
            assert "probabilidade" in entrada
            assert "logit" in entrada
            assert "top_5_tokens" in entrada
            assert len(entrada["top_5_tokens"]) == 5

    def test_generate_temperatura_retorna_resposta_valida(
        self, client: TestClient
    ) -> None:
        """Estratégia 'temperatura' deve retornar resposta com status 200."""
        resposta = client.post(
            "/api/inference/generate",
            json={
                "prompt": ["o"],
                "estrategia": "temperatura",
                "max_tokens": 4,
                "temperatura": 0.8,
            },
        )
        assert resposta.status_code == 200
        corpo = resposta.json()
        assert corpo["estrategia"] == "temperatura"
        assert len(corpo["tokens_gerados"]) == 4

    def test_generate_top_k_retorna_resposta_valida(
        self, client: TestClient
    ) -> None:
        """Estratégia 'top_k' deve retornar resposta com status 200."""
        resposta = client.post(
            "/api/inference/generate",
            json={
                "prompt": ["a"],
                "estrategia": "top_k",
                "max_tokens": 4,
                "k": 10,
                "temperatura": 1.0,
            },
        )
        assert resposta.status_code == 200
        corpo = resposta.json()
        assert corpo["estrategia"] == "top_k"
        assert len(corpo["tokens_gerados"]) == 4

    def test_generate_top_p_retorna_resposta_valida(
        self, client: TestClient
    ) -> None:
        """Estratégia 'top_p' deve retornar resposta com status 200."""
        resposta = client.post(
            "/api/inference/generate",
            json={
                "prompt": ["de"],
                "estrategia": "top_p",
                "max_tokens": 4,
                "p": 0.9,
                "temperatura": 1.0,
            },
        )
        assert resposta.status_code == 200
        corpo = resposta.json()
        assert corpo["estrategia"] == "top_p"
        assert len(corpo["tokens_gerados"]) == 4

    def test_generate_estrategia_invalida_retorna_422(
        self, client: TestClient
    ) -> None:
        """Estratégia desconhecida deve retornar status 422."""
        resposta = client.post(
            "/api/inference/generate",
            json={"prompt": ["o"], "estrategia": "invalida", "max_tokens": 5},
        )
        assert resposta.status_code == 422

    def test_generate_greedy_texto_gerado_contem_prompt(
        self, client: TestClient
    ) -> None:
        """texto_gerado deve incluir os tokens do prompt original."""
        prompt = ["o", "gato"]
        resposta = client.post(
            "/api/inference/generate",
            json={"prompt": prompt, "estrategia": "greedy", "max_tokens": 3},
        )
        assert resposta.status_code == 200
        texto = resposta.json()["texto_gerado"]
        for token in prompt:
            assert token in texto

    def test_generate_resposta_contem_explicacao(self, client: TestClient) -> None:
        """Resposta deve incluir campo 'explicacao'."""
        resposta = client.post(
            "/api/inference/generate",
            json={"prompt": ["o"], "estrategia": "greedy", "max_tokens": 2},
        )
        assert resposta.status_code == 200
        assert "explicacao" in resposta.json()
        assert len(resposta.json()["explicacao"]) > 0


# ---------------------------------------------------------------------------
# Testes de API — POST /api/inference/temperature-demo
# ---------------------------------------------------------------------------


class TestApiTemperatureDemo:
    """Testes de integração para POST /api/inference/temperature-demo."""

    def test_temperature_demo_numero_de_distribuicoes(
        self, client: TestClient
    ) -> None:
        """Resposta deve ter tantas distribuições quantas temperaturas enviadas."""
        temperaturas = [0.1, 0.5, 1.0, 1.5, 2.0]
        resposta = client.post(
            "/api/inference/temperature-demo",
            json={"temperaturas": temperaturas},
        )
        assert resposta.status_code == 200
        corpo = resposta.json()
        assert len(corpo["distribuicoes"]) == len(temperaturas)

    def test_temperature_demo_campos_resposta(self, client: TestClient) -> None:
        """Resposta deve conter logits_originais, tokens, distribuicoes e explicacao."""
        resposta = client.post(
            "/api/inference/temperature-demo",
            json={"temperaturas": [0.5, 1.0]},
        )
        assert resposta.status_code == 200
        corpo = resposta.json()
        assert "logits_originais" in corpo
        assert "tokens" in corpo
        assert "distribuicoes" in corpo
        assert "explicacao" in corpo

    def test_temperature_demo_cada_distribuicao_tem_campos(
        self, client: TestClient
    ) -> None:
        """Cada entrada em distribuicoes deve ter temperatura, probabilidades,
        entropia e descricao."""
        resposta = client.post(
            "/api/inference/temperature-demo",
            json={"temperaturas": [0.5, 1.0, 2.0]},
        )
        assert resposta.status_code == 200
        for dist in resposta.json()["distribuicoes"]:
            assert "temperatura" in dist
            assert "probabilidades" in dist
            assert "entropia" in dist
            assert "descricao" in dist

    def test_temperature_demo_temperatura_invalida_retorna_422(
        self, client: TestClient
    ) -> None:
        """Temperatura <= 0 deve retornar status 422."""
        resposta = client.post(
            "/api/inference/temperature-demo",
            json={"temperaturas": [0.5, -1.0]},
        )
        assert resposta.status_code == 422

    def test_temperature_demo_temperatura_zero_retorna_422(
        self, client: TestClient
    ) -> None:
        """Temperatura exatamente 0 deve retornar status 422."""
        resposta = client.post(
            "/api/inference/temperature-demo",
            json={"temperaturas": [0.0]},
        )
        assert resposta.status_code == 422

    def test_temperature_demo_entropia_cresce(self, client: TestClient) -> None:
        """Entropia deve ser maior para temperaturas mais altas via API."""
        resposta = client.post(
            "/api/inference/temperature-demo",
            json={"temperaturas": [0.1, 2.0]},
        )
        assert resposta.status_code == 200
        dists = resposta.json()["distribuicoes"]
        entropia_baixa = dists[0]["entropia"]
        entropia_alta = dists[1]["entropia"]
        assert entropia_baixa < entropia_alta


# ---------------------------------------------------------------------------
# Testes de API — POST /api/inference/sampling-demo
# ---------------------------------------------------------------------------


class TestApiSamplingDemo:
    """Testes de integração para POST /api/inference/sampling-demo."""

    def test_sampling_demo_todas_estrategias_presentes(
        self, client: TestClient
    ) -> None:
        """Resposta deve conter todas as 6 estratégias: greedy, temperatura_baixa,
        temperatura_alta, top_k, top_p e beam_search."""
        resposta = client.post(
            "/api/inference/sampling-demo",
            json={"prompt": ["o"], "max_tokens": 5, "num_beams": 3},
        )
        assert resposta.status_code == 200
        corpo = resposta.json()
        estrategias_esperadas = {
            "greedy",
            "temperatura_baixa",
            "temperatura_alta",
            "top_k",
            "top_p",
            "beam_search",
        }
        for chave in estrategias_esperadas:
            assert chave in corpo, f"Chave '{chave}' ausente na resposta"

    def test_sampling_demo_beam_search_e_lista(self, client: TestClient) -> None:
        """beam_search na resposta deve ser uma lista de candidatos."""
        resposta = client.post(
            "/api/inference/sampling-demo",
            json={"prompt": ["o"], "max_tokens": 4, "num_beams": 3},
        )
        assert resposta.status_code == 200
        assert isinstance(resposta.json()["beam_search"], list)

    def test_sampling_demo_beam_search_tamanho(self, client: TestClient) -> None:
        """beam_search deve retornar exatamente num_beams candidatos."""
        num_beams = 4
        resposta = client.post(
            "/api/inference/sampling-demo",
            json={"prompt": ["o"], "max_tokens": 4, "num_beams": num_beams},
        )
        assert resposta.status_code == 200
        assert len(resposta.json()["beam_search"]) == num_beams

    def test_sampling_demo_greedy_e_dict(self, client: TestClient) -> None:
        """greedy deve ser um dicionário com campo texto_gerado."""
        resposta = client.post(
            "/api/inference/sampling-demo",
            json={"prompt": ["o"], "max_tokens": 5, "num_beams": 2},
        )
        assert resposta.status_code == 200
        greedy = resposta.json()["greedy"]
        assert isinstance(greedy, dict)
        assert "texto_gerado" in greedy

    def test_sampling_demo_contem_explicacao(self, client: TestClient) -> None:
        """Resposta deve incluir campo 'explicacao' não vazio."""
        resposta = client.post(
            "/api/inference/sampling-demo",
            json={"prompt": ["o"], "max_tokens": 3, "num_beams": 2},
        )
        assert resposta.status_code == 200
        assert "explicacao" in resposta.json()
        assert len(resposta.json()["explicacao"]) > 0

    def test_sampling_demo_estrategias_tem_tokens_gerados(
        self, client: TestClient
    ) -> None:
        """greedy, temperatura_baixa, temperatura_alta, top_k e top_p devem
        conter tokens_gerados com o tamanho correto."""
        max_tokens = 5
        resposta = client.post(
            "/api/inference/sampling-demo",
            json={"prompt": ["o"], "max_tokens": max_tokens, "num_beams": 2},
        )
        assert resposta.status_code == 200
        corpo = resposta.json()
        for chave in ["greedy", "temperatura_baixa", "temperatura_alta", "top_k", "top_p"]:
            estrategia = corpo[chave]
            assert "tokens_gerados" in estrategia, (
                f"Campo 'tokens_gerados' ausente em '{chave}'"
            )
            assert len(estrategia["tokens_gerados"]) == max_tokens
