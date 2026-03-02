"""Testes para o gerenciador de modelos HuggingFace e o router /api/models.

Cobre:
- Testes de unidade sobre MODELOS_DISPONIVEIS e ModelManager (rápidos, sem download)
- Testes de integração via TestClient para os três endpoints do router models
- Testes lentos (marcados com @pytest.mark.slow) que requerem download do modelo
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ===========================================================================
# Testes de unidade: MODELOS_DISPONIVEIS
# ===========================================================================


class TestModelosDisponiveis:
    """Testes sobre o dicionário estático MODELOS_DISPONIVEIS."""

    def test_tem_exatamente_tres_modelos(self) -> None:
        """MODELOS_DISPONIVEIS deve conter exatamente 3 entradas."""
        from core.model_manager import MODELOS_DISPONIVEIS

        assert len(MODELOS_DISPONIVEIS) == 3

    def test_chaves_corretas(self) -> None:
        """As três chaves esperadas devem estar presentes no dicionário."""
        from core.model_manager import MODELOS_DISPONIVEIS

        chaves_esperadas = {
            "distilbert-base-uncased",
            "bert-base-uncased",
            "gpt2",
        }
        assert set(MODELOS_DISPONIVEIS.keys()) == chaves_esperadas

    def test_metadados_distilbert_corretos(self) -> None:
        """distilbert-base-uncased deve ter 6 camadas, 12 cabeças, d_model=768."""
        from core.model_manager import MODELOS_DISPONIVEIS

        info = MODELOS_DISPONIVEIS["distilbert-base-uncased"]
        assert info["num_camadas"] == 6
        assert info["num_cabecas"] == 12
        assert info["d_model"] == 768
        assert info["tipo"] == "encoder"

    def test_metadados_bert_corretos(self) -> None:
        """bert-base-uncased deve ter 12 camadas, 12 cabeças, d_model=768."""
        from core.model_manager import MODELOS_DISPONIVEIS

        info = MODELOS_DISPONIVEIS["bert-base-uncased"]
        assert info["num_camadas"] == 12
        assert info["num_cabecas"] == 12
        assert info["d_model"] == 768
        assert info["tipo"] == "encoder"

    def test_metadados_gpt2_corretos(self) -> None:
        """gpt2 deve ter 12 camadas, 12 cabeças, d_model=768 e ser do tipo decoder."""
        from core.model_manager import MODELOS_DISPONIVEIS

        info = MODELOS_DISPONIVEIS["gpt2"]
        assert info["num_camadas"] == 12
        assert info["num_cabecas"] == 12
        assert info["d_model"] == 768
        assert info["tipo"] == "decoder"


# ===========================================================================
# Testes de unidade: singleton get_model_manager
# ===========================================================================


class TestSingleton:
    """Testes sobre o padrão singleton de get_model_manager."""

    def test_get_model_manager_retorna_mesma_instancia(self) -> None:
        """Duas chamadas a get_model_manager() devem retornar o mesmo objeto."""
        from core.model_manager import get_model_manager

        primeira = get_model_manager()
        segunda = get_model_manager()
        assert primeira is segunda

    def test_get_model_manager_retorna_model_manager(self) -> None:
        """get_model_manager() deve retornar uma instância de ModelManager."""
        from core.model_manager import ModelManager, get_model_manager

        mgr = get_model_manager()
        assert isinstance(mgr, ModelManager)


# ===========================================================================
# Testes de unidade: listar_modelos_disponiveis
# ===========================================================================


class TestListarModelosDisponiveis:
    """Testes sobre ModelManager.listar_modelos_disponiveis."""

    def test_retorna_todos_os_tres_modelos(self, reset_model_manager) -> None:
        """listar_modelos_disponiveis deve retornar exatamente 3 itens."""
        from core.model_manager import get_model_manager

        mgr = get_model_manager()
        resultado = mgr.listar_modelos_disponiveis()
        assert len(resultado) == 3

    def test_nomes_correspondem_as_chaves_do_dicionario(
        self, reset_model_manager
    ) -> None:
        """Os nomes retornados devem coincidir com as chaves de MODELOS_DISPONIVEIS."""
        from core.model_manager import MODELOS_DISPONIVEIS, get_model_manager

        mgr = get_model_manager()
        nomes_retornados = {info.nome for info in mgr.listar_modelos_disponiveis()}
        assert nomes_retornados == set(MODELOS_DISPONIVEIS.keys())

    def test_nenhum_modelo_carregado_inicialmente(
        self, reset_model_manager
    ) -> None:
        """Com cache vazio, todos os modelos devem ter carregado=False."""
        from core.model_manager import get_model_manager

        mgr = get_model_manager()
        for info in mgr.listar_modelos_disponiveis():
            assert info.carregado is False, (
                f"Modelo '{info.nome}' apareceu como carregado sem ter sido carregado"
            )

    def test_tipo_de_retorno_e_modelo_carregado_info(
        self, reset_model_manager
    ) -> None:
        """Cada item da lista deve ser uma instância de ModeloCarregadoInfo."""
        from core.model_manager import ModeloCarregadoInfo, get_model_manager

        mgr = get_model_manager()
        for item in mgr.listar_modelos_disponiveis():
            assert isinstance(item, ModeloCarregadoInfo)


# ===========================================================================
# Testes de integração: GET /api/models/available-models
# ===========================================================================


class TestEndpointAvailableModels:
    """Testes de integração para GET /api/models/available-models."""

    def test_retorna_200(self, client: TestClient) -> None:
        """Endpoint deve retornar HTTP 200."""
        resp = client.get("/api/models/available-models")
        assert resp.status_code == 200

    def test_total_e_tres(self, client: TestClient) -> None:
        """Campo total deve ser 3."""
        resp = client.get("/api/models/available-models")
        assert resp.status_code == 200
        assert resp.json()["total"] == 3

    def test_todos_os_nomes_presentes(self, client: TestClient) -> None:
        """Os três modelos esperados devem estar na lista retornada."""
        resp = client.get("/api/models/available-models")
        assert resp.status_code == 200

        nomes = {m["nome"] for m in resp.json()["modelos"]}
        assert "distilbert-base-uncased" in nomes
        assert "bert-base-uncased" in nomes
        assert "gpt2" in nomes

    def test_estrutura_de_cada_modelo(self, client: TestClient) -> None:
        """Cada modelo na resposta deve conter os campos obrigatórios."""
        campos_obrigatorios = {
            "nome",
            "descricao",
            "num_camadas",
            "num_cabecas",
            "d_model",
            "tipo",
            "carregado",
        }
        resp = client.get("/api/models/available-models")
        assert resp.status_code == 200

        for modelo in resp.json()["modelos"]:
            assert campos_obrigatorios.issubset(
                modelo.keys()
            ), f"Campos ausentes em '{modelo.get('nome')}': {campos_obrigatorios - modelo.keys()}"

    def test_resposta_contem_nota(self, client: TestClient) -> None:
        """Resposta deve conter o campo 'nota'."""
        resp = client.get("/api/models/available-models")
        assert resp.status_code == 200
        assert "nota" in resp.json()
        assert len(resp.json()["nota"]) > 0


# ===========================================================================
# Testes de integração: POST /api/models/load-model (rápidos)
# ===========================================================================


class TestEndpointLoadModelRapido:
    """Testes de integração rápidos para POST /api/models/load-model."""

    def test_modelo_desconhecido_retorna_404(self, client: TestClient) -> None:
        """Modelo não cadastrado deve retornar HTTP 404."""
        resp = client.post(
            "/api/models/load-model",
            json={"nome_modelo": "modelo-inexistente-xyz"},
        )
        assert resp.status_code == 404

    def test_404_contem_modelos_disponiveis_no_detalhe(
        self, client: TestClient
    ) -> None:
        """O detalhe do 404 deve mencionar os modelos disponíveis."""
        resp = client.post(
            "/api/models/load-model",
            json={"nome_modelo": "nao-existe"},
        )
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        # pelo menos um dos nomes dos modelos deve aparecer no detalhe
        assert any(
            nome in detail
            for nome in (
                "distilbert-base-uncased",
                "bert-base-uncased",
                "gpt2",
            )
        )

    def test_corpo_vazio_usa_default_e_tenta_carregar(
        self, client: TestClient
    ) -> None:
        """Corpo sem nome_modelo deve usar o default 'distilbert-base-uncased'.

        O request não vai retornar 404 nem 422 — apenas vai tentar carregar
        o modelo padrão (podendo falhar com 200 e carregado=False se sem rede,
        mas nunca com 404).
        """
        # Não enviamos nome_modelo — o campo tem default configurado no Pydantic
        resp = client.post("/api/models/load-model", json={})
        # Com modelo padrão cadastrado, nunca deve ser 404 por nome desconhecido
        assert resp.status_code != 404


# ===========================================================================
# Testes de integração: POST /api/models/real-attention (rápidos)
# ===========================================================================


class TestEndpointRealAttentionRapido:
    """Testes de integração rápidos para POST /api/models/real-attention."""

    def test_modelo_nao_carregado_retorna_400(
        self, client: TestClient, reset_model_manager
    ) -> None:
        """Solicitar atenção real de modelo não carregado deve retornar HTTP 400."""
        resp = client.post(
            "/api/models/real-attention",
            json={
                "nome_modelo": "distilbert-base-uncased",
                "texto": "o gato senta no tapete",
            },
        )
        assert resp.status_code == 400

    def test_400_menciona_load_model(
        self, client: TestClient, reset_model_manager
    ) -> None:
        """Detalhe do 400 deve orientar o usuário a usar /load-model primeiro."""
        resp = client.post(
            "/api/models/real-attention",
            json={
                "nome_modelo": "distilbert-base-uncased",
                "texto": "transformers sao poderosos",
            },
        )
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "load-model" in detail or "carregar" in detail.lower()

    def test_texto_vazio_retorna_422(self, client: TestClient) -> None:
        """Texto vazio deve falhar na validação Pydantic e retornar 422."""
        resp = client.post(
            "/api/models/real-attention",
            json={"nome_modelo": "distilbert-base-uncased", "texto": ""},
        )
        assert resp.status_code == 422

    def test_texto_muito_longo_retorna_422(self, client: TestClient) -> None:
        """Texto com mais de 512 caracteres deve retornar 422."""
        texto_longo = "a" * 513
        resp = client.post(
            "/api/models/real-attention",
            json={"nome_modelo": "distilbert-base-uncased", "texto": texto_longo},
        )
        assert resp.status_code == 422


# ===========================================================================
# Testes lentos: requerem download de ~250 MB do HuggingFace Hub
# ===========================================================================


@pytest.mark.slow
class TestLoadModelDistilbert:
    """Testes de integração lentos: carregamento real do distilbert-base-uncased."""

    def test_carregar_distilbert_retorna_200(
        self, client: TestClient, reset_model_manager
    ) -> None:
        """POST /load-model com distilbert deve retornar HTTP 200."""
        resp = client.post(
            "/api/models/load-model",
            json={"nome_modelo": "distilbert-base-uncased"},
            timeout=300,
        )
        assert resp.status_code == 200

    def test_carregar_distilbert_carregado_true(
        self, client: TestClient, reset_model_manager
    ) -> None:
        """Após carregar com sucesso, carregado deve ser True."""
        resp = client.post(
            "/api/models/load-model",
            json={"nome_modelo": "distilbert-base-uncased"},
            timeout=300,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["carregado"] is True

    def test_carregar_distilbert_metadados_corretos(
        self, client: TestClient, reset_model_manager
    ) -> None:
        """Resposta de load-model deve conter metadados corretos para distilbert."""
        resp = client.post(
            "/api/models/load-model",
            json={"nome_modelo": "distilbert-base-uncased"},
            timeout=300,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_camadas"] == 6
        assert data["num_cabecas"] == 12
        assert data["d_model"] == 768
        assert data["nome"] == "distilbert-base-uncased"

    def test_carregar_distilbert_sem_erro(
        self, client: TestClient, reset_model_manager
    ) -> None:
        """Campo erro deve ser None após carregamento bem-sucedido."""
        resp = client.post(
            "/api/models/load-model",
            json={"nome_modelo": "distilbert-base-uncased"},
            timeout=300,
        )
        assert resp.status_code == 200
        assert resp.json()["erro"] is None


@pytest.mark.slow
class TestRealAttentionDistilbert:
    """Testes de integração lentos: extração de atenção real do distilbert."""

    @pytest.fixture(autouse=True)
    def _carregar_modelo(self, client: TestClient, reset_model_manager) -> None:
        """Carrega o distilbert antes de cada teste desta classe."""
        resp = client.post(
            "/api/models/load-model",
            json={"nome_modelo": "distilbert-base-uncased"},
            timeout=300,
        )
        assert resp.status_code == 200
        assert resp.json()["carregado"] is True

    def test_real_attention_retorna_200(self, client: TestClient) -> None:
        """POST /real-attention com modelo carregado deve retornar HTTP 200."""
        resp = client.post(
            "/api/models/real-attention",
            json={
                "nome_modelo": "distilbert-base-uncased",
                "texto": "o gato senta no tapete",
            },
        )
        assert resp.status_code == 200

    def test_tokens_incluem_cls_e_sep(self, client: TestClient) -> None:
        """distilbert acrescenta [CLS] e [SEP] — ambos devem estar nos tokens."""
        resp = client.post(
            "/api/models/real-attention",
            json={
                "nome_modelo": "distilbert-base-uncased",
                "texto": "o gato senta no tapete",
            },
        )
        assert resp.status_code == 200
        tokens = resp.json()["tokens"]
        assert "[CLS]" in tokens, "Token [CLS] ausente na resposta"
        assert "[SEP]" in tokens, "Token [SEP] ausente na resposta"

    def test_num_camadas_retornadas(self, client: TestClient) -> None:
        """distilbert tem 6 camadas — sem filtro, todas devem ser retornadas."""
        resp = client.post(
            "/api/models/real-attention",
            json={
                "nome_modelo": "distilbert-base-uncased",
                "texto": "transformers aprendem representacoes",
            },
        )
        assert resp.status_code == 200
        camadas = resp.json()["camadas"]
        assert len(camadas) == 6

    def test_pesos_de_atencao_somam_aproximadamente_um(
        self, client: TestClient
    ) -> None:
        """Cada linha dos pesos de atenção deve somar aproximadamente 1.0 (softmax)."""
        resp = client.post(
            "/api/models/real-attention",
            json={
                "nome_modelo": "distilbert-base-uncased",
                "texto": "o modelo aprende",
            },
        )
        assert resp.status_code == 200

        camadas = resp.json()["camadas"]
        for camada in camadas:
            for cabeca in camada["cabecas"]:
                for linha in cabeca["pesos"]:
                    soma = sum(linha)
                    assert abs(soma - 1.0) < 1e-3, (
                        f"Linha de atenção soma {soma:.6f} (esperado ~1.0) "
                        f"na camada {camada['camada']}, cabeça {cabeca['cabeca']}"
                    )

    def test_filtro_de_camada_especifica(self, client: TestClient) -> None:
        """Ao especificar camada=1, apenas a camada 1 deve ser retornada."""
        resp = client.post(
            "/api/models/real-attention",
            json={
                "nome_modelo": "distilbert-base-uncased",
                "texto": "aprendizado de maquina",
                "camada": 1,
            },
        )
        assert resp.status_code == 200
        camadas = resp.json()["camadas"]
        assert len(camadas) == 1
        assert camadas[0]["camada"] == 1

    def test_camada_inexistente_retorna_404(self, client: TestClient) -> None:
        """Camada 99 não existe no distilbert (6 camadas) — deve retornar 404."""
        resp = client.post(
            "/api/models/real-attention",
            json={
                "nome_modelo": "distilbert-base-uncased",
                "texto": "teste",
                "camada": 99,
            },
        )
        assert resp.status_code == 404

    def test_estrutura_da_resposta(self, client: TestClient) -> None:
        """Resposta deve conter todos os campos obrigatórios do PesosAtencaoResponse."""
        resp = client.post(
            "/api/models/real-attention",
            json={
                "nome_modelo": "distilbert-base-uncased",
                "texto": "redes neurais",
            },
        )
        assert resp.status_code == 200

        data = resp.json()
        campos_obrigatorios = {
            "modelo",
            "texto",
            "tokens",
            "num_tokens",
            "camadas",
            "explicacao",
        }
        assert campos_obrigatorios.issubset(data.keys()), (
            f"Campos ausentes: {campos_obrigatorios - data.keys()}"
        )

    def test_num_tokens_consistente_com_lista_de_tokens(
        self, client: TestClient
    ) -> None:
        """num_tokens deve coincidir com o comprimento da lista tokens."""
        resp = client.post(
            "/api/models/real-attention",
            json={
                "nome_modelo": "distilbert-base-uncased",
                "texto": "o gato e o cachorro",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_tokens"] == len(data["tokens"])


@pytest.mark.slow
class TestModelManagerCache:
    """Testes de unidade lentos: comportamento de cache do ModelManager."""

    def test_segundo_carregamento_usa_cache(
        self, client: TestClient, reset_model_manager
    ) -> None:
        """Após o primeiro carregamento, o segundo deve vir do cache (imediato)."""
        import time

        # Primeiro carregamento — pode demorar (download)
        resp1 = client.post(
            "/api/models/load-model",
            json={"nome_modelo": "distilbert-base-uncased"},
            timeout=300,
        )
        assert resp1.status_code == 200
        assert resp1.json()["carregado"] is True

        # Segundo carregamento — deve usar cache e ser significativamente mais rápido
        inicio = time.monotonic()
        resp2 = client.post(
            "/api/models/load-model",
            json={"nome_modelo": "distilbert-base-uncased"},
            timeout=30,
        )
        duracao = time.monotonic() - inicio

        assert resp2.status_code == 200
        assert resp2.json()["carregado"] is True
        # O cache deve responder em menos de 5 segundos
        assert duracao < 5.0, (
            f"Segunda chamada demorou {duracao:.2f}s - esperado < 5s (cache hit)"
        )

    def test_modelo_aparece_como_carregado_em_available_models(
        self, client: TestClient, reset_model_manager
    ) -> None:
        """Após carregar distilbert, GET /available-models deve mostrar carregado=True."""
        # Garante que o modelo está carregado
        resp_load = client.post(
            "/api/models/load-model",
            json={"nome_modelo": "distilbert-base-uncased"},
            timeout=300,
        )
        assert resp_load.status_code == 200
        assert resp_load.json()["carregado"] is True

        # Verifica reflexo no endpoint de listagem
        resp_list = client.get("/api/models/available-models")
        assert resp_list.status_code == 200

        modelos = {m["nome"]: m for m in resp_list.json()["modelos"]}
        assert modelos["distilbert-base-uncased"]["carregado"] is True
        # os outros não foram carregados
        assert modelos["bert-base-uncased"]["carregado"] is False
        assert modelos["gpt2"]["carregado"] is False
