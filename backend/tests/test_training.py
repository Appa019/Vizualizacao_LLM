"""Testes para o módulo de treinamento: core/mini_trainer.py e routers/training.py."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.mini_trainer import (
    MiniTransformerNet,
    computar_superficie_loss,
    gerar_dado_treino,
    gerar_dados_objetivos_treino,
    simular_gradient_descent,
)


# ---------------------------------------------------------------------------
# Testes unitários: MiniTransformerNet
# ---------------------------------------------------------------------------


class TestMiniTransformerNetForward:
    """Verifica que o forward pass produz distribuições de probabilidade válidas."""

    def test_saida_soma_para_um(self) -> None:
        """A saída do forward pass deve somar ~1 (distribuição de probabilidade)."""
        rede = MiniTransformerNet()
        entrada = np.zeros(20)
        entrada[5] = 1.0
        probs = rede.predict(entrada)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_saida_nao_tem_valores_negativos(self) -> None:
        """Todos os valores de saída devem ser >= 0."""
        rede = MiniTransformerNet()
        entrada = np.zeros(20)
        entrada[3] = 1.0
        probs = rede.predict(entrada)
        assert np.all(probs >= 0.0)

    def test_saida_tem_dimensao_correta(self) -> None:
        """A saída deve ter o mesmo tamanho que o vocabulário configurado."""
        tamanho_vocab = 15
        rede = MiniTransformerNet(
            tamanho_entrada=tamanho_vocab,
            tamanho_saida=tamanho_vocab,
        )
        entrada = np.zeros(tamanho_vocab)
        entrada[0] = 1.0
        probs = rede.predict(entrada)
        assert probs.shape == (tamanho_vocab,)


class TestMiniTransformerNetTreinamento:
    """Verifica o comportamento do train_step e convergência."""

    def test_loss_decresce_ao_longo_do_treino(self) -> None:
        """O loss deve diminuir após múltiplos passos de treino com a mesma amostra."""
        rede = MiniTransformerNet(taxa_aprendizado=0.05)
        # Treinar sempre no mesmo exemplo para forçar memorização
        entrada = np.zeros(20)
        entrada[7] = 1.0
        alvo = 8  # (7 + 1) % 20

        resultado_inicial = rede.train_step(entrada, alvo)
        loss_inicial = resultado_inicial.loss

        for _ in range(49):
            resultado_final = rede.train_step(entrada, alvo)

        assert resultado_final.loss < loss_inicial, (
            f"Loss final ({resultado_final.loss:.4f}) deveria ser menor que "
            f"o loss inicial ({loss_inicial:.4f})"
        )

    def test_gradientes_sao_nao_nulos_apos_passo(self) -> None:
        """Todas as normas de gradiente devem ser > 0 após um passo de treino."""
        rede = MiniTransformerNet()
        entrada = np.zeros(20)
        entrada[2] = 1.0
        resultado = rede.train_step(entrada, 3)

        for nome, norma in resultado.gradientes_norma.items():
            assert norma > 0.0, f"Gradiente de '{nome}' não deveria ser zero"

    def test_train_step_retorna_campos_corretos(self) -> None:
        """TrainStepResult deve conter todos os campos esperados com tipos corretos."""
        rede = MiniTransformerNet()
        entrada = np.zeros(20)
        entrada[0] = 1.0
        resultado = rede.train_step(entrada, 1)

        assert isinstance(resultado.passo, int)
        assert resultado.passo == 1
        assert isinstance(resultado.loss, float)
        assert resultado.loss > 0.0
        assert isinstance(resultado.acuracia, float)
        assert resultado.acuracia in {0.0, 1.0}
        assert isinstance(resultado.gradientes_norma, dict)
        assert set(resultado.gradientes_norma.keys()) == {"W1", "W2", "b1", "b2"}
        assert resultado.alvo == 1

    def test_passo_incrementa_a_cada_chamada(self) -> None:
        """O contador de passos deve incrementar a cada train_step."""
        rede = MiniTransformerNet()
        entrada = np.zeros(20)
        entrada[10] = 1.0

        for esperado in range(1, 6):
            resultado = rede.train_step(entrada, (10 + 1) % 20)
            assert resultado.passo == esperado

    def test_historico_loss_acumula(self) -> None:
        """O histórico de loss deve crescer a cada passo."""
        rede = MiniTransformerNet()
        entrada = np.zeros(20)
        entrada[4] = 1.0

        assert len(rede.historico_loss) == 0
        for n in range(1, 6):
            rede.train_step(entrada, 5)
            assert len(rede.historico_loss) == n


class TestPredicao:
    """Verifica que predict retorna distribuições de probabilidade válidas."""

    def test_predicao_soma_para_um(self) -> None:
        """predict deve retornar uma distribuição de probabilidade válida."""
        rede = MiniTransformerNet()
        entrada = np.zeros(20)
        entrada[9] = 1.0
        probs = rede.predict(entrada)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_predicao_todos_nao_negativos(self) -> None:
        """Todos os valores de predict devem ser >= 0."""
        rede = MiniTransformerNet()
        entrada = np.zeros(20)
        entrada[15] = 1.0
        probs = rede.predict(entrada)
        assert np.all(probs >= 0.0)


# ---------------------------------------------------------------------------
# Testes unitários: gerar_dado_treino
# ---------------------------------------------------------------------------


class TestGerarDadoTreino:
    """Verifica a forma e validade dos dados de treino gerados."""

    def test_entrada_tem_shape_correto(self) -> None:
        """O vetor de entrada deve ter o mesmo tamanho que o vocabulário."""
        tamanho_vocab = 20
        entrada, _ = gerar_dado_treino(tamanho_vocab=tamanho_vocab)
        assert entrada.shape == (tamanho_vocab,)

    def test_entrada_e_one_hot(self) -> None:
        """O vetor de entrada deve ser one-hot: exatamente um 1 e o resto 0."""
        entrada, _ = gerar_dado_treino(tamanho_vocab=20)
        assert entrada.sum() == 1.0
        assert np.all((entrada == 0.0) | (entrada == 1.0))

    def test_alvo_esta_dentro_do_range(self) -> None:
        """O índice alvo deve estar dentro do vocabulário."""
        tamanho_vocab = 20
        _, alvo = gerar_dado_treino(tamanho_vocab=tamanho_vocab)
        assert 0 <= alvo < tamanho_vocab

    def test_alvo_e_proximo_token_circular(self) -> None:
        """O alvo deve ser (idx_entrada + 1) % vocab — verificamos via soma."""
        # Como o índice de entrada é aleatório, executamos várias amostras
        # e checamos que alvo == (indice_entrada + 1) % vocab em todos os casos
        tamanho_vocab = 20
        for _ in range(30):
            entrada, alvo = gerar_dado_treino(tamanho_vocab=tamanho_vocab)
            idx_entrada = int(np.argmax(entrada))
            assert alvo == (idx_entrada + 1) % tamanho_vocab


# ---------------------------------------------------------------------------
# Testes unitários: computar_superficie_loss
# ---------------------------------------------------------------------------


class TestComputarSuperficieLoss:
    """Verifica propriedades da superfície de loss gerada."""

    def test_grade_tem_dimensoes_corretas(self) -> None:
        """loss_grid deve ter shape resolucao x resolucao."""
        resolucao = 15
        resultado = computar_superficie_loss(resolucao=resolucao)
        grid = np.array(resultado.loss_grid)
        assert grid.shape == (resolucao, resolucao)

    def test_w_valores_tem_tamanho_correto(self) -> None:
        """w1_valores e w2_valores devem ter comprimento igual à resolução."""
        resolucao = 20
        resultado = computar_superficie_loss(resolucao=resolucao)
        assert len(resultado.w1_valores) == resolucao
        assert len(resultado.w2_valores) == resolucao

    def test_todos_valores_de_loss_sao_positivos(self) -> None:
        """Todos os valores na grade de loss devem ser > 0."""
        resultado = computar_superficie_loss(resolucao=20)
        grid = np.array(resultado.loss_grid)
        assert np.all(grid > 0.0)

    def test_ponto_otimo_e_o_minimo_da_grade(self) -> None:
        """O ponto_otimo.loss deve corresponder ao valor mínimo da grade."""
        resultado = computar_superficie_loss(resolucao=25)
        grid = np.array(resultado.loss_grid)
        minimo_grade = float(grid.min())
        assert abs(resultado.ponto_otimo["loss"] - minimo_grade) < 1e-9

    def test_ponto_otimo_tem_campos_obrigatorios(self) -> None:
        """ponto_otimo deve conter as chaves 'w1', 'w2' e 'loss'."""
        resultado = computar_superficie_loss()
        assert {"w1", "w2", "loss"} == set(resultado.ponto_otimo.keys())


# ---------------------------------------------------------------------------
# Testes unitários: simular_gradient_descent
# ---------------------------------------------------------------------------


class TestSimularGradientDescent:
    """Verifica convergência e trajetória do gradient descent simulado."""

    def test_convergencia_com_taxa_baixa(self) -> None:
        """Com lr=0.1 e 50 iterações, loss_final deve ser menor que loss_inicial."""
        resultado = simular_gradient_descent(
            taxa_aprendizado=0.1,
            num_iteracoes=50,
            w1_inicial=2.5,
            w2_inicial=2.5,
        )
        assert resultado.loss_final < resultado.loss_inicial

    def test_numero_de_passos_nao_excede_num_iteracoes(self) -> None:
        """O número de passos registrados não deve exceder num_iteracoes."""
        num_iteracoes = 30
        resultado = simular_gradient_descent(
            taxa_aprendizado=0.1,
            num_iteracoes=num_iteracoes,
        )
        assert len(resultado.passos) <= num_iteracoes

    def test_taxa_aprendizado_registrada_corretamente(self) -> None:
        """A taxa de aprendizado deve ser preservada no resultado."""
        lr = 0.05
        resultado = simular_gradient_descent(taxa_aprendizado=lr)
        assert resultado.taxa_aprendizado == lr

    def test_passos_tem_campos_corretos(self) -> None:
        """Cada passo deve conter todos os campos de GradientDescentStep."""
        resultado = simular_gradient_descent(num_iteracoes=5)
        for passo in resultado.passos:
            assert hasattr(passo, "iteracao")
            assert hasattr(passo, "w1")
            assert hasattr(passo, "w2")
            assert hasattr(passo, "loss")
            assert hasattr(passo, "gradiente_w1")
            assert hasattr(passo, "gradiente_w2")

    def test_taxa_alta_nao_converge_ou_tem_loss_instavel(self) -> None:
        """Com lr=1.5 (alto), o algoritmo deve divergir ou não convergir."""
        resultado = simular_gradient_descent(
            taxa_aprendizado=1.5,
            num_iteracoes=50,
            w1_inicial=2.5,
            w2_inicial=2.5,
        )
        # Com lr muito alto, esperamos que NÃO marque como convergido
        # (o gradiente final provavelmente será grande, não < 1e-4)
        assert not resultado.convergiu

    def test_loss_inicial_e_final_correspondem_a_passos(self) -> None:
        """loss_inicial e loss_final devem corresponder ao primeiro e último passo."""
        resultado = simular_gradient_descent(num_iteracoes=20)
        assert resultado.loss_inicial == resultado.passos[0].loss
        assert resultado.loss_final == resultado.passos[-1].loss


# ---------------------------------------------------------------------------
# Testes unitários: gerar_dados_objetivos_treino
# ---------------------------------------------------------------------------


class TestGerarDadosObjetivosTreino:
    """Verifica a estrutura dos dados de MLM e CLM."""

    def setup_method(self) -> None:
        """Gera os dados uma vez para todos os testes desta classe."""
        self.dados = gerar_dados_objetivos_treino()

    def test_possui_campos_obrigatorios(self) -> None:
        """O dicionário deve conter frase_original, tokens, mlm e clm."""
        assert "frase_original" in self.dados
        assert "tokens" in self.dados
        assert "mlm" in self.dados
        assert "clm" in self.dados

    def test_mlm_possui_token_mask(self) -> None:
        """Os tokens MLM mascarados devem conter ao menos um [MASK]."""
        tokens_mlm = self.dados["mlm"]["tokens_com_mascara"]
        assert "[MASK]" in tokens_mlm

    def test_mlm_indices_mascarados_nao_vazio(self) -> None:
        """A lista de índices mascarados deve ter ao menos um elemento."""
        indices = self.dados["mlm"]["indices_mascarados"]
        assert len(indices) >= 1

    def test_clm_possui_pares_entrada_alvo(self) -> None:
        """O CLM deve conter pares de entrada/alvo para cada posição."""
        pares = self.dados["clm"]["pares_entrada_alvo"]
        assert len(pares) >= 1

    def test_clm_pares_tem_formato_correto(self) -> None:
        """Cada par CLM deve ter as chaves 'entrada' e 'alvo'."""
        for par in self.dados["clm"]["pares_entrada_alvo"]:
            assert "entrada" in par
            assert "alvo" in par

    def test_tokens_originais_sao_palavras_da_frase(self) -> None:
        """Os tokens mascarados originais devem pertencer à frase original."""
        frase = str(self.dados["frase_original"])
        palavras = frase.split()
        for token in self.dados["mlm"]["tokens_originais"]:
            assert token in palavras


# ---------------------------------------------------------------------------
# Testes de API: POST /api/training/train-step
# ---------------------------------------------------------------------------


class TestApiTrainStep:
    """Testes de integração para o endpoint POST /api/training/train-step."""

    def test_retorna_200_e_campos_esperados(
        self, client: TestClient, fresh_training_state: None
    ) -> None:
        """Deve retornar 200 com loss, acuracia, gradientes_norma e historico_loss."""
        resposta = client.post(
            "/api/training/train-step",
            json={"taxa_aprendizado": 0.01, "tamanho_vocab": 20, "resetar": False},
        )
        assert resposta.status_code == 200
        dados = resposta.json()
        assert "loss" in dados
        assert "acuracia" in dados
        assert "gradientes_norma" in dados
        assert "historico_loss" in dados
        assert "passo" in dados

    def test_loss_e_um_float_positivo(
        self, client: TestClient, fresh_training_state: None
    ) -> None:
        """O campo loss deve ser um número de ponto flutuante positivo."""
        resposta = client.post(
            "/api/training/train-step",
            json={"taxa_aprendizado": 0.01, "tamanho_vocab": 20, "resetar": False},
        )
        dados = resposta.json()
        assert isinstance(dados["loss"], float)
        assert dados["loss"] > 0.0

    def test_acuracia_e_zero_ou_um(
        self, client: TestClient, fresh_training_state: None
    ) -> None:
        """A acurácia deve ser 0.0 ou 1.0 (classificação binária por passo)."""
        resposta = client.post(
            "/api/training/train-step",
            json={"taxa_aprendizado": 0.01, "tamanho_vocab": 20, "resetar": False},
        )
        dados = resposta.json()
        assert dados["acuracia"] in {0.0, 1.0}

    def test_gradientes_norma_tem_quatro_chaves(
        self, client: TestClient, fresh_training_state: None
    ) -> None:
        """gradientes_norma deve ter W1, W2, b1 e b2."""
        resposta = client.post(
            "/api/training/train-step",
            json={"taxa_aprendizado": 0.01, "tamanho_vocab": 20, "resetar": False},
        )
        dados = resposta.json()
        assert set(dados["gradientes_norma"].keys()) == {"W1", "W2", "b1", "b2"}

    def test_historico_cresce_a_cada_chamada(
        self, client: TestClient, fresh_training_state: None
    ) -> None:
        """historico_loss deve ter um item a mais a cada passo de treino."""
        for esperado in range(1, 4):
            resposta = client.post(
                "/api/training/train-step",
                json={"taxa_aprendizado": 0.01, "tamanho_vocab": 20, "resetar": False},
            )
            dados = resposta.json()
            assert len(dados["historico_loss"]) == esperado

    def test_resetar_true_reinicia_do_passo_1(
        self, client: TestClient, fresh_training_state: None
    ) -> None:
        """Com resetar=True a rede deve ser recriada e o passo deve voltar a 1."""
        # Avançar alguns passos
        for _ in range(3):
            client.post(
                "/api/training/train-step",
                json={"taxa_aprendizado": 0.01, "tamanho_vocab": 20, "resetar": False},
            )
        # Agora resetar
        resposta = client.post(
            "/api/training/train-step",
            json={"taxa_aprendizado": 0.01, "tamanho_vocab": 20, "resetar": True},
        )
        assert resposta.status_code == 200
        dados = resposta.json()
        assert dados["passo"] == 1
        assert len(dados["historico_loss"]) == 1

    def test_resetar_true_limpa_historico(
        self, client: TestClient, fresh_training_state: None
    ) -> None:
        """Após resetar=True o histórico de loss deve ter apenas 1 entrada."""
        # Acumular histórico
        for _ in range(5):
            client.post(
                "/api/training/train-step",
                json={"taxa_aprendizado": 0.01, "tamanho_vocab": 20, "resetar": False},
            )
        # Resetar
        resposta = client.post(
            "/api/training/train-step",
            json={"taxa_aprendizado": 0.01, "tamanho_vocab": 20, "resetar": True},
        )
        dados = resposta.json()
        assert len(dados["historico_loss"]) == 1

    def test_predicao_top5_tem_cinco_itens(
        self, client: TestClient, fresh_training_state: None
    ) -> None:
        """predicao_top5 deve conter exatamente 5 candidatos."""
        resposta = client.post(
            "/api/training/train-step",
            json={"taxa_aprendizado": 0.01, "tamanho_vocab": 20, "resetar": False},
        )
        dados = resposta.json()
        assert len(dados["predicao_top5"]) == 5

    def test_taxa_aprendizado_invalida_retorna_422(
        self, client: TestClient, fresh_training_state: None
    ) -> None:
        """Uma taxa de aprendizado <= 0 deve retornar 422 (validação Pydantic)."""
        resposta = client.post(
            "/api/training/train-step",
            json={"taxa_aprendizado": 0.0, "tamanho_vocab": 20, "resetar": False},
        )
        assert resposta.status_code == 422


# ---------------------------------------------------------------------------
# Testes de API: POST /api/training/loss-surface
# ---------------------------------------------------------------------------


class TestApiLossSurface:
    """Testes de integração para o endpoint POST /api/training/loss-surface."""

    def test_retorna_200_e_campos_esperados(self, client: TestClient) -> None:
        """Deve retornar 200 com w1_valores, w2_valores e loss_grid."""
        resposta = client.post(
            "/api/training/loss-surface",
            json={"resolucao": 15},
        )
        assert resposta.status_code == 200
        dados = resposta.json()
        assert "w1_valores" in dados
        assert "w2_valores" in dados
        assert "loss_grid" in dados
        assert "ponto_otimo" in dados

    def test_dimensoes_da_grade_batem_com_resolucao(self, client: TestClient) -> None:
        """loss_grid deve ter shape resolucao x resolucao."""
        resolucao = 12
        resposta = client.post(
            "/api/training/loss-surface",
            json={"resolucao": resolucao},
        )
        dados = resposta.json()
        grid = dados["loss_grid"]
        assert len(grid) == resolucao
        assert all(len(linha) == resolucao for linha in grid)

    def test_w_valores_tem_comprimento_correto(self, client: TestClient) -> None:
        """w1_valores e w2_valores devem ter o mesmo comprimento que a resolução."""
        resolucao = 18
        resposta = client.post(
            "/api/training/loss-surface",
            json={"resolucao": resolucao},
        )
        dados = resposta.json()
        assert len(dados["w1_valores"]) == resolucao
        assert len(dados["w2_valores"]) == resolucao

    def test_todos_valores_de_loss_sao_positivos(self, client: TestClient) -> None:
        """Todos os valores na grade devem ser > 0 (garantia da implementação)."""
        resposta = client.post(
            "/api/training/loss-surface",
            json={"resolucao": 10},
        )
        dados = resposta.json()
        for linha in dados["loss_grid"]:
            for valor in linha:
                assert valor > 0.0

    def test_resolucao_fora_do_range_retorna_422(self, client: TestClient) -> None:
        """Resolução < 10 deve retornar 422 (Field constraint)."""
        resposta = client.post(
            "/api/training/loss-surface",
            json={"resolucao": 5},
        )
        assert resposta.status_code == 422


# ---------------------------------------------------------------------------
# Testes de API: POST /api/training/gradient-descent-demo
# ---------------------------------------------------------------------------


class TestApiGradientDescentDemo:
    """Testes de integração para POST /api/training/gradient-descent-demo."""

    def test_retorna_200_e_campos_esperados(self, client: TestClient) -> None:
        """Deve retornar 200 com passos, convergiu e info de loss."""
        resposta = client.post(
            "/api/training/gradient-descent-demo",
            json={
                "taxa_aprendizado": 0.1,
                "num_iteracoes": 50,
                "w1_inicial": 2.5,
                "w2_inicial": 2.5,
            },
        )
        assert resposta.status_code == 200
        dados = resposta.json()
        assert "passos" in dados
        assert "convergiu" in dados
        assert "loss_inicial" in dados
        assert "loss_final" in dados
        assert "taxa_aprendizado" in dados
        assert "reducao_percentual" in dados

    def test_passos_tem_campos_corretos(self, client: TestClient) -> None:
        """Cada passo deve conter w1, w2, loss, gradiente_w1 e gradiente_w2."""
        resposta = client.post(
            "/api/training/gradient-descent-demo",
            json={"taxa_aprendizado": 0.1, "num_iteracoes": 10},
        )
        dados = resposta.json()
        for passo in dados["passos"]:
            assert "iteracao" in passo
            assert "w1" in passo
            assert "w2" in passo
            assert "loss" in passo
            assert "gradiente_w1" in passo
            assert "gradiente_w2" in passo

    def test_loss_final_menor_que_inicial_com_lr_baixo(
        self, client: TestClient
    ) -> None:
        """Com lr=0.1, o loss final deve ser menor que o loss inicial."""
        resposta = client.post(
            "/api/training/gradient-descent-demo",
            json={
                "taxa_aprendizado": 0.1,
                "num_iteracoes": 50,
                "w1_inicial": 2.5,
                "w2_inicial": 2.5,
            },
        )
        dados = resposta.json()
        assert dados["loss_final"] < dados["loss_inicial"]

    def test_reducao_percentual_e_positiva(self, client: TestClient) -> None:
        """A redução percentual deve ser positiva quando o loss diminui."""
        resposta = client.post(
            "/api/training/gradient-descent-demo",
            json={
                "taxa_aprendizado": 0.1,
                "num_iteracoes": 50,
                "w1_inicial": 2.5,
                "w2_inicial": 2.5,
            },
        )
        dados = resposta.json()
        assert dados["reducao_percentual"] > 0.0

    def test_taxa_aprendizado_alta_nao_converge(self, client: TestClient) -> None:
        """Com lr=1.5 (alto), convergiu deve ser False."""
        resposta = client.post(
            "/api/training/gradient-descent-demo",
            json={
                "taxa_aprendizado": 1.5,
                "num_iteracoes": 50,
                "w1_inicial": 2.5,
                "w2_inicial": 2.5,
            },
        )
        assert resposta.status_code == 200
        dados = resposta.json()
        assert dados["convergiu"] is False

    def test_taxa_aprendizado_invalida_retorna_422(self, client: TestClient) -> None:
        """Uma taxa de aprendizado <= 0 deve retornar 422."""
        resposta = client.post(
            "/api/training/gradient-descent-demo",
            json={"taxa_aprendizado": -0.1, "num_iteracoes": 20},
        )
        assert resposta.status_code == 422


# ---------------------------------------------------------------------------
# Testes de API: GET /api/training/training-objectives
# ---------------------------------------------------------------------------


class TestApiTrainingObjectives:
    """Testes de integração para GET /api/training/training-objectives."""

    def test_retorna_200_e_campos_esperados(self, client: TestClient) -> None:
        """Deve retornar 200 com frase_original, tokens, mlm e clm."""
        resposta = client.get("/api/training/training-objectives")
        assert resposta.status_code == 200
        dados = resposta.json()
        assert "frase_original" in dados
        assert "tokens" in dados
        assert "mlm" in dados
        assert "clm" in dados

    def test_frase_original_e_string_nao_vazia(self, client: TestClient) -> None:
        """frase_original deve ser uma string não vazia."""
        resposta = client.get("/api/training/training-objectives")
        dados = resposta.json()
        assert isinstance(dados["frase_original"], str)
        assert len(dados["frase_original"]) > 0

    def test_tokens_e_lista_nao_vazia(self, client: TestClient) -> None:
        """tokens deve ser uma lista com ao menos um elemento."""
        resposta = client.get("/api/training/training-objectives")
        dados = resposta.json()
        assert isinstance(dados["tokens"], list)
        assert len(dados["tokens"]) > 0

    def test_mlm_possui_tokens_com_mask(self, client: TestClient) -> None:
        """mlm.tokens_com_mascara deve conter ao menos um [MASK]."""
        resposta = client.get("/api/training/training-objectives")
        dados = resposta.json()
        tokens_mascarados = dados["mlm"]["tokens_com_mascara"]
        assert "[MASK]" in tokens_mascarados

    def test_clm_possui_pares_entrada_alvo(self, client: TestClient) -> None:
        """clm.pares_entrada_alvo deve ser uma lista com ao menos um par."""
        resposta = client.get("/api/training/training-objectives")
        dados = resposta.json()
        pares = dados["clm"]["pares_entrada_alvo"]
        assert isinstance(pares, list)
        assert len(pares) >= 1

    def test_clm_pares_tem_chaves_corretas(self, client: TestClient) -> None:
        """Cada par CLM deve ter as chaves 'entrada' e 'alvo'."""
        resposta = client.get("/api/training/training-objectives")
        dados = resposta.json()
        for par in dados["clm"]["pares_entrada_alvo"]:
            assert "entrada" in par
            assert "alvo" in par

    def test_comparacao_esta_presente(self, client: TestClient) -> None:
        """O campo comparacao deve ser uma string explicativa não vazia."""
        resposta = client.get("/api/training/training-objectives")
        dados = resposta.json()
        assert "comparacao" in dados
        assert isinstance(dados["comparacao"], str)
        assert len(dados["comparacao"]) > 0
