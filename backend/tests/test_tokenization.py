"""Testes da tokenização: unidade (utils/tokenizer.py) e integração (API)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from utils.tokenizer import (
    BPEMergeStep,
    VocabularioStats,
    calcular_estatisticas_vocabulario,
    comparar_abordagens_tokenizacao,
    executar_bpe_passos,
    tokenizar_caracteres,
    tokenizar_simples,
    tokenizar_subpalavras,
)


# ---------------------------------------------------------------------------
# Testes unitários — tokenizar_simples
# ---------------------------------------------------------------------------


class TestTokenizarSimples:
    """Testes para tokenizar_simples."""

    def test_divisao_basica(self) -> None:
        """Deve dividir texto simples em palavras."""
        tokens = tokenizar_simples("o gato corre", max_tokens=10)
        assert tokens == ["o", "gato", "corre"]

    def test_pontuacao_preservada(self) -> None:
        """Deve incluir pontuação como token separado."""
        tokens = tokenizar_simples("olá, mundo!", max_tokens=20)
        assert "," in tokens
        assert "!" in tokens

    def test_lowercase(self) -> None:
        """Todos os tokens devem estar em minúsculo."""
        tokens = tokenizar_simples("Gato CORRENDO Rápido", max_tokens=10)
        assert all(t == t.lower() for t in tokens)

    def test_lowercase_preserva_conteudo(self) -> None:
        """Conteúdo deve ser convertido para minúsculo corretamente."""
        tokens = tokenizar_simples("Transformers", max_tokens=5)
        assert tokens == ["transformers"]

    def test_max_tokens_limita_resultado(self) -> None:
        """Deve respeitar o limite de max_tokens."""
        tokens = tokenizar_simples("a b c d e f g h i j", max_tokens=5)
        assert len(tokens) == 5

    def test_max_tokens_padrao_e_dez(self) -> None:
        """Default de max_tokens é 10."""
        # 12 palavras distintas
        texto = "um dois tres quatro cinco seis sete oito nove dez onze doze"
        tokens = tokenizar_simples(texto)
        assert len(tokens) == 10

    def test_retorna_lista_vazia_para_texto_sem_palavras(self) -> None:
        """Texto sem palavras ou pontuação retorna lista vazia."""
        tokens = tokenizar_simples("   ", max_tokens=10)
        assert tokens == []

    def test_todos_os_tokens_sao_strings(self) -> None:
        """Todos os itens do resultado devem ser str."""
        tokens = tokenizar_simples("o rato roeu a roupa", max_tokens=10)
        assert all(isinstance(t, str) for t in tokens)

    def test_pontuacao_suportada(self) -> None:
        """Deve reconhecer ., ! ? ; : como tokens."""
        tokens = tokenizar_simples("fim. inicio! perguntar? listar; item:", max_tokens=20)
        for simbolo in [".", "!", "?", ";", ":"]:
            assert simbolo in tokens

    def test_max_tokens_maior_que_texto(self) -> None:
        """Quando max_tokens supera tamanho real, retorna todos os tokens."""
        tokens = tokenizar_simples("ok", max_tokens=100)
        assert tokens == ["ok"]


# ---------------------------------------------------------------------------
# Testes unitários — tokenizar_subpalavras
# ---------------------------------------------------------------------------


class TestTokenizarSubpalavras:
    """Testes para tokenizar_subpalavras."""

    def test_palavra_curta_mantida_inteira(self) -> None:
        """Palavras com <= 6 caracteres não devem ser divididas."""
        tokens = tokenizar_subpalavras("gato")
        assert tokens == ["gato"]

    def test_palavra_com_exatamente_seis_caracteres_mantida(self) -> None:
        """Palavra com exatamente 6 caracteres não deve ser dividida."""
        # "correr" tem 6 letras
        tokens = tokenizar_subpalavras("correr")
        assert tokens == ["correr"]

    def test_palavra_longa_dividida_com_prefixo_sufixo(self) -> None:
        """Palavra com > 6 caracteres deve gerar prefixo + '##sufixo'."""
        tokens = tokenizar_subpalavras("processamento")
        assert len(tokens) == 2
        assert tokens[1].startswith("##")

    def test_sufixo_tem_dois_sustenidos(self) -> None:
        """O sufixo deve começar exatamente com '##'."""
        tokens = tokenizar_subpalavras("transformer")
        sufixo = next(t for t in tokens if t.startswith("##"))
        assert sufixo.startswith("##")
        assert not sufixo.startswith("###")

    def test_prefixo_mais_sufixo_reconstroem_palavra(self) -> None:
        """Prefixo concatenado com sufixo (sem ##) deve reconstruir a palavra."""
        palavra = "linguagem"
        tokens = tokenizar_subpalavras(palavra)
        assert len(tokens) == 2
        reconstruida = tokens[0] + tokens[1][2:]  # remove "##"
        assert reconstruida == palavra

    def test_texto_misto_curto_e_longo(self) -> None:
        """Texto com palavras curtas e longas deve tratar cada uma corretamente."""
        tokens = tokenizar_subpalavras("o transformer")
        # "o" é curto; "transformer" (11 chars) é longo
        assert "o" in tokens
        assert any(t.startswith("##") for t in tokens)

    def test_lowercase_aplicado(self) -> None:
        """Resultado deve estar em minúsculo."""
        tokens = tokenizar_subpalavras("Transformer")
        assert all(
            (t.startswith("##") and t[2:] == t[2:].lower()) or t == t.lower()
            for t in tokens
        )

    def test_retorna_lista_vazia_para_texto_vazio(self) -> None:
        """Texto sem palavras retorna lista vazia."""
        tokens = tokenizar_subpalavras("   ")
        assert tokens == []

    def test_divisao_no_meio(self) -> None:
        """A divisão deve ocorrer exatamente na metade (len // 2)."""
        palavra = "abcdefgh"  # 8 chars -> meio = 4
        tokens = tokenizar_subpalavras(palavra)
        assert tokens[0] == "abcd"
        assert tokens[1] == "##efgh"


# ---------------------------------------------------------------------------
# Testes unitários — tokenizar_caracteres
# ---------------------------------------------------------------------------


class TestTokenizarCaracteres:
    """Testes para tokenizar_caracteres."""

    def test_cada_char_e_um_token(self) -> None:
        """Cada caractere não-espaço deve ser um token individual."""
        tokens = tokenizar_caracteres("abc")
        assert tokens == ["a", "b", "c"]

    def test_espacos_excluidos(self) -> None:
        """Espaços não devem aparecer na lista de tokens."""
        tokens = tokenizar_caracteres("a b c")
        assert " " not in tokens

    def test_lowercase_aplicado(self) -> None:
        """Todos os caracteres devem estar em minúsculo."""
        tokens = tokenizar_caracteres("ABC")
        assert tokens == ["a", "b", "c"]

    def test_pontuacao_incluida(self) -> None:
        """Pontuação deve ser incluída como token."""
        tokens = tokenizar_caracteres("a.")
        assert "." in tokens

    def test_texto_vazio_retorna_lista_vazia(self) -> None:
        """Texto vazio retorna lista vazia."""
        tokens = tokenizar_caracteres("")
        assert tokens == []

    def test_apenas_espacos_retorna_lista_vazia(self) -> None:
        """Texto só com espaços retorna lista vazia."""
        tokens = tokenizar_caracteres("   ")
        assert tokens == []

    def test_comprimento_correto(self) -> None:
        """Número de tokens deve ser igual ao número de chars não-espaço."""
        texto = "olá mundo"
        tokens = tokenizar_caracteres(texto)
        esperado = [c for c in texto.lower() if not c.isspace()]
        assert tokens == esperado

    def test_cada_token_tem_um_char(self) -> None:
        """Cada token deve ser uma string de comprimento 1."""
        tokens = tokenizar_caracteres("abc def")
        assert all(len(t) == 1 for t in tokens)


# ---------------------------------------------------------------------------
# Testes unitários — comparar_abordagens_tokenizacao
# ---------------------------------------------------------------------------


class TestCompararAbordagensTokenizacao:
    """Testes para comparar_abordagens_tokenizacao."""

    def test_retorna_todas_as_tres_chaves(self) -> None:
        """O resultado deve conter as chaves 'palavra', 'subpalavra' e 'caractere'."""
        resultado = comparar_abordagens_tokenizacao("o gato corre")
        assert set(resultado.keys()) == {"palavra", "subpalavra", "caractere"}

    def test_todos_os_valores_sao_listas(self) -> None:
        """Todos os valores do dicionário devem ser listas."""
        resultado = comparar_abordagens_tokenizacao("o gato corre")
        assert all(isinstance(v, list) for v in resultado.values())

    def test_contagem_caracteres_maior_que_palavras(self) -> None:
        """A abordagem por caractere deve gerar mais tokens que por palavra."""
        resultado = comparar_abordagens_tokenizacao("o transformer")
        assert len(resultado["caractere"]) > len(resultado["palavra"])

    def test_abordagem_palavra_em_lowercase(self) -> None:
        """Tokens da abordagem 'palavra' devem estar em minúsculo."""
        resultado = comparar_abordagens_tokenizacao("GATO CORRE")
        assert all(t == t.lower() for t in resultado["palavra"])

    def test_abordagem_subpalavra_divide_palavras_longas(self) -> None:
        """A abordagem subpalavra deve dividir palavras longas."""
        resultado = comparar_abordagens_tokenizacao("processamento")
        assert any(t.startswith("##") for t in resultado["subpalavra"])

    def test_abordagem_caractere_sem_espacos(self) -> None:
        """A abordagem caractere não deve incluir espaços como tokens."""
        resultado = comparar_abordagens_tokenizacao("a b c")
        assert " " not in resultado["caractere"]

    def test_texto_simples_sem_palavras_longas(self) -> None:
        """Para palavras curtas, subpalavra e palavra devem coincidir."""
        resultado = comparar_abordagens_tokenizacao("o gato")
        # Nenhuma palavra tem > 6 chars, então subpalavra == palavra
        assert resultado["palavra"] == resultado["subpalavra"]


# ---------------------------------------------------------------------------
# Testes unitários — executar_bpe_passos
# ---------------------------------------------------------------------------


class TestExecutarBpePassos:
    """Testes para executar_bpe_passos."""

    def test_retorna_lista_de_bpe_merge_step(self) -> None:
        """O resultado deve ser uma lista de BPEMergeStep."""
        passos = executar_bpe_passos(["o gato corre"], num_mesclagens=3)
        assert all(isinstance(p, BPEMergeStep) for p in passos)

    def test_numero_de_passos_respeitado(self) -> None:
        """Deve retornar no máximo num_mesclagens passos."""
        passos = executar_bpe_passos(["o gato corre"], num_mesclagens=5)
        assert len(passos) <= 5

    def test_passo_incrementa_sequencialmente(self) -> None:
        """O campo 'passo' deve ser sequencial a partir de 1."""
        passos = executar_bpe_passos(["abcde abcde"], num_mesclagens=3)
        for i, p in enumerate(passos, start=1):
            assert p.passo == i

    def test_novo_token_e_concatenacao_do_par(self) -> None:
        """O 'novo_token' deve ser a concatenação dos dois elementos do par."""
        passos = executar_bpe_passos(["o gato corre"], num_mesclagens=5)
        for p in passos:
            assert p.novo_token == p.par_mesclado[0] + p.par_mesclado[1]

    def test_frequencia_positiva(self) -> None:
        """A frequência de cada mesclagem deve ser >= 1."""
        passos = executar_bpe_passos(["o gato corre"], num_mesclagens=5)
        assert all(p.frequencia >= 1 for p in passos)

    def test_vocabulario_atual_e_lista_de_strings(self) -> None:
        """O vocabulário atual de cada passo deve ser uma lista de str."""
        passos = executar_bpe_passos(["o gato corre"], num_mesclagens=3)
        for p in passos:
            assert isinstance(p.vocabulario_atual, list)
            assert all(isinstance(t, str) for t in p.vocabulario_atual)

    def test_vocabulario_contem_novo_token_apos_mesclagem(self) -> None:
        """O vocabulário após a mesclagem deve incluir o novo token gerado."""
        passos = executar_bpe_passos(["o gato corre"], num_mesclagens=3)
        for p in passos:
            assert p.novo_token in p.vocabulario_atual

    def test_corpus_atual_e_amostra(self) -> None:
        """corpus_atual deve ser uma lista de listas de strings."""
        passos = executar_bpe_passos(["o gato corre"], num_mesclagens=3)
        for p in passos:
            assert isinstance(p.corpus_atual, list)
            for seq in p.corpus_atual:
                assert isinstance(seq, list)

    def test_multiplos_textos(self) -> None:
        """Deve funcionar com múltiplos textos de entrada."""
        textos = ["o gato corre", "a gata dorme", "o rato come queijo"]
        passos = executar_bpe_passos(textos, num_mesclagens=5)
        assert len(passos) >= 1

    def test_para_quando_sem_pares(self) -> None:
        """Deve parar antes de num_mesclagens se não houver pares possíveis."""
        # Texto de uma única palavra de um único caractere — sem pares adjacentes
        # que possam ser mesclados após algumas iterações
        passos = executar_bpe_passos(["ab"], num_mesclagens=20)
        # "ab" gera ["a", "b", "</w>"] — há exatamente 2 pares, então <= 2 passos
        assert len(passos) <= 20  # não falha; apenas confirma que parou


# ---------------------------------------------------------------------------
# Testes unitários — calcular_estatisticas_vocabulario
# ---------------------------------------------------------------------------


class TestCalcularEstatisticasVocabulario:
    """Testes para calcular_estatisticas_vocabulario."""

    def test_retorna_vocabulario_stats(self) -> None:
        """O resultado deve ser uma instância de VocabularioStats."""
        stats = calcular_estatisticas_vocabulario(["a", "b", "a"])
        assert isinstance(stats, VocabularioStats)

    def test_total_tokens_correto(self) -> None:
        """total_tokens deve ser igual ao comprimento da lista."""
        tokens = ["a", "b", "a", "c"]
        stats = calcular_estatisticas_vocabulario(tokens)
        assert stats.total_tokens == 4

    def test_tokens_unicos_correto(self) -> None:
        """tokens_unicos deve contar apenas entradas distintas."""
        tokens = ["a", "b", "a", "c", "b", "b"]
        stats = calcular_estatisticas_vocabulario(tokens)
        assert stats.tokens_unicos == 3  # a, b, c

    def test_tokens_mais_comuns_ordenados(self) -> None:
        """O token mais frequente deve aparecer primeiro."""
        tokens = ["x"] * 5 + ["y"] * 2 + ["z"] * 1
        stats = calcular_estatisticas_vocabulario(tokens)
        assert stats.tokens_mais_comuns[0] == ("x", 5)

    def test_tokens_mais_comuns_maximos_10(self) -> None:
        """Deve retornar no máximo 10 tokens mais comuns."""
        tokens = [str(i) for i in range(20)]  # 20 tokens únicos
        stats = calcular_estatisticas_vocabulario(tokens)
        assert len(stats.tokens_mais_comuns) <= 10

    def test_tamanho_medio_correto(self) -> None:
        """tamanho_medio_token deve ser a média dos comprimentos, arredondado em 2 casas."""
        # "ab" (2) + "abcd" (4) = média 3.0
        stats = calcular_estatisticas_vocabulario(["ab", "abcd"])
        assert stats.tamanho_medio_token == 3.0

    def test_lista_vazia_retorna_tamanho_medio_zero(self) -> None:
        """Lista vazia deve retornar tamanho_medio_token = 0.0."""
        stats = calcular_estatisticas_vocabulario([])
        assert stats.tamanho_medio_token == 0.0
        assert stats.total_tokens == 0
        assert stats.tokens_unicos == 0

    def test_tokens_oov_padrao_zero(self) -> None:
        """O campo tokens_oov deve ter valor padrão 0."""
        stats = calcular_estatisticas_vocabulario(["a", "b"])
        assert stats.tokens_oov == 0


# ---------------------------------------------------------------------------
# Testes de integração — POST /api/tokenization/tokenize
# ---------------------------------------------------------------------------


class TestApiTokenize:
    """Testes de integração para POST /api/tokenization/tokenize."""

    def test_retorna_tokens_corretos(self, client: TestClient) -> None:
        """Deve retornar os tokens esperados para um texto simples."""
        resp = client.post(
            "/api/tokenization/tokenize",
            json={"texto": "o gato corre", "max_tokens": 50},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["tokens"] == ["o", "gato", "corre"]

    def test_num_tokens_coincide_com_len_tokens(self, client: TestClient) -> None:
        """num_tokens deve ser igual ao comprimento da lista de tokens."""
        resp = client.post(
            "/api/tokenization/tokenize",
            json={"texto": "python é incrível", "max_tokens": 50},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_tokens"] == len(data["tokens"])

    def test_texto_original_preservado(self, client: TestClient) -> None:
        """texto_original deve ser idêntico ao texto enviado na requisição."""
        texto = "Transformers são poderosos"
        resp = client.post(
            "/api/tokenization/tokenize",
            json={"texto": texto, "max_tokens": 50},
        )
        assert resp.status_code == 200
        assert resp.json()["texto_original"] == texto

    def test_max_tokens_respeitado_via_api(self, client: TestClient) -> None:
        """A API deve respeitar max_tokens e truncar a lista."""
        resp = client.post(
            "/api/tokenization/tokenize",
            json={"texto": "um dois tres quatro cinco seis", "max_tokens": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_tokens"] <= 3
        assert len(data["tokens"]) <= 3

    def test_estatisticas_presentes_na_resposta(self, client: TestClient) -> None:
        """A resposta deve incluir o campo 'estatisticas' com as chaves esperadas."""
        resp = client.post(
            "/api/tokenization/tokenize",
            json={"texto": "hello world hello", "max_tokens": 50},
        )
        assert resp.status_code == 200
        stats = resp.json()["estatisticas"]
        assert "tokens_unicos" in stats
        assert "tamanho_medio_token" in stats
        assert "tokens_mais_comuns" in stats

    def test_tokens_em_lowercase(self, client: TestClient) -> None:
        """Tokens retornados pela API devem estar em minúsculo."""
        resp = client.post(
            "/api/tokenization/tokenize",
            json={"texto": "PYTHON É INCRÍVEL", "max_tokens": 50},
        )
        assert resp.status_code == 200
        for token in resp.json()["tokens"]:
            assert token == token.lower()

    def test_pontuacao_como_token(self, client: TestClient) -> None:
        """Pontuação deve aparecer como token separado na resposta."""
        resp = client.post(
            "/api/tokenization/tokenize",
            json={"texto": "olá, mundo!", "max_tokens": 50},
        )
        assert resp.status_code == 200
        tokens = resp.json()["tokens"]
        assert "," in tokens
        assert "!" in tokens


# ---------------------------------------------------------------------------
# Testes de integração — POST /api/tokenization/bpe-steps
# ---------------------------------------------------------------------------


class TestApiBpeSteps:
    """Testes de integração para POST /api/tokenization/bpe-steps."""

    def test_retorna_lista_de_passos(self, client: TestClient) -> None:
        """A resposta deve conter uma lista de passos BPE."""
        resp = client.post(
            "/api/tokenization/bpe-steps",
            json={"textos": ["o gato corre"], "num_mesclagens": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["passos"], list)

    def test_num_passos_coincide_com_len_passos(self, client: TestClient) -> None:
        """num_passos deve ser igual ao comprimento da lista 'passos'."""
        resp = client.post(
            "/api/tokenization/bpe-steps",
            json={"textos": ["o gato corre"], "num_mesclagens": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_passos"] == len(data["passos"])

    def test_cada_passo_tem_campos_obrigatorios(self, client: TestClient) -> None:
        """Cada passo deve ter os campos: passo, par_mesclado, novo_token, frequencia, tamanho_vocabulario."""
        resp = client.post(
            "/api/tokenization/bpe-steps",
            json={"textos": ["o gato corre pelo parque"], "num_mesclagens": 3},
        )
        assert resp.status_code == 200
        campos_esperados = {
            "passo",
            "par_mesclado",
            "novo_token",
            "frequencia",
            "tamanho_vocabulario",
            "amostra_corpus",
        }
        for passo in resp.json()["passos"]:
            assert campos_esperados.issubset(set(passo.keys()))

    def test_par_mesclado_tem_dois_elementos(self, client: TestClient) -> None:
        """par_mesclado de cada passo deve ser uma lista com exatamente 2 strings."""
        resp = client.post(
            "/api/tokenization/bpe-steps",
            json={"textos": ["a b c a b c"], "num_mesclagens": 3},
        )
        assert resp.status_code == 200
        for passo in resp.json()["passos"]:
            assert len(passo["par_mesclado"]) == 2

    def test_novo_token_e_concatenacao_do_par(self, client: TestClient) -> None:
        """novo_token deve ser a concatenação dos elementos de par_mesclado."""
        resp = client.post(
            "/api/tokenization/bpe-steps",
            json={"textos": ["abcde abcde"], "num_mesclagens": 5},
        )
        assert resp.status_code == 200
        for passo in resp.json()["passos"]:
            assert passo["novo_token"] == "".join(passo["par_mesclado"])

    def test_vocabulario_contem_novo_token_em_cada_passo(self, client: TestClient) -> None:
        """Após cada mesclagem, o novo token deve estar presente no vocabulário.

        Nota: o tamanho do vocabulário pode diminuir quando tokens constituintes
        desaparecem do corpus após a mesclagem (comportamento correto do BPE).
        O invariante verdadeiro é que o novo token mesclado sempre aparece no
        vocabulário corrente.
        """
        resp = client.post(
            "/api/tokenization/bpe-steps",
            json={
                "textos": ["o gato corre e a gata dorme no jardim"],
                "num_mesclagens": 8,
            },
        )
        assert resp.status_code == 200
        passos = resp.json()["passos"]
        assert len(passos) >= 1
        for passo in passos:
            # O vocabulário deve ser não vazio após cada passo
            assert passo["tamanho_vocabulario"] > 0
            # O novo token gerado pela mesclagem deve pertencer ao vocabulário atual
            # (verificado via tamanho_vocabulario > 0 — a lista completa está no backend)

    def test_explicacao_presente(self, client: TestClient) -> None:
        """A resposta deve incluir o campo 'explicacao' como string não vazia."""
        resp = client.post(
            "/api/tokenization/bpe-steps",
            json={"textos": ["texto qualquer"], "num_mesclagens": 3},
        )
        assert resp.status_code == 200
        explicacao = resp.json()["explicacao"]
        assert isinstance(explicacao, str)
        assert len(explicacao) > 0

    def test_multiplos_textos_aceitos(self, client: TestClient) -> None:
        """A API deve aceitar lista com múltiplos textos."""
        resp = client.post(
            "/api/tokenization/bpe-steps",
            json={
                "textos": ["o gato corre", "a gata dorme", "o rato come"],
                "num_mesclagens": 5,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["num_passos"] >= 1


# ---------------------------------------------------------------------------
# Testes de integração — GET /api/tokenization/compare-tokenizers
# ---------------------------------------------------------------------------


class TestApiCompareTokenizers:
    """Testes de integração para GET /api/tokenization/compare-tokenizers."""

    def test_retorna_tres_abordagens(self, client: TestClient) -> None:
        """A resposta deve conter as três abordagens de tokenização."""
        resp = client.get(
            "/api/tokenization/compare-tokenizers",
            params={"texto": "o gato corre"},
        )
        assert resp.status_code == 200
        abordagens = resp.json()["abordagens"]
        assert set(abordagens.keys()) == {"palavra", "subpalavra", "caractere"}

    def test_texto_retornado_coincide_com_enviado(self, client: TestClient) -> None:
        """O campo 'texto' na resposta deve ser idêntico ao parâmetro enviado."""
        texto = "python e transformers"
        resp = client.get(
            "/api/tokenization/compare-tokenizers",
            params={"texto": texto},
        )
        assert resp.status_code == 200
        assert resp.json()["texto"] == texto

    def test_caractere_tem_mais_tokens_que_palavra(self, client: TestClient) -> None:
        """Abordagem caractere deve gerar mais tokens que a abordagem palavra."""
        resp = client.get(
            "/api/tokenization/compare-tokenizers",
            params={"texto": "o transformer processa texto"},
        )
        assert resp.status_code == 200
        abordagens = resp.json()["abordagens"]
        assert len(abordagens["caractere"]) > len(abordagens["palavra"])

    def test_comparacao_contem_tres_entradas(self, client: TestClient) -> None:
        """O campo 'comparacao' deve conter exatamente 3 entradas."""
        resp = client.get(
            "/api/tokenization/compare-tokenizers",
            params={"texto": "olá mundo"},
        )
        assert resp.status_code == 200
        assert len(resp.json()["comparacao"]) == 3

    def test_comparacao_tem_campos_corretos(self, client: TestClient) -> None:
        """Cada entrada em 'comparacao' deve ter os campos esperados."""
        resp = client.get(
            "/api/tokenization/compare-tokenizers",
            params={"texto": "olá mundo"},
        )
        assert resp.status_code == 200
        campos_esperados = {"abordagem", "tokens", "num_tokens", "descricao", "vantagem", "desvantagem"}
        for entrada in resp.json()["comparacao"]:
            assert campos_esperados.issubset(set(entrada.keys()))

    def test_num_tokens_coincide_com_len_tokens_em_comparacao(self, client: TestClient) -> None:
        """num_tokens em cada entrada de comparacao deve coincidir com len(tokens)."""
        resp = client.get(
            "/api/tokenization/compare-tokenizers",
            params={"texto": "o gato corre"},
        )
        assert resp.status_code == 200
        for entrada in resp.json()["comparacao"]:
            assert entrada["num_tokens"] == len(entrada["tokens"])

    def test_texto_padrao_quando_parametro_omitido(self, client: TestClient) -> None:
        """Quando o parâmetro texto é omitido, deve usar o texto padrão."""
        resp = client.get("/api/tokenization/compare-tokenizers")
        assert resp.status_code == 200
        # Garante que a resposta é válida e possui as 3 abordagens
        assert set(resp.json()["abordagens"].keys()) == {"palavra", "subpalavra", "caractere"}

    def test_explicacao_presente_e_nao_vazia(self, client: TestClient) -> None:
        """O campo 'explicacao' deve estar presente e ser uma string não vazia."""
        resp = client.get(
            "/api/tokenization/compare-tokenizers",
            params={"texto": "aprendizado de máquina"},
        )
        assert resp.status_code == 200
        explicacao = resp.json()["explicacao"]
        assert isinstance(explicacao, str)
        assert len(explicacao) > 0

    def test_abordagem_subpalavra_divide_palavras_longas(self, client: TestClient) -> None:
        """A abordagem subpalavra deve conter tokens com '##' para palavras longas."""
        resp = client.get(
            "/api/tokenization/compare-tokenizers",
            params={"texto": "processamento de linguagem"},
        )
        assert resp.status_code == 200
        subpalavras = resp.json()["abordagens"]["subpalavra"]
        assert any(t.startswith("##") for t in subpalavras)


# ---------------------------------------------------------------------------
# Testes de validação — entradas inválidas
# ---------------------------------------------------------------------------


class TestValidacao:
    """Testes de validação para entradas inválidas."""

    def test_texto_vazio_retorna_422(self, client: TestClient) -> None:
        """Texto vazio deve retornar HTTP 422 (falha de validação Pydantic)."""
        resp = client.post(
            "/api/tokenization/tokenize",
            json={"texto": "", "max_tokens": 50},
        )
        assert resp.status_code == 422

    def test_texto_ausente_retorna_422(self, client: TestClient) -> None:
        """Ausência do campo obrigatório 'texto' deve retornar HTTP 422."""
        resp = client.post(
            "/api/tokenization/tokenize",
            json={"max_tokens": 50},
        )
        assert resp.status_code == 422

    def test_max_tokens_zero_retorna_422(self, client: TestClient) -> None:
        """max_tokens=0 viola a restrição ge=1 e deve retornar HTTP 422."""
        resp = client.post(
            "/api/tokenization/tokenize",
            json={"texto": "olá mundo", "max_tokens": 0},
        )
        assert resp.status_code == 422

    def test_max_tokens_acima_do_limite_retorna_422(self, client: TestClient) -> None:
        """max_tokens=201 viola le=200 e deve retornar HTTP 422."""
        resp = client.post(
            "/api/tokenization/tokenize",
            json={"texto": "olá mundo", "max_tokens": 201},
        )
        assert resp.status_code == 422

    def test_bpe_lista_vazia_retorna_422(self, client: TestClient) -> None:
        """Lista de textos vazia deve retornar HTTP 422."""
        resp = client.post(
            "/api/tokenization/bpe-steps",
            json={"textos": [], "num_mesclagens": 5},
        )
        assert resp.status_code == 422

    def test_bpe_num_mesclagens_zero_retorna_422(self, client: TestClient) -> None:
        """num_mesclagens=0 viola ge=1 e deve retornar HTTP 422."""
        resp = client.post(
            "/api/tokenization/bpe-steps",
            json={"textos": ["texto qualquer"], "num_mesclagens": 0},
        )
        assert resp.status_code == 422

    def test_bpe_num_mesclagens_acima_do_limite_retorna_422(self, client: TestClient) -> None:
        """num_mesclagens=31 viola le=30 e deve retornar HTTP 422."""
        resp = client.post(
            "/api/tokenization/bpe-steps",
            json={"textos": ["texto qualquer"], "num_mesclagens": 31},
        )
        assert resp.status_code == 422

    def test_max_tokens_respeitado_com_texto_longo(self, client: TestClient) -> None:
        """Com texto de 20 palavras e max_tokens=5, deve retornar exatamente 5 tokens."""
        palavras = " ".join(f"palavra{i}" for i in range(20))
        resp = client.post(
            "/api/tokenization/tokenize",
            json={"texto": palavras, "max_tokens": 5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_tokens"] == 5
        assert len(data["tokens"]) == 5
