"""Testes para utils/hardware.py e os endpoints do router routers/setup.py.

Cobre:
- Testes unitários de detectar_hardware, recomendar_modelo e verificar_dependencias
- Testes de integração via FastAPI TestClient para GET /api/setup/hardware,
  /api/setup/status, /api/setup/health, GET / e GET /health.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from utils.hardware import detectar_hardware, recomendar_modelo, verificar_dependencias

# ---------------------------------------------------------------------------
# Chaves esperadas no retorno de detectar_hardware
# ---------------------------------------------------------------------------

_CHAVES_HARDWARE_ESPERADAS: frozenset[str] = frozenset(
    {
        "cpu",
        "nucleos",
        "ram_total_gb",
        "ram_disponivel_gb",
        "gpu",
        "gpu_disponivel",
        "sistema",
        "python_version",
        "torch_instalado",
        "torch_version",
        "transformers_instalado",
        "transformers_version",
    }
)


# ---------------------------------------------------------------------------
# Helpers para construir dicts de hardware fictício
# ---------------------------------------------------------------------------


def _hw_base(
    *,
    ram_disponivel_gb: float = 8.0,
    gpu_disponivel: bool = False,
    gpu: str | None = None,
    torch_instalado: bool = True,
    transformers_instalado: bool = True,
) -> dict:
    """Retorna um dicionário de hardware mínimo para uso nos testes de recomendar_modelo."""
    return {
        "cpu": "Test CPU",
        "nucleos": 4,
        "ram_total_gb": ram_disponivel_gb + 2.0,
        "ram_disponivel_gb": ram_disponivel_gb,
        "gpu": gpu,
        "gpu_disponivel": gpu_disponivel,
        "sistema": "Linux test",
        "python_version": "3.12.0",
        "torch_instalado": torch_instalado,
        "torch_version": "2.2.0",
        "transformers_instalado": transformers_instalado,
        "transformers_version": "4.40.0",
    }


# ===========================================================================
# Testes unitários — detectar_hardware
# ===========================================================================


class TestDetectarHardware:
    """Testes diretos da função detectar_hardware."""

    def test_nucleos_igual_os_cpu_count(self) -> None:
        """nucleos deve ser igual a os.cpu_count() (ou 1 como fallback)."""
        resultado = detectar_hardware()
        esperado = os.cpu_count() or 1
        assert resultado["nucleos"] == esperado

    def test_ram_total_maior_que_zero(self) -> None:
        """ram_total_gb deve ser positivo em qualquer máquina real."""
        resultado = detectar_hardware()
        assert resultado["ram_total_gb"] > 0, (
            f"ram_total_gb inesperadamente <= 0: {resultado['ram_total_gb']}"
        )

    def test_todas_as_chaves_presentes(self) -> None:
        """O dicionário retornado deve conter exatamente as 12 chaves esperadas."""
        resultado = detectar_hardware()
        chaves_retornadas = frozenset(resultado.keys())
        chaves_faltantes = _CHAVES_HARDWARE_ESPERADAS - chaves_retornadas
        assert not chaves_faltantes, f"Chaves ausentes no retorno: {chaves_faltantes}"


# ===========================================================================
# Testes unitários — recomendar_modelo
# ===========================================================================


class TestRecomendarModelo:
    """Testes para a lógica de recomendação de modelo baseada no hardware."""

    def test_sem_torch_retorna_simulacao(self) -> None:
        """Sem torch instalado a recomendação deve ser 'simulacao'."""
        hw = _hw_base(torch_instalado=False)
        modelo, razao = recomendar_modelo(hw)
        assert modelo == "simulacao"
        assert isinstance(razao, str)

    def test_sem_transformers_retorna_simulacao(self) -> None:
        """Sem transformers instalado a recomendação deve ser 'simulacao'."""
        hw = _hw_base(transformers_instalado=False)
        modelo, razao = recomendar_modelo(hw)
        assert modelo == "simulacao"

    def test_ram_insuficiente_retorna_simulacao(self) -> None:
        """RAM disponível < 4 GB deve resultar em 'simulacao'."""
        hw = _hw_base(ram_disponivel_gb=2.0)
        modelo, razao = recomendar_modelo(hw)
        assert modelo == "simulacao"
        assert "2.0" in razao or "insuficiente" in razao or "simulação" in razao

    def test_ram_media_sem_gpu_retorna_distilbert(self) -> None:
        """4 GB <= RAM < 8 GB sem GPU deve retornar 'distilbert-base-uncased'."""
        hw = _hw_base(ram_disponivel_gb=5.0, gpu_disponivel=False)
        modelo, _ = recomendar_modelo(hw)
        assert modelo == "distilbert-base-uncased"

    def test_ram_alta_retorna_gpt2(self) -> None:
        """RAM >= 16 GB deve retornar 'gpt2', independente de GPU."""
        hw = _hw_base(ram_disponivel_gb=16.0, gpu_disponivel=False)
        modelo, razao = recomendar_modelo(hw)
        assert modelo == "gpt2"
        assert "16.0" in razao or "potente" in razao

    def test_gpu_com_ram_suficiente_retorna_bert(self) -> None:
        """RAM >= 8 GB com GPU (sem VRAM >= 4) deve retornar 'bert-base-uncased'."""
        hw = _hw_base(ram_disponivel_gb=8.0, gpu_disponivel=True, gpu="Apple MPS")
        modelo, razao = recomendar_modelo(hw)
        assert modelo == "bert-base-uncased"

    def test_gpu_vram_alta_retorna_gpt2(self) -> None:
        """GPU com VRAM >= 4 GB deve retornar 'gpt2'."""
        hw = _hw_base(
            ram_disponivel_gb=6.0,
            gpu_disponivel=True,
            gpu="NVIDIA RTX 3060 (8.0 GB VRAM)",
        )
        modelo, _ = recomendar_modelo(hw)
        assert modelo == "gpt2"

    def test_retorno_e_tupla_de_duas_strings(self) -> None:
        """recomendar_modelo deve retornar uma tupla (str, str)."""
        hw = _hw_base(ram_disponivel_gb=8.0)
        resultado = recomendar_modelo(hw)
        assert isinstance(resultado, tuple)
        assert len(resultado) == 2
        assert all(isinstance(parte, str) for parte in resultado)


# ===========================================================================
# Testes unitários — verificar_dependencias
# ===========================================================================


class TestVerificarDependencias:
    """Testes para verificar_dependencias."""

    def test_numpy_sempre_instalado(self) -> None:
        """numpy é dependência obrigatória do projeto — deve estar instalado."""
        deps = verificar_dependencias()
        assert "numpy" in deps
        assert deps["numpy"]["instalado"] is True
        assert deps["numpy"]["versao"] is not None

    def test_estrutura_de_cada_entrada(self) -> None:
        """Cada entrada deve ter as chaves 'instalado' (bool) e 'versao' (str|None)."""
        deps = verificar_dependencias()
        for nome, info in deps.items():
            assert "instalado" in info, f"Chave 'instalado' ausente em '{nome}'"
            assert "versao" in info, f"Chave 'versao' ausente em '{nome}'"
            assert isinstance(info["instalado"], bool), (
                f"'instalado' não é bool em '{nome}': {type(info['instalado'])}"
            )

    def test_contem_dependencias_esperadas(self) -> None:
        """Deve verificar ao menos torch, transformers, numpy, scipy e sklearn."""
        deps = verificar_dependencias()
        for nome in ("torch", "transformers", "numpy", "scipy", "sklearn"):
            assert nome in deps, f"Dependência '{nome}' ausente no resultado"


# ===========================================================================
# Testes de integração — endpoints via TestClient
# ===========================================================================


class TestEndpointRaiz:
    """Testes para o endpoint GET /."""

    def test_status_online(self, client: TestClient) -> None:
        """GET / deve retornar status 'online'."""
        resposta = client.get("/")
        assert resposta.status_code == 200
        assert resposta.json()["status"] == "online"

    def test_versao_1_0_0(self, client: TestClient) -> None:
        """GET / deve retornar versao '1.0.0'."""
        resposta = client.get("/")
        assert resposta.json()["versao"] == "1.0.0"


class TestEndpointHealth:
    """Testes para o endpoint GET /health."""

    def test_status_healthy(self, client: TestClient) -> None:
        """GET /health deve retornar status 'healthy'."""
        resposta = client.get("/health")
        assert resposta.status_code == 200
        assert resposta.json()["status"] == "healthy"


class TestEndpointSetupHardware:
    """Testes para GET /api/setup/hardware."""

    def test_nucleos_positivo(self, client: TestClient) -> None:
        """nucleos deve ser maior que zero."""
        resposta = client.get("/api/setup/hardware")
        assert resposta.status_code == 200
        assert resposta.json()["nucleos"] > 0

    def test_ram_total_positiva(self, client: TestClient) -> None:
        """ram_total_gb deve ser maior que zero."""
        resposta = client.get("/api/setup/hardware")
        assert resposta.json()["ram_total_gb"] > 0

    def test_sistema_e_string(self, client: TestClient) -> None:
        """sistema deve ser uma string não vazia."""
        resposta = client.get("/api/setup/hardware")
        sistema = resposta.json()["sistema"]
        assert isinstance(sistema, str)
        assert len(sistema) > 0


class TestEndpointSetupStatus:
    """Testes para GET /api/setup/status."""

    def test_modelo_carregado_e_bool(self, client: TestClient) -> None:
        """modelo_carregado deve ser um booleano."""
        resposta = client.get("/api/setup/status")
        assert resposta.status_code == 200
        assert isinstance(resposta.json()["modelo_carregado"], bool)

    def test_hardware_presente_na_resposta(self, client: TestClient) -> None:
        """hardware deve estar presente e conter nucleos e ram_total_gb."""
        resposta = client.get("/api/setup/status")
        corpo = resposta.json()
        assert "hardware" in corpo
        assert "nucleos" in corpo["hardware"]
        assert "ram_total_gb" in corpo["hardware"]


class TestEndpointSetupHealthDetalhado:
    """Testes para GET /api/setup/health."""

    def test_backend_true(self, client: TestClient) -> None:
        """backend deve ser True — o servidor está respondendo."""
        resposta = client.get("/api/setup/health")
        assert resposta.status_code == 200
        assert resposta.json()["backend"] is True

    def test_campos_obrigatorios_presentes(self, client: TestClient) -> None:
        """Resposta deve conter os campos torch, transformers e modelo_carregado."""
        resposta = client.get("/api/setup/health")
        corpo = resposta.json()
        for campo in ("torch", "transformers", "modelo_carregado"):
            assert campo in corpo, f"Campo obrigatório '{campo}' ausente na resposta"
