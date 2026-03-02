"""Gerenciamento de modelos HuggingFace com carregamento lazy e cache."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

MODELOS_DISPONIVEIS: dict[str, dict[str, str | int]] = {
    "distilbert-base-uncased": {
        "descricao": "DistilBERT: versão destilada do BERT, 40% menor e 60% mais rápido",
        "num_camadas": 6,
        "num_cabecas": 12,
        "d_model": 768,
        "tipo": "encoder",
    },
    "bert-base-uncased": {
        "descricao": "BERT base: modelo bidirecional original do Google, 12 camadas",
        "num_camadas": 12,
        "num_cabecas": 12,
        "d_model": 768,
        "tipo": "encoder",
    },
    "gpt2": {
        "descricao": "GPT-2 small: modelo causal autoregressivo da OpenAI, 12 camadas",
        "num_camadas": 12,
        "num_cabecas": 12,
        "d_model": 768,
        "tipo": "decoder",
    },
}


@dataclass
class ModeloCarregadoInfo:
    """Informações sobre um modelo carregado."""

    nome: str
    num_camadas: int
    num_cabecas: int
    d_model: int
    tipo: str
    descricao: str
    carregado: bool = False
    erro: str | None = None


@dataclass
class PesosAtencaoReais:
    """Pesos de atenção extraídos de um modelo real."""

    modelo: str
    texto: str
    tokens: list[str]
    num_tokens: int
    camadas: list[dict[str, object]]  # por camada e cabeça


class ModelManager:
    """Gerencia carregamento e inferência de modelos HuggingFace.

    Usa lazy loading para não carregar modelos na inicialização.
    Mantém um cache em memória de modelos já carregados.

    Os imports de `transformers` e `torch` são feitos sob demanda
    para não bloquear a inicialização da API se não instalados.
    """

    def __init__(self) -> None:
        self._cache: dict[str, object] = {}
        self._tokenizadores: dict[str, object] = {}

    def listar_modelos_disponiveis(self) -> list[ModeloCarregadoInfo]:
        """Lista modelos disponíveis com seu status de carregamento.

        Returns:
            Lista de ModeloCarregadoInfo para cada modelo.
        """
        resultado: list[ModeloCarregadoInfo] = []
        for nome, info in MODELOS_DISPONIVEIS.items():
            resultado.append(
                ModeloCarregadoInfo(
                    nome=nome,
                    num_camadas=int(info["num_camadas"]),
                    num_cabecas=int(info["num_cabecas"]),
                    d_model=int(info["d_model"]),
                    tipo=str(info["tipo"]),
                    descricao=str(info["descricao"]),
                    carregado=nome in self._cache,
                )
            )
        return resultado

    async def carregar_modelo(self, nome_modelo: str) -> ModeloCarregadoInfo:
        """Carrega um modelo HuggingFace de forma assíncrona.

        Usa asyncio.to_thread para não bloquear o event loop durante
        o download e carregamento do modelo (operação bloqueante).

        Args:
            nome_modelo: Nome do modelo no HuggingFace Hub.

        Returns:
            ModeloCarregadoInfo com status do carregamento.

        Raises:
            ValueError: Se o modelo não está na lista de disponíveis.
        """
        if nome_modelo not in MODELOS_DISPONIVEIS:
            raise ValueError(
                f"Modelo '{nome_modelo}' não disponível. "
                f"Opções: {list(MODELOS_DISPONIVEIS.keys())}"
            )

        if nome_modelo in self._cache:
            logger.info("Modelo ja em cache.", extra={"modelo": nome_modelo})
            info = MODELOS_DISPONIVEIS[nome_modelo]
            return ModeloCarregadoInfo(
                nome=nome_modelo,
                num_camadas=int(info["num_camadas"]),
                num_cabecas=int(info["num_cabecas"]),
                d_model=int(info["d_model"]),
                tipo=str(info["tipo"]),
                descricao=str(info["descricao"]),
                carregado=True,
            )

        try:
            modelo, tokenizador = await asyncio.to_thread(
                self._carregar_modelo_sync, nome_modelo
            )
            self._cache[nome_modelo] = modelo
            self._tokenizadores[nome_modelo] = tokenizador

            info = MODELOS_DISPONIVEIS[nome_modelo]
            return ModeloCarregadoInfo(
                nome=nome_modelo,
                num_camadas=int(info["num_camadas"]),
                num_cabecas=int(info["num_cabecas"]),
                d_model=int(info["d_model"]),
                tipo=str(info["tipo"]),
                descricao=str(info["descricao"]),
                carregado=True,
            )

        except Exception as exc:
            logger.error(
                "Falha ao carregar modelo.",
                extra={"modelo": nome_modelo, "erro": str(exc)},
            )
            info = MODELOS_DISPONIVEIS[nome_modelo]
            return ModeloCarregadoInfo(
                nome=nome_modelo,
                num_camadas=int(info["num_camadas"]),
                num_cabecas=int(info["num_cabecas"]),
                d_model=int(info["d_model"]),
                tipo=str(info["tipo"]),
                descricao=str(info["descricao"]),
                carregado=False,
                erro=str(exc),
            )

    def _carregar_modelo_sync(
        self, nome_modelo: str
    ) -> tuple[object, object]:
        """Carrega modelo sincronamente (deve ser chamado via asyncio.to_thread).

        Args:
            nome_modelo: Nome do modelo no HuggingFace Hub.

        Returns:
            Tupla (modelo, tokenizador).
        """
        from transformers import AutoModel, AutoTokenizer  # type: ignore[import]

        logger.info("Iniciando download do modelo.", extra={"modelo": nome_modelo})
        tokenizador = AutoTokenizer.from_pretrained(nome_modelo)
        modelo = AutoModel.from_pretrained(
            nome_modelo, output_attentions=True
        )
        modelo.eval()
        logger.info("Modelo carregado com sucesso.", extra={"modelo": nome_modelo})
        return modelo, tokenizador

    async def obter_pesos_atencao_reais(
        self,
        nome_modelo: str,
        texto: str,
    ) -> PesosAtencaoReais:
        """Extrai pesos de atenção reais de um modelo carregado.

        Args:
            nome_modelo: Nome do modelo (deve estar no cache).
            texto: Texto de entrada para análise.

        Returns:
            PesosAtencaoReais com pesos por camada e cabeça.

        Raises:
            RuntimeError: Se o modelo não está carregado.
        """
        if nome_modelo not in self._cache:
            raise RuntimeError(
                f"Modelo '{nome_modelo}' não está carregado. "
                "Use o endpoint /load-model primeiro."
            )

        return await asyncio.to_thread(
            self._extrair_atencao_sync, nome_modelo, texto
        )

    def _extrair_atencao_sync(
        self, nome_modelo: str, texto: str
    ) -> PesosAtencaoReais:
        """Extrai pesos de atenção sincronamente.

        Args:
            nome_modelo: Nome do modelo.
            texto: Texto de entrada.

        Returns:
            PesosAtencaoReais com dados por camada.
        """
        import torch  # type: ignore[import]

        modelo = self._cache[nome_modelo]
        tokenizador = self._tokenizadores[nome_modelo]

        inputs = tokenizador(
            texto,
            return_tensors="pt",
            truncation=True,
            max_length=64,
        )

        with torch.no_grad():
            outputs = modelo(**inputs, output_attentions=True)

        tokens = tokenizador.convert_ids_to_tokens(
            inputs["input_ids"][0].tolist()
        )
        num_tokens = len(tokens)

        camadas: list[dict[str, object]] = []
        for idx_camada, atencao_camada in enumerate(outputs.attentions):
            # Shape: (batch=1, num_heads, seq_len, seq_len)
            pesos_numpy: FloatArray = atencao_camada[0].numpy()

            cabecas = [
                {
                    "cabeca": idx_cabeca + 1,
                    "pesos": pesos_numpy[idx_cabeca, :num_tokens, :num_tokens].tolist(),
                }
                for idx_cabeca in range(pesos_numpy.shape[0])
            ]

            # Atenção média entre todas as cabeças
            media_cabecas = pesos_numpy.mean(axis=0)[:num_tokens, :num_tokens]

            camadas.append(
                {
                    "camada": idx_camada + 1,
                    "cabecas": cabecas,
                    "media_cabecas": media_cabecas.tolist(),
                }
            )

        return PesosAtencaoReais(
            modelo=nome_modelo,
            texto=texto,
            tokens=tokens,
            num_tokens=num_tokens,
            camadas=camadas,
        )


_shared_manager: ModelManager | None = None


def get_model_manager() -> ModelManager:
    """Return the shared ModelManager singleton."""
    global _shared_manager
    if _shared_manager is None:
        _shared_manager = ModelManager()
    return _shared_manager
