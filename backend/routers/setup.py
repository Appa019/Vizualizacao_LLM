"""Router de configuração: detecção de hardware e setup one-click de modelo."""

from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from core.model_manager import get_model_manager, MODELOS_DISPONIVEIS
from utils.hardware import detectar_hardware, recomendar_modelo, verificar_dependencias

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/setup", tags=["configuração"])

_gerenciador = get_model_manager()


# ---------------------------------------------------------------------------
# Modelos de resposta
# ---------------------------------------------------------------------------


class HardwareResponse(BaseModel):
    """Informações de hardware do servidor."""

    cpu: str
    nucleos: int
    ram_total_gb: float
    ram_disponivel_gb: float
    gpu: str | None
    gpu_disponivel: bool
    sistema: str
    python_version: str
    torch_instalado: bool
    torch_version: str | None
    transformers_instalado: bool
    transformers_version: str | None


class ModeloInfoResponse(BaseModel):
    """Info resumida de um modelo."""

    nome: str
    descricao: str
    num_camadas: int
    num_cabecas: int
    d_model: int
    tipo: str
    carregado: bool


class SetupStatusResponse(BaseModel):
    """Estado atual do setup."""

    modelo_carregado: bool
    modelo_nome: str | None
    modelo_info: ModeloInfoResponse | None
    hardware: HardwareResponse


class EtapaSetup(BaseModel):
    """Uma etapa do processo de auto-configuração."""

    etapa: str
    status: str
    detalhe: str


class AutoConfigureResponse(BaseModel):
    """Resultado da auto-configuração."""

    sucesso: bool
    modelo_nome: str
    modelo_info: dict | None
    hardware: HardwareResponse
    recomendacao_razao: str
    etapas: list[EtapaSetup]
    erro: str | None


class HealthResponse(BaseModel):
    """Health check detalhado."""

    backend: bool
    torch: bool
    transformers: bool
    modelo_carregado: bool
    modelo_nome: str | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/hardware", response_model=HardwareResponse)
async def get_hardware() -> HardwareResponse:
    """Retorna informações de hardware do servidor."""
    hw = detectar_hardware()
    return HardwareResponse(**hw)


@router.get("/status", response_model=SetupStatusResponse)
async def get_status() -> SetupStatusResponse:
    """Retorna estado atual: modelo carregado, hardware, etc."""
    hw = detectar_hardware()
    hardware_resp = HardwareResponse(**hw)

    modelos = _gerenciador.listar_modelos_disponiveis()
    carregado = next((m for m in modelos if m.carregado), None)

    modelo_info = None
    if carregado:
        modelo_info = ModeloInfoResponse(
            nome=carregado.nome,
            descricao=carregado.descricao,
            num_camadas=carregado.num_camadas,
            num_cabecas=carregado.num_cabecas,
            d_model=carregado.d_model,
            tipo=carregado.tipo,
            carregado=True,
        )

    return SetupStatusResponse(
        modelo_carregado=carregado is not None,
        modelo_nome=carregado.nome if carregado else None,
        modelo_info=modelo_info,
        hardware=hardware_resp,
    )


@router.post("/auto-configure", response_model=AutoConfigureResponse)
async def auto_configure() -> AutoConfigureResponse:
    """Auto-configuração one-click: detecta hardware, recomenda e carrega modelo."""
    etapas: list[EtapaSetup] = []

    # Etapa 1: Detectar hardware
    etapas.append(EtapaSetup(etapa="Detectando hardware", status="em_andamento", detalhe=""))
    hw = detectar_hardware()
    hardware_resp = HardwareResponse(**hw)
    etapas[-1] = EtapaSetup(
        etapa="Detectando hardware",
        status="completo",
        detalhe=f"CPU: {hw['cpu']}, RAM: {hw['ram_total_gb']} GB, GPU: {hw['gpu'] or 'Não detectada'}",
    )

    # Etapa 2: Verificar dependências
    etapas.append(EtapaSetup(etapa="Verificando dependências", status="em_andamento", detalhe=""))
    deps = verificar_dependencias()
    torch_ok = deps.get("torch", {}).get("instalado", False)
    transf_ok = deps.get("transformers", {}).get("instalado", False)
    etapas[-1] = EtapaSetup(
        etapa="Verificando dependências",
        status="completo",
        detalhe=f"torch: {'✓ ' + str(deps['torch']['versao']) if torch_ok else '✗'}, "
        f"transformers: {'✓ ' + str(deps['transformers']['versao']) if transf_ok else '✗'}",
    )

    # Etapa 3: Selecionar modelo
    etapas.append(EtapaSetup(etapa="Selecionando modelo ideal", status="em_andamento", detalhe=""))
    modelo_nome, razao = recomendar_modelo(hw)
    etapas[-1] = EtapaSetup(
        etapa="Selecionando modelo ideal",
        status="completo",
        detalhe=f"{modelo_nome}: {razao}",
    )

    # Se modo simulação, não precisa carregar modelo
    if modelo_nome == "simulacao":
        etapas.append(EtapaSetup(
            etapa="Modo simulação ativado",
            status="completo",
            detalhe="Usando dados sintéticos — todas as visualizações funcionam sem modelo real",
        ))
        return AutoConfigureResponse(
            sucesso=True,
            modelo_nome="simulacao",
            modelo_info=None,
            hardware=hardware_resp,
            recomendacao_razao=razao,
            etapas=etapas,
            erro=None,
        )

    # Etapa 4: Baixar e carregar modelo
    etapas.append(EtapaSetup(
        etapa="Baixando e carregando modelo",
        status="em_andamento",
        detalhe=f"Carregando {modelo_nome} do HuggingFace Hub...",
    ))

    try:
        info = await _gerenciador.carregar_modelo(modelo_nome)

        if info.carregado:
            etapas[-1] = EtapaSetup(
                etapa="Baixando e carregando modelo",
                status="completo",
                detalhe=f"{info.nome} carregado ({info.num_camadas} camadas, {info.num_cabecas} cabeças, d_model={info.d_model})",
            )

            # Etapa 5: Teste de inferência real
            etapas.append(EtapaSetup(
                etapa="Testando inferência",
                status="em_andamento",
                detalhe="Executando teste...",
            ))
            try:
                resultado_teste = await _gerenciador.obter_pesos_atencao_reais(
                    modelo_nome, "Teste de inferência"
                )
                etapas[-1] = EtapaSetup(
                    etapa="Testando inferência",
                    status="completo",
                    detalhe=f"Modelo processou {resultado_teste.num_tokens} tokens com sucesso",
                )
            except Exception as exc:
                logger.warning("Teste de inferência falhou: %s", exc)
                etapas[-1] = EtapaSetup(
                    etapa="Testando inferência",
                    status="completo",
                    detalhe=f"Modelo carregado (teste com aviso: {exc})",
                )

            etapas.append(EtapaSetup(
                etapa="Configuração completa",
                status="completo",
                detalhe=f"Modelo {info.nome} pronto para uso",
            ))

            modelo_dict = {
                "nome": info.nome,
                "carregado": info.carregado,
                "descricao": info.descricao,
                "num_camadas": info.num_camadas,
                "num_cabecas": info.num_cabecas,
                "d_model": info.d_model,
                "tipo": info.tipo,
                "erro": info.erro,
                "mensagem": f"Modelo '{info.nome}' carregado com sucesso.",
            }

            return AutoConfigureResponse(
                sucesso=True,
                modelo_nome=modelo_nome,
                modelo_info=modelo_dict,
                hardware=hardware_resp,
                recomendacao_razao=razao,
                etapas=etapas,
                erro=None,
            )
        else:
            etapas[-1] = EtapaSetup(
                etapa="Baixando e carregando modelo",
                status="erro",
                detalhe=f"Falha: {info.erro}",
            )
            return AutoConfigureResponse(
                sucesso=False,
                modelo_nome=modelo_nome,
                modelo_info=None,
                hardware=hardware_resp,
                recomendacao_razao=razao,
                etapas=etapas,
                erro=info.erro,
            )

    except Exception as exc:
        logger.error("Erro na auto-configuração.", exc_info=True)
        etapas[-1] = EtapaSetup(
            etapa="Baixando e carregando modelo",
            status="erro",
            detalhe=str(exc),
        )
        return AutoConfigureResponse(
            sucesso=False,
            modelo_nome=modelo_nome,
            modelo_info=None,
            hardware=hardware_resp,
            recomendacao_razao=razao,
            etapas=etapas,
            erro=str(exc),
        )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check rápido do backend e dependências."""
    torch_ok = False
    transf_ok = False

    try:
        import torch
        torch_ok = True
    except ImportError:
        pass

    try:
        import transformers
        transf_ok = True
    except ImportError:
        pass

    modelos = _gerenciador.listar_modelos_disponiveis()
    carregado = next((m for m in modelos if m.carregado), None)

    return HealthResponse(
        backend=True,
        torch=torch_ok,
        transformers=transf_ok,
        modelo_carregado=carregado is not None,
        modelo_nome=carregado.nome if carregado else None,
    )
