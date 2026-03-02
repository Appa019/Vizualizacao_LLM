"""Detecção de hardware e recomendação de modelo."""

from __future__ import annotations

import logging
import os
import platform
import sys

logger = logging.getLogger(__name__)


def detectar_hardware() -> dict:
    """Detecta informações de hardware do servidor.

    Returns:
        Dicionário com detalhes de CPU, RAM, GPU e dependências.
    """
    info: dict = {
        "cpu": platform.processor() or platform.machine(),
        "nucleos": os.cpu_count() or 1,
        "ram_total_gb": 0.0,
        "ram_disponivel_gb": 0.0,
        "gpu": None,
        "gpu_disponivel": False,
        "sistema": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "torch_instalado": False,
        "torch_version": None,
        "transformers_instalado": False,
        "transformers_version": None,
    }

    # RAM via psutil (preferido) ou /proc/meminfo (fallback Linux)
    try:
        import psutil

        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024**3), 1)
        info["ram_disponivel_gb"] = round(mem.available / (1024**3), 1)
    except ImportError:
        info.update(_ram_fallback())

    # Torch
    try:
        import torch

        info["torch_instalado"] = True
        info["torch_version"] = torch.__version__

        if torch.cuda.is_available():
            info["gpu_disponivel"] = True
            try:
                props = torch.cuda.get_device_properties(0)
                vram_gb = round(props.total_mem / (1024**3), 1)
                info["gpu"] = f"{props.name} ({vram_gb} GB VRAM)"
            except Exception:
                info["gpu"] = "CUDA disponível"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["gpu_disponivel"] = True
            info["gpu"] = "Apple MPS"
    except ImportError:
        pass

    # Transformers
    try:
        import transformers

        info["transformers_instalado"] = True
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    return info


def _ram_fallback() -> dict:
    """Fallback para leitura de RAM via /proc/meminfo (Linux)."""
    result = {"ram_total_gb": 0.0, "ram_disponivel_gb": 0.0}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                if parts[0] == "MemTotal:":
                    result["ram_total_gb"] = round(int(parts[1]) / (1024**2), 1)
                elif parts[0] == "MemAvailable:":
                    result["ram_disponivel_gb"] = round(int(parts[1]) / (1024**2), 1)
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return result


def recomendar_modelo(hardware: dict) -> tuple[str, str]:
    """Recomenda o melhor modelo baseado no hardware.

    Args:
        hardware: Dicionário retornado por detectar_hardware().

    Returns:
        Tupla (nome_modelo, razão da recomendação).
    """
    ram = hardware.get("ram_disponivel_gb", 0)
    gpu = hardware.get("gpu_disponivel", False)
    torch_ok = hardware.get("torch_instalado", False)
    transformers_ok = hardware.get("transformers_instalado", False)

    if not torch_ok or not transformers_ok:
        return (
            "simulacao",
            "torch ou transformers não instalados — usando modo simulação com dados sintéticos",
        )

    if ram < 4:
        return (
            "simulacao",
            f"RAM disponível ({ram:.1f} GB) insuficiente para modelos reais — usando simulação",
        )

    gpu_info = hardware.get("gpu") or ""
    gpu_vram = 0.0
    if "VRAM" in gpu_info:
        try:
            gpu_vram = float(gpu_info.split("(")[1].split(" GB")[0])
        except (IndexError, ValueError):
            pass

    if ram >= 16 or gpu_vram >= 4:
        return (
            "gpt2",
            f"Hardware potente (RAM: {ram:.1f} GB"
            + (f", GPU: {gpu_info}" if gpu else "")
            + ") — GPT-2 permite demonstrar geração de texto autoregressiva",
        )

    if ram >= 8 or gpu:
        return (
            "bert-base-uncased",
            f"Hardware adequado (RAM: {ram:.1f} GB) — BERT base oferece 12 camadas de atenção para análise completa",
        )

    return (
        "distilbert-base-uncased",
        f"RAM limitada ({ram:.1f} GB) — DistilBERT é 40% menor e 60% mais rápido, ideal para este hardware",
    )


def verificar_dependencias() -> dict:
    """Verifica status das dependências necessárias.

    Returns:
        Dicionário com status de cada dependência.
    """
    deps: dict = {}

    for nome in ("torch", "transformers", "numpy", "scipy", "sklearn"):
        try:
            mod = __import__(nome if nome != "sklearn" else "sklearn")
            deps[nome] = {"instalado": True, "versao": getattr(mod, "__version__", "?")}
        except ImportError:
            deps[nome] = {"instalado": False, "versao": None}

    return deps
