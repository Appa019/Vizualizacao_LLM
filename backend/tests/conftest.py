"""Shared test fixtures for the backend test suite."""

import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import pytest
from fastapi.testclient import TestClient
from main import app


@pytest.fixture
def client():
    """Provide a FastAPI TestClient for endpoint tests."""
    return TestClient(app)


@pytest.fixture
def fresh_training_state():
    """Reset global training state before and after each test."""
    import routers.training as training_mod
    training_mod._rede_global = None
    yield
    training_mod._rede_global = None


@pytest.fixture
def reset_model_manager():
    """Reset the model manager singleton cache before and after each test."""
    from core.model_manager import get_model_manager
    mgr = get_model_manager()
    mgr._cache.clear()
    mgr._tokenizadores.clear()
    yield
    mgr._cache.clear()
    mgr._tokenizadores.clear()
