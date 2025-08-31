"""Pytest configuration and fixtures for DEWI tests."""

import os
from pathlib import Path
from typing import Generator
import numpy as np
import pytest

# Enable test mode for all tests
pytest_plugins = []

@pytest.fixture(autouse=True)
def set_test_env(monkeypatch):
    """Set test environment variables."""
    monkeypatch.setenv("DEWI_TEST_MODE", "1")

@pytest.fixture
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture
def tmp_dir(tmp_path):
    """Return a temporary directory for test output."""
    return tmp_path

@pytest.fixture
def dummy_embeddings():
    """Generate dummy embeddings for testing."""
    def _generate(n: int = 10, dim: int = 128) -> np.ndarray:
        return np.random.randn(n, dim).astype(np.float32)
    return _generate

@pytest.fixture
def dummy_payloads():
    """Generate dummy payloads for testing."""
    def _generate(n: int = 10):
        return [
            {
                "dewi": float(np.clip(np.random.beta(2, 2), 0, 1)),
                "ht_mean": float(np.random.gamma(2, 0.5)),
                "ht_q90": float(np.random.gamma(2, 0.5) * 1.5),
                "hi_mean": float(np.random.gamma(2, 0.3)),
                "hi_q90": float(np.random.gamma(2, 0.3) * 1.5),
                "I_hat": float(np.random.beta(2, 2)),
                "redundancy": float(np.random.beta(1, 5)),
                "noise": float(np.random.beta(1, 10))
            }
            for _ in range(n)
        ]
    return _generate
