"""Shared pytest fixtures."""
import pytest

from ._ma_parity_helpers import DEVICES


@pytest.fixture(params=DEVICES)
def dev(request):
    """Device the XuPy data lives on: "cpu" (numpy) or "gpu" (cupy)."""
    return request.param
