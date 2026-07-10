from __future__ import annotations

import pytest


@pytest.fixture
def benchmark_params() -> dict[str, float]:
    return {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.20}
