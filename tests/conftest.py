from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srn_model import SRNConfig


@pytest.fixture(scope="session")
def sample_config() -> SRNConfig:
    """Small deterministic config for fast unit tests."""
    return replace(
        SRNConfig(),
        vocab_size=128,
        max_seq_len=32,
        d_model=64,
        d_compressed=16,
        n_layers=2,
        n_memory_slots=8,
        n_experts=4,
        top_k_experts=2,
        d_expert=32,
        n_heads_route=4,
        causal_window=8,
    )


@pytest.fixture(scope="session")
def rng_seed() -> int:
    return 1234


@pytest.fixture(scope="session")
def numpy_rng(rng_seed: int) -> np.random.Generator:
    return np.random.default_rng(rng_seed)
