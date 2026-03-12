from __future__ import annotations

from srn_model import SRNConfig


def test_sample_config_is_valid(sample_config: SRNConfig) -> None:
    assert sample_config.d_model % sample_config.n_heads_route == 0
    assert sample_config.top_k_experts <= sample_config.n_experts
    assert sample_config.max_seq_len > 0


def test_seed_fixture_is_stable(rng_seed: int) -> None:
    assert rng_seed == 1234
