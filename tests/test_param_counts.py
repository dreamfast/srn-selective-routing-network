from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def _estimate_params(
    d_model: int,
    d_compressed: int,
    n_layers: int,
    n_memory_slots: int,
    n_experts: int,
    d_expert: int,
    max_seq_len: int,
    vocab_size: int,
) -> int:
    # Per-layer blocks
    dsr = (
        3 * (d_model**2)
        + 2 * d_model * n_memory_slots
        + 3 * d_model
        + n_memory_slots
        + max_seq_len * n_memory_slots
    )
    csp = (
        d_model**2
        + 3 * d_model * d_compressed
        + d_compressed**2
        + 2 * d_compressed
        + 2 * d_model
    )
    gem = n_experts * (2 * d_model * d_expert + 2 * d_model + d_expert + 1)
    ln = 6 * d_model
    per_layer = dsr + csp + gem + ln

    # Global embeddings and final LN
    global_params = (vocab_size + max_seq_len + 2) * d_model
    return n_layers * per_layer + global_params


def test_param_count_matches_expected_150m_band() -> None:
    cfg = OmegaConf.to_container(
        OmegaConf.load(Path("configs") / "srn-150m.yaml"), resolve=True
    )
    model = cfg["model"]
    estimated = _estimate_params(
        d_model=model["d_model"],
        d_compressed=model["d_compressed"],
        n_layers=model["n_layers"],
        n_memory_slots=model["n_memory_slots"],
        n_experts=model["n_experts"],
        d_expert=model["d_expert"],
        max_seq_len=model["max_seq_len"],
        vocab_size=32000,
    )

    assert 130_000_000 <= estimated <= 170_000_000
