"""Tests for Task 6: Sparse MoE path correctness and parity.

Verifies that the sparse (grouped token-by-expert) forward path produces
numerically equivalent results to the dense (einsum-all-experts) path,
and that the feature flag correctly switches between them.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from srn_model import GatedExpertMixture, SRNConfig, SRNModel


@pytest.fixture
def moe_config() -> SRNConfig:
    """Small config for MoE tests — dense mode (default)."""
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
        dropout=0.0,  # No dropout for deterministic comparison
        sparse_moe=False,
    )


@pytest.fixture
def sparse_moe_config(moe_config: SRNConfig) -> SRNConfig:
    """Same config but with sparse_moe enabled."""
    return replace(moe_config, sparse_moe=True)


def test_sparse_moe_config_flag_default() -> None:
    """sparse_moe defaults to False in SRNConfig."""
    config = SRNConfig()
    assert config.sparse_moe is False


def test_sparse_moe_config_flag_set() -> None:
    """sparse_moe can be set to True."""
    config = replace(SRNConfig(), sparse_moe=True)
    assert config.sparse_moe is True


def test_dense_sparse_parity(moe_config: SRNConfig) -> None:
    """Dense and sparse paths produce numerically equivalent output.

    Creates a single GEM module, runs the same input through both paths,
    and verifies the outputs match within floating-point tolerance.
    """
    torch.manual_seed(42)
    gem = GatedExpertMixture(moe_config, layer_idx=0)
    gem.eval()

    x = torch.randn(2, 16, moe_config.d_model)  # (B=2, N=16, D=64)

    # Run dense path
    with torch.no_grad():
        dense_config = replace(moe_config, sparse_moe=False)
        gem.config = dense_config
        dense_out, dense_aux = gem(x)

    # Run sparse path (same weights, same input)
    with torch.no_grad():
        sparse_config = replace(moe_config, sparse_moe=True)
        gem.config = sparse_config
        sparse_out, sparse_aux = gem(x)

    # Outputs should be numerically equivalent
    assert torch.allclose(dense_out, sparse_out, rtol=1e-4, atol=1e-5), (
        f"Dense-sparse output mismatch: max diff = "
        f"{(dense_out - sparse_out).abs().max().item():.6e}"
    )

    # Aux loss should be identical (computed from same routing)
    assert torch.allclose(dense_aux, sparse_aux, rtol=1e-5, atol=1e-6), (
        f"Dense-sparse aux loss mismatch: {dense_aux.item():.6e} vs {sparse_aux.item():.6e}"
    )


def test_sparse_parity_different_batch_sizes(moe_config: SRNConfig) -> None:
    """Parity holds across different batch sizes."""
    torch.manual_seed(123)
    gem = GatedExpertMixture(moe_config, layer_idx=0)
    gem.eval()

    for batch_size in [1, 4, 8]:
        x = torch.randn(batch_size, 16, moe_config.d_model)

        with torch.no_grad():
            gem.config = replace(moe_config, sparse_moe=False)
            dense_out, _ = gem(x)

            gem.config = replace(moe_config, sparse_moe=True)
            sparse_out, _ = gem(x)

        assert torch.allclose(dense_out, sparse_out, rtol=1e-4, atol=1e-5), (
            f"Parity failed at batch_size={batch_size}: max diff = "
            f"{(dense_out - sparse_out).abs().max().item():.6e}"
        )


def test_sparse_model_forward_pass(sparse_moe_config: SRNConfig) -> None:
    """Full SRN model forward pass works with sparse_moe=True."""
    torch.manual_seed(42)
    model = SRNModel(sparse_moe_config)
    model.eval()

    x = torch.randint(0, sparse_moe_config.vocab_size, (2, 16))  # (B=2, N=16)

    with torch.no_grad():
        logits, aux_loss = model(x)

    # Basic shape checks
    assert logits.shape == (2, 16, sparse_moe_config.vocab_size)
    assert aux_loss.dim() == 0  # scalar

    # No NaN/Inf
    assert not torch.isnan(logits).any(), "NaN in sparse MoE logits"
    assert not torch.isinf(logits).any(), "Inf in sparse MoE logits"
    assert not torch.isnan(aux_loss), "NaN in sparse MoE aux loss"


def test_sparse_model_backward_pass(sparse_moe_config: SRNConfig) -> None:
    """Backward pass works with sparse_moe=True (gradients flow)."""
    torch.manual_seed(42)
    config = replace(sparse_moe_config, dropout=0.0)
    model = SRNModel(config)
    model.train()

    x = torch.randint(0, config.vocab_size, (2, 16))
    y = torch.randint(0, config.vocab_size, (2, 16))

    logits, aux_loss = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1)
    ) + config.aux_loss_weight * aux_loss

    loss.backward()

    # Check that GEM expert weights received gradients
    for i, layer in enumerate(model.layers):
        gem = layer.gem
        assert gem.W_up.grad is not None, f"Layer {i}: W_up has no gradient"
        assert gem.W_down.grad is not None, f"Layer {i}: W_down has no gradient"
        assert gem.router.weight.grad is not None, f"Layer {i}: router has no gradient"

        # Gradients should be non-zero (not all dead)
        assert gem.W_up.grad.abs().sum() > 0, f"Layer {i}: W_up gradient is all zeros"
        assert gem.W_down.grad.abs().sum() > 0, f"Layer {i}: W_down gradient is all zeros"


def test_sparse_full_model_parity(moe_config: SRNConfig) -> None:
    """Full model produces same output in dense vs sparse mode."""
    torch.manual_seed(42)

    # Build model in dense mode
    model = SRNModel(moe_config)
    model.eval()

    x = torch.randint(0, moe_config.vocab_size, (2, 16))

    with torch.no_grad():
        dense_logits, dense_aux = model(x)

    # Switch to sparse mode (same weights)
    sparse_config = replace(moe_config, sparse_moe=True)
    model.config = sparse_config
    for layer in model.layers:
        layer.gem.config = sparse_config

    with torch.no_grad():
        sparse_logits, sparse_aux = model(x)

    assert torch.allclose(dense_logits, sparse_logits, rtol=1e-4, atol=1e-5), (
        f"Full model parity failed: max diff = "
        f"{(dense_logits - sparse_logits).abs().max().item():.6e}"
    )


def test_sparse_generation(sparse_moe_config: SRNConfig) -> None:
    """Model can generate tokens with sparse_moe=True."""
    torch.manual_seed(42)
    model = SRNModel(sparse_moe_config)
    model.eval()

    prompt = torch.zeros(1, 1, dtype=torch.long)
    generated = model.generate(prompt, max_tokens=20, temperature=0.8)

    assert generated.shape[1] == 21  # 1 prompt + 20 generated
    assert generated.min() >= 0
    assert generated.max() < sparse_moe_config.vocab_size
