"""Tests for Task 1: Hybrid Attention Support in SRNModel.

Verifies that SRNLayer correctly substitutes CausalSelfAttention for DSR
on every Nth layer when attention_every_n_layers > 0, while preserving
pure SRN behavior when the flag is 0.
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

from srn_model import SRNConfig, SRNModel, SRNLayer, DynamicSparseRouter
from dense_model import CausalSelfAttention


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hybrid_config() -> SRNConfig:
    """Small config with hybrid attention every 2nd layer (4 layers total)."""
    return replace(
        SRNConfig(),
        vocab_size=128,
        max_seq_len=32,
        d_model=64,
        d_compressed=16,
        n_layers=4,
        n_memory_slots=8,
        n_experts=4,
        top_k_experts=2,
        d_expert=32,
        n_heads_route=4,
        causal_window=8,
        dropout=0.0,
        attention_every_n_layers=2,
        attention_n_heads=4,
    )


@pytest.fixture
def pure_srn_config() -> SRNConfig:
    """Small config with pure SRN (no attention layers)."""
    return replace(
        SRNConfig(),
        vocab_size=128,
        max_seq_len=32,
        d_model=64,
        d_compressed=16,
        n_layers=4,
        n_memory_slots=8,
        n_experts=4,
        top_k_experts=2,
        d_expert=32,
        n_heads_route=4,
        causal_window=8,
        dropout=0.0,
        attention_every_n_layers=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_hybrid_forward_shape(hybrid_config: SRNConfig) -> None:
    """Hybrid model produces correct output shape."""
    model = SRNModel(hybrid_config)
    model.eval()

    B, N = 2, 16
    x = torch.randint(0, hybrid_config.vocab_size, (B, N))

    with torch.no_grad():
        logits, aux_loss = model(x)

    assert logits.shape == (B, N, hybrid_config.vocab_size)
    assert aux_loss.dim() == 0  # scalar


def test_hybrid_no_nan(hybrid_config: SRNConfig) -> None:
    """Hybrid model produces no NaN or Inf in logits."""
    torch.manual_seed(42)
    model = SRNModel(hybrid_config)
    model.eval()

    x = torch.randint(0, hybrid_config.vocab_size, (2, 16))

    with torch.no_grad():
        logits, aux_loss = model(x)

    assert not torch.isnan(logits).any(), "NaN in hybrid model logits"
    assert not torch.isinf(logits).any(), "Inf in hybrid model logits"
    assert not torch.isnan(aux_loss), "NaN in hybrid model aux_loss"


def test_hybrid_causal_masking(hybrid_config: SRNConfig) -> None:
    """Modifying a future token doesn't affect past logits (causality).

    This is critical: both attention layers and DSR layers must be causal.
    """
    torch.manual_seed(42)
    model = SRNModel(hybrid_config)
    model.eval()

    x = torch.randint(0, hybrid_config.vocab_size, (1, 16))

    with torch.no_grad():
        logits1, _ = model(x)

        # Modify token at position 8
        x_mod = x.clone()
        x_mod[0, 8] = (x[0, 8] + 1) % hybrid_config.vocab_size
        logits2, _ = model(x_mod)

    # Positions before 8 should be unaffected
    assert torch.allclose(logits1[0, :8], logits2[0, :8], atol=1e-5), (
        f"Causal masking violated in hybrid model: max diff = "
        f"{(logits1[0, :8] - logits2[0, :8]).abs().max().item():.6e}"
    )


def test_hybrid_layer_types(hybrid_config: SRNConfig) -> None:
    """Correct layers use attention vs DSR.

    With attention_every_n_layers=2 and n_layers=4:
    - Layer 0: attention (0 % 2 == 0)
    - Layer 1: DSR
    - Layer 2: attention (2 % 2 == 0)
    - Layer 3: DSR
    """
    model = SRNModel(hybrid_config)

    for i, layer in enumerate(model.layers):
        assert isinstance(layer, SRNLayer)
        if i % 2 == 0:
            # Attention layer
            assert layer.uses_attention is True
            assert layer.attn is not None
            assert isinstance(layer.attn, CausalSelfAttention)
            assert layer.router is None
        else:
            # DSR layer
            assert layer.uses_attention is False
            assert layer.router is not None
            assert isinstance(layer.router, DynamicSparseRouter)
            assert layer.attn is None


def test_pure_srn_unchanged(pure_srn_config: SRNConfig) -> None:
    """When attention_every_n_layers=0, all layers use DSR (no attention)."""
    model = SRNModel(pure_srn_config)

    for i, layer in enumerate(model.layers):
        assert layer.uses_attention is False, f"Layer {i} should not use attention"
        assert layer.router is not None, f"Layer {i} should have DSR router"
        assert layer.attn is None, f"Layer {i} should not have attention module"


def test_hybrid_aux_loss(hybrid_config: SRNConfig) -> None:
    """Aux loss is accumulated from GEM in all layers (attention and DSR).

    Both attention and DSR layers have GEM, so aux_loss should be non-zero
    and come from all 4 layers.
    """
    torch.manual_seed(42)
    model = SRNModel(hybrid_config)
    model.train()

    x = torch.randint(0, hybrid_config.vocab_size, (2, 16))
    logits, aux_loss = model(x)

    # Aux loss should be positive (from GEM load balancing in all layers)
    assert aux_loss.item() > 0, "Aux loss should be positive from GEM"

    # Verify backward works through hybrid model
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        torch.randint(0, hybrid_config.vocab_size, (2, 16)).view(-1),
    )
    total_loss = loss + hybrid_config.aux_loss_weight * aux_loss
    total_loss.backward()

    # Check gradients flow to both attention and DSR layers
    attn_layer = model.layers[0]  # attention layer
    dsr_layer = model.layers[1]   # DSR layer
    assert attn_layer.attn.qkv.weight.grad is not None, "No grad in attention layer"
    assert dsr_layer.router.W_q.weight.grad is not None, "No grad in DSR layer"


def test_hybrid_param_count(hybrid_config: SRNConfig) -> None:
    """Active param count changes correctly with hybrid attention.

    Attention layers have different param counts than DSR layers.
    The count_active_params() method must account for this.
    """
    # Hybrid model
    hybrid_model = SRNModel(hybrid_config)
    hybrid_active = hybrid_model.count_active_params()

    # Pure SRN model (same config but no attention)
    pure_config = replace(hybrid_config, attention_every_n_layers=0)
    pure_model = SRNModel(pure_config)
    pure_active = pure_model.count_active_params()

    # They should differ (attention layers have different param count than DSR)
    assert hybrid_active != pure_active, (
        f"Hybrid ({hybrid_active}) and pure SRN ({pure_active}) should have "
        f"different active param counts"
    )

    # Both should be positive and reasonable
    assert hybrid_active > 0
    assert pure_active > 0

    # Verify count_active_params is in the right ballpark vs actual params
    hybrid_total = hybrid_model.count_params()
    assert hybrid_active <= hybrid_total, (
        f"Active params ({hybrid_active}) should not exceed total ({hybrid_total})"
    )
