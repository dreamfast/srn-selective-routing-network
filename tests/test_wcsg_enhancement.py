"""Tests for Task 3: WCSG Score-Space Offset Enhancement.

Verifies that the low-rank score-space offset in DynamicSparseRouter
works correctly when enabled, preserves behavior when disabled, and
maintains causality through the gated application.
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

from srn_model import SRNConfig, SRNModel, DynamicSparseRouter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wcsg_config() -> SRNConfig:
    """Small config with WCSG offset enabled."""
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
        dropout=0.0,
        wcsg_key_offset=True,
        wcsg_key_offset_rank=16,
    )


@pytest.fixture
def no_wcsg_config() -> SRNConfig:
    """Small config with WCSG offset disabled (default)."""
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
        dropout=0.0,
        wcsg_key_offset=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_wcsg_forward(wcsg_config: SRNConfig) -> None:
    """Model with WCSG offset produces correct output shape."""
    model = SRNModel(wcsg_config)
    model.eval()

    B, N = 2, 16
    x = torch.randint(0, wcsg_config.vocab_size, (B, N))

    with torch.no_grad():
        logits, aux_loss = model(x)

    assert logits.shape == (B, N, wcsg_config.vocab_size)
    assert aux_loss.dim() == 0  # scalar


def test_wcsg_no_nan(wcsg_config: SRNConfig) -> None:
    """Model with WCSG offset produces no NaN or Inf."""
    torch.manual_seed(42)
    model = SRNModel(wcsg_config)
    model.eval()

    x = torch.randint(0, wcsg_config.vocab_size, (2, 16))

    with torch.no_grad():
        logits, aux_loss = model(x)

    assert not torch.isnan(logits).any(), "NaN in logits with WCSG offset"
    assert not torch.isinf(logits).any(), "Inf in logits with WCSG offset"
    assert not torch.isnan(aux_loss), "NaN in aux_loss with WCSG offset"


def test_wcsg_causal(wcsg_config: SRNConfig) -> None:
    """WCSG offset preserves causality: future tokens don't affect past logits.

    The offset uses the same causal gate (derived from causal_windowed_mean),
    so it must not leak future information.
    """
    torch.manual_seed(42)
    model = SRNModel(wcsg_config)
    model.eval()

    x = torch.randint(0, wcsg_config.vocab_size, (1, 16))

    with torch.no_grad():
        logits1, _ = model(x)

        # Modify token at position 8
        x_mod = x.clone()
        x_mod[0, 8] = (x[0, 8] + 1) % wcsg_config.vocab_size
        logits2, _ = model(x_mod)

    # Positions before 8 should be unaffected
    assert torch.allclose(logits1[0, :8], logits2[0, :8], atol=1e-5), (
        f"Causal masking violated with WCSG offset: max diff = "
        f"{(logits1[0, :8] - logits2[0, :8]).abs().max().item():.6e}"
    )


def test_wcsg_disabled_unchanged(no_wcsg_config: SRNConfig) -> None:
    """When wcsg_key_offset=False (default), no offset modules exist."""
    model = SRNModel(no_wcsg_config)

    for i, layer in enumerate(model.layers):
        router = layer.router
        assert router is not None, f"Layer {i} should have a router"
        assert router.W_offset_down is None, (
            f"Layer {i}: W_offset_down should be None when WCSG offset disabled"
        )
        assert router.W_offset_up is None, (
            f"Layer {i}: W_offset_up should be None when WCSG offset disabled"
        )

    # Verify forward still works
    model.eval()
    x = torch.randint(0, no_wcsg_config.vocab_size, (2, 16))
    with torch.no_grad():
        logits, _ = model(x)
    assert logits.shape == (2, 16, no_wcsg_config.vocab_size)
    assert not torch.isnan(logits).any()


def test_wcsg_param_count(
    wcsg_config: SRNConfig,
    no_wcsg_config: SRNConfig,
) -> None:
    """WCSG offset adds the expected number of parameters.

    With d=64, rank=16, k=8:
    - W_offset_down: 64*16 + 16 (weight + bias) = 1040
    - W_offset_up: 16*8 + 8 (weight + bias) = 136
    - Per layer: 1176
    - Total across 2 layers: 2352
    """
    wcsg_model = SRNModel(wcsg_config)
    no_wcsg_model = SRNModel(no_wcsg_config)

    wcsg_total = wcsg_model.count_params()
    no_wcsg_total = no_wcsg_model.count_params()

    # WCSG should have more params
    param_diff = wcsg_total - no_wcsg_total
    assert param_diff > 0, (
        f"WCSG model ({wcsg_total}) should have more params than "
        f"non-WCSG ({no_wcsg_total})"
    )

    # Verify the difference matches expected
    d = wcsg_config.d_model
    rank = wcsg_config.wcsg_key_offset_rank
    k = wcsg_config.n_memory_slots
    n_layers = wcsg_config.n_layers
    expected_per_layer = (d * rank + rank) + (rank * k + k)
    expected_total = expected_per_layer * n_layers
    assert param_diff == expected_total, (
        f"Param diff ({param_diff}) doesn't match expected ({expected_total}). "
        f"Per layer: {expected_per_layer}"
    )

    # Active param count should also reflect the difference
    wcsg_active = wcsg_model.count_active_params()
    no_wcsg_active = no_wcsg_model.count_active_params()
    assert wcsg_active > no_wcsg_active


def test_wcsg_near_zero_init(wcsg_config: SRNConfig) -> None:
    """WCSG offset weights are initialized near zero (std=0.001).

    This ensures the offset starts negligible and gradually learns,
    preventing it from disrupting pre-trained routing patterns.
    """
    model = SRNModel(wcsg_config)

    for i, layer in enumerate(model.layers):
        router = layer.router
        assert router is not None

        # Check W_offset_down weight std is near 0.001
        down_std = router.W_offset_down.weight.std().item()
        assert 0.0005 < down_std < 0.002, (
            f"Layer {i} W_offset_down std={down_std:.6f}, expected ~0.001"
        )

        # Check W_offset_up weight std is near 0.001
        up_std = router.W_offset_up.weight.std().item()
        assert 0.0005 < up_std < 0.002, (
            f"Layer {i} W_offset_up std={up_std:.6f}, expected ~0.001"
        )

        # Check biases are zero
        assert torch.allclose(
            router.W_offset_down.bias, torch.zeros_like(router.W_offset_down.bias)
        ), f"Layer {i} W_offset_down bias should be zero"
        assert torch.allclose(
            router.W_offset_up.bias, torch.zeros_like(router.W_offset_up.bias)
        ), f"Layer {i} W_offset_up bias should be zero"
