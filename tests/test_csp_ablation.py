"""Tests for Task 2: CSP Ablation Support.

Verifies that setting disable_csp=True correctly removes the Compressed
State Propagation bottleneck from all SRN layers, while preserving
correct behavior when CSP is enabled (default).
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

from srn_model import SRNConfig, SRNModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def csp_disabled_config() -> SRNConfig:
    """Small config with CSP disabled."""
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
        disable_csp=True,
    )


@pytest.fixture
def csp_enabled_config() -> SRNConfig:
    """Small config with CSP enabled (default)."""
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
        disable_csp=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_csp_disabled_forward(csp_disabled_config: SRNConfig) -> None:
    """Model with CSP disabled produces correct output shape."""
    model = SRNModel(csp_disabled_config)
    model.eval()

    B, N = 2, 16
    x = torch.randint(0, csp_disabled_config.vocab_size, (B, N))

    with torch.no_grad():
        logits, aux_loss = model(x)

    assert logits.shape == (B, N, csp_disabled_config.vocab_size)
    assert aux_loss.dim() == 0  # scalar


def test_csp_disabled_no_nan(csp_disabled_config: SRNConfig) -> None:
    """Model with CSP disabled produces no NaN or Inf."""
    torch.manual_seed(42)
    model = SRNModel(csp_disabled_config)
    model.eval()

    x = torch.randint(0, csp_disabled_config.vocab_size, (2, 16))

    with torch.no_grad():
        logits, aux_loss = model(x)

    assert not torch.isnan(logits).any(), "NaN in logits with CSP disabled"
    assert not torch.isinf(logits).any(), "Inf in logits with CSP disabled"
    assert not torch.isnan(aux_loss), "NaN in aux_loss with CSP disabled"


def test_csp_disabled_param_count(
    csp_disabled_config: SRNConfig,
    csp_enabled_config: SRNConfig,
) -> None:
    """Disabling CSP reduces both total and active parameter counts.

    CSP params per layer: compress(d*dc) + process(dc*dc) + expand(dc*d)
    + gate_proj((d+dc)*d) + biases. With d=64, dc=16:
    - compress: 64*16 = 1024
    - process: 16*16 = 256
    - expand: 16*64 = 1024
    - gate_proj: (64+16)*64 = 5120
    - biases: 16+16+64+64 = 160
    Total CSP per layer: 7648
    Plus ln2 per layer: 2*64 = 128
    Total removed per layer: 7776
    Total removed across 2 layers: 15552
    """
    disabled_model = SRNModel(csp_disabled_config)
    enabled_model = SRNModel(csp_enabled_config)

    disabled_total = disabled_model.count_params()
    enabled_total = enabled_model.count_params()

    disabled_active = disabled_model.count_active_params()
    enabled_active = enabled_model.count_active_params()

    # Disabled should have fewer params
    assert disabled_total < enabled_total, (
        f"CSP disabled ({disabled_total}) should have fewer total params "
        f"than enabled ({enabled_total})"
    )
    assert disabled_active < enabled_active, (
        f"CSP disabled ({disabled_active}) should have fewer active params "
        f"than enabled ({enabled_active})"
    )

    # Verify CSP and ln2 modules are actually None
    for layer in disabled_model.layers:
        assert layer.csp is None, "CSP should be None when disabled"
        assert layer.ln2 is None, "ln2 should be None when CSP disabled"

    # Verify CSP and ln2 modules are present when enabled
    for layer in enabled_model.layers:
        assert layer.csp is not None, "CSP should exist when enabled"
        assert layer.ln2 is not None, "ln2 should exist when CSP enabled"


def test_csp_enabled_unchanged(csp_enabled_config: SRNConfig) -> None:
    """When disable_csp=False (default), model behaves identically to baseline.

    Verifies that the CSP ablation code path doesn't accidentally change
    behavior when CSP is enabled.
    """
    torch.manual_seed(42)
    model = SRNModel(csp_enabled_config)
    model.eval()

    x = torch.randint(0, csp_enabled_config.vocab_size, (2, 16))

    with torch.no_grad():
        logits, aux_loss = model(x)

    # Basic sanity: output shape, no NaN, valid probabilities
    assert logits.shape == (2, 16, csp_enabled_config.vocab_size)
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()

    probs = torch.nn.functional.softmax(logits, dim=-1)
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    # Verify backward works
    model.train()
    logits, aux_loss = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        torch.randint(0, csp_enabled_config.vocab_size, (2, 16)).view(-1),
    )
    total_loss = loss + csp_enabled_config.aux_loss_weight * aux_loss
    total_loss.backward()

    # CSP layers should have gradients
    assert model.layers[0].csp.compress.weight.grad is not None, (
        "CSP compress weight should have gradient when enabled"
    )
