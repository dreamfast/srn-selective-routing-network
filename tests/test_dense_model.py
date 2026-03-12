"""Tests for Task 8: Dense GPT baseline model.

Verifies the DenseGPT model works correctly and exposes the same interface
as SRNModel for use in the shared training harness.
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

from dense_model import DenseConfig, DenseGPT


@pytest.fixture
def dense_config() -> DenseConfig:
    """Small dense config for fast unit tests."""
    return DenseConfig(
        vocab_size=128,
        max_seq_len=32,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        dropout=0.0,
        bias=False,
    )


def test_dense_config_defaults() -> None:
    """DenseConfig has sensible defaults."""
    config = DenseConfig()
    assert config.vocab_size == 65
    assert config.d_model == 1024
    assert config.n_layers == 16
    assert config.n_heads == 16
    assert config.d_ff == 4096
    assert config.bias is False


def test_dense_forward_pass(dense_config: DenseConfig) -> None:
    """Forward pass produces correct output shape and no NaN/Inf."""
    model = DenseGPT(dense_config)
    model.eval()

    x = torch.randint(0, dense_config.vocab_size, (2, 16))  # (B=2, N=16)

    with torch.no_grad():
        logits, aux_loss = model(x)

    # Shape checks
    assert logits.shape == (2, 16, dense_config.vocab_size)
    assert aux_loss.dim() == 0  # scalar
    assert aux_loss.item() == 0.0  # Dense model has no aux loss

    # No NaN/Inf
    assert not torch.isnan(logits).any(), "NaN in logits"
    assert not torch.isinf(logits).any(), "Inf in logits"


def test_dense_backward_pass(dense_config: DenseConfig) -> None:
    """Backward pass works and gradients flow to all parameters."""
    model = DenseGPT(dense_config)
    model.train()

    x = torch.randint(0, dense_config.vocab_size, (2, 16))
    y = torch.randint(0, dense_config.vocab_size, (2, 16))

    logits, aux_loss = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), y.view(-1)
    )
    loss.backward()

    # Check key parameters have gradients
    assert model.token_emb.weight.grad is not None
    assert model.layers[0].attn.qkv.weight.grad is not None
    assert model.layers[0].ffn.up.weight.grad is not None


def test_dense_causal_masking(dense_config: DenseConfig) -> None:
    """Modifying a future token doesn't affect past logits (causality)."""
    torch.manual_seed(42)
    model = DenseGPT(dense_config)
    model.eval()

    x = torch.randint(0, dense_config.vocab_size, (1, 16))

    with torch.no_grad():
        logits1, _ = model(x)

        # Modify token at position 8
        x_mod = x.clone()
        x_mod[0, 8] = (x[0, 8] + 1) % dense_config.vocab_size
        logits2, _ = model(x_mod)

    # Positions before 8 should be unaffected
    assert torch.allclose(logits1[0, :8], logits2[0, :8], atol=1e-5), (
        f"Causal masking violated: max diff = "
        f"{(logits1[0, :8] - logits2[0, :8]).abs().max().item():.6e}"
    )


def test_dense_generation(dense_config: DenseConfig) -> None:
    """Model can generate tokens autoregressively."""
    torch.manual_seed(42)
    model = DenseGPT(dense_config)
    model.eval()

    prompt = torch.zeros(1, 1, dtype=torch.long)
    generated = model.generate(prompt, max_tokens=20, temperature=0.8)

    assert generated.shape[1] == 21  # 1 prompt + 20 generated
    assert generated.min() >= 0
    assert generated.max() < dense_config.vocab_size


def test_dense_count_params(dense_config: DenseConfig) -> None:
    """count_params and count_active_params are equal for dense models."""
    model = DenseGPT(dense_config)
    total = model.count_params()
    active = model.count_active_params()

    assert total == active, "Dense model should have total == active params"
    assert total > 0


def test_dense_weight_tying(dense_config: DenseConfig) -> None:
    """LM head shares weights with token embedding."""
    model = DenseGPT(dense_config)
    assert model.lm_head.weight is model.token_emb.weight


def test_dense_interface_matches_srn(dense_config: DenseConfig) -> None:
    """DenseGPT exposes the same interface as SRNModel."""
    model = DenseGPT(dense_config)

    # Check required methods exist
    assert hasattr(model, "forward")
    assert hasattr(model, "generate")
    assert hasattr(model, "count_params")
    assert hasattr(model, "count_active_params")

    # Check forward returns (logits, aux_loss) tuple
    x = torch.randint(0, dense_config.vocab_size, (1, 8))
    with torch.no_grad():
        result = model(x)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_dense_valid_probabilities(dense_config: DenseConfig) -> None:
    """Softmax of logits produces valid probability distributions."""
    model = DenseGPT(dense_config)
    model.eval()

    x = torch.randint(0, dense_config.vocab_size, (2, 16))
    with torch.no_grad():
        logits, _ = model(x)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)


def test_dense_config_yaml_exists() -> None:
    """Dense baseline config YAML exists with model_type field."""
    from omegaconf import OmegaConf

    config_path = ROOT / "configs" / "dense-067b.yaml"
    assert config_path.exists(), "configs/dense-067b.yaml not found"

    cfg = OmegaConf.load(config_path)
    assert cfg.train.model_type == "dense"
    assert cfg.model.n_heads > 0
    assert cfg.model.d_ff > 0
