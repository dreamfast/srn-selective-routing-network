from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

from srn_model import SRNConfig
from train import parse_args, validate_srn_config


def test_yaml_config_overrides_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
train:
  seq_len: 128
model:
  max_seq_len: 128
  d_model: 256
  n_heads_route: 8
  n_experts: 4
  top_k_experts: 2
""".strip()
    )

    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    args = parse_args()

    assert args.seq_len == 128
    assert args.max_seq_len == 128
    assert args.d_model == 256
    assert args.n_heads_route == 8
    assert args.n_experts == 4
    assert args.top_k_experts == 2


def test_invalid_head_divisibility_raises() -> None:
    bad = SRNConfig(d_model=257, n_heads_route=8)
    with pytest.raises(ValueError, match="divisible"):
        validate_srn_config(bad, seq_len=128)


# ---------------------------------------------------------------------------
# Task 4: New ablation config field tests
# ---------------------------------------------------------------------------

def test_ablation_defaults() -> None:
    """New ablation fields have backward-compatible defaults."""
    config = SRNConfig()
    assert config.attention_every_n_layers == 0
    assert config.attention_n_heads == 8
    assert config.disable_csp is False
    assert config.wcsg_key_offset is False
    assert config.wcsg_key_offset_rank == 32


def test_attention_head_divisibility_raises() -> None:
    """attention_n_heads must divide d_model when hybrid attention is enabled."""
    bad = SRNConfig(d_model=512, attention_every_n_layers=4, attention_n_heads=7)
    with pytest.raises(ValueError, match="attention_n_heads"):
        validate_srn_config(bad, seq_len=128)


def test_wcsg_rank_validation_raises() -> None:
    """wcsg_key_offset_rank must be > 0 and < d_model."""
    # rank <= 0
    bad_zero = SRNConfig(wcsg_key_offset_rank=0)
    with pytest.raises(ValueError, match="wcsg_key_offset_rank"):
        validate_srn_config(bad_zero, seq_len=128)

    # rank >= d_model
    bad_large = SRNConfig(d_model=512, wcsg_key_offset_rank=512)
    with pytest.raises(ValueError, match="wcsg_key_offset_rank"):
        validate_srn_config(bad_large, seq_len=128)


def test_ablation_yaml_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ablation fields can be set via YAML config file."""
    config_path = tmp_path / "ablation.yaml"
    config_path.write_text(
        """
model:
  attention_every_n_layers: 4
  attention_n_heads: 8
  disable_csp: true
  wcsg_key_offset: true
  wcsg_key_offset_rank: 16
""".strip()
    )

    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    args = parse_args()

    assert args.attention_every_n_layers == 4
    assert args.attention_n_heads == 8
    assert args.disable_csp is True
    assert args.wcsg_key_offset is True
    assert args.wcsg_key_offset_rank == 16


def test_attention_exceeds_layers_warns() -> None:
    """Warning when attention_every_n_layers > n_layers (no attention layers created)."""
    config = SRNConfig(
        n_layers=8,
        attention_every_n_layers=16,
        attention_n_heads=8,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validate_srn_config(config, seq_len=128)
        assert len(w) == 1
        assert "no attention layers" in str(w[0].message).lower()


def test_hybrid_with_dense_model_type(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hybrid attention flags are ignored when model_type=dense (no crash)."""
    config_path = tmp_path / "dense_hybrid.yaml"
    config_path.write_text(
        """
train:
  model_type: dense
model:
  attention_every_n_layers: 4
  d_model: 64
  n_heads: 4
""".strip()
    )

    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    args = parse_args()

    # These should be set but not cause errors (dense model ignores them)
    assert args.model_type == "dense"
    assert args.attention_every_n_layers == 4
