from __future__ import annotations

import sys
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
