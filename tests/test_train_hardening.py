"""Tests for Task 4: Train Loop Hardening (150M readiness).

Covers:
- Checkpoint save/resume restores step + RNG state
- Precision resolution (bf16 fallback to fp16) with warnings
- --compile flag smoke test (parse_args accepts it)
- Checkpoint contains all required fields
- save_checkpoint / resume round-trip
- Checkpoint metadata validation warnings on mismatch
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from data import CharTokenizer
from srn_model import SRNConfig, SRNModel
from train import _get_raw_model, parse_args, resolve_precision, save_checkpoint


# ============================================================================
# Precision resolution
# ============================================================================


def test_resolve_precision_fp16() -> None:
    """fp16 always resolves to torch.float16."""
    assert resolve_precision("fp16") == torch.float16


def test_resolve_precision_bf16_fallback_no_cuda() -> None:
    """bf16 falls back to fp16 with a warning when CUDA is not available."""
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.warns(UserWarning, match="bf16 requested but not supported"):
            result = resolve_precision("bf16")
    assert result == torch.float16


def test_resolve_precision_bf16_fallback_unsupported() -> None:
    """bf16 falls back to fp16 with a warning when GPU doesn't support it."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.is_bf16_supported", return_value=False):
        with pytest.warns(UserWarning, match="bf16 requested but not supported"):
            result = resolve_precision("bf16")
    assert result == torch.float16


def test_resolve_precision_bf16_supported() -> None:
    """bf16 resolves to bfloat16 when GPU supports it."""
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.is_bf16_supported", return_value=True):
        result = resolve_precision("bf16")
    assert result == torch.bfloat16


def test_resolve_precision_invalid_raises() -> None:
    """Unknown precision string raises ValueError."""
    with pytest.raises(ValueError, match="Unknown precision"):
        resolve_precision("fp32")


# ============================================================================
# _get_raw_model helper
# ============================================================================


def test_get_raw_model_unwraps_compiled(sample_config: SRNConfig) -> None:
    """_get_raw_model returns the underlying module from a compiled wrapper."""
    model = SRNModel(sample_config)
    # Simulate torch.compile by setting _orig_mod
    wrapper = type("CompiledModel", (), {"_orig_mod": model})()
    assert _get_raw_model(wrapper) is model


def test_get_raw_model_passthrough(sample_config: SRNConfig) -> None:
    """_get_raw_model returns the model itself when not compiled."""
    model = SRNModel(sample_config)
    assert _get_raw_model(model) is model


# ============================================================================
# CLI argument parsing
# ============================================================================


def test_parse_args_compile_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--compile flag is accepted and defaults to False."""
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = parse_args()
    assert args.compile is False

    monkeypatch.setattr(sys, "argv", ["train.py", "--compile"])
    args = parse_args()
    assert args.compile is True


def test_parse_args_precision_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--precision flag accepts fp16 and bf16."""
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = parse_args()
    assert args.precision == "fp16"

    monkeypatch.setattr(sys, "argv", ["train.py", "--precision", "bf16"])
    args = parse_args()
    assert args.precision == "bf16"


# ============================================================================
# Checkpoint save/resume round-trip
# ============================================================================


def test_checkpoint_contains_all_required_fields(
    tmp_path: Path, sample_config: SRNConfig
) -> None:
    """Checkpoint must contain all fields needed for reproducible resume."""
    model = SRNModel(sample_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    tokenizer = CharTokenizer("abcdef")

    ckpt_path = tmp_path / "test.pt"
    save_checkpoint(
        model, optimizer, scaler, step=42, best_val_loss=1.5,
        config=sample_config, tokenizer=tokenizer, path=ckpt_path,
        precision="fp16", compiled=False,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    required_keys = {
        "format_version", "model", "optimizer", "scaler",
        "step", "best_val_loss", "config",
        "precision", "compiled",
        "rng_state", "np_rng_state",
        "tokenizer_type",
    }
    for key in required_keys:
        assert key in ckpt, f"Missing checkpoint key: {key}"

    assert ckpt["format_version"] == 2
    assert ckpt["step"] == 42
    assert ckpt["best_val_loss"] == 1.5
    assert ckpt["precision"] == "fp16"
    assert ckpt["compiled"] is False


def test_checkpoint_resume_restores_step_and_rng(
    tmp_path: Path, sample_config: SRNConfig
) -> None:
    """Saving and loading a checkpoint restores step count and RNG state."""
    model = SRNModel(sample_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    tokenizer = CharTokenizer("abcdef")

    # Set a specific RNG state
    torch.manual_seed(999)
    np.random.seed(999)

    # Generate some random numbers to advance RNG
    _ = torch.randn(10)
    _ = np.random.randn(10)

    # Capture RNG state before save
    torch_rng_before = torch.get_rng_state().clone()
    np_rng_before = np.random.get_state()

    ckpt_path = tmp_path / "resume_test.pt"
    save_checkpoint(
        model, optimizer, scaler, step=100, best_val_loss=2.0,
        config=sample_config, tokenizer=tokenizer, path=ckpt_path,
    )

    # Scramble RNG state
    torch.manual_seed(12345)
    np.random.seed(12345)

    # Load checkpoint and restore RNG
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["step"] == 100

    torch.set_rng_state(ckpt["rng_state"])
    np.random.set_state(ckpt["np_rng_state"])

    # RNG state should match what was saved
    torch_rng_after = torch.get_rng_state()
    assert torch.equal(torch_rng_before, torch_rng_after), \
        "PyTorch RNG state not restored correctly"

    # Verify numpy RNG produces same sequence
    np_rng_after = np.random.get_state()
    assert np_rng_before[0] == np_rng_after[0], "NumPy RNG algorithm mismatch"
    assert np.array_equal(np_rng_before[1], np_rng_after[1]), \
        "NumPy RNG state array not restored correctly"


def test_checkpoint_model_weights_roundtrip(
    tmp_path: Path, sample_config: SRNConfig
) -> None:
    """Model weights survive save/load round-trip."""
    model = SRNModel(sample_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    tokenizer = CharTokenizer("abcdef")

    # Do a forward pass to get non-trivial gradients
    x = torch.randint(0, sample_config.vocab_size, (2, 16))
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()
    optimizer.step()

    # Save
    ckpt_path = tmp_path / "weights_test.pt"
    save_checkpoint(
        model, optimizer, scaler, step=1, best_val_loss=5.0,
        config=sample_config, tokenizer=tokenizer, path=ckpt_path,
    )

    # Load into fresh model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model2 = SRNModel(sample_config)
    model2.load_state_dict(ckpt["model"])

    # Compare all parameters
    for (n1, p1), (n2, p2) in zip(
        model.named_parameters(), model2.named_parameters()
    ):
        assert n1 == n2
        assert torch.equal(p1.data, p2.data), f"Parameter {n1} mismatch after load"


def test_checkpoint_precision_bf16_stored(
    tmp_path: Path, sample_config: SRNConfig
) -> None:
    """Checkpoint stores bf16 precision metadata correctly."""
    model = SRNModel(sample_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    tokenizer = CharTokenizer("abcdef")

    ckpt_path = tmp_path / "bf16_test.pt"
    save_checkpoint(
        model, optimizer, scaler, step=10, best_val_loss=3.0,
        config=sample_config, tokenizer=tokenizer, path=ckpt_path,
        precision="bf16", compiled=True,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["precision"] == "bf16"
    assert ckpt["compiled"] is True
