"""Tests for Task 5: Promotion gate validation logic.

Tests the individual gate functions from validate_checkpoint.py
using small models and synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

# Allow imports from scripts/
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data import CharTokenizer
from srn_model import SRNConfig, SRNModel
from train import save_checkpoint
from validate_checkpoint import (
    gate_generate,
    gate_nan_inf,
    gate_reproducibility,
    gate_vram,
    run_promotion_gates,
)


def test_gate_nan_inf_passes_clean_model(sample_config: SRNConfig) -> None:
    """NaN/Inf gate passes for a freshly initialized model."""
    model = SRNModel(sample_config)
    model.eval()
    result = gate_nan_inf(model, torch.device("cpu"), sample_config)
    assert result.passed, result.detail


def test_gate_reproducibility_passes(sample_config: SRNConfig) -> None:
    """Reproducibility gate passes — same input gives same output."""
    model = SRNModel(sample_config)
    model.eval()
    result = gate_reproducibility(model, torch.device("cpu"), sample_config)
    assert result.passed, result.detail


def test_gate_vram_passes_generous_limit() -> None:
    """VRAM gate passes with a generous limit."""
    result = gate_vram(max_vram_mb=999999)
    assert result.passed, result.detail


def test_gate_vram_fails_tiny_limit() -> None:
    """VRAM gate fails with an impossibly small limit (if CUDA available)."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA — VRAM gate always passes on CPU")
    # Allocate some memory to ensure peak > 0
    _ = torch.zeros(1000, 1000, device="cuda")
    result = gate_vram(max_vram_mb=0.001)
    assert not result.passed


def test_gate_generate_passes(sample_config: SRNConfig) -> None:
    """Generate gate passes for a valid model + tokenizer."""
    from dataclasses import replace

    tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz .,!?the")
    config = replace(sample_config, vocab_size=tokenizer.vocab_size)
    model = SRNModel(config)
    model.eval()
    result = gate_generate(model, tokenizer, torch.device("cpu"))
    assert result.passed, result.detail


def test_run_promotion_gates_on_checkpoint(
    tmp_path: Path, sample_config: SRNConfig
) -> None:
    """Full promotion gate suite runs on a saved checkpoint."""
    from dataclasses import replace

    tokenizer = CharTokenizer("abcdefghijklmnopqrstuvwxyz .,!?the")
    config = replace(sample_config, vocab_size=tokenizer.vocab_size)
    model = SRNModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    ckpt_path = tmp_path / "test_ckpt.pt"
    save_checkpoint(
        model, optimizer, scaler, step=0, best_val_loss=5.0,
        config=config, tokenizer=tokenizer, path=ckpt_path,
    )

    results = run_promotion_gates(
        checkpoint_path=str(ckpt_path),
        max_vram_mb=999999,
        max_ppl=999999,
    )

    # All gates should pass for a fresh model
    for r in results:
        # val_perplexity may be skipped if Shakespeare data isn't available
        if "skipped" in r.detail:
            continue
        assert r.passed, f"Gate {r.name} failed: {r.detail}"
