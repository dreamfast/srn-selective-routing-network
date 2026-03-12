"""
Validation and comparison tests for the SRN architecture.

Tests:
1. Original NumPy implementation still runs correctly
2. PyTorch model produces valid outputs (shapes, no NaN, valid distributions)
3. Causal masking test: modifying future tokens doesn't affect past logits
4. Comparison table: original vs improved architecture

Usage:
    python validate.py
"""

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from srn_model import SRNConfig, SRNModel


def test_numpy_original() -> bool:
    """Verify the original NumPy SRN implementation still works."""
    print("Test 1: Original NumPy Implementation")
    print("─" * 50)

    try:
        # Import and run the original analysis
        sys.path.insert(0, ".")
        from srn_architecture import SRNConfig as NpConfig
        from srn_architecture import SRNModel as NpModel
        from srn_architecture import softmax

        config = NpConfig()
        model = NpModel(config)

        # Forward pass
        rng = np.random.RandomState(123)
        test_input = rng.randint(0, config.vocab_size, (2, 128))
        logits = model.forward(test_input)

        # Checks
        assert not np.isnan(logits).any(), "NaN in NumPy output"
        assert not np.isinf(logits).any(), "Inf in NumPy output"

        probs = softmax(logits, axis=-1)
        prob_sums = probs.sum(axis=-1)
        assert np.allclose(prob_sums, 1.0, atol=1e-5), "Probs don't sum to 1"

        print(f"  Output shape:  {logits.shape}")
        print(f"  Param count:   {model.count_params():,}")
        print(f"  NaN check:     PASSED")
        print(f"  Prob check:    PASSED")
        print(f"  ✅ PASSED\n")
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        return False


def test_pytorch_model() -> bool:
    """Verify the PyTorch SRN model produces valid outputs."""
    print("Test 2: PyTorch Model Validation")
    print("─" * 50)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = SRNConfig()
        model = SRNModel(config).to(device)

        # Forward pass
        B, N = 2, 128
        token_ids = torch.randint(0, config.vocab_size, (B, N), device=device)

        with torch.no_grad():
            logits, aux_loss = model(token_ids)

        # Shape check
        expected_shape = (B, N, config.vocab_size)
        assert logits.shape == expected_shape, f"Shape mismatch: {logits.shape} != {expected_shape}"

        # NaN/Inf check
        assert not torch.isnan(logits).any(), "NaN in PyTorch output"
        assert not torch.isinf(logits).any(), "Inf in PyTorch output"

        # Valid probability distribution
        probs = F.softmax(logits, dim=-1)
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), \
            "Probs don't sum to 1"

        # Output statistics in reasonable range
        mean_val = logits.mean().item()
        std_val = logits.std().item()
        assert -10 < mean_val < 10, f"Output mean out of range: {mean_val}"
        assert 0.1 < std_val < 100, f"Output std out of range: {std_val}"

        # Aux loss is finite and positive
        assert torch.isfinite(aux_loss), f"Aux loss not finite: {aux_loss}"
        assert aux_loss > 0, f"Aux loss not positive: {aux_loss}"

        print(f"  Output shape:  {tuple(logits.shape)}")
        print(f"  Param count:   {model.count_params():,}")
        print(f"  Logit mean:    {mean_val:.4f}")
        print(f"  Logit std:     {std_val:.4f}")
        print(f"  Aux loss:      {aux_loss.item():.4f}")
        print(f"  NaN check:     PASSED")
        print(f"  Prob check:    PASSED")
        print(f"  Range check:   PASSED")
        print(f"  ✅ PASSED\n")
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_causal_masking() -> bool:
    """Verify that modifying future tokens doesn't affect past logits.

    For each test position t, we modify the token at position t and verify
    that logits at all positions < t remain unchanged. This confirms the
    Windowed Causal Score Gating (WCSG) mechanism is truly causal.
    """
    print("Test 3: Causal Masking Verification")
    print("─" * 50)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = SRNConfig()
        model = SRNModel(config).to(device)
        model.eval()

        seq_len = 64
        test_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)

        with torch.no_grad():
            logits1, _ = model(test_ids)

        # Test at various positions including window boundaries
        test_positions = [1, 8, 16, config.causal_window - 1, config.causal_window,
                          config.causal_window + 1, seq_len - 1]
        test_positions = [t for t in test_positions if 0 < t < seq_len]

        all_passed = True
        for t in test_positions:
            modified = test_ids.clone()
            modified[0, t] = (test_ids[0, t] + 1) % config.vocab_size

            with torch.no_grad():
                logits2, _ = model(modified)

            # Positions before t should be identical
            if torch.allclose(logits1[0, :t], logits2[0, :t], atol=1e-5):
                print(f"  Position {t:>3d}: PASSED")
            else:
                max_diff = (logits1[0, :t] - logits2[0, :t]).abs().max().item()
                print(f"  Position {t:>3d}: FAILED (max diff: {max_diff:.8f})")
                all_passed = False

        if all_passed:
            print(f"  ✅ ALL POSITIONS PASSED\n")
        else:
            print(f"  ❌ SOME POSITIONS FAILED\n")
        return all_passed

    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_generation() -> bool:
    """Verify autoregressive generation produces valid output."""
    print("Test 4: Generation Validation")
    print("─" * 50)

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = SRNConfig()
        model = SRNModel(config).to(device)
        model.eval()

        prompt = torch.zeros(1, 1, dtype=torch.long, device=device)

        # Test greedy (deterministic)
        gen1 = model.generate(prompt, max_tokens=20, temperature=0.0)
        gen2 = model.generate(prompt, max_tokens=20, temperature=0.0)
        assert torch.equal(gen1, gen2), "Greedy generation not deterministic"
        print(f"  Greedy determinism: PASSED")

        # Test that all tokens are valid
        gen = model.generate(prompt, max_tokens=100, temperature=0.8)
        assert (gen >= 0).all() and (gen < config.vocab_size).all(), \
            f"Token out of range: [{gen.min()}, {gen.max()}]"
        print(f"  Token range:        PASSED ({gen.min().item()}-{gen.max().item()})")

        # Test with top-k
        gen_topk = model.generate(prompt, max_tokens=50, temperature=0.8, top_k=10)
        assert gen_topk.shape == (1, 51), f"Wrong shape: {gen_topk.shape}"
        print(f"  Top-k generation:   PASSED")

        print(f"  ✅ PASSED\n")
        return True

    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def print_comparison() -> None:
    """Print comparison table between original NumPy and improved PyTorch."""
    print("Architecture Comparison")
    print("─" * 60)

    # Original NumPy
    from srn_architecture import SRNConfig as NpConfig, SRNModel as NpModel
    np_config = NpConfig()
    np_model = NpModel(np_config)
    np_params = np_model.count_params()

    # PyTorch
    pt_config = SRNConfig()
    pt_model = SRNModel(pt_config)
    pt_params = pt_model.count_params()
    pt_active = pt_model.count_active_params()

    print(f"\n  {'Feature':<35} {'NumPy (Original)':<20} {'PyTorch (Improved)':<20}")
    print(f"  {'─'*35} {'─'*20} {'─'*20}")
    print(f"  {'Total parameters':<35} {np_params:>15,}    {pt_params:>15,}")
    print(f"  {'Active params/token':<35} {'N/A':>15}    {pt_active:>15,}")
    print(f"  {'Vocab size':<35} {np_config.vocab_size:>15,}    {pt_config.vocab_size:>15,}")
    print(f"  {'Max seq len':<35} {np_config.max_seq_len:>15,}    {pt_config.max_seq_len:>15,}")
    print(f"  {'d_model':<35} {np_config.d_model:>15}    {pt_config.d_model:>15}")
    print(f"  {'Causal routing':<35} {'❌ (global mean)':>15}    {'✅ (WCSG)':>15}")
    print(f"  {'Positional routing bias':<35} {'❌':>15}    {'✅':>15}")
    print(f"  {'Vectorized experts':<35} {'❌ (Python loop)':>15}    {'✅ (einsum)':>15}")
    print(f"  {'MoE load balancing':<35} {'❌':>15}    {'✅':>15}")
    print(f"  {'Dropout':<35} {'❌ (0.0 default)':>15}    {'✅ (0.1)':>15}")
    print(f"  {'Dead W_gate_slot param':<35} {'⚠️  Present':>15}    {'✅ Removed':>15}")
    print(f"  {'CSP double residual bug':<35} {'⚠️  Present':>15}    {'✅ Fixed':>15}")
    print(f"  {'Weight tying':<35} {'✅':>15}    {'✅':>15}")
    print(f"  {'GPU support':<35} {'❌':>15}    {'✅':>15}")
    print(f"  {'Training support':<35} {'❌':>15}    {'✅':>15}")
    print(f"  {'Generation':<35} {'❌':>15}    {'✅':>15}")
    print()


def main() -> None:
    print("=" * 60)
    print("SRN Validation Suite")
    print("=" * 60)
    print()

    results = []

    results.append(("NumPy Original", test_numpy_original()))
    results.append(("PyTorch Model", test_pytorch_model()))
    results.append(("Causal Masking", test_causal_masking()))
    results.append(("Generation", test_generation()))

    print_comparison()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:<25} {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed! 🎉")
    else:
        print("Some tests failed. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
