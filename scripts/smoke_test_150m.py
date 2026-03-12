"""Smoke test for SRN-150M configuration.

Runs a tiny training loop (5 steps) with the 150M architecture to verify:
- Model construction succeeds
- Forward/backward pass completes without errors
- No NaN/Inf in logits or gradients
- VRAM usage is reported
- Checkpoint save/load round-trip works

This is designed to run on any GPU (including RTX 2060 with 6GB) by using
a minimal batch size and sequence length. It does NOT train a useful model —
it only verifies the architecture doesn't crash.

Usage:
    python scripts/smoke_test_150m.py
    python scripts/smoke_test_150m.py --steps 10 --seq_len 128
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import CharTokenizer
from srn_model import SRNConfig, SRNModel
from train import resolve_precision, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test SRN-150M config")
    parser.add_argument("--steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length (reduce for low VRAM)")
    parser.add_argument("--micro_batch", type=int, default=2, help="Micro batch size")
    parser.add_argument("--precision", type=str, choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Save checkpoint here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    precision_dtype = resolve_precision(args.precision)
    precision_label = "bf16" if precision_dtype == torch.bfloat16 else "fp16"

    # Use 150M architecture but with a dummy vocab
    config = SRNConfig(
        vocab_size=256,  # Dummy — just needs to be valid
        max_seq_len=args.seq_len,
        d_model=896,
        d_compressed=224,
        n_layers=12,
        n_memory_slots=96,
        n_experts=8,
        top_k_experts=2,
        d_expert=384,
        n_heads_route=8,
        dropout=0.0,  # No dropout for deterministic smoke test
        causal_window=64,
        csp_internal_residual=False,
        aux_loss_weight=0.01,
    )

    print(f"\n{'='*60}")
    print("SRN-150M Smoke Test")
    print(f"{'='*60}")
    print(f"Precision: {precision_label}")
    print(f"Seq len: {args.seq_len}, Micro batch: {args.micro_batch}")
    print(f"Steps: {args.steps}")

    # Construct model
    print("\nConstructing model...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = SRNModel(config).to(device)
    total_params = model.count_params()
    active_params = model.count_active_params()
    print(f"Total params:  {total_params:>12,}")
    print(f"Active/token:  {active_params:>12,} ({active_params/total_params:.1%})")

    if torch.cuda.is_available():
        model_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Model VRAM:    {model_mem:.1f} MB")

    if args.compile:
        print("Compiling model...")
        model = torch.compile(model)
        print("Compiled.")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    use_scaler = (precision_dtype == torch.float16) and (device.type == "cuda")
    scaler_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(scaler_device, enabled=use_scaler)

    # Training loop
    print(f"\nRunning {args.steps} training steps...")
    model.train()
    t_start = time.time()

    for step in range(args.steps):
        x = torch.randint(0, config.vocab_size, (args.micro_batch, args.seq_len), device=device)
        y = torch.randint(0, config.vocab_size, (args.micro_batch, args.seq_len), device=device)

        optimizer.zero_grad()

        with torch.amp.autocast(device.type, dtype=precision_dtype):
            logits, aux_loss = model(x)
            ce_loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
            loss = ce_loss + config.aux_loss_weight * aux_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Check for NaN/Inf
        has_nan = torch.isnan(logits).any().item()
        has_inf = torch.isinf(logits).any().item()
        grad_nan = math.isnan(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else math.isnan(grad_norm)

        ppl = math.exp(min(ce_loss.item(), 20))

        status = "OK"
        if has_nan or has_inf or grad_nan:
            problems = []
            if has_nan:
                problems.append("NaN logits")
            if has_inf:
                problems.append("Inf logits")
            if grad_nan:
                problems.append("NaN grad")
            status = f"PROBLEM: {', '.join(problems)}"

        print(f"  step {step}: loss={ce_loss.item():.4f} ppl={ppl:.1f} "
              f"grad={grad_norm:.2f} aux={aux_loss.item():.3f} [{status}]")

        if has_nan or has_inf or grad_nan:
            print("\n*** SMOKE TEST FAILED — numerical instability detected ***")
            sys.exit(1)

    elapsed = time.time() - t_start
    tokens_total = args.steps * args.micro_batch * args.seq_len
    tok_per_sec = tokens_total / max(elapsed, 1e-6)

    print(f"\nCompleted {args.steps} steps in {elapsed:.1f}s ({tok_per_sec:.0f} tok/s)")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak VRAM: {peak_mem:.1f} MB")

    # Checkpoint round-trip test
    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "smoke_test.pt"

        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        tokenizer = CharTokenizer("".join(chr(i) for i in range(config.vocab_size) if chr(i).isprintable()))
        save_checkpoint(
            raw_model, optimizer, scaler, step=args.steps, best_val_loss=ce_loss.item(),
            config=config, tokenizer=tokenizer, path=ckpt_path,
            precision=precision_label, compiled=args.compile,
        )
        print(f"Checkpoint saved: {ckpt_path}")

        # Reload test
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model2 = SRNModel(ckpt["config"]).to(device)
        model2.load_state_dict(ckpt["model"])
        model2.eval()
        with torch.no_grad():
            test_x = torch.randint(0, config.vocab_size, (1, 32), device=device)
            test_logits, _ = model2(test_x)
            if torch.isnan(test_logits).any():
                raise RuntimeError("NaN in reloaded model logits")
            if torch.isinf(test_logits).any():
                raise RuntimeError("Inf in reloaded model logits")
        print("Checkpoint reload: OK")

    print(f"\n{'='*60}")
    print("SMOKE TEST PASSED")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
