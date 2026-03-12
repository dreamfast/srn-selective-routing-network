"""
Training loop for the SRN (Selective Routing Network) language model.

Features:
- AdamW optimizer with parameter groups (no weight decay on biases/LN/embeddings)
- Cosine annealing with linear warmup
- Mixed precision training (autocast + GradScaler)
- Gradient accumulation for effective batch size > micro batch
- Gradient clipping (max_norm=1.0)
- MoE load balancing auxiliary loss
- Expert utilization monitoring
- Checkpoint save/resume with optimizer and scheduler state
- Sample text generation during evaluation
- Peak VRAM profiling

Usage:
    python train.py                          # Train with defaults
    python train.py --max_steps 1000         # Quick test
    python train.py --resume                 # Resume from latest checkpoint
    python train.py --micro_batch 8          # Reduce batch for less VRAM
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import get_dataloaders
from srn_model import SRNConfig, SRNModel


# ============================================================================
# Learning Rate Schedule
# ============================================================================


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine annealing with linear warmup.

    Args:
        step: current training step
        warmup_steps: number of warmup steps
        max_steps: total training steps
        max_lr: peak learning rate
        min_lr: minimum learning rate (end of cosine decay)

    Returns:
        learning rate for this step
    """
    # Linear warmup (guard against warmup_steps=0)
    if step < warmup_steps:
        return max_lr * (step + 1) / max(warmup_steps, 1)

    # Cosine decay
    if step >= max_steps:
        return min_lr

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================================
# Parameter Groups (no weight decay on biases, LN, embeddings)
# ============================================================================


def get_param_groups(model: SRNModel, weight_decay: float) -> list[dict]:
    """Create optimizer parameter groups with selective weight decay.

    Weight decay is NOT applied to:
    - Bias parameters
    - LayerNorm parameters (weight and bias)
    - Embedding parameters
    - 1D parameters (biases, LN gamma/beta)

    Args:
        model: the SRN model
        weight_decay: weight decay value for decayed parameters

    Returns:
        list of parameter group dicts for the optimizer
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No decay for: biases, LayerNorm, embeddings, 1D params
        if param.dim() <= 1:
            no_decay_params.append(param)
        elif "emb" in name or "ln" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"Parameter groups: {n_decay:,} with decay, {n_no_decay:,} without decay")

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


# ============================================================================
# Expert Utilization Monitoring
# ============================================================================


@torch.no_grad()
def get_expert_utilization(model: SRNModel, x: torch.Tensor) -> dict:
    """Measure expert utilization across all GEM layers.

    Returns dict with min/max/std of expert usage fractions.
    """
    model.eval()
    B, N = x.shape
    device = x.device

    # Run forward pass and collect router weights from each layer
    pos_ids = torch.arange(N, device=device).unsqueeze(0)
    hidden = model.token_emb(x) + model.pos_emb(pos_ids)
    hidden = model.emb_drop(hidden)

    all_fracs = []
    for layer in model.layers:
        # Get router logits from GEM
        normed = layer.ln3(hidden)
        router_logits = layer.gem.router(normed)  # (B, N, E)
        _, topk_idx = torch.topk(router_logits, model.config.top_k_experts, dim=-1)

        # Count how often each expert is selected
        mask = torch.zeros_like(router_logits, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)
        fracs = mask.float().mean(dim=(0, 1))  # (E,)
        all_fracs.append(fracs)

        # Advance hidden state through the layer
        hidden, _ = layer(hidden)

    model.train()

    all_fracs = torch.stack(all_fracs)  # (n_layers, E)
    return {
        "min": all_fracs.min().item(),
        "max": all_fracs.max().item(),
        "std": all_fracs.std().item(),
        "per_layer": all_fracs.mean(dim=1).tolist(),
    }


# ============================================================================
# Evaluation
# ============================================================================


@torch.no_grad()
def evaluate(model: SRNModel, val_loader, device: torch.device, max_batches: int = 20) -> float:
    """Compute average validation loss.

    Args:
        model: the SRN model
        val_loader: validation DataLoader
        device: torch device
        max_batches: maximum number of batches to evaluate

    Returns:
        average cross-entropy loss on validation set
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for x, y in val_loader:
        if n_batches >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, _ = model(x)

        # Loss in fp32
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
        n_batches += 1

    model.train()
    return total_loss / max(n_batches, 1)


# ============================================================================
# Training
# ============================================================================


def train(args: argparse.Namespace) -> None:
    """Main training function."""

    # ---- Setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Data ----
    train_loader, val_loader, tokenizer = get_dataloaders(
        batch_size=args.micro_batch,
        seq_len=args.seq_len,
        num_workers=0,
    )

    # ---- Model ----
    config = SRNConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.seq_len,
    )
    model = SRNModel(config).to(device)

    total_params = model.count_params()
    active_params = model.count_active_params()
    print(f"\nModel: {total_params:,} params ({active_params:,} active/token, {active_params/total_params:.1%})")

    # ---- Optimizer ----
    param_groups = get_param_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ---- Mixed precision ----
    scaler = torch.amp.GradScaler("cuda")

    # ---- Resume ----
    start_step = 0
    best_val_loss = float("inf")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        latest_path = checkpoint_dir / "latest.pt"
        if latest_path.exists():
            print(f"\nResuming from {latest_path}...")
            # weights_only=False needed to deserialize SRNConfig dataclass
            # Only load checkpoints you created — not untrusted files
            ckpt = torch.load(latest_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scaler.load_state_dict(ckpt["scaler"])
            start_step = ckpt["step"] + 1
            best_val_loss = ckpt["best_val_loss"]
            # Restore RNG state for reproducibility
            if "rng_state" in ckpt:
                torch.set_rng_state(ckpt["rng_state"])
                if torch.cuda.is_available() and "cuda_rng_state" in ckpt:
                    torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
                np.random.set_state(ckpt["np_rng_state"])
            print(f"Resumed at step {start_step}, best val loss: {best_val_loss:.4f}")
        else:
            print("No checkpoint found, starting fresh.")

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"Training SRN — {args.max_steps} steps")
    print(f"Micro batch: {args.micro_batch}, Accum steps: {args.accum_steps}, "
          f"Effective batch: {args.micro_batch * args.accum_steps}")
    print(f"Seq len: {args.seq_len}, LR: {args.max_lr}, Weight decay: {args.weight_decay}")
    print(f"{'='*60}\n")

    model.train()
    train_iter = iter(train_loader)
    t_start = time.time()

    for step in range(start_step, args.max_steps):
        # Update learning rate
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_aux = 0.0

        for micro_step in range(args.accum_steps):
            # Get batch (cycle through data)
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits, aux_loss = model(x)
                # Loss in fp32 for stability
                ce_loss = F.cross_entropy(
                    logits.float().view(-1, logits.size(-1)),
                    y.view(-1),
                )
                loss = ce_loss + config.aux_loss_weight * aux_loss
                # Scale for gradient accumulation
                loss = loss / args.accum_steps

            # Backward with scaler
            scaler.scale(loss).backward()

            accum_loss += ce_loss.item() / args.accum_steps
            accum_aux += aux_loss.item() / args.accum_steps

        # Gradient clipping (unscale first)
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # ---- Logging ----
        if step % args.log_interval == 0:
            elapsed = time.time() - t_start
            tokens_per_sec = (step - start_step + 1) * args.micro_batch * args.accum_steps * args.seq_len / max(elapsed, 1e-6)

            print(
                f"step {step:>5d} | "
                f"loss {accum_loss:.4f} | "
                f"aux {accum_aux:.2f} | "
                f"lr {lr:.2e} | "
                f"grad {grad_norm:.2f} | "
                f"tok/s {tokens_per_sec:.0f}"
            )

        # ---- Memory profiling (first step only) ----
        if step == start_step and torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"\n>>> Peak GPU memory: {peak_mem:.1f} MB <<<\n")

        # ---- Evaluation ----
        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate(model, val_loader, device, max_batches=args.eval_batches)
            perplexity = math.exp(min(val_loss, 20))  # Clamp to avoid overflow

            print(f"\n{'─'*60}")
            print(f"Eval @ step {step}: val_loss={val_loss:.4f}, ppl={perplexity:.2f}")

            # Expert utilization
            sample_x, _ = next(iter(val_loader))
            expert_util = get_expert_utilization(model, sample_x.to(device))
            print(f"Expert util: min={expert_util['min']:.3f}, max={expert_util['max']:.3f}, "
                  f"std={expert_util['std']:.4f}")

            # Generate sample text
            prompt_text = "ROMEO:"
            prompt_ids = torch.tensor(
                [tokenizer.encode(prompt_text)], dtype=torch.long, device=device
            )
            generated = model.generate(prompt_ids, max_tokens=200, temperature=0.8)
            gen_text = tokenizer.decode(generated[0].tolist())
            print(f"\nGenerated:\n{gen_text[:300]}")
            print(f"{'─'*60}\n")

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scaler, step, best_val_loss, config, tokenizer,
                    checkpoint_dir / "best.pt",
                )
                print(f">>> New best model saved (val_loss={val_loss:.4f}) <<<\n")

            # Save latest
            save_checkpoint(
                model, optimizer, scaler, step, best_val_loss, config, tokenizer,
                checkpoint_dir / "latest.pt",
            )
            model.train()

    # ---- Final evaluation ----
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    val_loss = evaluate(model, val_loader, device, max_batches=50)
    perplexity = math.exp(min(val_loss, 20))
    print(f"Final val_loss: {val_loss:.4f}, perplexity: {perplexity:.2f}")
    print(f"Best val_loss:  {best_val_loss:.4f}")

    # Save final
    save_checkpoint(
        model, optimizer, scaler, args.max_steps, best_val_loss, config, tokenizer,
        checkpoint_dir / "final.pt",
    )

    # Final generation
    prompt_text = "ROMEO:\nO, "
    prompt_ids = torch.tensor(
        [tokenizer.encode(prompt_text)], dtype=torch.long, device=device
    )
    generated = model.generate(prompt_ids, max_tokens=500, temperature=0.8)
    gen_text = tokenizer.decode(generated[0].tolist())
    print(f"\nFinal generation:\n{'─'*60}\n{gen_text}\n{'─'*60}")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nPeak GPU memory: {peak_mem:.1f} MB")


def save_checkpoint(
    model, optimizer, scaler, step, best_val_loss, config, tokenizer, path
):
    """Save training checkpoint with all state needed for reproducible resume."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config,
        "tokenizer_chars": tokenizer.chars,
        # RNG state for reproducible resume
        "rng_state": torch.get_rng_state(),
        "np_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state()
    torch.save(ckpt, path)


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SRN language model")

    # Training
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--micro_batch", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=256)

    # Optimizer
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Evaluation
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)

    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", action="store_true")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
