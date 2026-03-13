"""Train a nanoGPT-style Transformer baseline on TinyStories."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from baseline_model import BaselineConfig, BaselineTransformer
from data import BPETokenizer


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup followed by cosine decay."""
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def get_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Match SRN optimizer grouping (no decay on bias/LN/embeddings)."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() <= 1 or "bias" in name or "ln" in name or "wte" in name or "wpe" in name:
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


class MemmapBatcher:
    """Random contiguous token chunk sampler from memmap arrays."""

    def __init__(self, train_path: str, val_path: str, seq_len: int) -> None:
        self.seq_len = seq_len
        self.train = np.memmap(train_path, dtype=np.int32, mode="r")
        self.val = np.memmap(val_path, dtype=np.int32, mode="r")

        if len(self.train) <= seq_len + 1:
            raise ValueError("train_tokens.bin is too short for configured seq_len")
        if len(self.val) <= seq_len + 1:
            raise ValueError("val_tokens.bin is too short for configured seq_len")

    def get_batch(self, split: str, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        source = self.train if split == "train" else self.val
        max_start = len(source) - self.seq_len - 1
        starts = np.random.randint(0, max_start + 1, size=batch_size)

        x_np = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y_np = np.empty((batch_size, self.seq_len), dtype=np.int64)

        for i, start in enumerate(starts):
            end = start + self.seq_len
            x_np[i] = source[start:end]
            y_np[i] = source[start + 1 : end + 1]

        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np)

        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y


@torch.no_grad()
def evaluate(
    model: BaselineTransformer,
    batcher: MemmapBatcher,
    device: torch.device,
    micro_batch: int,
    eval_batches: int,
    use_amp: bool,
) -> float:
    """Estimate validation loss over a fixed number of random batches."""
    model.eval()
    losses: list[float] = []

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_amp else torch.amp.autocast(device_type="cpu", enabled=False)
    with amp_ctx:
        for _ in range(eval_batches):
            x, y = batcher.get_batch("val", micro_batch, device)
            logits = model(x)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
            losses.append(float(loss.item()))

    model.train()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def generate_sample(
    model: BaselineTransformer,
    tokenizer: BPETokenizer,
    device: torch.device,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Generate sample text for qualitative comparison."""
    model.eval()
    prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(prompt_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
    model.train()
    return tokenizer.decode(out[0].tolist())


def save_checkpoint(
    path: Path,
    model: BaselineTransformer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    best_val_loss: float,
    config: BaselineConfig,
) -> None:
    """Persist checkpoint state for resume and comparison."""
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config.__dict__,
        "rng_state": torch.get_rng_state(),
        "np_rng_state": np.random.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state()
    torch.save(payload, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer baseline for SRN comparison")
    parser.add_argument("--train_tokens_path", type=str, default="data/tinystories/train_tokens.bin")
    parser.add_argument("--val_tokens_path", type=str, default="data/tinystories/val_tokens.bin")
    parser.add_argument("--manifest_path", type=str, default="data/tinystories/manifest.json")
    parser.add_argument("--tokenizer_path", type=str, default="data/tinystories/tokenizer.json")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/baseline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--log_interval", type=int, default=10)

    parser.add_argument("--micro_batch", type=int, default=16)
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=512)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=16)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_bias", action="store_true")

    parser.add_argument("--sample_prompt", type=str, default="Once upon a time")
    parser.add_argument("--sample_tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    manifest = json.loads(Path(args.manifest_path).read_text(encoding="utf-8"))
    vocab_size = int(manifest.get("vocab_size", 50257))
    tokenizer = BPETokenizer.from_file(args.tokenizer_path)
    if tokenizer.vocab_size != vocab_size:
        raise ValueError(f"Tokenizer vocab_size={tokenizer.vocab_size} does not match manifest {vocab_size}")

    config = BaselineConfig(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        bias=not args.no_bias,
    )

    model = BaselineTransformer(config).to(device)
    total_params = model.count_params()

    print(f"Device: {device}")
    print("Model: Baseline Transformer")
    print(
        "Config: "
        f"d_model={config.d_model}, n_layers={config.n_layers}, "
        f"n_heads={config.n_heads}, d_ff={config.d_ff}, seq_len={config.max_seq_len}"
    )
    print(f"Parameters: {total_params:,} total ({total_params / 1e6:.2f}M)")

    batcher = MemmapBatcher(args.train_tokens_path, args.val_tokens_path, args.seq_len)

    param_groups = get_param_groups(model, args.weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=1e-8,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"

    start_step = 0
    best_val_loss = float("inf")
    if args.resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt["step"]) + 1
        best_val_loss = float(ckpt["best_val_loss"])
        torch.set_rng_state(ckpt["rng_state"])
        np.random.set_state(ckpt["np_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in ckpt:
            torch.cuda.set_rng_state(ckpt["cuda_rng_state"])
        print(f"Resumed from step {start_step}, best_val_loss={best_val_loss:.4f}")

    effective_batch = args.micro_batch * args.accum_steps
    tokens_per_step = effective_batch * args.seq_len
    print(
        f"Training: steps={args.max_steps}, micro_batch={args.micro_batch}, accum_steps={args.accum_steps}, "
        f"effective_batch={effective_batch}, tokens/step={tokens_per_step}"
    )
    print(
        f"Optimizer: AdamW lr={args.lr}, wd={args.weight_decay}, betas=({args.beta1}, {args.beta2}), "
        f"warmup={args.warmup_steps}, cosine->min_lr={args.min_lr}"
    )

    model.train()
    total_start = time.time()

    for step in range(start_step, args.max_steps):
        step_start = time.time()
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.accum_steps):
            x, y = batcher.get_batch("train", args.micro_batch, device)

            amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if use_amp else torch.amp.autocast(device_type="cpu", enabled=False)
            with amp_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1))
                loss = loss / args.accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item())
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0:
            step_elapsed = time.time() - step_start
            total_elapsed = time.time() - total_start
            steps_done = step - start_step + 1
            tok_s = tokens_per_step / max(step_elapsed, 1e-6)
            avg_tok_s = (steps_done * tokens_per_step) / max(total_elapsed, 1e-6)
            train_loss = accum_loss
            train_ppl = math.exp(min(train_loss, 20.0))
            vram_str = ""
            if torch.cuda.is_available():
                vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
                vram_str = f" | vram {vram_mb:.0f}M"

            print(
                f"step {step:>5d} | "
                f"loss {train_loss:.4f} | "
                f"ppl {train_ppl:>7.2f} | "
                f"lr {lr:.2e} | "
                f"grad {grad_norm:.2f} | "
                f"tok/s {tok_s:.0f} (avg {avg_tok_s:.0f})"
                f"{vram_str}"
            )

        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate(
                model=model,
                batcher=batcher,
                device=device,
                micro_batch=args.micro_batch,
                eval_batches=args.eval_batches,
                use_amp=use_amp,
            )
            val_ppl = math.exp(min(val_loss, 20.0))
            print(f"\nEval @ step {step}: val_loss={val_loss:.4f}, ppl={val_ppl:.2f}")

            sample = generate_sample(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=args.sample_prompt,
                max_new_tokens=args.sample_tokens,
            )
            print(f"Sample:\n{sample[:500]}\n")

            if torch.cuda.is_available():
                peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
                print(f"Peak VRAM (cumulative): {peak_mb:.1f} MB")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(best_path, model, optimizer, scaler, step, best_val_loss, config)
                print(f"New best checkpoint saved: {best_path} (val_loss={val_loss:.4f})")

            save_checkpoint(latest_path, model, optimizer, scaler, step, best_val_loss, config)

    final_val_loss = evaluate(
        model=model,
        batcher=batcher,
        device=device,
        micro_batch=args.micro_batch,
        eval_batches=max(args.eval_batches, 50),
        use_amp=use_amp,
    )
    final_ppl = math.exp(min(final_val_loss, 20.0))
    print(f"Final val_loss: {final_val_loss:.4f}, perplexity: {final_ppl:.2f}")
    print(f"Best val_loss: {best_val_loss:.4f}")

    save_checkpoint(ckpt_dir / "final.pt", model, optimizer, scaler, args.max_steps, best_val_loss, config)

    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"Peak VRAM usage: {peak_mb:.1f} MB")


if __name__ == "__main__":
    main()
