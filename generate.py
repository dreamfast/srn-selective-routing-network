"""
Text generation and evaluation for trained SRN models.

Usage:
    python generate.py --checkpoint checkpoints/best.pt --prompt "ROMEO:" --max_tokens 500
    python generate.py --checkpoint checkpoints/best.pt --eval
    python generate.py --checkpoint checkpoints/best.pt --stats
"""

import argparse
import math

import torch
import torch.nn.functional as F

from data import TokenizerType, get_dataloaders, tokenizer_from_checkpoint
from srn_model import SRNConfig, SRNModel


def load_model(
    checkpoint_path: str, device: torch.device
) -> tuple[SRNModel, TokenizerType, dict]:
    """Load a trained SRN model from checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint file
        device: torch device to load onto

    Returns:
        model: loaded SRN model in eval mode
        tokenizer: tokenizer reconstructed from checkpoint metadata
        metadata: dict with step, best_val_loss, config
    """
    # weights_only=False needed to deserialize SRNConfig dataclass
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = ckpt["config"]
    model = SRNModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = tokenizer_from_checkpoint(ckpt)

    metadata = {
        "step": ckpt.get("step", -1),
        "best_val_loss": ckpt.get("best_val_loss", float("inf")),
        "config": config,
        "tokenizer_type": ckpt.get("tokenizer_type", "char"),
        "tokenizer_path": ckpt.get("tokenizer_path"),
    }

    return model, tokenizer, metadata


@torch.no_grad()
def compute_perplexity(
    model: SRNModel, val_loader, device: torch.device, max_batches: int = 50
) -> float:
    """Compute perplexity on the validation set.

    Perplexity = exp(mean cross-entropy loss per token).

    Args:
        model: SRN model in eval mode
        val_loader: validation DataLoader
        device: torch device
        max_batches: maximum batches to evaluate

    Returns:
        perplexity value (lower is better, 1.0 = perfect)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits, _ = model(x)

        # Per-token cross-entropy (no reduction)
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            y.view(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += y.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))  # Clamp to prevent overflow
    return perplexity


def print_model_stats(model: SRNModel, config: SRNConfig) -> None:
    """Print model architecture statistics."""
    total = model.count_params()
    active = model.count_active_params()

    print(f"\n{'─'*50}")
    print("MODEL STATISTICS")
    print(f"{'─'*50}")
    print(f"  Architecture:      SRN (Selective Routing Network)")
    print(f"  Total params:      {total:>12,}")
    print(f"  Active/token:      {active:>12,} ({active/total:.1%})")
    print(f"  VRAM (fp16):       {total * 2 / 1024**2:>10.1f} MB")
    print(f"  d_model:           {config.d_model}")
    print(f"  n_layers:          {config.n_layers}")
    print(f"  n_memory_slots:    {config.n_memory_slots}")
    print(f"  n_experts:         {config.n_experts}")
    print(f"  top_k_experts:     {config.top_k_experts}")
    print(f"  causal_window:     {config.causal_window}")
    print(f"  vocab_size:        {config.vocab_size}")
    print(f"  max_seq_len:       {config.max_seq_len}")
    print(f"{'─'*50}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with trained SRN")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt", type=str, default="ROMEO:\n", help="Text prompt for generation"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=500, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (0=greedy, 1=standard)",
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling (None=disabled)"
    )
    parser.add_argument(
        "--eval", action="store_true", dest="do_eval", help="Compute validation perplexity"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Print model statistics"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model, tokenizer, metadata = load_model(args.checkpoint, device)
    print(
        f"Loaded model from step {metadata['step']}, "
        f"best val_loss={metadata['best_val_loss']:.4f}"
    )

    # Stats
    if args.stats:
        print_model_stats(model, metadata["config"])

    # Evaluation
    if args.do_eval:
        print("\nComputing validation perplexity...")
        _, val_loader, _ = get_dataloaders(
            batch_size=16,
            seq_len=metadata["config"].max_seq_len,
            tokenizer_backend=metadata["tokenizer_type"],
            tokenizer_path=metadata.get("tokenizer_path"),
            tokenizer_override=tokenizer,
        )
        ppl = compute_perplexity(model, val_loader, device)
        print(f"Validation perplexity: {ppl:.2f}")

    # Generation
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print(f"{'─'*60}")

    prompt_ids = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
    )
    generated = model.generate(
        prompt_ids,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = tokenizer.decode(generated[0].tolist())
    print(text)
    print(f"{'─'*60}")
    print(f"Generated {generated.shape[1] - prompt_ids.shape[1]} tokens")


if __name__ == "__main__":
    main()
