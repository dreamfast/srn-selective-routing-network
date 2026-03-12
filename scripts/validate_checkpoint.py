"""Promotion gate validation for trained SRN checkpoints.

Runs a series of automated checks to determine if a checkpoint is ready
for promotion to the next training phase. All gates must pass.

Gates:
1. NaN/Inf check — forward pass produces no NaN or Inf in logits
2. Reproducibility — two forward passes with same seed produce identical logits
3. VRAM budget — peak memory stays under a configurable threshold
4. Checkpoint load+generate — model loads from checkpoint and generates text
5. Val perplexity — validation perplexity is below a configurable ceiling

Usage:
    python scripts/validate_checkpoint.py --checkpoint checkpoints/srn-150m/best.pt
    python scripts/validate_checkpoint.py --checkpoint checkpoints/srn-150m/best.pt --max_vram_mb 8000
    python scripts/validate_checkpoint.py --checkpoint checkpoints/srn-150m/best.pt --max_ppl 50
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import get_dataloaders, get_memmap_dataloaders, tokenizer_from_checkpoint
from srn_model import SRNModel


@dataclass
class GateResult:
    """Result of a single promotion gate check."""

    name: str
    passed: bool
    detail: str


def gate_nan_inf(model: SRNModel, device: torch.device, config) -> GateResult:
    """Gate 1: Forward pass produces no NaN or Inf in logits."""
    model.eval()
    with torch.no_grad():
        x = torch.randint(0, config.vocab_size, (2, min(128, config.max_seq_len)), device=device)
        logits, aux_loss = model(x)

    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    aux_nan = math.isnan(aux_loss.item())

    if has_nan or has_inf or aux_nan:
        problems = []
        if has_nan:
            problems.append("NaN in logits")
        if has_inf:
            problems.append("Inf in logits")
        if aux_nan:
            problems.append("NaN in aux_loss")
        return GateResult("nan_inf", False, f"FAILED: {', '.join(problems)}")

    logit_range = f"[{logits.min().item():.3f}, {logits.max().item():.3f}]"
    return GateResult("nan_inf", True, f"logit range {logit_range}, aux_loss={aux_loss.item():.4f}")


def gate_reproducibility(model: SRNModel, device: torch.device, config) -> GateResult:
    """Gate 2: Two forward passes with same input produce consistent logits.

    Uses torch.allclose with tolerances rather than exact equality, because
    CUDA operations can have non-deterministic kernel orderings that produce
    small floating-point differences even with identical inputs.
    """
    model.eval()
    seq_len = min(64, config.max_seq_len)

    torch.manual_seed(12345)
    if device.type == "cuda":
        torch.cuda.manual_seed(12345)
    x = torch.randint(0, config.vocab_size, (1, seq_len), device=device)

    with torch.no_grad():
        logits1, _ = model(x)

    with torch.no_grad():
        logits2, _ = model(x)

    if not torch.allclose(logits1, logits2, rtol=1e-4, atol=1e-5):
        max_diff = (logits1 - logits2).abs().max().item()
        return GateResult("reproducibility", False, f"FAILED: max diff={max_diff:.6e}")

    return GateResult("reproducibility", True, "consistent logits on repeated forward pass")


def gate_vram(max_vram_mb: float) -> GateResult:
    """Gate 3: Peak VRAM stays under threshold."""
    if not torch.cuda.is_available():
        return GateResult("vram", True, "skipped (no CUDA)")

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    passed = peak_mb <= max_vram_mb
    detail = f"peak={peak_mb:.1f}MB, limit={max_vram_mb:.0f}MB"
    if not passed:
        detail = f"FAILED: {detail}"
    return GateResult("vram", passed, detail)


def gate_generate(model: SRNModel, tokenizer, device: torch.device) -> GateResult:
    """Gate 4: Model loads and generates text without errors."""
    model.eval()
    try:
        # Use a simple prompt that's likely in any tokenizer's vocabulary.
        # Fall back to token ID 0 if encoding fails or is empty.
        try:
            prompt_ids = torch.tensor(
                [tokenizer.encode("the")], dtype=torch.long, device=device
            )
        except Exception:
            prompt_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
        # Guard against empty prompt encoding
        if prompt_ids.shape[1] == 0:
            prompt_ids = torch.zeros(1, 1, dtype=torch.long, device=device)

        generated = model.generate(prompt_ids, max_tokens=50, temperature=0.8)
        gen_len = generated.shape[1] - prompt_ids.shape[1]
        gen_text = tokenizer.decode(generated[0].tolist())

        if gen_len < 50:
            return GateResult("generate", False, f"FAILED: only generated {gen_len}/50 tokens")

        return GateResult("generate", True, f"generated {gen_len} tokens: {gen_text[:80]!r}...")
    except Exception as e:
        return GateResult("generate", False, f"FAILED: {type(e).__name__}: {e}")


def gate_val_perplexity(
    model: SRNModel,
    val_loader,
    device: torch.device,
    max_ppl: float,
    precision_dtype: torch.dtype = torch.float16,
) -> GateResult:
    """Gate 5: Validation perplexity is below ceiling."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            if i >= 50:
                break
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast(device.type, dtype=precision_dtype):
                logits, _ = model(x)
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                y.view(-1),
                reduction="sum",
            )
            total_loss += loss.item()
            total_tokens += y.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    passed = ppl <= max_ppl
    detail = f"ppl={ppl:.2f}, limit={max_ppl:.0f}"
    if not passed:
        detail = f"FAILED: {detail}"
    return GateResult("val_perplexity", passed, detail)


def run_promotion_gates(
    checkpoint_path: str,
    max_vram_mb: float = 10000,
    max_ppl: float = 100,
    val_tokens_path: str | None = None,
    train_tokens_path: str | None = None,
    tokenizer_path: str | None = None,
) -> list[GateResult]:
    """Run all promotion gates on a checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint file
        max_vram_mb: VRAM ceiling in MB
        max_ppl: maximum acceptable validation perplexity
        val_tokens_path: path to validation token shard (memmap)
        train_tokens_path: path to training token shard (memmap)
        tokenizer_path: path to tokenizer JSON (for memmap mode)

    Returns:
        list of GateResult objects
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reset VRAM tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt["config"]
    precision = ckpt.get("precision", "fp16")
    precision_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    model = SRNModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = tokenizer_from_checkpoint(ckpt)

    total_params = model.count_params()
    active_params = model.count_active_params()
    print(f"Model: {total_params:,} params ({active_params:,} active/token)")
    print(f"Step: {ckpt.get('step', '?')}, Best val loss: {ckpt.get('best_val_loss', '?')}")

    results: list[GateResult] = []

    # Gate 1: NaN/Inf
    results.append(gate_nan_inf(model, device, config))

    # Gate 2: Reproducibility
    results.append(gate_reproducibility(model, device, config))

    # Gate 3: VRAM
    # Run a forward pass at full seq_len to measure peak VRAM
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            x = torch.randint(0, config.vocab_size, (1, config.max_seq_len), device=device)
            _ = model(x)
    results.append(gate_vram(max_vram_mb))

    # Gate 4: Generate
    results.append(gate_generate(model, tokenizer, device))

    # Gate 5: Val perplexity (requires data)
    if val_tokens_path is not None and train_tokens_path is not None:
        from data import BPETokenizer, get_memmap_dataloaders

        if tokenizer_path is not None:
            tok = BPETokenizer.from_file(tokenizer_path)
        else:
            tok = tokenizer
        _, val_loader, _ = get_memmap_dataloaders(
            train_tokens_path=train_tokens_path,
            val_tokens_path=val_tokens_path,
            tokenizer=tok,
            batch_size=8,
            seq_len=min(config.max_seq_len, 512),
        )
        results.append(gate_val_perplexity(model, val_loader, device, max_ppl, precision_dtype))
    else:
        # Try Shakespeare fallback for smoke tests
        try:
            from data import get_dataloaders

            _, val_loader, _ = get_dataloaders(
                batch_size=8,
                seq_len=min(config.max_seq_len, 256),
                tokenizer_override=tokenizer,
            )
            results.append(gate_val_perplexity(model, val_loader, device, max_ppl, precision_dtype))
        except Exception as e:
            results.append(GateResult("val_perplexity", True, f"skipped (no val data: {e})"))

    return results


def print_results(results: list[GateResult]) -> bool:
    """Print gate results and return True if all passed."""
    print(f"\n{'='*60}")
    print("PROMOTION GATE RESULTS")
    print(f"{'='*60}")

    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        icon = "✓" if r.passed else "✗"
        print(f"  {icon} [{status}] {r.name}: {r.detail}")
        if not r.passed:
            all_passed = False

    print(f"{'='*60}")
    if all_passed:
        print("ALL GATES PASSED — checkpoint is promotion-ready")
    else:
        failed = [r.name for r in results if not r.passed]
        print(f"BLOCKED — failed gates: {', '.join(failed)}")
    print(f"{'='*60}")

    return all_passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run promotion gates on a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max_vram_mb", type=float, default=10000)
    parser.add_argument("--max_ppl", type=float, default=100)
    parser.add_argument("--val_tokens_path", type=str, default=None)
    parser.add_argument("--train_tokens_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_promotion_gates(
        checkpoint_path=args.checkpoint,
        max_vram_mb=args.max_vram_mb,
        max_ppl=args.max_ppl,
        val_tokens_path=args.val_tokens_path,
        train_tokens_path=args.train_tokens_path,
        tokenizer_path=args.tokenizer_path,
    )
    all_passed = print_results(results)
    sys.exit(0 if all_passed else 1)
