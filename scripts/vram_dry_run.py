"""VRAM dry-run gate for 1B-scale training.

Runs a synthetic forward+backward pass with the target model configuration
to measure peak VRAM usage before committing to a long training run.

The gate PASSES if peak VRAM is below (gpu_total * (1 - headroom_margin)).
Default headroom is 10% to leave room for CUDA context, fragmentation, and
PyTorch memory allocator overhead.

Usage:
    # SRN-1B on current GPU
    python scripts/vram_dry_run.py --config configs/srn-1b.yaml

    # Dense baseline
    python scripts/vram_dry_run.py --config configs/dense-067b.yaml

    # Custom headroom
    python scripts/vram_dry_run.py --config configs/srn-1b.yaml --headroom 0.15

    # Specify target GPU memory (for planning without the actual GPU)
    python scripts/vram_dry_run.py --config configs/srn-1b.yaml --gpu_mem_mb 32768
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omegaconf import OmegaConf

from dense_model import DenseConfig, DenseGPT
from srn_model import SRNConfig, SRNModel
from train import resolve_precision


@dataclass
class DryRunResult:
    """Result of a VRAM dry-run profiling pass."""

    model_type: str
    total_params: int
    active_params: int
    peak_vram_mb: float
    gpu_total_mb: float
    headroom: float
    budget_mb: float
    passed: bool
    forward_time_ms: float
    backward_time_ms: float
    precision: str


def build_model(
    cfg: dict, vocab_size: int, device: torch.device
) -> tuple:
    """Build model from flattened config dict.

    Returns:
        (model, config, model_type) tuple
    """
    model_section = cfg.get("model", {})
    train_section = cfg.get("train", {})
    model_type = train_section.get("model_type", "srn")

    if model_type == "dense":
        d_model = model_section.get("d_model", 1024)
        config = DenseConfig(
            vocab_size=vocab_size,
            max_seq_len=model_section.get("max_seq_len", 2048),
            d_model=d_model,
            n_layers=model_section.get("n_layers", 16),
            n_heads=model_section.get("n_heads", 16),
            d_ff=model_section.get("d_ff", d_model * 4),
            dropout=0.0,  # No dropout for profiling
            bias=model_section.get("bias", False),
        )
        model = DenseGPT(config).to(device)
    else:
        config = SRNConfig(
            vocab_size=vocab_size,
            max_seq_len=model_section.get("max_seq_len", 2048),
            d_model=model_section.get("d_model", 1792),
            d_compressed=model_section.get("d_compressed", 448),
            n_layers=model_section.get("n_layers", 16),
            n_memory_slots=model_section.get("n_memory_slots", 192),
            n_experts=model_section.get("n_experts", 16),
            top_k_experts=model_section.get("top_k_experts", 2),
            d_expert=model_section.get("d_expert", 896),
            n_heads_route=model_section.get("n_heads_route", 16),
            dropout=0.0,  # No dropout for profiling
            causal_window=model_section.get("causal_window", 128),
            csp_internal_residual=model_section.get("csp_internal_residual", False),
            aux_loss_weight=model_section.get("aux_loss_weight", 0.01),
            sparse_moe=model_section.get("sparse_moe", False),
        )
        model = SRNModel(config).to(device)

    return model, config, model_type


def run_dry_run(
    config_path: str,
    headroom: float = 0.10,
    gpu_mem_mb: float | None = None,
    precision: str = "fp16",
    vocab_size: int = 32000,
    micro_batch: int | None = None,
    seq_len: int | None = None,
) -> DryRunResult:
    """Run VRAM dry-run profiling.

    Args:
        config_path: path to YAML config file
        headroom: fraction of GPU memory to reserve (0.10 = 10%)
        gpu_mem_mb: override GPU total memory (for planning)
        precision: precision mode ("fp16" or "bf16")
        vocab_size: vocabulary size for the model
        micro_batch: override micro batch size from config
        seq_len: override sequence length from config

    Returns:
        DryRunResult with profiling data and pass/fail verdict
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        raise RuntimeError(
            "VRAM dry-run requires CUDA. Run inside Docker with GPU access."
        )

    # Validate inputs
    if not (0.0 <= headroom < 1.0):
        raise ValueError(f"headroom must be in [0.0, 1.0), got {headroom}")
    if gpu_mem_mb is not None and gpu_mem_mb <= 0:
        raise ValueError(f"gpu_mem_mb must be positive, got {gpu_mem_mb}")

    # Load config
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping")

    train_section = cfg.get("train", {})
    if micro_batch is None:
        micro_batch = train_section.get("micro_batch", 2)
    if seq_len is None:
        seq_len = train_section.get("seq_len", 2048)

    if micro_batch <= 0:
        raise ValueError(f"micro_batch must be positive, got {micro_batch}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")

    # Resolve precision
    precision_dtype = resolve_precision(precision)
    precision_label = "bf16" if precision_dtype == torch.bfloat16 else "fp16"

    # GPU memory budget
    if gpu_mem_mb is not None:
        gpu_total_mb = gpu_mem_mb
    else:
        gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2

    budget_mb = gpu_total_mb * (1 - headroom)

    # Reset memory tracking
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    # Build model
    print(f"Building model from {config_path}...")
    model, config, model_type = build_model(cfg, vocab_size, device)

    total_params = model.count_params()
    active_params = model.count_active_params()
    model_label = "Dense GPT" if model_type == "dense" else "SRN"

    print(f"  Model: {model_label}")
    print(f"  Params: {total_params:,} total, {active_params:,} active/token")

    # Optimizer (needed for backward pass memory)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    use_scaler = precision_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # Synthetic data
    x = torch.randint(0, vocab_size, (micro_batch, seq_len), device=device)
    y = torch.randint(0, vocab_size, (micro_batch, seq_len), device=device)

    # Reset after setup
    torch.cuda.reset_peak_memory_stats()

    # Forward pass
    print(f"\n  Forward pass (batch={micro_batch}, seq_len={seq_len}, precision={precision_label})...")
    torch.cuda.synchronize()
    t0 = time.time()

    optimizer.zero_grad()
    with torch.amp.autocast(device.type, dtype=precision_dtype):
        logits, aux_loss = model(x)
        ce_loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), y.view(-1)
        )
        aux_weight = getattr(config, "aux_loss_weight", 0.0)
        loss = ce_loss + aux_weight * aux_loss

    torch.cuda.synchronize()
    forward_ms = (time.time() - t0) * 1000

    # Backward pass
    print(f"  Backward pass...")
    torch.cuda.synchronize()
    t0 = time.time()

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    torch.cuda.synchronize()
    backward_ms = (time.time() - t0) * 1000

    # Measure peak VRAM
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Cleanup
    del model, optimizer, scaler, x, y, logits, aux_loss, ce_loss, loss
    torch.cuda.empty_cache()

    passed = peak_vram_mb <= budget_mb

    return DryRunResult(
        model_type=model_type,
        total_params=total_params,
        active_params=active_params,
        peak_vram_mb=peak_vram_mb,
        gpu_total_mb=gpu_total_mb,
        headroom=headroom,
        budget_mb=budget_mb,
        passed=passed,
        forward_time_ms=forward_ms,
        backward_time_ms=backward_ms,
        precision=precision_label,
    )


def print_result(result: DryRunResult) -> None:
    """Print dry-run result with pass/fail verdict."""
    icon = "✓" if result.passed else "✗"
    status = "PASS" if result.passed else "FAIL"

    print(f"\n{'='*60}")
    print(f"VRAM DRY-RUN RESULT: [{status}] {icon}")
    print(f"{'='*60}")
    print(f"  Model type:     {result.model_type}")
    print(f"  Total params:   {result.total_params:>12,}")
    print(f"  Active/token:   {result.active_params:>12,}")
    print(f"  Precision:      {result.precision}")
    print(f"  Peak VRAM:      {result.peak_vram_mb:>10.1f} MB")
    print(f"  GPU total:      {result.gpu_total_mb:>10.1f} MB")
    print(f"  Headroom:       {result.headroom*100:>10.0f}%")
    print(f"  Budget:         {result.budget_mb:>10.1f} MB")
    print(f"  Utilization:    {result.peak_vram_mb/result.gpu_total_mb*100:>10.1f}%")
    print(f"  Forward time:   {result.forward_time_ms:>10.1f} ms")
    print(f"  Backward time:  {result.backward_time_ms:>10.1f} ms")

    if result.passed:
        margin = result.budget_mb - result.peak_vram_mb
        print(f"  Margin:         {margin:>10.1f} MB ({margin/result.gpu_total_mb*100:.1f}%)")
    else:
        overshoot = result.peak_vram_mb - result.budget_mb
        print(f"  OVER BUDGET BY: {overshoot:>10.1f} MB")
        print(f"\n  *** DO NOT proceed with training — will OOM ***")

    print(f"{'='*60}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VRAM dry-run gate for 1B-scale training"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--headroom", type=float, default=0.10,
        help="Fraction of GPU memory to reserve (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--gpu_mem_mb", type=float, default=None,
        help="Override GPU total memory in MB (for planning)",
    )
    parser.add_argument(
        "--precision", type=str, choices=["fp16", "bf16"], default="fp16",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=32000,
    )
    parser.add_argument(
        "--micro_batch", type=int, default=None,
        help="Override micro batch size from config",
    )
    parser.add_argument(
        "--seq_len", type=int, default=None,
        help="Override sequence length from config",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_dry_run(
        config_path=args.config,
        headroom=args.headroom,
        gpu_mem_mb=args.gpu_mem_mb,
        precision=args.precision,
        vocab_size=args.vocab_size,
        micro_batch=args.micro_batch,
        seq_len=args.seq_len,
    )
    print_result(result)
    sys.exit(0 if result.passed else 1)
