#!/usr/bin/env python3
"""
Experiment Runner for SRN Ablation Studies
===========================================

Runs ablation experiments defined by YAML configs in configs/experiments/.
Supports selective experiment execution, GPU tier targeting, dry-run mode,
and structured results tracking.

Usage:
    python scripts/run_experiments.py --experiments 1,2a,3 --gpu 2060
    python scripts/run_experiments.py --all --gpu 4090
    python scripts/run_experiments.py --dry-run --all --gpu 2060
    python scripts/run_experiments.py --compare --gpu 2060

Results are saved as structured JSON in results/{experiment}-{gpu}.json
and can be compared with --compare.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

# Project root (parent of scripts/)
ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = ROOT / "configs" / "experiments"
RESULTS_DIR = ROOT / "results"

# Experiment registry: ID -> config file prefix
EXPERIMENT_REGISTRY = {
    "0": "exp0-srn",
    "0t": "exp0-transformer",
    "0ts": "exp0-transformer-small",
    "1": "exp1-hybrid",
    "2a": "exp2a-slots128",
    "2b": "exp2b-slots256",
    "3": "exp3-full",
    "4": "exp4-nocsp",
    "5": "exp5-topk4",
    "6": "exp6-wcsg",
}

GPU_TIERS = ["2060", "4090", "5090"]


def get_config_path(experiment_id: str, gpu: str) -> Path:
    """Get the config file path for an experiment and GPU tier."""
    prefix = EXPERIMENT_REGISTRY.get(experiment_id)
    if prefix is None:
        raise ValueError(
            f"Unknown experiment ID: {experiment_id!r}. "
            f"Valid IDs: {', '.join(sorted(EXPERIMENT_REGISTRY.keys()))}"
        )
    return CONFIGS_DIR / f"{prefix}-{gpu}.yaml"


def get_result_path(experiment_id: str, gpu: str) -> Path:
    """Get the result JSON path for an experiment and GPU tier."""
    prefix = EXPERIMENT_REGISTRY.get(experiment_id, experiment_id)
    return RESULTS_DIR / f"{prefix}-{gpu}.json"


def collect_results(checkpoint_dir: str) -> dict[str, Any]:
    """Collect training results from a checkpoint directory.

    Loads the best checkpoint and extracts metrics. Falls back to
    latest checkpoint if best doesn't exist.

    Args:
        checkpoint_dir: path to the checkpoint directory

    Returns:
        dict with training metrics (val_loss, step, config, etc.)
    """
    ckpt_dir = Path(checkpoint_dir)
    results: dict[str, Any] = {"checkpoint_dir": str(ckpt_dir)}

    # Try best.pt first, then latest.pt, then final.pt
    for ckpt_name in ("best.pt", "latest.pt", "final.pt"):
        ckpt_path = ckpt_dir / ckpt_name
        if ckpt_path.exists():
            try:
                import torch
                # weights_only=False needed to deserialize SRNConfig/DenseConfig
                # dataclass stored in checkpoint. Only load self-created checkpoints.
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                results["best_val_loss"] = ckpt.get("best_val_loss")
                results["step"] = ckpt.get("step")
                results["checkpoint_file"] = ckpt_name

                # Extract config info and param count from state dict
                config = ckpt.get("config")
                if config is not None:
                    results["config_type"] = type(config).__name__
                model_state = ckpt.get("model")
                if model_state is not None:
                    results["total_params"] = sum(
                        v.numel() for v in model_state.values()
                    )
                break
            except ImportError as e:
                results["load_error"] = f"PyTorch not available: {e}"
            except Exception as e:
                results["load_error"] = str(e)

    return results


def run_experiment(
    experiment_id: str,
    gpu: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a single experiment.

    Args:
        experiment_id: experiment ID (e.g. "1", "2a")
        gpu: GPU tier (e.g. "2060", "4090")
        dry_run: if True, print command but don't execute

    Returns:
        dict with experiment result metadata
    """
    config_path = get_config_path(experiment_id, gpu)
    result_path = get_result_path(experiment_id, gpu)

    result: dict[str, Any] = {
        "experiment_id": experiment_id,
        "gpu": gpu,
        "config_path": str(config_path),
        "result_path": str(result_path),
        "status": "pending",
    }

    if not config_path.exists():
        result["status"] = "skipped"
        result["error"] = f"Config file not found: {config_path}"
        print(f"  SKIP: {config_path.name} (not found)")
        return result

    # Build command
    cmd = [sys.executable, str(ROOT / "train.py"), "--config", str(config_path)]

    if dry_run:
        result["status"] = "dry_run"
        result["command"] = " ".join(cmd)
        print(f"  DRY RUN: {' '.join(cmd)}")
        return result

    # Execute — redirect stdout/stderr to log files to avoid pipe buffer
    # deadlock on long-running training (24h+). Logs are saved alongside results.
    print(f"  Running: {config_path.name}")
    result["start_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = EXPERIMENT_REGISTRY.get(experiment_id, experiment_id)
    log_path = RESULTS_DIR / f"{prefix}-{gpu}.log"

    try:
        with open(log_path, "w") as log_file:
            proc = subprocess.run(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(ROOT),
                timeout=86400,  # 24h timeout
            )
        result["returncode"] = proc.returncode
        result["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        result["log_path"] = str(log_path)

        if proc.returncode == 0:
            result["status"] = "completed"
            # Collect results from checkpoint
            from omegaconf import OmegaConf
            cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
            ckpt_dir = cfg.get("train", {}).get("checkpoint_dir", "checkpoints")
            result["metrics"] = collect_results(ckpt_dir)
        else:
            result["status"] = "failed"
            # Read last 2000 chars of log for error context
            try:
                log_content = log_path.read_text()
                result["stderr"] = log_content[-2000:] if log_content else ""
            except Exception:
                result["stderr"] = ""
            print(f"  FAILED (exit code {proc.returncode}). See {log_path}")

    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"] = "Exceeded 24h timeout"
        print(f"  TIMEOUT: {config_path.name}")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  ERROR: {e}")

    return result


def save_result(result: dict[str, Any]) -> None:
    """Save experiment result to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = Path(result.get("result_path", ""))
    if result_path.suffix == ".json":
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"  Result saved: {result_path}")


def compare_results(gpu: str) -> None:
    """Print comparison table of all experiment results for a GPU tier.

    Args:
        gpu: GPU tier to compare results for
    """
    print(f"\n{'='*70}")
    print(f"Experiment Results Comparison — {gpu}")
    print(f"{'='*70}")
    print(f"{'Experiment':<25} {'Status':<12} {'Val Loss':<12} {'Steps':<8}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*8}")

    for exp_id in sorted(EXPERIMENT_REGISTRY.keys()):
        result_path = get_result_path(exp_id, gpu)
        prefix = EXPERIMENT_REGISTRY[exp_id]

        if not result_path.exists():
            print(f"{prefix:<25} {'no result':<12} {'—':<12} {'—':<8}")
            continue

        with open(result_path) as f:
            result = json.load(f)

        status = result.get("status", "unknown")
        metrics = result.get("metrics", {})
        val_loss = metrics.get("best_val_loss")
        step = metrics.get("step")

        val_str = f"{val_loss:.4f}" if val_loss is not None else "—"
        step_str = str(step) if step is not None else "—"

        print(f"{prefix:<25} {status:<12} {val_str:<12} {step_str:<8}")

    print(f"{'='*70}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SRN ablation experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_experiments.py --experiments 1,2a,3 --gpu 2060
    python scripts/run_experiments.py --all --gpu 4090
    python scripts/run_experiments.py --dry-run --all --gpu 2060
    python scripts/run_experiments.py --compare --gpu 2060
        """,
    )

    parser.add_argument(
        "--experiments", type=str, default=None,
        help="Comma-separated experiment IDs (e.g. '1,2a,3'). "
             f"Valid: {', '.join(sorted(EXPERIMENT_REGISTRY.keys()))}",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all experiments",
    )
    parser.add_argument(
        "--gpu", type=str, required=True, choices=GPU_TIERS,
        help="GPU tier to run experiments for",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare results for the specified GPU tier",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.compare:
        compare_results(args.gpu)
        return

    # Determine which experiments to run
    if args.all:
        experiment_ids = sorted(EXPERIMENT_REGISTRY.keys())
    elif args.experiments:
        experiment_ids = [e.strip() for e in args.experiments.split(",")]
        # Validate
        for eid in experiment_ids:
            if eid not in EXPERIMENT_REGISTRY:
                print(f"ERROR: Unknown experiment ID: {eid!r}")
                print(f"Valid IDs: {', '.join(sorted(EXPERIMENT_REGISTRY.keys()))}")
                sys.exit(1)
    else:
        print("ERROR: Specify --experiments or --all")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"SRN Ablation Experiments — GPU: {args.gpu}")
    print(f"Experiments: {', '.join(experiment_ids)}")
    if args.dry_run:
        print("Mode: DRY RUN")
    print(f"{'='*60}\n")

    # Run experiments
    results = []
    for exp_id in experiment_ids:
        prefix = EXPERIMENT_REGISTRY[exp_id]
        print(f"\n[{exp_id}] {prefix}")
        result = run_experiment(exp_id, args.gpu, dry_run=args.dry_run)
        results.append(result)

        # Save result (even for failures/dry-runs)
        if not args.dry_run:
            save_result(result)

    # Summary
    print(f"\n{'='*60}")
    print("Summary:")
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    dry_run = sum(1 for r in results if r["status"] == "dry_run")

    if dry_run > 0:
        print(f"  Dry run: {dry_run} experiments")
    else:
        print(f"  Completed: {completed}, Failed: {failed}, Skipped: {skipped}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
