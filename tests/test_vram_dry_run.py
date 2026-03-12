"""Tests for Task 9: VRAM dry-run gate.

Verifies the dry-run profiling script correctly builds models, measures VRAM,
and reports pass/fail based on headroom margin.

Tests that require CUDA are marked with @pytest.mark.skipif and will be
skipped gracefully on CPU-only machines. CUDA integration tests use tiny
model configs to fit within the RTX 2060's 6GB VRAM.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

import torch
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vram_dry_run import DryRunResult, build_model, print_result, run_dry_run

HAS_CUDA = torch.cuda.is_available()
SKIP_NO_CUDA = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")

# Tiny SRN config YAML for CUDA tests (fits easily in 6GB)
TINY_SRN_YAML = """\
train:
  micro_batch: 1
  seq_len: 32
model:
  max_seq_len: 64
  d_model: 64
  d_compressed: 16
  n_layers: 2
  n_memory_slots: 8
  n_experts: 4
  top_k_experts: 2
  d_expert: 32
  n_heads_route: 4
  causal_window: 8
  csp_internal_residual: false
  aux_loss_weight: 0.01
  sparse_moe: false
"""

# Tiny Dense config YAML for CUDA tests
TINY_DENSE_YAML = """\
train:
  micro_batch: 1
  seq_len: 32
  model_type: dense
model:
  max_seq_len: 64
  d_model: 64
  n_layers: 2
  n_heads: 4
  d_ff: 256
  bias: false
"""


@pytest.fixture
def tiny_srn_config(tmp_path: Path) -> str:
    """Write tiny SRN config to a temp file and return its path."""
    p = tmp_path / "tiny_srn.yaml"
    p.write_text(TINY_SRN_YAML)
    return str(p)


@pytest.fixture
def tiny_dense_config(tmp_path: Path) -> str:
    """Write tiny Dense config to a temp file and return its path."""
    p = tmp_path / "tiny_dense.yaml"
    p.write_text(TINY_DENSE_YAML)
    return str(p)


def _cuda_cleanup() -> None:
    """Force CUDA memory cleanup between tests."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def passing_result() -> DryRunResult:
    """A DryRunResult that passes the headroom check."""
    return DryRunResult(
        model_type="srn",
        total_params=1_150_000_000,
        active_params=463_000_000,
        peak_vram_mb=22000.0,
        gpu_total_mb=32768.0,
        headroom=0.10,
        budget_mb=29491.2,
        passed=True,
        forward_time_ms=150.0,
        backward_time_ms=300.0,
        precision="fp16",
    )


@pytest.fixture
def failing_result() -> DryRunResult:
    """A DryRunResult that fails the headroom check."""
    return DryRunResult(
        model_type="dense",
        total_params=670_000_000,
        active_params=670_000_000,
        peak_vram_mb=31000.0,
        gpu_total_mb=32768.0,
        headroom=0.10,
        budget_mb=29491.2,
        passed=False,
        forward_time_ms=120.0,
        backward_time_ms=250.0,
        precision="fp16",
    )


# ---------------------------------------------------------------------------
# DryRunResult dataclass tests
# ---------------------------------------------------------------------------


def test_dry_run_result_fields(passing_result: DryRunResult) -> None:
    """DryRunResult has all expected fields."""
    assert passing_result.model_type == "srn"
    assert passing_result.total_params == 1_150_000_000
    assert passing_result.active_params == 463_000_000
    assert passing_result.peak_vram_mb == 22000.0
    assert passing_result.gpu_total_mb == 32768.0
    assert passing_result.headroom == 0.10
    assert passing_result.budget_mb == 29491.2
    assert passing_result.passed is True
    assert passing_result.forward_time_ms == 150.0
    assert passing_result.backward_time_ms == 300.0
    assert passing_result.precision == "fp16"


def test_dry_run_result_pass_logic() -> None:
    """Pass/fail is determined by peak_vram_mb <= budget_mb."""
    # Exactly at budget -> pass
    result = DryRunResult(
        model_type="srn",
        total_params=100,
        active_params=100,
        peak_vram_mb=1000.0,
        gpu_total_mb=2000.0,
        headroom=0.50,
        budget_mb=1000.0,
        passed=True,
        forward_time_ms=1.0,
        backward_time_ms=1.0,
        precision="fp16",
    )
    assert result.passed is True
    assert result.peak_vram_mb <= result.budget_mb


# ---------------------------------------------------------------------------
# build_model tests (CPU only — no CUDA needed)
# ---------------------------------------------------------------------------


def test_build_srn_model() -> None:
    """build_model creates an SRN model from config dict."""
    cfg = {
        "model": {
            "max_seq_len": 32,
            "d_model": 64,
            "d_compressed": 16,
            "n_layers": 2,
            "n_memory_slots": 8,
            "n_experts": 4,
            "top_k_experts": 2,
            "d_expert": 32,
            "n_heads_route": 4,
            "causal_window": 8,
            "csp_internal_residual": False,
            "aux_loss_weight": 0.01,
            "sparse_moe": False,
        },
        "train": {},
    }
    device = torch.device("cpu")
    model, config, model_type = build_model(cfg, vocab_size=128, device=device)

    assert model_type == "srn"
    assert config.d_model == 64
    assert config.n_layers == 2
    assert model.count_params() > 0


def test_build_dense_model() -> None:
    """build_model creates a Dense model when model_type is 'dense'."""
    cfg = {
        "model": {
            "max_seq_len": 32,
            "d_model": 64,
            "n_layers": 2,
            "n_heads": 4,
            "d_ff": 256,
            "bias": False,
        },
        "train": {"model_type": "dense"},
    }
    device = torch.device("cpu")
    model, config, model_type = build_model(cfg, vocab_size=128, device=device)

    assert model_type == "dense"
    assert config.d_model == 64
    assert config.n_layers == 2
    assert model.count_params() > 0
    assert model.count_params() == model.count_active_params()


def test_build_model_default_type() -> None:
    """build_model defaults to SRN when model_type is not specified."""
    cfg = {
        "model": {
            "d_model": 64,
            "d_compressed": 16,
            "n_layers": 2,
            "n_memory_slots": 8,
            "n_experts": 4,
            "top_k_experts": 2,
            "d_expert": 32,
            "n_heads_route": 4,
            "causal_window": 8,
        },
        "train": {},
    }
    device = torch.device("cpu")
    _, _, model_type = build_model(cfg, vocab_size=128, device=device)
    assert model_type == "srn"


def test_build_model_sparse_moe() -> None:
    """build_model respects sparse_moe flag."""
    cfg = {
        "model": {
            "d_model": 64,
            "d_compressed": 16,
            "n_layers": 2,
            "n_memory_slots": 8,
            "n_experts": 4,
            "top_k_experts": 2,
            "d_expert": 32,
            "n_heads_route": 4,
            "causal_window": 8,
            "sparse_moe": True,
        },
        "train": {},
    }
    device = torch.device("cpu")
    model, config, _ = build_model(cfg, vocab_size=128, device=device)
    assert config.sparse_moe is True


# ---------------------------------------------------------------------------
# print_result tests
# ---------------------------------------------------------------------------


def test_print_result_pass(passing_result: DryRunResult, capsys) -> None:
    """print_result shows PASS and margin for passing results."""
    print_result(passing_result)
    captured = capsys.readouterr()

    assert "PASS" in captured.out
    assert "Margin:" in captured.out
    assert "srn" in captured.out
    assert "fp16" in captured.out
    assert "DO NOT proceed" not in captured.out


def test_print_result_fail(failing_result: DryRunResult, capsys) -> None:
    """print_result shows FAIL and overshoot for failing results."""
    print_result(failing_result)
    captured = capsys.readouterr()

    assert "FAIL" in captured.out
    assert "OVER BUDGET BY" in captured.out
    assert "DO NOT proceed" in captured.out
    assert "dense" in captured.out


def test_print_result_contains_params(passing_result: DryRunResult, capsys) -> None:
    """print_result displays parameter counts."""
    print_result(passing_result)
    captured = capsys.readouterr()

    assert "1,150,000,000" in captured.out
    assert "463,000,000" in captured.out


# ---------------------------------------------------------------------------
# run_dry_run integration tests (CUDA only, tiny models)
# ---------------------------------------------------------------------------


@SKIP_NO_CUDA
def test_run_dry_run_srn_small(tiny_srn_config: str) -> None:
    """run_dry_run completes for a tiny SRN config on GPU."""
    _cuda_cleanup()
    result = run_dry_run(
        config_path=tiny_srn_config,
        headroom=0.10,
        precision="fp16",
        vocab_size=256,
        micro_batch=1,
        seq_len=32,
    )

    assert isinstance(result, DryRunResult)
    assert result.model_type == "srn"
    assert result.peak_vram_mb > 0
    assert result.forward_time_ms > 0
    assert result.backward_time_ms > 0
    assert result.total_params > 0
    assert result.gpu_total_mb > 0
    _cuda_cleanup()


@SKIP_NO_CUDA
def test_run_dry_run_dense_small(tiny_dense_config: str) -> None:
    """run_dry_run completes for a tiny dense config on GPU."""
    _cuda_cleanup()
    result = run_dry_run(
        config_path=tiny_dense_config,
        headroom=0.10,
        precision="fp16",
        vocab_size=256,
        micro_batch=1,
        seq_len=32,
    )

    assert isinstance(result, DryRunResult)
    assert result.model_type == "dense"
    assert result.peak_vram_mb > 0
    assert result.total_params > 0
    _cuda_cleanup()


@SKIP_NO_CUDA
def test_run_dry_run_gpu_mem_override(tiny_srn_config: str) -> None:
    """run_dry_run respects gpu_mem_mb override for budget calculation."""
    _cuda_cleanup()
    result = run_dry_run(
        config_path=tiny_srn_config,
        headroom=0.10,
        gpu_mem_mb=100000.0,  # 100 GB — should always pass
        precision="fp16",
        vocab_size=256,
        micro_batch=1,
        seq_len=32,
    )

    assert result.gpu_total_mb == 100000.0
    assert result.budget_mb == 90000.0
    assert result.passed is True
    _cuda_cleanup()


@SKIP_NO_CUDA
def test_run_dry_run_tight_budget_fails(tiny_srn_config: str) -> None:
    """run_dry_run fails when gpu_mem_mb is too small."""
    _cuda_cleanup()
    result = run_dry_run(
        config_path=tiny_srn_config,
        headroom=0.10,
        gpu_mem_mb=1.0,  # 1 MB — impossible to fit any model
        precision="fp16",
        vocab_size=256,
        micro_batch=1,
        seq_len=32,
    )

    assert result.passed is False
    assert result.peak_vram_mb > result.budget_mb
    _cuda_cleanup()


@SKIP_NO_CUDA
def test_run_dry_run_headroom_affects_budget(tiny_srn_config: str) -> None:
    """Different headroom values produce different budgets."""
    _cuda_cleanup()

    result_10 = run_dry_run(
        config_path=tiny_srn_config,
        headroom=0.10,
        gpu_mem_mb=10000.0,
        precision="fp16",
        vocab_size=256,
        micro_batch=1,
        seq_len=32,
    )
    _cuda_cleanup()

    result_50 = run_dry_run(
        config_path=tiny_srn_config,
        headroom=0.50,
        gpu_mem_mb=10000.0,
        precision="fp16",
        vocab_size=256,
        micro_batch=1,
        seq_len=32,
    )
    _cuda_cleanup()

    assert result_10.budget_mb == 9000.0
    assert result_50.budget_mb == 5000.0
    assert result_10.budget_mb > result_50.budget_mb


def test_run_dry_run_requires_cuda_on_cpu() -> None:
    """run_dry_run raises RuntimeError when CUDA is not available."""
    if HAS_CUDA:
        pytest.skip("Test only meaningful on CPU-only machines")

    config_path = str(ROOT / "configs" / "base.yaml")
    with pytest.raises(RuntimeError, match="CUDA"):
        run_dry_run(config_path=config_path)


# ---------------------------------------------------------------------------
# Input validation tests (CUDA required for run_dry_run to reach validation)
# ---------------------------------------------------------------------------


@SKIP_NO_CUDA
def test_run_dry_run_rejects_headroom_too_high(tiny_srn_config: str) -> None:
    """run_dry_run rejects headroom >= 1.0."""
    with pytest.raises(ValueError, match="headroom"):
        run_dry_run(config_path=tiny_srn_config, headroom=1.5)


@SKIP_NO_CUDA
def test_run_dry_run_rejects_negative_headroom(tiny_srn_config: str) -> None:
    """run_dry_run rejects negative headroom."""
    with pytest.raises(ValueError, match="headroom"):
        run_dry_run(config_path=tiny_srn_config, headroom=-0.1)


@SKIP_NO_CUDA
def test_run_dry_run_rejects_zero_gpu_mem(tiny_srn_config: str) -> None:
    """run_dry_run rejects gpu_mem_mb <= 0."""
    with pytest.raises(ValueError, match="gpu_mem_mb"):
        run_dry_run(config_path=tiny_srn_config, gpu_mem_mb=0.0)


@SKIP_NO_CUDA
def test_run_dry_run_rejects_negative_gpu_mem(tiny_srn_config: str) -> None:
    """run_dry_run rejects negative gpu_mem_mb."""
    with pytest.raises(ValueError, match="gpu_mem_mb"):
        run_dry_run(config_path=tiny_srn_config, gpu_mem_mb=-100.0)


@SKIP_NO_CUDA
def test_run_dry_run_rejects_zero_micro_batch(tiny_srn_config: str) -> None:
    """run_dry_run rejects micro_batch <= 0."""
    with pytest.raises(ValueError, match="micro_batch"):
        run_dry_run(config_path=tiny_srn_config, micro_batch=0)


@SKIP_NO_CUDA
def test_run_dry_run_rejects_zero_seq_len(tiny_srn_config: str) -> None:
    """run_dry_run rejects seq_len <= 0."""
    with pytest.raises(ValueError, match="seq_len"):
        run_dry_run(config_path=tiny_srn_config, seq_len=0)


# ---------------------------------------------------------------------------
# CLI argument parsing tests
# ---------------------------------------------------------------------------


def test_parse_args_required_config() -> None:
    """parse_args requires --config."""
    from scripts.vram_dry_run import parse_args

    # Temporarily override sys.argv
    old_argv = sys.argv
    try:
        sys.argv = ["vram_dry_run.py"]
        with pytest.raises(SystemExit):
            parse_args()
    finally:
        sys.argv = old_argv


def test_parse_args_defaults() -> None:
    """parse_args has correct defaults."""
    from scripts.vram_dry_run import parse_args

    old_argv = sys.argv
    try:
        sys.argv = ["vram_dry_run.py", "--config", "configs/srn-1b.yaml"]
        args = parse_args()
        assert args.config == "configs/srn-1b.yaml"
        assert args.headroom == 0.10
        assert args.gpu_mem_mb is None
        assert args.precision == "fp16"
        assert args.vocab_size == 32000
        assert args.micro_batch is None
        assert args.seq_len is None
    finally:
        sys.argv = old_argv


def test_parse_args_all_flags() -> None:
    """parse_args parses all flags correctly."""
    from scripts.vram_dry_run import parse_args

    old_argv = sys.argv
    try:
        sys.argv = [
            "vram_dry_run.py",
            "--config", "configs/dense-067b.yaml",
            "--headroom", "0.15",
            "--gpu_mem_mb", "32768",
            "--precision", "bf16",
            "--vocab_size", "50000",
            "--micro_batch", "4",
            "--seq_len", "1024",
        ]
        args = parse_args()
        assert args.config == "configs/dense-067b.yaml"
        assert args.headroom == 0.15
        assert args.gpu_mem_mb == 32768.0
        assert args.precision == "bf16"
        assert args.vocab_size == 50000
        assert args.micro_batch == 4
        assert args.seq_len == 1024
    finally:
        sys.argv = old_argv
