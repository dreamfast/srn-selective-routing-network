"""Tests for Task 5: Experiment YAML Configs.

Verifies that all experiment config files parse correctly, contain no
unknown keys, and include required dataset backend fields.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train import parse_args

EXPERIMENTS_DIR = ROOT / "configs" / "experiments"


def _get_experiment_configs() -> list[Path]:
    """Get all experiment YAML config files."""
    return sorted(EXPERIMENTS_DIR.glob("*.yaml"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_all_experiment_configs_parse() -> None:
    """All experiment YAML configs parse without errors."""
    configs = _get_experiment_configs()
    assert len(configs) >= 21, (
        f"Expected at least 21 experiment configs, found {len(configs)}"
    )

    for config_path in configs:
        cfg = OmegaConf.load(config_path)
        assert cfg is not None, f"Failed to parse {config_path.name}"

        # Must have train and model sections
        assert "train" in cfg, f"{config_path.name} missing 'train' section"
        assert "model" in cfg, f"{config_path.name} missing 'model' section"

        # Train section must have key fields
        train = cfg.train
        assert "max_steps" in train, f"{config_path.name} missing train.max_steps"
        assert "micro_batch" in train, f"{config_path.name} missing train.micro_batch"
        assert "seq_len" in train, f"{config_path.name} missing train.seq_len"

        # Model section must have key fields
        model = cfg.model
        assert "d_model" in model, f"{config_path.name} missing model.d_model"
        assert "n_layers" in model, f"{config_path.name} missing model.n_layers"
        assert "max_seq_len" in model, f"{config_path.name} missing model.max_seq_len"


def test_no_unknown_keys() -> None:
    """No experiment config contains keys unknown to the CLI parser.

    This catches typos and ensures all config keys are wired to argparse.
    """
    configs = _get_experiment_configs()
    assert len(configs) > 0

    # Get all known argparse keys
    import sys as _sys
    _sys.argv = ["train.py"]  # Reset argv for clean parse
    args = parse_args()
    known_keys = set(vars(args).keys())

    # Also allow top-level keys that are flattened by _flatten_loaded_config
    # and section names themselves
    section_names = {"train", "model"}

    for config_path in configs:
        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        assert isinstance(cfg, dict), f"{config_path.name} top-level must be a mapping"

        # Check top-level keys
        for key in cfg:
            if key in section_names:
                continue
            assert key in known_keys, (
                f"{config_path.name}: unknown top-level key '{key}'"
            )

        # Check keys within train/model sections
        for section in ("train", "model"):
            section_values = cfg.get(section, {})
            if section_values is None:
                continue
            for key in section_values:
                assert key in known_keys, (
                    f"{config_path.name}: unknown key '{key}' in '{section}' section"
                )


def test_dataset_backend_present() -> None:
    """All experiment configs include dataset backend fields for memmap.

    These are required for TinyStories experiments to work without
    additional CLI flags.
    """
    configs = _get_experiment_configs()
    assert len(configs) > 0

    required_fields = [
        "dataset_backend",
        "tokenizer_backend",
        "tokenizer_path",
        "train_tokens_path",
        "val_tokens_path",
    ]

    for config_path in configs:
        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        assert isinstance(cfg, dict)

        for field in required_fields:
            assert field in cfg, (
                f"{config_path.name} missing required field '{field}'"
            )

        # Verify memmap backend
        assert cfg["dataset_backend"] == "memmap", (
            f"{config_path.name}: dataset_backend should be 'memmap', "
            f"got '{cfg['dataset_backend']}'"
        )
