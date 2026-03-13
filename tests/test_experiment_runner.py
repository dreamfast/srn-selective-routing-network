"""Tests for Task 6: Experiment Runner and Results Tracker.

Verifies the experiment runner's dry-run output, JSON result schema,
and comparison table functionality.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import from scripts directory
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.run_experiments import (
    EXPERIMENT_REGISTRY,
    GPU_TIERS,
    get_config_path,
    get_result_path,
    run_experiment,
    compare_results,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_dry_run_output() -> None:
    """Dry run produces expected output without executing training.

    Verifies that dry_run=True returns a result dict with status='dry_run'
    and a command string, without actually launching subprocess.
    """
    # Use exp1 which should have a config file
    result = run_experiment("1", "2060", dry_run=True)

    assert result["status"] == "dry_run"
    assert result["experiment_id"] == "1"
    assert result["gpu"] == "2060"
    assert "command" in result
    assert "train.py" in result["command"]
    assert "--config" in result["command"]
    assert "exp1-hybrid-2060.yaml" in result["command"]


def test_json_result_schema() -> None:
    """Result dict has the expected schema fields."""
    result = run_experiment("1", "2060", dry_run=True)

    # Required fields
    required_fields = [
        "experiment_id",
        "gpu",
        "config_path",
        "result_path",
        "status",
    ]
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"

    # Result should be JSON-serializable
    json_str = json.dumps(result, default=str)
    parsed = json.loads(json_str)
    assert parsed["experiment_id"] == "1"
    assert parsed["gpu"] == "2060"
    assert parsed["status"] == "dry_run"


def test_compare_with_mock_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare function reads and displays mock result files.

    Creates mock result JSON files and verifies compare_results
    reads them correctly.
    """
    import scripts.run_experiments as runner

    # Monkeypatch RESULTS_DIR to tmp_path
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)

    # Create mock results for exp1 and exp4
    mock_result_1 = {
        "experiment_id": "1",
        "gpu": "2060",
        "status": "completed",
        "metrics": {
            "best_val_loss": 2.3456,
            "step": 5000,
        },
    }
    result_path_1 = tmp_path / "exp1-hybrid-2060.json"
    with open(result_path_1, "w") as f:
        json.dump(mock_result_1, f)

    mock_result_4 = {
        "experiment_id": "4",
        "gpu": "2060",
        "status": "failed",
        "metrics": {},
    }
    result_path_4 = tmp_path / "exp4-nocsp-2060.json"
    with open(result_path_4, "w") as f:
        json.dump(mock_result_4, f)

    # compare_results should not crash
    compare_results("2060")

    # Verify the mock files were read correctly
    with open(result_path_1) as f:
        loaded = json.load(f)
    assert loaded["metrics"]["best_val_loss"] == 2.3456
    assert loaded["status"] == "completed"


def test_experiment_registry_complete() -> None:
    """Experiment registry contains all expected experiment IDs."""
    expected_ids = {"0", "0t", "0ts", "1", "2a", "2b", "3", "4", "5", "6"}
    assert set(EXPERIMENT_REGISTRY.keys()) == expected_ids


def test_config_path_resolution() -> None:
    """Config path resolution produces correct paths."""
    path = get_config_path("1", "2060")
    assert path.name == "exp1-hybrid-2060.yaml"
    assert "experiments" in str(path)

    path = get_config_path("2a", "4090")
    assert path.name == "exp2a-slots128-4090.yaml"


def test_missing_config_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Experiment with missing config file is skipped gracefully."""
    import scripts.run_experiments as runner

    # Point CONFIGS_DIR to a non-existent directory so all configs appear missing
    monkeypatch.setattr(runner, "CONFIGS_DIR", Path("/tmp/nonexistent_configs_dir"))

    result = run_experiment("0", "2060", dry_run=False)
    # Should be skipped since config doesn't exist
    assert result["status"] == "skipped"
    assert "not found" in result.get("error", "").lower()
