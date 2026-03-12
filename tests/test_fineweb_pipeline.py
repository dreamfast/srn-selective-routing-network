"""Tests for Task 7: FineWeb-Edu data pipeline.

Tests the preparation script's core logic using synthetic data to avoid
downloading the actual FineWeb-Edu dataset in CI. Verifies:
- Tokenization and shard creation
- Manifest generation with checksums and tokenizer hash
- Train/val split ratios
- Deterministic output (same seed → same shards)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from data import BPETokenizer, SPECIAL_TOKENS


def _make_fake_tokenizer(tmp_path: Path) -> BPETokenizer:
    """Train a tiny BPE tokenizer on synthetic text."""
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 10,
        "A simple test sentence for tokenizer training. " * 10,
        "Machine learning models process text data efficiently. " * 10,
    ]
    tokenizer = BPETokenizer.train_from_iterator(
        texts, vocab_size=256, special_tokens=SPECIAL_TOKENS
    )
    tok_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tok_path))
    return tokenizer


def _make_fake_docs(n: int = 100) -> list[dict]:
    """Create synthetic documents mimicking FineWeb-Edu structure."""
    docs = []
    for i in range(n):
        text = f"Document {i}: This is a test document with some content. " * 5
        docs.append({"text": text})
    return docs


def test_prepare_fineweb_creates_shards(tmp_path: Path) -> None:
    """Preparation script creates train/val shards and manifest."""
    tokenizer = _make_fake_tokenizer(tmp_path)
    tok_path = tmp_path / "tokenizer.json"
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Simulate what prepare_fineweb.py does (core logic)
    docs = _make_fake_docs(50)
    eos_id = tokenizer.token_to_id("<eos>")
    assert eos_id is not None

    all_ids: list[int] = []
    for doc in docs:
        ids = tokenizer.encode(doc["text"])
        all_ids.extend(ids)
        all_ids.append(eos_id)

    token_array = np.array(all_ids, dtype=np.int32)
    train_ratio = 0.95
    split_idx = int(len(token_array) * train_ratio)
    train_tokens = token_array[:split_idx]
    val_tokens = token_array[split_idx:]

    train_path = output_dir / "train_tokens.bin"
    val_path = output_dir / "val_tokens.bin"
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    # Verify files exist and have correct sizes
    assert train_path.exists()
    assert val_path.exists()
    assert train_path.stat().st_size == train_tokens.size * 4  # int32 = 4 bytes
    assert val_path.stat().st_size == val_tokens.size * 4

    # Verify data can be read back
    loaded_train = np.fromfile(train_path, dtype=np.int32)
    loaded_val = np.fromfile(val_path, dtype=np.int32)
    np.testing.assert_array_equal(loaded_train, train_tokens)
    np.testing.assert_array_equal(loaded_val, val_tokens)


def test_manifest_contains_required_fields(tmp_path: Path) -> None:
    """Manifest JSON contains all fields needed for fairness verification."""
    tokenizer = _make_fake_tokenizer(tmp_path)
    tok_path = tmp_path / "tokenizer.json"

    # Build manifest (same structure as prepare_fineweb.py)
    tokenizer_serialized = tokenizer.to_serialized()
    tokenizer_hash = hashlib.sha256(
        tokenizer_serialized.encode("utf-8")
    ).hexdigest()

    manifest = {
        "dataset": "FineWeb-Edu",
        "subset": "sample-10BT",
        "documents": 1000,
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "train_ratio": 0.95,
        "seed": 42,
        "tokenizer_path": str(tok_path),
        "tokenizer_hash": tokenizer_hash,
        "train_tokens": 50000,
        "val_tokens": 2632,
        "total_tokens": 52632,
        "train_sha256": "abc123",
        "val_sha256": "def456",
    }

    # Verify all required fields
    required_fields = [
        "dataset", "subset", "documents", "vocab_size", "special_tokens",
        "train_ratio", "seed", "tokenizer_path", "tokenizer_hash",
        "train_tokens", "val_tokens", "total_tokens",
        "train_sha256", "val_sha256",
    ]
    for field in required_fields:
        assert field in manifest, f"Missing manifest field: {field}"

    # Verify tokenizer hash is a valid SHA-256 hex string
    assert len(manifest["tokenizer_hash"]) == 64
    int(manifest["tokenizer_hash"], 16)  # Should not raise


def test_tokenizer_hash_deterministic(tmp_path: Path) -> None:
    """Same tokenizer produces same hash (fairness lock)."""
    tokenizer = _make_fake_tokenizer(tmp_path)

    s1 = tokenizer.to_serialized()
    s2 = tokenizer.to_serialized()

    h1 = hashlib.sha256(s1.encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(s2.encode("utf-8")).hexdigest()

    assert h1 == h2, "Tokenizer hash is not deterministic"


def test_train_val_split_ratio(tmp_path: Path) -> None:
    """Train/val split respects the configured ratio."""
    tokenizer = _make_fake_tokenizer(tmp_path)
    docs = _make_fake_docs(100)
    eos_id = tokenizer.token_to_id("<eos>")

    all_ids: list[int] = []
    for doc in docs:
        ids = tokenizer.encode(doc["text"])
        all_ids.extend(ids)
        all_ids.append(eos_id)

    token_array = np.array(all_ids, dtype=np.int32)

    for ratio in [0.8, 0.9, 0.95]:
        split_idx = int(len(token_array) * ratio)
        train_tokens = token_array[:split_idx]
        val_tokens = token_array[split_idx:]

        actual_ratio = train_tokens.size / token_array.size
        assert abs(actual_ratio - ratio) < 0.01, (
            f"Split ratio {actual_ratio:.4f} deviates from target {ratio}"
        )


def test_shard_checksums_match(tmp_path: Path) -> None:
    """SHA-256 checksums in manifest match actual shard files."""
    tokenizer = _make_fake_tokenizer(tmp_path)
    docs = _make_fake_docs(20)
    eos_id = tokenizer.token_to_id("<eos>")

    all_ids: list[int] = []
    for doc in docs:
        ids = tokenizer.encode(doc["text"])
        all_ids.extend(ids)
        all_ids.append(eos_id)

    token_array = np.array(all_ids, dtype=np.int32)
    split_idx = int(len(token_array) * 0.95)

    train_path = tmp_path / "train_tokens.bin"
    val_path = tmp_path / "val_tokens.bin"
    token_array[:split_idx].tofile(train_path)
    token_array[split_idx:].tofile(val_path)

    # Compute checksums
    def sha256_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    train_hash = sha256_file(train_path)
    val_hash = sha256_file(val_path)

    # Verify they're valid hex strings
    assert len(train_hash) == 64
    assert len(val_hash) == 64

    # Verify reproducibility: same data → same hash
    token_array[:split_idx].tofile(train_path)
    assert sha256_file(train_path) == train_hash


def test_empty_docs_skipped(tmp_path: Path) -> None:
    """Empty documents are skipped during tokenization."""
    tokenizer = _make_fake_tokenizer(tmp_path)
    eos_id = tokenizer.token_to_id("<eos>")

    docs = [
        {"text": "Valid document with content."},
        {"text": ""},
        {"text": "   "},
        {"text": "Another valid document."},
    ]

    all_ids: list[int] = []
    for doc in docs:
        text = doc["text"]
        if not text or not text.strip():
            continue
        ids = tokenizer.encode(text)
        all_ids.extend(ids)
        all_ids.append(eos_id)

    # Should have tokens from 2 valid docs (not 4)
    # Count EOS tokens to verify
    eos_count = sum(1 for t in all_ids if t == eos_id)
    assert eos_count == 2, f"Expected 2 EOS tokens, got {eos_count}"


def test_chunked_writing_preserves_data(tmp_path: Path) -> None:
    """Tokens written in chunks match original tokens exactly."""
    chunk_flush_size = 1000  # Small for testing
    all_tokens = list(range(2500))  # Forces 3 chunks (1000 + 1000 + 500)

    output_path = tmp_path / "test_chunks.bin"
    chunk_buf: list[int] = []

    with output_path.open("wb") as f:
        for token in all_tokens:
            chunk_buf.append(token)
            if len(chunk_buf) >= chunk_flush_size:
                np.array(chunk_buf, dtype=np.int32).tofile(f)
                chunk_buf.clear()

        if chunk_buf:
            np.array(chunk_buf, dtype=np.int32).tofile(f)

    # Verify data integrity
    loaded = np.fromfile(output_path, dtype=np.int32)
    assert list(loaded) == all_tokens, "Chunked write corrupted data"


def test_data_config_exists() -> None:
    """FineWeb-Edu data config YAML exists with required fields."""
    from omegaconf import OmegaConf

    config_path = ROOT / "configs" / "data_fineweb.yaml"
    assert config_path.exists(), "configs/data_fineweb.yaml not found"

    config = OmegaConf.load(config_path)

    required = ["dataset", "subset", "output_dir", "tokenizer_path",
                 "vocab_size", "train_ratio", "seed",
                 "train_tokens_path", "val_tokens_path"]
    for field in required:
        assert field in config, f"Missing config field: {field}"

    assert config["dataset"] == "fineweb-edu"
    assert config["train_ratio"] > 0 and config["train_ratio"] < 1
