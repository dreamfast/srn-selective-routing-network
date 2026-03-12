"""
Data pipeline for SRN training on TinyShakespeare.

Provides:
- CharTokenizer: character-level tokenizer with encode/decode
- ShakespeareDataset: PyTorch Dataset returning (input, target) pairs
- get_dataloaders(): convenience function for train/val DataLoaders

The dataset is ~1.1MB of Shakespeare text (~1.1M characters, ~65 unique).
Character-level tokenization is simplest for a proof-of-concept — the model
must learn spelling, word boundaries, and grammar from scratch.
"""

import os
from pathlib import Path
from typing import Tuple

import requests
import torch
from torch.utils.data import DataLoader, Dataset

# TinyShakespeare URL and expected properties
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "tinyshakespeare.txt"


# ============================================================================
# Character-Level Tokenizer
# ============================================================================


class CharTokenizer:
    """Character-level tokenizer.

    Builds a sorted vocabulary from the input text. Each unique character
    gets a unique integer ID. Encode/decode are simple lookups.

    Attributes:
        chars: sorted list of unique characters
        vocab_size: number of unique characters
    """

    def __init__(self, text: str) -> None:
        self.chars = sorted(set(text))
        self._char_to_id = {c: i for i, c in enumerate(self.chars)}
        self._id_to_char = {i: c for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        """Encode text to list of integer token IDs."""
        return [self._char_to_id[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode list of integer token IDs to text."""
        return "".join(self._id_to_char[i] for i in ids)


# ============================================================================
# Dataset
# ============================================================================


class ShakespeareDataset(Dataset):
    """PyTorch Dataset for character-level language modeling.

    Returns (input_ids, target_ids) pairs where target is input shifted by 1.
    Each sample is a contiguous chunk of `seq_len` tokens from the text.

    Args:
        data: 1D LongTensor of encoded text
        seq_len: sequence length for each sample
    """

    def __init__(self, data: torch.Tensor, seq_len: int) -> None:
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        # Number of non-overlapping chunks (minus 1 for the target shift)
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        x = self.data[start:end]  # (seq_len,)
        y = self.data[start + 1 : end + 1]  # (seq_len,) — shifted by 1
        return x, y


# ============================================================================
# Data Loading
# ============================================================================


def download_shakespeare() -> str:
    """Download TinyShakespeare if not already present. Returns the text."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        text = DATA_FILE.read_text(encoding="utf-8")
        print(f"Loaded cached data: {len(text):,} characters from {DATA_FILE}")
        return text

    print(f"Downloading TinyShakespeare from {SHAKESPEARE_URL}...")
    response = requests.get(SHAKESPEARE_URL, timeout=30)
    response.raise_for_status()
    text = response.text

    # Verify reasonable size (~1.1MB)
    assert len(text) > 1_000_000, f"Downloaded text too small: {len(text)} chars"
    assert len(text) < 2_000_000, f"Downloaded text too large: {len(text)} chars"

    DATA_FILE.write_text(text, encoding="utf-8")
    print(f"Saved {len(text):,} characters to {DATA_FILE}")
    return text


def get_dataloaders(
    batch_size: int = 16,
    seq_len: int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, CharTokenizer]:
    """Create train and validation DataLoaders for TinyShakespeare.

    Split: first 90% for training, last 10% for validation (deterministic).

    Args:
        batch_size: batch size for DataLoaders
        seq_len: sequence length per sample
        num_workers: number of DataLoader workers

    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        tokenizer: CharTokenizer fitted on the full text
    """
    text = download_shakespeare()
    tokenizer = CharTokenizer(text)

    print(f"Vocabulary: {tokenizer.vocab_size} unique characters")
    print(f"Sample chars: {tokenizer.chars[:20]}...")

    # Encode full text
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # 90/10 split (deterministic — first 90% train, last 10% val)
    split_idx = int(len(encoded) * 0.9)
    train_data = encoded[:split_idx]
    val_data = encoded[split_idx:]

    print(f"Train: {len(train_data):,} tokens ({len(train_data) // seq_len} chunks)")
    print(f"Val:   {len(val_data):,} tokens ({len(val_data) // seq_len} chunks)")

    train_dataset = ShakespeareDataset(train_data, seq_len)
    val_dataset = ShakespeareDataset(val_data, seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,  # Avoid partial batches
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )

    return train_loader, val_loader, tokenizer


# ============================================================================
# Quick validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Data Pipeline — Validation")
    print("=" * 60)

    train_loader, val_loader, tokenizer = get_dataloaders(
        batch_size=4, seq_len=128
    )

    # Tokenizer roundtrip test
    test_text = "To be, or not to be, that is the question."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    assert decoded == test_text, f"Roundtrip failed: '{decoded}' != '{test_text}'"
    print(f"\nTokenizer roundtrip: PASSED")
    print(f"  '{test_text}' -> {encoded[:10]}... -> '{decoded[:20]}...'")

    # Vocab size check
    assert tokenizer.vocab_size <= 100, f"Vocab too large: {tokenizer.vocab_size}"
    print(f"Vocab size: {tokenizer.vocab_size} (expected ≤ 65)")

    # Batch shape check
    for x, y in train_loader:
        assert x.shape == (4, 128), f"Bad input shape: {x.shape}"
        assert y.shape == (4, 128), f"Bad target shape: {y.shape}"
        assert x.dtype == torch.long, f"Bad dtype: {x.dtype}"
        assert (x >= 0).all() and (x < tokenizer.vocab_size).all(), "Token ID out of range"
        assert (y >= 0).all() and (y < tokenizer.vocab_size).all(), "Target ID out of range"
        # Verify shift: y should be x shifted by 1
        # (This only holds within a chunk, not across chunks)
        print(f"\nBatch shapes: input={tuple(x.shape)}, target={tuple(y.shape)}")
        print(f"Token range: [{x.min().item()}, {x.max().item()}]")
        print(f"First input:  {tokenizer.decode(x[0, :30].tolist())!r}")
        print(f"First target: {tokenizer.decode(y[0, :30].tolist())!r}")
        break

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    print("\n" + "=" * 60)
    print("All checks passed!")
    print("=" * 60)
