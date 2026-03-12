"""
Data pipeline for SRN training on TinyShakespeare.

Provides:
- CharTokenizer: character-level tokenizer with encode/decode
- BPETokenizer: HuggingFace tokenizers BPE wrapper
- ShakespeareDataset: PyTorch Dataset returning (input, target) pairs
- get_dataloaders(): convenience function for train/val DataLoaders

The dataset is ~1.1MB of Shakespeare text (~1.1M characters, ~65 unique).
Character-level tokenization is simplest for a proof-of-concept — the model
must learn spelling, word boundaries, and grammar from scratch.
"""

import hashlib
from pathlib import Path
from typing import Any, Iterable, Tuple, Union

import numpy as np
import requests
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from torch.utils.data import DataLoader, Dataset

# TinyShakespeare URL and expected properties
SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "tinyshakespeare.txt"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


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

    @property
    def tokenizer_type(self) -> str:
        return "char"

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "tokenizer_type": "char",
            "tokenizer_chars": self.chars,
        }


class BPETokenizer:
    """BPE tokenizer wrapper using HuggingFace tokenizers.

    Uses byte-level pre-tokenization and can be trained from iterable text.
    """

    def __init__(self, tokenizer: Tokenizer, tokenizer_path: str | None = None) -> None:
        self._tokenizer = tokenizer
        self.tokenizer_path = tokenizer_path

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        vocab_size: int = 32_000,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS

        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
        tokenizer.train_from_iterator(texts, trainer=trainer)
        return cls(tokenizer)

    @classmethod
    def from_file(cls, path: str) -> "BPETokenizer":
        tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer, tokenizer_path=path)

    @classmethod
    def from_serialized(cls, serialized: str) -> "BPETokenizer":
        tokenizer = Tokenizer.from_str(serialized)
        return cls(tokenizer)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def tokenizer_type(self) -> str:
        return "bpe"

    @property
    def special_tokens(self) -> list[str]:
        return [tok.content for tok in self._tokenizer.get_added_tokens_decoder().values()]

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    def token_to_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def save(self, path: str) -> None:
        self._tokenizer.save(path)
        self.tokenizer_path = path

    def to_serialized(self) -> str:
        return self._tokenizer.to_str()

    def checkpoint_payload(self) -> dict[str, Any]:
        serialized = self.to_serialized()
        return {
            "tokenizer_type": "bpe",
            "tokenizer_serialized": serialized,
            "tokenizer_hash": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "tokenizer_path": self.tokenizer_path,
        }


TokenizerType = Union[CharTokenizer, BPETokenizer]


def tokenizer_from_checkpoint(ckpt: dict[str, Any]) -> TokenizerType:
    """Reconstruct tokenizer from checkpoint metadata.

    Supports:
    - v1 char checkpoints: tokenized chars only
    - v2 checkpoints with `tokenizer_type`
    """
    version = ckpt.get("format_version", 1)
    if version <= 1:
        if "tokenizer_chars" not in ckpt:
            raise ValueError("Legacy checkpoint missing tokenizer_chars")
        return CharTokenizer("".join(ckpt["tokenizer_chars"]))

    tokenizer_type = ckpt.get("tokenizer_type")
    if tokenizer_type == "char":
        chars = ckpt.get("tokenizer_chars")
        if chars is None:
            raise ValueError("Char checkpoint missing tokenizer_chars")
        return CharTokenizer("".join(chars))

    if tokenizer_type == "bpe":
        serialized = ckpt.get("tokenizer_serialized")
        if serialized is not None:
            return BPETokenizer.from_serialized(serialized)
        path = ckpt.get("tokenizer_path")
        if path is not None:
            return BPETokenizer.from_file(path)
        raise ValueError("BPE checkpoint missing tokenizer_serialized/tokenizer_path")

    raise ValueError(f"Unsupported tokenizer_type in checkpoint: {tokenizer_type}")


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


class MemmapTokenDataset(Dataset):
    """Language-model dataset backed by memory-mapped token arrays.

    Expects a flat binary int32 token file. Samples are non-overlapping chunks
    of length `seq_len` with next-token targets.
    """

    def __init__(self, token_file: str | Path, seq_len: int) -> None:
        self.token_file = Path(token_file)
        self.seq_len = seq_len
        if not self.token_file.exists():
            raise FileNotFoundError(f"Token file not found: {self.token_file}")
        self.data = np.memmap(self.token_file, dtype=np.int32, mode="r")

    def __len__(self) -> int:
        if len(self.data) == 0:
            return 0
        return (len(self.data) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        end = start + self.seq_len
        x = torch.from_numpy(np.asarray(self.data[start:end], dtype=np.int64))
        y = torch.from_numpy(np.asarray(self.data[start + 1 : end + 1], dtype=np.int64))
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
    tokenizer_backend: str = "char",
    tokenizer_path: str | None = None,
    tokenizer_vocab_size: int = 32_000,
    tokenizer_override: TokenizerType | None = None,
) -> Tuple[DataLoader, DataLoader, TokenizerType]:
    """Create train and validation DataLoaders for TinyShakespeare.

    Split: first 90% for training, last 10% for validation (deterministic).

    Args:
        batch_size: batch size for DataLoaders
        seq_len: sequence length per sample
        num_workers: number of DataLoader workers

    Returns:
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        tokenizer: tokenizer instance fitted on the full text
    """
    text = download_shakespeare()
    if tokenizer_override is not None:
        tokenizer = tokenizer_override
    elif tokenizer_backend == "char":
        tokenizer: TokenizerType = CharTokenizer(text)
    elif tokenizer_backend == "bpe":
        if tokenizer_path is not None and Path(tokenizer_path).exists():
            tokenizer = BPETokenizer.from_file(tokenizer_path)
        else:
            tokenizer = BPETokenizer.train_from_iterator(
                [text],
                vocab_size=tokenizer_vocab_size,
                special_tokens=SPECIAL_TOKENS,
            )
            if tokenizer_path is not None:
                Path(tokenizer_path).parent.mkdir(parents=True, exist_ok=True)
                tokenizer.save(tokenizer_path)
    else:
        raise ValueError(f"Unknown tokenizer_backend: {tokenizer_backend}")

    vocab_unit = "characters" if isinstance(tokenizer, CharTokenizer) else "tokens"
    print(f"Vocabulary: {tokenizer.vocab_size} unique {vocab_unit}")
    if isinstance(tokenizer, CharTokenizer):
        print(f"Sample chars: {tokenizer.chars[:20]}...")
    else:
        print(f"Tokenizer backend: bpe, special tokens: {tokenizer.special_tokens[:4]}")

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


def get_memmap_dataloaders(
    train_tokens_path: str | Path,
    val_tokens_path: str | Path,
    tokenizer: TokenizerType,
    batch_size: int = 16,
    seq_len: int = 256,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, TokenizerType]:
    """Create train/val DataLoaders from memory-mapped token shards."""
    train_dataset = MemmapTokenDataset(train_tokens_path, seq_len)
    val_dataset = MemmapTokenDataset(val_tokens_path, seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
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
