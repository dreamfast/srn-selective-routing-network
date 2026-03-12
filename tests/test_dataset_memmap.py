from __future__ import annotations

from pathlib import Path

import numpy as np

from data import MemmapTokenDataset


def _write_tokens(path: Path, values: np.ndarray) -> None:
    values.astype(np.int32).tofile(path)


def test_memmap_len_and_bounds(tmp_path: Path) -> None:
    token_path = tmp_path / "tokens.bin"
    _write_tokens(token_path, np.arange(0, 101, dtype=np.int32))

    ds = MemmapTokenDataset(token_path, seq_len=10)
    assert len(ds) == 10


def test_sequence_shift_targets_correct(tmp_path: Path) -> None:
    token_path = tmp_path / "tokens.bin"
    _write_tokens(token_path, np.arange(0, 41, dtype=np.int32))

    ds = MemmapTokenDataset(token_path, seq_len=8)
    x, y = ds[2]
    assert x.shape[0] == 8
    assert y.shape[0] == 8
    assert np.all((x.numpy() + 1) == y.numpy())


def test_train_val_nonoverlap(tmp_path: Path) -> None:
    train_path = tmp_path / "train.bin"
    val_path = tmp_path / "val.bin"
    _write_tokens(train_path, np.arange(0, 1000, dtype=np.int32))
    _write_tokens(val_path, np.arange(2000, 2600, dtype=np.int32))

    train_ds = MemmapTokenDataset(train_path, seq_len=20)
    val_ds = MemmapTokenDataset(val_path, seq_len=20)

    train_x, _ = train_ds[0]
    val_x, _ = val_ds[0]
    assert set(train_x.tolist()).isdisjoint(set(val_x.tolist()))
