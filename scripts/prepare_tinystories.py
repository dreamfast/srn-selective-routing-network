"""Prepare TinyStories token shards and manifest for training.

Outputs:
- train_tokens.bin (int32)
- val_tokens.bin (int32)
- tokenizer.json
- manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset

from data import BPETokenizer, SPECIAL_TOKENS


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare TinyStories memmap shards")
    parser.add_argument("--output_dir", type=str, default="data/tinystories")
    parser.add_argument("--tokenizer_path", type=str, default="data/tinystories/tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--max_stories", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train")
    if args.max_stories is not None:
        ds = ds.select(range(min(args.max_stories, len(ds))))

    texts = [item["text"] for item in ds]
    if len(texts) == 0:
        raise ValueError("TinyStories dataset is empty")

    tokenizer_path = Path(args.tokenizer_path)
    if tokenizer_path.exists():
        print(f"Loading existing tokenizer: {tokenizer_path}")
        tokenizer = BPETokenizer.from_file(str(tokenizer_path))
    else:
        print("Training BPE tokenizer...")
        tokenizer = BPETokenizer.train_from_iterator(
            texts,
            vocab_size=args.vocab_size,
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))

    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("Tokenizer is missing <eos> token")

    print("Tokenizing stories...")
    all_ids: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)
        all_ids.append(eos_id)

    token_array = np.array(all_ids, dtype=np.int32)
    split_idx = int(len(token_array) * args.train_ratio)
    train_tokens = token_array[:split_idx]
    val_tokens = token_array[split_idx:]

    train_path = output_dir / "train_tokens.bin"
    val_path = output_dir / "val_tokens.bin"
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    tokenizer_serialized = tokenizer.to_serialized()
    manifest = {
        "dataset": "TinyStories",
        "stories": len(texts),
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "train_ratio": args.train_ratio,
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_hash": hashlib.sha256(tokenizer_serialized.encode("utf-8")).hexdigest(),
        "train_tokens": int(train_tokens.size),
        "val_tokens": int(val_tokens.size),
        "train_sha256": _sha256(train_path),
        "val_sha256": _sha256(val_path),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote {train_tokens.size:,} train tokens -> {train_path}")
    print(f"Wrote {val_tokens.size:,} val tokens -> {val_path}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
