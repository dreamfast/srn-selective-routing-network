"""Prepare TinyStories token shards and manifest for training.

Supports three tokenizer modes:
  --pretrained gpt2       Load a pretrained tokenizer from HuggingFace Hub
  --tokenizer_path X.json Load a previously saved tokenizer JSON file
  (default)               Train a new BPE tokenizer from the dataset

Outputs:
- train_tokens.bin (int32)
- val_tokens.bin (int32)
- tokenizer.json (saved copy of the tokenizer)
- manifest.json (dataset metadata + checksums)

Usage:
    # GPT-2 tokenizer (recommended for 150M experiments)
    python scripts/prepare_tinystories.py --pretrained gpt2

    # Train custom BPE from scratch
    python scripts/prepare_tinystories.py --vocab_size 32000

    # Quick test with fewer stories
    python scripts/prepare_tinystories.py --pretrained gpt2 --max_stories 10000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="HuggingFace model ID for pretrained tokenizer (e.g. 'gpt2')",
    )
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--max_stories", type=int, default=None)
    parser.add_argument(
        "--eos_token", type=str, default=None,
        help="EOS token string (auto-detected for pretrained tokenizers)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    # Deterministic shuffle BEFORE any splitting (fixes distribution mismatch)
    ds = ds.shuffle(seed=args.seed)
    print(f"  Shuffled with seed={args.seed}")

    if args.max_stories is not None:
        ds = ds.select(range(min(args.max_stories, len(ds))))
        print(f"  Limited to {len(ds):,} stories")

    n_stories = len(ds)
    if n_stories == 0:
        raise ValueError("TinyStories dataset is empty after filtering")

    texts = [item["text"] for item in ds]

    # --- Load or train tokenizer ---
    tokenizer_save_path = output_dir / "tokenizer.json"

    if args.pretrained is not None:
        print(f"Loading pretrained tokenizer: {args.pretrained}")
        tokenizer = BPETokenizer.from_pretrained(args.pretrained)
        tokenizer.save(str(tokenizer_save_path))
        print(f"  Saved local copy -> {tokenizer_save_path}")
    elif args.tokenizer_path is not None:
        tokenizer_path = Path(args.tokenizer_path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
        print(f"Loading existing tokenizer: {tokenizer_path}")
        tokenizer = BPETokenizer.from_file(str(tokenizer_path))
    else:
        print(f"Training BPE tokenizer (vocab_size={args.vocab_size})...")
        tokenizer = BPETokenizer.train_from_iterator(
            texts,
            vocab_size=args.vocab_size,
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.save(str(tokenizer_save_path))

    print(f"  Vocab size: {tokenizer.vocab_size:,}")

    # --- Resolve EOS token ---
    eos_id = None
    if args.eos_token is not None:
        eos_id = tokenizer.token_to_id(args.eos_token)
        if eos_id is None:
            raise ValueError(f"EOS token {args.eos_token!r} not found in tokenizer")
        eos_token = args.eos_token
    else:
        # Auto-detect: try common EOS tokens
        for candidate in ["", "<eos>", "</s>"]:
            eos_id = tokenizer.token_to_id(candidate)
            if eos_id is not None:
                eos_token = candidate
                break
        if eos_id is None:
            raise ValueError(
                "Could not auto-detect EOS token. "
                "Use --eos_token to specify one. "
                f"Tried: , <eos>, </s>"
            )

    print(f"  EOS token: {eos_token!r} (id={eos_id})")

    # --- Tokenize stories (batched + streaming to disk) ---
    BATCH_SIZE = 10_000  # Stories per encode_batch() call
    print(f"Tokenizing {n_stories:,} stories (batched, {BATCH_SIZE} stories/batch)...")
    t0 = time.time()

    all_tokens_path = output_dir / "_all_tokens.bin"
    total_tokens = 0
    stories_processed = 0

    with all_tokens_path.open("wb") as f:
        batch_texts: list[str] = []
        for i, text in enumerate(texts):
            if text and text.strip():
                batch_texts.append(text)

            if len(batch_texts) >= BATCH_SIZE or (i == len(texts) - 1 and batch_texts):
                all_ids = tokenizer.encode_batch(batch_texts)
                chunk_buf: list[int] = []
                for ids in all_ids:
                    chunk_buf.extend(ids)
                    chunk_buf.append(eos_id)
                chunk_array = np.array(chunk_buf, dtype=np.int32)
                chunk_array.tofile(f)
                total_tokens += len(chunk_buf)
                stories_processed += len(batch_texts)
                batch_texts.clear()

                if stories_processed % 100_000 == 0 or stories_processed == n_stories:
                    elapsed = time.time() - t0
                    print(f"  {stories_processed:>10,}/{n_stories:,} stories "
                          f"({stories_processed / n_stories * 100:.0f}%) "
                          f"| {total_tokens:,} tokens "
                          f"| {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  Tokenized {stories_processed:,} stories -> {total_tokens:,} tokens in {elapsed:.1f}s")

    if total_tokens == 0:
        all_tokens_path.unlink(missing_ok=True)
        raise ValueError("No tokens produced — all stories may be empty")

    # --- Split into train/val shards at document boundaries ---
    all_tokens = np.memmap(all_tokens_path, dtype=np.int32, mode="r")
    target_idx = int(len(all_tokens) * args.train_ratio)

    # Search backwards from target_idx for the last EOS token
    search_start = max(0, target_idx - 1_000_000)
    search_region = all_tokens[search_start:target_idx]
    eos_positions = np.where(search_region == eos_id)[0]

    if len(eos_positions) > 0:
        split_idx = search_start + eos_positions[-1] + 1
        print(f"  Adjusted split from {target_idx:,} to {split_idx:,} "
              f"(clean document boundary)")
    else:
        split_idx = target_idx
        print(f"  WARNING: No EOS token found near split point. "
              f"Using raw split at {split_idx:,}.")

    train_path = output_dir / "train_tokens.bin"
    val_path = output_dir / "val_tokens.bin"

    # Stream from memmap — no full RAM copy
    all_tokens[:split_idx].tofile(train_path)
    n_train = split_idx

    all_tokens[split_idx:].tofile(val_path)
    n_val = len(all_tokens) - split_idx

    del all_tokens
    all_tokens_path.unlink()

    # --- Manifest ---
    tokenizer_serialized = tokenizer.to_serialized()
    manifest = {
        "dataset": "TinyStories",
        "stories": n_stories,
        "vocab_size": tokenizer.vocab_size,
        "eos_token": eos_token,
        "eos_id": eos_id,
        "pretrained": args.pretrained,
        "special_tokens": SPECIAL_TOKENS,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "tokenizer_path": str(tokenizer_save_path),
        "tokenizer_hash": hashlib.sha256(tokenizer_serialized.encode("utf-8")).hexdigest(),
        "train_tokens": n_train,
        "val_tokens": n_val,
        "train_sha256": _sha256(train_path),
        "val_sha256": _sha256(val_path),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print("TinyStories Preparation Complete")
    print(f"{'='*60}")
    print(f"  Stories:       {n_stories:>12,}")
    print(f"  Train tokens:  {n_train:>12,}")
    print(f"  Val tokens:    {n_val:>12,}")
    print(f"  Total tokens:  {total_tokens:>12,}")
    print(f"  Vocab size:    {tokenizer.vocab_size:>12,}")
    print(f"  Train shard:   {train_path}")
    print(f"  Val shard:     {val_path}")
    print(f"  Manifest:      {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
