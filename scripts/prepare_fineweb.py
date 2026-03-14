"""Prepare FineWeb-Edu token shards and manifest for 1B-scale training.

FineWeb-Edu is a large-scale filtered web dataset (~1.3T tokens). This script
downloads a configurable subset, tokenizes it with a locked BPE tokenizer,
and produces deterministic train/val memmap shards with checksums.

Fairness policy: The manifest records the tokenizer hash, dataset subset,
document count, and shard checksums. These must match between SRN and
baseline runs to ensure a valid comparison.

Outputs:
- train_tokens.bin (int32 memmap)
- val_tokens.bin (int32 memmap)
- tokenizer.json (BPE tokenizer, reused or trained)
- manifest.json (checksums, token counts, tokenizer hash)

Usage:
    # GPT-2 tokenizer (recommended for 1B experiments — matches TinyStories runs)
    python scripts/prepare_fineweb.py --pretrained gpt2

    # Prepare with existing tokenizer (fairness: reuse from TinyStories or prior run)
    python scripts/prepare_fineweb.py --tokenizer_path data/tinystories/tokenizer.json

    # Train new tokenizer from FineWeb-Edu subset
    python scripts/prepare_fineweb.py --train_tokenizer --vocab_size 32000

    # Limit document count for testing
    python scripts/prepare_fineweb.py --max_docs 10000 --pretrained gpt2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import BPETokenizer, SPECIAL_TOKENS


def _sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FineWeb-Edu memmap shards for SRN training"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/fineweb-edu",
        help="Output directory for shards and manifest",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="HuggingFace model ID for pretrained tokenizer (e.g. 'gpt2')",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default=None,
        help="Path to existing tokenizer.json (reuse for fairness)",
    )
    parser.add_argument(
        "--train_tokenizer", action="store_true",
        help="Train a new BPE tokenizer from the dataset",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=32000,
        help="Vocab size for new tokenizer (only with --train_tokenizer)",
    )
    parser.add_argument(
        "--subset", type=str, default="sample-10BT",
        help="FineWeb-Edu subset name (default: sample-10BT, ~10B tokens)",
    )
    parser.add_argument(
        "--max_docs", type=int, default=None,
        help="Maximum number of documents to process (None = all)",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.95,
        help="Fraction of tokens for training (rest for validation)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible shuffling",
    )
    parser.add_argument(
        "--tokenizer_train_docs", type=int, default=50000,
        help="Number of documents to use for tokenizer training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load dataset ──────────────────────────────────────────────
    from datasets import load_dataset

    print(f"Loading FineWeb-Edu subset: {args.subset}")
    t0 = time.time()
    # trust_remote_code=True is required by FineWeb-Edu's dataset script.
    # This executes code from the HuggingFace Hub — only use trusted datasets.
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.subset,
        split="train",
        trust_remote_code=True,
    )
    print(f"  Loaded {len(ds):,} documents in {time.time() - t0:.1f}s")

    # Deterministic shuffle for reproducible train/val split
    ds = ds.shuffle(seed=args.seed)

    if args.max_docs is not None:
        ds = ds.select(range(min(args.max_docs, len(ds))))
        print(f"  Limited to {len(ds):,} documents")

    n_docs = len(ds)
    if n_docs == 0:
        raise ValueError("FineWeb-Edu dataset is empty after filtering")

    # ── Step 2: Tokenizer ─────────────────────────────────────────────────
    tokenizer_out_path = output_dir / "tokenizer.json"

    if args.pretrained is not None:
        print(f"Loading pretrained tokenizer: {args.pretrained}")
        tokenizer = BPETokenizer.from_pretrained(args.pretrained)
        tokenizer.save(str(tokenizer_out_path))
        print(f"  Saved local copy -> {tokenizer_out_path}")
    elif args.tokenizer_path is not None and Path(args.tokenizer_path).exists():
        print(f"Loading existing tokenizer: {args.tokenizer_path}")
        tokenizer = BPETokenizer.from_file(args.tokenizer_path)
        # Copy tokenizer to output dir for self-contained manifest
        if Path(args.tokenizer_path).resolve() != tokenizer_out_path.resolve():
            import shutil
            shutil.copy2(args.tokenizer_path, tokenizer_out_path)
    elif args.train_tokenizer:
        print(f"Training BPE tokenizer (vocab_size={args.vocab_size}) "
              f"from {min(args.tokenizer_train_docs, n_docs):,} documents...")
        train_subset = ds.select(range(min(args.tokenizer_train_docs, n_docs)))
        train_texts = (doc["text"] for doc in train_subset)
        tokenizer = BPETokenizer.train_from_iterator(
            train_texts,
            vocab_size=args.vocab_size,
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.save(str(tokenizer_out_path))
        print(f"  Saved tokenizer: {tokenizer_out_path}")
    else:
        raise ValueError(
            "Must provide --pretrained, --tokenizer_path, or --train_tokenizer. "
            "For fairness, reuse the same tokenizer across SRN and baseline."
        )

    # Resolve EOS token — try common candidates
    eos_id = None
    eos_token = None
    for candidate in ["<|endoftext|>", "<eos>", "</s>"]:
        eos_id = tokenizer.token_to_id(candidate)
        if eos_id is not None:
            eos_token = candidate
            break
    if eos_id is None:
        raise ValueError(
            "Could not find EOS token in tokenizer. "
            "Tried: <|endoftext|>, <eos>, </s>"
        )

    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    print(f"  EOS token: {eos_token!r} (id={eos_id})")

    # Validate train_ratio
    if not (0 < args.train_ratio < 1):
        raise ValueError(
            f"train_ratio must be between 0 and 1 (exclusive), got {args.train_ratio}"
        )

    # ── Step 3: Tokenize (batched + chunked to avoid OOM) ───────────────
    # Uses encode_batch() which parallelizes across all CPU cores via the
    # HuggingFace tokenizers Rust backend. Typically 10-20x faster than
    # single-threaded encode() in a Python loop.
    #
    # Documents are processed in batches of BATCH_SIZE, then token IDs are
    # flushed to disk in chunks to avoid holding all tokens in memory.
    BATCH_SIZE = 10_000  # Documents per encode_batch() call
    print(f"Tokenizing {n_docs:,} documents (batched, {BATCH_SIZE} docs/batch)...")
    t0 = time.time()

    # Write all tokens to a single temp file first
    all_tokens_path = output_dir / "_all_tokens.bin"
    total_tokens = 0
    docs_processed = 0

    with all_tokens_path.open("wb") as f:
        # Process documents in batches
        batch_texts: list[str] = []
        for doc in ds:
            text = doc["text"]
            if text and text.strip():
                batch_texts.append(text)

            if len(batch_texts) >= BATCH_SIZE:
                # Encode entire batch in parallel (Rust threads)
                all_ids = tokenizer.encode_batch(batch_texts)
                # Flatten with EOS separators and write to disk
                chunk_buf: list[int] = []
                for ids in all_ids:
                    chunk_buf.extend(ids)
                    chunk_buf.append(eos_id)
                chunk_array = np.array(chunk_buf, dtype=np.int32)
                chunk_array.tofile(f)
                total_tokens += len(chunk_buf)
                docs_processed += len(batch_texts)
                batch_texts.clear()

                elapsed = time.time() - t0
                docs_per_sec = docs_processed / max(elapsed, 1e-6)
                print(f"  {docs_processed:>10,}/{n_docs:,} docs "
                      f"({docs_processed / n_docs * 100:.0f}%) "
                      f"| {total_tokens:,} tokens "
                      f"| {docs_per_sec:,.0f} docs/s")

        # Flush remaining documents
        if batch_texts:
            all_ids = tokenizer.encode_batch(batch_texts)
            chunk_buf = []
            for ids in all_ids:
                chunk_buf.extend(ids)
                chunk_buf.append(eos_id)
            chunk_array = np.array(chunk_buf, dtype=np.int32)
            chunk_array.tofile(f)
            total_tokens += len(chunk_buf)
            docs_processed += len(batch_texts)

    elapsed = time.time() - t0
    print(f"  Tokenized {docs_processed:,} docs -> {total_tokens:,} tokens in {elapsed:.1f}s")

    if total_tokens == 0:
        all_tokens_path.unlink(missing_ok=True)
        raise ValueError("No tokens produced — all documents may be empty")

    # ── Step 4: Split into train/val shards ───────────────────────────────
    # Memory-map the temp file and split by index
    all_tokens = np.memmap(all_tokens_path, dtype=np.int32, mode="r")
    split_idx = int(len(all_tokens) * args.train_ratio)

    train_path = output_dir / "train_tokens.bin"
    val_path = output_dir / "val_tokens.bin"

    # Write train shard
    train_tokens = np.array(all_tokens[:split_idx])
    train_tokens.tofile(train_path)
    n_train = len(train_tokens)
    del train_tokens

    # Write val shard
    val_tokens = np.array(all_tokens[split_idx:])
    val_tokens.tofile(val_path)
    n_val = len(val_tokens)
    del val_tokens

    # Clean up temp file
    del all_tokens
    all_tokens_path.unlink()

    # ── Step 5: Manifest ──────────────────────────────────────────────────
    tokenizer_serialized = tokenizer.to_serialized()
    tokenizer_hash = hashlib.sha256(
        tokenizer_serialized.encode("utf-8")
    ).hexdigest()

    manifest = {
        "dataset": "FineWeb-Edu",
        "subset": args.subset,
        "documents": n_docs,
        "vocab_size": tokenizer.vocab_size,
        "eos_token": eos_token,
        "eos_id": eos_id,
        "pretrained": args.pretrained,
        "special_tokens": SPECIAL_TOKENS,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "tokenizer_path": str(tokenizer_out_path),
        "tokenizer_hash": tokenizer_hash,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "total_tokens": total_tokens,
        "train_sha256": _sha256(train_path),
        "val_sha256": _sha256(val_path),
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print("FineWeb-Edu Preparation Complete")
    print(f"{'='*60}")
    print(f"  Documents:     {n_docs:>12,}")
    print(f"  Train tokens:  {n_train:>12,}")
    print(f"  Val tokens:    {n_val:>12,}")
    print(f"  Total tokens:  {total_tokens:>12,}")
    print(f"  Vocab size:    {tokenizer.vocab_size:>12,}")
    print(f"  Tokenizer hash: {tokenizer_hash[:16]}...")
    print(f"  Train shard:   {train_path}")
    print(f"  Val shard:     {val_path}")
    print(f"  Manifest:      {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
