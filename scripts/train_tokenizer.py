"""Train and save a BPE tokenizer artifact.

Usage:
    python scripts/train_tokenizer.py --input data/tinyshakespeare.txt --output data/tokenizer.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from data import BPETokenizer, SPECIAL_TOKENS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--output", type=str, required=True, help="Output tokenizer JSON")
    parser.add_argument("--vocab_size", type=int, default=32000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    tokenizer = BPETokenizer.train_from_iterator(
        [text],
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))

    print(f"Saved tokenizer to {output_path}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: {tokenizer.special_tokens[:4]}")


if __name__ == "__main__":
    main()
