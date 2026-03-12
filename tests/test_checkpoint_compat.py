from __future__ import annotations

from data import BPETokenizer, CharTokenizer, SPECIAL_TOKENS, tokenizer_from_checkpoint


def test_v1_char_checkpoint_loads() -> None:
    char_tok = CharTokenizer("abc")
    ckpt = {
        "tokenizer_chars": char_tok.chars,
    }
    loaded = tokenizer_from_checkpoint(ckpt)
    assert loaded.tokenizer_type == "char"
    assert loaded.decode(loaded.encode("abc")) == "abc"


def test_v2_bpe_checkpoint_loads() -> None:
    bpe_tok = BPETokenizer.train_from_iterator(
        ["friends romans countrymen"],
        vocab_size=128,
        special_tokens=SPECIAL_TOKENS,
    )
    ckpt = {
        "format_version": 2,
        **bpe_tok.checkpoint_payload(),
    }
    loaded = tokenizer_from_checkpoint(ckpt)
    assert loaded.tokenizer_type == "bpe"
    assert loaded.encode("friends") == bpe_tok.encode("friends")
