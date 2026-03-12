from __future__ import annotations

from data import BPETokenizer, SPECIAL_TOKENS, tokenizer_from_checkpoint


def test_bpe_roundtrip_nonempty() -> None:
    tok = BPETokenizer.train_from_iterator(
        ["To be, or not to be."],
        vocab_size=128,
        special_tokens=SPECIAL_TOKENS,
    )
    ids = tok.encode("To be")
    assert len(ids) > 0
    decoded = tok.decode(ids)
    assert isinstance(decoded, str)
    assert len(decoded) > 0


def test_special_tokens_present() -> None:
    tok = BPETokenizer.train_from_iterator(
        ["alpha beta gamma"],
        vocab_size=128,
        special_tokens=SPECIAL_TOKENS,
    )
    specials = set(tok.special_tokens)
    for token in SPECIAL_TOKENS:
        assert token in specials


def test_checkpoint_tokenizer_reload_consistent_ids() -> None:
    tok = BPETokenizer.train_from_iterator(
        ["romeo juliet mercutio"],
        vocab_size=128,
        special_tokens=SPECIAL_TOKENS,
    )
    ckpt = {
        "format_version": 2,
        **tok.checkpoint_payload(),
    }
    reloaded = tokenizer_from_checkpoint(ckpt)

    source = "romeo"
    assert tok.encode(source) == reloaded.encode(source)
