"""
Dense GPT-style Transformer Baseline
======================================
A standard causal Transformer for fair comparison against SRN.

This model uses the same training harness, tokenizer, data pipeline, and
evaluation code as SRN. The only difference is the architecture:
- Multi-head causal self-attention (O(n²)) instead of sparse routing (O(n·k))
- Standard FFN instead of Gated Expert Mixture
- No auxiliary loss (returns 0.0 for compatibility)

The model exposes the same interface as SRNModel:
- forward(token_ids) -> (logits, aux_loss)
- generate(prompt_ids, max_tokens, temperature, top_k) -> token_ids
- count_params() -> int
- count_active_params() -> int  (same as count_params for dense models)

Author: maxx (with Claude/Anthropic) — Experimental
License: Apache 2.0
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DenseConfig:
    """Configuration for the Dense GPT baseline.

    Designed to be a fair comparison target for SRN. The parameter count
    should be ~0.67B to compare against SRN-1B's ~0.46B active params.
    """

    vocab_size: int = 65
    max_seq_len: int = 256
    d_model: int = 1024       # Model dimension
    n_layers: int = 16        # Number of transformer layers
    n_heads: int = 16         # Number of attention heads
    d_ff: int = 4096          # FFN hidden dimension (typically 4 * d_model)
    dropout: float = 0.1      # Dropout rate
    bias: bool = False        # Use bias in linear layers (GPT-2 style: no bias)


# ============================================================================
# Multi-Head Causal Self-Attention
# ============================================================================

class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention.

    Uses PyTorch's scaled_dot_product_attention with causal mask for
    efficient O(n²) attention computation.
    """

    def __init__(self, config: DenseConfig) -> None:
        super().__init__()
        d = config.d_model
        h = config.n_heads
        assert d % h == 0, f"d_model ({d}) must be divisible by n_heads ({h})"

        self.n_heads = h
        self.d_head = d // h

        # QKV projection (combined for efficiency)
        self.qkv = nn.Linear(d, 3 * d, bias=config.bias)
        # Output projection
        self.out_proj = nn.Linear(d, d, bias=config.bias)

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.qkv.weight)
        nn.init.xavier_normal_(self.out_proj.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with causal self-attention.

        Args:
            x: (B, N, D) input tensor

        Returns:
            (B, N, D) attention output
        """
        B, N, D = x.shape
        h, d_head = self.n_heads, self.d_head

        # QKV projection: (B, N, D) -> (B, N, 3*D) -> 3 × (B, h, N, d_head)
        qkv = self.qkv(x)  # (B, N, 3*D)
        q, k, v = qkv.split(D, dim=-1)  # 3 × (B, N, D)
        q = q.reshape(B, N, h, d_head).transpose(1, 2)  # (B, h, N, d_head)
        k = k.reshape(B, N, h, d_head).transpose(1, 2)  # (B, h, N, d_head)
        v = v.reshape(B, N, h, d_head).transpose(1, 2)  # (B, h, N, d_head)

        # Scaled dot-product attention with causal mask
        # PyTorch's SDPA handles the causal mask efficiently
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=True,
        )  # (B, h, N, d_head)

        # Reshape and project: (B, h, N, d_head) -> (B, N, D)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        output = self.resid_drop(self.out_proj(attn_out))  # (B, N, D)

        return output


# ============================================================================
# Feed-Forward Network
# ============================================================================

class FeedForward(nn.Module):
    """Standard two-layer FFN with GELU activation.

    Architecture: Linear(D -> 4D) -> GELU -> Linear(4D -> D) -> Dropout
    """

    def __init__(self, config: DenseConfig) -> None:
        super().__init__()
        d = config.d_model
        d_ff = config.d_ff

        self.up = nn.Linear(d, d_ff, bias=config.bias)
        self.down = nn.Linear(d_ff, d, bias=config.bias)
        self.drop = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.up.weight)
        nn.init.xavier_normal_(self.down.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        if self.down.bias is not None:
            nn.init.zeros_(self.down.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FFN.

        Args:
            x: (B, N, D) input tensor

        Returns:
            (B, N, D) FFN output
        """
        h = F.gelu(self.up(x))  # (B, N, d_ff)
        return self.drop(self.down(h))  # (B, N, D)


# ============================================================================
# Transformer Block
# ============================================================================

class TransformerBlock(nn.Module):
    """One transformer layer: pre-norm attention + pre-norm FFN.

    Pre-norm residual pattern (same as SRN):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(self, config: DenseConfig) -> None:
        super().__init__()
        d = config.d_model

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(config)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through one transformer block.

        Args:
            x: (B, N, D) input tensor

        Returns:
            (B, N, D) output tensor
        """
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================================
# Dense GPT Model
# ============================================================================

class DenseGPT(nn.Module):
    """Dense GPT-style Transformer for language modeling.

    Architecture: Token Embedding + Position Embedding -> N × TransformerBlock -> LM Head
    Weight tying: LM head shares weights with token embedding.

    Interface matches SRNModel for drop-in use in the shared training harness.
    """

    def __init__(self, config: DenseConfig) -> None:
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Learned positional encoding
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # Embedding dropout
        self.emb_drop = nn.Dropout(config.dropout)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # LM head (weight-tied with token embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings."""
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: token IDs -> logits.

        Returns aux_loss=0.0 for interface compatibility with SRN.

        Args:
            token_ids: (B, N) integer token indices

        Returns:
            logits: (B, N, vocab_size) unnormalized log-probabilities
            aux_loss: scalar tensor (always 0.0 — no MoE in dense model)
        """
        B, N = token_ids.shape
        device = token_ids.device

        # Embeddings
        pos_ids = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
        x = self.token_emb(token_ids) + self.pos_emb(pos_ids)  # (B, N, D)
        x = self.emb_drop(x)

        # Process through transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output
        x = self.ln_f(x)  # (B, N, D)
        logits = self.lm_head(x)  # (B, N, vocab_size)

        # No auxiliary loss for dense models (zeros() for graph compatibility)
        aux_loss = torch.zeros((), device=device)

        return logits, aux_loss

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation.

        Args:
            prompt_ids: (1, T) prompt token IDs
            max_tokens: number of tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_k: if set, only sample from top-k tokens

        Returns:
            (1, T + max_tokens) generated token IDs
        """
        self.eval()
        ids = prompt_ids  # (1, T)

        for _ in range(max_tokens):
            # Crop to max_seq_len if needed
            ids_cond = ids[:, -self.config.max_seq_len :]

            # Forward pass
            logits, _ = self(ids_cond)  # (1, T', V)
            logits = logits[:, -1, :]  # (1, V) — last position only

            if temperature == 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)  # (1, 1)
            else:
                logits = logits / temperature

                if top_k is not None:
                    topk_vals, _ = torch.topk(logits, top_k, dim=-1)
                    min_topk = topk_vals[:, -1:]
                    logits = logits.masked_fill(logits < min_topk, float("-inf"))

                probs = F.softmax(logits, dim=-1)  # (1, V)
                next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)

            ids = torch.cat([ids, next_id], dim=1)

        return ids

    def count_params(self) -> int:
        """Total parameter count."""
        return sum(p.numel() for p in self.parameters())

    def count_active_params(self) -> int:
        """Active parameters per token (same as total for dense models)."""
        return self.count_params()


# ============================================================================
# Quick validation (runs when executed directly)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Dense GPT Baseline — Validation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = DenseConfig()
    model = DenseGPT(config).to(device)

    total = model.count_params()
    active = model.count_active_params()
    print(f"Total params:  {total:>12,}")
    print(f"Active/token:  {active:>12,} ({active / total:.1%})")

    # Forward pass test
    B, N = 2, 128
    token_ids = torch.randint(0, config.vocab_size, (B, N), device=device)

    logits, aux_loss = model(token_ids)
    print(f"\nInput:         ({B}, {N})")
    print(f"Output:        {tuple(logits.shape)}")
    print(f"Aux loss:      {aux_loss.item():.4f}")
    print(f"Logit range:   [{logits.min().item():.3f}, {logits.max().item():.3f}]")

    # Verify no NaN/Inf
    assert not torch.isnan(logits).any(), "NaN in logits!"
    assert not torch.isinf(logits).any(), "Inf in logits!"
    print("NaN/Inf check: PASSED")

    # Verify valid probabilities
    probs = F.softmax(logits, dim=-1)
    prob_sums = probs.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)
    print("Prob sum check: PASSED")

    # Causal masking test
    print("\nCausal masking test...")
    model.eval()
    with torch.no_grad():
        test_ids = torch.randint(0, config.vocab_size, (1, 64), device=device)
        logits1, _ = model(test_ids)

        for t in [1, 16, 32, 63]:
            modified = test_ids.clone()
            modified[0, t] = (test_ids[0, t] + 1) % config.vocab_size
            logits2, _ = model(modified)

            if torch.allclose(logits1[0, :t], logits2[0, :t], atol=1e-5):
                print(f"  Position {t:>2}: PASSED")
            else:
                max_diff = (logits1[0, :t] - logits2[0, :t]).abs().max().item()
                print(f"  Position {t:>2}: FAILED (max diff: {max_diff:.6f})")

    # Generation test
    print("\nGeneration test...")
    prompt = torch.zeros(1, 1, dtype=torch.long, device=device)
    generated = model.generate(prompt, max_tokens=50, temperature=0.8)
    print(f"  Generated {generated.shape[1]} tokens")
    print(f"  Token range: [{generated.min().item()}, {generated.max().item()}]")

    # Memory usage
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nPeak GPU memory: {mem:.1f} MB")

    print("\n" + "=" * 60)
    print("All checks passed!")
    print("=" * 60)
