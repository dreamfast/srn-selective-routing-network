"""Minimal nanoGPT-style decoder-only Transformer baseline."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BaselineConfig:
    """Configuration for the Transformer baseline."""

    vocab_size: int = 50257
    max_seq_len: int = 512
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12
    d_ff: int = 3072
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    """Standard multi-head causal self-attention."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        bsz, seq_len, dim = x.shape

        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(dim, dim=2)  # 3 x (B, T, C)

        q = q.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, T, d)
        k = k.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, T, d)
        v = v.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (B, h, T, d)

        scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        scores = scores.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], float("-inf"))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        attn = self.attn_dropout(attn)

        y = attn @ v  # (B, h, T, d)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)  # (B, T, C)
        y = self.resid_dropout(self.proj(y))
        return y


class MLP(nn.Module):
    """Position-wise feed-forward network with GELU."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BaselineTransformer(nn.Module):
    """Decoder-only Transformer with tied token embedding and LM head."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T)
        _, seq_len = idx.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        pos = torch.arange(0, seq_len, device=idx.device, dtype=torch.long).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)  # (B, T, C)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)  # (B, T, vocab)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len :]
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                next_idx = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                    logits = logits.masked_fill(logits < values[:, -1:], float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_idx), dim=1)
        return idx

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    cfg = BaselineConfig()
    model = BaselineTransformer(cfg)
    total = model.count_params()
    print(f"Baseline config: d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}, d_ff={cfg.d_ff}")
    print(f"Total parameters: {total:,}")
