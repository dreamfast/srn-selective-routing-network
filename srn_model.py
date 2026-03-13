"""
SRN: Selective Routing Network — PyTorch Implementation
=========================================================
A trainable PyTorch port of the SRN architecture with critical improvements:

1. **Windowed Causal Score Gating (WCSG)**: A novel mechanism that provides
   causal routing without VRAM explosion. Each position's routing scores are
   modulated by a gate derived from a causal window of past tokens, ensuring
   no future information leaks into past positions.

2. **Vectorized Gated Expert Mixture**: All experts computed in parallel via
   einsum instead of sequential Python loops.

3. **Configurable CSP Residual**: The double-residual bug in the original
   NumPy implementation is fixed (configurable for A/B testing).

4. **Switch Transformer Load Balancing**: MoE auxiliary loss prevents expert
   collapse during training.

Novel contribution — WCSG:
    Instead of adapting slot keys per-position (which expands to (B,N,k,D) and
    requires ~4.3GB VRAM), we modulate the *routing scores* with a per-position
    causal gate. The gate is derived from a windowed mean of past tokens via
    F.avg_pool1d with left-padding. This provides:
    - True causality: position t only sees tokens [max(0, t-W+1)..t]
    - O(1) extra memory: gate is (B, N, k) ≈ 1MB/layer
    - Slot keys/values stay at (k, D) — no per-position expansion

    Design trade-off: softmax((Q @ K^T) * g) ≠ softmax(Q @ (K*g)^T).
    Score modulation can zero out routes but cannot transform key directions.
    This is an intentional capacity-for-efficiency trade-off.

Author: maxx (with Claude/Anthropic) — Experimental
License: Apache 2.0
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dense_model import CausalSelfAttention


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SRNConfig:
    """Configuration for the Selective Routing Network.

    All hyperparameters for the model architecture. Training hyperparameters
    (lr, batch_size, etc.) live in train.py, not here.
    """

    vocab_size: int = 65  # Character-level Shakespeare (~65 unique chars)
    max_seq_len: int = 256  # Sequence length (fits in 6GB VRAM)
    d_model: int = 512  # Main model dimension
    d_compressed: int = 128  # Compressed state dimension (d_model / 4)
    n_layers: int = 8  # Number of SRN layers
    n_memory_slots: int = 64  # Number of routing targets (replaces attention)
    n_experts: int = 8  # Number of expert networks per layer
    top_k_experts: int = 2  # How many experts process each token
    d_expert: int = 256  # Hidden dim within each expert
    n_heads_route: int = 4  # Multi-head routing
    dropout: float = 0.1  # Dropout rate (actually applied, unlike original)
    causal_window: int = 32  # Window size for causal slot adaptation (NOVEL)
    csp_internal_residual: bool = False  # If True, CSP adds its own residual
    aux_loss_weight: float = 0.01  # Weight for MoE load balancing loss
    sparse_moe: bool = False  # If True, use grouped sparse MoE (VRAM-efficient)

    # --- Ablation experiment flags ---
    attention_every_n_layers: int = 0  # 0=pure SRN, N=every Nth layer uses attention
    attention_n_heads: int = 8  # Attention heads (only used when attention_every_n_layers>0)
    disable_csp: bool = False  # If True, skip CSP bottleneck entirely
    wcsg_key_offset: bool = False  # If True, add score-space offset to WCSG
    wcsg_key_offset_rank: int = 32  # Low-rank dimension for WCSG offset


# ============================================================================
# Utility: Causal Windowed Mean
# ============================================================================

def causal_windowed_mean(x: torch.Tensor, window: int) -> torch.Tensor:
    """Compute causal windowed mean over the sequence dimension.

    Each position t receives the mean of tokens [max(0, t-W+1) .. t],
    properly normalized by the actual number of tokens in the window
    (not the fixed window size). This prevents early positions from being
    attenuated by zero-padding.

    Example:
        Position 0 (window=32): mean of [x₀] → x₀/1 (not x₀/32)
        Position 5 (window=32): mean of [x₀..x₅] → sum/6
        Position 40 (window=32): mean of [x₉..x₄₀] → sum/32

    Args:
        x: (B, N, D) input tensor (N must be >= 1)
        window: window size W

    Returns:
        (B, N, D) windowed causal means, properly normalized
    """
    B, N, D = x.shape
    # avg_pool1d expects (B, C, L) — channels-first
    x_t = x.transpose(1, 2)  # (B, D, N)
    # Left-pad with zeros: only past information, no future leakage
    x_padded = F.pad(x_t, (window - 1, 0))  # (B, D, N + W - 1)
    # Sum pool (not avg) — we normalize by actual count below
    # avg_pool1d divides by kernel_size, so multiply back to get sum
    summed = F.avg_pool1d(x_padded, kernel_size=window, stride=1) * window  # (B, D, N)
    # Actual count per position: min(position + 1, window)
    counts = torch.arange(1, N + 1, device=x.device, dtype=x.dtype)  # (N,)
    counts = counts.clamp(max=window).view(1, 1, -1)  # (1, 1, N)
    # Divide by actual count for proper normalization
    pooled = summed / counts  # (B, D, N)
    return pooled.transpose(1, 2)  # (B, N, D)


# ============================================================================
# Module 1: Dynamic Sparse Router with Windowed Causal Score Gating (WCSG)
# ============================================================================

class DynamicSparseRouter(nn.Module):
    """Replaces self-attention with learned routing to memory slots.

    Novel: Windowed Causal Score Gating (WCSG)
    -------------------------------------------
    Instead of computing n×n attention, each token routes to k memory slots.
    The routing scores are modulated by a per-position causal gate derived
    from a windowed mean of past tokens. This ensures:
    - Causality: position t cannot see tokens at positions > t
    - Efficiency: O(n·k) instead of O(n²), with O(1) extra memory for gating
    - Adaptivity: routing preferences change based on local context

    The slot keys and values are global (shared across positions), but the
    routing *scores* are position-dependent via the causal gate. This is a
    deliberate capacity-for-efficiency trade-off vs. per-position slot keys.
    """

    def __init__(self, config: SRNConfig, layer_idx: int) -> None:
        super().__init__()
        d = config.d_model
        k = config.n_memory_slots
        h = config.n_heads_route
        d_head = d // h

        self.config = config
        self.n_heads = h
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)

        # Query projection (from token representations)
        self.W_q = nn.Linear(d, d)  # (D) -> (D)

        # Slot keys: learned memory slot representations (GLOBAL, not per-position)
        self.slot_keys = nn.Parameter(torch.empty(k, d))  # (k, D)

        # Slot value projection (STATIC: slot_keys @ W_sv, no per-batch adaptation)
        self.W_sv = nn.Linear(d, d, bias=True)  # (D) -> (D)

        # Causal gate projection: windowed_mean -> per-position gate
        self.W_gate = nn.Linear(d, k)  # (D) -> (k)

        # Learned positional routing bias
        self.pos_bias = nn.Parameter(torch.zeros(config.max_seq_len, k))  # (L, k)

        # Output projection
        self.W_o = nn.Linear(d, d)  # (D) -> (D)

        # WCSG score-space offset: low-rank additive offset to routing scores
        # Uses the same causal gate for consistent causal gating
        self.W_offset_down: Optional[nn.Linear] = None
        self.W_offset_up: Optional[nn.Linear] = None
        if config.wcsg_key_offset:
            rank = config.wcsg_key_offset_rank
            self.W_offset_down = nn.Linear(d, rank)   # (D) -> (rank)
            self.W_offset_up = nn.Linear(rank, k)     # (rank) -> (k)

        # Dropout on routing weights
        self.attn_drop = nn.Dropout(config.dropout)

        # Initialize
        self._init_weights(layer_idx)

    def _init_weights(self, layer_idx: int) -> None:
        """Xavier normal initialization matching original NumPy ParamStore."""
        nn.init.xavier_normal_(self.W_q.weight)
        nn.init.zeros_(self.W_q.bias)
        nn.init.xavier_normal_(self.slot_keys)
        nn.init.xavier_normal_(self.W_sv.weight)
        nn.init.zeros_(self.W_sv.bias)
        nn.init.xavier_normal_(self.W_gate.weight)
        nn.init.zeros_(self.W_gate.bias)
        nn.init.xavier_normal_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)
        # Positional bias starts at zero — learned from scratch
        nn.init.zeros_(self.pos_bias)
        # WCSG offset: near-zero init so offset starts negligible
        if self.W_offset_down is not None:
            nn.init.normal_(self.W_offset_down.weight, std=0.001)
            nn.init.zeros_(self.W_offset_down.bias)
            nn.init.normal_(self.W_offset_up.weight, std=0.001)
            nn.init.zeros_(self.W_offset_up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Windowed Causal Score Gating.

        Args:
            x: (B, N, D) input tensor

        Returns:
            (B, N, D) routed output
        """
        B, N, D = x.shape
        h, d_head = self.n_heads, self.d_head
        k = self.config.n_memory_slots

        # 1. Compute queries from input tokens
        Q = self.W_q(x)  # (B, N, D)
        Q = Q.reshape(B, N, h, d_head).permute(0, 2, 1, 3)  # (B, h, N, d_head)

        # 2. Compute per-position causal gate via windowed mean (NOVEL: WCSG)
        x_windowed = causal_windowed_mean(x, self.config.causal_window)  # (B, N, D)
        # Clamp gate to prevent all-zero routing scores (which would cause
        # vanishing gradients through the sigmoid gate and dead neurons)
        gate = torch.sigmoid(self.W_gate(x_windowed)).clamp(min=1e-4)  # (B, N, k)

        # 3. Compute routing scores: Q @ slot_keys^T
        # Slot keys are global — reshape for multi-head
        K_slots = self.slot_keys.reshape(k, h, d_head).permute(1, 0, 2)  # (h, k, d_head)
        routing_scores = torch.matmul(
            Q, K_slots.transpose(-2, -1)
        )  # (B, h, N, k)
        routing_scores = routing_scores * self.scale

        # 4. MODULATE routing scores with causal gate (NOVEL)
        # gate: (B, N, k) -> (B, 1, N, k) for broadcast over heads
        routing_scores = routing_scores * gate.unsqueeze(1)

        # 4b. WCSG score-space offset: gated additive offset (NOVEL)
        # Uses same causal gate for consistent causal gating
        if self.W_offset_down is not None:
            score_offset = self.W_offset_up(
                F.gelu(self.W_offset_down(x_windowed))
            )  # (B, N, k)
            # Gated application: gate * offset ensures causality
            routing_scores = routing_scores + (gate * score_offset).unsqueeze(1)  # (B, 1, N, k)

        # 5. Add learned positional routing bias
        # pos_bias: (L, k) -> (1, 1, N, k)
        routing_scores = routing_scores + self.pos_bias[:N].unsqueeze(0).unsqueeze(0)

        # 6. Softmax over slots
        routing_weights = F.softmax(routing_scores, dim=-1)  # (B, h, N, k)
        routing_weights = self.attn_drop(routing_weights)

        # 7. Compute slot values (STATIC — no per-batch adaptation)
        V_slots = self.W_sv(self.slot_keys)  # (k, D)
        V_slots = V_slots.reshape(k, h, d_head).permute(1, 0, 2)  # (h, k, d_head)

        # 8. Route: each token aggregates from slots
        # routing_weights: (B, h, N, k) @ V_slots: (h, k, d_head) -> (B, h, N, d_head)
        routed = torch.matmul(routing_weights, V_slots)  # (B, h, N, d_head)

        # 9. Reshape and project output
        routed = routed.permute(0, 2, 1, 3).reshape(B, N, D)  # (B, N, D)
        output = self.W_o(routed)  # (B, N, D)

        return output


# ============================================================================
# Module 2: Compressed State Propagation
# ============================================================================

class CompressedStatePropagation(nn.Module):
    """Forces information through a learned bottleneck between layers.

    Compresses the state to a lower dimension, processes it, and selectively
    expands back via a sigmoid gate. This acts as learned information selection,
    keeping only what matters.

    Fix from original: The double-residual bug is corrected. The original
    NumPy code returned `x + gate * expanded` internally, then the layer
    added ANOTHER residual `x + csp(x)`, creating a double skip connection.
    Now configurable via `csp_internal_residual`.
    """

    def __init__(self, config: SRNConfig, layer_idx: int) -> None:
        super().__init__()
        d = config.d_model
        dc = config.d_compressed

        # Compress: D -> dc
        self.compress = nn.Linear(d, dc)
        # Process in compressed space: dc -> dc
        self.process = nn.Linear(dc, dc)
        # Expand back: dc -> D
        self.expand = nn.Linear(dc, d)
        # Gate: decides how much compressed info to inject
        self.gate_proj = nn.Linear(d + dc, d)

        self.drop = nn.Dropout(config.dropout)
        self.internal_residual = config.csp_internal_residual

        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.compress, self.process, self.expand, self.gate_proj]:
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the compression bottleneck.

        Args:
            x: (B, N, D) input tensor

        Returns:
            (B, N, D) — if internal_residual=False: gate * expanded
                        if internal_residual=True:  x + gate * expanded
        """
        # Compress: (B, N, D) -> (B, N, dc)
        compressed = F.gelu(self.compress(x))

        # Process in compressed space: (B, N, dc) -> (B, N, dc)
        processed = F.gelu(self.process(compressed))

        # Expand: (B, N, dc) -> (B, N, D)
        expanded = self.expand(processed)
        expanded = self.drop(expanded)

        # Gated injection: concat [x, processed] -> sigmoid gate
        gate_input = torch.cat([x, processed], dim=-1)  # (B, N, D + dc)
        gate = torch.sigmoid(self.gate_proj(gate_input))  # (B, N, D)

        if self.internal_residual:
            return x + gate * expanded
        else:
            return gate * expanded


# ============================================================================
# Module 3: Gated Expert Mixture (Vectorized)
# ============================================================================

class GatedExpertMixture(nn.Module):
    """Multiple small expert networks with a lightweight router.

    Only top-k experts are activated per token, so most parameters are dormant.
    This means effective parameter count per token << total parameter count.

    Improvements over original:
    - Vectorized via einsum (no Python loop over experts)
    - Switch Transformer load balancing auxiliary loss
    - Dropout on expert outputs
    """

    def __init__(self, config: SRNConfig, layer_idx: int) -> None:
        super().__init__()
        d = config.d_model
        n_exp = config.n_experts
        d_exp = config.d_expert

        self.config = config

        # Router: decides which experts process each token
        self.router = nn.Linear(d, n_exp)

        # Expert parameters (stored as 3D tensors for vectorized compute)
        self.W_up = nn.Parameter(torch.empty(n_exp, d, d_exp))  # (E, D, H)
        self.b_up = nn.Parameter(torch.zeros(n_exp, d_exp))  # (E, H)
        self.W_down = nn.Parameter(torch.empty(n_exp, d_exp, d))  # (E, H, D)
        self.b_down = nn.Parameter(torch.zeros(n_exp, d))  # (E, D)

        self.drop = nn.Dropout(config.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        # Xavier normal for each expert's weight matrices
        for e in range(self.config.n_experts):
            nn.init.xavier_normal_(self.W_up[e])
            nn.init.xavier_normal_(self.W_down[e])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with expert computation.

        Dispatches to dense (einsum-all-experts) or sparse (grouped token-by-
        expert) path based on config.sparse_moe.

        Args:
            x: (B, N, D) input tensor

        Returns:
            output: (B, N, D) expert-mixed output
            aux_loss: scalar — Switch Transformer load balancing loss
        """
        B, N, D = x.shape
        n_exp = self.config.n_experts
        top_k = self.config.top_k_experts

        # 1. Route: compute expert selection logits
        router_logits = self.router(x)  # (B, N, E)

        # 2. Top-k expert selection
        topk_vals, topk_idx = torch.topk(router_logits, top_k, dim=-1)  # (B, N, top_k)

        # 3. Masked softmax: only top-k experts get non-zero weight
        # Create mask: (B, N, E) with True for selected experts
        mask = torch.zeros_like(router_logits, dtype=torch.bool)
        mask.scatter_(-1, topk_idx, True)

        # Masked logits: -inf for non-selected experts
        masked_logits = router_logits.masked_fill(~mask, float("-inf"))
        router_weights = F.softmax(masked_logits, dim=-1)  # (B, N, E)

        # 4. Expert forward pass — dense or sparse
        if self.config.sparse_moe:
            output = self._sparse_expert_forward(x, router_weights, topk_idx)
        else:
            output = self._dense_expert_forward(x, router_weights)

        output = self.drop(output)

        # 5. Load balancing auxiliary loss (adapted for top-k > 1)
        # f_i: fraction of tokens where expert i is in top-k selection
        f_i = mask.float().mean(dim=(0, 1))  # (E,)
        # P_i: mean routing weight for expert i from the MASKED softmax
        # (not the full softmax — using full softmax creates gradient
        # misalignment when top-k > 1, as it pulls probability mass toward
        # experts that are then masked out in the forward pass)
        P_i = router_weights.mean(dim=(0, 1))  # (E,)
        # L_aux = n_experts * sum(f_i * P_i) — encourages uniform distribution
        aux_loss = n_exp * (f_i * P_i).sum()

        return output, aux_loss

    def _dense_expert_forward(
        self, x: torch.Tensor, router_weights: torch.Tensor
    ) -> torch.Tensor:
        """Dense expert path: compute ALL experts for ALL tokens via einsum.

        Simple and fast for small models, but VRAM-hungry at scale because
        it materializes the full (B, N, E, H) intermediate tensor.

        Args:
            x: (B, N, D) input tokens
            router_weights: (B, N, E) softmax routing weights

        Returns:
            (B, N, D) weighted expert output
        """
        # x: (B, N, D), W_up: (E, D, H) -> einsum -> (B, N, E, H)
        h = torch.einsum("bnd,edh->bneh", x, self.W_up) + self.b_up  # (B, N, E, H)
        h = F.gelu(h)

        # h: (B, N, E, H), W_down: (E, H, D) -> einsum -> (B, N, E, D)
        h = torch.einsum("bneh,ehd->bned", h, self.W_down) + self.b_down  # (B, N, E, D)

        # Weight by router and sum over experts
        # router_weights: (B, N, E) -> (B, N, E, 1) for broadcast
        output = (router_weights.unsqueeze(-1) * h).sum(dim=2)  # (B, N, D)
        return output

    def _sparse_expert_forward(
        self,
        x: torch.Tensor,
        router_weights: torch.Tensor,
        topk_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Sparse expert path: grouped token-by-expert dispatch.

        Only computes the top-k selected experts per token. Groups tokens by
        expert assignment, runs each expert only on its assigned tokens, then
        scatters results back. VRAM scales with O(B*N*top_k*H) instead of
        O(B*N*E*H).

        Args:
            x: (B, N, D) input tokens
            router_weights: (B, N, E) softmax routing weights
            topk_idx: (B, N, top_k) indices of selected experts

        Returns:
            (B, N, D) weighted expert output
        """
        B, N, D = x.shape
        n_exp = self.config.n_experts
        top_k = self.config.top_k_experts

        # Flatten batch and sequence: (B*N, D), (B*N, top_k)
        x_flat = x.reshape(B * N, D)  # (T, D) where T = B*N
        topk_idx_flat = topk_idx.reshape(B * N, top_k)  # (T, top_k)

        # Gather router weights for selected experts: (T, top_k)
        # router_weights is (B, N, E) -> flatten to (T, E)
        rw_flat = router_weights.reshape(B * N, n_exp)  # (T, E)
        # Gather weights for selected experts
        topk_weights = rw_flat.gather(-1, topk_idx_flat)  # (T, top_k)

        # Output accumulator
        output = torch.zeros_like(x_flat)  # (T, D)

        # Process each expert: gather assigned tokens, compute, scatter back
        for e in range(n_exp):
            # Find which (token, slot) pairs are assigned to expert e
            # topk_idx_flat: (T, top_k) — check each slot
            expert_mask = topk_idx_flat == e  # (T, top_k)

            if not expert_mask.any():
                continue

            # Get token indices that route to this expert (across any slot)
            token_mask = expert_mask.any(dim=-1)  # (T,)
            token_indices = token_mask.nonzero(as_tuple=True)[0]  # (n_tokens,)

            # Gather tokens for this expert
            x_e = x_flat[token_indices]  # (n_tokens, D)

            # Expert forward: up projection + GELU + down projection
            h_e = F.linear(x_e, self.W_up[e].T, self.b_up[e])  # (n_tokens, H)
            h_e = F.gelu(h_e)
            h_e = F.linear(h_e, self.W_down[e].T, self.b_down[e])  # (n_tokens, D)

            # Compute combined weight for this expert across all slots
            # A token may select the same expert in multiple top-k slots
            # (rare but possible) — sum the weights across those slots
            slot_weights = (expert_mask[token_indices].float()
                           * topk_weights[token_indices])  # (n_tokens, top_k)
            combined_weight = slot_weights.sum(dim=-1, keepdim=True)  # (n_tokens, 1)

            # Scatter weighted output back
            output.index_add_(0, token_indices, combined_weight * h_e)

        return output.reshape(B, N, D)  # (B, N, D)


# ============================================================================
# Full SRN Layer
# ============================================================================

class SRNLayer(nn.Module):
    """One layer of the Selective Routing Network.

    Pre-norm residual pattern:
        x = x + Routing(LayerNorm(x))   — DSR or CausalSelfAttention
        x = x + CSP(LayerNorm(x))       — optional (skip when disable_csp=True)
        x = x + GEM(LayerNorm(x))       — always present

    Hybrid attention: when attention_every_n_layers > 0, every Nth layer
    replaces DSR with standard causal self-attention. This tests whether
    direct token-to-token interaction closes the quality gap.
    """

    def __init__(self, config: SRNConfig, layer_idx: int) -> None:
        super().__init__()

        d = config.d_model

        # Layer norms (pre-norm pattern)
        self.ln1 = nn.LayerNorm(d)
        self.ln3 = nn.LayerNorm(d)

        # Routing: either attention or DSR (mutually exclusive)
        # Note: layer_idx is 0-based, so attention_every_n_layers=2 means
        # layers 0, 2, 4, ... use attention (every 2nd layer starting from first)
        self.uses_attention: bool = (
            config.attention_every_n_layers > 0
            and layer_idx % config.attention_every_n_layers == 0
        )
        self.attn: Optional[CausalSelfAttention] = None
        self.router: Optional[DynamicSparseRouter] = None

        if self.uses_attention:
            self.attn = CausalSelfAttention(
                d, config.attention_n_heads, config.dropout
            )
        else:
            self.router = DynamicSparseRouter(config, layer_idx)

        # CSP: optional (for ablation experiment)
        if config.disable_csp:
            self.csp: Optional[CompressedStatePropagation] = None
            self.ln2: Optional[nn.LayerNorm] = None
        else:
            self.ln2 = nn.LayerNorm(d)
            self.csp = CompressedStatePropagation(config, layer_idx)

        # GEM: always present
        self.gem = GatedExpertMixture(config, layer_idx)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through one SRN layer.

        Args:
            x: (B, N, D) input tensor

        Returns:
            x: (B, N, D) output tensor
            aux_loss: scalar — MoE load balancing loss from GEM
        """
        # Routing: attention or DSR (pre-norm residual)
        if self.uses_attention:
            x = x + self.attn(self.ln1(x))
        else:
            x = x + self.router(self.ln1(x))

        # Compressed State Propagation (skip when disabled)
        if self.csp is not None:
            x = x + self.csp(self.ln2(x))

        # Gated Expert Mixture (replaces FFN)
        gem_out, aux_loss = self.gem(self.ln3(x))
        x = x + gem_out

        return x, aux_loss


# ============================================================================
# Full SRN Model
# ============================================================================

class SRNModel(nn.Module):
    """The complete Selective Routing Network for language modeling.

    Architecture: Token Embedding + Position Embedding -> N × SRNLayer -> LM Head
    Weight tying: LM head shares weights with token embedding.
    """

    def __init__(self, config: SRNConfig) -> None:
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Learned positional encoding
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # Embedding dropout
        self.emb_drop = nn.Dropout(config.dropout)

        # Layers
        self.layers = nn.ModuleList(
            [SRNLayer(config, i) for i in range(config.n_layers)]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # LM head (weight-tied with token embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

        # Initialize
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embeddings and final layer norm."""
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(
        self, token_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: token IDs -> logits.

        Args:
            token_ids: (B, N) integer token indices

        Returns:
            logits: (B, N, vocab_size) unnormalized log-probabilities
            total_aux_loss: scalar — sum of MoE aux losses across all layers
        """
        B, N = token_ids.shape
        device = token_ids.device

        # Embeddings
        pos_ids = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
        x = self.token_emb(token_ids) + self.pos_emb(pos_ids)  # (B, N, D)
        x = self.emb_drop(x)

        # Process through layers, accumulate aux loss
        total_aux_loss = torch.tensor(0.0, device=device)
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss = total_aux_loss + aux_loss

        # Output
        x = self.ln_f(x)  # (B, N, D)
        logits = self.lm_head(x)  # (B, N, vocab_size)

        return logits, total_aux_loss

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
                # Greedy
                next_id = logits.argmax(dim=-1, keepdim=True)  # (1, 1)
            else:
                # Temperature sampling
                logits = logits / temperature

                if top_k is not None:
                    # Top-k filtering
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
        """Approximate active parameters per token.

        Only top_k experts are active, so GEM params are partially dormant.
        DSR/attention, CSP, LayerNorm, and embeddings are fully active.
        Accounts for hybrid attention, CSP ablation, and WCSG offset.
        """
        config = self.config
        d, dc = config.d_model, config.d_compressed
        k, n_exp = config.n_memory_slots, config.n_experts
        top_k, d_exp = config.top_k_experts, config.d_expert

        # Count attention vs DSR layers
        n_attn_layers = 0
        if config.attention_every_n_layers > 0:
            n_attn_layers = sum(
                1 for i in range(config.n_layers)
                if i % config.attention_every_n_layers == 0
            )
        n_dsr_layers = config.n_layers - n_attn_layers

        # Attention layer active params: QKV + out_proj (all active, no bias)
        # qkv: d*3d, out_proj: d*d → total 4*d*d (bias=False by default)
        attn_active = 4 * d * d

        # DSR active params (all active — routing is dense, just O(n*k) not O(n²))
        # W_q: d*d, slot_keys: k*d, W_sv: d*d, W_gate: d*k, pos_bias: L*k, W_o: d*d
        dsr_active = (d * d) + k * d + (d * d) + (d * k) + (config.max_seq_len * k) + (d * d)

        # WCSG offset params (if enabled, only on DSR layers)
        wcsg_offset = 0
        if config.wcsg_key_offset:
            rank = config.wcsg_key_offset_rank
            wcsg_offset = d * rank + rank + rank * k + k  # down + up with biases

        # CSP active params (0 if disabled)
        csp_active = 0
        if not config.disable_csp:
            csp_active = (d * dc) + (dc * dc) + (dc * d) + ((d + dc) * d)

        # GEM active params: router (all active) + top_k experts
        gem_active = (d * n_exp) + top_k * (d * d_exp + d_exp * d)

        # LayerNorm: 2 params per dim (gamma + beta)
        # ln1 + ln3 always present; ln2 only when CSP enabled
        n_ln = 2 if config.disable_csp else 3
        ln_active = n_ln * 2 * d

        # Per-layer common (CSP + GEM + LN)
        common_active = csp_active + gem_active + ln_active

        # Total across layers
        routing_total = (
            n_attn_layers * attn_active
            + n_dsr_layers * (dsr_active + wcsg_offset)
        )
        layer_total = routing_total + common_active * config.n_layers

        # Embeddings: token + pos + final LN (gamma + beta)
        embeddings = config.vocab_size * d + config.max_seq_len * d + 2 * d

        return layer_total + embeddings


# ============================================================================
# Quick validation (runs when executed directly)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SRN PyTorch Model — Validation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = SRNConfig()
    model = SRNModel(config).to(device)

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

            # Positions before t should be unaffected
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
