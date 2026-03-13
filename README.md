# SRN: Selective Routing Network

A neural network architecture that replaces Transformer self-attention with dynamic sparse routing to learned memory slots. Designed to enable large model capacity on consumer GPUs (6GB VRAM).

**Key approach: Windowed Causal Score Gating (WCSG)** — a score-side gating mechanism for causal, position-dependent routing with O(1) extra memory overhead, instead of the O(n²) attention matrix.

## Table of Contents

- [SRN: Selective Routing Network](#srn-selective-routing-network)
  - [Table of Contents](#table-of-contents)
  - [Architecture](#architecture)
    - [Dynamic Sparse Router (DSR)](#dynamic-sparse-router-dsr)
    - [Compressed State Propagation (CSP)](#compressed-state-propagation-csp)
    - [Gated Expert Mixture (GEM)](#gated-expert-mixture-gem)
  - [Windowed Causal Score Gating (WCSG)](#windowed-causal-score-gating-wcsg)
    - [The Problem](#the-problem)
    - [The Solution: Modulate Scores, Not Keys](#the-solution-modulate-scores-not-keys)
    - [Why This Works](#why-this-works)
    - [The Design Trade-off](#the-design-trade-off)
    - [Related Work](#related-work)
  - [Results](#results)
  - [Quick Start](#quick-start)
    - [Prerequisites](#prerequisites)
    - [Train](#train)
    - [Generate Text](#generate-text)
    - [Validate Architecture](#validate-architecture)
    - [Without Docker](#without-docker)
  - [Model Configuration](#model-configuration)
  - [Training Details](#training-details)
  - [Complexity Comparison](#complexity-comparison)
  - [Bugs Fixed from Original NumPy PoC](#bugs-fixed-from-original-numpy-poc)
  - [Project Structure](#project-structure)
  - [License](#license)
  - [Author](#author)

## Architecture

SRN layers consist of three modules stacked with pre-norm residual connections:

```
Input token IDs: (B, N)
    ↓
Token Embedding + Positional Embedding + Dropout
    ↓
┌─────────────────────────────────────────────┐
│  SRN Layer (×8)                             │
│                                             │
│  x = x + DSR(LayerNorm(x))    ← routing    │
│  x = x + CSP(LayerNorm(x))    ← bottleneck │
│  x = x + GEM(LayerNorm(x))    ← experts    │
│                                             │
│  DSR uses WCSG for causal routing           │
│  GEM returns aux_loss for load balancing    │
└─────────────────────────────────────────────┘
    ↓
Final LayerNorm
    ↓
LM Head (weight-tied with token embedding)
    ↓
Logits: (B, N, vocab_size)
```

### Dynamic Sparse Router (DSR)

Replaces self-attention. Each token computes query vectors that are scored against k learned memory slots via multi-head routing. The routing scores are modulated by WCSG (see below) to ensure causality, then softmaxed into routing weights. Each token aggregates information from the static slot values weighted by these scores.

- **Complexity:** O(n·k·d) instead of O(n²·d)
- **Memory:** Slot keys/values are (k, D) — shared across positions, no per-position expansion

### Compressed State Propagation (CSP)

Forces information through a learned bottleneck between layers. Compresses the state from D dimensions to D/4, processes it, then expands back via a sigmoid gate. This acts as learned information selection — only what matters passes through.

### Gated Expert Mixture (GEM)

Replaces the dense FFN. Multiple small expert networks (8 total) with a lightweight router that selects the top-k (2) experts per token. All experts are computed in parallel via `torch.einsum` (no Python loops). Includes Switch Transformer load balancing auxiliary loss to prevent expert collapse.

- **Parameter efficiency:** Only 2 of 8 experts fire per token, so ~75% of GEM parameters are dormant

## Windowed Causal Score Gating (WCSG)

WCSG is a core mechanism in SRN. It addresses a practical problem: **how do you make slot-based routing causal (so position t can't see future tokens) without blowing up memory?**

### The Problem

In a standard Transformer, causal masking is straightforward — you mask the attention matrix so position t only attends to positions ≤ t. But SRN doesn't have an attention matrix. It has **global memory slots** (shared across all positions), and tokens route to these slots via learned scores.

The naive approach to making this causal would be to create **per-position slot keys** — adapting the slot representations for each position based on past context. But this expands the slot key tensor from (k, D) to (B, N, k, D), which at our model dimensions would require **~4.3GB of VRAM per layer** — completely infeasible.

### The Solution: Modulate Scores, Not Keys

Instead of adapting the slot keys themselves, WCSG modulates the **routing scores** with a per-position causal gate:

```
1. Compute a causal windowed mean of past tokens:
   For position t, average tokens [max(0, t-W+1) .. t]
   Implementation: F.avg_pool1d with left-padding (no future leakage)
   → (B, N, D)

2. Project to a per-position gate:
   gate = sigmoid(windowed_mean @ W_gate)
   → (B, N, k)  — one scalar per slot per position

3. Compute raw routing scores:
   scores = Q @ slot_keys^T
   → (B, h, N, k)

4. Modulate scores with the causal gate:
   scores = scores * gate.unsqueeze(1)
   This multiplicatively shapes which slots each position prefers,
   based on its local causal context.

5. Add learned positional routing bias:
   scores += pos_bias[:N]
   → (B, h, N, k)

6. Softmax → routing weights → aggregate from static slot values
```

### Why This Works

- **Truly causal:** The windowed mean at position t only includes tokens [max(0, t-W+1) .. t]. The left-padding in `avg_pool1d` guarantees no future information leaks. This has been verified empirically — perturbing token t does not change logits at positions < t.

- **O(1) extra memory:** The gate tensor is (B, N, k) ≈ 1MB per layer. Slot keys and values stay at (k, D) — no per-position expansion.

- **Proper normalization:** Early positions (where the window extends before the sequence start) are normalized by the *actual* number of tokens in the window, not the fixed window size W. Position 0 divides by 1, position 5 divides by min(6, W). This prevents early-position attenuation from zero-padding.

- **Numerical safety:** The gate is clamped to `min=1e-4` to prevent all-zero routing scores, which would cause vanishing gradients through the sigmoid and dead neurons.

### The Design Trade-off

Score modulation is not equivalent to key adaptation:

```
softmax((Q @ K^T) * g) ≠ softmax(Q @ (K*g)^T)
```

The gate can suppress routes (drive scores toward zero) but cannot transform key directions. A gate of 0.01 effectively says "don't route here," while a gate of 1.0 says "route normally." But it can't rotate the key space to create new routing patterns that don't exist in the base keys.

This is an **intentional capacity-for-efficiency trade-off.** Full per-position key adaptation would be more expressive but requires ~4.3GB/layer. WCSG achieves ~1MB/layer while still providing position-dependent, causal routing behavior.

### Related Work

This project sits in an active research area: routing tokens to memory-like structures with gating. The claim here is not that the overall idea is unprecedented, but that WCSG takes a different mechanism within that design space.

- **GSA** (gated slot attention variants) — closest in spirit, but primarily gates memory writes/forgetting; WCSG gates read-side routing scores using local causal context
- **DeepSeek Engram** — hash-based memory lookup with context-aware control; related direction, but not learned slot routing via score modulation
- **Memory layer / external memory papers** — broadly related to token-memory interaction, but with different routing and gating specifics
- **Slot Attention** (Locatello et al., 2020) — iterative slot refinement, but not causal and not for autoregressive LMs
- **Switch Transformer** (Fedus et al., 2022) — sparse MoE routing, but routes to experts not memory slots, and uses no causal gating
- **Routing Transformers** (Roy et al., 2021) — learned routing for attention, but still O(n²) within clusters

WCSG's contribution is a practical read-side alternative: causal, position-dependent score modulation for global memory-slot routing with O(1) extra memory overhead.

## Results

Trained on TinyShakespeare (character-level, ~1.1MB) for 5,000 steps on an RTX 2060:

| Metric | Value |
|--------|-------|
| Best validation loss | 2.269 |
| Validation perplexity | 10.06 |
| Peak GPU memory | 2.3 GB |
| Total parameters | 27.8M |
| Active per token | 15.2M (54.5%) |
| Training time | ~50 minutes |

The model learns Shakespeare's formatting (speaker labels, verse structure, punctuation) and common English character patterns. With only 5K steps of character-level training, word-level coherence is limited — this is a proof-of-concept, not a production model.

### TinyStories SRN vs Transformer (Fit Snapshot)

Early apples-to-apples fit comparison on RTX 2060 6GB with TinyStories setup:

| Model | Params | Micro batch | Peak VRAM |
|-------|--------|-------------|-----------|
| SRN (WCSG) | ~150M | 16 | 4624M |
| Transformer baseline | 152M | 2 | 4519M |

This is a useful capacity/efficiency data point: at similar parameter count, SRN fits a much larger micro batch than the dense Transformer on the same GPU.

## Quick Start

### Prerequisites

- Docker with NVIDIA Container Toolkit (GPU support)
- NVIDIA GPU with 6GB+ VRAM

### Train

```bash
# Build the container
docker compose build

# Download dataset and train (5000 steps, ~50 min on RTX 2060)
docker compose run --rm srn python train.py --max_steps 5000

# Or customize training
docker compose run --rm srn python train.py \
  --max_steps 20000 \
  --eval_interval 500 \
  --log_interval 100 \
  --lr 3e-4
```

### Generate Text

```bash
# Generate from a prompt
docker compose run --rm srn python generate.py \
  --checkpoint checkpoints/best.pt \
  --prompt "ROMEO:" \
  --max_tokens 500 \
  --temperature 0.8

# With top-k sampling
docker compose run --rm srn python generate.py \
  --checkpoint checkpoints/best.pt \
  --prompt "To be, or not to be" \
  --max_tokens 300 \
  --temperature 0.6 \
  --top_k 10

# Evaluate perplexity
docker compose run --rm srn python generate.py \
  --checkpoint checkpoints/best.pt \
  --eval --stats
```

### Validate Architecture

```bash
# Run validation suite (causal masking tests, architecture comparison)
docker compose run --rm srn python validate.py

# Run the original NumPy proof-of-concept
docker compose run --rm srn python srn_architecture.py
```

### Without Docker

```bash
# Requires Python 3.10+, PyTorch 2.x with CUDA
pip install torch numpy tqdm requests
python train.py --max_steps 5000
```

## Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 512 | Model dimension |
| `n_layers` | 8 | Number of SRN layers |
| `n_memory_slots` | 64 | Routing targets per layer |
| `n_experts` | 8 | Expert networks per layer |
| `top_k_experts` | 2 | Active experts per token |
| `d_expert` | 256 | Expert hidden dimension |
| `n_heads_route` | 4 | Multi-head routing |
| `d_compressed` | 128 | CSP bottleneck (d_model/4) |
| `causal_window` | 32 | WCSG window size |
| `max_seq_len` | 256 | Context length |
| `vocab_size` | 65 | Character-level Shakespeare |

## Training Details

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95)) |
| Schedule | Cosine decay with 100-step linear warmup |
| Loss | CrossEntropy + 0.01 × MoE auxiliary load balancing |
| Precision | Mixed FP16 (forward) / FP32 (loss) |
| Batch size | 32 effective (micro=16, accumulation=2) |
| Gradient clipping | max_norm=1.0 |
| Sequence length | 256 |

## Complexity Comparison

| Property | Transformer | SRN |
|----------|-------------|-----|
| Core operation | O(n²·d) attention | O(n·k·d) routing |
| At seq_len=32K | 2.20T ops | 4.29G ops (512x fewer) |
| Parameter activity | 100% active | ~55% active per token |
| Memory scaling | Quadratic in seq_len | Linear in seq_len |

## Bugs Fixed from Original NumPy PoC

The PyTorch port discovered and fixed several issues in the original `srn_architecture.py`:

1. **Causal masking violation** — `x.mean(axis=1)` computed a global mean across all positions, leaking future information into past positions. Fixed with the causal windowed mean.
2. **Dead parameter** — `W_gate_slot` (262K params) was allocated but never used in the forward pass. Removed entirely.
3. **Double-residual bug** — CSP internally computed `x + gate * expanded`, then the layer added another residual `x + csp(x)`, creating a double skip connection. Made configurable via `csp_internal_residual`.
4. **Dropout never applied** — Dropout rate was configured but never actually called during forward passes. Now properly applied.
5. **Sequential expert loop** — Expert computation used a Python loop instead of vectorized operations. Replaced with `torch.einsum` for parallel computation.

## Project Structure

```
srn_model.py           # PyTorch SRN model with WCSG (core implementation)
train.py               # Training loop (AdamW, cosine schedule, mixed precision, grad accum)
generate.py            # Text generation CLI + perplexity evaluation
validate.py            # Validation suite + comparison with original NumPy PoC
data.py                # TinyShakespeare dataset + character-level tokenizer
srn_architecture.py    # Original NumPy proof-of-concept (reference, untouched)
Dockerfile             # PyTorch 2.5.1 + CUDA 12.4
docker-compose.yml     # GPU passthrough + volume mounts
requirements.txt       # numpy, tqdm, requests
```

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

## Author

Built by Nathan Sapwell with Claude (Anthropic). Experimental research — use at your own risk. Honestly I am not a smart maths guy so I am just messing with this to learn about LLM stuff.
