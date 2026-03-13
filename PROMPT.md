# SRN Ablation Experiment Run — Desktop (5090 + 4090)

## What is this project?

SRN (Selective Routing Network) is a neural architecture that replaces Transformer self-attention with dynamic sparse routing to learned memory slots. We trained an SRN-150M and a Transformer baseline on TinyStories and found a **0.79 val_loss gap** (SRN: 2.61, Transformer: 1.82). We built an experiment framework with 6 ablation experiments to figure out *why* the gap exists.

## What's already done

- All code is implemented and tested (113 tests pass)
- 30 YAML experiment configs exist in `configs/experiments/` for 3 GPU tiers (2060/4090/5090)
- An experiment runner CLI exists at `scripts/run_experiments.py`
- TinyStories data needs to be prepared on this machine (see below)

## GPU selection

GPU is selected via `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Then edit `.env`:

```env
# Use GPU 0 (5090)
NVIDIA_GPU=0

# Use GPU 1 (4090)
NVIDIA_GPU=1

# Use both GPUs
NVIDIA_GPU=all

# HuggingFace token (for gated datasets/models)
HF_TOKEN=hf_your_token_here
```

Docker compose reads this automatically. To run on a specific GPU:

```bash
# Edit .env to set NVIDIA_GPU=1, then:
docker compose run --rm srn python train.py --config configs/experiments/exp0-srn-4090.yaml

# Or override inline without editing .env:
NVIDIA_GPU=1 docker compose run --rm srn python train.py --config configs/experiments/exp0-srn-4090.yaml
```

To run two experiments in parallel (one per GPU), open two terminals:

```bash
# Terminal 1
NVIDIA_GPU=0 docker compose run --rm srn python train.py --config ...

# Terminal 2
NVIDIA_GPU=1 docker compose run --rm srn python train.py --config ...
```

## KNOWN ISSUE: RTX 5090 needs newer PyTorch

The current Dockerfile uses `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` which only supports up to sm_90 (RTX 4090). The RTX 5090 is sm_120 (Blackwell) and needs PyTorch 2.6+ with CUDA 12.8+.

**To use the 5090**, update the Dockerfile base image:

```dockerfile
# Change this:
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# To something like (check https://hub.docker.com/r/pytorch/pytorch/tags for latest):
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
```

Then rebuild: `docker compose build`

**The 4090 works fine with the current image.** If you only have a 4090, no changes needed.

## What we need to do

### Step 0: Prepare the data

The TinyStories dataset needs to be tokenized before training. Run:

```bash
docker compose run --rm srn python scripts/prepare_tinystories.py --vocab_size 32000
```

This creates `data/tinystories/{train_tokens.bin, val_tokens.bin, tokenizer.json, manifest.json}`.

### Step 1: Run the three baselines (MOST IMPORTANT)

These establish the controlled comparison numbers. Run all three — they give us the real gap and decompose it into compute vs architecture.

**If using both GPUs**, run SRN on one and Transformer on the other:

```bash
# Terminal 1 — GPU 0: SRN baseline
NVIDIA_GPU=0 docker compose run --rm srn python train.py \
  --config configs/experiments/exp0-srn-4090.yaml

# Terminal 2 — GPU 1: Transformer 152M baseline
NVIDIA_GPU=1 docker compose run --rm srn python train.py \
  --config configs/experiments/exp0-transformer-4090.yaml
```

Then run the compute-fair Transformer on whichever GPU finishes first:

```bash
NVIDIA_GPU=0 docker compose run --rm srn python train.py \
  --config configs/experiments/exp0-transformer-small-4090.yaml
```

**Use the config tier that matches your GPU** — `*-4090.yaml` for RTX 4090, `*-5090.yaml` for RTX 5090 (after Dockerfile fix), `*-2060.yaml` for RTX 2060.

**What we expect:**
- Exp0 (SRN): ~162M params, ~112M active/token → val_loss around 2.5-2.6
- Exp0-T (Transformer 152M): ~152M params, 100% active → val_loss around 1.8-1.9
- Exp0-Ts (Transformer 112M): ~113M params, 100% active → val_loss between the two

**The gap decomposition:**
```
Total Gap        = SRN loss − Transformer-152M loss
Compute Gap      = Transformer-152M loss − Transformer-112M loss
Architecture Gap = SRN loss − Transformer-112M loss
```

### Step 2: Run the ablation experiments

After baselines are done, run the 6 ablations. Parallelizable across both GPUs in pairs:

| Priority | ID | Name | What it tests |
|----------|----|------|---------------|
| 1 | Exp1 | Hybrid Attention | Inject attention every 4th layer — is routing the bottleneck? |
| 2 | Exp4 | No CSP | Disable compressed state — is the bottleneck hurting? |
| 3 | Exp5 | Top-k 4 | Double expert activation — is 2-of-8 too sparse? |
| 4 | Exp6 | WCSG Offset | Add score-space offset — is WCSG too limited? |
| 5 | Exp2a | 128 Slots | Double slot count — is 96 slots a bottleneck? |
| 6 | Exp2b | 256 Slots | 4× slot count — diminishing returns? |

Run them in pairs (one per GPU, two terminals):

```bash
# Pair 1
NVIDIA_GPU=0 docker compose run --rm srn python train.py --config configs/experiments/exp1-hybrid-4090.yaml
NVIDIA_GPU=1 docker compose run --rm srn python train.py --config configs/experiments/exp4-nocsp-4090.yaml

# Pair 2
NVIDIA_GPU=0 docker compose run --rm srn python train.py --config configs/experiments/exp5-topk4-4090.yaml
NVIDIA_GPU=1 docker compose run --rm srn python train.py --config configs/experiments/exp6-wcsg-4090.yaml

# Pair 3
NVIDIA_GPU=0 docker compose run --rm srn python train.py --config configs/experiments/exp2a-slots128-4090.yaml
NVIDIA_GPU=1 docker compose run --rm srn python train.py --config configs/experiments/exp2b-slots256-4090.yaml
```

### Step 3 (optional): Full dataset run

Exp3 trains on the full TinyStories dataset (2.5× more steps). Only run if baselines suggest data volume matters:

```bash
NVIDIA_GPU=0 docker compose run --rm srn python train.py \
  --config configs/experiments/exp3-full-4090.yaml
```

## Using the experiment runner (alternative)

The runner handles results tracking but runs experiments sequentially:

```bash
# Dry run — show commands without executing
docker compose run --rm srn python scripts/run_experiments.py --dry-run --all --gpu 4090

# Run baselines
docker compose run --rm srn python scripts/run_experiments.py --experiments 0,0t,0ts --gpu 4090

# Run all ablations
docker compose run --rm srn python scripts/run_experiments.py --experiments 1,2a,2b,4,5,6 --gpu 4090

# Compare results when done
docker compose run --rm srn python scripts/run_experiments.py --compare --gpu 4090
```

For parallel execution across both GPUs, use the direct `train.py` commands above.

## Important notes

- **Docker is required** — all Python runs inside the container
- **GPU selection is via `.env`** — set `NVIDIA_GPU=0` or `NVIDIA_GPU=1` or override inline
- Each experiment saves checkpoints to `checkpoints/exp{ID}-{name}-{gpu}/`
- All experiments are 5,000 steps except Exp3 (full dataset)
- **Estimated time per experiment on 4090:** ~30-45 min
- **Estimated total with both GPUs:** ~3-5 hours for everything

## What to look for in results

After each experiment, check:
1. **Best val_loss** — the key metric
2. **Training loss curve** — is it still decreasing? (might need more steps)
3. **VRAM usage** — verify it fits (logged at step 0)
4. **tok/s** — throughput for time estimates

The big questions we're answering:
- **Exp1 (Hybrid):** If adding 3 attention layers closes most of the gap → routing alone can't match attention for this task
- **Exp4 (No CSP):** If removing CSP helps → the bottleneck is destroying useful information
- **Exp5 (Top-k 4):** If more experts helps → 2-of-8 is too sparse
- **Exp6 (WCSG Offset):** If offset helps → score-space modulation needs more expressiveness
- **Exp2a/2b (Slots):** If more slots helps → 96 slots is an information bottleneck
