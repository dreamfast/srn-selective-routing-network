#!/usr/bin/env bash
# =============================================================================
# SRN 1B Training Launch Script
# =============================================================================
#
# Pre-flight checklist and sequential launch for the 1B scaling experiment.
# Run from the project root: ./scripts/launch_1b.sh
#
# This script runs INSIDE Docker:
#   docker compose run --rm srn bash scripts/launch_1b.sh
#
# Or step-by-step (recommended for first run):
#   docker compose run --rm srn bash
#   # then run each section manually
#
# Hardware: RTX 5090 (32GB), bf16 precision
# Budget: 5B tokens (76K steps), ~2-3 days per model
# =============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "============================================================"
echo "  SRN 1B Training — Pre-flight Checklist"
echo "============================================================"

# ── Step 1: Check GPU ──────────────────────────────────────────
echo -e "\n${YELLOW}[1/6] Checking GPU...${NC}"
python3 -c "
import torch
assert torch.cuda.is_available(), 'No CUDA GPU found!'
props = torch.cuda.get_device_properties(0)
print(f'  GPU: {props.name}')
print(f'  VRAM: {props.total_memory / 1024**3:.1f} GB')
print(f'  Compute: sm_{props.major}{props.minor}')
assert props.total_memory > 30 * 1024**3, f'Need 32GB+ GPU, got {props.total_memory / 1024**3:.1f} GB'
print(f'  bf16 support: {torch.cuda.is_bf16_supported()}')
"
echo -e "${GREEN}  ✓ GPU OK${NC}"

# ── Step 2: Check data ─────────────────────────────────────────
echo -e "\n${YELLOW}[2/6] Checking FineWeb-Edu data...${NC}"
if [ ! -f "data/fineweb-edu/train_tokens.bin" ] || [ ! -f "data/fineweb-edu/val_tokens.bin" ]; then
	echo -e "${RED}  ✗ FineWeb-Edu data not found!${NC}"
	echo "  Run: python scripts/prepare_fineweb.py --pretrained gpt2"
	echo "  (This downloads ~10B tokens and takes several hours)"
	exit 1
fi
TRAIN_SIZE=$(stat -c%s "data/fineweb-edu/train_tokens.bin" 2>/dev/null || stat -f%z "data/fineweb-edu/train_tokens.bin")
VAL_SIZE=$(stat -c%s "data/fineweb-edu/val_tokens.bin" 2>/dev/null || stat -f%z "data/fineweb-edu/val_tokens.bin")
echo "  train_tokens.bin: $(echo "scale=2; $TRAIN_SIZE / 1073741824" | bc) GB"
echo "  val_tokens.bin:   $(echo "scale=2; $VAL_SIZE / 1073741824" | bc) GB"
if [ ! -f "data/fineweb-edu/tokenizer.json" ]; then
	echo -e "${RED}  ✗ tokenizer.json not found!${NC}"
	exit 1
fi
echo -e "${GREEN}  ✓ Data OK${NC}"

# ── Step 3: Check configs ──────────────────────────────────────
echo -e "\n${YELLOW}[3/6] Validating configs...${NC}"
python3 -c "
from omegaconf import OmegaConf

# SRN config
srn = OmegaConf.load('configs/srn-1b-hybrid.yaml')
assert srn.model.top_k_experts == 2, f'Expected top_k=2, got {srn.model.top_k_experts}'
assert srn.model.attention_every_n_layers == 4, 'Missing hybrid attention'
assert srn.train.precision == 'bf16', 'Expected bf16'
srn_eff = srn.train.micro_batch * srn.train.accum_steps * srn.train.seq_len
print(f'  SRN:   {srn.train.micro_batch}×{srn.train.accum_steps}×{srn.train.seq_len} = {srn_eff:,} tok/step')

# Dense config
dense = OmegaConf.load('configs/dense-411m.yaml')
assert dense.train.model_type == 'dense', 'Missing model_type=dense'
assert dense.train.precision == 'bf16', 'Expected bf16'
dense_eff = dense.train.micro_batch * dense.train.accum_steps * dense.train.seq_len
print(f'  Dense: {dense.train.micro_batch}×{dense.train.accum_steps}×{dense.train.seq_len} = {dense_eff:,} tok/step')

assert srn_eff == dense_eff, f'Batch mismatch: SRN={srn_eff} vs Dense={dense_eff}'
print(f'  Effective batch: {srn_eff:,} tokens/step ✓')
print(f'  Max steps: {srn.train.max_steps:,} ({srn.train.max_steps * srn_eff / 1e9:.2f}B tokens)')
"
echo -e "${GREEN}  ✓ Configs OK${NC}"

# ── Step 4: VRAM dry-run (SRN) ─────────────────────────────────
echo -e "\n${YELLOW}[4/6] VRAM dry-run: SRN 1B hybrid...${NC}"
python3 scripts/vram_dry_run.py \
	--config configs/srn-1b-hybrid.yaml \
	--precision bf16
SRN_EXIT=$?
if [ $SRN_EXIT -ne 0 ]; then
	echo -e "${RED}  ✗ SRN VRAM dry-run FAILED — will OOM${NC}"
	echo "  Try reducing micro_batch to 1 or enabling gradient checkpointing"
	exit 1
fi
echo -e "${GREEN}  ✓ SRN fits in VRAM${NC}"

# ── Step 5: VRAM dry-run (Dense) ───────────────────────────────
echo -e "\n${YELLOW}[5/6] VRAM dry-run: Dense 411M...${NC}"
python3 scripts/vram_dry_run.py \
	--config configs/dense-411m.yaml \
	--precision bf16
DENSE_EXIT=$?
if [ $DENSE_EXIT -ne 0 ]; then
	echo -e "${RED}  ✗ Dense VRAM dry-run FAILED — will OOM${NC}"
	exit 1
fi
echo -e "${GREEN}  ✓ Dense fits in VRAM${NC}"

# ── Step 6: Summary ───────────────────────────────────────────
echo -e "\n${YELLOW}[6/6] Pre-flight complete!${NC}"
echo "============================================================"
echo -e "${GREEN}  All checks passed. Ready to train.${NC}"
echo "============================================================"
echo ""
echo "  To train SRN 1B hybrid:"
echo "    python train.py --config configs/srn-1b-hybrid.yaml"
echo ""
echo "  To train Dense 411M baseline:"
echo "    python train.py --config configs/dense-411m.yaml"
echo ""
echo "  To resume after interruption:"
echo "    python train.py --config configs/srn-1b-hybrid.yaml --resume"
echo "    python train.py --config configs/dense-411m.yaml --resume"
echo ""
echo "  Recommended order: SRN first (longer), then Dense."
echo "  Use tmux/screen to keep sessions alive."
echo "============================================================"
