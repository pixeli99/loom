#!/bin/bash
# LOOM at the 1.7B scale (L=15, hidden 1280, 30 routed + 2 shared experts, top-8).
# Paper setting: global batch 1024 x 1024 tokens, 57k steps (~60B tokens), lr 6e-5, WSD.
#
#   H=9 DATA_PATH=data/pretrain-1024 VAL_PATH=data/val-1024 bash scripts/loom_1p7b.sh
#
# MoE capacity is computed per micro-batch, so keep MICRO_BATCH_SAMPLES=8 to reproduce
# the paper runs exactly.
set -euo pipefail
export ARCH_SIZE="${ARCH_SIZE:-T_moe_L15_E32}"
export ARCH_N_LAYERS="${ARCH_N_LAYERS:-15}"
export LR="${LR:-6e-5}"
export TOTAL_STEPS="${TOTAL_STEPS:-57000}"
export LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.01}"
export LR_DECAY_RATIO="${LR_DECAY_RATIO:-0.1}"
export LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
export MICRO_BATCH_SAMPLES="${MICRO_BATCH_SAMPLES:-8}"
export SNAPSHOT_EVERY_STEPS="${SNAPSHOT_EVERY_STEPS:-2000}"
export SNAPSHOT_KEEP_LAST="${SNAPSHOT_KEEP_LAST:-2}"
export RUN_TAG="${RUN_TAG:-loom_1p7b_H${H:?set H (number of loops)}}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh"
