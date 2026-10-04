#!/bin/bash
# LOOM at the 329M scale (L=10, hidden 512, 30 routed + 2 shared experts, top-8).
# Paper setting: global batch 1024 x 1024 tokens, 10k steps (~10.7B tokens), lr 2e-4, WSD 10/80/10.
#
#   H=9 DATA_PATH=data/fineweb-edu-1024 bash scripts/loom_329m.sh
#
# MoE capacity is computed per micro-batch, so keep MICRO_BATCH_SAMPLES=16 to reproduce
# the paper runs exactly (samples per rank = 1024 / world size must be a multiple of it).
set -euo pipefail
export ARCH_SIZE="${ARCH_SIZE:-T_moe_L10}"
export ARCH_N_LAYERS="${ARCH_N_LAYERS:-10}"
export LR="${LR:-2e-4}"
export TOTAL_STEPS="${TOTAL_STEPS:-10000}"
export LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.1}"
export LR_DECAY_RATIO="${LR_DECAY_RATIO:-0.1}"
export LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
export MICRO_BATCH_SAMPLES="${MICRO_BATCH_SAMPLES:-16}"
export SNAPSHOT_EVERY_STEPS="${SNAPSHOT_EVERY_STEPS:-1000}"
export SNAPSHOT_KEEP_LAST="${SNAPSHOT_KEEP_LAST:-2}"
export RUN_TAG="${RUN_TAG:-loom_329m_H${H:?set H (number of loops)}}"
exec bash "$(dirname "${BASH_SOURCE[0]}")/train.sh"
