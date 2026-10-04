#!/bin/bash
# Launch one LOOM run with torchrun (single node or multi-node).
#
# The architecture overrides below are the LOOM recipe used in the paper; the model size,
# schedule and data come from the environment. The presets in scripts/loom_329m.sh and
# scripts/loom_1p7b.sh set those for the two paper scales.
#
#   H=9 DATA_PATH=data/fineweb-edu-1024 bash scripts/train.sh
#   H=3 DATA_PATH=... TOTAL_STEPS=20 GLOBAL_BATCH_SIZE=16 MICRO_BATCH_SAMPLES=8 bash scripts/train.sh
#
# Multi-node: run the same command on every node with NNODES, NODE_RANK, MASTER_ADDR and
# MASTER_PORT set (WORLD_SIZE / RANK as injected by most schedulers are also accepted).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---- what to run ------------------------------------------------------------
H="${H:?set H (number of loops): 1, 3, 6, 9 or 12}"
ARCH_SIZE="${ARCH_SIZE:-T_moe_L15_E32}"
ARCH_N_LAYERS="${ARCH_N_LAYERS:-15}"
ARCH_NET="${ARCH_NET:-looped_transformer}"
DATA_PATH="${DATA_PATH:?set DATA_PATH to a packed DocumentDataset directory}"
# Where weights go. Default: checkpoints/<project>/<run> under the repo.
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
MICRO_BATCH_SAMPLES="${MICRO_BATCH_SAMPLES:-4}"
LR="${LR:-6e-5}"
TOTAL_STEPS="${TOTAL_STEPS:-20000}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.1}"
LR_DECAY_RATIO="${LR_DECAY_RATIO:-0.1}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.1}"
VAL_PATH="${VAL_PATH:-}"                 # eval on this pack's epoch_0 instead of DATA_PATH's holdout
CKPT_EVERY="${CKPT_EVERY:-500}"          # live checkpoint (overwritten) -> resume granularity
export SNAPSHOT_EVERY_STEPS="${SNAPSHOT_EVERY_STEPS:-2000}"   # immutable full copies (19 GB at 1.7B)
export SNAPSHOT_KEEP_LAST="${SNAPSHOT_KEEP_LAST:-1}"           # prune older snapshots; 0 keeps all
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
BP_K="${BP_K:-3}"
SEED="${SEED:-0}"

# ---- topology ---------------------------------------------------------------
# WORLD_SIZE / RANK are read as node count / node rank when NNODES / NODE_RANK are unset.
NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch;print(torch.cuda.device_count())')}"
NNODES="${NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
WORLD="$((NNODES * NPROC_PER_NODE))"

if (( GLOBAL_BATCH_SIZE % WORLD != 0 )); then
  echo "ERROR: global_batch_size ${GLOBAL_BATCH_SIZE} not divisible by world size ${WORLD}" >&2
  exit 1
fi
SAMPLES_PER_RANK=$((GLOBAL_BATCH_SIZE / WORLD))
if (( SAMPLES_PER_RANK % MICRO_BATCH_SAMPLES != 0 )); then
  echo "ERROR: samples/rank ${SAMPLES_PER_RANK} not divisible by micro ${MICRO_BATCH_SAMPLES}" >&2
  exit 1
fi

# ---- runtime env ------------------------------------------------------------
export DIST_BACKEND="${DIST_BACKEND:-nccl}"
export MOE_BACKEND="${MOE_BACKEND:-npu_fused}"   # packed expert weights + grouped GEMM (also the CUDA path)
export MOE_ROUTING="${MOE_ROUTING:-python}"      # token permute/unpermute in PyTorch
export PREFIXLM_ATTN_BACKEND="${PREFIXLM_ATTN_BACKEND:-fa2}"
export PREFIXLM_FUSION_ROPE="${PREFIXLM_FUSION_ROPE:-0}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
export MODULE_DIAG_FORCE_OFF="${MODULE_DIAG_FORCE_OFF:-0}"
export MODULE_DIAG_INTERVAL="${MODULE_DIAG_INTERVAL:-50}"
export MODULE_DIAG_ACT="${MODULE_DIAG_ACT:-1}"
export MODULE_DIAG_RANK="${MODULE_DIAG_RANK:-1}"
export MODULE_DIAG_GRAD="${MODULE_DIAG_GRAD:-0}"
export MODULE_DIAG_LIVE_PLOT="${MODULE_DIAG_LIVE_PLOT:-0}"
export EXPERT_FREQ_EVERY="${EXPERT_FREQ_EVERY:-1}"
export LAMBDA_LOG_EVERY="${LAMBDA_LOG_EVERY:-10}"
export NUM_LOOPS="$H"
export H_LIST_OVERRIDE="$H"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# ---- run name / checkpoint dir ---------------------------------------------
RUN_TAG="${RUN_TAG:-loom_L${ARCH_N_LAYERS}_H${H}}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
FULL_RUN_NAME="${FULL_RUN_NAME:-${RUN_TAG}_s${TOTAL_STEPS}_${RUN_TIMESTAMP}}"

# ---- LOOM recipe -------------------------------------------------------------
# residual scaling   arch.cycle_residual_scale, arch.cycle_scale_lambda
# embedding inject   arch.embed_inject_*
# per-loop routers   arch.moe.router_per_loop, arch.moe.router_num_loops
# Looping Residual   arch.attn_res=true arch.attn_res_scope=dual_axis, arch.dual_axis_*
# MoE output norm    arch.ffn_branch_rms
# segmented backprop arch.bp_segment_len (backward + optimizer step every BP_K loops)
LAM="${CYCLE_SCALE_LAMBDA:-0.5}"
BETA="${DUAL_AXIS_BETA_INIT:-0.5}"
# Ablation switches; defaults are the LOOM values.
CYCLE_RESIDUAL_SCALE="${CYCLE_RESIDUAL_SCALE:-true}"         # false: loop residual γ=1 (inject schedule unaffected)
EMBED_INJECT_MODE="${EMBED_INJECT_MODE:-loop_start_mix_soft}" # none: no embedding re-injection at loop start
FFN_BRANCH_RMS="${FFN_BRANCH_RMS:-affine}"                    # none: no RMSNorm on the MoE output
read -r -d '' ARCH_OVERRIDES <<OVR || true
arch.ffn_type=moe
arch.moe.shared_expert=true arch.moe.shared_expert_size_multiplier=1
arch.moe.top_k_includes_shared=true
arch.moe.gate_type=sigmoid arch.moe.router_dtype=float32
arch.moe.load_balance_bias_lr=2.0e-3 arch.moe.router_z_loss_coef=1.0e-3
arch.moe.router_per_loop=true arch.moe.router_num_loops=${H}
arch.cycle_residual_scale=${CYCLE_RESIDUAL_SCALE}
arch.cycle_scale_lambda=${LAM} arch.cycle_scale_lambda_learnable=false arch.cycle_scale_lambda_mode=shared
arch.cycle_eps_shared_span=0.0 arch.cycle_eps_t0=0.0 arch.cycle_eps_time_mode=inv_h
arch.cycle_eps_t_floor=1.0 arch.cycle_eps_span_mode=progress
arch.layernorm_scaling=false arch.layernorm_scaling_power=0.5 arch.layernorm_scaling_ell_mode=ell
arch.ffn_branch_rms=${FFN_BRANCH_RMS} arch.attn_branch_rms=none arch.attn_branch_rms_loops=0
arch.embed_inject_mode=${EMBED_INJECT_MODE} arch.embed_inject_learnable=false
arch.embed_inject_from_t1=false arch.embed_inject_when=start arch.embed_inject_norm=none
arch.embed_inject_time_mode=inv_t arch.embed_inject_depth_scale=true arch.embed_inject_swap=false
arch.attn_res=true arch.attn_res_scope=dual_axis arch.attn_res_max_slots=0
arch.dual_axis_apply=attn arch.dual_axis_residual_mode=x_plus_r
arch.dual_axis_update_src=o arch.dual_axis_l_persist=false arch.dual_axis_nd_mode=ema
arch.dual_axis_beta_learnable=false
arch.dual_axis_beta_init=${BETA} arch.dual_axis_beta_p_init=${BETA}
arch.dual_axis_beta_h_init=${BETA} arch.dual_axis_beta_l_init=${BETA}
arch.dual_axis_heads=1 arch.dual_axis_h_layout=shared
arch.dual_axis_h_write=every_layer arch.dual_axis_h_update_src=l
arch.dual_axis_fuse_mode=sum arch.dual_axis_content=false arch.dual_axis_beta_l_per_loop=true
arch.bp_segment_len=${BP_K} arch.loop_id_embed_mode=none
OVR

TRAIN_OVERRIDES="lr_schedule=wsd lr_warmup_ratio=${LR_WARMUP_RATIO} lr_decay_ratio=${LR_DECAY_RATIO} lr_min_ratio=${LR_MIN_RATIO}"
TRAIN_OVERRIDES="${TRAIN_OVERRIDES} checkpoint_every_steps=${CKPT_EVERY} log_interval=10 eval_interval=${EVAL_INTERVAL}"
TRAIN_OVERRIDES="${TRAIN_OVERRIDES} clip_grad_norm=1.0 skip_bad_step=true"
VAL_ARGS=""
if [ -n "${VAL_PATH}" ]; then
  [ -d "${VAL_PATH}/epoch_0" ] || { echo "FATAL: VAL_PATH ${VAL_PATH} has no epoch_0" >&2; exit 1; }
  VAL_ARGS="data.val_path=${VAL_PATH}"
fi
TRAIN_OVERRIDES="${TRAIN_OVERRIDES} skip_loss_delta=0.18 skip_loss_rel=0.06 skip_loss_ratio=0.09 skip_loss_hard=1.5 skip_loss_ema=0.9 skip_delta_warmup_steps=0"
TRAIN_OVERRIDES="${TRAIN_OVERRIDES} skip_abort_on_spike=${SKIP_ABORT_ON_SPIKE:-true} skip_abort_consecutive=${SKIP_ABORT_CONSECUTIVE:-0}"

CKPT_ARGS=()
if [[ -n "${CHECKPOINT_PATH}" ]]; then
  CKPT_ARGS+=("checkpoint_path='${CHECKPOINT_PATH}'")
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM:-}" && "${RESUME_FROM}" != "none" ]]; then
  RESUME_ARGS+=("resume_from='${RESUME_FROM}'")
fi

cat <<INFO
================== LOOM ===================
H (loops)        : ${H}     arch ${ARCH_SIZE} L${ARCH_N_LAYERS}
world            : ${NNODES} nodes x ${NPROC_PER_NODE} = ${WORLD} ranks (node_rank ${NODE_RANK})
batch            : global ${GLOBAL_BATCH_SIZE}, ${SAMPLES_PER_RANK}/rank, micro ${MICRO_BATCH_SAMPLES}, accum $((SAMPLES_PER_RANK / MICRO_BATCH_SAMPLES))
schedule         : lr ${LR}, ${TOTAL_STEPS} steps, WSD ${LR_WARMUP_RATIO}/${LR_DECAY_RATIO}
tokens (nominal) : $((GLOBAL_BATCH_SIZE * TOTAL_STEPS / 1000))k x seq_len
data             : ${DATA_PATH}
attention / moe  : ${PREFIXLM_ATTN_BACKEND} / ${MOE_BACKEND}+${MOE_ROUTING}
ablation         : bp_segment_len=${BP_K} cycle_residual_scale=${CYCLE_RESIDUAL_SCALE} embed_inject_mode=${EMBED_INJECT_MODE} ffn_branch_rms=${FFN_BRANCH_RMS}
run name         : ${FULL_RUN_NAME}
checkpoints      : ${CHECKPOINT_PATH:-checkpoints/<project>/<run> (under cwd)}
resume           : ${RESUME_FROM:-none} (set RESUME_FROM=<run or snapshot dir> to continue)
master           : ${MASTER_ADDR}:${MASTER_PORT}
===========================================
INFO

exec torchrun \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  pretrain.py \
  "arch/net@arch=${ARCH_NET}" \
  "arch/size@arch=${ARCH_SIZE}" \
  arch.num_loops="${H}" \
  arch.n_layers="${ARCH_N_LAYERS}" \
  lm_mode=causal \
  global_batch_size="${GLOBAL_BATCH_SIZE}" \
  micro_batch_samples="${MICRO_BATCH_SAMPLES}" \
  lr="${LR}" \
  total_steps="${TOTAL_STEPS}" \
  downstream_eval_interval=0 \
  optimizer=adamw beta1=0.9 beta2=0.95 weight_decay=0.1 ema=null \
  seed="${SEED}" \
  run_name="${FULL_RUN_NAME}" \
  data.path="${DATA_PATH}" \
  data.format=document \
  ${VAL_ARGS} \
  ${ARCH_OVERRIDES} \
  ${TRAIN_OVERRIDES} \
  "${CKPT_ARGS[@]}" \
  "${RESUME_ARGS[@]}"
