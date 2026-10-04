#!/bin/bash
# 0-shot lm_eval for one LOOM checkpoint (single node).
#
#   bash eval/run.sh /path/to/ckpt
#   CKPT_PATH=/path/to/ckpt NPROC_PER_NODE=1 bash eval/run.sh
#
# The checkpoint and the model code tree are the only things outside eval/.
# MODEL_CODE_ROOT picks the tree (default: the repo this eval/ sits in); the result
# files record the files actually imported and their commit.
set -euo pipefail
# The body is one brace group: bash parses it whole before running, so editing this file
# while a run is in progress cannot re-execute part of it.
{
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$EVAL_DIR"

if [[ -z "${CKPT_PATH:-}" && $# -ge 1 ]]; then
  CKPT_PATH="$1"
fi
: "${CKPT_PATH:?CKPT_PATH is required (env or first argument)}"
CKPT_PATH="$(cd "$CKPT_PATH" && pwd)"

MODEL_CODE_ROOT="$(cd "${MODEL_CODE_ROOT:-${EVAL_DIR}/..}" && pwd)"
[[ -f "${MODEL_CODE_ROOT}/pretrain.py" && -d "${MODEL_CODE_ROOT}/models" ]] \
  || { echo "ERROR: MODEL_CODE_ROOT=${MODEL_CODE_ROOT} has no pretrain.py / models/" >&2; exit 1; }

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LMEVAL_TASKS="${LMEVAL_TASKS:-openbookqa,winogrande,arc_challenge,arc_easy,hellaswag,social_iqa,piqa}"
if [[ -z "${EVAL_OUTPUT_DIR:-}" ]]; then
  if [[ "$(basename "$(dirname "$CKPT_PATH")")" == snapshots ]]; then
    # Snapshots are pruned by SNAPSHOT_KEEP_LAST; keep their results in the run dir.
    EVAL_OUTPUT_DIR="$(dirname "$(dirname "$CKPT_PATH")")/eval_results/$(basename "$CKPT_PATH")/official_lmeval7_${STAMP}"
  else
    EVAL_OUTPUT_DIR="${CKPT_PATH}/eval_results/official_lmeval7_${STAMP}"
  fi
fi

export CKPT_PATH MODEL_CODE_ROOT EVAL_OUTPUT_DIR LMEVAL_TASKS
# Datasets and the tokenizer are cached under EVAL_HF_ROOT (default eval/hf_cache).
# EVAL_OFFLINE=1 reuses that cache without network access.
export EVAL_OFFLINE="${EVAL_OFFLINE:-0}"
# eval/ first (the lm_eval wrapper), then the model code tree.
export PYTHONPATH="${EVAL_DIR}:${MODEL_CODE_ROOT}"
export LMEVAL_TASK_DIR="${EVAL_DIR}/tasks"
export EVAL_NUM_FEWSHOT="${EVAL_NUM_FEWSHOT:-0}"
export EVAL_BATCH_SIZE_MCQ="${EVAL_BATCH_SIZE_MCQ:-32}"
export LMEVAL_LIMIT="${LMEVAL_LIMIT:-}"
export LMEVAL_LOG_SAMPLES="${LMEVAL_LOG_SAMPLES:-0}"
export LMEVAL_BOOTSTRAP_ITERS="${LMEVAL_BOOTSTRAP_ITERS:-1000}"

# Same kernels as scripts/train.sh.
export DIST_BACKEND="${DIST_BACKEND:-nccl}"
export MOE_BACKEND="${MOE_BACKEND:-npu_fused}"
export MOE_ROUTING="${MOE_ROUTING:-python}"
export PREFIXLM_ATTN_BACKEND="${PREFIXLM_ATTN_BACKEND:-fa2}"
export PREFIXLM_FUSION_ROPE="${PREFIXLM_FUSION_ROPE:-0}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export MODULE_DIAG_FORCE_OFF=1
# Scoring numerics. Capacity dropping on makes a request's score depend on what it is packed
# with, so it is off. bf16 parameters still flip answers with batch size (bf16 MoE combine);
# fp32 does not. The holdout check always runs with capacity on and bf16 parameters, as
# pretrain.run_eval did. See eval/README.md, "Scoring numerics".
export MOE_SKIP_CAPACITY="${MOE_SKIP_CAPACITY:-1}"
export EVAL_PARAM_DTYPE="${EVAL_PARAM_DTYPE:-fp32}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch;print(torch.cuda.device_count())')}"
MASTER_PORT="${MASTER_PORT:-$((29700 + RANDOM % 200))}"
export MASTER_PORT

mkdir -p "$EVAL_OUTPUT_DIR"
LOG="${EVAL_OUTPUT_DIR}/eval.log"
echo "ckpt=${CKPT_PATH}"
echo "code=${MODEL_CODE_ROOT}  nproc=${NPROC_PER_NODE}  out=${EVAL_OUTPUT_DIR}"

set +e  # keep going to report rc
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  "${EVAL_DIR}/lm_eval_loom.py" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
echo "OUT=${EVAL_OUTPUT_DIR}  rc=${rc}"
exit "$rc"
}
