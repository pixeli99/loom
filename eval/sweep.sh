#!/bin/bash
# Evaluate several checkpoints on this node, one GPU per checkpoint, then print the table.
#
#   bash eval/sweep.sh /path/H3 /path/H6 /path/H9 /path/H12
#   GPUS=0,1 STAMP=h_sweep bash eval/sweep.sh ...
#
# Checkpoints are dealt round-robin to GPUS (default: all visible); each GPU runs its share
# in order. Every other knob is passed through to run.sh via the environment.
set -uo pipefail
# The body is one brace group: bash parses it whole before running, so editing this file
# while a run is in progress cannot re-execute part of it.
{
EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ $# -ge 1 ]] || { echo "usage: bash eval/sweep.sh CKPT [CKPT ...]" >&2; exit 1; }

GPUS="${GPUS:-$(python -c 'import torch;print(",".join(map(str, range(torch.cuda.device_count()))))')}"
IFS=, read -r -a GPU_LIST <<< "$GPUS"
export STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

# Same rule as run.sh: snapshots are pruned, so their results go to the run dir.
out_dir() {
  local ck; ck="$(cd "$1" && pwd)"
  if [[ "$(basename "$(dirname "$ck")")" == snapshots ]]; then
    echo "$(dirname "$(dirname "$ck")")/eval_results/$(basename "$ck")/official_lmeval7_${STAMP}"
  else
    echo "${ck}/eval_results/official_lmeval7_${STAMP}"
  fi
}

CKPTS=("$@")
OUTS=()
for c in "${CKPTS[@]}"; do
  o="$(out_dir "$c")" || exit 1
  mkdir -p "$o"
  OUTS+=("$o")
done

pids=()
for g in "${!GPU_LIST[@]}"; do
  (
    for ((i = g; i < ${#CKPTS[@]}; i += ${#GPU_LIST[@]})); do
      CUDA_VISIBLE_DEVICES="${GPU_LIST[$g]}" NPROC_PER_NODE=1 \
        CKPT_PATH="${CKPTS[$i]}" EVAL_OUTPUT_DIR="${OUTS[$i]}" \
        bash "${EVAL_DIR}/run.sh" > "${OUTS[$i]}/launcher.log" 2>&1 \
        || echo "FAILED ${CKPTS[$i]} (log: ${OUTS[$i]}/launcher.log)"
    done
  ) &
  pids+=($!)
done

# Each run.sh writes only to its launcher.log, and the training platform restarts a pod
# whose stdout stays silent for ~12 min, so print the latest line of every log periodically.
(
  while sleep "${HEARTBEAT_SECONDS:-120}"; do
    for i in "${!CKPTS[@]}"; do
      last="$(tail -c 400 "${OUTS[$i]}/launcher.log" 2>/dev/null | tr '\r' '\n' | grep -v '^[[:space:]]*$' | tail -n 1)"
      echo "[sweep $(date +%H:%M:%S)] $(basename "${CKPTS[$i]}"): ${last}"
    done
  done
) &
heartbeat=$!
wait "${pids[@]}"
kill "$heartbeat" 2>/dev/null

done_outs=()
for o in "${OUTS[@]}"; do [[ -f "${o}/results.json" ]] && done_outs+=("$o"); done
[[ ${#done_outs[@]} -gt 0 ]] && python "${EVAL_DIR}/collect.py" "${done_outs[@]}"
exit
}
