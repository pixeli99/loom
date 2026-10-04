# Zero-shot evaluation

`lm_eval.simple_evaluate` (lm-eval 0.4.11), 0-shot, fully causal sum log-likelihood over the continuation.
Tasks: ARC-Easy, ARC-Challenge, HellaSwag, OpenBookQA, PIQA, Winogrande, SIQA.
Metrics: `acc_norm` for ARC / HellaSwag / OpenBookQA / PIQA, `acc` for Winogrande / SIQA.

## Evaluate one checkpoint

```bash
bash eval/run.sh checkpoints/<project>/<run>                  # run directory
bash eval/run.sh checkpoints/<project>/<run>/snapshots/step_0010000
```

`CKPT_PATH` (or the first argument) may be a run directory, a snapshot under `<run>/snapshots/`,
or an archive directory holding `fsdp2_epoch_N` plus a `final_meta.json` whose `source` names the run directory.
Results go to `<ckpt>/eval_results/official_lmeval7_<timestamp>/`; for a snapshot they go to
`<run>/eval_results/step_NNNNNNN/...`, because snapshots are pruned by `SNAPSHOT_KEEP_LAST`.
Each result directory has `results.json` (scores, checkpoint info, code provenance, holdout check), `summary.txt` and `eval.log`.

Datasets are downloaded on first use into `eval/hf_cache` (`EVAL_HF_ROOT` to move it, `EVAL_OFFLINE=1` to forbid network access afterwards).

## Several checkpoints, one GPU each

```bash
bash eval/sweep.sh ckpt_H3 ckpt_H6 ckpt_H9 ckpt_H12     # prints the table when all finish
python eval/collect.py ckpt_H3 ckpt_H6                  # re-print from the newest results
```

## Holdout check

Before scoring, the checkpoint's own `eval/loss` is recomputed exactly as `pretrain.py` computed it during training
(same holdout batches, same per-rank packing, 1M-token budget, bf16 parameters, MoE capacity on) and compared with the value
logged at that step in `tensorboard_scalars.json`. A gap above `EVAL_HOLDOUT_MAX_GAP` (default 0.005) aborts the run:
it means the weights, tokenizer or code tree do not match, not that the model is weak.

- The check needs the training data (`data.path` / `data.val_path` in the run's `all_config.yaml`). Use `EVAL_HOLDOUT_CHECK=0` to skip it.
- For checkpoints saved between two evals there is no logged value; `EVAL_HOLDOUT_CHECK=record` recomputes and stores the loss without comparing.
- The per-rank packing depends on the training world size. It is read from the number of `__*_0.distcp` files, or from `source_world_size`
  in `snapshot_meta.json` / `final_meta.json` for single-file model-only exports (`tools/export_model_only_dcp.py`), or from `EVAL_HOLDOUT_TRAIN_WORLD`.

## Scoring numerics

- MoE capacity is off while scoring (`MOE_SKIP_CAPACITY=1`). With capacity on, capacity is computed over the whole packed batch,
  so a question's score depends on which other questions share its batch.
- Parameters are fp32 while scoring (`EVAL_PARAM_DTYPE=fp32`). In bf16, the expert outputs are combined in bf16 and rounding flips
  the expert choice of individual tokens with batch size, which flips a few answers per task. fp32 runs through SDPA and is about 35% slower.
- With both settings, scores do not depend on `EVAL_BATCH_SIZE_MCQ` (default 32).

## Code tree

Model code is imported only through `PYTHONPATH`: `run.sh` sets `PYTHONPATH=eval:$MODEL_CODE_ROOT`, and `MODEL_CODE_ROOT`
defaults to the repository root. `results.json` records the files actually imported, the commit and whether
`models/ utils/ pretrain.py` had uncommitted changes. If the imported files are not under `MODEL_CODE_ROOT`, the run aborts.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `CKPT_PATH` | required (or first argument) | checkpoint directory |
| `CKPT_EPOCH` | largest `fsdp2_epoch_*` | epoch to load |
| `CKPT_CONFIG_DIR` | auto | directory with `all_config.yaml` and `train_metadata.yaml` |
| `MODEL_CODE_ROOT` | repository root | model code tree |
| `EVAL_OUTPUT_DIR` | see above | result directory |
| `NPROC_PER_NODE` | visible GPUs | GPUs per checkpoint (requests are split across them) |
| `EVAL_BATCH_SIZE_MCQ` | 32 | requests per forward pass |
| `EVAL_PARAM_DTYPE` | fp32 | parameter dtype while scoring (`bf16` / `fp32`) |
| `MOE_SKIP_CAPACITY` | 1 | disable MoE capacity while scoring |
| `EVAL_HOLDOUT_CHECK` | 1 | `0` skips the holdout check, `record` stores without comparing |
| `EVAL_HOLDOUT_MAX_GAP` | 0.005 | holdout tolerance |
| `LMEVAL_TASKS` | the 7 tasks | comma-separated |
| `LMEVAL_LIMIT` | empty (all) | first N examples per task, for smoke tests |
| `LMEVAL_LOG_SAMPLES` | 0 | `1` writes `samples_<task>.jsonl` |
| `EVAL_TOKENIZER_PATH` | from `train_metadata.yaml` | tokenizer override |
| `EVAL_HF_ROOT` | `eval/hf_cache` | HuggingFace cache for datasets and the tokenizer |
| `EVAL_OFFLINE` | 0 | `1` uses the cache only |

## Files

```text
eval/
  run.sh            evaluate one checkpoint (single-node torchrun)
  sweep.sh          several checkpoints, one GPU each
  collect.py        result table
  lm_eval_loom.py   lm_eval model adapter, holdout check, result writing
  loom_loader.py    load a checkpoint from DCP, record code provenance
  datasets_cache.py / distributed_utils.py
  tasks/            Winogrande (winogrande_debiased, whose validation split is the standard 1,267 examples)
                    and SIQA (jet-ai/social_i_qa, same schema and 1,954 validation examples) task definitions
```
