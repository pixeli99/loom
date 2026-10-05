# LOOM: Looping Beyond Twice

Code for **Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts**
(Di He, Pengxiang Li, Da Chang, Qingyan Meng, Lu Yin, Shiwei Liu). [arXiv:2610.01153](https://arxiv.org/abs/2610.01153)

A looped Transformer applies the same blocks several times, which adds depth without adding parameters.
In MoE language models, the gain from extra loops usually stops after about two loops.
LOOM is a training recipe for looped MoE models: each loop should add new computation, and the recurrent state should stay stable.
With LOOM, models from 100M to 1.7B parameters train stably with 9 to 12 loops.

<p align="center">
  <img src="assets/weave.gif" width="100%" alt="Nine loops weave through 30 experts; one token uses 14 distinct experts, and the first 48 tokens of the sentence form a cloth.">
</p>

Each loop uses different experts. The 30 vertical threads are the routed experts of layer 8 in the 1.7B LOOM model with 9 loops, and each row is one loop.
Where the loop's router picks an expert, the row passes over that thread (coloured); everywhere else, it passes under.
Over 9 loops, the token "guitar" uses 14 distinct experts, where loops that reused the same experts would use 6.
Averaged over 16,384 held-out tokens and all 15 layers, the count is 14.6, and two consecutive loops share about 3 of their 6 experts.
At the end, the view pulls back to the same weave for the first 48 tokens of the sentence.
The routing is measured by `tools/animations/collect_routing.py`, and `tools/animations/weave.py` draws the animation.

| Component | What it does | Config |
|---|---|---|
| Residual scaling | scales each loop's residual update so hidden-state variance stays bounded as loops are added | `arch.cycle_residual_scale`, `arch.cycle_scale_lambda` |
| Embedding re-injection | mixes the input embedding back into the state at the start of every loop | `arch.embed_inject_mode=loop_start_mix_soft`, `arch.embed_inject_*` |
| Per-loop routers | a separate router for each loop, so different loops can use different experts | `arch.moe.router_per_loop`, `arch.moe.router_num_loops` |
| Looping Residual | carries earlier loops' outputs forward into later loops | `arch.attn_res=true`, `arch.attn_res_scope=dual_axis`, `arch.dual_axis_*` |
| MoE output RMSNorm | RMSNorm on the MoE branch output before the residual add | `arch.ffn_branch_rms=affine` |
| Segmented backprop | backward and optimizer step every K loops (K = 3) | `arch.bp_segment_len` |

All of these are set in `scripts/train.sh`.


## Install

```bash
# Python 3.12, CUDA 12.x. Install torch and flash-attn for your CUDA version first.
pip install -r requirements.txt
```

## Data

Training reads a pre-tokenized, document-packed corpus. To build one from FineWeb-Edu with the SmolLM tokenizer (vocabulary 49,152):

```bash
python tools/pack_document_data.py \
    --parquet 'fineweb-edu/sample/10BT/*.parquet' \
    --tokenizer HuggingFaceTB/SmolLM-360M \
    --out data/fineweb-edu-1024 \
    --seq-len 1024 --epochs 9 --holdout-tokens 30000000 --workers 32
```

`epoch_0` to `epoch_8` are different shuffles of the training documents, and `epoch_9` is a held-out set of documents that never appear in training.
`pretrain.py` evaluates on `epoch_9`, or on `epoch_0` of a separate pack given as `data.val_path` (`VAL_PATH`).
`--jsonl` reads `.jsonl` / `.jsonl.zst` shards (for example Dolma), and `tools/merge_document_packs.py` merges packs that use the same tokenizer.

## Train

`H` is the number of loops. The two presets reproduce the paper's two scales:

```bash
# 329M total parameters: L=10, hidden 512, 30 routed + 2 shared experts, top-8; 10k steps of 1M tokens
H=9 DATA_PATH=data/fineweb-edu-1024 bash scripts/loom_329m.sh

# 1.7B total parameters: L=15, hidden 1280, same expert layout; 57k steps of 1M tokens (about 60B tokens)
H=9 DATA_PATH=data/<corpus>-1024 VAL_PATH=data/<val>-1024 bash scripts/loom_1p7b.sh
```

- The global batch is 1024 sequences of 1024 tokens. The world size must divide 1024, and samples per rank must be a multiple of `MICRO_BATCH_SAMPLES`.
- MoE capacity is computed per micro-batch, so the micro-batch size changes the computation. The presets use the paper values (16 for 329M, 8 for 1.7B).
- Multi-node: run the same command on every node with `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT` set.
- Quick test on one GPU: `H=3 DATA_PATH=... TOTAL_STEPS=20 GLOBAL_BATCH_SIZE=16 MICRO_BATCH_SAMPLES=8 bash scripts/loom_329m.sh`.

Ablations turn off one component of the recipe:

| Ablation | Setting |
|---|---|
| no residual scaling | `CYCLE_RESIDUAL_SCALE=false` |
| no embedding re-injection | `EMBED_INJECT_MODE=none` |
| no MoE output RMSNorm | `FFN_BRANCH_RMS=none` |
| no segmented backprop (one backward through all loops) | `BP_K=0` |

### Checkpoints

Runs are written to `checkpoints/<project>/<run>/` (`CHECKPOINT_PATH` to change).

- `fsdp2_epoch_N/` is the live checkpoint. It is overwritten every `CKPT_EVERY` steps (default 500).
- `snapshots/step_NNNNNNN/` are full copies, saved every `SNAPSHOT_EVERY_STEPS` steps. Only the newest `SNAPSHOT_KEEP_LAST` are kept.
- To continue a run, set `RESUME_FROM=<run dir or snapshot dir>`.
- Each checkpoint stores model and Adam state together in torch DCP format. `tools/export_model_only_dcp.py` writes a model-only copy, about one third of the size, and checks it bit for bit.

Training logs go to TensorBoard under the run directory, and every scalar is also saved to `tensorboard_scalars.json`.

## Evaluate

```bash
bash eval/run.sh checkpoints/<project>/<run>
```

This runs the seven 0-shot tasks from the paper (ARC-E, ARC-C, HellaSwag, OpenBookQA, PIQA, Winogrande, SIQA) with lm-eval 0.4.11.
Before scoring, it checks that the loaded weights reproduce the `eval/loss` logged during training.
See [eval/README.md](eval/README.md).

## Layout

```text
pretrain.py                     training loop (FSDP2, WSD schedule, segmented backprop, spike guard)
models/baselines/looped_transformer.py   looped MoE Transformer
models/moe.py                   MoE layer (shared + routed experts, sigmoid router, per-loop routers)
models/dual_axis_carry.py       Looping Residual
models/attn_res.py              Attention Residuals (arXiv:2603.15031); attn_res_scope=dual_axis uses dual_axis_carry.py instead
config/                         Hydra configs (arch/size: T_moe_L10 = 329M, T_moe_L15_E32 = 1.7B)
dataset_document.py             document-packed dataset
scripts/                        launchers
tools/                          data packing, model-only export
eval/                           0-shot evaluation
```

## Citation

```bibtex
@article{he2026loom,
  title   = {Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts},
  author  = {He, Di and Li, Pengxiang and Chang, Da and Meng, Qingyan and Yin, Lu and Liu, Shiwei},
  journal = {arXiv preprint arXiv:2610.01153},
  year    = {2026}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
