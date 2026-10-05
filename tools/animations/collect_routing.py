"""Record which experts each loop's router picks, for the README animation.

Runs a trained checkpoint on held-out documents and stores, for every MoE layer, loop
and token, the routed experts chosen by top-k. Writes routing.json next to this file.

    PYTHONPATH=eval:. python tools/animations/collect_routing.py <ckpt> <packed data dir> [n_docs] [seq_len] [example_doc]

example_doc is the index of the held-out document whose first 64 tokens are drawn (default 0).
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MOE_SKIP_CAPACITY", "1")
from lm_eval_loom import _build_fully_causal_ll_batch  # noqa: E402
from loom_loader import load_checkpoint_for_eval, set_param_dtype  # noqa: E402
from models.moe import MoEFFN  # noqa: E402

ckpt_path, data_dir = sys.argv[1], Path(sys.argv[2])
n_docs = int(sys.argv[3]) if len(sys.argv) > 3 else 32
seq_len = int(sys.argv[4]) if len(sys.argv) > 4 else 512
example_doc = int(sys.argv[5]) if len(sys.argv) > 5 else 0

ckpt = load_checkpoint_for_eval(ckpt_path)
model = ckpt.model.eval()
set_param_dtype(model, torch.float32)
device = next(model.parameters()).device

layers = [m for m in model.modules() if isinstance(m, MoEFFN)]
record = defaultdict(list)  # (layer, loop) -> [tensor [tokens, k]]
for li, moe in enumerate(layers):
    orig = moe._route

    def wrapped(x2d, loop_idx=0, _orig=orig, _li=li):
        scores, idx, unbiased = _orig(x2d, loop_idx=loop_idx)
        record[(_li, int(loop_idx))].append(idx.detach().cpu())
        return scores, idx, unbiased

    moe._route = wrapped

tokens = np.load(data_dir / "tokens.npy", mmap_mode="r")
starts = np.load(data_dir / "epoch_0" / "doc_start.npy")
lens = np.load(data_dir / "epoch_0" / "doc_len.npy")
docs = [np.asarray(tokens[s : s + seq_len], dtype=np.int32) for s, l in zip(starts, lens) if l >= seq_len][:n_docs]
# The drawn example goes first (it may be shorter than seq_len).
ex_s, ex_l = int(starts[example_doc]), int(lens[example_doc])
docs.insert(0, np.asarray(tokens[ex_s : ex_s + min(ex_l, seq_len)], dtype=np.int32))

with torch.no_grad():
    for d in docs:
        model(carry=None, batch=_build_fully_causal_ll_batch([d], device))

n_layers = len(layers)
loops = sorted({lp for _, lp in record})
sel = np.stack([np.stack([torch.cat(record[(li, lp)]).numpy() for lp in loops]) for li in range(n_layers)])
# sel: [layer, loop, token, k]
example_doc = docs[0]
out = {
    "ckpt": str(ckpt.info.get("run_name") or ckpt_path),
    "num_loops": len(loops),
    "n_layers": n_layers,
    "top_k": int(sel.shape[-1]),
    "n_routed": int(layers[0].num_experts),
    "n_tokens": int(sel.shape[2]),
    "example_text": ckpt.tokenizer.decode(example_doc[:64].tolist()),
    "example_tokens": [ckpt.tokenizer.decode([t]) for t in example_doc[:64].tolist()],
    "selections": sel.astype(np.int16).tolist() if sel.size < 3_000_000 else None,
}
# compact summaries
uniq = np.zeros(sel.shape[:1] + sel.shape[2:3])
for li in range(n_layers):
    for t in range(sel.shape[2]):
        uniq[li, t] = len(np.unique(sel[li, :, t, :]))
overlap = np.zeros((n_layers, len(loops) - 1))
for li in range(n_layers):
    for lp in range(1, len(loops)):
        a, b = sel[li, lp], sel[li, lp - 1]
        overlap[li, lp - 1] = np.mean([len(set(x) & set(y)) for x, y in zip(a, b)])
out["distinct_per_token_by_layer"] = uniq.mean(1).tolist()
cum = np.zeros(len(loops))
for lp in range(len(loops)):
    flat = np.sort(sel[:, : lp + 1].transpose(0, 2, 1, 3).reshape(n_layers, sel.shape[2], -1), axis=-1)
    cum[lp] = (1 + (np.diff(flat, axis=-1) != 0).sum(-1)).mean()
out["cum_distinct_by_loop"] = cum.tolist()  # mean over layers and tokens
out["overlap_prev_loop_by_layer"] = overlap.tolist()
out["example_layer_loop_sets"] = sel[:, :, :64, :].astype(int).tolist()
Path(__file__).with_name("routing.json").write_text(json.dumps(out))
print(f"layers={n_layers} loops={len(loops)} k={sel.shape[-1]} tokens={sel.shape[2]}")
print("distinct experts per token over all loops, by layer:", np.round(uniq.mean(1), 2).tolist())
print("cumulative distinct by loop:", np.round(cum, 2).tolist())
print("overlap with previous loop, by layer (mean over loops):", np.round(overlap.mean(1), 2).tolist())
