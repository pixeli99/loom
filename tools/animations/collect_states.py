"""Record the hidden state after every loop, for the README animation.

Runs a checkpoint on held-out documents and stores, per loop, the RMS of the hidden
state and of the loop's update, plus, for a sample of 96 tokens, 2-D PCA trajectories and the cosine
between every pair of that token's states.
Appends one entry to states.json next to this file.

    PYTHONPATH=eval:. python tools/animations/collect_states.py <name> <ckpt> <packed data dir> [n_docs] [seq_len]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MOE_SKIP_CAPACITY", "1")
from lm_eval_loom import _build_fully_causal_ll_batch  # noqa: E402
from loom_loader import load_checkpoint_for_eval, set_param_dtype  # noqa: E402

name, ckpt_path, data_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
n_docs = int(sys.argv[4]) if len(sys.argv) > 4 else 8
seq_len = int(sys.argv[5]) if len(sys.argv) > 5 else 512

ckpt = load_checkpoint_for_eval(ckpt_path)
model = ckpt.model.eval()
set_param_dtype(model, torch.float32)
device = next(model.parameters()).device
core = next(m for m in model.modules() if hasattr(m, "forward_range") and hasattr(m, "num_loops"))
H = int(core.num_loops)

states = []  # per document: [H + 1, tokens, D]
orig = core.forward_range


def per_loop(z, x, t_start, t_end, *, reset_bank=False, **kw):
    path = [z.detach().float().cpu()]
    for t in range(int(t_start), int(t_end) + 1):
        z = orig(z, x, t, t, reset_bank=reset_bank and t == int(t_start), **kw)
        path.append(z.detach().float().cpu())
    states.append(torch.stack(path))
    return z


core.forward_range = per_loop

tokens = np.load(data_dir / "tokens.npy", mmap_mode="r")
starts = np.load(data_dir / "epoch_0" / "doc_start.npy")
lens = np.load(data_dir / "epoch_0" / "doc_len.npy")
docs = [np.asarray(tokens[s : s + seq_len], dtype=np.int32) for s, l in zip(starts, lens) if l >= seq_len][:n_docs]
with torch.no_grad():
    for d in docs:
        model(carry=None, batch=_build_fully_causal_ll_batch([d], device))

Z = torch.cat(states, dim=1)  # [H + 1, tokens, D]; index 0 is the input to loop 1
rms = Z.pow(2).mean(-1).sqrt()                      # [H + 1, tokens]
upd = (Z[1:] - Z[:-1]).pow(2).mean(-1).sqrt()       # [H, tokens]
cos = torch.nn.functional.cosine_similarity(Z[1:], Z[:-1], dim=-1)

rng = np.random.default_rng(0)
pick = rng.choice(Z.shape[1], size=96, replace=False)
P = Z[:, pick]                                      # [H + 1, 96, D]
flat = P.reshape(-1, P.shape[-1])
mu = flat.mean(0, keepdim=True)
_, _, V = torch.pca_lowrank(flat - mu, q=2, center=False)
traj = ((P - mu) @ V[:, :2]).numpy()                # [H + 1, 96, 2]

Pn = torch.nn.functional.normalize(P, dim=-1).transpose(0, 1)   # [96, H + 1, D]
gram = (Pn @ Pn.transpose(1, 2)).numpy()                     # true cosines between a token's states

entry = {
    "name": name,
    "ckpt": str(ckpt.info.get("run_name") or Path(ckpt_path).name),
    "step": ckpt.info.get("step"),
    "num_loops": H,
    "hidden": int(Z.shape[-1]),
    "n_tokens": int(Z.shape[1]),
    "rms_by_loop": rms.mean(1).tolist(),
    "rms_p90_by_loop": rms.quantile(0.9, dim=1).tolist(),
    "update_rms_by_loop": upd.mean(1).tolist(),
    "cos_prev_by_loop": cos.mean(1).tolist(),
    "traj_pca2": traj.round(4).tolist(),
    "gram": gram.round(5).tolist(),
}
out = Path(__file__).with_name("states.json")
data = json.loads(out.read_text()) if out.exists() else {}
data[name] = entry
out.write_text(json.dumps(data))
print(name, "H", H, "rms", np.round(entry["rms_by_loop"], 3).tolist())
print(name, "update", np.round(entry["update_rms_by_loop"], 3).tolist())
print(name, "cos", np.round(entry["cos_prev_by_loop"], 3).tolist())
