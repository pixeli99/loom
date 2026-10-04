"""Write a model-only copy of an FSDP2 DCP checkpoint and verify it bit for bit.

Each __R_0.distcp shard holds rank R's slice of both the model and the Adam
state, so the optimizer cannot be removed by deleting files. This reads every
"model.*" tensor (fp32) on CPU without a process group, saves them under the
same "model" prefix the trainer and eval/hrm_loader.py use, reloads the new
copy and compares every tensor with torch.equal.

    python tools/export_model_only_dcp.py <src fsdp2_epoch_N> <dst dir>
"""
import sys
import time

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

src, dst = sys.argv[1], sys.argv[2]
t0 = time.time()
md = dcp.FileSystemReader(src).read_metadata().state_dict_metadata
model_md = {k: v for k, v in md.items() if k.startswith("model.")}
non_tensor = [k for k, v in model_md.items() if not isinstance(v, TensorStorageMetadata)]
if non_tensor:
    sys.exit(f"FATAL: non-tensor model entries, refusing: {non_tensor[:5]}")
state = {k[len("model."):]: torch.empty(v.size, dtype=v.properties.dtype) for k, v in model_md.items()}
dcp.load({"model": state}, checkpoint_id=src, no_dist=True)
dcp.save({"model": state}, checkpoint_id=dst, no_dist=True)

check = {k: torch.empty_like(v) for k, v in state.items()}
dcp.load({"model": check}, checkpoint_id=dst, no_dist=True)
bad = [k for k in state if not torch.equal(state[k], check[k])]
if bad:
    sys.exit(f"FATAL: {len(bad)} tensors differ after round trip: {bad[:5]}")
n = sum(v.numel() for v in state.values())
print(f"OK {len(state)} tensors, {n:,} params, {n * 4 / 2**30:.2f} GiB fp32, bit-exact, {time.time() - t0:.0f}s")
