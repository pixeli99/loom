"""0-shot lm_eval for LOOM checkpoints.

Protocol:
  official lm_eval.simple_evaluate, 0-shot, full-text continuation,
  fully causal SUM log-likelihood, acc / acc_norm.

Before scoring, the run's own eval/loss at the checkpoint step is recomputed (same
holdout batches as pretrain.run_eval) and must match to EVAL_HOLDOUT_MAX_GAP. A gap
there means the weights or code tree are wrong, not that the model is weak.

Launch::

    bash eval/run.sh /path/to/ckpt
"""

from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from datasets_cache import load_dataset, setup_hf_datasets_cache
from distributed_utils import (
    clear_prep_done_flag,
    init_eval_distributed,
    signal_prep_done,
    wait_for_prep_done,
)
from loom_loader import EvalCheckpoint, load_checkpoint_for_eval, provenance, set_param_dtype
from models.common import wrap_tensor
from models.prefixlm_attention import compute_aux_seq_tensors_scalars
from utils.device import get_device, set_device


DEFAULT_TASKS = (
    "openbookqa",
    "winogrande",
    "arc_challenge",
    "arc_easy",
    "hellaswag",
    "social_iqa",
    "piqa",
)


class _DistAccel:
    """Minimal Accelerator stand-in for lm_eval's gather / barrier calls."""

    def __init__(self, rank: int, world_size: int, local_rank: int, device: torch.device):
        self.process_index = rank
        self.local_process_index = local_rank
        self.num_processes = world_size
        self.device = device

    @property
    def is_local_main_process(self) -> bool:
        return self.local_process_index == 0

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.num_processes <= 1:
            return tensor.unsqueeze(0) if tensor.ndim == 0 else tensor
        src = tensor.detach().contiguous()
        gathered = [torch.empty_like(src) for _ in range(self.num_processes)]
        dist.all_gather(gathered, src)
        return torch.stack(gathered)

    def wait_for_everyone(self) -> None:
        if self.num_processes > 1 and dist.is_initialized():
            dist.barrier()


def _build_fully_causal_ll_batch(
    inp_toks: list[np.ndarray],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Pack sequences with prefix_lens=0 so attention is fully causal (lm_eval)."""
    prefix_lens = np.zeros(len(inp_toks), dtype=np.int32)
    causal_lens = np.asarray([int(x.size) for x in inp_toks], dtype=np.int32)
    if causal_lens.size == 0 or int(causal_lens.max(initial=0)) <= 0:
        raise ValueError("empty causal batch")
    n_seq = int(causal_lens.shape[0])
    batch_max_tokens = max(int(causal_lens.max()), n_seq + 1, 1)
    flat_inputs = np.concatenate(inp_toks)
    flat_pos = np.concatenate([np.arange(int(x.size), dtype=np.int32) for x in inp_toks])
    tensors, scalars = compute_aux_seq_tensors_scalars(
        prefix_lens, causal_lens, batch_max_tokens
    )
    batch: dict[str, torch.Tensor] = {
        "inputs": torch.from_numpy(flat_inputs).to(device, non_blocking=True),
        "position_ids": torch.from_numpy(flat_pos).to(device, non_blocking=True),
    }
    for key, value in tensors.items():
        batch[key] = torch.from_numpy(value).to(device, non_blocking=True)
    for key, value in scalars.items():
        batch[key] = wrap_tensor(torch.tensor(value, device="cpu"))
    return batch


@register_model("loom")
class LoomLM(TemplateLM):
    """TemplateLM adapter: official causal SUM log-likelihood over packed PrefixLM=0."""

    def __init__(
        self,
        ckpt: EvalCheckpoint,
        *,
        rank: int,
        world_size: int,
        local_rank: int,
        batch_size: int = 32,
        max_length: int = 1024,
    ):
        super().__init__()
        self._ckpt = ckpt
        self.tokenizer = ckpt.tokenizer
        self._device = get_device(local_rank)
        self._rank = rank
        self._world_size = world_size
        self.batch_size = int(batch_size)
        self._max_length = int(max_length)
        self.accelerator = _DistAccel(rank, world_size, local_rank, self._device)
        self.backend = "causal"

    @property
    def eot_token_id(self) -> int:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            raise ValueError("tokenizer has no eos_token_id")
        return int(eos)

    @property
    def prefix_token_id(self) -> int:
        bos = self.tokenizer.bos_token_id
        return int(bos) if bos is not None else self.eot_token_id

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def device(self) -> torch.device:
        return self._device

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        **kwargs,
    ) -> list[int]:
        # Training packs each document as `text + eos` from position 0, no BOS.
        del add_special_tokens
        return list(self.tokenizer.encode(string, add_special_tokens=False))

    @torch.inference_mode()
    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        **kwargs,
    ) -> list[tuple[float, bool]]:
        results: list[tuple[float, bool]] = [(0.0, False)] * len(requests)
        batch_size = max(int(self.batch_size), 1)
        max_len = self.max_length
        device = self.device
        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm or (self.rank != 0),
            desc="lm_eval loglikelihood",
        )
        for start in range(0, len(requests), batch_size):
            chunk = requests[start : start + batch_size]
            inps: list[np.ndarray] = []
            conts: list[np.ndarray] = []
            valid_idx: list[int] = []
            for i, (_pair, ctx, cont) in enumerate(chunk):
                if not ctx or not cont:
                    continue
                if len(cont) > max_len:
                    continue
                whole = list(ctx) + list(cont)
                inp = whole[-(max_len + 1) :][:-1]
                if not inp:
                    continue
                inps.append(np.asarray(inp, dtype=np.int32))
                conts.append(np.asarray(cont, dtype=np.int32))
                valid_idx.append(i)
            if not inps:
                pbar.update(len(chunk))
                continue
            batch = _build_fully_causal_ll_batch(inps, device)
            _, logits = self._ckpt.model(carry=None, batch=batch)
            cu = batch["cu_seqlens"].tolist()
            for j, i in enumerate(valid_idx):
                cl = int(conts[j].size)
                end = cu[j] + int(inps[j].size)
                rows = logits[end - cl : end].float()
                cont = torch.from_numpy(conts[j]).to(device=device, dtype=torch.long)
                logp = torch.log_softmax(rows, dim=-1).gather(-1, cont[:, None]).squeeze(-1)
                greedy_ok = bool((rows.argmax(dim=-1) == cont).all().item())
                results[start + i] = (float(logp.double().sum().item()), greedy_ok)
            pbar.update(len(chunk))
        pbar.close()
        return results

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        raise NotImplementedError("LoomLM only implements multiple-choice loglikelihood")

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        raise NotImplementedError("LoomLM only implements multiple-choice loglikelihood")


_task_dir_env = os.environ.get("LMEVAL_TASK_DIR", "").strip()
_LOCAL_TASK_DIR = (
    Path(_task_dir_env) if _task_dir_env else Path(__file__).resolve().parent / "tasks"
)


def _parse_tasks() -> list[str]:
    raw = os.environ.get("LMEVAL_TASKS", ",".join(DEFAULT_TASKS)).strip()
    return [t.strip() for t in raw.split(",") if t.strip()]


_DATASET_WARMUP = (
    ("allenai/openbookqa", "main", "train"),
    ("allenai/openbookqa", "main", "test"),
    ("allenai/winogrande", "winogrande_debiased", "train"),
    ("allenai/winogrande", "winogrande_debiased", "validation"),
    ("allenai/ai2_arc", "ARC-Challenge", "train"),
    ("allenai/ai2_arc", "ARC-Challenge", "test"),
    ("allenai/ai2_arc", "ARC-Easy", "train"),
    ("allenai/ai2_arc", "ARC-Easy", "test"),
    ("Rowan/hellaswag", None, "train"),
    ("Rowan/hellaswag", None, "validation"),
    ("jet-ai/social_i_qa", None, "train"),
    ("jet-ai/social_i_qa", None, "validation"),
    ("baber/piqa", None, "train"),
    ("baber/piqa", None, "validation"),
)


def _warmup_hf_datasets(tasks: list[str], task_manager, num_fewshot: int | None) -> None:
    """Materialize raw + process_docs arrow caches on rank 0 (avoid N-way races on a shared filesystem)."""
    for path, name, split in _DATASET_WARMUP:
        print(f"[lm_eval_loom] warmup {path} {name} {split}", flush=True)
        kwargs = {"split": split, "trust_remote_code": True}
        ds = load_dataset(path, name, **kwargs) if name else load_dataset(path, **kwargs)
        print(f"[lm_eval_loom] warmup ok n={len(ds)}", flush=True)
    from lm_eval.tasks import get_task_dict

    shots = 0 if num_fewshot is None else int(num_fewshot)
    task_dict = get_task_dict(tasks, task_manager)
    for name, task in task_dict.items():
        if shots:
            task.set_config(key="num_fewshot", value=shots)
        print(f"[lm_eval_loom] warmup build_all_requests {name} shots={shots}", flush=True)
        task.build_all_requests(limit=None, rank=0, world_size=1)
        print(f"[lm_eval_loom] warmup task {name} n={len(task.instances)}", flush=True)


_LOCAL_DS_CACHE = Path(os.environ.get("EVAL_LOCAL_DS_CACHE", "/tmp/loom_lmeval_hf_datasets"))
_SHARED_DS_NAMES = (
    "allenai___openbookqa",
    "allenai___winogrande",
    "allenai___ai2_arc",
    "Rowan___hellaswag",
    "jet-ai___social_i_qa",
    "baber___piqa",
)


def _stage_local_dataset_cache(local_rank: int) -> None:
    """Copy warmed arrow caches onto node-local disk so local ranks do not write the shared filesystem."""
    src_root = Path(os.environ.get("HF_DATASETS_CACHE", str(_LOCAL_DS_CACHE)))
    dest = _LOCAL_DS_CACHE
    ready = dest / ".ready"
    if local_rank == 0:
        dest.mkdir(parents=True, exist_ok=True)
        for name in _SHARED_DS_NAMES:
            src = src_root / name
            dst = dest / name
            if src.is_dir() and src.resolve() != dst.resolve() and not dst.exists():
                print(f"[lm_eval_loom] stage {name} -> {dst}", flush=True)
                # Copy then rename: a concurrent eval on the same node never sees a half copy.
                tmp = dest / f".{name}.{os.getpid()}"
                shutil.copytree(src, tmp)
                try:
                    os.rename(tmp, dst)
                except OSError:
                    shutil.rmtree(tmp, ignore_errors=True)
        ready.touch()
    else:
        while not ready.exists():
            time.sleep(0.5)
    os.environ["HF_DATASETS_CACHE"] = str(dest)
    import datasets.config as ds_config

    ds_config.HF_DATASETS_CACHE = dest
    print(f"[lm_eval_loom] HF_DATASETS_CACHE={dest}", flush=True)


def _parse_num_fewshot() -> int | None:
    raw = os.environ.get("EVAL_NUM_FEWSHOT", "0").strip()
    if raw in {"", "auto", "none", "None"}:
        return None
    return int(raw)


def _train_eval_loss_at(ckpt: EvalCheckpoint) -> float | None:
    """The run's own eval/loss at the checkpoint step, from tensorboard_scalars.json
    (in the ckpt dir, its config dir, or the run a snapshot / archive was taken from)."""
    step = ckpt.info.get("step")
    roots = [Path(ckpt.info["ckpt_path"]), Path(ckpt.info["config_dir"])]
    for meta, key in (("snapshot_meta.json", "source_checkpoint_path"), ("final_meta.json", "source")):
        meta_path = roots[0] / meta
        if meta_path.is_file():
            src = json.loads(meta_path.read_text()).get(key)
            if src:
                roots.append(Path(src))
    for root in roots:
        path = root / "tensorboard_scalars.json"
        if step is None or not path.is_file():
            continue
        points = json.loads(path.read_text()).get("eval/loss") or []
        hits = [v for s, v in points if int(s) == int(step)]
        if hits:
            return float(hits[-1])
    return None


def _train_world_size(weights_dir: str) -> int:
    """Ranks that wrote the checkpoint (one __<rank>_0.distcp each) = the run's eval sharding.
    A model-only export is rewritten as one file, so its meta records the source rank count."""
    for meta in ("snapshot_meta.json", "final_meta.json"):
        meta_path = Path(weights_dir).parent / meta
        if meta_path.is_file():
            n = json.loads(meta_path.read_text()).get("source_world_size")
            if n:
                return int(n)
    n = len(list(Path(weights_dir).glob("__*_0.distcp")))
    if n <= 0:
        raise FileNotFoundError(f"no __*_0.distcp under {weights_dir}")
    return n


@contextmanager
def _training_numerics(ckpt: EvalCheckpoint):
    """Run as pretrain.run_eval does: fwd_bwd_dtype parameters and MoE capacity dropping on.
    The eval-time dtype and capacity setting are restored afterwards (fp32 masters are kept,
    not re-derived from bf16)."""
    fwd_dtype = getattr(torch, ckpt.config.fwd_bwd_dtype)
    params = list(ckpt.model.parameters())
    saved = [p.data for p in params] if any(p.dtype != fwd_dtype for p in params) else None
    saved_env = os.environ.get("MOE_SKIP_CAPACITY")
    os.environ["MOE_SKIP_CAPACITY"] = "0"
    if saved is not None:
        set_param_dtype(ckpt.model, fwd_dtype)
    try:
        yield
    finally:
        if saved is not None:
            for p, data in zip(params, saved):
                p.data = data
        if saved_env is None:
            os.environ.pop("MOE_SKIP_CAPACITY", None)
        else:
            os.environ["MOE_SKIP_CAPACITY"] = saved_env


@torch.inference_mode()
def _holdout_ce(ckpt: EvalCheckpoint, device: torch.device, rank: int, world_size: int) -> dict | None:
    """Recompute the run's own eval/loss: same holdout split, same per-rank multipack
    batches (pads included, they share MoE capacity), same token budget and stop rule
    as pretrain.run_eval, under the same numerics. Matching it to ~1e-4 pins weights +
    tokenizer + code tree."""
    if os.environ.get("EVAL_HOLDOUT_CHECK", "1") in {"0", "false", "False"}:
        return None
    from models.common import IGNORE_LABEL_ID
    from pretrain import _make_dataset

    cfg = ckpt.config
    train_world = int(os.environ.get("EVAL_HOLDOUT_TRAIN_WORLD", "0")) or _train_world_size(ckpt.info["weights"])
    val_path = cfg.data.val_path or cfg.data.path
    epoch = 0 if cfg.data.val_path is not None else int(cfg.eval_epoch)
    batch_max_length = int(cfg.micro_batch_samples) * int(ckpt.info["max_seq_len"])
    mine = list(range(rank, train_world, world_size))
    iters = {
        r: iter(_make_dataset(
            cfg, dataset_path=val_path, batch_max_length=batch_max_length, drop_last_batch=False,
            rank=r, world_size=train_world, fixed_epoch=epoch,
        ))
        for r in mine
    }
    stats = torch.zeros(2, device=device, dtype=torch.float64)
    steps = 0
    while True:
        for r in mine:
            batch, scalars = next(iters[r])
            labels = batch.pop("labels").to(device, torch.long)
            model_batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            for k, v in scalars.items():
                if not str(k).startswith("_resume_"):
                    model_batch[k] = wrap_tensor(torch.tensor(v, device="cpu"))
            _, logits = ckpt.model(carry=None, batch=model_batch)
            stats[0] += F.cross_entropy(
                logits.float(), labels, ignore_index=IGNORE_LABEL_ID, reduction="sum"
            ).double()
            stats[1] += (labels != IGNORE_LABEL_ID).sum()
        steps += 1
        seen = stats[1].clone()
        if world_size > 1 and dist.is_initialized():
            dist.all_reduce(seen)
        if seen.item() >= cfg.eval_token_budget:
            break
    if world_size > 1 and dist.is_initialized():
        dist.all_reduce(stats)
    ce = float(stats[0] / stats[1])
    ref = _train_eval_loss_at(ckpt)
    return {
        "source": f"{val_path}/epoch_{epoch}",
        "train_world": train_world,
        "steps": steps,
        "tokens": int(stats[1]),
        "ce": ce,
        "train_eval_loss_at_step": ref,
        "gap": None if ref is None else ce - ref,
    }


def _write_outputs(
    results: dict,
    out_dir: Path,
    ckpt: EvalCheckpoint,
    tasks: list[str],
    num_fewshot: int | None,
    extra: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in results.items() if k != "samples"}
    slim["ckpt_path"] = ckpt.info["ckpt_path"]
    slim["checkpoint"] = ckpt.info
    slim["harness"] = "lm_eval.simple_evaluate"
    slim["protocol"] = "official_0shot_causal_sum_ll"
    slim["tasks"] = tasks
    slim["num_fewshot"] = num_fewshot
    slim.update(extra)
    (out_dir / "results.json").write_text(
        json.dumps(slim, indent=2, default=str),
        encoding="utf-8",
    )
    for task, rows in (results.get("samples") or {}).items():
        with open(out_dir / f"samples_{task}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                keep = {k: row.get(k) for k in ("doc_id", "target", "acc", "acc_norm")}
                keep["ll"] = [float(r[0]) for r in row.get("filtered_resps", [])]
                f.write(json.dumps(keep, default=str) + "\n")
    try:
        from lm_eval.utils import make_table

        table = make_table(results)
    except Exception as exc:  # noqa: BLE001
        table = f"(make_table failed: {exc})\n"
    prov = extra["provenance"]
    hold = extra.get("holdout")
    info = ckpt.info
    lines = [
        f"ckpt={info['ckpt_path']}  step={info['step']}  H={info['num_loops']}",
        f"code={prov['code_root']}@{prov['commit'][:12]}  dirty={prov['model_code_dirty']}",
        f"tokenizer={info['tokenizer_path']}",
        f"harness=lm_eval.simple_evaluate  scoring=causal_sum_ll  shots={num_fewshot}  "
        f"bs={extra['batch_size']}  world={extra['world_size']}  "
        f"param_dtype={info['param_dtype']}  moe_skip_capacity={prov['env'].get('MOE_SKIP_CAPACITY')}",
        "holdout: skipped" if hold is None else (
            f"holdout ({info['fwd_dtype']}, capacity on) CE={hold['ce']:.4f} on {hold['tokens']} tok  "
            f"train eval/loss@step={hold['train_eval_loss_at_step']}  gap={hold['gap']}"
        ),
        f"tasks={','.join(tasks)}",
        "",
        table,
        "",
    ]
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main() -> None:
    setup_hf_datasets_cache()
    rank, world_size, is_main = init_eval_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    set_device(local_rank)
    device = get_device(local_rank)
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    ckpt_path = os.environ.get("CKPT_PATH", "").strip()
    if not ckpt_path:
        raise SystemExit("CKPT_PATH is required")
    out_dir = Path(os.environ.get("EVAL_OUTPUT_DIR", str(Path(ckpt_path) / "eval_results" / "official_lmeval")))
    batch_size = int(os.environ.get("EVAL_BATCH_SIZE_MCQ", "32"))
    num_fewshot = _parse_num_fewshot()
    limit_raw = os.environ.get("LMEVAL_LIMIT", "").strip()
    limit = float(limit_raw) if limit_raw else None
    if limit is not None and limit >= 1:
        limit = int(limit)
    log_samples = os.environ.get("LMEVAL_LOG_SAMPLES", "0") not in {"0", "false", "False"}
    bootstrap = int(os.environ.get("LMEVAL_BOOTSTRAP_ITERS", "1000"))
    max_length = int(os.environ.get("LMEVAL_MAX_LENGTH", "1024"))
    tasks = _parse_tasks()

    from lm_eval.evaluator import simple_evaluate
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(
        verbosity="INFO" if is_main else "ERROR",
        include_path=str(_LOCAL_TASK_DIR),
    )

    if is_main:
        print(
            f"[lm_eval_loom] rank0/{world_size} ckpt={ckpt_path} tasks={tasks} "
            f"bs={batch_size} shots={num_fewshot} task_dir={_LOCAL_TASK_DIR} out={out_dir}",
            flush=True,
        )
        clear_prep_done_flag()
        _warmup_hf_datasets(tasks, task_manager, num_fewshot)
        signal_prep_done()
    else:
        print(f"[lm_eval_loom] rank {rank}: waiting for dataset warmup", flush=True)
        wait_for_prep_done(rank, world_size)
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
    _stage_local_dataset_cache(local_rank)
    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    ckpt_epoch = os.environ.get("CKPT_EPOCH", "").strip()
    ckpt = load_checkpoint_for_eval(ckpt_path, int(ckpt_epoch) if ckpt_epoch else None)
    prov = provenance(ckpt.config.arch.name)
    if is_main:
        print(f"[lm_eval_loom] checkpoint {json.dumps(ckpt.info, default=str)}", flush=True)
        print(f"[lm_eval_loom] provenance {json.dumps(prov)}", flush=True)

    t0 = time.time()
    with _training_numerics(ckpt):
        holdout = _holdout_ce(ckpt, device, rank, world_size)
    if is_main and holdout is not None:
        print(f"[lm_eval_loom] holdout {json.dumps(holdout)} ({time.time() - t0:.0f}s)", flush=True)
        max_gap = float(os.environ.get("EVAL_HOLDOUT_MAX_GAP", "0.005"))
        if holdout["gap"] is None:
            # A live checkpoint saved between eval steps has no reference; EVAL_HOLDOUT_CHECK=record
            # keeps the recomputed CE in results.json so it can be read against neighbouring steps.
            if os.environ.get("EVAL_HOLDOUT_CHECK", "1") != "record":
                raise SystemExit(
                    f"no eval/loss at step {ckpt.info.get('step')} in any tensorboard_scalars.json; "
                    "nothing to check the load against (EVAL_HOLDOUT_CHECK=0 skips the check, "
                    "EVAL_HOLDOUT_CHECK=record keeps the CE without checking)"
                )
            print(f"[lm_eval_loom] holdout: no eval/loss at step {ckpt.info.get('step')}, CE recorded unchecked", flush=True)
        elif abs(holdout["gap"]) > max_gap:
            raise SystemExit(
                f"holdout CE {holdout['ce']:.4f} is {holdout['gap']:+.4f} off the run's own "
                f"eval/loss; weights, tokenizer or code tree do not match (EVAL_HOLDOUT_MAX_GAP={max_gap})"
            )

    lm = LoomLM(
        ckpt,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        batch_size=batch_size,
        max_length=max_length,
    )

    if world_size > 1 and dist.is_initialized():
        dist.barrier()

    t1 = time.time()
    results = simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        log_samples=log_samples,
        bootstrap_iters=bootstrap,
        fewshot_random_seed=int(os.environ.get("LMEVAL_FEWSHOT_SEED", "1234")),
        verbosity="INFO" if is_main else "ERROR",
        task_manager=task_manager,
    )

    if is_main:
        if results is None:
            raise SystemExit("lm_eval.simple_evaluate returned None on rank 0")
        extra = {
            "provenance": prov,
            "holdout": holdout,
            "batch_size": batch_size,
            "world_size": world_size,
            "limit": limit,
            "seconds": {"holdout": t1 - t0, "lm_eval": time.time() - t1},
        }
        _write_outputs(results, out_dir, ckpt, tasks, num_fewshot, extra)

    if world_size > 1 and dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
