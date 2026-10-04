from typing import Literal, Optional
from dataclasses import dataclass
from pathlib import Path
from glob import glob
import math
import os
import time
import json
import yaml
import shutil
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import tqdm
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf

from models.layers import Carry
from models.common import wrap_tensor
from models.transformer import TransformerBlock
from models.adam_atan2 import AdamATan2
from utils.functions import load_model_class, get_model_source_path
from utils.tensorboard_logger import TensorBoardLogger
from utils.module_diag import ModuleDiagCollector
from utils.device import empty_cache, get_device, get_dist_backend, set_device, synchronize, use_pin_memory
from dataset_new import V1DatasetMeta  # metadata schema shared with DocumentDataset
from dataset_document import DocumentDataset, DocumentDatasetConfig


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    head: str


class DataConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    path: str
    val_path: Optional[str] = None
    # Full-sequence CE for document Causal LM (QA / PrefixLM removed).
    target_only: bool = False
    # document only — Causal LM packing (doc_start/doc_len). QA inst_*/resp_* removed.
    format: Literal["document"] = "document"


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    data: DataConfig

    # Language-modeling objective: full causal attention + full-sequence CE only.
    # PrefixLM / QA (prefix_target|prefix_full) removed.
    lm_mode: Literal["causal"] = "causal"

    # Hyperparams (all batch sizes are in samples)
    global_batch_size: int  # global samples per optimizer step
    micro_batch_samples: int  # samples per NPU per micro forward
    gradient_accumulation_steps: Optional[int] = None  # auto if None
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    lr_warmup_ratio: Optional[float] = None
    # LR schedule: "cosine" (warmup + cosine to lr_min_ratio) or "wsd"
    # (Warmup–Stable–Decay: warmup → flat → linear decay over last lr_decay_ratio).
    lr_schedule: str = "cosine"
    lr_decay_ratio: Optional[float] = None  # WSD only; fraction of steps for final decay
    lr_decay_steps: int = 0  # filled from lr_decay_ratio in init_train

    weight_decay: float
    beta1: float
    beta2: float
    # Optional separate peak LR / WD for learnable cycle λ (θ) and AttnRes β.
    # null → fall back to global lr / weight_decay (legacy single-group behavior).
    cycle_lambda_lr: Optional[float] = None
    cycle_lambda_weight_decay: Optional[float] = None
    # If true: divide λ θ-grads by (∂λ/∂θ)² before optim so Δλ≈−lr·∂L/∂λ (pack-invariant).
    cycle_lambda_grad_in_lambda_space: bool = True
    # Optimizer for λ: adamw (main group) | adamw_sep (independent AdamW) | sgd | sign_sgd.
    cycle_lambda_optim: str = "sgd"
    dual_axis_beta_lr: Optional[float] = None
    dual_axis_beta_weight_decay: Optional[float] = None
    optimizer: str = "adamw"
    ema: Optional[float] = None
    # Global L2 grad clip before optim.step (FSDP2/DTensor-safe via clip_grad_norm_).
    # <=0 or null → disabled.
    clip_grad_norm: Optional[float] = 1.0
    # Skip optim.step on non-finite loss/grad or sudden loss jump (onset spike guard).
    skip_bad_step: bool = True
    # Adaptive threshold: thr = max(skip_loss_delta, skip_loss_rel * baseline).
    # skip_loss_delta is a *floor* (not a fixed trigger) so early high-loss phases
    # need larger absolute jumps (e.g. loss≈6 → thr≈0.6 with rel=0.1).
    skip_loss_delta: float = 0.20
    skip_loss_rel: float = 0.06
    # Also skip if loss >= baseline * (1 + ratio). Catches 3.26–3.28 → 3.6
    # even when the absolute Δ is only ~0.34 (rel×EMA floor can miss it).
    skip_loss_ratio: float = 0.09
    # Always skip on huge absolute jumps (e.g. 3.9→8.2), independent of rel.
    skip_loss_hard: float = 1.5
    # EMA baseline for Δloss (0=use last good loss only). Higher = smoother.
    skip_loss_ema: float = 0.9
    # For the first N optimizer steps, only nonfinite guards (no Δloss skip).
    skip_delta_warmup_steps: int = 0
    # If true: after recording a SPIKE, abort training (alarm file + non-zero exit).
    skip_abort_on_spike: bool = False
    # Abort after this many consecutive *global steps* that each skipped at least
    # one K-seg. 0 = off. Isolated skips do not trip this (counter resets on a clean step).
    skip_abort_consecutive: int = 0
    sample_multiplier: Optional[float] = None  # total_samples = num_params * sample_multiplier / max_seq_len
    total_steps: int = -1  # -1: derive from sample_multiplier/epochs; >1: fixed optimizer steps
    fwd_bwd_dtype: str = "bfloat16"

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Resume / fine-tune from checkpoint
    resume_from: Optional[str] = None
    resume_epoch: Optional[int] = None
    weights_only_resume_from_ema: bool = False  # Swap EMA into model + reset optim

    # Extras
    seed: int = 0
    checkpoint_interval: int = 1
    # Save live ckpt + immutable snapshots/step_* every N optimizer steps (0=off).
    checkpoint_every_steps: int = 0
    log_interval: int = 1
    tensorboard_plot_interval: int = 100
    eval_interval: int = 500
    eval_token_budget: int = 1_000_000
    eval_epoch: int = 9  # epoch_{eval_epoch} indices used when data.val_path is null
    # Mid-train downstream (GSM8k/MATH/...) eval: save + pause so launcher can run
    # 8-NPU eval then resume. 0 = disabled. Distinct from eval_interval (val loss).
    downstream_eval_interval: int = 0


@dataclass
class BatchPlan:
    micro_batch_samples: int
    grad_accum_steps: int
    samples_per_npu_per_step: int
    samples_per_global_step: int
    batch_max_length: int  # dataloader token cap = micro_batch_samples * max_seq_len


@dataclass
class TrainState:
    model: nn.Module
    carry: Optional[Carry]
    
    optim: Optimizer

    step: int
    total_steps: int
    total_samples: int
    num_params: int
    batch_plan: BatchPlan
    # Optional separate SGD for cycle λ (pack-invariant Euclidean steps in λ-space).
    optim_lambda: Optional[Optimizer] = None


def round_steps_to_thousands(steps: float) -> int:
    rounded = int(round(steps / 1000) * 1000)
    if rounded == 0:
        return max(1, int(round(steps)))
    return rounded


def resolve_batch_plan(config: PretrainConfig, world_size: int, max_seq_len: int) -> BatchPlan:
    assert config.global_batch_size % world_size == 0, (
        f"global_batch_size (samples) {config.global_batch_size} must be divisible by world_size {world_size}."
    )
    samples_per_npu = config.global_batch_size // world_size

    if config.gradient_accumulation_steps is None:
        if samples_per_npu % config.micro_batch_samples != 0:
            raise ValueError(
                f"samples_per_npu ({samples_per_npu}) must be divisible by micro_batch_samples "
                f"({config.micro_batch_samples})."
            )
        grad_accum = samples_per_npu // config.micro_batch_samples
    else:
        grad_accum = config.gradient_accumulation_steps
        expected = config.micro_batch_samples * grad_accum
        if expected != samples_per_npu:
            raise ValueError(
                f"micro_batch_samples ({config.micro_batch_samples}) * gradient_accumulation_steps "
                f"({grad_accum}) = {expected}, but expected samples_per_npu = {samples_per_npu}."
            )

    return BatchPlan(
        micro_batch_samples=config.micro_batch_samples,
        grad_accum_steps=grad_accum,
        samples_per_npu_per_step=samples_per_npu,
        samples_per_global_step=config.global_batch_size,
        batch_max_length=config.micro_batch_samples * max_seq_len,
    )


def compute_training_budget(
    num_params: int,
    batch_plan: BatchPlan,
    config: PretrainConfig,
    train_metadata: V1DatasetMeta,
) -> tuple[int, int]:
    if config.total_steps > 1:
        total_steps = config.total_steps
        total_samples = total_steps * batch_plan.samples_per_global_step
        return total_steps, total_samples

    if config.sample_multiplier is not None:
        sample_budget = int(num_params * config.sample_multiplier / train_metadata.max_seq_len)
        raw_steps = sample_budget / batch_plan.samples_per_global_step
        total_steps = round_steps_to_thousands(raw_steps)
        total_samples = total_steps * batch_plan.samples_per_global_step
        return total_steps, total_samples

    samples_per_epoch = train_metadata.total_length // train_metadata.max_seq_len
    total_samples = config.epochs * samples_per_epoch
    total_steps = int(total_samples // batch_plan.samples_per_global_step)
    return total_steps, total_samples


def _make_dataset(
    config: PretrainConfig,
    *,
    dataset_path: str,
    batch_max_length: int,
    drop_last_batch: bool,
    rank: int,
    world_size: int,
    fixed_epoch: Optional[int] = None,
    start_data_epoch: int = 0,
    resume_sampler_start_index: int = 0,
):
    """Build document Causal LM dataset (QA / inst_*/resp_* packing removed)."""
    if config.data.format != "document":
        raise ValueError(
            f"Only data.format=document is supported (got {config.data.format!r}). "
            "QA inst_*/resp_* packing has been removed."
        )
    return DocumentDataset(DocumentDatasetConfig(
        seed=config.seed,
        dataset_path=dataset_path,
        drop_last_batch=drop_last_batch,
        batch_max_length=batch_max_length,
        rank=rank,
        num_replicas=world_size,
        fixed_epoch=fixed_epoch,
        start_data_epoch=start_data_epoch,
        resume_sampler_start_index=resume_sampler_start_index,
    ))


def create_dataloader(
    config: PretrainConfig,
    batch_max_length: int,
    drop_last_batch: bool,
    rank: int,
    world_size: int,
    *,
    fixed_epoch: Optional[int] = None,
    start_data_epoch: int = 0,
    resume_sampler_start_index: int = 0,
):
    dataset = _make_dataset(
        config,
        dataset_path=config.data.path,
        batch_max_length=batch_max_length,
        drop_last_batch=drop_last_batch,
        rank=rank,
        world_size=world_size,
        fixed_epoch=fixed_epoch,
        start_data_epoch=start_data_epoch,
        resume_sampler_start_index=resume_sampler_start_index,
    )
    num_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", "1"))
    loader_kwargs: dict = {
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory(),
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 8
        loader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(dataset, **loader_kwargs)
    return dataloader, dataset.metadata


def _epoch_doc_count(dataset_path: str, epoch_idx: int) -> int:
    """Number of packed docs in epoch_N (permutation does not change count)."""
    root = Path(dataset_path)
    n_epochs = sum(1 for p in root.glob("epoch_*") if p.is_dir())
    if n_epochs <= 0:
        return 0
    epoch_idx = int(epoch_idx) % n_epochs
    path = root / f"epoch_{epoch_idx}" / "doc_len.npy"
    if not path.is_file():
        return 0
    arr = np.load(path, mmap_mode="r")
    return int(arr.shape[0])


def apply_resume_data_skip(
    config: PretrainConfig,
    train_state: TrainState,
    resume_progress: dict,
    seek_epoch: int,
    seek_index: int,
    rank: int = 0,
) -> tuple[int, int]:
    """Advance the resume dataloader cursor by ~RESUME_SKIP_STEPS global steps of data.

    Default 1000 so a rollback does not immediately replay the batches that
    triggered a spike. Set RESUME_SKIP_STEPS=0 to seek exactly at the ckpt cursor.
    Overflow wraps into the next data epoch.
    """
    raw = os.environ.get("RESUME_SKIP_STEPS", "1000").strip()
    try:
        skip_steps = int(raw) if raw else 0
    except (TypeError, ValueError):
        skip_steps = 1000
    if skip_steps <= 0:
        return seek_epoch, seek_index

    orig_step = int(resume_progress.get("step", 0) or 0)
    orig_idx = int(resume_progress.get("sampler_start_index", 0) or 0)
    orig_epoch = int(resume_progress.get("data_epoch", 0) or 0)
    gbs = max(int(getattr(train_state.batch_plan, "samples_per_global_step", 0) or 0), 1)
    # Document packing: ~2.56 docs per packed sample (matches 500-step cursor deltas).
    fallback_per_step = float(gbs) * 2.56
    if orig_epoch == 0 and orig_step > 0 and orig_idx > 0:
        per_step = float(orig_idx) / float(orig_step)
    else:
        per_step = fallback_per_step
    skip_docs = max(int(round(per_step * skip_steps)), 0)
    if skip_docs <= 0:
        return seek_epoch, seek_index

    epoch = int(seek_epoch)
    index = int(seek_index) + skip_docs
    dataset_path = str(config.data.path)
    n_epochs = sum(1 for p in Path(dataset_path).glob("epoch_*") if p.is_dir())
    n_epochs = max(int(n_epochs), 1)
    for _ in range(n_epochs + 2):
        n_docs = _epoch_doc_count(dataset_path, epoch)
        if n_docs <= 0 or index < n_docs:
            break
        index -= n_docs
        epoch += 1
    if rank == 0:
        print(
            f"[Resume] skip {skip_steps} steps of data (~{skip_docs} docs, "
            f"{per_step:.1f} docs/step) → data_epoch={epoch} sampler_start={index}",
            flush=True,
        )
    return epoch, index


def create_eval_dataloader(
    config: PretrainConfig,
    batch_plan: BatchPlan,
    rank: int,
    world_size: int,
    *,
    num_workers: int = 1,
    persistent_workers: bool = True,
):
    val_path = config.data.val_path or config.data.path
    eval_epoch = 0 if config.data.val_path is not None else config.eval_epoch
    dataset = _make_dataset(
        config,
        dataset_path=val_path,
        batch_max_length=batch_plan.batch_max_length,
        drop_last_batch=False,
        rank=rank,
        world_size=world_size,
        fixed_epoch=eval_epoch,
    )
    loader_kwargs: dict = {
        "batch_size": None,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory(),
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 8
    dataloader = DataLoader(dataset, **loader_kwargs)
    return dataloader, dataset.metadata


def apply_fsdp(module: nn.Module, param_dtype: torch.dtype):
    fully_shard(module,
                mp_policy=MixedPrecisionPolicy(param_dtype=param_dtype,
                                               reduce_dtype=torch.get_default_dtype()),  # Use master dtype for reduction
                reshard_after_forward=False)  # Trade off NPU memory for less comms
    
    assert isinstance(module, FSDPModule)
    # Disable gradient division during reduce-scatter (AdamW is scale invariant).
    if hasattr(module, "set_reduce_scatter_divide_factor"):
        module.set_reduce_scatter_divide_factor(1.0)
    elif hasattr(module, "set_gradient_divide_factor"):
        module.set_gradient_divide_factor(1.0)
    if hasattr(module, "set_force_sum_reduction_for_comms"):
        module.set_force_sum_reduction_for_comms(True)


def _is_cycle_lambda_param(name: str) -> bool:
    n = name.replace(".", "_").lower()
    return "cycle_lambda_raw" in n or "inject_lambda_raw" in n


def _is_dual_axis_beta_param(name: str) -> bool:
    n = name.replace(".", "_").lower()
    return any(
        tok in n
        for tok in ("beta_h_raw", "beta_l_raw", "beta_p_raw", "gate_raw")
    )


def create_optimizer(
    config: PretrainConfig, model: nn.Module
) -> tuple[Optimizer, Optional[Optimizer]]:
    """Build AdamW/AdamATan2 (+ optional separate λ optimizer).

    ``cycle_lambda_optim``:
      - adamw: λ as param-group in main AdamW
      - adamw_sep / adamw_indep: independent AdamW for λ only (μP scalar moments)
      - sgd / sign_sgd: separate SGD after jac compensation
    """
    lam_peak = (
        float(config.cycle_lambda_lr)
        if config.cycle_lambda_lr is not None
        else float(config.lr)
    )
    lam_wd = (
        float(config.cycle_lambda_weight_decay)
        if config.cycle_lambda_weight_decay is not None
        else float(config.weight_decay)
    )
    beta_peak = (
        float(config.dual_axis_beta_lr)
        if config.dual_axis_beta_lr is not None
        else float(config.lr)
    )
    beta_wd = (
        float(config.dual_axis_beta_weight_decay)
        if config.dual_axis_beta_weight_decay is not None
        else float(config.weight_decay)
    )
    lam_opt_name = str(getattr(config, "cycle_lambda_optim", "sgd") or "sgd").lower()

    main_params: list[nn.Parameter] = []
    lam_params: list[nn.Parameter] = []
    beta_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_cycle_lambda_param(name):
            lam_params.append(p)
        elif _is_dual_axis_beta_param(name):
            beta_params.append(p)
        else:
            main_params.append(p)

    use_lam_sgd = bool(lam_params) and lam_opt_name in ("sgd", "sign_sgd", "signsgd")
    use_lam_sign = lam_opt_name in ("sign_sgd", "signsgd")
    use_lam_adamw_sep = bool(lam_params) and lam_opt_name in ("adamw_sep", "adamw_indep")
    use_lam_sep = use_lam_sgd or use_lam_adamw_sep
    # peak_lr is consumed by update_lr (warmup/cosine relative to each group's peak).
    groups: list[dict] = [
        {
            "params": main_params,
            "lr": 0.0,
            "weight_decay": float(config.weight_decay),
            "group_name": "main",
            "peak_lr": float(config.lr),
        }
    ]
    if lam_params and not use_lam_sep:
        groups.append(
            {
                "params": lam_params,
                "lr": 0.0,
                "weight_decay": lam_wd,
                "group_name": "cycle_lambda",
                "peak_lr": lam_peak,
            }
        )
    if beta_params:
        groups.append(
            {
                "params": beta_params,
                "lr": 0.0,
                "weight_decay": beta_wd,
                "group_name": "dual_axis_beta",
                "peak_lr": beta_peak,
            }
        )

    if not any(g["params"] for g in groups):
        raise ValueError("create_optimizer: no trainable parameters")

    if config.optimizer == "adamw":
        opt = torch.optim.AdamW(
            groups,
            lr=0.0,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam_atan2":
        opt = AdamATan2(
            groups,
            lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
            ema=config.ema,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    opt_lam: Optional[Optimizer] = None
    if use_lam_sgd:
        opt_lam = torch.optim.SGD(
            [
                {
                    "params": lam_params,
                    "lr": 0.0,
                    "weight_decay": lam_wd,
                    "group_name": "cycle_lambda",
                    "peak_lr": lam_peak,
                }
            ],
            lr=0.0,
            momentum=0.0,
            weight_decay=lam_wd,
        )
    elif use_lam_adamw_sep:
        # Independent AdamW for scalar λ — own moments, not mixed with main params.
        opt_lam = torch.optim.AdamW(
            [
                {
                    "params": lam_params,
                    "lr": 0.0,
                    "weight_decay": lam_wd,
                    "group_name": "cycle_lambda",
                    "peak_lr": lam_peak,
                }
            ],
            lr=0.0,
            betas=(config.beta1, config.beta2),
            weight_decay=lam_wd,
        )

    # Log group sizes once (rank0 only if available).
    try:
        is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        is_rank0 = True
    if is_rank0:
        for g in groups:
            n = sum(int(p.numel()) for p in g["params"])
            print(
                f"[Optim] group={g['group_name']} n_param={len(g['params'])} "
                f"n_elem={n} peak_lr={g['peak_lr']} wd={g['weight_decay']}",
                flush=True,
            )
        if opt_lam is not None:
            if use_lam_adamw_sep:
                tag = "AdamW-sep"
            elif use_lam_sign:
                tag = "SignSGD"
            else:
                tag = "SGD"
            print(
                f"[Optim] group=cycle_lambda({tag}) n_param={len(lam_params)} "
                f"peak_lr={lam_peak} wd={lam_wd}",
                flush=True,
            )
    # Stash sign flag for compensate path (SGD only).
    if opt_lam is not None:
        setattr(opt_lam, "_cycle_lambda_sign_sgd", bool(use_lam_sign))
    return opt, opt_lam


def clip_grads_if_enabled(train_state: TrainState) -> Optional[float]:
    """Clip all trainable grads to ``clip_grad_norm`` (default 1.0). Returns total norm.

    Uses ``torch.nn.utils.clip_grad_norm_`` so FSDP2/DTensor shards reduce ||g||² correctly.
    Separate λ optimizer params are still model parameters → clipped once with the rest.
    """
    cfg = getattr(train_state, "pretrain_config", None)
    max_norm = getattr(cfg, "clip_grad_norm", None) if cfg is not None else None
    if max_norm is None:
        return None
    try:
        max_norm_f = float(max_norm)
    except (TypeError, ValueError):
        return None
    if max_norm_f <= 0.0:
        return None
    params = [p for p in train_state.model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return None
    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm_f)
    if torch.is_tensor(total_norm):
        total_norm_f = float(total_norm.detach().float().item())
    else:
        total_norm_f = float(total_norm)
    train_state._last_grad_norm = total_norm_f  # type: ignore[attr-defined]
    return total_norm_f


def global_mean_loss_from_metrics(metrics: Optional[dict]) -> Optional[float]:
    """All-reduce local loss [sum, count] → global mean (identical on every rank)."""
    if not metrics or "loss" not in metrics:
        return None
    pair = metrics["loss"]
    try:
        loss_sum = pair[0].detach().float().reshape(())
        loss_cnt = pair[1].detach().float().reshape(())
    except Exception:
        return None
    buf = torch.stack([loss_sum, loss_cnt])
    if dist.is_initialized():
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
    cnt = float(buf[1].item())
    if cnt <= 0.0:
        return None
    return float((buf[0] / buf[1]).item())


def _grads_finite(train_state: TrainState) -> bool:
    """Rank-local nonfinite-gradient check with a single host sync.

    Testing each parameter separately meant one device->host sync per parameter
    per step (hundreds here). Reducing to one scalar first keeps the same
    rank-local semantics: ranks are reconciled by the all_reduce(MAX) on the skip
    flag in the caller.
    """
    grads = []
    for p in train_state.model.parameters():
        g = _local_grad_view(getattr(p, "grad", None))
        if g is not None:
            grads.append(g)
    if not grads:
        return True
    norms = torch._foreach_norm(grads)
    return bool(torch.isfinite(torch.stack(norms).sum()).item())


def _local_grad_view(t) -> Optional[Tensor]:
    """Local shard of a gradient without triggering a collective (see _ascend_safe_local_tensor)."""
    return _ascend_safe_local_tensor(t)


def _record_spike_event(train_state: TrainState, step: int, reason: str, loss_value: Optional[float]) -> None:
    """Rank0 console + optional SPIKE_SUMMARY_FILE / checkpoint spike_events.jsonl."""
    train_state._last_skip_reason = reason  # type: ignore[attr-defined]
    train_state._skip_count = int(getattr(train_state, "_skip_count", 0)) + 1  # type: ignore[attr-defined]
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return
    prev = getattr(train_state, "_prev_train_loss", None)
    msg = (
        f"[SPIKE] step={step} skip=1 reason={reason} "
        f"loss={loss_value} prev={prev} skip_count={train_state._skip_count}"  # type: ignore[attr-defined]
    )
    print(msg, flush=True)
    for path in (
        os.environ.get("SPIKE_SUMMARY_FILE", ""),
        os.environ.get("SPIKE_EVENTS_JSONL", ""),
    ):
        if not path:
            continue
        try:
            import json as _json
            rec = {
                "step": int(step),
                "reason": reason,
                "loss": None if loss_value is None else float(loss_value),
                "prev_loss": None if prev is None else float(prev),
                "skip_count": int(train_state._skip_count),  # type: ignore[attr-defined]
            }
            with open(path, "a", encoding="utf-8") as f:
                if path.endswith(".jsonl"):
                    f.write(_json.dumps(rec, sort_keys=True) + "\n")
                else:
                    f.write(msg + "\n")
        except Exception:
            pass


def should_skip_bad_step(
    train_state: TrainState,
    loss_value: Optional[float],
    *,
    check_delta: bool = True,
    seg_idx: Optional[int] = None,
) -> tuple[bool, str]:
    cfg = getattr(train_state, "pretrain_config", None)
    if cfg is None or not bool(getattr(cfg, "skip_bad_step", True)):
        return False, ""
    if loss_value is not None and not math.isfinite(loss_value):
        return True, f"nonfinite_loss={loss_value}"
    if not _grads_finite(train_state):
        return True, "nonfinite_grad"
    if not check_delta:
        return False, ""
    # Early training: loss is high & noisy — only keep nonfinite guards.
    try:
        warm = int(getattr(cfg, "skip_delta_warmup_steps", 0) or 0)
    except (TypeError, ValueError):
        warm = 0
    if warm > 0 and int(getattr(train_state, "step", 0)) < warm:
        return False, ""
    try:
        delta_floor = float(getattr(cfg, "skip_loss_delta", 0.25) or 0.25)
    except (TypeError, ValueError):
        delta_floor = 0.25
    try:
        rel = float(getattr(cfg, "skip_loss_rel", 0.06) or 0.0)
    except (TypeError, ValueError):
        rel = 0.06
    try:
        ratio = float(getattr(cfg, "skip_loss_ratio", 0.09) or 0.0)
    except (TypeError, ValueError):
        ratio = 0.09
    try:
        hard = float(getattr(cfg, "skip_loss_hard", 1.5) or 0.0)
    except (TypeError, ValueError):
        hard = 1.5
    # Per-segment baselines: early K-loops naturally have higher CE than late ones;
    # comparing seg0 to a late-seg/global EMA caused permanent skip storms.
    if seg_idx is not None:
        ema_map = getattr(train_state, "_seg_loss_ema", None) or {}
        prev_map = getattr(train_state, "_prev_seg_loss", None) or {}
        ema = ema_map.get(int(seg_idx))
        prev = prev_map.get(int(seg_idx))
        baseline = ema if (ema is not None and math.isfinite(float(ema))) else prev
        # Cold-start this segment index: accept first observation, no Δ skip yet.
        if baseline is None:
            return False, ""
    else:
        prev = getattr(train_state, "_prev_train_loss", None)
        ema = getattr(train_state, "_train_loss_ema", None)
        baseline = ema if (ema is not None and math.isfinite(float(ema))) else prev
    if (
        baseline is not None
        and loss_value is not None
        and math.isfinite(float(baseline))
        and math.isfinite(loss_value)
    ):
        base = float(baseline)
        dloss = float(loss_value) - base
        tag = f"seg{seg_idx}" if seg_idx is not None else "g"
        if hard > 0.0 and dloss >= hard:
            return True, f"hard_dloss={dloss:.4f}/{tag}"
        if ratio > 0.0 and base > 0.0 and float(loss_value) >= base * (1.0 + ratio):
            return True, (
                f"ratio_dloss={dloss:.4f}/ratio={ratio:.3f}/"
                f"{base:.4f}→{float(loss_value):.4f}/{tag}"
            )
        thr = max(delta_floor, rel * max(base, 1e-8)) if rel > 0.0 else delta_floor
        if dloss >= thr:
            return True, f"adapt_dloss={dloss:.4f}/thr={thr:.4f}/{tag}"
    return False, ""


def step_optimizers(
    train_state: TrainState,
    *,
    step_lambda: bool = True,
    loss_value: Optional[float] = None,
    step_for_log: Optional[int] = None,
    check_delta: bool = True,
    update_prev_loss: bool = True,
    seg_idx: Optional[int] = None,
) -> bool:
    """Clip grads (if enabled) then step, or skip on bad loss/grad. Returns True if stepped."""
    cfg = getattr(train_state, "pretrain_config", None)
    skip, reason = should_skip_bad_step(
        train_state, loss_value, check_delta=check_delta, seg_idx=seg_idx
    )
    # All ranks must agree (local nonfinite_grad can otherwise desync the process group).
    if dist.is_initialized():
        try:
            device = next(train_state.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        flag = torch.zeros(1, dtype=torch.int32, device=device)
        if skip:
            flag[0] = 1
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if int(flag.item()) == 1 and not skip:
            skip, reason = True, (reason or "peer_skip")
    log_step = int(step_for_log if step_for_log is not None else train_state.step + 1)
    if skip:
        train_state.optim.zero_grad(set_to_none=True)
        if train_state.optim_lambda is not None:
            train_state.optim_lambda.zero_grad(set_to_none=True)
        _record_spike_event(train_state, log_step, reason, loss_value)
        # Do NOT auto-raise baselines after consecutive skips — that locked H9 into
        # accepting elevated early-seg CE and stepping a collapsing model.
        train_state._consec_skips = int(getattr(train_state, "_consec_skips", 0)) + 1  # type: ignore[attr-defined]
        train_state._step_had_skip = True  # type: ignore[attr-defined]
        abort = bool(getattr(cfg, "skip_abort_on_spike", False)) if cfg is not None else False
        spike_env = os.environ.get("SPIKE_ABORT", "").strip()
        if spike_env in ("1", "true", "True", "yes"):
            abort = True
        elif spike_env in ("0", "false", "False", "no", "off", "OFF"):
            abort = False
        if abort:
            train_state._abort_training = True  # type: ignore[attr-defined]
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                alarm = (
                    f"[SPIKE_ABORT] step={log_step} reason={reason} "
                    f"loss={loss_value} — stopping training NOW"
                )
                print(alarm, flush=True)
                for path in (
                    os.environ.get("SPIKE_SUMMARY_FILE", ""),
                    os.environ.get("SPIKE_ALARM_FILE", ""),
                ):
                    if not path:
                        continue
                    try:
                        with open(path, "a", encoding="utf-8") as f:
                            f.write(alarm + "\n")
                    except Exception:
                        pass
                ckpt = getattr(cfg, "checkpoint_path", None) if cfg is not None else None
                if ckpt:
                    try:
                        with open(os.path.join(str(ckpt), "SPIKE_ABORT.txt"), "w", encoding="utf-8") as f:
                            f.write(alarm + "\n")
                    except Exception:
                        pass
        # Keep last *good* prev; do not poison baseline with the spike loss.
        train_state._last_grad_norm = None  # type: ignore[attr-defined]
        return False

    clip_grads_if_enabled(train_state)
    train_state.optim.step()
    if step_lambda and train_state.optim_lambda is not None:
        train_state.optim_lambda.step()
    # Always project after main step: λ may live in AdamW-main and is updated even
    # when step_lambda=False (segmented BPTT). Missing this → WD drove λ to 1.26.
    core = getattr(train_state.model, "model", train_state.model)
    proj = getattr(core, "project_cycle_lambda_raw_away_from_dead_jac", None)
    if callable(proj):
        proj()
    if update_prev_loss and loss_value is not None and math.isfinite(loss_value):
        lv = float(loss_value)
        try:
            ema_c = float(getattr(cfg, "skip_loss_ema", 0.9) or 0.0) if cfg is not None else 0.9
        except (TypeError, ValueError):
            ema_c = 0.9
        if seg_idx is not None:
            si = int(seg_idx)
            prev_map = getattr(train_state, "_prev_seg_loss", None)
            if not isinstance(prev_map, dict):
                prev_map = {}
            ema_map = getattr(train_state, "_seg_loss_ema", None)
            if not isinstance(ema_map, dict):
                ema_map = {}
            prev_map[si] = lv
            prev_ema = ema_map.get(si)
            if ema_c <= 0.0 or prev_ema is None or not math.isfinite(float(prev_ema)):
                ema_map[si] = lv
            else:
                ema_map[si] = ema_c * float(prev_ema) + (1.0 - ema_c) * lv
            train_state._prev_seg_loss = prev_map  # type: ignore[attr-defined]
            train_state._seg_loss_ema = ema_map  # type: ignore[attr-defined]
        train_state._prev_train_loss = lv  # type: ignore[attr-defined]
        prev_ema = getattr(train_state, "_train_loss_ema", None)
        if ema_c <= 0.0 or prev_ema is None or not math.isfinite(float(prev_ema)):
            train_state._train_loss_ema = lv  # type: ignore[attr-defined]
        else:
            train_state._train_loss_ema = (  # type: ignore[attr-defined]
                ema_c * float(prev_ema) + (1.0 - ema_c) * lv
            )
    train_state._last_skip_reason = None  # type: ignore[attr-defined]
    train_state._consec_skips = 0  # type: ignore[attr-defined]
    return True


def create_model_and_carry(config: PretrainConfig, train_metadata: V1DatasetMeta, micro_batch_samples: int):
    model_cfg = config.arch.model_dump() | train_metadata.model_dump() | config.data.model_dump()
    fwd_bwd_dtype = getattr(torch, config.fwd_bwd_dtype)

    # Instantiate model with head
    model_cls = load_model_class(config.arch.name)
    head_cls = load_model_class(config.arch.head)

    with torch.device(get_device()):
        model: nn.Module = model_cls(model_cfg)
        carry = model.initial_carry(micro_batch_samples, dtype=fwd_bwd_dtype)  # pyright: ignore[reportCallIssue]
        # Attach loss head
        model = head_cls(model, model_cfg)

    # ----FSDP----
    # Broadcast buffers
    for buffer in model.buffers():
        dist.broadcast(buffer, src=0)

    # Detect TransformerBlock recursively and apply FSDP
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            apply_fsdp(module, fwd_bwd_dtype)

    apply_fsdp(model, fwd_bwd_dtype)

    # ----Create optimizer----
    optim, optim_lambda = create_optimizer(config, model)

    return model, carry, optim, optim_lambda


def init_train(config: PretrainConfig, rank: int, world_size: int):
    with open(os.path.join(config.data.path, "metadata.json"), "r") as f:
        peek_meta = V1DatasetMeta(**json.load(f))
    peek_meta.max_seq_len -= 1

    batch_plan = resolve_batch_plan(config, world_size, peek_meta.max_seq_len)

    # Dataset
    train_loader, train_metadata = create_dataloader(
        config, batch_plan.batch_max_length, drop_last_batch=True, rank=rank, world_size=world_size
    )
    eval_loader = None
    if config.eval_interval > 0:
        eval_loader, _ = create_eval_dataloader(config, batch_plan, rank=rank, world_size=world_size)

    # Model
    model, carry, optim, optim_lambda = create_model_and_carry(
        config, train_metadata, batch_plan.micro_batch_samples
    )

    # Train state
    num_params = sum(p.numel() for p in model.parameters())
    total_steps, total_samples = compute_training_budget(num_params, batch_plan, config, train_metadata)
    if config.lr_warmup_ratio is not None:
        config.lr_warmup_steps = max(1, round(total_steps * config.lr_warmup_ratio))
    decay_ratio = getattr(config, "lr_decay_ratio", None)
    if decay_ratio is not None:
        config.lr_decay_steps = max(0, round(total_steps * float(decay_ratio)))
    else:
        config.lr_decay_steps = int(getattr(config, "lr_decay_steps", 0) or 0)
    train_state = TrainState(
        model=model,
        carry=carry,
        optim=optim,
        step=0,
        total_steps=total_steps,
        total_samples=total_samples,
        num_params=num_params,
        batch_plan=batch_plan,
        optim_lambda=optim_lambda,
    )
    # Soft attach for helpers that need PretrainConfig (λ-space jac compensation).
    train_state.pretrain_config = config  # type: ignore[attr-defined]
    return train_state, train_loader, eval_loader, train_metadata


def compensate_cycle_lambda_grads_if_enabled(
    train_state: TrainState, config: Optional[PretrainConfig] = None
) -> dict[str, float]:
    """Apply λ-space Jacobian compensation on the inner LoopedTransformer (if present)."""
    cfg = config if config is not None else getattr(train_state, "pretrain_config", None)
    if cfg is None or not bool(getattr(cfg, "cycle_lambda_grad_in_lambda_space", False)):
        return {}
    core = getattr(train_state.model, "model", train_state.model)
    fn = getattr(core, "compensate_cycle_lambda_grads_to_lambda_space", None)
    if not callable(fn):
        return {}
    out = fn() or {}
    # SignSGD in λ: after /jac²/ws, g≈∂L/∂λ/jac; set g←sign(g)/jac ⇒ Δλ=−lr·sign(∂L/∂λ).
    opt_lam = getattr(train_state, "optim_lambda", None)
    if opt_lam is not None and bool(getattr(opt_lam, "_cycle_lambda_sign_sgd", False)):
        for attr in ("cycle_lambda_raw", "inject_lambda_raw"):
            p = getattr(core, attr, None)
            if p is None or p.grad is None:
                continue
            g = p.grad.detach().float()
            out[f"{attr}_grad_mean_pre_sign"] = float(g.mean().item())
            out[f"{attr}_grad_abs_pre_sign"] = float(g.abs().mean().item())
            out[f"{attr}_grad_sign_frac"] = float(g.sign().mean().item())
            jac = core._lambda_pack_jacobian(p.data).clamp_min(1e-4)
            p.grad.sign_()
            p.grad.div_(jac)
        out["cycle_lambda_sign_sgd"] = 1.0
    # AttnRes β grads: read on ALL ranks (local shard / replicated). Never densify.
    try:
        for name, p in train_state.model.named_parameters():
            if "beta" not in name.lower() or p.grad is None:
                continue
            if type(p).__name__ == "DTensor":
                continue  # skip sharded — rank0-only densify hangs
            g = p.grad.detach().float()
            short = name.split(".")[-1]
            out[f"{short}_grad_mean"] = float(g.mean().item())
            out[f"{short}_grad_abs"] = float(g.abs().mean().item())
    except Exception:
        pass
    return out


def apply_scheduled_cycle_lambda(train_state: TrainState) -> None:
    """Frozen shared λ: linear start→end over total_steps (no-op if sched=none)."""
    core = getattr(train_state.model, "model", train_state.model)
    fn = getattr(core, "apply_cycle_lambda_schedule", None)
    if not callable(fn):
        return
    fn(int(train_state.step), int(train_state.total_steps))


def update_lr(config: PretrainConfig, train_state: TrainState, step: Optional[int] = None) -> float:
    step = train_state.step if step is None else step
    total = max(int(train_state.total_steps), 1)
    warmup = max(int(config.lr_warmup_steps), 0)
    sched = str(getattr(config, "lr_schedule", "cosine") or "cosine").lower()
    decay = max(int(getattr(config, "lr_decay_steps", 0) or 0), 0)

    if sched in ("wsd", "warmup_stable_decay"):
        # Warmup → stable (scale=1) → linear decay over last `decay` steps.
        if warmup > 0 and step < warmup:
            scale = min(1.0, step / max(warmup, 1))
        elif decay > 0 and step >= max(total - decay, warmup):
            # progress 0→1 across the decay window
            start = max(total - decay, warmup)
            denom = max(total - start, 1)
            progress = min(1.0, max(0.0, (step - start) / denom))
            scale = 1.0 + (float(config.lr_min_ratio) - 1.0) * progress
        else:
            scale = 1.0
    else:
        # Linear warmup + cosine schedule (per-group peak_lr; main uses config.lr).
        if step < warmup:
            scale = min(1.0, step / max(warmup, 1))
        else:
            denom = max(total - warmup, 1)
            progress = (step - warmup) / denom
            scale = config.lr_min_ratio + max(
                0.0, (1 - config.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
            )

    main_lr = float(config.lr) * scale
    for param_group in train_state.optim.param_groups:
        peak = float(param_group.get("peak_lr", config.lr))
        lr_g = peak * scale
        # AdamATan2 historically stored Tensor lr; keep float for AdamW / DCP safety.
        if config.optimizer == "adam_atan2":
            param_group["lr"] = torch.tensor(lr_g, dtype=torch.get_default_dtype(), device="cpu")
        else:
            param_group["lr"] = lr_g
    if train_state.optim_lambda is not None:
        # SignSGD target-band: lr≈(λ_init−1.6)/T must stay constant — do not inherit
        # Adam warmup/cosine (that would shrink early Δλ and defeat the schedule).
        lam_opt = str(getattr(config, "cycle_lambda_optim", "sgd") or "sgd").lower()
        lam_sched = os.environ.get("CYCLE_LAMBDA_LR_SCHEDULE", "").strip().lower()
        if not lam_sched:
            lam_sched = "constant" if lam_opt in ("sign_sgd", "sign-sgd", "signsgd") else "main"
        lam_scale = 1.0 if lam_sched == "constant" else scale
        for param_group in train_state.optim_lambda.param_groups:
            peak = float(param_group.get("peak_lr", config.cycle_lambda_lr or config.lr))
            param_group["lr"] = float(peak) * lam_scale

    return main_lr


def _ascend_safe_local_tensor(t) -> Optional[Tensor]:
    """Plain Tensor, or DTensor's already-materialized ``_local_tensor`` (no ``to_local``).

    Calling ``DTensor.to_local()`` mid-step has hung Ascend HCCL on H≥9. Reading the
    private local buffer is rank-local and collective-free.
    """
    if t is None:
        return None
    if type(t).__name__ == "DTensor":
        loc = getattr(t, "_local_tensor", None)
        if loc is None or not isinstance(loc, Tensor):
            return None
        t = loc
    if not isinstance(t, Tensor) or t.numel() == 0 or t.device.type == "meta":
        return None
    return t.detach()


def _collect_lambda_grad_snapshot(core) -> dict[str, float]:
    """Read-only λ/θ grad + jac (no /jac² rewrite; Ascend-safe, no densify/collective).

    Prefer plain Parameter; for DTensor use ``_local_tensor`` only — never ``to_local``.
    module_diag.csv also records grad/cycle_lambda_raw_* when mid-diag is on.
    """
    out: dict[str, float] = {}
    try:
        for attr in ("cycle_lambda_raw", "inject_lambda_raw"):
            p = getattr(core, attr, None)
            if p is None or getattr(p, "grad", None) is None:
                continue
            g = _ascend_safe_local_tensor(p.grad)
            if g is None:
                continue
            g_f = g.float()
            out[f"{attr}_grad_theta_mean"] = float(g_f.mean().item())
            out[f"{attr}_grad_theta_abs"] = float(g_f.abs().mean().item())
            jac_fn = getattr(core, "_lambda_pack_jacobian", None)
            data = _ascend_safe_local_tensor(p.data if hasattr(p, "data") else p)
            if callable(jac_fn) and data is not None:
                jac = jac_fn(data).clamp_min(1e-4).detach().float()
                # g_θ = (∂L/∂λ)·jac ⇒ ∂L/∂λ ≈ g_θ/jac
                g_lam = g_f / jac
                out[f"{attr}_grad_lambda_mean"] = float(g_lam.mean().item())
                out[f"{attr}_grad_lambda_abs"] = float(g_lam.abs().mean().item())
                out["cycle_lambda_jac_mean"] = float(jac.mean().item())
    except Exception:
        pass
    return out


def _log_lambda_beta_snapshot(train_state: TrainState, step: int, jac_m: Optional[dict] = None) -> None:
    """Rank0 dump of λ / AttnRes-β grad stats (no DTensor densify — Ascend-safe).

    Mid-run: jac_m + optional packed λ from raw (local shard / plain Param only).
    When grad_in_lambda_space=false, still collect θ/λ grads read-only.
    """
    every = int(os.environ.get("LAMBDA_LOG_EVERY", "0") or 0)
    if every <= 0 or step <= 0 or step % every != 0:
        return
    try:
        is_rank0 = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        is_rank0 = True
    if not is_rank0:
        return
    parts: list[str] = [f"[λβ step {step}]"]
    merged: dict[str, float] = dict(jac_m or {})
    # Cheap packed-λ readout (no densify): plain Parameter or local tensor only.
    try:
        core = getattr(train_state.model, "model", train_state.model)
        # Always fill λ grads if missing (lamspace off → jac_m empty).
        if not any("grad_theta" in k or "grad_lambda" in k for k in merged):
            merged.update(_collect_lambda_grad_snapshot(core))
        raw = getattr(core, "cycle_lambda_raw", None)
        if raw is not None and hasattr(core, "_pack_lambda"):
            t = _ascend_safe_local_tensor(raw)
            if t is not None and t.numel() == 1:
                lam = core._pack_lambda(t)
                lam_f = float(lam.reshape(()).item())
                parts.append(f"cycle_lambda={lam_f:.4f}")
                train_state._last_cycle_lambda = lam_f  # type: ignore[attr-defined]
                latched = getattr(core, "_cycle_lambda_settled_latched", None)
                if latched is not None:
                    parts.append(f"settle_latched={bool(latched.item())}")
            else:
                # Fallback: config / cached float (no densify).
                try:
                    lam_attr = getattr(core, "cycle_scale_lambda", None)
                    if lam_attr is not None:
                        parts.append(f"cycle_lambda≈{float(lam_attr):.4f}")
                except Exception:
                    pass
        else:
            try:
                lam_attr = getattr(core, "cycle_scale_lambda", None)
                if lam_attr is not None:
                    parts.append(f"cycle_lambda={float(lam_attr):.4f}(frozen)")
            except Exception:
                pass
            parts.append("grad/cycle_lambda=0(frozen)")
        try:
            beta_fn = getattr(core, "get_attn_res_beta_metrics", None)
            if callable(beta_fn):
                bm = beta_fn() or {}
                for k in ("attn_res_beta_h", "attn_res_beta_l", "attn_res_beta_p"):
                    if k in bm:
                        parts.append(f"{k}={float(bm[k]):.4f}")
            else:
                for attr in ("dual_axis_beta_h_init", "dual_axis_beta_l_init"):
                    v = getattr(core, attr, None)
                    if v is not None:
                        parts.append(f"{attr}={float(v):.4f}(frozen)")
        except Exception:
            pass
        if raw is None:
            parts.append("grad/attn_res_beta=0(frozen)")
        # AttnRes β grad norms — plain or _local_tensor only (no to_local).
        for name, p in core.named_parameters():
            if "beta" not in name.lower():
                continue
            g = _ascend_safe_local_tensor(getattr(p, "grad", None))
            if g is None:
                continue
            parts.append(f"grad/{name}_meanabs={float(g.abs().mean().item()):.4g}")
            if len([x for x in parts if x.startswith("grad/")]) >= 4:
                break
    except Exception:
        pass
    for k, v in sorted(merged.items()):
        if k.startswith("cycle_lambda") or "grad" in k or "sign" in k or "jac" in k:
            try:
                parts.append(f"{k}={float(v):.4g}")
            except (TypeError, ValueError):
                parts.append(f"{k}={v}")
    if len(parts) > 1:
        print(" ".join(parts), flush=True)


def _train_batch_impl(train_state: TrainState, batch: dict[str, Tensor], grad_accum_steps: int, **kwargs):
    train_state.carry, loss, metrics = train_state.model(batch=batch, carry=train_state.carry, **kwargs)
    (loss / grad_accum_steps).backward()
    return metrics


def _bp_segment_len(model: nn.Module) -> int:
    inner = getattr(model, "model", None)
    return max(0, int(getattr(inner, "bp_segment_len", 0) or 0))


def run_segmented_global_step(
    train_state: TrainState,
    microbatches: list[dict[str, Tensor]],
    **kwargs,
):
    """FPRM-style: every K loops, CE → backward → optim.step → detach z (+ AttnRes).

    Always consumes all H loops (no early-exit) *unless* a segment spikes — then the
    remaining segments of this global step are skipped (no further BP / optim).

    ε(t)/γ(t) stay on global t. Later segments recompute embeddings with updated weights.

    AttnRes dual-axis: H-axis N/D is snapshotted (detached) and restored into the
    next segment — forward memory connects across K boundaries; only BPTT is cut.

    λ stepped once per global step (last segment): per-segment SignSGD cancelled.
    Every segment runs skip_bad_step (Δloss / relative / nonfinite).
    """
    apply_scheduled_cycle_lambda(train_state)
    head = train_state.model
    core = getattr(head, "model", head)
    K = max(1, int(core.bp_segment_len))
    H = int(core.num_loops)
    n_micro = max(1, len(microbatches))
    zs: list[Optional[Tensor]] = [None] * n_micro
    banks: list = [None] * n_micro
    last_metrics = None
    last_jac: dict[str, float] = {}
    t = 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    while t <= H:
        t_end = min(t + K - 1, H)
        last_seg = t_end >= H
        train_state.optim.zero_grad()
        # λ: credit only the last BPTT segment (early/late CE grads oppose — p2d/p2e).
        # Zero at last-seg start so SignSGD/Adam see final-loop ∂L/∂λ, matching AdamW-with-main
        # direction (phase1 ↓) instead of sum-cancel / wrong-sign (p2e ↑).
        if train_state.optim_lambda is not None and (t == 1 or last_seg):
            train_state.optim_lambda.zero_grad()
        for i, batch in enumerate(microbatches):
            extra = dict(kwargs)
            extra["loop_t_start"] = t
            extra["loop_t_end"] = t_end
            extra["reset_bank"] = t == 1 and banks[i] is None
            extra["bank_snapshot"] = banks[i]
            new_z, loss, metrics = head(batch=batch, carry=zs[i], **extra)
            (loss / n_micro).backward()
            zs[i] = new_z.detach() if torch.is_tensor(new_z) else new_z
            if hasattr(core, "cross_loop_snapshot"):
                banks[i] = core.cross_loop_snapshot()
            else:
                bank = getattr(core, "attn_res_bank", None)
                banks[i] = bank.snapshot() if bank is not None else None
            last_metrics = metrics
        if last_seg:
            last_jac = compensate_cycle_lambda_grads_if_enabled(train_state)
            # Log λ/β + pre-sign grads while θ.grad still populated.
            _log_lambda_beta_snapshot(train_state, train_state.step + 1, last_jac)
        loss_g = global_mean_loss_from_metrics(last_metrics)
        seg_idx = (t - 1) // K
        # Every K-segment: Δloss + nonfinite. Per-seg EMA baseline (not global)
        # so early-K CE is not compared to late-K. First spike also aborts the
        # remaining segments of this global step (and the job if SPIKE_ABORT=1).
        stepped = step_optimizers(
            train_state,
            step_lambda=last_seg,
            loss_value=loss_g,
            step_for_log=train_state.step + 1,
            check_delta=True,
            update_prev_loss=True,
            seg_idx=seg_idx,
        )
        if not stepped:
            # Spike on this K-segment: drop remaining segments of this global step.
            if rank == 0:
                print(
                    f"[SPIKE_SEG] step={train_state.step + 1} seg_loops={t}-{t_end}/{H} "
                    f"loss={loss_g} — skip remaining segments, next global step",
                    flush=True,
                )
            break
        t = t_end + 1
    # Stash jac for module_diag merge (caller reads train_state._last_lambda_jac).
    train_state._last_lambda_jac = last_jac  # type: ignore[attr-defined]
    return last_metrics


if os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1":
    train_batch = _train_batch_impl
else:
    train_batch = torch.compile(dynamic=False)(_train_batch_impl)


@torch.inference_mode()
def reduce_metrics(local_metrics: dict[str, Tensor], prefix: str):
    metric_keys = list(sorted(local_metrics.keys()))  # Sort keys to guarantee all processes use the same order.
    # Reduce and reconstruct
    metric_values = torch.stack([local_metrics[k][0] for k in metric_keys] + [local_metrics[k][1] for k in metric_keys])
    dist.reduce(metric_values, dst=0)
    # Split and normalize
    metrics, metrics_div = metric_values.chunk(2, dim=-1)
    metrics = (metrics / metrics_div).cpu().numpy().tolist()
    return {prefix + name: metrics[idx] for idx, name in enumerate(metric_keys)}


def _all_reduce_sum(tensor: Tensor) -> Tensor:
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


@torch.inference_mode()
def run_eval(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: DataLoader,
    extra_args_override: Optional[dict] = None,
) -> dict[str, float]:
    train_state.model.eval()
    fwd_bwd_dtype = getattr(torch, config.fwd_bwd_dtype)
    micro_batch_samples = train_state.batch_plan.micro_batch_samples

    total_loss = torch.zeros((), device=get_device(), dtype=torch.float32)
    total_tokens = torch.zeros((), device=get_device(), dtype=torch.float32)
    eval_extra_args = train_state.model.compute_train_extra_args(train_state)  # pyright: ignore[reportCallIssue]
    if extra_args_override:
        eval_extra_args = {**eval_extra_args, **extra_args_override}

    for batch, batch_info in eval_loader:
        # Strip resume cursors — same as train loop; must not enter model forward.
        model_batch_info = {
            k: v for k, v in batch_info.items()
            if not str(k).startswith("_resume_")
        }
        batch_payload = batch | {k: wrap_tensor(torch.tensor(v, device="cpu")) for k, v in model_batch_info.items()}
        carry = train_state.model.model.initial_carry(micro_batch_samples, dtype=fwd_bwd_dtype)  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        _, _, metrics = train_state.model(batch=batch_payload, carry=carry, **eval_extra_args)

        total_loss += metrics["loss"][0].float()
        total_tokens += metrics["loss"][1].float()

        global_tokens = total_tokens.clone()
        _all_reduce_sum(global_tokens)
        if global_tokens.item() >= config.eval_token_budget:
            break

    stats = torch.stack([total_loss, total_tokens])
    _all_reduce_sum(stats)
    avg_loss = (stats[0] / stats[1]).item() if stats[1].item() > 0 else float("nan")

    train_state.model.train()
    return {"eval/loss": avg_loss, "eval/tokens": stats[1].item()}


def run_heff_sweep_eval(
    config: PretrainConfig,
    train_state: TrainState,
    rank: int,
    world_size: int,
    heff_list: list[int],
) -> dict[int, float]:
    """Same weights, early-exit at each H_eff (paper-style deeper-at-eval protocol)."""
    synchronize()
    empty_cache()
    eval_loader, _ = create_eval_dataloader(
        config,
        train_state.batch_plan,
        rank=rank,
        world_size=world_size,
        num_workers=0,
        persistent_workers=False,
    )
    out: dict[int, float] = {}
    try:
        for h in heff_list:
            h = int(h)
            h = max(1, min(h, int(getattr(train_state.model.model, "H_cycles", h))))
            metrics = run_eval(
                config,
                train_state,
                eval_loader,
                extra_args_override={"H_cycles_eff": h},
            )
            out[h] = float(metrics["eval/loss"])
            if rank == 0:
                print(
                    f"[eval H_eff={h}] loss={metrics['eval/loss']:.4f} "
                    f"tokens={metrics['eval/tokens']:.0f}",
                    flush=True,
                )
    finally:
        del eval_loader
    return out


def run_final_eval(
    config: PretrainConfig,
    train_state: TrainState,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    synchronize()
    empty_cache()
    final_eval_loader, _ = create_eval_dataloader(
        config,
        train_state.batch_plan,
        rank=rank,
        world_size=world_size,
        num_workers=0,
        persistent_workers=False,
    )
    try:
        return run_eval(config, train_state, final_eval_loader)
    finally:
        del final_eval_loader


def _progress_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "train_progress.yaml")


def _pending_downstream_eval_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "pending_downstream_eval.json")


def _rng_path(ckpt_dir: str, rank: int) -> str:
    return os.path.join(ckpt_dir, f"rng.{rank}.pt")


def save_rng_state(ckpt_dir: str, rank: int) -> None:
    """Fast per-rank RNG snapshot (CPU torch/numpy/python + NPU/CUDA if present)."""
    state: dict = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state()
        except Exception:
            pass
    try:
        if hasattr(torch, "npu") and torch.npu.is_available():
            state["npu"] = torch.npu.get_rng_state()
    except Exception:
        pass
    torch.save(state, _rng_path(ckpt_dir, rank))


def load_rng_state(ckpt_dir: str, rank: int) -> None:
    path = _rng_path(ckpt_dir, rank)
    if not os.path.isfile(path):
        return
    state = torch.load(path, map_location="cpu", weights_only=False)
    torch.set_rng_state(state["torch"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state(state["cuda"])
        except Exception:
            pass
    if "npu" in state:
        try:
            if hasattr(torch, "npu") and torch.npu.is_available():
                torch.npu.set_rng_state(state["npu"])
        except Exception:
            pass


def save_train_progress(
    config: PretrainConfig,
    train_state: TrainState,
    *,
    loop_epoch: int,
    data_epoch: int,
    sampler_start_index: int,
    rank: int,
) -> None:
    if config.checkpoint_path is None:
        return
    # Tiny YAML on rank0 + fast RNG blob on every rank (no FSDP gather).
    save_rng_state(config.checkpoint_path, rank)
    if rank != 0:
        return
    payload = {
        "step": int(train_state.step),
        "total_steps": int(train_state.total_steps),
        "loop_epoch": int(loop_epoch),
        "data_epoch": int(data_epoch),
        "sampler_start_index": int(sampler_start_index),
        # Backward-compatible alias used by older readers.
        "epoch": int(loop_epoch),
    }
    with open(_progress_path(config.checkpoint_path), "wt") as f:
        yaml.dump(payload, f)


def write_pending_downstream_eval(
    config: PretrainConfig,
    train_state: TrainState,
    *,
    loop_epoch: int,
    rank: int,
) -> None:
    if rank != 0 or config.checkpoint_path is None:
        return
    payload = {
        "step": int(train_state.step),
        "total_steps": int(train_state.total_steps),
        "epoch": int(loop_epoch),
        "loop_epoch": int(loop_epoch),
        "checkpoint_path": config.checkpoint_path,
    }
    with open(_pending_downstream_eval_path(config.checkpoint_path), "wt") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(
        f"[DownstreamEval] Pause requested at step {train_state.step}/{train_state.total_steps} "
        f"(interval={config.downstream_eval_interval})",
        flush=True,
    )


def _step_snapshot_dirname(step: int) -> str:
    """Canonical mid-train snapshot folder name: step_0001000 (7-digit, sortable)."""
    return f"step_{int(step):07d}"


def snapshot_checkpoint_at_step(
    config: PretrainConfig,
    train_state: TrainState,
    *,
    ckpt_epoch: int,
    rank: int,
    is_final: bool = False,
) -> None:
    """Keep a full, immutable copy of weights+cursor at this step for rollback.

    After mid-train pauses at e.g. 2500 / 5000 / 7500 you get::

        snapshots/step_0002500/   # full ckpt @ 2500 — keep forever
        snapshots/step_0005000/   # full ckpt @ 5000 — keep forever
        snapshots/step_0007500/   # full ckpt @ 7500 — keep forever

    These directories **coexist**. Later training only overwrites the **live**
    ``fsdp2_epoch_*`` under ``checkpoint_path``, never older snapshot folders.

    Rollback / resume from a past step::

        RESUME_FROM=.../snapshots/step_0005000
    """
    if config.checkpoint_path is None:
        return
    if dist.is_initialized():
        dist.barrier()
    if rank != 0:
        if dist.is_initialized():
            dist.barrier()
        return

    step = int(train_state.step)
    snap_name = _step_snapshot_dirname(step)
    snap_dir = os.path.join(config.checkpoint_path, "snapshots", snap_name)
    os.makedirs(snap_dir, exist_ok=True)

    src_fsdp = os.path.join(config.checkpoint_path, f"fsdp2_epoch_{ckpt_epoch}")
    dst_fsdp = os.path.join(snap_dir, f"fsdp2_epoch_{ckpt_epoch}")
    if os.path.isdir(src_fsdp):
        if os.path.exists(dst_fsdp):
            shutil.rmtree(dst_fsdp)
        shutil.copytree(src_fsdp, dst_fsdp)
        # Compat symlink: many eval launchers historically hardcode fsdp2_epoch_1.
        # When ckpt_epoch != 1, point epoch_1 -> the real weights so those tools work.
        if int(ckpt_epoch) != 1:
            link_path = os.path.join(snap_dir, "fsdp2_epoch_1")
            if os.path.islink(link_path) or os.path.exists(link_path):
                if os.path.islink(link_path) or os.path.isfile(link_path):
                    os.unlink(link_path)
                else:
                    shutil.rmtree(link_path)
            os.symlink(f"fsdp2_epoch_{ckpt_epoch}", link_path)

    for name in ("train_progress.yaml", "all_config.yaml", "train_metadata.yaml"):
        src = os.path.join(config.checkpoint_path, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(snap_dir, name))

    # Carry + RNG for every rank — required for full resume/rollback.
    for carry_path in glob(os.path.join(config.checkpoint_path, f"carry_epoch_{ckpt_epoch}.*.pt")):
        shutil.copy2(carry_path, os.path.join(snap_dir, os.path.basename(carry_path)))
    for rng_path in glob(os.path.join(config.checkpoint_path, "rng.*.pt")):
        shutil.copy2(rng_path, os.path.join(snap_dir, os.path.basename(rng_path)))

    with open(os.path.join(snap_dir, "snapshot_meta.json"), "wt", encoding="utf-8") as f:
        json.dump(
            {
                "step": step,
                "step_dirname": snap_name,
                "ckpt_epoch": int(ckpt_epoch),
                "source_checkpoint_path": config.checkpoint_path,
                "snapshot_dir": snap_dir,
                "resumable": True,
                "final": bool(is_final),
                "resume_from": snap_dir,
                "note": (
                    "Immutable full checkpoint at this step. "
                    "Coexists with other snapshots/step_*/. "
                    "To roll back: RESUME_FROM=<this snapshot_dir>."
                ),
            },
            f,
            indent=2,
        )

    # Maintain an index of all kept steps (for humans + tooling).
    snaps_root = os.path.join(config.checkpoint_path, "snapshots")
    index: dict = {"steps": [], "layout": "snapshots/step_NNNNNNN/ = full ckpt at that step (rollback-capable)"}
    for name in sorted(os.listdir(snaps_root)):
        if not name.startswith("step_"):
            continue
        meta_p = os.path.join(snaps_root, name, "snapshot_meta.json")
        entry = {"dirname": name, "path": os.path.join(snaps_root, name)}
        if os.path.isfile(meta_p):
            try:
                with open(meta_p, "rt", encoding="utf-8") as mf:
                    m = json.load(mf)
                entry["step"] = m.get("step")
            except Exception:
                pass
        index["steps"].append(entry)
    with open(os.path.join(snaps_root, "index.json"), "wt", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    # Rotation: each snapshot is a full copy of the weights + optimizer state
    # (~19 GB for the 1.6B MoE), so keeping every step fills a disk fast.
    # SNAPSHOT_KEEP_LAST=0 keeps them all (the original behaviour).
    if is_final:
        # Downstream evaluation needs one stable path that always means "end of
        # training", independent of the step number and of rotation.
        link_path = os.path.join(snaps_root, "final")
        if os.path.islink(link_path) or os.path.exists(link_path):
            if os.path.islink(link_path) or os.path.isfile(link_path):
                os.unlink(link_path)
            else:
                shutil.rmtree(link_path)
        os.symlink(snap_name, link_path)
        print(f"[Checkpoint] final snapshot -> {link_path} ({snap_name})", flush=True)

    final_target = None
    final_link = os.path.join(snaps_root, "final")
    if os.path.islink(final_link):
        final_target = os.path.basename(os.readlink(final_link))

    keep_last = int(os.environ.get("SNAPSHOT_KEEP_LAST", "1") or 0)
    if keep_last > 0:
        kept = sorted(
            name for name in os.listdir(snaps_root) if name.startswith("step_")
        )
        protected = {n for n in kept[-keep_last:]}
        if final_target:
            protected.add(final_target)
        for stale in [n for n in kept if n not in protected]:
            stale_dir = os.path.join(snaps_root, stale)
            try:
                shutil.rmtree(stale_dir)
                print(f"[Checkpoint] pruned old snapshot {stale}", flush=True)
            except OSError as exc:
                print(f"[Checkpoint] could not prune {stale}: {exc}", flush=True)
        index["steps"] = [e for e in index["steps"] if e["dirname"] in protected]
        with open(os.path.join(snaps_root, "index.json"), "wt", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    layout_path = os.path.join(config.checkpoint_path, "CHECKPOINT_LAYOUT.md")
    # Always refresh so the multi-step rollback story stays visible.
    with open(layout_path, "wt", encoding="utf-8") as f:
        f.write(
            "# Checkpoint layout (multi-step rollback)\n\n"
            "Goal: keep **many** step checkpoints at once so you can roll back to "
            "any previous mid-train moment (e.g. 2500 / 5000 / 7500).\n\n"
            "## Directory tree\n\n"
            "```text\n"
            "<run_dir>/                          # live run root\n"
            "  fsdp2_epoch_*                     # LIVE weights (overwritten as training continues)\n"
            "  train_progress.yaml               # LIVE resume cursor\n"
            "  snapshots/\n"
            "    index.json                      # list of all kept steps\n"
            "    step_0007500/                   # FULL ckpt @ 7500  (SNAPSHOT_KEEP_LAST newest kept)\n"
            "  eval_results/\n"
            "    step_0002500_YYYYMMDD_HHMMSS/   # mid-eval metrics for that step\n"
            "```\n\n"
            "## What gets snapshotted\n\n"
            f"- Trigger: every `downstream_eval_interval` step (pause → snapshot → eval → resume).\n"
            "- Each `snapshots/step_NNNNNNN/` contains: `fsdp2_epoch_*`, all `carry_epoch_*.pt`, "
            "all `rng.*.pt`, `train_progress.yaml`, configs.\n"
            "- `SNAPSHOT_KEEP_LAST` (default 1) prunes older snapshots after each write; "
            "set it to 0 to keep every step.\n\n"
            "## How to resume / roll back\n\n"
            "- Continue from latest live state:\n"
            "  `RESUME_FROM=<run_dir>`\n"
            "- Roll back to a past step (example 5000):\n"
            "  `RESUME_FROM=<run_dir>/snapshots/step_0005000`\n"
        )
    print(
        f"[Checkpoint] Snapshot kept at {snap_dir} "
        f"(rollback: RESUME_FROM={snap_dir})",
        flush=True,
    )
    if dist.is_initialized():
        dist.barrier()


def _checkpointing_disabled() -> bool:
    """Benchmark / smoke runs should not leave a ~19 GB checkpoint behind."""
    return os.environ.get("SKIP_CHECKPOINTS", "0").lower() in ("1", "true", "yes")


def save_fsdp_checkpoint(
    config: PretrainConfig,
    train_state: TrainState,
    *,
    ckpt_epoch: int,
    resume_loop_epoch: int,
    data_epoch: int,
    sampler_start_index: int,
    rank: int,
) -> None:
    """Save FSDP2 DCP + per-rank carry/RNG + tiny train_progress (all ranks in dcp.save).

    ``ckpt_epoch`` names fsdp2/carry files. ``resume_loop_epoch`` / data cursors say where
    training should continue (O(1) multipack seek; no data replay).
    """
    if _checkpointing_disabled():
        if dist.is_initialized():
            dist.barrier()
        return
    if config.checkpoint_path is None:
        return
    if dist.is_initialized():
        dist.barrier()
    checkpoint_id = os.path.join(config.checkpoint_path, f"fsdp2_epoch_{ckpt_epoch}")
    if rank == 0:
        print(f"[Checkpoint] Saving epoch {ckpt_epoch} to {checkpoint_id}", flush=True)
    # update_lr() stores lr as a 0-dim Tensor; DCP then writes TensorStorageMetadata.
    # Fresh optimizers expect a Python float (BytesStorageMetadata) → resume crash.
    # Normalize to float before save (update_lr will re-tensorize on the next step).
    for pg in train_state.optim.param_groups:
        lr = pg.get("lr")
        if isinstance(lr, torch.Tensor):
            pg["lr"] = float(lr.detach().cpu().item())
    dcp.save(
        {
            "model": train_state.model.state_dict(),
            "optim": get_optimizer_state_dict(train_state.model, train_state.optim),
        },  # pyright: ignore[reportPrivateImportUsage]
        checkpoint_id=checkpoint_id,
    )
    torch.save(train_state.carry, os.path.join(config.checkpoint_path, f"carry_epoch_{ckpt_epoch}.{rank}.pt"))
    save_train_progress(
        config,
        train_state,
        loop_epoch=resume_loop_epoch,
        data_epoch=data_epoch,
        sampler_start_index=sampler_start_index,
        rank=rank,
    )
    if dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print(
            f"[Checkpoint] Saved epoch {ckpt_epoch} "
            f"(step={train_state.step}, resume_loop_epoch={resume_loop_epoch}, "
            f"data_epoch={data_epoch}, sampler_start={sampler_start_index})",
            flush=True,
        )


def load_checkpoint(config: PretrainConfig, train_state: TrainState, rank: int) -> dict:
    """Resume model+optim+step+carry+RNG. Returns progress dict for dataloader seek."""
    progress: dict = {
        "step": 0,
        "loop_epoch": 1,
        "data_epoch": 0,
        "sampler_start_index": 0,
    }
    if config.resume_from is None:
        return progress

    epoch = config.resume_epoch
    if epoch is None:
        ckpt_files = glob(os.path.join(config.resume_from, "fsdp2_epoch_*"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint found in {config.resume_from}")
        epoch = max(int(Path(f).stem.split("_")[-1]) for f in ckpt_files)

    checkpoint_id = os.path.join(config.resume_from, f"fsdp2_epoch_{epoch}")
    print(f"[Resume] Loading model + optimizer from {checkpoint_id}")
    optim_state = get_optimizer_state_dict(train_state.model, train_state.optim)
    # Adam state is lazy: unused dual_axis β_p (etc.) may never appear in the
    # saved optim dict. Fresh get_optimizer_state_dict can still emit those keys
    # → strict DCP load fails with "Missing key ... beta_p_raw.step".
    dcp.load(
        {"model": train_state.model.state_dict(), "optim": optim_state},
        checkpoint_id=checkpoint_id,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )
    set_optimizer_state_dict(train_state.model, train_state.optim, optim_state)

    for param_group in train_state.optim.param_groups:
        param_group["betas"] = (config.beta1, config.beta2)
        gname = str(param_group.get("group_name", "main"))
        if gname == "cycle_lambda" and config.cycle_lambda_weight_decay is not None:
            param_group["weight_decay"] = float(config.cycle_lambda_weight_decay)
        elif gname == "dual_axis_beta" and config.dual_axis_beta_weight_decay is not None:
            param_group["weight_decay"] = float(config.dual_axis_beta_weight_decay)
        elif gname == "main" or "weight_decay" not in param_group:
            param_group["weight_decay"] = config.weight_decay
        if gname == "cycle_lambda" and config.cycle_lambda_lr is not None:
            param_group["peak_lr"] = float(config.cycle_lambda_lr)
        elif gname == "dual_axis_beta" and config.dual_axis_beta_lr is not None:
            param_group["peak_lr"] = float(config.dual_axis_beta_lr)
        elif "peak_lr" not in param_group:
            param_group["peak_lr"] = float(config.lr)
        if "ema" in param_group:
            param_group["ema"] = config.ema

    if config.weights_only_resume_from_ema:
        if not hasattr(train_state.optim, "swap_ema"):
            raise ValueError("weights_only_resume_from_ema requires optimizer=adam_atan2")
        print("[Resume] Swapping EMA into model and resetting optimizer state")
        train_state.optim.swap_ema()
        train_state.optim._init_state()

    progress_file = _progress_path(config.resume_from)
    if os.path.isfile(progress_file):
        with open(progress_file, "rt") as f:
            loaded = yaml.safe_load(f) or {}
        progress["step"] = int(loaded.get("step", 0))
        progress["loop_epoch"] = int(loaded.get("loop_epoch", loaded.get("epoch", epoch)))
        progress["data_epoch"] = int(loaded.get("data_epoch", 0))
        progress["sampler_start_index"] = int(loaded.get("sampler_start_index", 0))
        if progress["step"] > 0:
            train_state.step = progress["step"]
            print(
                f"[Resume] Restored step={progress['step']} loop_epoch={progress['loop_epoch']} "
                f"data_epoch={progress['data_epoch']} sampler_start={progress['sampler_start_index']}"
            )

    carry_path = os.path.join(config.resume_from, f"carry_epoch_{epoch}.{rank}.pt")
    if os.path.isfile(carry_path):
        train_state.carry = torch.load(carry_path, map_location=get_device(), weights_only=False)
        print(f"[Resume] Restored carry from {carry_path}")

    # RNG is restored later — immediately before the training loop — so dataloader /
    # logger setup cannot advance the generator past the saved step boundary.
    print(f"[Resume] Done (RNG restore deferred until train loop).")
    return progress


def save_code_and_config(config: PretrainConfig, train_metadata: V1DatasetMeta):
    if config.checkpoint_path is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    with open(os.path.join(config.checkpoint_path, "all_config.yaml"), "wt") as f:
        yaml.dump(config.model_dump(), f)
    with open(os.path.join(config.checkpoint_path, "train_metadata.yaml"), "wt") as f:
        yaml.dump(train_metadata.model_dump(), f)


def assert_document_layout(dataset_path: str) -> None:
    """Require document Causal LM indices (doc_start/doc_len). Reject QA inst_*/resp_*."""
    root = Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(f"data.path not found: {dataset_path}")

    candidates = sorted(root.glob("epoch_*"))
    if not candidates:
        candidates = [root]

    for cand in candidates:
        if not cand.is_dir():
            continue
        has_doc = (cand / "doc_start.npy").is_file() and (cand / "doc_len.npy").is_file()
        has_qa = (cand / "inst_start.npy").is_file() and (cand / "resp_start.npy").is_file()
        if has_qa:
            raise ValueError(
                f"QA inst_*/resp_* layout under {cand} is no longer supported; "
                "use document Causal LM packs (doc_start.npy / doc_len.npy)."
            )
        if has_doc:
            return

    meta_path = root / "metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text())
        layout = meta.get("layout") or (meta.get("tokenizer_info") or {}).get("layout")
        if layout == "document_causal_lm":
            return

    raise FileNotFoundError(
        f"cannot find document Causal LM layout under {dataset_path}: "
        f"need epoch_*/doc_start.npy and doc_len.npy"
    )


def resolve_data_format(config: PretrainConfig) -> PretrainConfig:
    """Force document Causal LM packing (QA / auto detection removed)."""
    fmt = getattr(config.data, "format", "document")
    if fmt != "document":
        raise ValueError(
            f"Only data.format=document is supported (got {fmt!r}). "
            "QA inst_*/resp_* packing has been removed."
        )
    assert_document_layout(config.data.path)
    return config


def apply_lm_mode(config: PretrainConfig) -> PretrainConfig:
    """Force full causal attention + full-sequence CE (PrefixLM / QA removed)."""
    if config.lm_mode != "causal":
        raise ValueError(
            f"Only lm_mode=causal is supported (got {config.lm_mode!r}). "
            "PrefixLM modes prefix_target/prefix_full have been removed."
        )
    config.data.target_only = False
    config.arch.attn_type = "causal"
    config.lm_mode = "causal"
    return config


def load_synced_config(hydra_config: DictConfig, rank: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**OmegaConf.to_container(hydra_config, resolve=True))  # type: ignore
        config = resolve_data_format(config)
        config = apply_lm_mode(config)

        # Naming
        if config.project_name is None:
            config.project_name = f"{Path(config.data.path).stem.capitalize()}_HLM-torch"
        if config.run_name is None:
            base_name = os.environ.get("MLP_TASK_NAME")
            if base_name is None:
                base_name = config.arch.name.split('@')[-1]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config.run_name = f"{base_name}_{timestamp}"
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    dist.broadcast_object_list(objects, src=0)
    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    WORLD_SIZE = 1
    RANK = 0
    DEVICE_ID = 0

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend=get_dist_backend())

        WORLD_SIZE = dist.get_world_size()
        RANK = dist.get_rank()
        DEVICE_ID = int(os.environ["LOCAL_RANK"])

        set_device(DEVICE_ID)

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK)

    # Seed RNGs to ensure consistency (overwritten by RNG snapshot on resume).
    torch.random.manual_seed(config.seed + RANK)
    random.seed(config.seed + RANK)
    np.random.seed(config.seed + RANK)

    # --- Training
    train_state, train_loader, eval_loader, train_metadata = init_train(config, rank=RANK, world_size=WORLD_SIZE)
    resume_progress = load_checkpoint(config, train_state, rank=RANK)

    # Rebuild dataloader: skip already-seen docs, or reshuffle if cursor is missing.
    if config.resume_from is not None and (
        resume_progress.get("sampler_start_index", 0) > 0
        or resume_progress.get("data_epoch", 0) > 0
        or resume_progress.get("step", 0) > 0
    ):
        have_cursor = (
            int(resume_progress.get("sampler_start_index", 0)) > 0
            or int(resume_progress.get("data_epoch", 0)) > 0
        )
        reshuffle = os.environ.get("RESUME_RESHUFFLE", "1").strip() in ("1", "true", "True", "yes")
        seek_epoch = int(resume_progress.get("data_epoch", 0))
        seek_index = int(resume_progress.get("sampler_start_index", 0))
        if reshuffle and not have_cursor:
            # Data restarted from epoch 0: change seed so we do not replay the prefix.
            config.seed = int(config.seed) + int(resume_progress.get("step", 0))
            seek_epoch = 0
            seek_index = 0
            if RANK == 0:
                print(f"[Resume] no data cursor — reshuffle seed={config.seed}", flush=True)
        seek_epoch, seek_index = apply_resume_data_skip(
            config, train_state, resume_progress, seek_epoch, seek_index, rank=RANK
        )
        try:
            train_loader._iterator = None  # type: ignore[attr-defined]
        except Exception:
            pass
        del train_loader
        train_loader, _ = create_dataloader(
            config,
            train_state.batch_plan.batch_max_length,
            drop_last_batch=True,
            rank=RANK,
            world_size=WORLD_SIZE,
            start_data_epoch=seek_epoch,
            resume_sampler_start_index=seek_index,
        )
        if RANK == 0:
            print(
                f"[Resume] Dataloader seek data_epoch={seek_epoch} "
                f"sampler_start={seek_index} seed={config.seed}",
                flush=True,
            )

    # Progress bar and logger
    progress_bar = None
    tb_logger = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)

        tb_log_dir = os.path.join(config.checkpoint_path, "tensorboard")
        tb_plot_dir = os.path.join(config.checkpoint_path, "tensorboard_plots")
        tb_logger = TensorBoardLogger(
            log_dir=tb_log_dir,
            plot_dir=tb_plot_dir,
        )
        tb_logger.log({"num_params": train_state.num_params}, step=0)
        tb_logger.log({"train/total_samples": train_state.total_samples}, step=0)
        tb_logger.log({"train/total_steps": train_state.total_steps}, step=0)
        print(
            f"[Train] params={train_state.num_params:,}, "
            f"seq_len={train_metadata.max_seq_len}, "
            f"samples={train_state.total_samples:,}, "
            f"steps={train_state.total_steps:,}, "
            f"lr_warmup_steps={config.lr_warmup_steps}, "
            f"global_batch={train_state.batch_plan.samples_per_global_step} samples/step, "
            f"micro_batch={train_state.batch_plan.micro_batch_samples} samples/NPU, "
            f"grad_accum={train_state.batch_plan.grad_accum_steps}, "
            f"samples/npu/step={train_state.batch_plan.samples_per_npu_per_step}, "
            f"eval_interval={config.eval_interval}, "
            f"downstream_eval_interval={config.downstream_eval_interval}, "
            f"eval_token_budget={config.eval_token_budget:,}"
        )
        save_code_and_config(config, train_metadata)

    # Optional H/L attention+MoE activation / gradnorm diagnostics
    module_diag: ModuleDiagCollector | None = None
    _mdi_raw = str(os.environ.get("MODULE_DIAG_INTERVAL", "0") or "0").strip()
    try:
        module_diag_interval = int(float(_mdi_raw)) if _mdi_raw else 0
    except (TypeError, ValueError):
        module_diag_interval = 0
    if str(os.environ.get("MODULE_DIAG_FORCE_OFF", "0")).strip() in ("1", "true", "True", "yes"):
        module_diag_interval = 0
    if module_diag_interval > 0:
        diag_dir = os.path.join(config.checkpoint_path, "module_diag")
        module_diag = ModuleDiagCollector(
            train_state.model,
            interval=module_diag_interval,
            out_dir=diag_dir,
        )
        if RANK == 0:
            print(
                f"[Train] module_diag enabled: interval={module_diag_interval}, "
                f"plot_every={module_diag.plot_every_steps}, "
                f"grad={int(module_diag.collect_grad)}, "
                f"act={int(module_diag.collect_act)}, "
                f"rank={int(module_diag.collect_rank)}, "
                f"layers={len(module_diag.layer_tags)}, out={diag_dir}",
                flush=True,
            )

    expert_logger = None
    expert_every = int(os.environ.get("EXPERT_FREQ_EVERY", "1") or 0)
    if expert_every > 0:
        from utils.expert_freq import ExpertFreqLogger, set_active_expert_logger

        expert_path = os.path.join(config.checkpoint_path, "expert_freq.jsonl")
        expert_logger = ExpertFreqLogger(train_state.model, expert_path, every=expert_every)
        if RANK == 0:
            set_active_expert_logger(expert_logger)
            print(
                f"[Train] expert_freq every={expert_every} → {expert_path}",
                flush=True,
            )

    if RANK == 0 and progress_bar is not None and train_state.step > 0:
        progress_bar.n = train_state.step
        progress_bar.refresh()

    pause_for_downstream_eval = False
    start_loop_epoch = int(resume_progress.get("loop_epoch", 1)) if config.resume_from else 1
    # Cursor of the *next* multipack batch within the current data epoch file.
    data_epoch_cursor = int(resume_progress.get("data_epoch", 0))
    sampler_start_cursor = int(resume_progress.get("sampler_start_index", 0))
    epoch_num_samples = 0  # filled from dataloader batch_info; for epoch-% display


    # Restore RNG at the last moment so setup above cannot desync generators.
    if config.resume_from is not None:
        load_rng_state(config.resume_from, RANK)
        if RANK == 0:
            print("[Resume] RNG restored immediately before train loop", flush=True)

    # Same-weight early-exit sweep (Geiping / looped-LM protocol): no further training.
    _heff_sweep = os.environ.get("EVAL_H_EFF_SWEEP", "").strip()
    if _heff_sweep:
        heff_list = [int(x) for x in _heff_sweep.split(",") if x.strip()]
        if RANK == 0:
            print(f"[eval] H_eff sweep (same weights): {heff_list}", flush=True)
        sweep_losses = run_heff_sweep_eval(
            config, train_state, rank=RANK, world_size=WORLD_SIZE, heff_list=heff_list
        )
        if RANK == 0:
            print("[eval] H_eff sweep summary:", flush=True)
            for h in sorted(sweep_losses):
                print(f"  H_eff={h} loss={sweep_losses[h]:.4f}", flush=True)
            mono = all(
                sweep_losses[a] >= sweep_losses[b]
                for a, b in zip(sorted(sweep_losses)[:-1], sorted(sweep_losses)[1:])
            )
            print(
                f"[eval] deeper_better (loss non-increasing in H_eff)={mono}",
                flush=True,
            )
        if dist.is_initialized():
            dist.barrier()
        return

    # Optional bit-exact smoke / debug (may be unsupported on some NPU kernels).
    if os.environ.get("RESUME_DETERMINISTIC", "0") == "1":
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            if RANK == 0:
                print(f"[Resume] deterministic_algorithms unavailable: {exc}", flush=True)

    # Training Loop
    # When total_steps is fixed, do NOT stop after config.epochs data-passes —
    # small packs finish epochs=4 long before total_steps (mcqzs ~800/3k).
    if config.total_steps > 1:
        max_loop_epochs = 10**9
        if RANK == 0:
            print(
                f"[Train] total_steps={train_state.total_steps} is the stop condition; "
                f"config.epochs={config.epochs} only names checkpoints / legacy budget "
                f"(will cycle data epoch_* files as needed)",
                flush=True,
            )
    else:
        max_loop_epochs = config.epochs

    for epoch in range(start_loop_epoch, max_loop_epochs + 1):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {epoch}")

        # ############ Train Iter
        train_state.model.train()
        grad_accum_steps = train_state.batch_plan.grad_accum_steps
        accum_counter = 0
        lr = 0.0
        train_extra_args: dict = {}
        metrics = None
        moe_step_profile = os.environ.get("MOE_STEP_PROFILE", "0") == "1"
        moe_profile_interval = max(1, int(os.environ.get("MOE_STEP_PROFILE_INTERVAL", "10")))
        moe_profile_fb_ms = 0.0
        moe_profile_opt_ms = 0.0
        moe_profile_lb_ms = 0.0
        moe_profile_step_t0 = 0.0
        collect_diag_this_step = False
        seg_buffer: list[dict[str, Tensor]] = []
        use_seg = _bp_segment_len(train_state.model) > 0

        for batch, batch_info in train_loader:
            if "_resume_data_epoch" in batch_info:
                data_epoch_cursor = int(batch_info["_resume_data_epoch"])
            if "_resume_sampler_start" in batch_info:
                sampler_start_cursor = int(batch_info["_resume_sampler_start"])
            if "_resume_epoch_num_samples" in batch_info:
                epoch_num_samples = int(batch_info["_resume_epoch_num_samples"])
            # Strip resume cursors — they must not enter the model forward.
            model_batch_info = {
                k: v for k, v in batch_info.items()
                if not str(k).startswith("_resume_")
            }
            if accum_counter == 0:
                apply_scheduled_cycle_lambda(train_state)
                optim_step = train_state.step + 1
                saved_step = train_state.step
                train_state.step = optim_step
                lr = update_lr(config, train_state)
                train_extra_args = train_state.model.compute_train_extra_args(train_state)  # pyright: ignore[reportCallIssue]
                train_state.step = saved_step
                train_state.optim.zero_grad()
                if train_state.optim_lambda is not None:
                    train_state.optim_lambda.zero_grad()
                seg_buffer = []
                collect_diag_this_step = bool(module_diag is not None and module_diag.should_collect(optim_step))
                if collect_diag_this_step:
                    assert module_diag is not None
                    module_diag.begin_step()
                if expert_logger is not None and RANK == 0:
                    expert_logger.begin_step()
                if moe_step_profile:
                    synchronize()
                    moe_profile_step_t0 = time.perf_counter()

            packed_batch = batch | {k: wrap_tensor(torch.tensor(v, device="cpu")) for k, v in model_batch_info.items()}
            if use_seg:
                seg_buffer.append(packed_batch)
                accum_counter += 1
                if accum_counter < grad_accum_steps:
                    continue
                metrics = run_segmented_global_step(train_state, seg_buffer, **train_extra_args)
                seg_buffer = []
            else:
                metrics = train_batch(
                    train_state,
                    packed_batch,
                    grad_accum_steps,
                    **train_extra_args,
                )
                accum_counter += 1
                if accum_counter < grad_accum_steps:
                    continue

            diag_grad_metrics: dict[str, float] = {}
            diag_act_metrics: dict[str, float] = {}
            # μP-style λ-space compensation BEFORE grad logging / Adam (non-segmented).
            # Segmented path already compensated (+ stashed jac on train_state).
            jac_m: dict[str, float] = {}
            if not use_seg:
                jac_m = compensate_cycle_lambda_grads_if_enabled(train_state, config)
            else:
                jac_m = dict(getattr(train_state, "_last_lambda_jac", {}) or {})
            if collect_diag_this_step and module_diag is not None:
                if module_diag.collect_grad:
                    diag_grad_metrics = module_diag.collect_gradnorms()
                else:
                    # Always log λ / AttnRes-β grads (cheap) even when full gradnorm is off.
                    diag_grad_metrics = module_diag.collect_scale_param_grads()
                if jac_m:
                    diag_grad_metrics = {**diag_grad_metrics, **jac_m}
                diag_act_metrics = module_diag.finalize_step(train_state.step + 1)
                module_diag.end_step()

            if moe_step_profile:
                synchronize()
                moe_profile_fb_ms = (time.perf_counter() - moe_profile_step_t0) * 1000
                t0 = time.perf_counter()
            if not use_seg:
                _log_lambda_beta_snapshot(train_state, train_state.step + 1, jac_m)
                loss_g = global_mean_loss_from_metrics(metrics)
                step_optimizers(
                    train_state,
                    loss_value=loss_g,
                    step_for_log=train_state.step + 1,
                )
            # Segmented: λβ already logged inside run_segmented_global_step (pre-step).
            # Segmented path already steps (+ compensates) inside run_segmented_global_step.
            if moe_step_profile:
                synchronize()
                moe_profile_opt_ms = (time.perf_counter() - t0) * 1000
            inner_model = getattr(train_state.model, "model", None)
            if inner_model is not None and hasattr(inner_model, "update_moe_load_balance_biases"):
                # Do NOT update expert_bias on skipped spikes — otherwise a skip storm
                # poisons MoE routing while θ is frozen (seen: loss 3.3 → 6 after resume).
                if not getattr(train_state, "_last_skip_reason", None):
                    if moe_step_profile:
                        t0 = time.perf_counter()
                    inner_model.update_moe_load_balance_biases()
                    if moe_step_profile:
                        synchronize()
                        moe_profile_lb_ms = (time.perf_counter() - t0) * 1000
                elif hasattr(inner_model, "discard_moe_load_balance_stats"):
                    inner_model.discard_moe_load_balance_stats()
            train_state.step += 1
            if getattr(train_state, "_step_had_skip", False):
                n_consec = int(getattr(train_state, "_consec_skip_steps", 0)) + 1
                train_state._consec_skip_steps = n_consec  # type: ignore[attr-defined]
                lim = 0
                try:
                    lim = int(getattr(config, "skip_abort_consecutive", 0) or 0)
                except (TypeError, ValueError):
                    lim = 0
                env_lim = os.environ.get("SKIP_ABORT_CONSECUTIVE", "").strip()
                if env_lim.isdigit():
                    lim = int(env_lim)
                if lim > 0 and n_consec >= lim:
                    train_state._abort_training = True  # type: ignore[attr-defined]
                    if RANK == 0:
                        alarm = (
                            f"[SPIKE_ABORT] step={train_state.step} "
                            f"reason=consecutive_skips={n_consec}/{lim} — stopping training NOW"
                        )
                        print(alarm, flush=True)
                        for path in (
                            os.environ.get("SPIKE_SUMMARY_FILE", ""),
                            os.environ.get("SPIKE_ALARM_FILE", ""),
                        ):
                            if not path:
                                continue
                            try:
                                with open(path, "a", encoding="utf-8") as f:
                                    f.write(alarm + "\n")
                            except Exception:
                                pass
            else:
                train_state._consec_skip_steps = 0  # type: ignore[attr-defined]
            train_state._step_had_skip = False  # type: ignore[attr-defined]
            accum_counter = 0
            if expert_logger is not None and RANK == 0:
                try:
                    expert_logger.flush(train_state.step)
                except Exception:
                    pass

            if metrics is not None:
                reduced_metrics = reduce_metrics(metrics, prefix="train/")
                log_payload = reduced_metrics | train_extra_args | {"train/lr": lr}
                gnorm = getattr(train_state, "_last_grad_norm", None)
                if gnorm is not None:
                    log_payload["train/grad_norm"] = float(gnorm)
                skip_reason = getattr(train_state, "_last_skip_reason", None)
                if skip_reason:
                    log_payload["train/skip_step"] = 1.0
                skip_n = getattr(train_state, "_skip_count", 0)
                if skip_n:
                    log_payload["train/skip_count"] = float(skip_n)
                if RANK == 0:
                    progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                    assert tb_logger is not None
                    tb_logger.log(log_payload, step=train_state.step)
                    log_every = max(1, int(getattr(config, "log_interval", 1) or 1))
                    if train_state.step % log_every == 0 or train_state.step >= train_state.total_steps:
                        metric_str = " ".join(
                            f"{name.split('/')[-1]}={value:.4f}" if name != "train/lr" else f"lr={value:.6g}"
                            for name, value in sorted(log_payload.items())
                        )
                        # Cheap epoch tag (integer %). Do NOT mirror metrics into tqdm postfix —
                        # that doubles terminal I/O and was slowing the step loop slightly.
                        if epoch_num_samples > 0:
                            epoch_pct_i = min(100, (100 * int(sampler_start_cursor)) // int(epoch_num_samples))
                        else:
                            epoch_pct_i = 0
                        # ETA from tqdm remaining time when available
                        eta = ""
                        try:
                            fmt = getattr(progress_bar, "format_dict", {}) or {}
                            remaining = fmt.get("remaining")
                            if remaining is not None:
                                eta = f" eta={float(remaining):.0f}s"
                        except Exception:
                            eta = ""
                        # Attach recent λ / act summary if present on train_state
                        extra = ""
                        lam_v = getattr(train_state, "_last_cycle_lambda", None)
                        if lam_v is not None:
                            extra += f" cycle_lambda={float(lam_v):.4f}"
                        an = getattr(train_state, "_last_act_norm_mean", None)
                        av = getattr(train_state, "_last_act_var_mean", None)
                        if an is not None:
                            extra += f" act_norm={float(an):.3f}"
                        if av is not None:
                            extra += f" act_var={float(av):.3f}"
                        print(
                            f"[step {train_state.step}/{train_state.total_steps}] "
                            f"[ep{epoch} {epoch_pct_i}%] {metric_str}{extra}{eta}",
                            flush=True,
                        )
                        metrics_jsonl = os.environ.get("STEP_METRICS_JSONL", "")
                        if metrics_jsonl:
                            import json as _json
                            rec = {
                                "step": int(train_state.step),
                                "lr": float(lr),
                                **{k: float(v) for k, v in reduced_metrics.items()},
                            }
                            with open(metrics_jsonl, "a", encoding="utf-8") as _mf:
                                _mf.write(_json.dumps(rec, sort_keys=True) + "\n")
                    if collect_diag_this_step and module_diag is not None:
                        diag_payload = module_diag.log_and_store(
                            train_state.step,
                            diag_act_metrics,
                            diag_grad_metrics,
                            tb_logger=tb_logger,
                        )
                        # Compact console: means + λ + AttnRes β values + λ/β grads
                        summary = {
                            k.removeprefix("module_diag/"): v
                            for k, v in diag_payload.items()
                            if (
                                k.endswith("_mean")
                                or "cycle_lambda" in k
                                or "attn_res_beta" in k
                                or k.startswith("module_diag/grad/")
                            )
                        }
                        if summary:
                            summary_str = " ".join(f"{k}={v:.4g}" for k, v in sorted(summary.items()))
                            print(f"[module_diag step {train_state.step}] {summary_str}", flush=True)
                            an = summary.get("H/moe/act_norm_mean") or summary.get("H/attn/act_norm_mean")
                            av = summary.get("H/moe/act_var_mean") or summary.get("H/attn/act_var_mean")
                            if an is not None:
                                train_state._last_act_norm_mean = float(an)  # type: ignore[attr-defined]
                            if av is not None:
                                train_state._last_act_var_mean = float(av)  # type: ignore[attr-defined]
                    if moe_step_profile and train_state.step % moe_profile_interval == 0:
                        total_ms = moe_profile_fb_ms + moe_profile_opt_ms + moe_profile_lb_ms
                        cap_msg = ""
                        if os.environ.get("MOE_CAPACITY_PROFILE", "0") == "1":
                            from models.moe import pop_capacity_profile

                            cap_stats = pop_capacity_profile()
                            cap_msg = (
                                f" cap_fast={cap_stats['fast']} cap_slow={cap_stats['slow']}"
                            )
                        print(
                            f"[moe_profile step {train_state.step}] "
                            f"fwd+bwd={moe_profile_fb_ms:.1f}ms "
                            f"optim={moe_profile_opt_ms:.1f}ms "
                            f"lb_update={moe_profile_lb_ms:.1f}ms "
                            f"sum={total_ms:.1f}ms "
                            f"world_size={WORLD_SIZE}{cap_msg}",
                            flush=True,
                        )

            collect_diag_this_step = False

            if RANK == 0 and tb_logger is not None and train_state.step % config.tensorboard_plot_interval == 0:
                tb_logger.save_plots()

            if (
                eval_loader is not None
                and config.eval_interval > 0
                and train_state.step % config.eval_interval == 0
            ):
                if dist.is_initialized():
                    dist.barrier()
                eval_metrics = run_eval(config, train_state, eval_loader)
                if RANK == 0:
                    assert tb_logger is not None
                    tb_logger.log(eval_metrics, step=train_state.step)
                    tb_logger.save_plots()
                    print(
                        f"[eval step {train_state.step}/{train_state.total_steps}] "
                        f"loss={eval_metrics['eval/loss']:.4f} "
                        f"tokens={eval_metrics['eval/tokens']:.0f}",
                        flush=True,
                    )

            # Optional per-step barrier for rank alignment; default 0 for throughput.
            barrier_interval = int(os.environ.get("BARRIER_INTERVAL", "0"))
            if dist.is_initialized() and barrier_interval > 0 and train_state.step % barrier_interval == 0:
                dist.barrier()

            del metrics

            if getattr(train_state, "_abort_training", False):
                if RANK == 0:
                    print("[SPIKE_ABORT] exiting train loop after spike alarm", flush=True)
                break

            if train_state.step >= train_state.total_steps:
                break

            # Periodic step checkpoints (immutable snapshots/step_* for resume/rollback).
            ckpt_every = int(getattr(config, "checkpoint_every_steps", 0) or 0)
            if (
                config.checkpoint_path is not None
                and ckpt_every > 0
                and train_state.step > 0
                and train_state.step % ckpt_every == 0
                and train_state.step < train_state.total_steps
            ):
                save_fsdp_checkpoint(
                    config,
                    train_state,
                    ckpt_epoch=epoch,
                    resume_loop_epoch=epoch,
                    data_epoch=data_epoch_cursor,
                    sampler_start_index=sampler_start_cursor,
                    rank=RANK,
                )
                # The live fsdp2_epoch_* above is enough to resume; immutable
                # snapshots are 19 GB each, so keep them on their own (coarser)
                # interval. SNAPSHOT_EVERY_STEPS=0 disables them entirely.
                snap_every = int(os.environ.get("SNAPSHOT_EVERY_STEPS", ckpt_every) or 0)
                if snap_every > 0 and train_state.step % snap_every == 0:
                    snapshot_checkpoint_at_step(
                        config, train_state, ckpt_epoch=epoch, rank=RANK
                    )

            # Mid-train downstream eval: save + pause (launcher runs 8-NPU eval, then resumes).
            # Only pause when step < total_steps so the final step is handled by normal exit + POST_TRAIN_EVAL.
            if (
                config.downstream_eval_interval > 0
                and train_state.step > 0
                and train_state.step % config.downstream_eval_interval == 0
                and train_state.step < train_state.total_steps
            ):
                save_fsdp_checkpoint(
                    config,
                    train_state,
                    ckpt_epoch=epoch,
                    resume_loop_epoch=epoch,
                    data_epoch=data_epoch_cursor,
                    sampler_start_index=sampler_start_cursor,
                    rank=RANK,
                )
                # Keep a durable copy at this step (live fsdp2_epoch_* is overwritten later).
                snapshot_checkpoint_at_step(
                    config, train_state, ckpt_epoch=epoch, rank=RANK
                )
                write_pending_downstream_eval(config, train_state, loop_epoch=epoch, rank=RANK)
                if dist.is_initialized():
                    dist.barrier()
                pause_for_downstream_eval = True
                break

        if pause_for_downstream_eval:
            break

        if (
            eval_loader is not None
            and config.eval_interval > 0
            and train_state.step > 0
            and train_state.step >= train_state.total_steps
            and train_state.step % config.eval_interval != 0
        ):
            if dist.is_initialized():
                dist.barrier()
            eval_metrics = run_final_eval(config, train_state, rank=RANK, world_size=WORLD_SIZE)
            if RANK == 0:
                if tb_logger is not None:
                    tb_logger.log(eval_metrics, step=train_state.step)
                    tb_logger.save_plots()
                print(
                    f"[eval final step {train_state.step}/{train_state.total_steps}] "
                    f"loss={eval_metrics['eval/loss']:.4f} "
                    f"tokens={eval_metrics['eval/tokens']:.0f}",
                    flush=True,
                )
            if dist.is_initialized():
                dist.barrier()

        ############ Checkpointing
        if config.checkpoint_path is not None and (
            (epoch % config.checkpoint_interval == 0)
            or (epoch == config.epochs)
            or (train_state.step >= train_state.total_steps)
        ):
            # End of data-epoch: next resume starts the following loop epoch / next epoch_* file.
            next_loop = epoch if train_state.step >= train_state.total_steps else epoch + 1
            next_data = data_epoch_cursor if train_state.step >= train_state.total_steps else epoch
            next_sampler = sampler_start_cursor if train_state.step >= train_state.total_steps else 0
            save_fsdp_checkpoint(
                config,
                train_state,
                ckpt_epoch=epoch,
                resume_loop_epoch=next_loop,
                data_epoch=next_data,
                sampler_start_index=next_sampler,
                rank=RANK,
            )
            if train_state.step >= train_state.total_steps:
                # Both periodic snapshot sites are gated on step < total_steps, so
                # without this the end-of-training weights exist only in the live
                # fsdp2_epoch_* directory, which resume overwrites. Downstream
                # tasks need an immutable copy.
                snapshot_checkpoint_at_step(
                    config, train_state, ckpt_epoch=epoch, rank=RANK, is_final=True,
                )

        if train_state.step >= train_state.total_steps:
            break
        if getattr(train_state, "_abort_training", False):
            break

    # finalize
    aborted = bool(getattr(train_state, "_abort_training", False))
    if RANK == 0 and tb_logger is not None:
        tb_logger.close()
    if dist.is_initialized():
        dist.destroy_process_group()
    if aborted:
        raise SystemExit(2)


if __name__ == "__main__":
    launch()
