"""Load a LOOM training checkpoint for evaluation.

``CKPT_PATH`` may be
  * a run directory (``all_config.yaml`` + ``train_metadata.yaml`` + ``fsdp2_epoch_N``),
  * a snapshot (``<run>/snapshots/step_NNNNNNN``), or
  * an archive directory that only holds
    ``fsdp2_epoch_N`` and a ``final_meta.json`` whose ``source`` names the run directory
    the configs live in. ``CKPT_CONFIG_DIR`` overrides where configs are read from.

Model code is whatever ``models`` / ``pretrain`` resolve to on ``PYTHONPATH``
(``run.sh`` sets it from ``MODEL_CODE_ROOT``). Nothing here inserts a tree path,
and ``provenance()`` records which files were actually imported.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed.checkpoint as dcp
import yaml
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase


@dataclass
class EvalCheckpoint:
    model: nn.Module
    tokenizer: PreTrainedTokenizerBase
    config: Any
    info: dict[str, Any] = field(default_factory=dict)


def _resolve_config_dir(ckpt_path: Path) -> Path:
    override = os.environ.get("CKPT_CONFIG_DIR", "").strip()
    candidates = [Path(override)] if override else []
    candidates.append(ckpt_path)
    meta_path = ckpt_path / "final_meta.json"
    if meta_path.is_file():
        source = json.loads(meta_path.read_text()).get("source")
        if source:
            candidates.append(Path(source))
    for cand in candidates:
        if (cand / "all_config.yaml").is_file() and (cand / "train_metadata.yaml").is_file():
            return cand
    raise FileNotFoundError(
        f"no all_config.yaml + train_metadata.yaml in {[str(c) for c in candidates]}; "
        "set CKPT_CONFIG_DIR to the run directory"
    )


def _resolve_epoch(ckpt_path: Path, epoch: Optional[int]) -> int:
    if epoch is not None:
        return int(epoch)
    # Digits only: sibling dirs such as fsdp2_epoch_1.model_only (export staging) are not epochs.
    found = [s for f in glob(str(ckpt_path / "fsdp2_epoch_*")) if (s := Path(f).name.split("_")[-1]).isdigit()]
    if not found:
        raise FileNotFoundError(f"no fsdp2_epoch_<N> in {ckpt_path}")
    return max(int(s) for s in found)


def _step_in(path: Path) -> tuple[Optional[int], dict]:
    if not path.is_file():
        return None, {}
    text = path.read_text()
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        return None, {}
    return (int(data["step"]) if "step" in data else None), data


def _read_step(ckpt_path: Path, config_dir: Path) -> Optional[int]:
    # snapshot_meta.json before train_progress.yaml: snapshot files are hardlinked into
    # pin / anneal-init dirs, and a resume rewriting train_progress.yaml in place there
    # resets the snapshot's copy too (the step-34000 snapshots read step 0 after 09-24).
    # A run dir can also carry a snapshot_meta.json naming itself (model-only export of a
    # crashed run); if that run is later resumed, its train_progress.yaml is the live step.
    progress, _ = _step_in(ckpt_path / "train_progress.yaml")
    snap, meta = _step_in(ckpt_path / "snapshot_meta.json")
    src = meta.get("source_checkpoint_path")
    if snap is not None and progress is not None and snap != progress and src and Path(src).resolve() == ckpt_path:
        raise RuntimeError(
            f"{ckpt_path}: snapshot_meta.json says step {snap}, train_progress.yaml says {progress}; "
            "the run dir moved on after the meta was written, evaluate a snapshot instead"
        )
    for step in (_step_in(ckpt_path / "final_meta.json")[0], snap, progress):
        if step is not None:
            return step
    return None


def _tokenizer_path(train_metadata) -> str:
    override = os.environ.get("EVAL_TOKENIZER_PATH", "").strip()
    if override:
        return override
    info = train_metadata.tokenizer_info or {}
    path = info.get("name_or_path") or info.get("tokenizer_path")
    if not path:
        raise ValueError("train_metadata.yaml has no tokenizer path; set EVAL_TOKENIZER_PATH")
    if str(path).endswith("tokenizer.json"):
        path = str(Path(path).parent)
    return str(path)


def _eval_param_dtype(fwd_bwd_dtype: str) -> torch.dtype:
    raw = os.environ.get("EVAL_PARAM_DTYPE", "").strip() or fwd_bwd_dtype
    aliases = {"bf16": "bfloat16", "fp32": "float32", "fp16": "float16"}
    dtype = getattr(torch, aliases.get(raw, raw), None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"EVAL_PARAM_DTYPE={raw!r} is not a torch dtype")
    return dtype


def set_param_dtype(model: nn.Module, dtype: torch.dtype) -> None:
    """Cast parameters only, as FSDP2 MixedPrecisionPolicy(param_dtype=...) does."""
    for p in model.parameters():
        p.data = p.data.to(dtype)


def load_checkpoint_for_eval(ckpt_path: str, epoch: Optional[int] = None) -> EvalCheckpoint:
    from pretrain import PretrainConfig, V1DatasetMeta
    from utils.device import get_device
    from utils.functions import load_model_class

    ckpt = Path(ckpt_path).resolve()
    config_dir = _resolve_config_dir(ckpt)
    epoch = _resolve_epoch(ckpt, epoch)
    weights = ckpt / f"fsdp2_epoch_{epoch}"

    config = PretrainConfig(**yaml.safe_load((config_dir / "all_config.yaml").read_text()))
    train_metadata = V1DatasetMeta(**yaml.safe_load((config_dir / "train_metadata.yaml").read_text()))

    model_cfg = config.arch.model_dump() | train_metadata.model_dump() | config.data.model_dump()
    model_cls = load_model_class(config.arch.name)
    head_cls = load_model_class(config.arch.head)
    with torch.device(get_device()):
        model: nn.Module = head_cls(model_cls(model_cfg), model_cfg)

    state = model.state_dict()
    ckpt_keys = {
        k[len("model."):]
        for k in dcp.FileSystemReader(str(weights)).read_metadata().state_dict_metadata
        if k.startswith("model.")
    }
    missing = sorted(set(state) - ckpt_keys)
    if missing:
        raise KeyError(f"{len(missing)} model keys absent from {weights}: {missing[:8]}")
    unused = sorted(ckpt_keys - set(state))
    step_before = _read_step(ckpt, config_dir)
    dcp.load({"model": state}, checkpoint_id=str(weights), no_dist=True)
    if _read_step(ckpt, config_dir) != step_before:
        raise RuntimeError(f"{ckpt} advanced during loading (live run dir); evaluate a snapshot instead")

    # DCP holds fp32 masters. Training computes under FSDP2 MixedPrecisionPolicy(param_dtype=bf16):
    # parameters are cast, buffers (RoPE tables, expert_bias) are not. EVAL_PARAM_DTYPE picks
    # the scoring dtype; the holdout check always switches to fwd_bwd_dtype (see training_numerics).
    param_dtype = _eval_param_dtype(config.fwd_bwd_dtype)
    set_param_dtype(model, param_dtype)
    model.eval()

    tok_path = _tokenizer_path(train_metadata)
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    if train_metadata.vocab_size is not None and len(tokenizer) > int(train_metadata.vocab_size):
        raise ValueError(
            f"tokenizer {tok_path} has {len(tokenizer)} tokens > model vocab {train_metadata.vocab_size}"
        )

    info = {
        "ckpt_path": str(ckpt),
        "weights": str(weights),
        "config_dir": str(config_dir),
        "epoch": epoch,
        "step": step_before,
        "run_name": config.run_name,
        "num_loops": config.arch.model_dump().get("num_loops"),
        "arch": config.arch.name,
        "fwd_dtype": config.fwd_bwd_dtype,
        "param_dtype": str(param_dtype).removeprefix("torch."),
        "tokenizer_path": tok_path,
        "tokenizer_len": len(tokenizer),
        "unused_ckpt_keys": unused,
        "num_params": sum(p.numel() for p in model.parameters()),
        # Already the per-sample supervised length pretrain.py packs to (file max - 1).
        "max_seq_len": int(train_metadata.max_seq_len),
    }
    return EvalCheckpoint(model=model, tokenizer=tokenizer, config=config, info=info)


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=30, check=True
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<git failed: {exc}>"


def provenance(arch_name: str) -> dict[str, Any]:
    """Which files were really imported, and which commit they came from."""
    import models.lm_head
    import models.moe
    import models.prefixlm_attention
    import pretrain
    from utils.functions import load_model_class

    arch_mod = sys.modules[load_model_class(arch_name).__module__]
    files = {
        "pretrain": pretrain.__file__,
        "arch": arch_mod.__file__,
        "moe": models.moe.__file__,
        "prefixlm_attention": models.prefixlm_attention.__file__,
        "lm_head": models.lm_head.__file__,
    }
    code_root = Path(pretrain.__file__).resolve().parent
    expected = os.environ.get("MODEL_CODE_ROOT", "").strip()
    if expected and Path(expected).resolve() != code_root:
        raise RuntimeError(f"MODEL_CODE_ROOT={expected} but pretrain was imported from {code_root}")
    dirty = _git(code_root, "status", "--porcelain", "--", "models", "utils", "pretrain.py")
    return {
        "code_root": str(code_root),
        "commit": _git(code_root, "rev-parse", "HEAD"),
        "model_code_dirty": bool(dirty) if not dirty.startswith("<git failed") else dirty,
        "files": {k: str(Path(v).resolve()) for k, v in files.items()},
        "env": {
            k: os.environ.get(k)
            for k in (
                "PREFIXLM_ATTN_BACKEND",
                "MOE_BACKEND",
                "MOE_ROUTING",
                "MOE_SKIP_CAPACITY",
                "MOE_CAPACITY_IMPL",
            )
        },
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
