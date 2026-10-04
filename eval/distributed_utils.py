"""Distributed helpers for multi-GPU evaluation."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import TypeVar

import torch.distributed as dist

from utils.device import get_dist_backend, set_device

T = TypeVar("T")


def init_eval_distributed() -> tuple[int, int, bool]:
    """Initialize HCCL if launched via torchrun. Returns (rank, world_size, is_main)."""
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, True

    if not dist.is_initialized():
        dist.init_process_group(backend=get_dist_backend())

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    set_device(int(os.environ["LOCAL_RANK"]))
    return rank, world_size, rank == 0


def prep_done_flag() -> Path:
    """Shared-filesystem flag for multi-node; /tmp is only OK for single-node."""
    port = os.environ.get("MASTER_PORT", "0")
    base = os.environ.get("EVAL_PREP_FLAG_DIR", "").strip()
    if base:
        path = Path(base)
        path.mkdir(parents=True, exist_ok=True)
        return path / f"loom_eval_{port}_prep.done"
    return Path(tempfile.gettempdir()) / f"loom_eval_{port}_prep.done"


def clear_prep_done_flag() -> None:
    flag = prep_done_flag()
    if flag.exists():
        flag.unlink()


def signal_prep_done() -> None:
    prep_done_flag().touch()


def wait_for_prep_done(rank: int, world_size: int) -> None:
    """Let rank 0 load datasets without holding HCCL collectives open."""
    if world_size <= 1 or rank == 0:
        return
    flag = prep_done_flag()
    while not flag.exists():
        time.sleep(0.5)


def shard_by_rank(items: list[T], rank: int, world_size: int) -> list[tuple[int, T]]:
    return [(i, item) for i, item in enumerate(items) if i % world_size == rank]


def shard_by_rank_balanced(
    items: list[T],
    rank: int,
    world_size: int,
    weight_fn=None,
) -> list[tuple[int, T]]:
    """Greedy load-balanced sharding by estimated workload (default: prompt char length)."""
    if world_size <= 1:
        return list(enumerate(items))

    weights = [float(weight_fn(item)) if weight_fn is not None else 1.0 for item in items]
    order = sorted(range(len(items)), key=lambda i: weights[i], reverse=True)

    rank_loads = [0.0] * world_size
    rank_indices: list[list[int]] = [[] for _ in range(world_size)]
    for idx in order:
        target = min(range(world_size), key=lambda r: rank_loads[r])
        rank_indices[target].append(idx)
        rank_loads[target] += weights[idx]

    return [(idx, items[idx]) for idx in rank_indices[rank]]


def gather_shard_balance_stats(
    prompts: list[str],
    rank: int,
    world_size: int,
    *,
    char_weights: list[float] | None = None,
    token_weights: list[float] | None = None,
) -> list[dict[str, float | int]]:
    """Per-rank sample count and estimated workload for modulo vs balanced sharding."""
    if char_weights is None:
        char_weights = [float(len(p)) for p in prompts]
    if token_weights is None:
        token_weights = char_weights

    def _balanced_indices(weights: list[float]) -> list[list[int]]:
        order = sorted(range(len(prompts)), key=lambda i: weights[i], reverse=True)
        rank_loads = [0.0] * world_size
        rank_indices: list[list[int]] = [[] for _ in range(world_size)]
        for idx in order:
            target = min(range(world_size), key=lambda r: rank_loads[r])
            rank_indices[target].append(idx)
            rank_loads[target] += weights[idx]
        return rank_indices

    def _stats(indices: list[int], weights: list[float]) -> dict[str, float | int]:
        ws = [weights[i] for i in indices]
        return {
            "count": len(indices),
            "weight_sum": int(sum(ws)),
            "weight_max": int(max(ws) if ws else 0),
            "weight_avg": round(sum(ws) / len(ws), 1) if ws else 0.0,
        }

    modulo_indices = [i for i in range(len(prompts)) if i % world_size == rank]
    char_indices = _balanced_indices(char_weights)
    token_indices = _balanced_indices(token_weights)

    local = {
        "modulo": _stats(modulo_indices, char_weights),
        "balanced_chars": _stats(char_indices[rank], char_weights),
        "balanced_tokens": _stats(token_indices[rank], token_weights),
        # Legacy alias for callers expecting "balanced"
        "balanced": _stats(token_indices[rank], token_weights),
    }

    if not dist.is_initialized() or world_size == 1:
        return [local]

    gathered: list[dict | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return gathered  # type: ignore[return-value]


def gather_indexed_results(local_results: list[tuple[int, str]], total_size: int) -> list[str]:
    """All-gather (idx, text) pairs from every rank and rebuild the full ordered list."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        outputs = [""] * total_size
        for idx, text in local_results:
            outputs[idx] = text
        return outputs

    gathered: list[list[tuple[int, str]] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_results)

    outputs = [""] * total_size
    for rank_items in gathered:
        for idx, text in rank_items:
            outputs[idx] = text
    return outputs


def broadcast_from_main(obj: T | None, rank: int) -> T:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return obj  # type: ignore[return-value]
    container: list[T | None] = [obj] if rank == 0 else [None]
    dist.broadcast_object_list(container, src=0)
    return container[0]  # type: ignore[return-value]


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()
