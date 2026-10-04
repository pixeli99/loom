"""Per-step MoE expert call frequency (layer × loop). Rank0 JSONL, no collectives."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

from torch import Tensor, nn

from models.transformer import TransformerBlock

_ACTIVE: Optional["ExpertFreqLogger"] = None


def get_active_expert_logger() -> Optional["ExpertFreqLogger"]:
    return _ACTIVE


def set_active_expert_logger(logger: Optional["ExpertFreqLogger"]) -> None:
    global _ACTIVE
    _ACTIVE = logger


class ExpertFreqLogger:
    """Accumulate routing counts during forward; flush one JSONL line per step."""

    def __init__(self, model: nn.Module, out_path: str | Path, *, every: int = 1) -> None:
        self.every = max(1, int(every))
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._moe_tag_by_id: dict[int, str] = {}
        self._sum: dict[tuple[str, str, int], float] = defaultdict(float)
        self._cycle = ""
        self._discover(model)

    def _discover(self, model: nn.Module) -> None:
        backbone = getattr(model, "model", model)
        for level_name, short in (("H_level", "H"), ("L_level", "L")):
            level = getattr(backbone, level_name, None)
            if level is None:
                continue
            core = getattr(level, "core", None)
            if core is None or not hasattr(core, "layers"):
                continue
            for i, block in enumerate(core.layers):
                if not isinstance(block, TransformerBlock):
                    continue
                if getattr(block, "ffn_type", "") == "moe":
                    self._moe_tag_by_id[id(block.mlp)] = f"{short}/L{i:02d}"
        if self._moe_tag_by_id:
            return
        layers = getattr(backbone, "layers", None)
        if layers is None:
            return
        for i, block in enumerate(layers):
            if not isinstance(block, TransformerBlock):
                continue
            if getattr(block, "ffn_type", "") == "moe":
                self._moe_tag_by_id[id(block.mlp)] = f"H/L{i:02d}"

    def begin_step(self) -> None:
        self._sum.clear()
        self._cycle = ""

    def set_cycle(self, level: str, idx: int) -> None:
        self._cycle = f"c{level}{int(idx):02d}"

    def record(self, moe_module: nn.Module, load_counts: Tensor) -> None:
        tag = self._moe_tag_by_id.get(id(moe_module))
        if tag is None:
            return
        counts = load_counts.detach()
        if counts.device.type != "cpu":
            counts = counts.float().cpu()
        else:
            counts = counts.float()
        cyc = self._cycle
        for eid in range(int(counts.numel())):
            self._sum[(tag, cyc, eid)] += float(counts[eid].item())

    def should_flush(self, step: int) -> bool:
        return step > 0 and step % self.every == 0

    def flush(self, step: int) -> None:
        if not self.should_flush(step) or not self._sum:
            self._sum.clear()
            return
        layers: dict[str, dict[str, list[float]]] = {}
        n_exp = 0
        for (tag, cyc, eid), v in self._sum.items():
            n_exp = max(n_exp, eid + 1)
            bucket = layers.setdefault(tag, {})
            row = bucket.setdefault(cyc or "cH00", [])
            while len(row) <= eid:
                row.append(0.0)
            row[eid] += float(v)
        rec = {"step": int(step), "n_experts": n_exp, "layers": layers}
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._sum.clear()
