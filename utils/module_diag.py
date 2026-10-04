"""Per-layer attention / MoE(MLP) activation + gradnorm diagnostics for HRM H/L stacks.

Tracks:
  - Aggregate (mean over H/L recurrent cycles): H/L × attn/ffn × layer × {grad_norm, act_norm, act_var}
  - Per-cycle activations: same keys with ``/cH00`` / ``/cL02`` cycle suffix

Plots emphasize differences across layers, recurrent cycles, and H vs L modules.
"""

from __future__ import annotations

import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch import Tensor, nn

from models.transformer import TransformerBlock

# Set by ModuleDiagCollector while enabled; TransformerBlock reads this.
_ACTIVE_COLLECTOR: Optional["ModuleDiagCollector"] = None

# Consistent colors for H/attn, H/ffn, L/attn, L/ffn (matches summary_means style)
_MODULE_COLORS = {
    "H/attn": "#1f77b4",
    "H/moe": "#ff7f0e",
    "H/mlp": "#ff7f0e",
    "L/attn": "#2ca02c",
    "L/moe": "#d62728",
    "L/mlp": "#d62728",
}

# Plots only use activations by default (gradnorm collect/plot is expensive).
_ACT_METRICS: tuple[tuple[str, str], ...] = (
    ("act_norm", "Act token-L2"),
    ("act_var", "Act variance"),
)
_PLOT_DPI = 100


def get_active_collector() -> Optional["ModuleDiagCollector"]:
    return _ACTIVE_COLLECTOR


class ModuleDiagCollector:
    """Collect post-attn / post-moe activations and submodule grad norms every N steps."""

    def __init__(
        self,
        model: nn.Module,
        *,
        interval: int = 10,
        out_dir: str | Path,
        ffn_tag: str = "moe",
    ) -> None:
        self.interval = max(0, int(interval))  # 0 = disabled
        self.out_dir = Path(out_dir)
        self.plot_dir = self.out_dir / "plots"
        self.excel_path = self.out_dir / "module_diag.xlsx"
        self.csv_path = self.out_dir / "module_diag.csv"
        self.ffn_tag = ffn_tag
        self.enabled = False
        self._block_tags: dict[int, str] = {}
        self._attn_modules: dict[str, nn.Module] = {}
        self._ffn_modules: dict[str, nn.Module] = {}
        # (tag, stage, cycle, stat) -> sum; cycle "" = aggregate over all cycles
        self._act_sum: dict[tuple[str, str, str, str], float] = defaultdict(float)
        self._act_count: dict[tuple[str, str, str], int] = defaultdict(int)
        self._cycle_level: Optional[str] = None  # "H" or "L"
        self._cycle_idx: Optional[int] = None
        self._seen_cycles: set[str] = set()
        self._rows: list[dict[str, Any]] = []
        self._history: dict[str, list[tuple[int, float]]] = defaultdict(list)
        # MoE expert routing: (tag, cycle, expert_id) -> token count sum this step
        self._moe_tag_by_id: dict[int, str] = {}
        self._moe_load_sum: dict[tuple[str, str, int], float] = defaultdict(float)
        # Rank-collapse metrics: (tag, stage, cycle, stat) -> sum
        self._rank_sum: dict[tuple[str, str, str, str], float] = defaultdict(float)
        self._rank_count: dict[tuple[str, str, str], int] = defaultdict(int)
        self.collect_rank = os.environ.get("MODULE_DIAG_RANK", "1") == "1"
        # Gradnorm allreduce is costly; off unless MODULE_DIAG_GRAD=1.
        self.collect_grad = os.environ.get("MODULE_DIAG_GRAD", "0") == "1"
        # Act/norm allreduce can hang on H≥9 if ranks disagree on cycle keys.
        # MODULE_DIAG_ACT=0 → only λ/β (+ optional scale grads), skip act tensors.
        self.collect_act = os.environ.get("MODULE_DIAG_ACT", "1") == "1"
        # PNG redraw period (train steps). Default = 5 × MODULE_DIAG_INTERVAL.
        _plot_every = os.environ.get("MODULE_DIAG_PLOT_EVERY", "").strip()
        self.plot_every_steps = (
            max(1, int(_plot_every)) if _plot_every else self.interval * 5
        )

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._discover(model)
        # Learnable cycle λ (softplus) owner — LoopedTransformer.get_cycle_lambda_metrics
        self._lambda_owner: Any | None = None
        self._beta_owners: list[Any] = []
        for m in model.modules():
            if self._lambda_owner is None and hasattr(m, "get_cycle_lambda_metrics"):
                self._lambda_owner = m
            if hasattr(m, "get_attn_res_beta_metrics"):
                self._beta_owners.append(m)
        self._load_existing_csv()
        backbone = getattr(model, "model", model)
        for level_name in ("H_level", "L_level"):
            level = getattr(backbone, level_name, None)
            if level is None:
                continue
            core = getattr(level, "core", None)
            if core is None or not getattr(core, "layers", None):
                continue
            block0 = core.layers[0]
            if isinstance(block0, TransformerBlock):
                self.ffn_tag = "moe" if getattr(block0, "ffn_type", "") == "moe" else "mlp"
            break

    def _load_existing_csv(self) -> None:
        """Resume-safe: reload prior module_diag.csv so plots/CSV stay continuous."""
        if not self.csv_path.is_file():
            return
        import csv

        try:
            with open(self.csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows: list[dict[str, Any]] = []
                for raw in reader:
                    row: dict[str, Any] = {}
                    for k, v in raw.items():
                        if k is None:
                            continue
                        if k == "step":
                            row[k] = int(float(v)) if v not in ("", None) else 0
                        else:
                            try:
                                row[k] = float(v) if v not in ("", None) else float("nan")
                            except ValueError:
                                continue
                    if "step" in row:
                        rows.append(row)
            if not rows:
                return
            self._rows = rows
            self._history = defaultdict(list)
            for row in rows:
                step = int(row["step"])
                for k, v in row.items():
                    if k == "step" or not isinstance(v, (int, float)):
                        continue
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        continue
                    self._history[k].append((step, float(v)))
            print(
                f"[module_diag] Restored {len(self._rows)} rows from {self.csv_path}",
                flush=True,
            )
        except Exception as exc:
            print(f"[module_diag] WARN: failed to load {self.csv_path}: {exc}", flush=True)

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
                tag = f"{short}/L{i:02d}"
                self._block_tags[id(block)] = tag
                self._attn_modules[tag] = block.attn
                self._ffn_modules[tag] = block.mlp
                if getattr(block, "ffn_type", "") == "moe":
                    self._moe_tag_by_id[id(block.mlp)] = tag

        # Looped / plain Transformer: layers live on backbone (no H_level/L_level).
        if not self._attn_modules:
            layers = getattr(backbone, "layers", None)
            if layers is not None:
                for i, block in enumerate(layers):
                    if not isinstance(block, TransformerBlock):
                        continue
                    # Reuse H/* tags so existing norm/var plot styles apply.
                    tag = f"H/L{i:02d}"
                    self._block_tags[id(block)] = tag
                    self._attn_modules[tag] = block.attn
                    self._ffn_modules[tag] = block.mlp
                    if getattr(block, "ffn_type", "") == "moe":
                        self._moe_tag_by_id[id(block.mlp)] = tag
                    if i == 0:
                        self.ffn_tag = (
                            "moe" if getattr(block, "ffn_type", "") == "moe" else "mlp"
                        )

    @property
    def layer_tags(self) -> list[str]:
        return sorted(self._attn_modules.keys(), key=lambda t: (t[0], t))

    def should_collect(self, next_step: int) -> bool:
        return self.interval > 0 and next_step > 0 and next_step % self.interval == 0

    def begin_step(self) -> None:
        global _ACTIVE_COLLECTOR
        # When collect_act=False, hooks stay off — still finalize λ/β + scale grads.
        self.enabled = bool(self.collect_act)
        self._act_sum.clear()
        self._act_count.clear()
        self._seen_cycles.clear()
        self._moe_load_sum.clear()
        self._rank_sum.clear()
        self._rank_count.clear()
        self._cycle_level = None
        self._cycle_idx = None
        _ACTIVE_COLLECTOR = self

    def end_step(self) -> None:
        global _ACTIVE_COLLECTOR
        self.enabled = False
        self._cycle_level = None
        self._cycle_idx = None
        if _ACTIVE_COLLECTOR is self:
            _ACTIVE_COLLECTOR = None

    def set_cycle(self, level: str, idx: int) -> None:
        """Mark current H/L recurrent cycle (called from HRM forward)."""
        if not self.enabled:
            return
        self._cycle_level = level
        self._cycle_idx = int(idx)
        self._seen_cycles.add(f"{level}{int(idx):02d}")

    def clear_cycle(self) -> None:
        self._cycle_level = None
        self._cycle_idx = None

    def record_activation(self, block: nn.Module, stage: str, tensor: Tensor) -> None:
        """stage: 'attn' or 'moe'/'mlp'. Records mean token-L2 and element variance."""
        if not self.enabled:
            return
        tag = self._block_tags.get(id(block))
        if tag is None:
            return
        t = tensor.detach().float()
        token_l2 = torch.linalg.vector_norm(t, ord=2, dim=-1).mean()
        var = t.var(unbiased=False)
        key_stage = stage if stage != "mlp" else self.ffn_tag
        norm_v = float(token_l2.item())
        var_v = float(var.item())

        # Aggregate over all cycles
        self._act_sum[(tag, key_stage, "", "norm")] += norm_v
        self._act_sum[(tag, key_stage, "", "var")] += var_v
        self._act_count[(tag, key_stage, "")] += 1

        # Per-cycle (when HRM marks cycle)
        if self._cycle_level is not None and self._cycle_idx is not None:
            cyc = f"{self._cycle_level}{self._cycle_idx:02d}"
            self._act_sum[(tag, key_stage, cyc, "norm")] += norm_v
            self._act_sum[(tag, key_stage, cyc, "var")] += var_v
            self._act_count[(tag, key_stage, cyc)] += 1

        if self.collect_rank and stage == "attn":
            self._record_rank_collapse(tag, t)

    def record_moe_expert_load(self, moe_module: nn.Module, load_counts: Tensor) -> None:
        """Per-layer per-loop expert token counts (MoE routing diagnostic)."""
        if not self.enabled:
            return
        tag = self._moe_tag_by_id.get(id(moe_module))
        if tag is None:
            return
        cyc = ""
        if self._cycle_level is not None and self._cycle_idx is not None:
            cyc = f"c{self._cycle_level}{self._cycle_idx:02d}"
        counts = load_counts.detach().float().cpu()
        for eid in range(counts.numel()):
            self._moe_load_sum[(tag, cyc, int(eid))] += float(counts[eid].item())

    def _record_rank_collapse(self, tag: str, tensor: Tensor) -> None:
        """Effective rank + top-1 SV ratio (rank-collapse diagnostic). Rank0 only."""
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
        t = tensor.detach().float()
        if t.dim() == 3:
            t = t.reshape(-1, t.size(-1))
        if t.size(0) < 4 or t.size(-1) < 2:
            return
        n = min(64, int(t.size(0)))
        idx = torch.randint(0, int(t.size(0)), (n,), device=t.device)
        x = t[idx] - t[idx].mean(dim=0, keepdim=True)
        try:
            s = torch.linalg.svdvals(x)
        except Exception:
            return
        s = s.clamp_min(1e-9)
        p = s / s.sum()
        eff = float(torch.exp(-(p * p.log()).sum()).item())
        top1 = float((s[0] / s.sum()).item())
        # Mean pairwise cosine (high → representation collapse)
        xn = torch.nn.functional.normalize(x, dim=-1)
        gram = xn @ xn.T
        off = gram[~torch.eye(n, dtype=torch.bool, device=gram.device)]
        mean_cos = float(off.mean().item()) if off.numel() else 0.0
        cyc = ""
        if self._cycle_level is not None and self._cycle_idx is not None:
            cyc = f"c{self._cycle_level}{self._cycle_idx:02d}"
        for stat, val in (("eff_rank", eff), ("top1_sv", top1), ("mean_cos", mean_cos)):
            self._rank_sum[(tag, "attn", cyc, stat)] += val
        self._rank_count[(tag, "attn", cyc)] += 1

    @torch.no_grad()
    def collect_gradnorms(self) -> dict[str, float]:
        """Frobenius grad norms for attn / moe params (FSDP-safe via allreduce of ||g||^2)."""
        device = get_diag_device()
        out: dict[str, float] = {}
        for tag in self.layer_tags:
            for kind, module in (("attn", self._attn_modules[tag]), (self.ffn_tag, self._ffn_modules[tag])):
                local_sq = 0.0
                for p in module.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    # Skip DTensor — to_local mid-step hangs Ascend HCCL on H≥9.
                    if type(g).__name__ == "DTensor" or hasattr(g, "to_local"):
                        continue
                    local_sq += float(g.float().pow(2).sum().item())
                sq = torch.tensor(local_sq, device=device, dtype=torch.float32)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(sq, op=dist.ReduceOp.SUM)
                out[f"{tag}/{kind}/grad_norm"] = float(sq.sqrt().item())
        out.update(self.collect_scale_param_grads())
        return out

    @torch.no_grad()
    def collect_scale_param_grads(self) -> dict[str, float]:
        """Grad norms / mean-abs for learnable λ (θ) and AttnRes β logits."""
        device = get_diag_device()
        # Predetermined keys — dynamic per-rank key sets hang Ascend allreduce
        # on the first MODULE_DIAG_INTERVAL step (H≥9 / FSDP shard mismatch).
        buckets: dict[str, float] = {
            "grad/cycle_lambda_raw_sq": 0.0,
            "grad/dual_axis_beta_raw_sq": 0.0,
            "grad/cycle_lambda_raw_abs": 0.0,
            "grad/dual_axis_beta_raw_abs": 0.0,
            "grad/cycle_lambda_raw_n": 0.0,
            "grad/dual_axis_beta_raw_n": 0.0,
            "grad/cycle_lambda_raw_meanabs": 0.0,
            "grad/cycle_lambda_raw_count": 0.0,
            "grad/beta_h_raw_sq": 0.0,
            "grad/beta_h_raw_meanabs": 0.0,
            "grad/beta_h_raw_count": 0.0,
            "grad/beta_l_raw_sq": 0.0,
            "grad/beta_l_raw_meanabs": 0.0,
            "grad/beta_l_raw_count": 0.0,
            "grad/beta_p_raw_sq": 0.0,
            "grad/beta_p_raw_meanabs": 0.0,
            "grad/beta_p_raw_count": 0.0,
            "grad/gate_raw_sq": 0.0,
            "grad/gate_raw_meanabs": 0.0,
            "grad/gate_raw_count": 0.0,
        }
        root = self._model if hasattr(self, "_model") and self._model is not None else None
        if root is None and self._lambda_owner is not None:
            root = self._lambda_owner

        def _local(t: Tensor) -> Tensor | None:
            # Ascend: DTensor.to_local mid-step hangs HCCL on H≥9.
            # Prefer already-materialized _local_tensor (no collective).
            if type(t).__name__ == "DTensor":
                loc = getattr(t, "_local_tensor", None)
                if loc is None or not isinstance(loc, Tensor):
                    return None
                return loc.detach().float()
            g = t.detach()
            if hasattr(g, "to_local") and type(g).__name__ == "DTensor":
                return None
            return g.float()

        if root is not None:
            for name, p in root.named_parameters():
                if p.grad is None:
                    continue
                n = name.replace(".", "_").lower()
                g = _local(p.grad)
                if g is None:
                    continue
                sq = float(g.pow(2).sum().item())
                ab = float(g.abs().sum().item())
                ne = float(g.numel())
                if "cycle_lambda_raw" in n or "inject_lambda_raw" in n:
                    buckets["grad/cycle_lambda_raw_sq"] += sq
                    buckets["grad/cycle_lambda_raw_abs"] += ab
                    buckets["grad/cycle_lambda_raw_n"] += ne
                    buckets["grad/cycle_lambda_raw_meanabs"] += float(g.abs().mean().item())
                    buckets["grad/cycle_lambda_raw_count"] += 1.0
                elif any(tok in n for tok in ("beta_h_raw", "beta_l_raw", "beta_p_raw", "gate_raw")):
                    buckets["grad/dual_axis_beta_raw_sq"] += sq
                    buckets["grad/dual_axis_beta_raw_abs"] += ab
                    buckets["grad/dual_axis_beta_raw_n"] += ne
                    for tok in ("beta_h_raw", "beta_l_raw", "beta_p_raw", "gate_raw"):
                        if tok in n:
                            buckets[f"grad/{tok}_sq"] += sq
                            buckets[f"grad/{tok}_meanabs"] += float(g.abs().mean().item())
                            buckets[f"grad/{tok}_count"] += 1.0

        # Allreduce scalar buckets (fixed schema — never grow keys after this).
        keys = list(buckets.keys())
        vec = torch.tensor([buckets[k] for k in keys], device=device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        out: dict[str, float] = {}
        for i, k in enumerate(keys):
            out[k] = float(vec[i].item())

        def _finalize_norm(prefix: str) -> None:
            sq = out.get(f"{prefix}_sq", 0.0)
            out[f"{prefix}_norm"] = float(sq ** 0.5)
            n = out.get(f"{prefix}_n", 0.0)
            if n > 0:
                out[f"{prefix}_rms"] = float((sq / n) ** 0.5)
                out[f"{prefix}_meanabs"] = float(out.get(f"{prefix}_abs", 0.0) / n)

        _finalize_norm("grad/cycle_lambda_raw")
        _finalize_norm("grad/dual_axis_beta_raw")
        # Average per-tensor meanabs if counted
        for tok in ("cycle_lambda_raw", "beta_h_raw", "beta_l_raw", "beta_p_raw", "gate_raw"):
            c = out.get(f"grad/{tok}_count", 0.0)
            if c > 0 and f"grad/{tok}_meanabs" in out:
                out[f"grad/{tok}_meanabs"] = float(out[f"grad/{tok}_meanabs"] / c)
            if f"grad/{tok}_sq" in out:
                out[f"grad/{tok}_norm"] = float(out[f"grad/{tok}_sq"] ** 0.5)
        return out

    def _activation_means(self) -> dict[str, float]:
        # Predetermined H cycles — never derive length from per-rank _seen_cycles
        # (that disagrees across ranks and hangs Ascend allreduce at first INTERVAL).
        # Prefer NUM_LOOPS (current H). Never use last token of a multi-H list
        # (H_LIST_OVERRIDE="9 6" used to collect only 6 cycles during H=9).
        n_loops = 0
        raw_n = str(os.environ.get("NUM_LOOPS", "") or "").strip()
        if raw_n.isdigit():
            n_loops = int(raw_n)
        if n_loops <= 0:
            raw_h = str(os.environ.get("H_LIST_OVERRIDE", "") or "").strip()
            if raw_h.isdigit():
                n_loops = int(raw_h)
            else:
                nums = [int(p) for p in raw_h.replace(",", " ").split() if p.isdigit()]
                if len(nums) == 1:
                    n_loops = nums[0]
        if n_loops <= 0:
            n_loops = 12
        pred_h = [f"H{i:02d}" for i in range(n_loops)]
        pred_l = sorted(c for c in self._seen_cycles if c.startswith("L"))
        keys = []
        for tag in self.layer_tags:
            level = tag[0]
            extra = pred_h if level == "H" else pred_l
            cycles = [""] + extra
            for stage in ("attn", self.ffn_tag):
                for cyc in cycles:
                    for stat in ("norm", "var"):
                        if cyc:
                            keys.append(f"{tag}/{stage}/c{cyc}/act_{stat}")
                        else:
                            keys.append(f"{tag}/{stage}/act_{stat}")
        local_vals = []
        for key in keys:
            if "/c" in key and "/act_" in key:
                head, act_stat = key.rsplit("/act_", 1)
                tag_stage, cyc_token = head.rsplit("/c", 1)
                tag, stage = tag_stage.rsplit("/", 1)
                cyc = cyc_token
                stat = act_stat
            else:
                tag, stage, act_stat = key.rsplit("/", 2)
                stat = act_stat.removeprefix("act_")
                cyc = ""
            count = self._act_count.get((tag, stage, cyc), 0)
            if count > 0:
                local_vals.append(self._act_sum[(tag, stage, cyc, stat)] / count)
            else:
                local_vals.append(0.0)

        if not (dist.is_available() and dist.is_initialized()):
            out = {k: v for k, v in zip(keys, local_vals)}
        else:
            vals = torch.tensor(local_vals, device=get_diag_device(), dtype=torch.float32)
            dist.all_reduce(vals, op=dist.ReduceOp.AVG)
            out = {k: float(v) for k, v in zip(keys, vals.tolist())}

        # Rank-collapse only (expert freqs live in expert_freq.jsonl — too wide for CSV).
        for (tag, stage, cyc, stat), v in self._rank_sum.items():
            cnt = self._rank_count.get((tag, stage, cyc), 0)
            if cnt > 0:
                out[f"{tag}/{stage}/{cyc}/{stat}"] = float(v / cnt)
        return out

    def finalize_step(self, step: int) -> dict[str, float]:
        """Reduce act metrics for this step. Caller must collect grads before end_step."""
        del step  # reserved for future use
        out: dict[str, float] = {}
        if self.collect_act:
            out.update(self._activation_means())
        if self._lambda_owner is not None:
            try:
                out.update(self._lambda_owner.get_cycle_lambda_metrics())
            except Exception as exc:
                print(f"[module_diag] WARN cycle_lambda metrics: {exc}", flush=True)
        # AttnRes β value trajectory (mean across DualAxisCarry modules if several)
        if self._beta_owners:
            try:
                acc: dict[str, list[float]] = {}
                for owner in self._beta_owners:
                    for k, v in owner.get_attn_res_beta_metrics().items():
                        acc.setdefault(k, []).append(float(v))
                for k, vals in acc.items():
                    out[k] = float(sum(vals) / max(len(vals), 1))
            except Exception as exc:
                print(f"[module_diag] WARN attn_res_beta metrics: {exc}", flush=True)
        return out

    def log_and_store(
        self,
        step: int,
        act_metrics: dict[str, float],
        grad_metrics: dict[str, float],
        tb_logger: Any | None = None,
    ) -> dict[str, float]:
        payload = {f"module_diag/{k}": v for k, v in {**act_metrics, **grad_metrics}.items()}

        # Layer-mean aggregates: H/attn, H/moe, L/attn, L/moe (cycle-averaged acts)
        for level in ("H", "L"):
            for stage in ("attn", self.ffn_tag):
                gn = [
                    grad_metrics[k]
                    for k in grad_metrics
                    if k.startswith(f"{level}/") and f"/{stage}/grad_norm" in k
                ]
                an = [
                    act_metrics[k]
                    for k in act_metrics
                    if k.startswith(f"{level}/")
                    and f"/{stage}/act_norm" in k
                    and "/c" not in k
                ]
                av = [
                    act_metrics[k]
                    for k in act_metrics
                    if k.startswith(f"{level}/")
                    and f"/{stage}/act_var" in k
                    and "/c" not in k
                ]
                if gn:
                    payload[f"module_diag/{level}/{stage}/grad_norm_mean"] = sum(gn) / len(gn)
                if an:
                    payload[f"module_diag/{level}/{stage}/act_norm_mean"] = sum(an) / len(an)
                if av:
                    payload[f"module_diag/{level}/{stage}/act_var_mean"] = sum(av) / len(av)

            # Per-cycle layer means for activations
            cyc_ids = sorted({c for c in self._seen_cycles if c.startswith(level)})
            for cyc in cyc_ids:
                for stage in ("attn", self.ffn_tag):
                    an = [
                        act_metrics[k]
                        for k in act_metrics
                        if k.startswith(f"{level}/")
                        and f"/{stage}/c{cyc}/act_norm" in k
                    ]
                    av = [
                        act_metrics[k]
                        for k in act_metrics
                        if k.startswith(f"{level}/")
                        and f"/{stage}/c{cyc}/act_var" in k
                    ]
                    if an:
                        payload[f"module_diag/{level}/{stage}/c{cyc}/act_norm_mean"] = sum(an) / len(an)
                    if av:
                        payload[f"module_diag/{level}/{stage}/c{cyc}/act_var_mean"] = sum(av) / len(av)

        row: dict[str, Any] = {"step": step}
        for k, v in sorted(payload.items()):
            short = k.removeprefix("module_diag/")
            row[short] = v
            hist = self._history[short]
            if hist and hist[-1][0] == step:
                hist[-1] = (step, float(v))
            else:
                hist.append((step, float(v)))
        # Replace same-step row if present (resume edge cases)
        if self._rows and int(self._rows[-1].get("step", -1)) == step:
            self._rows[-1] = row
        else:
            self._rows.append(row)

        if tb_logger is not None:
            # Only level×stage means (≈8 scalars). Skip per-layer / per-cycle → no TB PNG flood.
            tb_payload = {
                k: v
                for k, v in payload.items()
                if k in {
                    f"module_diag/{lv}/{st}/{m}"
                    for lv in ("H", "L")
                    for st in ("attn", self.ffn_tag)
                    for m in ("act_norm_mean", "act_var_mean", "grad_norm_mean")
                }
            }
            if tb_payload:
                tb_logger.log(tb_payload, step=step)

        self._flush_tables()
        # Never block the train loop with matplotlib on rank0 while other ranks advance —
        # that desyncs NCCL and has crashed Ascend FA after the first diag step.
        # Offline plots: recipes/.../moe_e8_plot_ckpt / rebuild_plots_from_csv.
        if os.environ.get("MODULE_DIAG_LIVE_PLOT", "0") == "1":
            if len(self._rows) == 1 or step % self.plot_every_steps == 0:
                self.save_plots()
        return payload

    def _flush_tables(self) -> None:
        if not self._rows:
            return
        cols: list[str] = ["step"]
        seen = set(cols)
        for row in self._rows:
            for k in row:
                if k not in seen:
                    cols.append(k)
                    seen.add(k)

        import csv

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for row in self._rows:
                writer.writerow({c: row.get(c, "") for c in cols})

        # xlsx rewrite mid-step hung rank0 on H≥9 (other ranks entered HCCL).
        if os.environ.get("MODULE_DIAG_XLSX", "0") == "1":
            try:
                _write_simple_xlsx(self.excel_path, cols, self._rows)
            except Exception:
                pass

    # ------------------------------------------------------------------ plots
    def save_plots(self) -> None:
        if not self._history:
            return
        if self.plot_dir.exists():
            shutil.rmtree(self.plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        cycle_dir = self.plot_dir / "cycles"
        cycle_dir.mkdir(parents=True, exist_ok=True)

        self._plot_summary_means()
        self._plot_layers_profile()
        self._plot_combined_layer_lines()
        self._plot_scale_ratios()
        self._plot_unroll_cycle_stability()
        self._plot_per_cycle_figures(cycle_dir)
        self._plot_hl_by_layer(cycle_dir)
        self._plot_hl_by_cycle(cycle_dir)
        self._plot_cycle_compare(cycle_dir)
        self._plot_cycle_lambda()
        self._plot_attn_res_beta()
        self._plot_scale_param_grads()

    def _plot_attn_res_beta(self) -> None:
        """AttnRes β_h / β_l / β_p value trajectory."""
        keys = sorted(k for k in self._history if k.startswith("attn_res_beta"))
        if not keys:
            return
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for k in keys:
            steps, vals = zip(*self._history[k])
            ax.plot(steps, vals, label=k.replace("attn_res_", ""), linewidth=1.4)
        ax.axhline(0.55, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, label="init≈0.55")
        ax.set_xlabel("step")
        ax.set_ylabel("β")
        ax.set_ylim(0.0, 1.05)
        ax.set_title("AttnRes dual-axis β trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "attn_res_beta_vs_step.png", dpi=_PLOT_DPI)
        plt.close(fig)

    def _plot_cycle_lambda(self) -> None:
        """Learnable λ trajectory (residual / inject / per-layer)."""
        keys = sorted(k for k in self._history if k.startswith("cycle_lambda") and "grad/" not in k)
        if not keys:
            return
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for k in keys:
            steps, vals = zip(*self._history[k])
            ax.plot(steps, vals, label=k, linewidth=1.5)
        ax.axhline(2.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="λ≈2 ref")
        ax.axhline(1.5, color="orange", linestyle=":", linewidth=1.0, alpha=0.6, label="λ_min=1.5")
        ax.set_xlabel("step")
        ax.set_ylabel("λ")
        ax.set_title("Learnable cycle_scale_lambda trajectory")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "cycle_lambda_vs_step.png", dpi=_PLOT_DPI)
        plt.close(fig)

    def _plot_scale_param_grads(self) -> None:
        """λ (θ) and AttnRes β logit gradient norms / mean-|grad|."""
        keys = sorted(
            k
            for k in self._history
            if k.startswith("grad/")
            and (k.endswith("_norm") or k.endswith("_meanabs") or k.endswith("_rms"))
        )
        if not keys:
            return
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharex=True)
        for k in keys:
            steps, vals = zip(*self._history[k])
            ax = axes[0] if "cycle_lambda" in k else axes[1]
            ax.plot(steps, vals, label=k.replace("grad/", ""), linewidth=1.3)
        axes[0].set_title("λ (θ) grads")
        axes[1].set_title("AttnRes β grads")
        for ax in axes:
            ax.set_xlabel("step")
            ax.set_ylabel("grad")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc="best")
        fig.tight_layout()
        fig.savefig(self.plot_dir / "scale_param_grads_vs_step.png", dpi=_PLOT_DPI)
        plt.close(fig)

    def _plot_summary_means(self) -> None:
        """2 rows (act_norm / act_var) × H/attn H/ffn L/attn L/ffn."""
        summary_keys = [
            k
            for k in self._history
            if (k.endswith("act_norm_mean") or k.endswith("act_var_mean")) and "/c" not in k
        ]
        if not summary_keys:
            return
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        groups = [
            ("act_norm_mean", axes[0], "Activation mean token-L2"),
            ("act_var_mean", axes[1], "Activation variance"),
        ]
        for suffix, ax, title in groups:
            for k in sorted(self._history):
                if not k.endswith(suffix) or "/c" in k:
                    continue
                label = k.replace(f"/{suffix}", "")
                steps, vals = zip(*self._history[k])
                color = _MODULE_COLORS.get(label)
                ax.plot(steps, vals, label=label, linewidth=1.6, color=color)
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, ncol=2, loc="best")
        axes[-1].set_xlabel("step")
        fig.suptitle(
            "H/L activation diagnostics (layer means)\nH/attn · H/%s · L/attn · L/%s"
            % (self.ffn_tag, self.ffn_tag)
        )
        fig.tight_layout()
        fig.savefig(self.plot_dir / "summary_means.png", dpi=_PLOT_DPI)
        plt.close(fig)

    def _latest_step_values(self) -> tuple[int, dict[str, float]]:
        if not self._rows:
            return 0, {}
        row = self._rows[-1]
        step = int(row.get("step", 0))
        return step, {k: float(v) for k, v in row.items() if k != "step" and isinstance(v, (int, float))}

    def _layer_indices(self, level: str) -> list[tuple[int, str]]:
        out = []
        for tag in self.layer_tags:
            if not tag.startswith(f"{level}/"):
                continue
            idx = int(tag.split("/L")[1])
            out.append((idx, tag))
        return sorted(out)

    def _plot_layers_profile(self) -> None:
        """columns=layer index, x=step; H+L on same axes; act_norm / act_var only."""
        idx_map: dict[int, dict[str, str]] = defaultdict(dict)
        for level in ("H", "L"):
            for layer_i, tag in self._layer_indices(level):
                idx_map[layer_i][level] = tag
        if not idx_map:
            return
        layer_ids = sorted(idx_map)
        n_cols = len(layer_ids)
        fig, axes = plt.subplots(
            2, n_cols, figsize=(max(10, 2.2 * n_cols), 5.5), sharex=True, squeeze=False
        )
        for c, layer_i in enumerate(layer_ids):
            for r, (metric, ylab) in enumerate(_ACT_METRICS):
                ax = axes[r, c]
                for level in ("H", "L"):
                    tag = idx_map[layer_i].get(level)
                    if tag is None:
                        continue
                    for stage in ("attn", self.ffn_tag):
                        label = f"{level}/{stage}"
                        self._plot_line_if_present(
                            ax,
                            f"{tag}/{stage}/{metric}",
                            label=label if (c == 0 and r == 0) else None,
                            color=_MODULE_COLORS.get(label),
                        )
                if r == 0:
                    ax.set_title(f"L{layer_i:02d}", fontsize=9)
                if c == 0:
                    ax.set_ylabel(ylab, fontsize=8)
                if r == len(_ACT_METRICS) - 1:
                    ax.set_xlabel("step", fontsize=7)
                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=6)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=2)
        fig.suptitle(
            f"Layer activations vs step (columns=layer)\n"
            f"H/attn · H/{self.ffn_tag} · L/attn · L/{self.ffn_tag}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(self.plot_dir / "layers_profile_latest.png", dpi=_PLOT_DPI)
        plt.close(fig)

    def _plot_combined_layer_lines(self) -> None:
        """2 rows (act_norm/act_var) × 4 cols; each subplot = one line per layer vs step."""
        colspec = [(lv, st) for lv in ("H", "L") for st in ("attn", self.ffn_tag)]
        fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
        cmap = plt.get_cmap("tab20")
        for r, (metric, ylab) in enumerate(_ACT_METRICS):
            for c, (level, stage) in enumerate(colspec):
                ax = axes[r, c]
                layer_keys = sorted(
                    [
                        k
                        for k in self._history
                        if k.startswith(f"{level}/L")
                        and f"/{stage}/{metric}" in k
                        and "_mean" not in k
                        and "/c" not in k
                    ]
                )
                if not layer_keys:
                    ax.set_visible(False)
                    continue
                for i, k in enumerate(layer_keys):
                    steps, vals = zip(*self._history[k])
                    ax.plot(steps, vals, linewidth=1.1, label=k.split("/")[1], color=cmap(i % 20))
                if r == 0:
                    ax.set_title(f"{level}/{stage}", fontsize=10)
                if c == 0:
                    ax.set_ylabel(ylab, fontsize=9)
                if r == len(_ACT_METRICS) - 1:
                    ax.set_xlabel("step", fontsize=8)
                ax.grid(True, alpha=0.25)
                if len(layer_keys) <= 10:
                    ax.legend(fontsize=5, ncol=2, loc="best")
        fig.suptitle(
            "H/attn · H/%s · L/attn · L/%s — per-layer activations vs step" % (self.ffn_tag, self.ffn_tag)
        )
        fig.tight_layout()
        fig.savefig(self.plot_dir / "combined_layers_HL.png", dpi=_PLOT_DPI)
        plt.close(fig)

    def _plot_scale_ratios(self) -> None:
        """Paper-style depth/cycle stability ratios (Curse of Depth + Residual Scaling).

        - layer_ratio: L04/L00 (and H04/H00) act_norm & act_var — want ≈1
        - cycle_ratio: cH01/cH00 (and late/early L cycles) — want ≈1
        """
        ffn = self.ffn_tag
        steps_set: set[int] = set()
        for k, series in self._history.items():
            for s, _ in series:
                steps_set.add(int(s))
        if not steps_set:
            return
        steps = sorted(steps_set)

        def series_map(key: str) -> dict[int, float]:
            if key not in self._history:
                return {}
            return {int(s): float(v) for s, v in self._history[key]}

        def ratio_curve(num_k: str, den_k: str) -> tuple[list[int], list[float]]:
            num, den = series_map(num_k), series_map(den_k)
            xs, ys = [], []
            for s in steps:
                if s in num and s in den and den[s] != 0:
                    xs.append(s)
                    ys.append(num[s] / den[s])
            return xs, ys

        fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)

        # (0,0) layer act_norm ratios
        ax = axes[0, 0]
        for label, num, den, color in [
            (f"L L04/L00 {ffn}", f"L/L04/{ffn}/act_norm", f"L/L00/{ffn}/act_norm", "#1f77b4"),
            (f"H L04/L00 {ffn}", f"H/L04/{ffn}/act_norm", f"H/L00/{ffn}/act_norm", "#d62728"),
            ("L L04/L00 attn", "L/L04/attn/act_norm", "L/L00/attn/act_norm", "#2ca02c"),
            ("H L04/L00 attn", "H/L04/attn/act_norm", "H/L00/attn/act_norm", "#ff7f0e"),
        ]:
            xs, ys = ratio_curve(num, den)
            if xs:
                ax.plot(xs, ys, label=label, linewidth=1.6, color=color)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        ax.axhline(1.8, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_ylabel("layer depth ratio (act_norm)")
        ax.set_title("Intra-module depth (Curse of Depth / LNS)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

        # (0,1) layer act_var ratios
        ax = axes[0, 1]
        for label, num, den, color in [
            (f"L L04/L00 {ffn}", f"L/L04/{ffn}/act_var", f"L/L00/{ffn}/act_var", "#1f77b4"),
            (f"H L04/L00 {ffn}", f"H/L04/{ffn}/act_var", f"H/L00/{ffn}/act_var", "#d62728"),
        ]:
            xs, ys = ratio_curve(num, den)
            if xs:
                ax.plot(xs, ys, label=label, linewidth=1.6, color=color)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_ylabel("layer depth ratio (act_var)")
        ax.set_title("Intra-module variance growth")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

        # (1,0) H cycle ratios (last/first — works for H=2 cH01/cH00 and H=3 cH02/cH00)
        ax = axes[1, 0]
        h_cycles = sorted(
            {
                k.split("/")[2]
                for k in self._history
                if k.startswith(f"H/{ffn}/cH") and k.endswith("/act_norm_mean")
            }
        )
        if len(h_cycles) >= 2:
            early, late = h_cycles[0], h_cycles[-1]
            for label, stage, color in [
                (f"H {ffn} {late}/{early}", ffn, "#d62728"),
                (f"H attn {late}/{early}", "attn", "#ff7f0e"),
            ]:
                xs, ys = ratio_curve(
                    f"H/{stage}/{late}/act_norm_mean",
                    f"H/{stage}/{early}/act_norm_mean",
                )
                if xs:
                    ax.plot(xs, ys, label=label, linewidth=1.6, color=color)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        ax.axhline(0.8, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_ylabel("cycle ratio (act_norm)")
        ax.set_xlabel("step")
        ax.set_title("Cross-cycle H (last/first)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

        # (1,1) L cycle late/early
        ax = axes[1, 1]
        # Prefer last vs first L cycle present
        l_cycles = sorted(
            {
                k.split("/")[2]
                for k in self._history
                if k.startswith(f"L/{ffn}/cL") and k.endswith("/act_norm_mean")
            }
        )
        if len(l_cycles) >= 2:
            early, late = l_cycles[0], l_cycles[-1]
            xs, ys = ratio_curve(
                f"L/{ffn}/{late}/act_norm_mean",
                f"L/{ffn}/{early}/act_norm_mean",
            )
            if xs:
                ax.plot(xs, ys, label=f"L {ffn} {late}/{early}", linewidth=1.6, color="#1f77b4")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_ylabel("cycle ratio (act_norm)")
        ax.set_xlabel("step")
        ax.set_title("Cross-cycle L (late/early)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

        fig.suptitle("Depth & loop residual scaling diagnostics")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(self.plot_dir / "scale_ratios.png", dpi=_PLOT_DPI)
        plt.close(fig)

    def _plot_unroll_cycle_stability(self) -> None:
        """Stability of one forward unroll: H calls (e.g. 3) vs L calls (e.g. 9).

        For H=3, L_cycles=3 → 3*(3L+1H): H uses cH00..cH02, L uses cL00..cL08.
        Plots act_norm / act_var of each call vs training step, plus last/first ratios
        and a H↔L spread comparison.
        """
        ffn = self.ffn_tag

        def cycle_ids(level: str) -> list[str]:
            ids = sorted(
                {
                    k.split("/")[2]
                    for k in self._history
                    if k.startswith(f"{level}/{ffn}/c{level}") and k.endswith("/act_norm_mean")
                }
            )
            return ids

        h_ids = cycle_ids("H")
        l_ids = cycle_ids("L")
        if len(h_ids) < 2 and len(l_ids) < 2:
            return

        def series_map(key: str) -> dict[int, float]:
            if key not in self._history:
                return {}
            return {int(s): float(v) for s, v in self._history[key]}

        def steps_union(*maps: dict[int, float]) -> list[int]:
            s: set[int] = set()
            for m in maps:
                s.update(m.keys())
            return sorted(s)

        # Prefer ffn; also plot attn as dashed overlays when available.
        fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
        cmap_h = plt.get_cmap("Reds")
        cmap_l = plt.get_cmap("Blues")

        def plot_level_raw(ax, level: str, ids: list[str], metric: str, cmap) -> None:
            n = max(len(ids), 1)
            for i, cid in enumerate(ids):
                key = f"{level}/{ffn}/{cid}/{metric}"
                sm = series_map(key)
                if not sm:
                    continue
                xs = sorted(sm)
                ys = [sm[x] for x in xs]
                color = cmap(0.35 + 0.55 * i / max(n - 1, 1))
                ax.plot(xs, ys, linewidth=1.5, label=cid, color=color)
            ax.axhline(0, color="none")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, ncol=3, loc="best")

        def plot_level_ratio(ax, level: str, ids: list[str], metric: str, cmap) -> None:
            if len(ids) < 2:
                ax.set_visible(False)
                return
            base = series_map(f"{level}/{ffn}/{ids[0]}/{metric}")
            n = max(len(ids) - 1, 1)
            for i, cid in enumerate(ids[1:], start=1):
                cur = series_map(f"{level}/{ffn}/{cid}/{metric}")
                xs, ys = [], []
                for s in steps_union(base, cur):
                    if s in base and s in cur and base[s] != 0:
                        xs.append(s)
                        ys.append(cur[s] / base[s])
                if not xs:
                    continue
                color = cmap(0.35 + 0.55 * (i - 1) / max(n - 1, 1))
                ax.plot(xs, ys, linewidth=1.4, label=f"{cid}/{ids[0]}", color=color)
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, ncol=2, loc="best")

        # Row 0: raw act_norm for H (3) and L (9)
        plot_level_raw(axes[0, 0], "H", h_ids, "act_norm_mean", cmap_h)
        axes[0, 0].set_title(f"H calls act_norm (n={len(h_ids)})")
        axes[0, 0].set_ylabel("act_norm (layer-mean)")
        plot_level_raw(axes[0, 1], "L", l_ids, "act_norm_mean", cmap_l)
        axes[0, 1].set_title(f"L calls act_norm (n={len(l_ids)})")

        # Row 1: raw act_var
        plot_level_raw(axes[1, 0], "H", h_ids, "act_var_mean", cmap_h)
        axes[1, 0].set_title(f"H calls act_var (n={len(h_ids)})")
        axes[1, 0].set_ylabel("act_var (layer-mean)")
        plot_level_raw(axes[1, 1], "L", l_ids, "act_var_mean", cmap_l)
        axes[1, 1].set_title(f"L calls act_var (n={len(l_ids)})")

        # Row 2: ratio to first call + H↔L spread relationship
        plot_level_ratio(axes[2, 0], "H", h_ids, "act_norm_mean", cmap_h)
        axes[2, 0].set_title("H stability: call_t / call_0 (act_norm)")
        axes[2, 0].set_ylabel("ratio")
        axes[2, 0].set_xlabel("step")

        ax = axes[2, 1]
        # Spread = max/min across calls at each step (want ≈1 if stable)
        def spread_curve(level: str, ids: list[str], metric: str) -> tuple[list[int], list[float]]:
            maps = [series_map(f"{level}/{ffn}/{cid}/{metric}") for cid in ids]
            maps = [m for m in maps if m]
            if len(maps) < 2:
                return [], []
            xs, ys = [], []
            for s in steps_union(*maps):
                vals = [m[s] for m in maps if s in m]
                if len(vals) < 2:
                    continue
                lo = min(vals)
                if lo == 0:
                    continue
                xs.append(s)
                ys.append(max(vals) / lo)
            return xs, ys

        if len(h_ids) >= 2:
            xs, ys = spread_curve("H", h_ids, "act_norm_mean")
            if xs:
                ax.plot(xs, ys, linewidth=1.8, color="#d62728", label=f"H max/min (n={len(h_ids)})")
            xs, ys = spread_curve("H", h_ids, "act_var_mean")
            if xs:
                ax.plot(xs, ys, linewidth=1.4, linestyle="--", color="#d62728", alpha=0.75, label="H var max/min")
        if len(l_ids) >= 2:
            xs, ys = spread_curve("L", l_ids, "act_norm_mean")
            if xs:
                ax.plot(xs, ys, linewidth=1.8, color="#1f77b4", label=f"L max/min (n={len(l_ids)})")
            xs, ys = spread_curve("L", l_ids, "act_var_mean")
            if xs:
                ax.plot(xs, ys, linewidth=1.4, linestyle="--", color="#1f77b4", alpha=0.75, label="L var max/min")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_title("H↔L relationship: within-forward call spread")
        ax.set_ylabel("max/min across calls")
        ax.set_xlabel("step")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

        n_h, n_l = len(h_ids), len(l_ids)
        fig.suptitle(
            f"Forward unroll cycle stability ({ffn})\n"
            f"H calls={n_h}, L calls={n_l}"
            + (f"  ≈ {n_h}×({n_l // n_h}L+1H)" if n_h > 0 and n_l % n_h == 0 else ""),
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out = self.plot_dir / "unroll_cycle_stability.png"
        fig.savefig(out, dpi=_PLOT_DPI)
        plt.close(fig)
        # Also copy into cycles/ for the placement probe summary path
        cycle_dir = self.plot_dir / "cycles"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(out, cycle_dir / "unroll_cycle_stability.png")
        except Exception:
            pass

    def _discover_cycles(self) -> list[str]:
        cyc_ids: set[str] = set()
        for k in self._history:
            if "/cH" not in k and "/cL" not in k:
                continue
            part = k.split("/c")[1]
            cyc = part.split("/")[0]
            if len(cyc) >= 3 and cyc[0] in "HL" and cyc[1:].isdigit():
                cyc_ids.add(cyc)
        return sorted(cyc_ids)

    def _plot_line_if_present(
        self,
        ax: Any,
        key: str,
        *,
        label: str | None,
        color: str | None = None,
        linewidth: float = 1.3,
    ) -> bool:
        if key not in self._history:
            return False
        steps, vals = zip(*self._history[key])
        ax.plot(
            steps,
            vals,
            linewidth=linewidth,
            label=label,
            color=color if color is not None else _MODULE_COLORS.get(label or ""),
        )
        return True

    def _plot_per_cycle_figures(self, cycle_dir: Path) -> None:
        """(1) Per cycle: cols=layers, rows=act_norm/act_var only."""
        cyc_ids = self._discover_cycles()
        for cyc in cyc_ids:
            level = cyc[0]
            layers = self._layer_indices(level)
            if not layers:
                continue
            n_cols = len(layers)
            fig, axes = plt.subplots(
                2,
                n_cols,
                figsize=(max(10, 2.2 * n_cols), 5.5),
                sharex=True,
                squeeze=False,
            )
            for c, (layer_i, tag) in enumerate(layers):
                for r, (metric, ylab) in enumerate(_ACT_METRICS):
                    ax = axes[r, c]
                    for stage in ("attn", self.ffn_tag):
                        key = f"{tag}/{stage}/c{cyc}/{metric}"
                        if key not in self._history:
                            key = f"{tag}/{stage}/{metric}"
                        self._plot_line_if_present(
                            ax,
                            key,
                            label=f"{level}/{stage}" if c == 0 and r == 0 else None,
                            color=_MODULE_COLORS.get(f"{level}/{stage}"),
                        )
                    if r == 0:
                        ax.set_title(f"L{layer_i:02d}", fontsize=9)
                    if c == 0:
                        ax.set_ylabel(ylab, fontsize=8)
                    if r == len(_ACT_METRICS) - 1:
                        ax.set_xlabel("step", fontsize=7)
                    ax.grid(True, alpha=0.25)
                    ax.tick_params(labelsize=6)

            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=2)
            fig.suptitle(f"(1) {level}-module cycle {cyc}: columns=layers · act_norm / act_var")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(cycle_dir / f"{level}_cycle_{cyc[1:]}.png", dpi=_PLOT_DPI)
            plt.close(fig)

    def _plot_hl_by_layer(self, cycle_dir: Path) -> None:
        """(2) columns=layers; H+L on same axes; cycle-averaged acts only."""
        idx_map: dict[int, dict[str, str]] = defaultdict(dict)
        for level in ("H", "L"):
            for layer_i, tag in self._layer_indices(level):
                idx_map[layer_i][level] = tag
        if not idx_map:
            return
        layer_ids = sorted(idx_map)
        n_cols = len(layer_ids)
        fig, axes = plt.subplots(
            2, n_cols, figsize=(max(10, 2.2 * n_cols), 5.5), sharex=True, squeeze=False
        )
        for c, layer_i in enumerate(layer_ids):
            for r, (metric, ylab) in enumerate(_ACT_METRICS):
                ax = axes[r, c]
                for level in ("H", "L"):
                    tag = idx_map[layer_i].get(level)
                    if tag is None:
                        continue
                    for stage in ("attn", self.ffn_tag):
                        label = f"{level}/{stage}"
                        self._plot_line_if_present(
                            ax,
                            f"{tag}/{stage}/{metric}",
                            label=label if (c == 0 and r == 0) else None,
                            color=_MODULE_COLORS.get(label),
                        )
                if r == 0:
                    ax.set_title(f"L{layer_i:02d}", fontsize=9)
                if c == 0:
                    ax.set_ylabel(ylab, fontsize=8)
                if r == len(_ACT_METRICS) - 1:
                    ax.set_xlabel("step", fontsize=7)
                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=6)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=2)
        fig.suptitle(
            f"(2) H+L by layer (cycle-averaged acts): columns=layers\n"
            f"H/attn · H/{self.ffn_tag} · L/attn · L/{self.ffn_tag}"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(cycle_dir / "HL_cols_layer.png", dpi=_PLOT_DPI)
        plt.close(fig)

    def _plot_hl_by_cycle(self, cycle_dir: Path) -> None:
        """(3) columns=cycle index; H+L on same axes; layer-mean acts only."""
        cyc_ids = self._discover_cycles()
        if not cyc_ids:
            return
        cycle_nums = sorted({int(c[1:]) for c in cyc_ids})
        n_cols = len(cycle_nums)
        fig, axes = plt.subplots(
            2, n_cols, figsize=(max(10, 2.4 * n_cols), 5.5), sharex=True, squeeze=False
        )
        for c, num in enumerate(cycle_nums):
            for r, (metric, ylab) in enumerate(_ACT_METRICS):
                ax = axes[r, c]
                for level in ("H", "L"):
                    cyc = f"{level}{num:02d}"
                    if cyc not in cyc_ids:
                        continue
                    for stage in ("attn", self.ffn_tag):
                        label = f"{level}/{stage}"
                        key = f"{level}/{stage}/c{cyc}/{metric}_mean"
                        if key not in self._history:
                            continue
                        self._plot_line_if_present(
                            ax,
                            key,
                            label=label if (c == 0 and r == 0) else None,
                            color=_MODULE_COLORS.get(label),
                        )
                if r == 0:
                    present = [f"c{lv}{num:02d}" for lv in ("H", "L") if f"{lv}{num:02d}" in cyc_ids]
                    ax.set_title(f"cycle {num:02d} ({', '.join(present)})", fontsize=8)
                if c == 0:
                    ax.set_ylabel(ylab, fontsize=8)
                if r == len(_ACT_METRICS) - 1:
                    ax.set_xlabel("step", fontsize=7)
                ax.grid(True, alpha=0.25)
                ax.tick_params(labelsize=6)

        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", fontsize=8, ncol=2)
        fig.suptitle(
            "(3) H+L by cycle (layer-averaged acts): columns=cycle index\n"
            "H and L with the same cycle index share a column"
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(cycle_dir / "HL_cols_cycle.png", dpi=_PLOT_DPI)
        plt.close(fig)

    def _plot_cycle_compare(self, cycle_dir: Path) -> None:
        """Compare cycles over training: layer-mean act_norm / act_var per cycle vs step."""
        for level in ("H", "L"):
            cyc_mean_keys = [
                k
                for k in self._history
                if k.startswith(f"{level}/")
                and "/c" in k
                and k.endswith("_mean")
                and ("act_norm_mean" in k or "act_var_mean" in k)
            ]
            if not cyc_mean_keys:
                continue
            fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
            for col, stage in enumerate(("attn", self.ffn_tag)):
                for row, metric in enumerate(("act_norm_mean", "act_var_mean")):
                    ax = axes[row, col]
                    for k in sorted(self._history):
                        if not k.startswith(f"{level}/{stage}/c"):
                            continue
                        if not k.endswith(metric):
                            continue
                        mid = k[len(f"{level}/{stage}/") :]
                        cyc_label = mid.split("/")[0]
                        steps, vals = zip(*self._history[k])
                        ax.plot(steps, vals, linewidth=1.4, label=cyc_label)
                    overall = f"{level}/{stage}/{metric}"
                    if overall in self._history:
                        steps, vals = zip(*self._history[overall])
                        ax.plot(
                            steps,
                            vals,
                            linewidth=2.0,
                            linestyle="--",
                            color="black",
                            label="all-cycles",
                            alpha=0.7,
                        )
                    ax.set_title(f"{level}/{stage} {metric.replace('_mean', '')}")
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=7, ncol=2, loc="best")
                    if row == 1:
                        ax.set_xlabel("step")
            fig.suptitle(f"{level}-module: activation by recurrent cycle (layer means)")
            fig.tight_layout()
            fig.savefig(cycle_dir / f"{level}_cycles_compare.png", dpi=_PLOT_DPI)
            plt.close(fig)


def replot_from_csv(csv_path: str | Path, plot_dir: str | Path | None = None, ffn_tag: str = "moe") -> Path:
    """Rebuild plots from an existing module_diag.csv (e.g. finished run without cycle tags)."""
    import csv

    csv_path = Path(csv_path)
    plot_dir = Path(plot_dir) if plot_dir is not None else csv_path.parent / "plots"
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row: dict[str, Any] = {}
            for k, v in raw.items():
                if k is None:
                    continue
                if k == "step":
                    row[k] = int(float(v)) if v not in ("", None) else 0
                else:
                    try:
                        row[k] = float(v) if v not in ("", None) else float("nan")
                    except ValueError:
                        continue
            rows.append(row)

    col = ModuleDiagCollector.__new__(ModuleDiagCollector)
    col.ffn_tag = ffn_tag
    col.plot_dir = plot_dir
    col._history = defaultdict(list)
    col._rows = rows
    # Infer layer tags from column names
    tags: set[str] = set()
    for row in rows:
        for k, v in row.items():
            if k == "step":
                continue
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                continue
            col._history[k].append((int(row["step"]), float(v)))
            # H/L00/attn/...
            parts = k.split("/")
            if len(parts) >= 2 and parts[0] in ("H", "L") and parts[1].startswith("L"):
                tags.add(f"{parts[0]}/{parts[1]}")
    col._attn_modules = {t: None for t in sorted(tags)}  # type: ignore[assignment]
    col.save_plots()
    return plot_dir


def get_diag_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    try:
        import torch_npu  # noqa: F401

        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu", torch.npu.current_device())
    except Exception:
        pass
    return torch.device("cpu")


def _write_simple_xlsx(path: Path, cols: list[str], rows: list[dict[str, Any]]) -> None:
    """Write a single-sheet xlsx with stdlib only (no openpyxl)."""
    import zipfile
    from xml.sax.saxutils import escape

    def cell_ref(r: int, c: int) -> str:
        name = ""
        n = c
        while n:
            n, rem = divmod(n - 1, 26)
            name = chr(65 + rem) + name
        return f"{name}{r}"

    def cell_xml(r: int, c: int, value: Any) -> str:
        ref = cell_ref(r, c)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return f'<c r="{ref}" t="inlineStr"><is><t></t></is></c>'
            return f'<c r="{ref}"><v>{value}</v></c>'
        text = escape("" if value is None else str(value))
        return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'

    sheet_rows = []
    header_cells = "".join(cell_xml(1, c + 1, cols[c]) for c in range(len(cols)))
    sheet_rows.append(f'<row r="1">{header_cells}</row>')
    for i, row in enumerate(rows, start=2):
        cells = "".join(cell_xml(i, c + 1, row.get(cols[c], "")) for c in range(len(cols)))
        sheet_rows.append(f'<row r="{i}">{cells}</row>')

    sheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData></worksheet>'
    )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="module_diag" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>'
    )
    wb_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", wb_rels)
        zf.writestr("xl/worksheets/sheet1.xml", sheet)
