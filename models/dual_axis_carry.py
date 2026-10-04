"""Dual-axis loop attention — N/D mix with pluggable update rules.

Locked residual wiring for current campaign:
  x ← x + fuse(r_H, r_L); N/D updated from branch output o.

Update modes (AttnRes notes / age–content decomposition):
  ema            — N←βN+e·o, D←βD+e           (recursive age decay)
  accum          — N←N+e·o, D←D+e              (no β decay)
  window         — recompute over last W steps, equal weights on e
  window_age     — last W with e_t·β^{W-1-t}
  window_softmax — last W: α=softmax(s), mix Σ α o
  window_smage   — last W: α ∝ softmax(s)·β^{age}, then normalize

β may be learnable (sigmoid) or fixed (buffer). Learnable β previously
drifted →0 (carry-last), which helps short H8 but hurts deepen.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1.0 - 1e-4)
    return math.log(p / (1.0 - p))


# Per-step buffer entry: (score_or_e [...,H,1], o_h [...,H,D]) both fp32
_Buf = List[Tuple[Tensor, Tensor]]


class DualAxisCarry(nn.Module):
    """N/D dual-axis carry with pluggable update + content score."""

    def __init__(
        self,
        n_layers: int,
        hidden_size: int,
        beta_init: float = 0.55,
        beta_h_init: Optional[float] = None,
        beta_l_init: Optional[float] = None,
        n_heads: int = 1,
        use_content: bool = True,
        content_mode: str = "exp",  # exp | softplus | softmax (legacy seq)
        content_temp: float = 1.0,
        fuse_half: bool = False,
        fuse_mode: str = "sum",
        blend_lambda: float = 0.25,
        l_persist: bool = False,
        apply: str = "attn",
        residual_mode: str = "x_plus_r",
        update_src: str = "o",
        # --- update mechanism ---
        nd_mode: str = "ema",
        beta_learnable: bool = True,
        window_size: int = 4,
        # Update H-axis N/D only every `h_update_every` loops (notes: ~3-step memory).
        # L-axis still updates every layer call. When H is frozen, reuse last H readout.
        h_update_every: int = 1,
        # Floor on learnable β: β = β_min + (1-β_min)*sigmoid(raw). Stops β→0 collapse.
        beta_h_min: float = 0.0,
        beta_l_min: float = 0.0,
        # Scale AttnRes write by loop depth: none | sqrt_t | inv_t
        r_scale_mode: str = "none",
        # Decay β_H used for H-axis N/D *writes* with loop t (not residual rscale):
        #   none | sqrt_t | inv_t  →  β_eff = β_H / sqrt(t) or β_H / t
        beta_h_decay: str = "none",
        # post: update N/D then readout (current). pre: readout memory then update.
        update_order: str = "post",
        # H-axis storage layout:
        #   per_layer — legacy: one N/D per layer (L slots), persist across loops
        #   shared    — one N/D for all layers in a loop; persist across loops
        #   both      — shared H + per-layer P (third axis); L-axis always present
        h_layout: str = "per_layer",
        # When to write the shared/per_layer H(P) N/D within a loop:
        #   every_layer — write at every layer call (legacy for per_layer)
        #   last_layer  — write only at layer n_layers-1 (TRM-like slow H)
        #   loop_mean   — accumulate o over layers; one EMA write at end_loop with mean(o)
        #   first_layer — write only at layer 0 (cross-loop handover)
        #   stride      — write when layer_idx % h_write_stride == 0
        #   first_last  — write at layer 0 and last layer only
        #   mid_layer   — write once at layer n_layers//2
        #   first_mid_last — write at layer 0, n_layers//2, and last (3 sparse points)
        h_write: str = "every_layer",
        # Stride for h_write=stride (layers 0, m, 2m, ...). Ignored otherwise.
        h_write_stride: int = 2,
        # What updates shared H (and cascade-style last write): o | l (r_L after L update)
        h_update_src: str = "o",
        beta_p_init: Optional[float] = None,
        # If True, β_L is a vector indexed by loop_t (1..max_loops); else one shared β_L.
        beta_l_per_loop: bool = False,
        max_loops: int = 16,
    ) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.hidden_size = int(hidden_size)
        self.n_heads = max(1, int(n_heads))
        if self.hidden_size % self.n_heads != 0:
            raise ValueError(
                f"hidden_size={self.hidden_size} not divisible by dual_axis_heads={self.n_heads}"
            )
        self.head_dim = self.hidden_size // self.n_heads
        self.use_content = bool(use_content)
        self.content_mode = str(content_mode or "exp").lower()
        if self.content_mode not in ("exp", "softmax", "softplus"):
            raise ValueError(
                f"dual_axis_content_mode={self.content_mode!r}; expected exp|softmax|softplus"
            )
        self.content_temp = max(float(content_temp), 1e-4)
        self.fuse_half = bool(fuse_half)
        fuse_mode = str(fuse_mode or "sum").lower()
        if fuse_half and fuse_mode not in ("half", "sum", "cascade"):
            fuse_mode = "half"
        if fuse_mode not in ("sum", "half", "dedup", "cascade", "blend", "gated", "h_only", "l_only"):
            raise ValueError(
                f"dual_axis_fuse_mode={fuse_mode!r}; "
                "expected sum|half|dedup|cascade|blend|gated|h_only|l_only"
            )
        self.fuse_mode = fuse_mode
        self.blend_lambda = float(blend_lambda)
        self.l_persist = bool(l_persist)
        apply = str(apply or "attn").lower()
        if apply not in ("both", "attn", "mlp", "none"):
            raise ValueError(f"dual_axis_apply={apply!r}; expected both|attn|mlp|none")
        self.apply = apply
        self.apply_attn = apply in ("both", "attn")
        self.apply_mlp = apply in ("both", "mlp")
        residual_mode = str(residual_mode or "x_plus_r").lower()
        if residual_mode not in ("x_plus_r", "replace_r", "o_plus_r", "x_plus_o_plus_r"):
            raise ValueError(
                f"dual_axis_residual_mode={residual_mode!r}; "
                "expected x_plus_r|replace_r|o_plus_r|x_plus_o_plus_r"
            )
        self.residual_mode = residual_mode
        update_src = str(update_src or "o").lower()
        if update_src not in ("o", "x"):
            raise ValueError(f"dual_axis_update_src={update_src!r}; expected o|x")
        self.update_src = update_src

        nd_mode = str(nd_mode or "ema").lower()
        if nd_mode not in (
            "ema",
            "accum",
            "window",
            "window_age",
            "window_softmax",
            "window_smage",
        ):
            raise ValueError(
                f"dual_axis_nd_mode={nd_mode!r}; expected "
                "ema|accum|window|window_age|window_softmax|window_smage"
            )
        self.nd_mode = nd_mode
        self.beta_learnable = bool(beta_learnable)
        self.window_size = max(1, int(window_size))
        self._use_window = nd_mode.startswith("window")
        self.h_update_every = max(1, int(h_update_every))
        self.beta_h_min = float(min(max(beta_h_min, 0.0), 0.99))
        self.beta_l_min = float(min(max(beta_l_min, 0.0), 0.99))
        rsm = str(r_scale_mode or "none").lower()
        if rsm not in ("none", "", "sqrt_t", "sqrt", "inv_t", "inv"):
            raise ValueError(
                f"dual_axis_r_scale_mode={r_scale_mode!r}; expected none|sqrt_t|inv_t"
            )
        self.r_scale_mode = "none" if rsm in ("", "none") else rsm
        bhd = str(beta_h_decay or "none").lower()
        if bhd not in ("none", "", "sqrt_t", "sqrt", "inv_t", "inv"):
            raise ValueError(
                f"dual_axis_beta_h_decay={beta_h_decay!r}; expected none|sqrt_t|inv_t"
            )
        self.beta_h_decay = "none" if bhd in ("", "none") else bhd
        uo = str(update_order or "post").lower()
        if uo not in ("post", "pre"):
            raise ValueError(f"dual_axis_update_order={update_order!r}; expected post|pre")
        self.update_order = uo

        hl = str(h_layout or "per_layer").lower()
        if hl not in ("per_layer", "shared", "both"):
            raise ValueError(
                f"dual_axis_h_layout={h_layout!r}; expected per_layer|shared|both"
            )
        self.h_layout = hl
        self.use_shared_h = hl in ("shared", "both")
        self.use_per_layer_h = hl in ("per_layer", "both")
        hw = str(h_write or "every_layer").lower().replace("+", "_").replace("-", "_")
        # Accept first+last / first-last aliases → first_last
        if hw in ("firstlast",):
            hw = "first_last"
        _ok_write = (
            "every_layer",
            "last_layer",
            "loop_mean",
            "first_layer",
            "stride",
            "first_last",
            "mid_layer",
            "first_mid_last",
        )
        if hw not in _ok_write:
            raise ValueError(
                f"dual_axis_h_write={h_write!r}; expected "
                "every_layer|last_layer|loop_mean|first_layer|stride|"
                "first_last|mid_layer|first_mid_last"
            )
        self.h_write = hw
        self.h_write_stride = max(1, int(h_write_stride))
        hus = str(h_update_src or "o").lower()
        if hus not in ("o", "l"):
            raise ValueError(f"dual_axis_h_update_src={h_update_src!r}; expected o|l")
        self.h_update_src = hus
        self.beta_l_per_loop = bool(beta_l_per_loop)
        self.max_loops = max(1, int(max_loops))

        if self.fuse_mode == "gated":
            g0 = min(max(float(blend_lambda), 1e-3), 1.0 - 1e-3)
            self.gate_raw = nn.Parameter(
                torch.tensor([_logit(g0)], dtype=torch.float32)
            )
        else:
            self.register_parameter("gate_raw", None)

        bh0 = float(beta_h_init if beta_h_init is not None else beta_init)
        bl0 = float(beta_l_init if beta_l_init is not None else beta_init)
        bp0 = float(beta_p_init if beta_p_init is not None else bh0)
        if self.beta_learnable:
            if self.n_heads == 1:
                self.beta_h_raw = nn.Parameter(
                    torch.tensor([_logit(bh0)], dtype=torch.float32)
                )
                if self.beta_l_per_loop:
                    self.beta_l_raw = nn.Parameter(
                        torch.full(
                            (self.max_loops,), _logit(bl0), dtype=torch.float32
                        )
                    )
                else:
                    self.beta_l_raw = nn.Parameter(
                        torch.tensor([_logit(bl0)], dtype=torch.float32)
                    )
                self.beta_p_raw = nn.Parameter(
                    torch.tensor([_logit(bp0)], dtype=torch.float32)
                )
            else:
                stairs_h = torch.linspace(0.2, 0.9, self.n_heads)
                stairs_l = torch.linspace(0.2, 0.9, self.n_heads)
                stairs_p = torch.linspace(0.2, 0.9, self.n_heads)
                self.beta_h_raw = nn.Parameter(
                    torch.tensor([_logit(float(p)) for p in stairs_h], dtype=torch.float32)
                )
                if self.beta_l_per_loop:
                    # [max_loops, n_heads]
                    self.beta_l_raw = nn.Parameter(
                        torch.tensor(
                            [[_logit(float(p)) for p in stairs_l] for _ in range(self.max_loops)],
                            dtype=torch.float32,
                        )
                    )
                else:
                    self.beta_l_raw = nn.Parameter(
                        torch.tensor([_logit(float(p)) for p in stairs_l], dtype=torch.float32)
                    )
                self.beta_p_raw = nn.Parameter(
                    torch.tensor([_logit(float(p)) for p in stairs_p], dtype=torch.float32)
                )
        else:
            self.register_parameter("beta_h_raw", None)
            self.register_parameter("beta_l_raw", None)
            self.register_parameter("beta_p_raw", None)
            self.register_buffer(
                "beta_h_fixed", torch.tensor([bh0], dtype=torch.float32)
            )
            if self.beta_l_per_loop:
                self.register_buffer(
                    "beta_l_fixed",
                    torch.full((self.max_loops,), bl0, dtype=torch.float32),
                )
            else:
                self.register_buffer(
                    "beta_l_fixed", torch.tensor([bl0], dtype=torch.float32)
                )
            self.register_buffer(
                "beta_p_fixed", torch.tensor([bp0], dtype=torch.float32)
            )

        if self.use_content:
            self.content_h = nn.Parameter(torch.zeros(self.n_heads, self.head_dim))
            self.content_l = nn.Parameter(torch.zeros(self.n_heads, self.head_dim))
            self.content_p = nn.Parameter(torch.zeros(self.n_heads, self.head_dim))
        else:
            self.register_parameter("content_h", None)
            self.register_parameter("content_l", None)
            self.register_parameter("content_p", None)

        self.loop_t: int = 1
        # Per-layer H (or third-axis P when h_layout=both)
        self.h_attn_n: List[Optional[Tensor]] = [None] * self.n_layers
        self.h_attn_d: List[Optional[Tensor]] = [None] * self.n_layers
        self.h_mlp_n: List[Optional[Tensor]] = [None] * self.n_layers
        self.h_mlp_d: List[Optional[Tensor]] = [None] * self.n_layers
        # Shared H: one N/D for the whole loop (all layers read the same)
        self.h_shared_attn_n: Optional[Tensor] = None
        self.h_shared_attn_d: Optional[Tensor] = None
        self.h_shared_mlp_n: Optional[Tensor] = None
        self.h_shared_mlp_d: Optional[Tensor] = None
        self.l_attn_n: Optional[Tensor] = None
        self.l_attn_d: Optional[Tensor] = None
        self.l_mlp_n: Optional[Tensor] = None
        self.l_mlp_d: Optional[Tensor] = None
        # loop_mean accumulators (reset each begin_loop)
        self.h_mean_attn_sum: Optional[Tensor] = None
        self.h_mean_attn_count: int = 0
        self.h_mean_mlp_sum: Optional[Tensor] = None
        self.h_mean_mlp_count: int = 0
        # Window buffers (only when nd_mode is window*)
        self.h_attn_buf: List[_Buf] = [[] for _ in range(self.n_layers)]
        self.h_mlp_buf: List[_Buf] = [[] for _ in range(self.n_layers)]
        self.h_shared_attn_buf: _Buf = []
        self.h_shared_mlp_buf: _Buf = []
        self.l_attn_buf: _Buf = []
        self.l_mlp_buf: _Buf = []

    def combine_residual(self, x: Tensor, branch_out: Tensor, r: Tensor) -> Tensor:
        # Optional loop-depth scale on AttnRes write (Residual Scaling of Looped Transformers).
        # "sqrt_t": r /= sqrt(loop_t); "inv_t": r /= loop_t; "none"/"": identity.
        scale_mode = str(getattr(self, "r_scale_mode", "none") or "none").lower()
        if scale_mode in ("sqrt_t", "sqrt"):
            r = r * (1.0 / math.sqrt(float(max(self.loop_t, 1))))
        elif scale_mode in ("inv_t", "inv"):
            r = r * (1.0 / float(max(self.loop_t, 1)))
        if self.residual_mode == "replace_r":
            return r
        if self.residual_mode == "o_plus_r":
            return branch_out + r
        if self.residual_mode == "x_plus_o_plus_r":
            # Keep vanilla x+o; AttnRes memory is an additive carry (pairs with update_order=pre).
            return x + branch_out + r
        return x + r

    def beta_h(self) -> Tensor:
        if self.beta_learnable:
            b = torch.sigmoid(self.beta_h_raw)
            if self.beta_h_min > 0.0:
                b = self.beta_h_min + (1.0 - self.beta_h_min) * b
            return b.mean() if b.numel() > 1 else b.squeeze()
        return self.beta_h_fixed.squeeze()

    def beta_h_for_update(self) -> Tensor:
        """β_H for writing H N/D; optional 1/sqrt(t) or 1/t decay (deepen unlock)."""
        b = self.beta_h()
        mode = str(getattr(self, "beta_h_decay", "none") or "none").lower()
        if mode in ("sqrt_t", "sqrt"):
            return b * (1.0 / math.sqrt(float(max(self.loop_t, 1))))
        if mode in ("inv_t", "inv"):
            return b * (1.0 / float(max(self.loop_t, 1)))
        return b

    def beta_l(self) -> Tensor:
        idx = min(max(int(self.loop_t) - 1, 0), self.max_loops - 1)
        if self.beta_learnable:
            raw = self.beta_l_raw
            if self.beta_l_per_loop:
                # [max_loops] or [max_loops, n_heads]
                b = torch.sigmoid(raw[idx])
            else:
                b = torch.sigmoid(raw)
            if self.beta_l_min > 0.0:
                b = self.beta_l_min + (1.0 - self.beta_l_min) * b
            return b.mean() if b.numel() > 1 else b.squeeze()
        fixed = self.beta_l_fixed
        if self.beta_l_per_loop:
            return fixed[idx]
        return fixed.squeeze()

    def beta_p(self) -> Tensor:
        """Third-axis (per-layer P) β; falls back to β_H if unused."""
        if self.beta_learnable:
            b = torch.sigmoid(self.beta_p_raw)
            if self.beta_h_min > 0.0:
                b = self.beta_h_min + (1.0 - self.beta_h_min) * b
            return b.mean() if b.numel() > 1 else b.squeeze()
        return self.beta_p_fixed.squeeze()

    @torch.no_grad()
    def get_attn_res_beta_metrics(self) -> dict[str, float]:
        """Scalars for module_diag: effective AttnRes β (post-sigmoid / mins)."""
        out: dict[str, float] = {}

        def _to_float(t: Tensor) -> float:
            x = t.detach()
            loc = getattr(x, "_local_tensor", None)
            if loc is not None:
                x = loc
            elif type(x).__name__ == "DTensor" or hasattr(x, "to_local") or hasattr(x, "full_tensor"):
                return float("nan")
            return float(x.float().reshape(-1).mean().item())

        try:
            out["attn_res_beta_h"] = _to_float(self.beta_h())
            out["attn_res_beta_l"] = _to_float(self.beta_l())
            out["attn_res_beta_p"] = _to_float(self.beta_p())
            # Per-loop β_L snapshot (first / mid / last) when per-loop table exists
            if self.beta_learnable and self.beta_l_per_loop and self.beta_l_raw is not None:
                raw = self.beta_l_raw.detach()
                loc = getattr(raw, "_local_tensor", None)
                if loc is not None:
                    raw = loc
                elif type(raw).__name__ == "DTensor" or hasattr(raw, "to_local"):
                    raw = None
                if raw is None:
                    raise StopIteration
                b = torch.sigmoid(raw.float())
                if self.beta_l_min > 0.0:
                    b = self.beta_l_min + (1.0 - self.beta_l_min) * b
                # mean over heads if present
                if b.dim() > 1:
                    b = b.mean(dim=-1)
                n = int(b.numel())
                if n >= 1:
                    out["attn_res_beta_l_loop0"] = float(b[0].item())
                    out["attn_res_beta_l_loop_mid"] = float(b[n // 2].item())
                    out["attn_res_beta_l_loop_last"] = float(b[n - 1].item())
                    out["attn_res_beta_l_mean"] = float(b.mean().item())
        except Exception:
            return out
        return out

    def _clear_mean_acc(self) -> None:
        self.h_mean_attn_sum = None
        self.h_mean_attn_count = 0
        self.h_mean_mlp_sum = None
        self.h_mean_mlp_count = 0

    def clear(self) -> None:
        self.h_attn_n = [None] * self.n_layers
        self.h_attn_d = [None] * self.n_layers
        self.h_mlp_n = [None] * self.n_layers
        self.h_mlp_d = [None] * self.n_layers
        self.h_shared_attn_n = self.h_shared_attn_d = None
        self.h_shared_mlp_n = self.h_shared_mlp_d = None
        self.l_attn_n = self.l_attn_d = None
        self.l_mlp_n = self.l_mlp_d = None
        self.h_attn_buf = [[] for _ in range(self.n_layers)]
        self.h_mlp_buf = [[] for _ in range(self.n_layers)]
        self.h_shared_attn_buf = []
        self.h_shared_mlp_buf = []
        self.l_attn_buf = []
        self.l_mlp_buf = []
        self._clear_mean_acc()
        self.loop_t = 1

    def begin_loop(self, t: int) -> None:
        self.loop_t = max(1, int(t))
        if not self.l_persist:
            self.l_attn_n = self.l_attn_d = None
            self.l_mlp_n = self.l_mlp_d = None
            self.l_attn_buf = []
            self.l_mlp_buf = []
        self._clear_mean_acc()

    def _should_update_h(self, N: Optional[Tensor]) -> bool:
        """Write H-axis N/D this loop? Always on first write; else every h_update_every."""
        if N is None:
            return True
        return ((self.loop_t - 1) % self.h_update_every) == 0

    def _want_write_h_at_layer(self, layer_idx: int, N: Optional[Tensor]) -> bool:
        if self.h_write == "loop_mean":
            return False  # deferred to end_loop
        if not self._should_update_h(N):
            return False
        ell = int(layer_idx)
        last = self.n_layers - 1
        if self.h_write == "last_layer":
            return ell == last
        if self.h_write == "first_layer":
            return ell == 0
        if self.h_write == "first_last":
            return ell == 0 or ell == last
        if self.h_write == "mid_layer":
            return ell == (self.n_layers // 2)
        if self.h_write == "first_mid_last":
            return ell == 0 or ell == last or ell == (self.n_layers // 2)
        if self.h_write == "stride":
            return (ell % self.h_write_stride) == 0
        # every_layer
        return True

    def _want_write_p_at_layer(self, layer_idx: int, N: Optional[Tensor]) -> bool:
        """Per-layer P always writes every layer (when enabled); still respects h_update_every."""
        if not self._should_update_h(N):
            return False
        return True

    def end_loop(self) -> None:
        """Flush loop_mean accumulators into shared H (one EMA step with mean(o))."""
        if self.h_write != "loop_mean" or not self.use_shared_h:
            return
        bh = self.beta_h_for_update()
        if self.apply_attn and self.h_mean_attn_count > 0 and self.h_mean_attn_sum is not None:
            if self._should_update_h(self.h_shared_attn_n):
                mean = self.h_mean_attn_sum / float(self.h_mean_attn_count)
                n_h, d_h, _, buf = self._nd_update(
                    self.h_shared_attn_n,
                    self.h_shared_attn_d,
                    mean,
                    bh,
                    self.content_h,
                    self.h_shared_attn_buf if self._use_window else None,
                )
                self.h_shared_attn_n, self.h_shared_attn_d = n_h, d_h
                if buf is not None:
                    self.h_shared_attn_buf = buf
        if self.apply_mlp and self.h_mean_mlp_count > 0 and self.h_mean_mlp_sum is not None:
            if self._should_update_h(self.h_shared_mlp_n):
                mean = self.h_mean_mlp_sum / float(self.h_mean_mlp_count)
                n_h, d_h, _, buf = self._nd_update(
                    self.h_shared_mlp_n,
                    self.h_shared_mlp_d,
                    mean,
                    bh,
                    self.content_h,
                    self.h_shared_mlp_buf if self._use_window else None,
                )
                self.h_shared_mlp_n, self.h_shared_mlp_d = n_h, d_h
                if buf is not None:
                    self.h_shared_mlp_buf = buf
        self._clear_mean_acc()

    def _readout_nd(
        self, N: Tensor, D: Tensor, o_dtype: torch.dtype
    ) -> Tensor:
        readout_h = N.float() / D.float().clamp_min(1e-6)
        return self._from_heads(readout_h).to(dtype=o_dtype)

    def _to_heads(self, o: Tensor) -> Tensor:
        *lead, d = o.shape
        if d != self.hidden_size:
            raise RuntimeError(f"expected hidden={self.hidden_size}, got {d}")
        return o.reshape(*lead, self.n_heads, self.head_dim)

    def _from_heads(self, x: Tensor) -> Tensor:
        *lead, h, d = x.shape
        return x.reshape(*lead, h * d)

    def _content_score(
        self, o_h: Tensor, weight: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Returns (e_or_ones, raw_logit_s). Both [..., heads, 1]."""
        if weight is None or not self.use_content:
            ones = torch.ones(
                *o_h.shape[:-1], 1, device=o_h.device, dtype=o_h.dtype
            )
            return ones, torch.zeros_like(ones)
        o_n = F.rms_norm(o_h, (o_h.shape[-1],))
        logits = (o_n * weight).sum(dim=-1, keepdim=True) / self.content_temp
        if self.content_mode == "softmax":
            t_len = max(int(logits.shape[1]), 1)
            e = torch.softmax(logits, dim=1) * float(t_len)
        elif self.content_mode == "softplus":
            e = F.softplus(logits) + 1e-6
        else:
            e = torch.exp(logits.clamp(-20.0, 20.0))
        return e, logits

    def _readout_from_buf(
        self, buf: _Buf, beta: Tensor, o_dtype: torch.dtype
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Recompute N,D,readout from window buffer."""
        assert buf, "empty window buf"
        scores = []  # unnormalized weights
        o_list = []
        wlen = len(buf)
        for i, (sc, o_h) in enumerate(buf):
            age = wlen - 1 - i
            if self.nd_mode in ("window_softmax", "window_smage"):
                # sc is logit
                s = sc
            else:
                s = sc  # e already
            if self.nd_mode == "window_age":
                b = beta.to(device=s.device, dtype=s.dtype)
                scores.append(s * (b ** float(age)))
            elif self.nd_mode == "window_smage":
                b = beta.to(device=s.device, dtype=s.dtype)
                scores.append(s + float(age) * torch.log(b.clamp_min(1e-6)))
                # will softmax below; store log-space as logits
            elif self.nd_mode == "window_softmax":
                scores.append(s)
            else:  # window equal on e
                scores.append(s)
            o_list.append(o_h)

        stacked_o = torch.stack(o_list, dim=0)  # [W, ..., H, D]
        stacked_s = torch.stack(scores, dim=0)  # [W, ..., H, 1]

        if self.nd_mode in ("window_softmax", "window_smage"):
            alpha = torch.softmax(stacked_s, dim=0)
            N_new = (alpha * stacked_o).sum(dim=0)
            D_new = alpha.sum(dim=0).clamp_min(1e-6)
            readout_h = N_new  # already convex combo
        else:
            N_new = (stacked_s * stacked_o).sum(dim=0)
            D_new = stacked_s.sum(dim=0).clamp_min(1e-6)
            readout_h = N_new / D_new

        readout = self._from_heads(readout_h).to(dtype=o_dtype)
        return N_new, D_new, readout

    def _nd_update(
        self,
        N: Optional[Tensor],
        D: Optional[Tensor],
        o: Tensor,
        beta: Tensor,
        content_w: Optional[Tensor],
        buf: Optional[_Buf] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Tensor, Optional[_Buf]]:
        o_dtype = o.dtype
        o_h = self._to_heads(o).float()
        e, logits = self._content_score(o_h, content_w)
        b = beta.to(device=o_h.device, dtype=o_h.dtype)

        if self._use_window:
            assert buf is not None
            if self.nd_mode in ("window_softmax", "window_smage"):
                buf.append((logits, o_h))
            else:
                buf.append((e, o_h))
            if len(buf) > self.window_size:
                del buf[0 : len(buf) - self.window_size]
            N_new, D_new, readout = self._readout_from_buf(buf, b, o_dtype)
            return N_new, D_new, readout, buf

        if self.nd_mode == "accum":
            if N is None or D is None:
                N_new = e * o_h
                D_new = e
            else:
                N_new = N.float() + e * o_h
                D_new = D.float() + e
        else:  # ema
            if N is None or D is None:
                N_new = e * o_h
                D_new = e
            else:
                N_new = b * N.float() + e * o_h
                D_new = b * D.float() + e

        readout_h = N_new / D_new.clamp_min(1e-6)
        readout = self._from_heads(readout_h).to(dtype=o_dtype)
        return N_new, D_new, readout, buf

    def _fuse(self, h: Tensor, l: Tensor, o: Tensor, p: Optional[Tensor] = None) -> Tensor:
        if self.fuse_mode == "h_only":
            return h if p is None else h + p
        if self.fuse_mode == "l_only":
            return l
        parts = [h, l] if p is None else [h, l, p]
        mem = sum(parts) / float(len(parts)) if self.fuse_mode == "half" else None
        if self.fuse_mode == "half":
            return mem  # type: ignore[return-value]
        if self.fuse_mode == "blend":
            lam = self.blend_lambda
            base = sum(parts) / float(len(parts))
            return (1.0 - lam) * o + lam * base
        if self.fuse_mode == "gated":
            g = torch.sigmoid(self.gate_raw).to(device=o.device, dtype=o.dtype)
            base = sum(parts) / float(len(parts))
            return (1.0 - g) * o + g * base
        if self.fuse_mode == "dedup":
            # keep legacy 2-axis meaning when no P
            if p is None:
                return h + l - o
            return h + l + p - o
        # sum (default)
        out = h + l
        if p is not None:
            out = out + p
        return out

    def _axis_readout(
        self,
        N: Optional[Tensor],
        D: Optional[Tensor],
        buf: Optional[_Buf],
        beta: Tensor,
        like: Tensor,
    ) -> Tensor:
        """Read current N/D (or window) without updating; zeros if empty."""
        if self._use_window and buf:
            _, _, r = self._readout_from_buf(buf, beta, like.dtype)
            return r
        if N is None or D is None:
            return torch.zeros_like(like)
        return self._readout_nd(N, D, like.dtype)

    def _mix_shared_or_both(self, layer_idx: int, branch_out: Tensor, *, kind: str) -> Tensor:
        """L + shared-H (+ optional per-layer P). post-order only."""
        ell = int(layer_idx)
        bh, bl, bp = self.beta_h_for_update(), self.beta_l(), self.beta_p()
        is_attn = kind == "attn"

        # --- L-axis: always update from o ---
        if is_attn:
            n_l, d_l, r_l, self.l_attn_buf = self._nd_update(
                self.l_attn_n,
                self.l_attn_d,
                branch_out,
                bl,
                self.content_l,
                self.l_attn_buf if self._use_window else None,
            )
            self.l_attn_n, self.l_attn_d = n_l, d_l
        else:
            n_l, d_l, r_l, self.l_mlp_buf = self._nd_update(
                self.l_mlp_n,
                self.l_mlp_d,
                branch_out,
                bl,
                self.content_l,
                self.l_mlp_buf if self._use_window else None,
            )
            self.l_mlp_n, self.l_mlp_d = n_l, d_l

        # --- shared H ---
        r_h = torch.zeros_like(branch_out)
        if self.use_shared_h:
            if is_attn:
                N, D, buf = self.h_shared_attn_n, self.h_shared_attn_d, self.h_shared_attn_buf
            else:
                N, D, buf = self.h_shared_mlp_n, self.h_shared_mlp_d, self.h_shared_mlp_buf
            if self.h_write == "loop_mean":
                # Accumulate o; H N/D frozen until end_loop; still readout current H.
                src = branch_out
                if is_attn:
                    if self.h_mean_attn_sum is None:
                        self.h_mean_attn_sum = src
                    else:
                        self.h_mean_attn_sum = self.h_mean_attn_sum + src
                    self.h_mean_attn_count += 1
                else:
                    if self.h_mean_mlp_sum is None:
                        self.h_mean_mlp_sum = src
                    else:
                        self.h_mean_mlp_sum = self.h_mean_mlp_sum + src
                    self.h_mean_mlp_count += 1
                r_h = self._axis_readout(
                    N, D, buf if self._use_window else None, bh, branch_out
                )
            else:
                src = r_l if self.h_update_src == "l" else branch_out
                if self._want_write_h_at_layer(ell, N):
                    n_h, d_h, r_h, buf = self._nd_update(
                        N, D, src, bh, self.content_h, buf if self._use_window else None
                    )
                    if is_attn:
                        self.h_shared_attn_n, self.h_shared_attn_d = n_h, d_h
                        self.h_shared_attn_buf = buf if buf is not None else []
                    else:
                        self.h_shared_mlp_n, self.h_shared_mlp_d = n_h, d_h
                        self.h_shared_mlp_buf = buf if buf is not None else []
                else:
                    r_h = self._axis_readout(
                        N, D, buf if self._use_window else None, bh, branch_out
                    )

        # --- per-layer P (third axis) ---
        r_p: Optional[Tensor] = None
        if self.use_per_layer_h and self.h_layout == "both":
            if is_attn:
                N, D, buf = self.h_attn_n[ell], self.h_attn_d[ell], self.h_attn_buf[ell]
            else:
                N, D, buf = self.h_mlp_n[ell], self.h_mlp_d[ell], self.h_mlp_buf[ell]
            if self._want_write_p_at_layer(ell, N):
                n_p, d_p, r_p, buf = self._nd_update(
                    N, D, branch_out, bp, self.content_p, buf if self._use_window else None
                )
                if is_attn:
                    self.h_attn_n[ell], self.h_attn_d[ell] = n_p, d_p
                    self.h_attn_buf[ell] = buf if buf is not None else []
                else:
                    self.h_mlp_n[ell], self.h_mlp_d[ell] = n_p, d_p
                    self.h_mlp_buf[ell] = buf if buf is not None else []
            else:
                r_p = self._axis_readout(
                    N, D, buf if self._use_window else None, bp, branch_out
                )

        return self._fuse(r_h, r_l, branch_out, r_p)

    def mix_attn(self, layer_idx: int, attn_out: Tensor) -> Tensor:
        if not self.apply_attn:
            return attn_out
        # New layouts: shared / both (post-order)
        if self.h_layout in ("shared", "both"):
            if self.update_order == "pre":
                raise NotImplementedError(
                    "dual_axis_update_order=pre not supported with h_layout shared|both"
                )
            if self.fuse_mode == "cascade":
                raise NotImplementedError(
                    "dual_axis_fuse_mode=cascade not supported with h_layout shared|both"
                )
            return self._mix_shared_or_both(layer_idx, attn_out, kind="attn")

        ell = int(layer_idx)
        bh, bl = self.beta_h_for_update(), self.beta_l()
        if self.fuse_mode == "cascade":
            n_l, d_l, r_l, self.l_attn_buf = self._nd_update(
                self.l_attn_n,
                self.l_attn_d,
                attn_out,
                bl,
                self.content_l,
                self.l_attn_buf if self._use_window else None,
            )
            self.l_attn_n, self.l_attn_d = n_l, d_l
            if self._want_write_h_at_layer(ell, self.h_attn_n[ell]):
                src = r_l if self.h_update_src == "l" else attn_out
                n_h, d_h, r_h, self.h_attn_buf[ell] = self._nd_update(
                    self.h_attn_n[ell],
                    self.h_attn_d[ell],
                    src,
                    bh,
                    self.content_h,
                    self.h_attn_buf[ell] if self._use_window else None,
                )
                self.h_attn_n[ell], self.h_attn_d[ell] = n_h, d_h
            else:
                r_h = self._axis_readout(
                    self.h_attn_n[ell],
                    self.h_attn_d[ell],
                    self.h_attn_buf[ell] if self._use_window else None,
                    bh,
                    attn_out,
                )
            return r_h
        # --- parallel H/L (legacy per_layer) ---
        if self.update_order == "pre":
            # Read memory BEFORE writing current o → pairs with x_plus_o_plus_r.
            r_h = self._axis_readout(
                self.h_attn_n[ell],
                self.h_attn_d[ell],
                self.h_attn_buf[ell] if self._use_window else None,
                bh,
                attn_out,
            )
            r_l = self._axis_readout(
                self.l_attn_n,
                self.l_attn_d,
                self.l_attn_buf if self._use_window else None,
                bl,
                attn_out,
            )
            mem = self._fuse(r_h, r_l, attn_out)
            if self._want_write_h_at_layer(ell, self.h_attn_n[ell]):
                n_h, d_h, _, self.h_attn_buf[ell] = self._nd_update(
                    self.h_attn_n[ell],
                    self.h_attn_d[ell],
                    attn_out,
                    bh,
                    self.content_h,
                    self.h_attn_buf[ell] if self._use_window else None,
                )
                self.h_attn_n[ell], self.h_attn_d[ell] = n_h, d_h
            n_l, d_l, _, self.l_attn_buf = self._nd_update(
                self.l_attn_n,
                self.l_attn_d,
                attn_out,
                bl,
                self.content_l,
                self.l_attn_buf if self._use_window else None,
            )
            self.l_attn_n, self.l_attn_d = n_l, d_l
            return mem
        if self._want_write_h_at_layer(ell, self.h_attn_n[ell]):
            n_h, d_h, r_h, self.h_attn_buf[ell] = self._nd_update(
                self.h_attn_n[ell],
                self.h_attn_d[ell],
                attn_out,
                bh,
                self.content_h,
                self.h_attn_buf[ell] if self._use_window else None,
            )
            self.h_attn_n[ell], self.h_attn_d[ell] = n_h, d_h
        else:
            r_h = self._axis_readout(
                self.h_attn_n[ell],
                self.h_attn_d[ell],
                self.h_attn_buf[ell] if self._use_window else None,
                bh,
                attn_out,
            )
        n_l, d_l, r_l, self.l_attn_buf = self._nd_update(
            self.l_attn_n,
            self.l_attn_d,
            attn_out,
            bl,
            self.content_l,
            self.l_attn_buf if self._use_window else None,
        )
        self.l_attn_n, self.l_attn_d = n_l, d_l
        return self._fuse(r_h, r_l, attn_out)

    def mix_mlp(self, layer_idx: int, mlp_out: Tensor) -> Tensor:
        if not self.apply_mlp:
            return mlp_out
        if self.h_layout in ("shared", "both"):
            if self.update_order == "pre":
                raise NotImplementedError(
                    "dual_axis_update_order=pre not supported with h_layout shared|both"
                )
            if self.fuse_mode == "cascade":
                raise NotImplementedError(
                    "dual_axis_fuse_mode=cascade not supported with h_layout shared|both"
                )
            return self._mix_shared_or_both(layer_idx, mlp_out, kind="mlp")

        ell = int(layer_idx)
        bh, bl = self.beta_h_for_update(), self.beta_l()
        if self.fuse_mode == "cascade":
            n_l, d_l, r_l, self.l_mlp_buf = self._nd_update(
                self.l_mlp_n,
                self.l_mlp_d,
                mlp_out,
                bl,
                self.content_l,
                self.l_mlp_buf if self._use_window else None,
            )
            self.l_mlp_n, self.l_mlp_d = n_l, d_l
            if self._want_write_h_at_layer(ell, self.h_mlp_n[ell]):
                n_h, d_h, r_h, self.h_mlp_buf[ell] = self._nd_update(
                    self.h_mlp_n[ell],
                    self.h_mlp_d[ell],
                    r_l if self.h_update_src == "l" else mlp_out,
                    bh,
                    self.content_h,
                    self.h_mlp_buf[ell] if self._use_window else None,
                )
                self.h_mlp_n[ell], self.h_mlp_d[ell] = n_h, d_h
            else:
                r_h = self._axis_readout(
                    self.h_mlp_n[ell],
                    self.h_mlp_d[ell],
                    self.h_mlp_buf[ell] if self._use_window else None,
                    bh,
                    mlp_out,
                )
            return r_h
        if self.update_order == "pre":
            r_h = self._axis_readout(
                self.h_mlp_n[ell],
                self.h_mlp_d[ell],
                self.h_mlp_buf[ell] if self._use_window else None,
                bh,
                mlp_out,
            )
            r_l = self._axis_readout(
                self.l_mlp_n,
                self.l_mlp_d,
                self.l_mlp_buf if self._use_window else None,
                bl,
                mlp_out,
            )
            mem = self._fuse(r_h, r_l, mlp_out)
            if self._want_write_h_at_layer(ell, self.h_mlp_n[ell]):
                n_h, d_h, _, self.h_mlp_buf[ell] = self._nd_update(
                    self.h_mlp_n[ell],
                    self.h_mlp_d[ell],
                    mlp_out,
                    bh,
                    self.content_h,
                    self.h_mlp_buf[ell] if self._use_window else None,
                )
                self.h_mlp_n[ell], self.h_mlp_d[ell] = n_h, d_h
            n_l, d_l, _, self.l_mlp_buf = self._nd_update(
                self.l_mlp_n,
                self.l_mlp_d,
                mlp_out,
                bl,
                self.content_l,
                self.l_mlp_buf if self._use_window else None,
            )
            self.l_mlp_n, self.l_mlp_d = n_l, d_l
            return mem
        if self._want_write_h_at_layer(ell, self.h_mlp_n[ell]):
            n_h, d_h, r_h, self.h_mlp_buf[ell] = self._nd_update(
                self.h_mlp_n[ell],
                self.h_mlp_d[ell],
                mlp_out,
                bh,
                self.content_h,
                self.h_mlp_buf[ell] if self._use_window else None,
            )
            self.h_mlp_n[ell], self.h_mlp_d[ell] = n_h, d_h
        else:
            r_h = self._axis_readout(
                self.h_mlp_n[ell],
                self.h_mlp_d[ell],
                self.h_mlp_buf[ell] if self._use_window else None,
                bh,
                mlp_out,
            )
        n_l, d_l, r_l, self.l_mlp_buf = self._nd_update(
            self.l_mlp_n,
            self.l_mlp_d,
            mlp_out,
            bl,
            self.content_l,
            self.l_mlp_buf if self._use_window else None,
        )
        self.l_mlp_n, self.l_mlp_d = n_l, d_l
        return self._fuse(r_h, r_l, mlp_out)

    def snapshot(self) -> dict:
        def _det(xs: List[Optional[Tensor]]) -> List[Optional[Tensor]]:
            return [v.detach() if v is not None else None for v in xs]

        def _det_buf(bufs: List[_Buf]) -> List[_Buf]:
            out: List[_Buf] = []
            for buf in bufs:
                out.append([(a.detach(), b.detach()) for a, b in buf])
            return out

        snap = {
            "h_attn_n": _det(self.h_attn_n),
            "h_attn_d": _det(self.h_attn_d),
            "h_mlp_n": _det(self.h_mlp_n),
            "h_mlp_d": _det(self.h_mlp_d),
            "h_shared_attn_n": self.h_shared_attn_n.detach() if self.h_shared_attn_n is not None else None,
            "h_shared_attn_d": self.h_shared_attn_d.detach() if self.h_shared_attn_d is not None else None,
            "h_shared_mlp_n": self.h_shared_mlp_n.detach() if self.h_shared_mlp_n is not None else None,
            "h_shared_mlp_d": self.h_shared_mlp_d.detach() if self.h_shared_mlp_d is not None else None,
            "loop_t": int(self.loop_t),
        }
        if self._use_window:
            snap["h_attn_buf"] = _det_buf(self.h_attn_buf)
            snap["h_mlp_buf"] = _det_buf(self.h_mlp_buf)
            snap["h_shared_attn_buf"] = [(a.detach(), b.detach()) for a, b in self.h_shared_attn_buf]
            snap["h_shared_mlp_buf"] = [(a.detach(), b.detach()) for a, b in self.h_shared_mlp_buf]
        return snap

    def restore(self, snap: dict) -> None:
        if not snap:
            return
        for key in ("h_attn_n", "h_attn_d", "h_mlp_n", "h_mlp_d"):
            if key in snap and snap[key] is not None:
                setattr(self, key, list(snap[key]))
        for key in (
            "h_shared_attn_n",
            "h_shared_attn_d",
            "h_shared_mlp_n",
            "h_shared_mlp_d",
        ):
            if key in snap:
                setattr(self, key, snap[key])
        if self._use_window:
            if "h_attn_buf" in snap:
                self.h_attn_buf = [list(b) for b in snap["h_attn_buf"]]
            if "h_mlp_buf" in snap:
                self.h_mlp_buf = [list(b) for b in snap["h_mlp_buf"]]
            if "h_shared_attn_buf" in snap:
                self.h_shared_attn_buf = list(snap["h_shared_attn_buf"])
            if "h_shared_mlp_buf" in snap:
                self.h_shared_mlp_buf = list(snap["h_shared_mlp_buf"])
        if not self.l_persist:
            self.l_attn_n = self.l_attn_d = None
            self.l_mlp_n = self.l_mlp_d = None
            self.l_attn_buf = []
            self.l_mlp_buf = []
        if "loop_t" in snap:
            self.loop_t = int(snap["loop_t"])
