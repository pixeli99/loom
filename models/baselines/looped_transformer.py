"""Weight-tied Looped Llama + soft residual scaling (+ optional inject / AttnRes).

Papers:
  Curse of Depth → LNS γ_ℓ = ℓ^{-p}
  Residual Scaling → soft ε(t)=λ/(t_eff+t0)
  Attention Residuals (arXiv:2603.15031) → depth-wise softmax mix
  Input injection (Anil/Zhu looped LM) → re-anchor with input each loop
  Embed mix (W14) → loop-boundary convex mix (β-sensitive; prefer add_soft)

Default: full BP through every loop; no early-exit.
Optional FPRM-style segmented BP (bp_segment_len=K>0) is driven by the train
loop: every K loops → CE → backward → optim.step → detach z (+ AttnRes bank).
ε(t) and inject γ=t/(H√L) stay on the global loop index t=1..H (not reset per
segment). AttnRes scope=loop keeps the full-H value bank; segment boundaries
only detach prior slots so later loops can still mix over earlier loops.
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from pydantic import ConfigDict
from torch import Tensor, nn

from models.attn_res import AttnResBank
from models.dual_axis_carry import DualAxisCarry
from models.transformer import Transformer, TransformerConfig, Cache


class LoopedTransformerConfig(TransformerConfig):
    model_config = ConfigDict(extra="ignore")

    num_loops: int = 1

    cycle_residual_scale: bool = False
    cycle_scale_lambda: float = 2.0
    # Frozen-λ training schedule: none keeps cycle_scale_lambda; linear interpolates
    # start→end over total_steps (step 0 = start, step=total_steps = end).
    cycle_scale_lambda_end: float = 0.0
    cycle_scale_lambda_sched: str = "none"  # none | linear
    cycle_scale_lambda_learnable: bool = False
    # Floor on learnable λ. Pack:
    #   softplus — λ = λ_min + softplus(θ)
    #   sigmoid  — λ = λ_min + amp * sigmoid(θ)   (bounded; mid-range grads like AttnRes β)
    #   tanh     — λ = λ_min + amp * tanh(θ)      (bounded; mid jac=amp; α=amp)
    # Init so λ≈cycle_scale_lambda (default 2).
    cycle_scale_lambda_min: float = 0.0
    cycle_scale_lambda_pack: str = "softplus"  # softplus | sigmoid | tanh
    cycle_scale_lambda_amp: float = 1.0  # sigmoid/tanh: λ=min+amp·pack(θ)
    # Softplus ceiling (0=off). Prevents FSDP-sum×SGD runaway; tanh already bounded by amp.
    cycle_scale_lambda_max: float = 2.5
    # CoD/residual-scaling: match loop-Δz RMS ratios so λ gets a stability signal
    # (CE alone does not drive λ→1.6–1.7; final-z RMS is LN-flat). Unitless.
    cycle_lambda_rms_match: bool = False
    # rms7b: detach CE→λ only in [mid,hi]; rms8: detach whenever λ≥mid (stops hi overshoot).
    cycle_lambda_detach_mode: str = "rms7b"
    # Depth-transfer attractor (fixed-λ BEST); prior=0 inside [lo,hi].
    cycle_lambda_depth_prior: float = 1.6
    cycle_lambda_depth_prior_coef: float = 1.0
    cycle_lambda_depth_prior_lo: float = 1.58
    cycle_lambda_depth_prior_hi: float = 1.75
    # Sigmoid/tanh init clamp: |frac|∈[eps,1−eps] so jac≠0 at requested λ≈1.5/2.
    cycle_scale_lambda_sig_eps: float = 0.05
    # How learnable λ is packaged:
    #   shared    — one λ for residual ε and soft-inject g (Arm A)
    #   split     — λ_res for residual, λ_inj for soft inject (independent)
    #   per_layer — λ_ℓ per physical layer for residual; λ_inj scalar for inject
    cycle_scale_lambda_mode: str = "shared"
    # t0 removed from recipes (always 0). Kept for backward-compat overrides only.
    cycle_eps_t0: float = 0.0
    cycle_eps_shared_span: float = 0.0
    cycle_eps_span_mode: str = "progress"  # progress | clamp
    # Time schedule for residual/inject ε (span=0):
    #   inv_t       ε ∝ λ / t_eff          (dense-style; MoE late loops may be too small)
    #   inv_sqrt_t  ε ∝ λ / √t_eff         (slower decay; MoE deepen candidate)
    #   inv_H       ε ∝ λ / H              (same scale every loop; H=num_loops)
    cycle_eps_time_mode: str = "inv_t"
    # First-loop-only residual soften. extra=0 → off. extra=1, loops=1 → ε1=λ/2.
    cycle_eps_first_loops: int = 1
    cycle_eps_first_extra: float = 0.0
    # Floor on loop index: t_eff = max(t, floor). floor=1 → off (t≥1).
    # floor=2 → loop1 uses 1/(t+1); floor=3 → loop1 1/(t+2), loop2 1/(t+1); later 1/t.
    cycle_eps_t_floor: float = 1.0

    # Embed reinject at loop boundary (t>=2); default convex z←(1−γ)z+γ·x
    # Soft-ε family: add_soft | mix_soft | soft_comp | soft_prod | eps*_ramp
    # CoD inject family — ONLY t,H,L (and optional shared residual t0). No inject β.
    #   sqrtL     γ = t/(H·√L)
    #   sqrtLp1   γ = t/(H·√(L+1))
    #   sqrtLp2   γ = t/(H·√(L+2))      # PRIMARY: near mix030@L8 (0.316); =sqrtL_t0 if t0=2
    #   sqrtLp3   γ = t/(H·√(L+3))      # PRIMARY: closest mix030@L8 (0.3015)
    #   sqrtLm1   γ = t/(H·√(max(L-1,1)))
    #   sqrtL_t0  γ = t/(H·√(L+t0))     # alias of Lp2 when t0=2
    # Soft-ε family / hand-β mix: legacy
    # depth_xfer: optional legacy (needs β★,L★) — prefer NOT to use
    embed_inject_mode: str = "none"
    embed_inject_beta: float = 0.25
    embed_inject_learnable: bool = False
    # legacy depth_xfer only (unused by recommended recipes)
    embed_inject_beta_ref: float = 0.30
    embed_inject_L_ref: float = 8.0
    # RMSNorm at inject mix z←(1−γ)z+γx. none | z | x | both (parameter-free).
    embed_inject_norm: str = "none"
    # If true, also inject at t=1 (default skips t=1 because z≡embed ⇒ mix is a no-op
    # unless t1 style remaps the mix). empty = (1−g)z; rms_x = (1−g)z + g·RMS(embed).
    embed_inject_from_t1: bool = False
    embed_inject_t1_style: str = "rms_x"  # rms_x | empty | same | unit
    # empty@t=1 keep. <0 → (1−g); ≥0 → z←keep·z (user "small coefficient").
    embed_inject_t1_keep: float = -1.0
    # start = before layers (default). end = after layers, t=1 mix is no longer a no-op.
    embed_inject_when: str = "start"  # start | end | both
    # If >0, z←RMS(z) at start of t=1..N and skip mix on those loops. 0 + style=unit → N=1.
    embed_inject_unit_loops: int = 0
    # Soft-inject time denom, independent of residual cycle_eps_time_mode.
    #   inherit — use residual schedule (old: inv_h ⇒ g=λ/H every loop)
    #   inv_t / inv_h / inv_sqrt_t — inject-only. Residual stays on cycle_eps_time_mode.
    embed_inject_time_mode: str = "inherit"
    # If true, soft-inject also ÷√L (g=λ/(t√L) with inv_t). Independent of residual LNS.
    embed_inject_depth_scale: bool = False
    # False (1): z←(1−g)z + g·x   g=λ/t ⇒ early loops heavier embed
    # True  (2): z←g·z + (1−g)·x  g=λ/t ⇒ late loops heavier embed
    embed_inject_swap: bool = False

    # Loop-ID embedding: tell the model which loop index t∈[1..H] it is on.
    # none | add (H learnable vecs at loop start) | add_layer (H×L vecs before each layer)
    #       | sin (fixed PE) | scale (multiplicative 1+α·tanh(s_t))
    loop_id_embed_mode: str = "none"
    loop_id_embed_scale: float = 1.0
    loop_id_embed_max: int = 16  # embedding / scalar table size (covers H≤16)

    # AttnRes scopes:
    #   intra       — clear bank each loop (within one loop only)
    #   loop        — full history (deprecated: slow/OOM)
    #   loop_carry      — O(1) EMA hidden carry across loops; intra bank within loop
    #   loop_lag1       — mix with previous loop's bank only (max 2 loops in bank)
    #   loop_hybrid     — lag1 AttnRes + prev-loop hidden snapshot (no EMA accumulation)
    #   loop_carry_lag1 — lag1 AttnRes + EMA hidden carry (combine carry eval + lag1 stability)
    #   dual_axis       — dual-axis loop: H/L running-mean on attn+mlp (no bank; ≈intra speed)
    attn_res: bool = False
    attn_res_scope: str = "intra"
    attn_res_max_slots: int = 256  # 0 = auto by scope
    dual_axis_fuse_half: bool = False  # legacy → fuse_mode=half
    dual_axis_fuse_mode: str = "sum"  # sum | half | cascade | dedup | blend | gated
    dual_axis_blend_lambda: float = 0.25
    dual_axis_beta_init: float = 0.55
    dual_axis_beta_h_init: float = 0.55
    dual_axis_beta_l_init: float = 0.55
    dual_axis_heads: int = 1
    dual_axis_content: bool = True
    dual_axis_content_mode: str = "exp"  # e^{w·RMSNorm(o)/τ}
    dual_axis_content_temp: float = 1.0
    dual_axis_l_persist: bool = False
    dual_axis_apply: str = "attn"
    dual_axis_residual_mode: str = "x_plus_r"  # x_plus_r | replace_r | o_plus_r | x_plus_o_plus_r
    dual_axis_update_src: str = "o"  # o | x
    dual_axis_nd_mode: str = "ema"  # ema|accum|window|window_age|window_softmax|window_smage
    dual_axis_beta_learnable: bool = True
    dual_axis_window_size: int = 4
    # Write H-axis N/D every N loops (1=every loop; 3≈align with K=3 BP). L still every call.
    dual_axis_h_update_every: int = 1
    # Learnable β floors: β = min + (1-min)*sigmoid(raw). 0 = unconstrained.
    dual_axis_beta_h_min: float = 0.0
    dual_axis_beta_l_min: float = 0.0
    # Scale AttnRes residual write by loop index: none | sqrt_t | inv_t
    dual_axis_r_scale_mode: str = "none"
    # Decay β_H on H N/D writes: none | sqrt_t | inv_t (≠ residual rscale)
    dual_axis_beta_h_decay: str = "none"
    # post: update then readout; pre: readout memory then update (use with x_plus_o_plus_r)
    dual_axis_update_order: str = "post"
    # H storage: per_layer | shared (1 N/D) | both (shared + per-layer P)
    dual_axis_h_layout: str = "per_layer"
    # Write H within loop: every_layer | last_layer | loop_mean | first_layer | stride | first_last | mid_layer
    dual_axis_h_write: str = "every_layer"
    # Stride for dual_axis_h_write=stride (default 2 → layers 0,2,4,...)
    dual_axis_h_write_stride: int = 2
    # Shared-H update source: o | l (r_L)
    dual_axis_h_update_src: str = "o"
    dual_axis_beta_p_init: float = 0.55
    # β_L: False = one shared β across loops; True = β_L[t] per loop index
    dual_axis_beta_l_per_loop: bool = False

    # 0 = full BP through all H loops (one backward per batch).
    # K>0 = train-loop segmented BP every K loops (eval still full-H, no inner step).
    bp_segment_len: int = 0


class LoopedTransformer(Transformer):
    def __init__(self, config_dict: dict) -> None:
        config = LoopedTransformerConfig(**config_dict)
        super().__init__(config)
        self.num_loops = max(1, int(config.num_loops))
        self.cycle_residual_scale = bool(config.cycle_residual_scale)
        self.cycle_scale_lambda = float(config.cycle_scale_lambda)
        self.cycle_scale_lambda_end = float(
            getattr(config, "cycle_scale_lambda_end", 0.0) or 0.0
        )
        self.cycle_scale_lambda_sched = str(
            getattr(config, "cycle_scale_lambda_sched", "none") or "none"
        ).lower()
        self._cycle_lambda_sched_start = float(self.cycle_scale_lambda)
        self.cycle_scale_lambda_min = max(
            0.0, float(getattr(config, "cycle_scale_lambda_min", 0.0) or 0.0)
        )
        self.cycle_scale_lambda_learnable = bool(
            getattr(config, "cycle_scale_lambda_learnable", False)
        )
        self.cycle_scale_lambda_pack = str(
            getattr(config, "cycle_scale_lambda_pack", "softplus") or "softplus"
        ).lower()
        if self.cycle_scale_lambda_pack not in (
            "softplus",
            "softplus_center",
            "sigmoid",
            "tanh",
            "exp_prior",
        ):
            raise ValueError(
                f"cycle_scale_lambda_pack={self.cycle_scale_lambda_pack!r}; "
                "expected softplus|softplus_center|sigmoid|tanh|exp_prior"
            )
        self.cycle_scale_lambda_amp = max(
            1e-6, float(getattr(config, "cycle_scale_lambda_amp", 1.0) or 1.0)
        )
        self.cycle_scale_lambda_max = float(
            getattr(config, "cycle_scale_lambda_max", 2.5) or 0.0
        )
        self.cycle_lambda_rms_match = bool(
            getattr(config, "cycle_lambda_rms_match", False)
        )
        self.cycle_lambda_detach_mode = str(
            getattr(config, "cycle_lambda_detach_mode", "rms7b") or "rms7b"
        ).strip().lower()
        # Depth-transfer prior for λ (residual-scaling / fixed-λ BEST ≈1.6).
        # Used as stopgrad attractor gated by loop-Δ imbalance — not pack LR hacks.
        self.cycle_lambda_depth_prior = float(
            getattr(config, "cycle_lambda_depth_prior", 1.6) or 0.0
        )
        self.cycle_lambda_depth_prior_coef = float(
            getattr(config, "cycle_lambda_depth_prior_coef", 1.0) or 1.0
        )
        self.cycle_lambda_depth_prior_lo = float(
            getattr(config, "cycle_lambda_depth_prior_lo", 1.58) or 1.58
        )
        self.cycle_lambda_depth_prior_hi = float(
            getattr(config, "cycle_lambda_depth_prior_hi", 1.75) or 1.75
        )
        # Legacy per-segment stash (unused when full-H ref/last path is active).
        self._lambda_rms_states: list[Tensor] = []
        # Full-H Δz-RMS (not final-z RMS): LN/stream keeps ||z|| flat ⇒ z-RMS match≈0
        # even when intermediate act_norms shrink. Match RMS(z_t−z_{t−1}) instead.
        # ref at t=2 (detached); live Δ at t=H. Survives bp_segment_len boundaries.
        self._lambda_rms_ref: Optional[Tensor] = None
        self._lambda_rms_last: Optional[Tensor] = None
        self._z_prev_for_lambda_rms: Optional[Tensor] = None
        self.cycle_scale_lambda_sig_eps = float(
            getattr(config, "cycle_scale_lambda_sig_eps", 0.05) or 0.05
        )
        self.cycle_scale_lambda_mode = str(
            getattr(config, "cycle_scale_lambda_mode", "shared") or "shared"
        ).lower()
        if self.cycle_scale_lambda_mode not in ("shared", "split", "per_layer"):
            raise ValueError(
                f"cycle_scale_lambda_mode={self.cycle_scale_lambda_mode!r}; "
                "expected shared|split|per_layer"
            )
        self.cycle_eps_t0 = float(config.cycle_eps_t0)
        self.cycle_eps_shared_span = float(getattr(config, "cycle_eps_shared_span", 0.0) or 0.0)
        self.cycle_eps_span_mode = str(getattr(config, "cycle_eps_span_mode", "progress") or "progress").lower()
        self.cycle_eps_time_mode = str(
            getattr(config, "cycle_eps_time_mode", "inv_t") or "inv_t"
        ).lower()
        self.cycle_eps_first_loops = int(getattr(config, "cycle_eps_first_loops", 1) or 0)
        self.cycle_eps_first_extra = float(getattr(config, "cycle_eps_first_extra", 0.0) or 0.0)
        self.cycle_eps_t_floor = float(getattr(config, "cycle_eps_t_floor", 1.0) or 1.0)
        if self.cycle_eps_time_mode not in (
            "inv_t",
            "inv_sqrt_t",
            "inv_h",
            "const_h",
            "t",
            "sqrt_t",
        ):
            raise ValueError(
                f"cycle_eps_time_mode={self.cycle_eps_time_mode!r}; "
                "expected inv_t|inv_sqrt_t|inv_h"
            )
        # Aliases
        if self.cycle_eps_time_mode == "t":
            self.cycle_eps_time_mode = "inv_t"
        if self.cycle_eps_time_mode == "sqrt_t":
            self.cycle_eps_time_mode = "inv_sqrt_t"
        if self.cycle_eps_time_mode == "const_h":
            self.cycle_eps_time_mode = "inv_h"
        self._n_layers = max(1, int(config.n_layers))
        self._sqrt_L = math.sqrt(self._n_layers)
        self._depth_via_ell = bool(config.layernorm_scaling) or bool(config.residual_depth_scale)

        # Pack λ from θ; init so packed(θ0) ≈ cycle_scale_lambda.
        self.cycle_lambda_raw: Optional[nn.Parameter] = None
        self.inject_lambda_raw: Optional[nn.Parameter] = None
        if self.cycle_scale_lambda_learnable and self.cycle_residual_scale:
            lam0 = max(float(self.cycle_scale_lambda), 1e-3)
            raw0 = self._lambda_raw0_from_init(lam0)
            mode = self.cycle_scale_lambda_mode
            if mode == "per_layer":
                self.cycle_lambda_raw = nn.Parameter(
                    torch.full((self._n_layers,), raw0, dtype=torch.float32)
                )
                # Inject keeps a separate scalar λ_inj (same init); no extra γ.
                self.inject_lambda_raw = nn.Parameter(
                    torch.tensor([raw0], dtype=torch.float32)
                )
            elif mode == "split":
                self.cycle_lambda_raw = nn.Parameter(torch.tensor([raw0], dtype=torch.float32))
                self.inject_lambda_raw = nn.Parameter(torch.tensor([raw0], dtype=torch.float32))
            else:  # shared
                self.cycle_lambda_raw = nn.Parameter(torch.tensor([raw0], dtype=torch.float32))

        self.embed_inject_mode = str(getattr(config, "embed_inject_mode", "none") or "none")
        self.embed_inject_beta = float(getattr(config, "embed_inject_beta", 0.25) or 0.0)
        self.embed_inject_learnable = bool(getattr(config, "embed_inject_learnable", False))
        self.embed_inject_norm = str(getattr(config, "embed_inject_norm", "none") or "none").lower()
        if self.embed_inject_norm not in ("none", "z", "x", "both"):
            raise ValueError(
                f"embed_inject_norm={self.embed_inject_norm!r}; expected none|z|x|both"
            )
        self.embed_inject_from_t1 = bool(getattr(config, "embed_inject_from_t1", False))
        self.embed_inject_t1_style = str(
            getattr(config, "embed_inject_t1_style", "rms_x") or "rms_x"
        ).lower()
        if self.embed_inject_t1_style not in ("rms_x", "empty", "same", "unit"):
            raise ValueError(
                f"embed_inject_t1_style={self.embed_inject_t1_style!r}; expected rms_x|empty|same|unit"
            )
        self.embed_inject_t1_keep = float(getattr(config, "embed_inject_t1_keep", -1.0) or -1.0)
        self.embed_inject_when = str(getattr(config, "embed_inject_when", "start") or "start").lower()
        if self.embed_inject_when not in ("start", "end", "both"):
            raise ValueError(
                f"embed_inject_when={self.embed_inject_when!r}; expected start|end|both"
            )
        self.embed_inject_unit_loops = int(getattr(config, "embed_inject_unit_loops", 0) or 0)
        self.embed_inject_time_mode = str(
            getattr(config, "embed_inject_time_mode", "inherit") or "inherit"
        ).lower()
        if self.embed_inject_time_mode not in (
            "inherit",
            "none",
            "",
            "inv_t",
            "inv_h",
            "inv_sqrt_t",
            "t",
            "sqrt_t",
            "const_h",
        ):
            raise ValueError(
                f"embed_inject_time_mode={self.embed_inject_time_mode!r}; "
                "expected inherit|inv_t|inv_h|inv_sqrt_t"
            )
        if self.embed_inject_time_mode == "t":
            self.embed_inject_time_mode = "inv_t"
        if self.embed_inject_time_mode == "sqrt_t":
            self.embed_inject_time_mode = "inv_sqrt_t"
        if self.embed_inject_time_mode == "const_h":
            self.embed_inject_time_mode = "inv_h"
        self.embed_inject_depth_scale = bool(
            getattr(config, "embed_inject_depth_scale", False)
        )
        self.embed_inject_swap = bool(getattr(config, "embed_inject_swap", False))
        self.embed_gamma_raw: Optional[nn.Parameter] = None
        if self.embed_inject_learnable and self.embed_inject_mode not in ("", "none"):
            # mix/add: init softplus≈β; *_soft: init softplus≈1 (scale on ε(t))
            if self.embed_inject_mode in ("loop_start_add_soft", "loop_start_mix_soft", "loop_start_mix_eps1_ramp", "loop_start_mix_eps1_half", "loop_start_mix_eps3_ramp", "loop_start_mix_soft_comp", "loop_start_mix_soft_prod"):
                g0 = 1.0
            else:
                g0 = max(float(self.embed_inject_beta), 1e-3)
            raw = math.log(math.expm1(g0)) if g0 < 20.0 else g0
            self.embed_gamma_raw = nn.Parameter(torch.tensor([raw], dtype=torch.float32))

        # Loop-ID signal (orthogonal to router_per_loop; works for dense + MoE)
        self.loop_id_embed_mode = str(
            getattr(config, "loop_id_embed_mode", "none") or "none"
        ).lower()
        self.loop_id_embed_scale = float(getattr(config, "loop_id_embed_scale", 1.0) or 1.0)
        _lid_max = max(
            int(getattr(config, "loop_id_embed_max", 16) or 16),
            int(self.num_loops),
            1,
        )
        self._loop_id_t: int = 1  # 1-based current loop (for add_layer)
        self._loop_id_max_h: int = _lid_max
        self.loop_id_emb: Optional[nn.Embedding] = None
        self.loop_id_raw: Optional[nn.Parameter] = None
        self.loop_id_layer_emb: Optional[nn.Embedding] = None
        if self.loop_id_embed_mode == "add":
            # H learnable vectors, added once at each loop start
            self.loop_id_emb = nn.Embedding(_lid_max, int(config.hidden_size))
            nn.init.normal_(self.loop_id_emb.weight, mean=0.0, std=0.02)
        elif self.loop_id_embed_mode == "add_layer":
            # H×L learnable vectors, added before every layer in every loop
            n_layers = int(config.n_layers)
            self.loop_id_layer_emb = nn.Embedding(
                _lid_max * max(n_layers, 1), int(config.hidden_size)
            )
            nn.init.normal_(self.loop_id_layer_emb.weight, mean=0.0, std=0.02)
        elif self.loop_id_embed_mode == "scale":
            self.loop_id_raw = nn.Parameter(torch.zeros(_lid_max, dtype=torch.float32))
        elif self.loop_id_embed_mode not in ("", "none", "sin"):
            raise ValueError(
                f"loop_id_embed_mode={self.loop_id_embed_mode!r}; "
                "expected none|add|add_layer|sin|scale"
            )

        self.attn_res = bool(getattr(config, "attn_res", False))
        self.attn_res_scope = str(getattr(config, "attn_res_scope", "intra") or "intra").lower()
        self.attn_res_max_slots = int(getattr(config, "attn_res_max_slots", 256) or 0)
        self.bp_segment_len = max(0, int(getattr(config, "bp_segment_len", 0) or 0))
        self.dual_axis_fuse_half = bool(getattr(config, "dual_axis_fuse_half", False))
        self.dual_axis_fuse_mode = str(
            getattr(config, "dual_axis_fuse_mode", "sum") or "sum"
        ).lower()
        self.dual_axis_blend_lambda = float(
            getattr(config, "dual_axis_blend_lambda", 0.25) or 0.25
        )
        self.dual_axis_beta_init = float(getattr(config, "dual_axis_beta_init", 0.55) or 0.55)
        self.dual_axis_content_mode = str(
            getattr(config, "dual_axis_content_mode", "exp") or "exp"
        ).lower()
        self.dual_axis_content_temp = float(
            getattr(config, "dual_axis_content_temp", 1.0) or 1.0
        )
        self.dual_axis_beta_h_init = float(
            getattr(config, "dual_axis_beta_h_init", self.dual_axis_beta_init) or self.dual_axis_beta_init
        )
        self.dual_axis_beta_l_init = float(
            getattr(config, "dual_axis_beta_l_init", self.dual_axis_beta_init) or self.dual_axis_beta_init
        )
        self.dual_axis_heads = max(1, int(getattr(config, "dual_axis_heads", 1) or 1))
        self.dual_axis_content = bool(getattr(config, "dual_axis_content", True))
        self.dual_axis_l_persist = bool(getattr(config, "dual_axis_l_persist", False))
        self.dual_axis_apply = str(getattr(config, "dual_axis_apply", "attn") or "attn").lower()
        self.dual_axis_residual_mode = str(
            getattr(config, "dual_axis_residual_mode", "x_plus_r") or "x_plus_r"
        ).lower()
        self.dual_axis_update_src = str(
            getattr(config, "dual_axis_update_src", "o") or "o"
        ).lower()
        self.dual_axis_nd_mode = str(
            getattr(config, "dual_axis_nd_mode", "ema") or "ema"
        ).lower()
        self.dual_axis_beta_learnable = bool(
            getattr(config, "dual_axis_beta_learnable", True)
        )
        self.dual_axis_window_size = max(
            1, int(getattr(config, "dual_axis_window_size", 4) or 4)
        )
        self.dual_axis_h_update_every = max(
            1, int(getattr(config, "dual_axis_h_update_every", 1) or 1)
        )
        self.dual_axis_beta_h_min = float(
            getattr(config, "dual_axis_beta_h_min", 0.0) or 0.0
        )
        self.dual_axis_beta_l_min = float(
            getattr(config, "dual_axis_beta_l_min", 0.0) or 0.0
        )
        self.dual_axis_r_scale_mode = str(
            getattr(config, "dual_axis_r_scale_mode", "none") or "none"
        ).lower()
        self.dual_axis_beta_h_decay = str(
            getattr(config, "dual_axis_beta_h_decay", "none") or "none"
        ).lower()
        self.dual_axis_update_order = str(
            getattr(config, "dual_axis_update_order", "post") or "post"
        ).lower()
        self.dual_axis_h_layout = str(
            getattr(config, "dual_axis_h_layout", "per_layer") or "per_layer"
        ).lower()
        self.dual_axis_h_write = str(
            getattr(config, "dual_axis_h_write", "every_layer") or "every_layer"
        ).lower()
        self.dual_axis_h_write_stride = max(
            1, int(getattr(config, "dual_axis_h_write_stride", 2) or 2)
        )
        self.dual_axis_h_update_src = str(
            getattr(config, "dual_axis_h_update_src", "o") or "o"
        ).lower()
        self.dual_axis_beta_p_init = float(
            getattr(config, "dual_axis_beta_p_init", self.dual_axis_beta_h_init)
            or self.dual_axis_beta_h_init
        )
        self.dual_axis_beta_l_per_loop = bool(
            getattr(config, "dual_axis_beta_l_per_loop", False)
        )
        self.attn_res_bank: Optional[AttnResBank] = None
        self.dual_axis: Optional[DualAxisCarry] = None
        self._z_carry: Optional[Tensor] = None
        self._z_prev_loop: Optional[Tensor] = None
        self._prev_loop_bank: Optional[list[Tensor]] = None
        if self.attn_res and self.attn_res_scope == "dual_axis" and self.dual_axis_apply != "none":
            # Winner N/D: β (learnable) × content e^{w·RMSNorm(o)/τ}; H carry across K-seg.
            self.dual_axis = DualAxisCarry(
                n_layers=self._n_layers,
                hidden_size=int(config.hidden_size),
                beta_init=self.dual_axis_beta_init,
                beta_h_init=self.dual_axis_beta_h_init,
                beta_l_init=self.dual_axis_beta_l_init,
                n_heads=self.dual_axis_heads,
                use_content=self.dual_axis_content,
                content_mode=self.dual_axis_content_mode,
                content_temp=self.dual_axis_content_temp,
                fuse_half=self.dual_axis_fuse_half,
                fuse_mode=self.dual_axis_fuse_mode,
                blend_lambda=self.dual_axis_blend_lambda,
                l_persist=self.dual_axis_l_persist,
                apply=self.dual_axis_apply,
                residual_mode=self.dual_axis_residual_mode,
                update_src=self.dual_axis_update_src,
                nd_mode=self.dual_axis_nd_mode,
                beta_learnable=self.dual_axis_beta_learnable,
                window_size=self.dual_axis_window_size,
                h_update_every=self.dual_axis_h_update_every,
                beta_h_min=self.dual_axis_beta_h_min,
                beta_l_min=self.dual_axis_beta_l_min,
                r_scale_mode=self.dual_axis_r_scale_mode,
                beta_h_decay=self.dual_axis_beta_h_decay,
                update_order=self.dual_axis_update_order,
                h_layout=self.dual_axis_h_layout,
                h_write=self.dual_axis_h_write,
                h_write_stride=self.dual_axis_h_write_stride,
                h_update_src=self.dual_axis_h_update_src,
                beta_p_init=self.dual_axis_beta_p_init,
                beta_l_per_loop=self.dual_axis_beta_l_per_loop,
                max_loops=max(1, int(self.num_loops)),
            )
        elif self.attn_res and self.attn_res_scope != "dual_axis":
            per_loop = 1 + 2 * int(self._n_layers)
            slots = self.attn_res_max_slots if self.attn_res_max_slots > 0 else 256
            scope = self.attn_res_scope
            if scope == "loop":
                slots = max(slots, int(self.num_loops) * per_loop + 8)
            elif scope in ("loop_lag1", "loop_hybrid", "loop_carry_lag1"):
                slots = max(slots, 2 * per_loop + 8)
            else:
                # intra, loop_carry: one loop depth in bank at a time
                slots = max(slots, per_loop + 8)
            self.attn_res_max_slots = slots
            self.attn_res_bank = AttnResBank(config.hidden_size, max_slots=slots)

    def _slots_per_loop(self) -> int:
        return 1 + 2 * int(self._n_layers)

    def _uses_dual_axis(self) -> bool:
        return self.dual_axis is not None

    def _uses_lag1_bank(self) -> bool:
        return self.attn_res_scope in ("loop_lag1", "loop_hybrid", "loop_carry_lag1")

    def _uses_ema_carry(self) -> bool:
        return self.attn_res_scope in ("loop_carry", "loop_carry_lag1")

    def _uses_prev_loop_z(self) -> bool:
        return self.attn_res_scope == "loop_hybrid"

    def _save_prev_loop_bank(self, bank: AttnResBank) -> None:
        n_per = self._slots_per_loop()
        if len(bank._values) >= n_per:
            self._prev_loop_bank = [v.detach() for v in bank._values[-n_per:]]

    def _update_ema_carry(self, t: int, z: Tensor) -> None:
        h = float(t)
        if self._z_carry is None:
            self._z_carry = z.detach()
        else:
            self._z_carry = self._z_carry * (h - 1.0) / h + z.detach() * (1.0 / h)

    def reset_cross_loop_state(self) -> None:
        self._z_carry = None
        self._z_prev_loop = None
        self._prev_loop_bank = None
        if self.attn_res_bank is not None:
            self.attn_res_bank.clear()
        if self.dual_axis is not None:
            self.dual_axis.clear()

    def cross_loop_snapshot(self) -> dict:
        """Detach cross-loop state at K-segment BP boundary.

        Dual-axis H N/D **values** continue into the next segment via restore;
        only the autograd graph is cut (forward AttnRes memory stays connected).
        """
        out: dict = {}
        if self.attn_res_bank is not None:
            out["bank"] = self.attn_res_bank.snapshot()
        if self.dual_axis is not None:
            out["dual_axis"] = self.dual_axis.snapshot()
        if self._uses_ema_carry() and self._z_carry is not None:
            out["z_carry"] = self._z_carry.detach()
        if self._uses_prev_loop_z() and self._z_prev_loop is not None:
            out["z_prev_loop"] = self._z_prev_loop.detach()
        if self._uses_lag1_bank() and self._prev_loop_bank is not None:
            out["prev_loop_bank"] = [v.detach() for v in self._prev_loop_bank]
        return out

    def cross_loop_restore(self, snap: dict) -> None:
        if not snap:
            return
        bank = self.attn_res_bank
        if bank is not None and "bank" in snap:
            bank.restore(snap["bank"])
        if self.dual_axis is not None and "dual_axis" in snap:
            self.dual_axis.restore(snap["dual_axis"])
        if self._uses_ema_carry():
            self._z_carry = snap.get("z_carry")
        if self._uses_prev_loop_z():
            self._z_prev_loop = snap.get("z_prev_loop")
        if self._uses_lag1_bank():
            plb = snap.get("prev_loop_bank")
            self._prev_loop_bank = list(plb) if plb else None

    def _cycle_t_eff(self, t: int) -> float:
        t = float(max(1, int(t)))
        floor = float(getattr(self, "cycle_eps_t_floor", 1.0) or 1.0)
        if floor > 1.0:
            t = max(t, floor)
        span = float(self.cycle_eps_shared_span)
        if span <= 0.0:
            return t
        if self.cycle_eps_span_mode == "clamp":
            return min(t, span)
        return (t / float(self.num_loops)) * span

    def _cycle_time_denom(self, t: int, *, time_mode: Optional[str] = None) -> float:
        """Denominator from loop index only: t_eff or √t_eff (+ optional legacy t0)."""
        mode = str(
            time_mode
            if time_mode is not None
            else (getattr(self, "cycle_eps_time_mode", "inv_t") or "inv_t")
        ).lower()
        if mode == "inv_h":
            return float(max(int(getattr(self, "num_loops", 1) or 1), 1))
        te = self._cycle_t_eff(t) + max(0.0, self.cycle_eps_t0)
        te = max(te, 1e-6)
        if mode == "inv_sqrt_t":
            return math.sqrt(te)
        return te

    def _lambda_raw0_from_init(self, lam0: float) -> float:
        """Invert packaging so packed(θ0) ≈ lam0 (clamp away from exact ±1 for sigmoid/tanh).

        For pack=sigmoid/tanh, frac is clamped so jac≠0 at requested λ≈1.5/2.
        """
        if self.cycle_scale_lambda_pack == "sigmoid":
            # λ = λ_min + amp * σ(θ)  →  σ = (λ−λ_min)/amp
            frac = (float(lam0) - self.cycle_scale_lambda_min) / max(self.cycle_scale_lambda_amp, 1e-8)
            eps = float(getattr(self, "cycle_scale_lambda_sig_eps", 0.05) or 0.05)
            eps = min(max(eps, 1e-3), 0.2)
            frac = min(max(frac, eps), 1.0 - eps)
            return math.log(frac / (1.0 - frac))
        if self.cycle_scale_lambda_pack == "tanh":
            # λ = λ_min + amp * tanh(θ)  →  tanh = (λ−λ_min)/amp
            frac = (float(lam0) - self.cycle_scale_lambda_min) / max(self.cycle_scale_lambda_amp, 1e-8)
            eps = float(getattr(self, "cycle_scale_lambda_sig_eps", 0.05) or 0.05)
            eps = min(max(eps, 1e-3), 0.2)
            frac = min(max(frac, -1.0 + eps), 1.0 - eps)
            # artanh(frac)
            return 0.5 * math.log((1.0 + frac) / (1.0 - frac))
        if self.cycle_scale_lambda_pack == "exp_prior":
            # λ = λ_prior · exp(θ); θ=0 ⇒ λ=prior (~1.6). Init θ=log(lam0/prior).
            # AdamW WD on θ pulls toward prior without an auxiliary CE term.
            prior = float(getattr(self, "cycle_lambda_depth_prior", 1.6) or 1.6)
            prior = max(prior, 1e-3)
            return math.log(max(float(lam0), 1e-3) / prior)
        if self.cycle_scale_lambda_pack == "softplus_center":
            # λ = λ_prior + softplus(θ) − softplus(0); θ=0 ⇒ λ=prior.
            # Same softplus nonlinearity; WD→0 centers at paper λ≈1.6 (no aux loss).
            prior = float(getattr(self, "cycle_lambda_depth_prior", 1.6) or 1.6)
            sp0 = math.log(2.0)  # softplus(0)
            delta = max(float(lam0) - prior + sp0, 0.05)
            return math.log(math.expm1(delta)) if delta < 20.0 else delta
        # softplus: λ = λ_min + softplus(θ)
        # Floor softplus≥0.05 so "init≈λ_min" still has healthy jac (σ(θ)≳0.05).
        # Exact softplus=0 ⇒ θ→−∞ ⇒ jac→0 ⇒ λ frozen (observed on soft_init15).
        delta = max(float(lam0) - self.cycle_scale_lambda_min, 0.05)
        return math.log(math.expm1(delta)) if delta < 20.0 else delta

    def _pack_lambda(self, raw: Tensor) -> Tensor:
        """Pack θ→λ. softplus / sigmoid / tanh."""
        if self.cycle_scale_lambda_pack == "sigmoid":
            lam = self.cycle_scale_lambda_amp * torch.sigmoid(raw)
            if self.cycle_scale_lambda_min > 0.0:
                lam = lam + self.cycle_scale_lambda_min
            return lam
        if self.cycle_scale_lambda_pack == "tanh":
            lam = self.cycle_scale_lambda_amp * torch.tanh(raw)
            if self.cycle_scale_lambda_min > 0.0:
                lam = lam + self.cycle_scale_lambda_min
            return lam
        if self.cycle_scale_lambda_pack == "exp_prior":
            prior = float(getattr(self, "cycle_lambda_depth_prior", 1.6) or 1.6)
            return float(prior) * torch.exp(raw)
        if self.cycle_scale_lambda_pack == "softplus_center":
            prior = float(getattr(self, "cycle_lambda_depth_prior", 1.6) or 1.6)
            # softplus(0)=ln2; θ=0 ⇒ λ=prior.
            return float(prior) + F.softplus(raw) - math.log(2.0)
        lam = F.softplus(raw)
        if self.cycle_scale_lambda_min > 0.0:
            lam = lam + self.cycle_scale_lambda_min
        if self.cycle_scale_lambda_max > 0.0:
            lam = lam.clamp(max=self.cycle_scale_lambda_max)
        return lam

    def _lambda_pack_jacobian(self, raw: Tensor) -> Tensor:
        """∂λ/∂θ for current pack (elementwise). Used for λ-space grad compensation."""
        if self.cycle_scale_lambda_pack == "sigmoid":
            s = torch.sigmoid(raw)
            return self.cycle_scale_lambda_amp * s * (1.0 - s)
        if self.cycle_scale_lambda_pack == "tanh":
            th = torch.tanh(raw)
            return self.cycle_scale_lambda_amp * (1.0 - th * th)
        if self.cycle_scale_lambda_pack == "exp_prior":
            # ∂λ/∂θ = λ_prior · exp(θ) = λ
            prior = float(getattr(self, "cycle_lambda_depth_prior", 1.6) or 1.6)
            return float(prior) * torch.exp(raw)
        # softplus / softplus_center: softplus'(θ) = sigmoid(θ)
        return torch.sigmoid(raw)

    @torch.no_grad()
    def compensate_cycle_lambda_grads_to_lambda_space(self, eps: float = 1e-4) -> dict[str, float]:
        """Pull θ-grads back so an optimizer step yields Δλ ≈ −lr · ∂L/∂λ (pack-invariant).

        Autograd: ∂L/∂θ = (∂L/∂λ)·jac, jac=∂λ/∂θ.
        Euclidean descent on λ: Δλ = −lr·∂L/∂λ ⇒ Δθ = Δλ/jac = −lr·∂L/∂λ/jac
          = −lr·(∂L/∂θ)/jac².
        So we must divide θ-grads by **jac²** (not jac). Dividing once only gives
        Δλ = −lr·jac·∂L/∂λ, which re-introduces pack/floor shrinkage (softplus near
        λ_min freezes again). This is the dense→MoE μP transfer for a scalar residual
        gain: step in the coordinate you care about (λ), independent of packaging.

        Also divides by world_size: FSDP2 ``fully_shard`` sum-reduces grads; plain SGD
        on λ is not scale-invariant (Adam is). Without ``/ws``, Δλ blows up ~64×
        (lamspace_p2c H6: λ rose 2.00→2.03). With ``/ws`` alone, Euclidean SGD at
        main lr is too small (p2a: 2.00→1.991) — use ``cycle_lambda_optim=sign_sgd``
        for pack-invariant unit steps in λ (Δλ≈−lr each step when CE wants lower λ).
        """
        out: dict[str, float] = {}
        jac_vals: list[float] = []
        ws = 1
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                ws = max(1, int(dist.get_world_size()))
        except Exception:
            ws = 1
        for attr in ("cycle_lambda_raw", "inject_lambda_raw"):
            p = getattr(self, attr, None)
            if p is None or p.grad is None:
                continue
            raw = p.data
            jac = self._lambda_pack_jacobian(raw).clamp_min(eps)
            g_pre = p.grad.detach().float()
            # Stash θ-space + λ-space proxy grads for [λβ] logs (all ranks; no gather).
            out[f"{attr}_grad_theta_mean"] = float(g_pre.mean().item())
            out[f"{attr}_grad_theta_abs"] = float(g_pre.abs().mean().item())
            # Autograd g_θ = (∂L/∂λ)·jac ⇒ ∂L/∂λ ≈ g_θ/jac (before /jac²).
            g_lam = g_pre / jac.detach().float()
            out[f"{attr}_grad_lambda_mean"] = float(g_lam.mean().item())
            out[f"{attr}_grad_lambda_abs"] = float(g_lam.abs().mean().item())
            # /jac² so SGD/Adam on θ realizes Euclidean steps in λ.
            p.grad.div_(jac * jac)
            if ws > 1:
                p.grad.div_(float(ws))
            jac_vals.append(float(jac.detach().float().mean().item()))
        if jac_vals:
            out["cycle_lambda_jac_mean"] = sum(jac_vals) / len(jac_vals)
            out["cycle_lambda_grad_ws"] = float(ws)
        return out

    @torch.no_grad()
    def project_cycle_lambda_raw_away_from_dead_jac(self, softplus_floor: float = 0.05) -> dict[str, float]:
        """Keep pack Jacobian bounded away from 0 (structural, not LR tuning).

        softplus: θ→−∞ ⇒ softplus'→0 ⇒ λ frozen + /jac² blows up (soft_init15 H9 collapse).
        Project θ so softplus(θ) ≥ floor ⇒ σ(θ) ≳ floor/(1+floor) > 0.
        tanh: keep |tanh| ≤ 1−eps so sech² stays healthy (same spirit as init clamp).

        rms10/rms11: also project λ ≤ depth_prior_hi so raw λ matches residual ceiling
        (soft15 rms10 saw raw λ→2.0 while residual was clamped → H12=4.04).
        """
        out: dict[str, float] = {}
        for attr in ("cycle_lambda_raw", "inject_lambda_raw"):
            p = getattr(self, attr, None)
            if p is None:
                continue
            if self.cycle_scale_lambda_pack == "softplus":
                floor = max(float(softplus_floor), 1e-3)
                # softplus(θ)=floor ⇒ θ = log(expm1(floor))
                thr = math.log(math.expm1(floor)) if floor < 20.0 else floor
                before = float(p.data.float().mean().item())
                p.data.clamp_(min=thr)
                out[f"{attr}_proj_softplus"] = float(p.data.float().mean().item()) - before
            elif self.cycle_scale_lambda_pack == "softplus_center":
                # θ≥0 ⇒ λ≥λ_prior (WD may overshoot below prior — p2g H6 λ→1.26).
                before = float(p.data.float().mean().item())
                p.data.clamp_(min=0.0)
                out[f"{attr}_proj_softplus_center"] = float(p.data.float().mean().item()) - before
                # settle latch: force θ=0 ⇒ λ=prior. Forward-path snap was a no-op under
                # FSDP; post-step project is the reliable place (no aux / no new HP).
                mode = str(getattr(self, "cycle_lambda_detach_mode", "") or "")
                latched = getattr(self, "_cycle_lambda_settled_latched", None)
                if mode == "settle" and latched is not None:
                    keep = 1.0 - latched.to(dtype=p.dtype)
                    p.data.mul_(keep)
            elif self.cycle_scale_lambda_pack == "exp_prior":
                # θ≥0 ⇒ λ≥λ_prior. WD pulls to θ=0; CE may push up; never below prior.
                before = float(p.data.float().mean().item())
                p.data.clamp_(min=0.0)
                out[f"{attr}_proj_exp_prior"] = float(p.data.float().mean().item()) - before
            elif self.cycle_scale_lambda_pack == "tanh":
                eps = float(getattr(self, "cycle_scale_lambda_sig_eps", 0.05) or 0.05)
                eps = min(max(eps, 1e-3), 0.2)
                # artanh(±(1−eps))
                lim = 0.5 * math.log((2.0 - eps) / eps)
                before = float(p.data.float().mean().item())
                p.data.clamp_(min=-lim, max=lim)
                out[f"{attr}_proj_tanh"] = float(p.data.float().mean().item()) - before
                # settle latch: θ→0 ⇒ λ=min (tanh_center: min=1.6). Same as softplus_center.
                mode = str(getattr(self, "cycle_lambda_detach_mode", "") or "")
                latched = getattr(self, "_cycle_lambda_settled_latched", None)
                if mode == "settle" and latched is not None:
                    keep = 1.0 - latched.to(dtype=p.dtype)
                    p.data.mul_(keep)
            # Operating-band ceiling for residual-scaling modes (match forward clamp).
            mode = str(getattr(self, "cycle_lambda_detach_mode", "rms7b") or "rms7b")
            if mode in ("rms10", "rms11", "rms12"):
                hi = float(getattr(self, "cycle_lambda_depth_prior_hi", 1.75) or 1.75)
                H = float(max(int(getattr(self, "num_loops", 1) or 1), 1))
                hi = min(hi, 1.65 + 0.10 * (9.0 / H) ** 0.5)
                lam_min = float(self.cycle_scale_lambda_min or 0.0)
                amp = float(self.cycle_scale_lambda_amp or 1.0)
                before = float(p.data.float().mean().item())
                if self.cycle_scale_lambda_pack == "softplus":
                    # λ=min+softplus(θ) ≤ hi ⇒ softplus(θ) ≤ hi−min
                    room = max(hi - lam_min, 1e-3)
                    thr_hi = math.log(math.expm1(room)) if room < 20.0 else room
                    p.data.clamp_(max=thr_hi)
                elif self.cycle_scale_lambda_pack == "tanh":
                    # λ=min+amp·tanh(θ) ≤ hi ⇒ tanh(θ) ≤ (hi−min)/amp
                    frac = min(max((hi - lam_min) / max(amp, 1e-6), -0.999), 0.999)
                    thr_hi = 0.5 * math.log((1.0 + frac) / max(1.0 - frac, 1e-6))
                    p.data.clamp_(max=thr_hi)
                out[f"{attr}_proj_hi"] = float(p.data.float().mean().item()) - before
        return out

    def apply_cycle_lambda_schedule(self, step: int, total_steps: int) -> float:
        """Linear λ(t) for frozen shared λ. No-op unless sched=linear."""
        start = float(getattr(self, "_cycle_lambda_sched_start", self.cycle_scale_lambda))
        end = float(getattr(self, "cycle_scale_lambda_end", start) or start)
        sched = str(getattr(self, "cycle_scale_lambda_sched", "none") or "none").lower()
        if sched not in ("linear", "lin"):
            return float(self.cycle_scale_lambda)
        total = max(int(total_steps), 1)
        p = min(max(float(step), 0.0) / float(total), 1.0)
        lam = start + (end - start) * p
        self.cycle_scale_lambda = float(lam)
        return float(lam)

    def _cycle_lambda_res_live(self) -> Union[float, Tensor]:
        """Packed residual λ without CE-detach (for depth-prior / stream-gain grads).

        rms8 freezes CE→λ via `_cycle_lambda_res` whenever λ≥mid; if prior also called
        the detached path, soft_init20 started frozen at λ=2.0 forever. Prior must stay live.
        """
        if self.cycle_lambda_raw is not None:
            lam = self._pack_lambda(self.cycle_lambda_raw)
            if lam.numel() == 1:
                return lam.reshape(())
            return lam.reshape(-1)
        return self.cycle_scale_lambda

    def _cycle_lambda_res(self) -> Union[float, Tensor]:
        """Residual-scale λ (scalar or per-layer vector). Init≈cycle_scale_lambda.

        rms7 / μP fixed-point: once λ sits in the operating band, *detach* it for the
        residual path so CE cannot keep yanking λ (rms6 only gated stream-gain/prior;
        soft20 H12 rms6b still saw 86 late loss spikes from CE→λ). Outside the band,
        λ stays live so stream-gain + depth prior can transfer it in.
        """
        if self.cycle_lambda_raw is not None:
            lam = self._pack_lambda(self.cycle_lambda_raw)
            if lam.numel() == 1:
                lam = lam.reshape(())
            else:
                lam = lam.reshape(-1)  # (n_layers,)
            if self.training and isinstance(lam, Tensor):
                lo = float(getattr(self, "cycle_lambda_depth_prior_lo", 1.58) or 1.58)
                hi = float(getattr(self, "cycle_lambda_depth_prior_hi", 1.75) or 1.75)
                # H-aware hi (residual-scaling transferability).
                H = float(max(int(getattr(self, "num_loops", 1) or 1), 1))
                hi = min(hi, 1.65 + 0.10 * (9.0 / H) ** 0.5)  # H9:1.75, H12:1.736
                mid = 0.5 * (lo + hi)
                lam_d = lam.detach()
                # rms7b: freeze ONLY in [mid, hi] (climb zone [lo,mid) stays live).
                # rms8: freeze CE whenever λ≥mid — includes λ>hi overshoot.
                #   Prior/stream MUST use `_cycle_lambda_res_live` (not this path).
                # Freezing at lo (rms7) trapped soft_init15 at λ≈1.582 → eval~7.2.
                mode = str(getattr(self, "cycle_lambda_detach_mode", "rms7b") or "rms7b")
                if mode in ("none", "off", "ce", "ce_only"):
                    # Pure CE→λ: no band-detach, no STE/hi ceiling.
                    pass
                elif mode == "settle":
                    # Permanent CE-detach once λ has descended far enough toward prior.
                    # Evidence:
                    #   fixed λ∈{1.6,2.0} survives H12; softcenter CE↑↔WD↓ near 1.6–1.7
                    #   permanently collapses; p2s3 snap-on-latch@hi≈1.75 works on H6/H9
                    #   but H12 explodes @step≈610 while λ≈1.82 (never reached old hi).
                    #   Old hi=min(1.75, 1.65+0.10√(9/H)) *shrinks* with H — wrong for
                    #   settle (deeper needs *earlier* freeze of residual scale).
                    # Latch threshold from existing init/prior + same H_ref=9 already used
                    # above — no new free knobs. H6:1.76 H9:1.80 H12:1.83.
                    # Rising-edge snap θ←0 ⇒ λ=depth_prior (μP-style coordinate fix).
                    init_lam = float(getattr(self, "cycle_scale_lambda", 2.0) or 2.0)
                    prior = float(getattr(self, "cycle_lambda_depth_prior", 1.6) or 1.6)
                    latch_hi = prior + (init_lam - prior) * (H / (H + 9.0))
                    if not hasattr(self, "_cycle_lambda_settled_latched"):
                        self.register_buffer(
                            "_cycle_lambda_settled_latched",
                            torch.zeros((), dtype=torch.bool, device=lam.device),
                            persistent=False,
                        )
                    if not hasattr(self, "_cycle_lambda_settle_prev"):
                        self.register_buffer(
                            "_cycle_lambda_settle_prev",
                            torch.zeros((), dtype=torch.bool, device=lam.device),
                            persistent=False,
                        )
                    reached = (lam_d <= latch_hi).reshape(-1).any()
                    self._cycle_lambda_settled_latched = (
                        self._cycle_lambda_settled_latched | reached
                    )
                    newly = self._cycle_lambda_settled_latched & (
                        ~self._cycle_lambda_settle_prev
                    )
                    self._cycle_lambda_settle_prev = (
                        self._cycle_lambda_settled_latched.clone()
                    )
                    # Snap raw→0 when latch first fires (softplus_center/tanh_center:
                    # θ=0 ⇒ λ=depth_prior). Mul keeps it Ascend-safe (no .item()).
                    if self.cycle_lambda_raw is not None:
                        keep = (1.0 - newly.to(dtype=self.cycle_lambda_raw.dtype))
                        self.cycle_lambda_raw.data.mul_(keep)
                        lam = self._pack_lambda(self.cycle_lambda_raw)
                        if lam.numel() == 1:
                            lam = lam.reshape(())
                        else:
                            lam = lam.reshape(-1)
                        lam_d = lam.detach()
                    lam = torch.where(self._cycle_lambda_settled_latched, lam_d, lam)
                elif mode == "rms8":
                    ce_frozen = lam_d >= mid
                    lam = torch.where(ce_frozen, lam_d, lam)
                elif mode == "rms9":
                    # Hysteresis latch (no .item(): Ascend crash at latch ~step940).
                    if not hasattr(self, "_cycle_lambda_settled_latched"):
                        self.register_buffer(
                            "_cycle_lambda_settled_latched",
                            torch.zeros((), dtype=torch.bool, device=lam.device),
                            persistent=False,
                        )
                    in_band = (lam_d >= mid) & (lam_d <= hi)
                    self._cycle_lambda_settled_latched = (
                        self._cycle_lambda_settled_latched | in_band.reshape(-1).any()
                    )
                    ce_frozen = self._cycle_lambda_settled_latched | in_band
                    lam = torch.where(ce_frozen, lam_d, lam)
                elif mode == "rms10":
                    # Hard operating ceiling: residual uses min(λ, hi). CE live below hi
                    # (fixes rms8 early-freeze H12=4.5); cannot express λ>hi in residual
                    # (fixes rms7b rebound→2.5). Prior still sees live λ via _res_live.
                    hi_t = lam.new_tensor(hi)
                    lam = torch.minimum(lam, hi_t)
                elif mode == "rms11":
                    # Ceiling + band settle (soft20 rms10 H12=3.4644 was +0.009 vs H9):
                    #   λ_raw > hi → residual=min(λ,hi), CE grad blocked by clamp; not "settled"
                    #   mid≤λ_raw≤hi → detach CE (rms7b fixed-point)
                    #   λ_raw < mid → live climb (init15)
                    hi_t = lam.new_tensor(hi)
                    raw_d = lam.detach()
                    clamping = raw_d > hi_t
                    lam_c = torch.minimum(lam, hi_t)
                    lam_cd = lam_c.detach()
                    settled = (lam_cd >= mid) & (lam_cd <= hi_t) & (~clamping)
                    lam = torch.where(settled, lam_cd, lam_c)
                elif mode == "rms12":
                    # STE ceiling (μP-style coordinate: care about λ≤hi in forward, Euclidean
                    # step in λ for CE). soft20 rms11 froze: min(λ,hi) has 0-grad when λ>hi,
                    # and with raw hi-proj ⇒ settled forever at hi with grad/λ=0.
                    # Forward: min(λ,hi). Backward: identity ⇒ CE can pull λ down through hi.
                    # No band-detach — tanh20 rms10 already showed deepen with live CE+ceiling.
                    hi_t = lam.new_tensor(hi)
                    lam_c = torch.minimum(lam, hi_t)
                    lam = lam + (lam_c - lam).detach()
                else:
                    ce_frozen = (lam_d >= mid) & (lam_d <= hi)
                    lam = torch.where(ce_frozen, lam_d, lam)
            return lam
        return self.cycle_scale_lambda

    def _cycle_lambda_inj(self) -> Union[float, Tensor]:
        """Soft-inject λ. split/per_layer → inject_lambda_raw; else share residual λ (scalar mean)."""
        if self.inject_lambda_raw is not None:
            return self._pack_lambda(self.inject_lambda_raw).reshape(())
        lam = self._cycle_lambda_res()
        if isinstance(lam, Tensor) and lam.numel() > 1:
            return lam.mean()
        return lam

    def _cycle_lambda(self) -> Union[float, Tensor]:
        """Backward-compat alias: residual λ (scalar if per-layer → mean)."""
        lam = self._cycle_lambda_res()
        if isinstance(lam, Tensor) and lam.numel() > 1:
            return lam.mean()
        return lam

    def _schedule_eps(self, t: int) -> Union[float, Tensor]:
        """Soft-inject gain. Residual schedule unless embed_inject_time_mode is set.

        Optional embed_inject_depth_scale ÷√L so inv_t ⇒ g=λ/(t√L).
        """
        lam = self._cycle_lambda_inj()
        inj_mode = str(getattr(self, "embed_inject_time_mode", "inherit") or "inherit").lower()
        if inj_mode in ("", "none", "inherit"):
            eps = lam / self._cycle_time_denom(t)
        else:
            eps = lam / self._cycle_time_denom(t, time_mode=inj_mode)
        if bool(getattr(self, "embed_inject_depth_scale", False)):
            eps = eps / self._sqrt_L
        return eps

    def _cycle_eps(self, t: int) -> Union[float, Tensor]:
        """Loop residual factor. Off → 1.0; on → λ_res/time_denom [×1/√L if no LNS].

        time_denom = t_eff (inv_t) or √t_eff (inv_sqrt_t). span=0 ⇒ t_eff=t.
        Per-layer mode returns a length-L vector (one ε per physical layer).
        """
        if not self.cycle_residual_scale:
            return 1.0
        lam = self._cycle_lambda_res()
        denom = self._cycle_time_denom(t)
        extra = float(getattr(self, "cycle_eps_first_extra", 0.0) or 0.0)
        n_first = int(getattr(self, "cycle_eps_first_loops", 1) or 0)
        if extra > 0.0 and n_first > 0 and int(t) <= n_first:
            # Leading loops only; inject schedule (_schedule_eps) is unchanged.
            denom = denom + extra
        if self._depth_via_ell:
            return lam / denom
        return lam / (denom * self._sqrt_L)

    @staticmethod
    def _param_dense(p: Tensor) -> Tensor:
        """Host-side metrics only. Never full_tensor/to_local — those hang HCCL on H≥9."""
        t = p.detach()
        loc = getattr(t, "_local_tensor", None)
        if isinstance(loc, Tensor):
            return loc
        # DTensor without a materialized shard: skip (empty → caller no-ops).
        if type(t).__name__ == "DTensor" or hasattr(t, "full_tensor") or hasattr(t, "to_local"):
            return t.new_zeros((0,))
        return t

    @torch.no_grad()
    def get_cycle_lambda_metrics(self) -> dict[str, float]:
        """Scalars for train logging / module_diag CSV (λ trajectory)."""
        out: dict[str, float] = {}
        if self.cycle_lambda_raw is None and self.inject_lambda_raw is None:
            if self.cycle_residual_scale:
                out["cycle_lambda"] = float(self.cycle_scale_lambda)
                out["cycle_lambda_res"] = float(self.cycle_scale_lambda)
                out["cycle_lambda_inj"] = float(self.cycle_scale_lambda)
            return out
        if self.cycle_lambda_raw is not None:
            # Avoid F.softplus on DTensor outside forward (no sharding strategy).
            raw = self._param_dense(self.cycle_lambda_raw).float().reshape(-1)
            if raw.numel() == 0:
                return out
            lam = self._pack_lambda(raw).cpu()
            if self.cycle_scale_lambda_min > 0.0:
                out["cycle_lambda_min"] = float(self.cycle_scale_lambda_min)
            out["cycle_lambda_amp"] = float(self.cycle_scale_lambda_amp)
            # pack encoded as float for CSV: 0=softplus, 1=sigmoid, 2=tanh
            if self.cycle_scale_lambda_pack == "sigmoid":
                out["cycle_lambda_pack_id"] = 1.0
            elif self.cycle_scale_lambda_pack == "tanh":
                out["cycle_lambda_pack_id"] = 2.0
            else:
                out["cycle_lambda_pack_id"] = 0.0
            out["cycle_lambda_pack_sigmoid"] = (
                1.0 if self.cycle_scale_lambda_pack == "sigmoid" else 0.0
            )
            out["cycle_lambda_pack_tanh"] = (
                1.0 if self.cycle_scale_lambda_pack == "tanh" else 0.0
            )
            if lam.numel() == 1:
                v = float(lam.item())
                out["cycle_lambda"] = v
                out["cycle_lambda_res"] = v
            else:
                out["cycle_lambda_res_mean"] = float(lam.mean().item())
                out["cycle_lambda_res_min"] = float(lam.min().item())
                out["cycle_lambda_res_max"] = float(lam.max().item())
                out["cycle_lambda"] = float(lam.mean().item())
                for i, vi in enumerate(lam.tolist()):
                    out[f"cycle_lambda_L{i}"] = float(vi)
        if self.inject_lambda_raw is not None:
            raw_i = self._param_dense(self.inject_lambda_raw).float().reshape(-1)
            if raw_i.numel() > 0:
                out["cycle_lambda_inj"] = float(self._pack_lambda(raw_i).reshape(()).item())
        elif "cycle_lambda_res" in out:
            out["cycle_lambda_inj"] = out["cycle_lambda_res"]
        elif "cycle_lambda" in out:
            out["cycle_lambda_inj"] = out["cycle_lambda"]
        return out

    def _inject_beta(self) -> Union[float, Tensor]:
        if self.embed_gamma_raw is not None:
            return F.softplus(self.embed_gamma_raw).reshape(())
        return max(0.0, float(self.embed_inject_beta))

    def _inject_is_additive(self) -> bool:
        return self.embed_inject_mode in ("loop_start_add_soft", "loop_start_add")

    def _rms(self, t: Tensor) -> Tensor:
        return F.rms_norm(t, (t.size(-1),), eps=1e-6)

    def _mix_inject(self, z: Tensor, x: Tensor, g: Union[float, Tensor]) -> Tensor:
        """Convex mix; optional parameter-free RMSNorm on z and/or x before mix."""
        mode = self.embed_inject_norm
        z_m = self._rms(z) if mode in ("z", "both") else z
        x_m = self._rms(x) if mode in ("x", "both") else x
        return (1.0 - g) * z_m + g * x_m

    def _apply_convex_inject(self, z: Tensor, x: Tensor, g: Union[float, Tensor]) -> Tensor:
        """Scheme (1) z←(1−g)z+g x; scheme (2) swap → z←g z+(1−g) x.

        Always clamp g to [0, 1] so λ/t > 1 (e.g. λ=1.5 at t=1 if enabled) stays convex.
        """
        if isinstance(g, Tensor):
            g = g.clamp(0.0, 1.0)
        else:
            g = min(max(float(g), 0.0), 1.0)
        if bool(getattr(self, "embed_inject_swap", False)):
            g = (1.0 - g) if not isinstance(g, Tensor) else (1.0 - g)
        return self._mix_inject(z, x, g)

    def _inject_gate(self, t: int, *, allow_t1: bool | None = None) -> Union[float, Tensor]:
        """Inject strength at loop t (1-based). 0 → no inject."""
        mode = self.embed_inject_mode
        if mode in ("", "none"):
            return 0.0
        if allow_t1 is None:
            allow_t1 = bool(getattr(self, "embed_inject_from_t1", False))
        if t <= 1 and not allow_t1:
            return 0.0
        H = float(self.num_loops)
        L = float(getattr(self, "_n_layers", 0) or (self._sqrt_L ** 2))

        # Soft inject uses _schedule_eps (g=λ/t), NOT residual _cycle_eps.
        # Ablating residual/CoD scaling must not collapse soft inject to g=1.
        if mode in ("loop_start_add_soft", "loop_start_mix_soft"):
            eps = self._schedule_eps(t)
            if self.embed_gamma_raw is not None:
                return F.softplus(self.embed_gamma_raw).reshape(()) * eps
            return eps
        if mode == "loop_start_mix_soft_comp":
            g = self._schedule_eps(1) - self._schedule_eps(t)
            if not isinstance(g, Tensor):
                g = max(0.0, min(float(g), 1.0))
            else:
                g = g.clamp(0.0, 1.0)
            if self.embed_gamma_raw is not None:
                return F.softplus(self.embed_gamma_raw).reshape(()) * g
            return g
        if mode == "loop_start_mix_soft_prod":
            g = self._schedule_eps(t) * (float(t) / H)
            if not isinstance(g, Tensor):
                g = min(float(g), 1.0)
            else:
                g = g.clamp(max=1.0)
            if self.embed_gamma_raw is not None:
                return F.softplus(self.embed_gamma_raw).reshape(()) * g
            return g
        if mode in ("loop_start_mix_eps1_ramp", "loop_start_mix_eps1_half", "loop_start_mix_eps3_ramp"):
            if mode == "loop_start_mix_eps3_ramp":
                base = self._schedule_eps(3)
            else:
                base = self._schedule_eps(1)
                if mode == "loop_start_mix_eps1_half":
                    base = base * 0.5 if isinstance(base, Tensor) else base * 0.5
            g = base * (float(t) / H)
            if not isinstance(g, Tensor):
                g = min(float(g), 1.0)
            else:
                g = g.clamp(max=1.0)
            if self.embed_gamma_raw is not None:
                return F.softplus(self.embed_gamma_raw).reshape(()) * g
            return g

        # --- CoD / depth-transfer (preferred; terminal independent of H) ---
        # Equivalent forms: γ = α·t/(H·√L) with α=β★·√L★ for depth_xfer.
        if mode == "loop_start_mix_sqrtL":
            g = float(t) / (H * math.sqrt(max(L, 1.0)))
            return min(max(g, 0.0), 1.0)
        if mode == "loop_start_mix_sqrtLp1":
            g = float(t) / (H * math.sqrt(max(L, 1.0) + 1.0))
            return min(max(g, 0.0), 1.0)
        if mode == "loop_start_mix_sqrtLp2":
            g = float(t) / (H * math.sqrt(max(L, 1.0) + 2.0))
            return min(max(g, 0.0), 1.0)
        if mode == "loop_start_mix_sqrtLp3":
            g = float(t) / (H * math.sqrt(max(L, 1.0) + 3.0))
            return min(max(g, 0.0), 1.0)
        if mode == "loop_start_mix_sqrtLm1":
            g = float(t) / (H * math.sqrt(max(L - 1.0, 1.0)))
            return min(max(g, 0.0), 1.0)
        if mode == "loop_start_mix_sqrtL_t0":
            # Tie to residual soft_lam t0 (already chosen); no inject-specific hyperparam.
            g = float(t) / (H * math.sqrt(max(L, 1.0) + max(0.0, float(self.cycle_eps_t0))))
            return min(max(g, 0.0), 1.0)
        if mode == "loop_start_mix_depth_xfer":
            # LEGACY — introduces β★,L★; prefer sqrtL / sqrtLp1 / sqrtL_t0 instead.
            Lref = max(1e-6, float(self.embed_inject_L_ref))
            bref = max(0.0, float(self.embed_inject_beta_ref))
            beta_L = bref * math.sqrt(Lref / max(L, 1e-6))
            g = beta_L * (float(t) / H)
            return min(max(g, 0.0), 1.0)

        # --- legacy hand-β ---
        g = self._inject_beta()
        if mode == "loop_start_mix_flat":
            return g if isinstance(g, Tensor) else min(float(g), 1.0)
        if mode == "loop_start_mix":
            scale = float(t) / H
            return g * scale if isinstance(g, Tensor) else min(float(g) * scale, 1.0)
        if mode == "loop_start_add":
            return g * (float(t) / H)
        return 0.0

    def _apply_loop_id(self, z: Tensor, t: int) -> Tensor:
        """Inject loop-index identity at loop start (add / sin / scale). add_layer is per-layer."""
        mode = self.loop_id_embed_mode
        if mode in ("", "none", "add_layer"):
            return z
        scale = float(self.loop_id_embed_scale)
        if mode == "add" and self.loop_id_emb is not None:
            idx = min(max(t - 1, 0), self.loop_id_emb.num_embeddings - 1)
            return z + scale * self.loop_id_emb.weight[idx].to(dtype=z.dtype)
        if mode == "sin":
            d = z.shape[-1]
            device, dtype = z.device, z.dtype
            half = (d + 1) // 2
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, device=device, dtype=dtype)
                / max(half - 1, 1)
            )
            ang = float(t) * freqs
            pe = torch.empty(d, device=device, dtype=dtype)
            pe[0::2] = torch.sin(ang[: (d + 1) // 2])
            pe[1::2] = torch.cos(ang[: d // 2])
            return z + scale * pe
        if mode == "scale" and self.loop_id_raw is not None:
            idx = min(max(t - 1, 0), int(self.loop_id_raw.numel()) - 1)
            # 1 + α·tanh(s_t): mild multiplicative loop tag (α default 1)
            return z * (1.0 + scale * torch.tanh(self.loop_id_raw[idx]).to(dtype=z.dtype))
        return z

    def _maybe_inject_loop_id_layer(self, x: Tensor, layer_id: int) -> Tensor:
        """add_layer: H×L table, idx=(t−1)*L + layer_id, before each block."""
        if self.loop_id_embed_mode != "add_layer" or self.loop_id_layer_emb is None:
            return x
        L = len(self.layers)
        t_idx = min(max(int(self._loop_id_t) - 1, 0), int(self._loop_id_max_h) - 1)
        li = min(max(int(layer_id), 0), L - 1)
        idx = t_idx * L + li
        idx = min(idx, self.loop_id_layer_emb.num_embeddings - 1)
        scale = float(self.loop_id_embed_scale)
        return x + scale * self.loop_id_layer_emb.weight[idx].to(dtype=x.dtype)

    def forward_range(
        self,
        z: Tensor,
        x: Tensor,
        t_start: int,
        t_end: int,
        *,
        reset_bank: bool = False,
        **kwargs,
    ) -> Tensor:
        """Run loops t_start..t_end inclusive (1-based). ε/γ use global t and full H."""
        bank = self.attn_res_bank if self.attn_res else None
        dual = self.dual_axis if self._uses_dual_axis() else None
        if reset_bank:
            if bank is not None:
                bank.clear()
            self.reset_cross_loop_state()
        # Fresh RMS stash per forward_range call (segmented BP: one segment at a time).
        self._lambda_rms_states = []

        additive = self._inject_is_additive()
        try:
            from utils.module_diag import get_active_collector as _get_diag
        except Exception:
            _get_diag = None  # type: ignore[assignment]

        t0 = max(1, int(t_start))
        t1 = min(int(self.num_loops), int(t_end))
        for t in range(t0, t1 + 1):
            if _get_diag is not None:
                _diag = _get_diag()
                if _diag is not None:
                    _diag.set_cycle("H", t - 1)
            try:
                from utils.expert_freq import get_active_expert_logger
                _elog = get_active_expert_logger()
                if _elog is not None:
                    _elog.set_cycle("H", t - 1)
            except Exception:
                pass

            when = str(getattr(self, "embed_inject_when", "start") or "start").lower()
            t1_style = str(getattr(self, "embed_inject_t1_style", "rms_x") or "rms_x").lower()
            n_unit = int(getattr(self, "embed_inject_unit_loops", 0) or 0)
            if n_unit <= 0 and t1_style == "unit" and bool(getattr(self, "embed_inject_from_t1", False)):
                n_unit = 1
            did_unit = False
            if n_unit > 0 and int(t) <= n_unit:
                # Scale-invariant pin. Skip mix so embed scale cannot leak back in.
                z = self._rms(z)
                did_unit = True
            g = self._inject_gate(t) if (when in ("start", "both") and not did_unit) else 0.0
            g_pos = bool(g > 0.0) if not isinstance(g, Tensor) else True
            if g_pos and (isinstance(g, Tensor) or float(g) > 0.0):
                if isinstance(g, Tensor):
                    g = g.clamp(0.0, 1.0)
                else:
                    g = min(max(float(g), 0.0), 1.0)
                if int(t) == 1 and bool(getattr(self, "embed_inject_from_t1", False)) and t1_style == "empty":
                    # Option (3): shrink the loop-1 stream, no embed add.
                    keep = float(getattr(self, "embed_inject_t1_keep", -1.0) or -1.0)
                    if keep >= 0.0:
                        z = keep * z
                    else:
                        z = (1.0 - g) * z
                elif int(t) == 1 and bool(getattr(self, "embed_inject_from_t1", False)) and t1_style == "rms_x":
                    # Option (2): mix raw z with unit embed. Raw mix(z,x) is a no-op at t=1.
                    z = (1.0 - g) * z + g * self._rms(x)
                elif additive:
                    z = z + g * x
                else:
                    z = self._apply_convex_inject(z, x, g)

            # Loop-ID after inject so each loop body sees a distinct tag
            self._loop_id_t = t
            z = self._apply_loop_id(z, t)

            self.set_cycle_residual_eps(self._cycle_eps(t))

            if dual is not None:
                dual.begin_loop(t)

            if bank is not None:
                if self.attn_res_scope in ("intra", "loop_carry"):
                    bank.clear()
                elif self._uses_lag1_bank():
                    bank.clear()
                    if self._prev_loop_bank:
                        bank.restore(self._prev_loop_bank)
                # scope == "loop": keep full bank (legacy)

            if self._uses_ema_carry() and t > 1 and self._z_carry is not None:
                z = z + self._z_carry
            if self._uses_prev_loop_z() and t > 1 and self._z_prev_loop is not None:
                z = z + self._z_prev_loop

            if dual is not None:
                z = Transformer.forward(self, z, cache=None, dual_axis_carry=dual, moe_loop_idx=t - 1, **kwargs)
                dual.end_loop()
            elif bank is not None:
                z = Transformer.forward(self, z, cache=None, attn_res_bank=bank, moe_loop_idx=t - 1, **kwargs)
            else:
                z = Transformer.forward(self, z, cache=None, moe_loop_idx=t - 1, **kwargs)

            # End inject: after residual, mix is never a no-op at t=1 (z ≠ embed).
            if when in ("end", "both"):
                g_end = self._inject_gate(t, allow_t1=True)
                g_end_pos = bool(g_end > 0.0) if not isinstance(g_end, Tensor) else True
                if g_end_pos and (isinstance(g_end, Tensor) or float(g_end) > 0.0):
                    if isinstance(g_end, Tensor):
                        g_end = g_end.clamp(0.0, 1.0)
                    else:
                        g_end = min(max(float(g_end), 0.0), 1.0)
                    z = self._apply_convex_inject(z, x, g_end)

            if bank is not None and self._uses_lag1_bank():
                self._save_prev_loop_bank(bank)

            if self._uses_ema_carry():
                self._update_ema_carry(t, z)
            if self._uses_prev_loop_z():
                self._z_prev_loop = z.detach()
            # Loop-Δz RMS match (full-H): ref=RMS(z_2−z_1); live Δ at t=H.
            # Final-z RMS is LN-flat (match≈0); Δz tracks residual-scale λ.
            if self.training and self.cycle_lambda_rms_match:
                if t == 1:
                    self._z_prev_for_lambda_rms = z.detach()
                    self._lambda_rms_ref = None
                    self._lambda_rms_last = None
                else:
                    prev = self._z_prev_for_lambda_rms
                    if prev is not None:
                        if t == 2:
                            d0 = z.detach() - prev
                            self._lambda_rms_ref = (
                                d0.to(torch.float32).pow(2).mean().clamp_min(1e-12).sqrt()
                            )
                        if t == int(self.num_loops):
                            # Live Δ: grad through z_H (and thus ε(λ) on last segment).
                            self._lambda_rms_last = z - prev
                    self._z_prev_for_lambda_rms = z.detach()
        return z

    def pop_lambda_rms_states(self) -> list[Tensor]:
        """Consume per-loop hidden states for stream-RMS matching aux loss (legacy)."""
        out = self._lambda_rms_states
        self._lambda_rms_states = []
        return out

    def pop_lambda_rms_full_h(self) -> tuple[Optional[Tensor], Optional[Tensor]]:
        """Return (delta_rms_ref_detached, delta_last_live) for full-H Δz-RMS; clear last."""
        ref = self._lambda_rms_ref
        last = self._lambda_rms_last
        self._lambda_rms_last = None
        # Keep ref across mid-H segments; cleared on next t=1.
        return ref, last

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        carry: Optional[Tensor],
        x: Tensor,
        cache: Optional[list[Cache]] = None,
        **kwargs,
    ) -> tuple[Optional[Tensor], Tensor]:
        del cache
        t0 = kwargs.pop("loop_t_start", None)
        t1 = kwargs.pop("loop_t_end", None)
        reset_bank = bool(kwargs.pop("reset_bank", carry is None))
        snap = kwargs.pop("bank_snapshot", None)
        segmented = t0 is not None
        if snap is not None:
            if isinstance(snap, dict):
                self.cross_loop_restore(snap)
            elif self.attn_res_bank is not None:
                self.attn_res_bank.restore(snap)
            reset_bank = False
        z0 = x if carry is None else carry
        z = self.forward_range(
            z0,
            x,
            1 if t0 is None else int(t0),
            self.num_loops if t1 is None else int(t1),
            reset_bank=reset_bank,
            **kwargs,
        )
        # Full-batch path keeps carry=None (no cross-step state). Segmented BP
        # returns z so the train loop can detach and continue the same sequence.
        return (z if segmented else None), z

    def compute_train_extra_args(self, train_state):
        return {}

    def initial_carry(self, batch_size: int, dtype: torch.dtype) -> None:
        return None
